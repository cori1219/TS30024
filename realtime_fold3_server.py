#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensor Logger HTTP Push → 실시간 균형 이상 분류 서버
(Conv DAE v0.0.x, fold3_ckpt.pt + fold3_norm.npz 사용)

test 시점과 최대한 비슷하게:
- 데이터 윈도우 구성: 100 Hz, 30초 (TARGET_HZ, WINDOW_SEC)
- 채널 정규화: fold3 train 폴드에서 구한 mean/std 사용 (ChannelNormalizer와 동일)
- 인코딩: Conv AE (AE1D)
- Kalman smoothing 후 시간 평균 풀링
- LatentClassifier 로 로짓 → sigmoid → threshold (0.5 기준)

※ main.py 는 수정하지 않고, 이 파일 + export_fold3_norm.py 만 추가로 사용.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


# ----------------- 학습 때와 맞춰야 하는 설정 -----------------
TARGET_HZ   = 100          # --target_hz
WINDOW_SEC  = 30           # --window_sec
WIN_LEN     = TARGET_HZ * WINDOW_SEC
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

LATENT_DIM  = 32           # --latent
HIDDEN_DIM  = 64           # --hidden

# fold3 모델/정규화 파일
CKPT_PATH   = Path("./runs/fold3_ckpt.pt")
NORM_PATH   = Path("./runs/fold3_norm.npz")

# test 시와 동일하게 threshold=0.5 사용 (v0.0.x 에서는 따로 튜닝 안 했으니까)
THRESHOLD   = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- 모델 정의 (main.py v0.0.x 와 동일 구조) -----------------
class Encoder1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):  # [B, C, T]
        return self.net(x)  # [B, latent, T]


class Decoder1D(nn.Module):
    def __init__(self, latent=32, hidden=64, out_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent, hidden, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, out_ch, 7, padding=3),
        )

    def forward(self, z):  # [B, latent, T]
        return self.net(z)  # [B, C, T]


class AE1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.latent_dim = latent
        self.enc = Encoder1D(in_ch, hidden, latent)
        self.dec = Decoder1D(latent, hidden, in_ch)

    def forward(self, x):
        z = self.enc(x)
        xr = self.dec(z)
        return xr, z

    def pooled_latent(self, z):
        # [B, D, T] -> [B, D]
        return z.mean(dim=-1)


class LatentClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  # [B, D]
        return self.net(x).squeeze(-1)


class SimpleKalman:
    """학습 코드와 동일한 latent smoothing."""

    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var

    def filter(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: [T, D]
        """
        T, D = seq.shape
        out = np.zeros_like(seq)
        x = np.zeros(D)
        p = np.ones(D)

        for t in range(T):
            # prediction
            x_pred = x
            p_pred = p + self.q

            # update
            z = seq[t]
            k = p_pred / (p_pred + self.r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred

            out[t] = x

        return out


# ----------------- Sensor Logger 세션 버퍼 -----------------
class SessionBuffer:
    """
    Sensor Logger HTTP Push 스트림을 sessionId 별로 누적.

    - accelerometer / gyroscope 를 각 DF에 timestamp index로 쌓고
    - outer join + 100 Hz resample
    - 마지막 30초 (WIN_LEN 샘플)만 잘라서 [T, 6] numpy 배열로 반환
    """

    def __init__(self):
        self.acc_df = pd.DataFrame(columns=["ax", "ay", "az"])
        self.gyr_df = pd.DataFrame(columns=["gx", "gy", "gz"])

    def _trim_history(self, now_ts, keep_sec: int = 60):
        cutoff = now_ts - pd.to_timedelta(keep_sec, unit="s")
        if not self.acc_df.empty:
            self.acc_df = self.acc_df[self.acc_df.index >= cutoff]
        if not self.gyr_df.empty:
            self.gyr_df = self.gyr_df[self.gyr_df.index >= cutoff]

    def update_from_payload(self, payload: List[dict]):
        for d in payload:
            name = str(d.get("name", "")).lower()
            t_ns = int(d.get("time"))
            ts = pd.to_datetime(t_ns, unit="ns", utc=True)
            values = d.get("values", {})

            if name == "accelerometer":
                ax = float(values.get("x", np.nan))
                ay = float(values.get("y", np.nan))
                az = float(values.get("z", np.nan))
                self.acc_df.loc[ts, ["ax", "ay", "az"]] = [ax, ay, az]

            elif name == "gyroscope":
                gx = float(values.get("x", np.nan))
                gy = float(values.get("y", np.nan))
                gz = float(values.get("z", np.nan))
                self.gyr_df.loc[ts, ["gx", "gy", "gz"]] = [gx, gy, gz]

            self._trim_history(ts)

    def _merged_resampled(self) -> Optional[pd.DataFrame]:
        df_all = None
        if not self.acc_df.empty:
            df_all = self.acc_df.sort_index()
        if not self.gyr_df.empty:
            g = self.gyr_df.sort_index()
            df_all = g if df_all is None else df_all.join(g, how="outer")

        if df_all is None or df_all.empty:
            return None

        rule = pd.to_timedelta(1 / TARGET_HZ, unit="s")
        idx = pd.date_range(df_all.index.min(), df_all.index.max(), freq=rule)
        df_all = df_all.infer_objects(copy=False)
        df_all = df_all.reindex(df_all.index.union(idx)).interpolate(method="time").reindex(idx)

        for c in SENSOR_COLS:
            if c not in df_all.columns:
                df_all[c] = np.nan
        df_all = df_all[SENSOR_COLS].interpolate(limit_direction="both")

        return df_all

    def current_window(self) -> Optional[np.ndarray]:
        df_all = self._merged_resampled()
        if df_all is None or df_all.empty:
            return None

        if len(df_all) < WIN_LEN:
            return None

        df_win = df_all.iloc[-WIN_LEN:]
        return df_win.to_numpy(dtype=np.float32)  # [T, 6]

    def num_samples(self) -> int:
        df_all = self._merged_resampled()
        return 0 if df_all is None else len(df_all)


# ----------------- fold3 모델 + 정규화 로딩 -----------------
def load_fold3_model_and_norm():
    # 모델
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    ae = AE1D(in_ch=6, hidden=HIDDEN_DIM, latent=LATENT_DIM).to(DEVICE)
    clf = LatentClassifier(in_dim=LATENT_DIM, hidden=HIDDEN_DIM).to(DEVICE)

    ae.load_state_dict(ckpt["ae"])
    clf.load_state_dict(ckpt["clf_head"])
    ae.eval()
    clf.eval()

    # 정규화 통계 (ChannelNormalizer.mean/std)
    if not NORM_PATH.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {NORM_PATH}. "
            f"Run export_fold3_norm.py first."
        )
    norm = np.load(NORM_PATH)
    mean = norm["mean"].astype(np.float32)  # [6]
    std  = norm["std"].astype(np.float32)   # [6]

    print(f"[INFO] Loaded fold3 model from {CKPT_PATH}")
    print(f"[INFO] Loaded fold3 norm from  {NORM_PATH}")
    print(f"[INFO] Device: {DEVICE}, threshold={THRESHOLD:.3f}")
    return ae, clf, mean, std


@torch.no_grad()
def predict_window(
    ae: AE1D,
    clf: LatentClassifier,
    arr: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    threshold: float,
):
    """
    arr: [T, 6] (T = 30초 * 100Hz)
    global_mean / global_std: ChannelNormalizer 로 학습된 [6] 벡터
    """

    x = arr.copy()  # [T, 6]
    # test 때와 동일한 방식: 채널별 (x - mean)/std
    x = (x - global_mean.reshape(1, -1)) / (global_std.reshape(1, -1) + 1e-6)

    # [T, 6] -> [1, 6, T]
    x = np.transpose(x, (1, 0))  # [6, T]
    x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)  # [1, 6, T]

    # AE 인코딩
    _, z = ae(x_tensor)          # z: [1, D, T]
    z_np = z.cpu().numpy()
    B, D, T = z_np.shape
    z_np = np.transpose(z_np, (0, 2, 1))  # [B, T, D]

    # Kalman smoothing
    kal = SimpleKalman(dim=D, process_var=1e-3, measure_var=1e-2)
    zf = np.zeros_like(z_np)
    for b in range(B):
        zf[b] = kal.filter(z_np[b])

    # 시간 평균 풀링
    z_pool = zf.mean(axis=1)  # [B, D]
    z_pool_tensor = torch.from_numpy(z_pool).float().to(DEVICE)

    logits = clf(z_pool_tensor).cpu().numpy()  # [B]
    prob = 1.0 / (1.0 + np.exp(-logits[0]))
    pred = int(prob >= threshold)
    text = "abnormal" if pred == 1 else "normal"

    return float(prob), pred, text


# ----------------- Flask 서버 -----------------
app = Flask(__name__)

SESSIONS: Dict[str, SessionBuffer] = {}

AE_MODEL, CLF_HEAD, GLOBAL_MEAN, GLOBAL_STD = load_fold3_model_and_norm()


@app.route("/data", methods=["POST"])
def handle_data():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        print("[ERROR] JSON parse failed:", e)
        return jsonify({"status": "error", "msg": "invalid JSON"}), 400

    session_id = data.get("sessionId", "default")
    payload    = data.get("payload", [])

    if not isinstance(payload, list):
        return jsonify({"status": "error", "msg": "payload must be a list"}), 400

    buf = SESSIONS.get(session_id)
    if buf is None:
        buf = SessionBuffer()
        SESSIONS[session_id] = buf

    buf.update_from_payload(payload)

    window = buf.current_window()
    num    = buf.num_samples()

    if window is None:
        # 아직 30초 분량이 안 쌓인 경우
        return jsonify({
            "status": "buffering",
            "sessionId": session_id,
            "num_samples_resampled": int(num),
            "required_samples": int(WIN_LEN),
        })

    prob, pred, text_label = predict_window(
        AE_MODEL, CLF_HEAD, window,
        GLOBAL_MEAN, GLOBAL_STD,
        THRESHOLD,
    )

    print(f"[PRED] session={session_id} prob={prob:.3f} pred={text_label}")

    return jsonify({
        "status": "ok",
        "sessionId": session_id,
        "window_sec": WINDOW_SEC,
        "target_hz": TARGET_HZ,
        "prob_label1": prob,      # label=1 확률 (이걸 abnormal 쪽으로 쓰는지, normal로 쓰는지만 프로젝트 정의에 맞게)
        "threshold": THRESHOLD,
        "pred_label": int(pred),  # 0 or 1
        "pred_text": text_label,  # "normal" / "abnormal"
    })


if __name__ == "__main__":
    # 예: 0.0.0.0:8000 에서 리슨
    app.run(host="0.0.0.0", port=8000, debug=False)

