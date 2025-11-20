#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensor Logger HTTP Push → 1초마다 데이터 들어오면
40번(40초) 모아서 앞뒤 5초씩 자르고, 가운데 30초만 분류한 뒤 바로 종료하는 서버.

- 학습 설정과 정합:
  - target_hz = 100, window_sec = 30 → WIN_SAMPLES = 3000
  - 학습 때는 긴 시계열에서 앞뒤 trim_sec=5 잘라내고 30초 윈도우를 뽑았음.
  - 여기서는 40초 스트림을 받고, 앞뒤 5초씩 날려서 가운데 30초를 모델에 넣음.

- 모델:
  - Conv1D AE (AE1D) + Kalman + latent mean pooling + LatentClassifier
  - fold3_ckpt.pt + fold3_norm.npz 사용
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import math
import os
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


# ----------------- 학습 설정과 동일 -----------------
TARGET_HZ   = 100              # --target_hz
WINDOW_SEC  = 30               # --window_sec (모델 입력 30초)
TRIM_SEC    = 5                # 앞뒤 5초 컷
TOTAL_SEC   = WINDOW_SEC + 2*TRIM_SEC  # 40초

WIN_SAMPLES = TARGET_HZ * WINDOW_SEC   # 3000
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

LATENT_DIM  = 32              # --latent
HIDDEN_DIM  = 64              # --hidden

CKPT_PATH   = Path("./runs/fold3_ckpt.pt")
NORM_PATH   = Path("./runs/fold3_norm.npz")

THRESHOLD   = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- 모델 정의 (main.py와 동일 구조) -----------------
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
    """학습 시와 동일: Linear → ReLU → Dropout → Linear"""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # [B, D]
        return self.net(x).squeeze(-1)


class SimpleKalman:
    """잠재 시퀀스 Kalman smoothing."""
    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var
        self.dim = dim

    def filter(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: [T, D]
        """
        T, D = seq.shape
        out = np.zeros_like(seq)
        x = np.zeros(D)
        p = np.ones(D)
        for t in range(T):
            x_pred = x
            p_pred = p + self.q
            z = seq[t]
            k = p_pred / (p_pred + self.r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred
            out[t] = x
        return out


def resample_df(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """DatetimeIndex 기반 100 Hz resample + time interpolation."""
    rule = pd.to_timedelta(1 / target_hz, unit="s")
    idx = pd.date_range(df.index.min(), df.index.max(), freq=rule)
    df = df.infer_objects(copy=False)
    df = df.reindex(df.index.union(idx)).interpolate(method="time").reindex(idx)
    return df


def load_fold3_model_and_norm():
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    ae = AE1D(in_ch=len(SENSOR_COLS), hidden=HIDDEN_DIM, latent=LATENT_DIM).to(DEVICE)
    clf = LatentClassifier(in_dim=LATENT_DIM, hidden=HIDDEN_DIM).to(DEVICE)

    ae.load_state_dict(ckpt["ae"])
    clf.load_state_dict(ckpt["clf_head"])

    ae.eval()
    clf.eval()

    if not NORM_PATH.is_file():
        raise FileNotFoundError(
            f"Normalization stats not found: {NORM_PATH}. "
            f"Run export_fold3_norm.py first."
        )
    norm = np.load(NORM_PATH)
    mean = norm["mean"].astype(np.float32)
    std  = norm["std"].astype(np.float32)

    print(f"[INFO] Loaded fold3 model from {CKPT_PATH}")
    print(f"[INFO] Loaded fold3 norm from  {NORM_PATH}")
    print(f"[INFO] Device={DEVICE}, threshold={THRESHOLD:.3f}")
    return ae, clf, mean, std


AE_MODEL, CLF_HEAD, GLOBAL_MEAN, GLOBAL_STD = load_fold3_model_and_norm()


# ----------------- 세션 버퍼: 40번 요청 모으기 -----------------
class SessionState:
    def __init__(self):
        self.acc_rows: List[Tuple[int, float, float, float]] = []
        self.gyr_rows: List[Tuple[int, float, float, float]] = []
        self.batch_count: int = 0
        self.classified: bool = False
        self.result: Optional[dict] = None

    def append_payload(self, payload: List[dict]):
        """
        Sensor Logger payload 에서 acc/gyro 데이터만 뽑아서 row 리스트에 쌓기.
        """
        for entry in payload:
            if not isinstance(entry, dict):
                continue

            name = str(entry.get("name", "")).lower()
            t_raw = entry.get("time", None)
            if t_raw is None:
                continue

            try:
                t_ns = int(t_raw)
            except Exception:
                continue

            vals = entry.get("values", entry)
            try:
                x = float(vals.get("x", 0.0))
                y = float(vals.get("y", 0.0))
                z = float(vals.get("z", 0.0))
            except Exception:
                continue

            if "accelerometer" in name and "uncalibrated" not in name:
                self.acc_rows.append((t_ns, x, y, z))
            elif "gyroscope" in name and "uncalibrated" not in name:
                self.gyr_rows.append((t_ns, x, y, z))

    def merged_df(self) -> Optional[pd.DataFrame]:
        """
        누적된 acc_rows / gyr_rows -> timestamp index DF(ax..gz) 로 변환.
        """
        df_acc = None
        df_gyr = None

        if self.acc_rows:
            arr_acc = np.array(self.acc_rows, dtype=np.float64)
            df_acc = pd.DataFrame(arr_acc, columns=["time_ns", "ax", "ay", "az"])
            df_acc["timestamp"] = pd.to_datetime(df_acc["time_ns"].astype("int64"), unit="ns", utc=True)
            df_acc = df_acc.set_index("timestamp")[["ax", "ay", "az"]].sort_index()

        if self.gyr_rows:
            arr_gyr = np.array(self.gyr_rows, dtype=np.float64)
            df_gyr = pd.DataFrame(arr_gyr, columns=["time_ns", "gx", "gy", "gz"])
            df_gyr["timestamp"] = pd.to_datetime(df_gyr["time_ns"].astype("int64"), unit="ns", utc=True)
            df_gyr = df_gyr.set_index("timestamp")[["gx", "gy", "gz"]].sort_index()

        if df_acc is None and df_gyr is None:
            return None

        if df_acc is not None and df_gyr is not None:
            df = df_acc.join(df_gyr, how="outer")
        elif df_acc is not None:
            df = df_acc
            for c in ["gx", "gy", "gz"]:
                if c not in df.columns:
                    df[c] = np.nan
        else:
            df = df_gyr
            for c in ["ax", "ay", "az"]:
                if c not in df.columns:
                    df[c] = np.nan

        df = df.sort_index()
        df = df.astype(np.float32)
        df = df.interpolate(method="time").ffill().bfill()
        return df

    def build_middle_30s_window(self) -> np.ndarray:
        """
        40초 스트림이라고 가정하고:
        전체 시간축에서 앞뒤 5초씩 잘라서 가운데 30초만 남김.
        그 구간을 100Hz로 resample 해서 최소 3000샘플 확보한 뒤,
        정확히 마지막 3000샘플을 [T,6] float32 로 반환.
        """
        df = self.merged_df()
        if df is None or df.empty:
            raise ValueError("no data in session")

        t_min = df.index.min()
        t_max = df.index.max()

        # 가운데 30초 구간: [t_min + 5s, t_max - 5s]
        mid_start = t_min + pd.to_timedelta(TRIM_SEC, unit="s")
        mid_end   = t_max - pd.to_timedelta(TRIM_SEC, unit="s")

        df_mid = df[(df.index >= mid_start) & (df.index <= mid_end)]
        if df_mid.empty:
            raise ValueError("no data in middle 30s segment")

        # 100 Hz로 resample
        df_mid = resample_df(df_mid, TARGET_HZ)

        if len(df_mid) < WIN_SAMPLES:
            raise ValueError(f"middle segment too short after resample: {len(df_mid)} < {WIN_SAMPLES}")

        # 정확히 3000개로 맞추기 (더 많으면 뒤쪽 3000만 사용)
        df_mid = df_mid.iloc[-WIN_SAMPLES:]

        for c in SENSOR_COLS:
            if c not in df_mid.columns:
                df_mid[c] = np.nan
        df_mid = df_mid[SENSOR_COLS].interpolate(limit_direction="both")

        arr = df_mid.to_numpy(dtype=np.float32)  # [T, 6]
        return arr


SESSIONS: Dict[str, SessionState] = {}


# ----------------- 30초 윈도우 분류 -----------------
@torch.no_grad()
def classify_window(arr: np.ndarray) -> Tuple[float, int]:
    """
    arr: [T, 6] (T=3000)
    - fold3_norm.npz 기반 채널 정규화
    - AE + Kalman + mean pooling + classifier
    """
    if arr.shape != (WIN_SAMPLES, len(SENSOR_COLS)):
        raise ValueError(f"invalid window shape: {arr.shape}, expected ({WIN_SAMPLES}, 6)")

    x_np = (arr - GLOBAL_MEAN.reshape(1, -1)) / (GLOBAL_STD.reshape(1, -1) + 1e-6)
    x = torch.from_numpy(x_np.T).unsqueeze(0).float().to(DEVICE)  # [1, 6, T]

    _, z = AE_MODEL(x)                              # z: [1, D, T]
    z_np = z.squeeze(0).permute(1, 0).cpu().numpy() # [T, D]

    kal = SimpleKalman(dim=z_np.shape[1], process_var=1e-3, measure_var=1e-2)
    z_f = kal.filter(z_np)                          # [T, D]

    z_pool = z_f.mean(axis=0, keepdims=True)        # [1, D]
    z_pool_t = torch.from_numpy(z_pool).float().to(DEVICE)

    logit = CLF_HEAD(z_pool_t).item()
    prob = 1.0 / (1.0 + math.exp(-logit))
    label = int(prob >= THRESHOLD)
    return float(prob), label


# ----------------- Flask 서버 -----------------
app = Flask(__name__)


@app.route("/data", methods=["POST"])
def handle_data():
    """
    Sensor Logger HTTP Push 엔드포인트.

    - 같은 sessionId 로 들어오는 요청은 1초마다 온다고 가정.
    - 각 요청마다 payload 를 세션 버퍼에 쌓고, batch_count += 1.
    - batch_count < 40 → 무조건 buffering (예측 X)
    - batch_count == 40 → 지금까지 쌓인 데이터로
        - 앞뒤 5초 컷 (중앙 30초만 사용)
        - 30초 윈도우 분류
        - 결과 반환 + 프로세스 강제 종료
    """
    try:
        body = request.get_json(force=True)
    except Exception as e:
        print("[ERROR] JSON parse error:", e)
        return jsonify({"status": "error", "reason": "invalid_json"}), 400

    if not isinstance(body, dict):
        return jsonify({"status": "error", "reason": "body_not_object"}), 400

    session_id = str(body.get("sessionId", "default"))
    payload = body.get("payload", None)

    if not isinstance(payload, list) or len(payload) == 0:
        return jsonify({"status": "error", "reason": "missing_or_empty_payload"}), 400

    state = SESSIONS.get(session_id)
    if state is None:
        state = SessionState()
        SESSIONS[session_id] = state

    # 이미 분류 끝난 세션이면 같은 결과만 계속 리턴
    if state.classified and state.result is not None:
        return jsonify({
            "status": "ok",
            "done": True,
            "window_sec": WINDOW_SEC,
            "trim_sec": TRIM_SEC,
            "target_hz": TARGET_HZ,
            **state.result,
        })

    # 새 payload 누적
    state.append_payload(payload)
    state.batch_count += 1

    # 아직 40번 안 찼으면 → 무조건 버퍼링
    if state.batch_count < TOTAL_SEC:
        print(f"[INFO] session={session_id} buffering... batch={state.batch_count}/{TOTAL_SEC}")
        return jsonify({
            "status": "buffering",
            "done": False,
            "batch_count": state.batch_count,
            "total_required": TOTAL_SEC,
        })

    # 딱 40번째에서만 분류 시도
    try:
        arr = state.build_middle_30s_window()  # [3000, 6]
        prob, label = classify_window(arr)
        state.classified = True
        state.result = {
            "prob_label1": prob,
            "threshold": THRESHOLD,
            "label": int(label),
        }
        print(
            f"[RESULT] session={session_id} 40s(5s+30s+5s) classified: "
            f"prob_label1={prob:.4f}, label={label}"
        )

        resp = jsonify({
            "status": "ok",
            "done": True,
            "window_sec": WINDOW_SEC,
            "trim_sec": TRIM_SEC,
            "target_hz": TARGET_HZ,
            **state.result,
        })

        # 응답 보낸 뒤 프로세스 강제 종료 (Flask 서버 포함)
        def delayed_exit():
            time.sleep(0.5)  # 응답 나갈 시간 조금 주고
            os._exit(0)

        threading.Thread(target=delayed_exit, daemon=True).start()

        return resp

    except Exception as e:
        print(f"[ERROR] classification failed for session={session_id}: {e}")
        return jsonify({
            "status": "error",
            "done": False,
            "reason": str(e),
        }), 500


if __name__ == "__main__":
    # Sensor Logger HTTP Push 설정 예:
    #   URL: http://<서버 IP>:8000/data
    #   Batch period: 1s
    #
    # → 같은 sessionId 로 40번(대략 40초) 보내면
    #   가운데 30초(앞뒤 5초 컷)로 한 번만 분류하고
    #   결과 응답 후 프로세스가 자동 종료됨.
    app.run(host="0.0.0.0", port=8000, debug=True)

