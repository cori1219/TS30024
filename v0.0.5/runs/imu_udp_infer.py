#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP IMU → 40s 수집 → 앞/뒤 5s 컷(=30s) → 균일 리샘플 → (훈련 통계) z-score → Encoder
→ Latent Kalman smoothing → 시간평균(GAP) → MLP 분류기 → logit/prob 출력

※ 두 번째 학습 스크립트의 'test' 경로와 동일한 전처리/추론 파이프라인:
  - 채널별 정규화: 반드시 '훈련 폴드에서 추정한 mean/std'를 전달해야 함 (ax,ay,az,gx,gy,gz 순)
  - latent 칼만 스무딩: process_var=1e-3, measure_var=1e-2
  - 분류기: LatentClassifier(MLP)와 동일 구조(Linear 32→64→1)

실행 예:
  python imu_udp_infer.py --port 9000 --target-hz 50 \
    --ckpt fold3_ckpt.pt \
    --mean 0.01,0.00,-0.02,0.0,0.0,0.0 \
    --std  0.98,1.02,1.01,0.95,1.03,0.99
"""

import argparse
import socket
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================ 모델 정의 (학습 스크립트와 키/구조 호환) ============================

class Encoder(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B,6,T]
        return self.net(x)  # [B,32,T]


class Decoder(nn.Module):
    def __init__(self, out_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, out_ch, kernel_size=7, padding=3),
        )

    def forward(self, z):  # z: [B,32,T]
        return self.net(z)  # [B,6,T]


class AE(nn.Module):
    def __init__(self, in_ch=6, out_ch=6):
        super().__init__()
        self.enc = Encoder(in_ch)
        self.dec = Decoder(out_ch)

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z


class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
        )

    def forward(self, z_vec32):
        return self.net(z_vec32)


class ClfHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, z_vec32):
        return self.net(z_vec32)  # [B,1]


# -------- Kalman (두 번째 코드의 SimpleKalman과 동일 파라미터) --------
class SimpleKalman:
    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var

    def filter(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape
        out = np.zeros_like(seq)
        x = np.zeros(D, dtype=seq.dtype)
        p = np.ones(D, dtype=seq.dtype)
        for t in range(T):
            x_pred = x
            p_pred = p + self.q
            z = seq[t]
            k = p_pred / (p_pred + self.r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred
            out[t] = x
        return out


class InferenceModel(nn.Module):
    def __init__(self, ckpt_path, device="cpu"):
        super().__init__()
        self.ae = AE(in_ch=6, out_ch=6)
        self.proj_head = ProjHead()
        self.clf_head = ClfHead()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 예상 키: 'ae', 'proj_head', 'clf_head'
        self.ae.load_state_dict(ckpt["ae"], strict=True)
        self.proj_head.load_state_dict(ckpt["proj_head"], strict=True)
        self.clf_head.load_state_dict(ckpt["clf_head"], strict=True)

        self.to(device)
        self.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x):  # x: [B,6,T], z-kalman → GAP → clf
        z = self.ae.enc(x)  # [B,32,T]

        # ===== Kalman smoothing (test 경로와 동일) =====
        z_np = z.detach().cpu().numpy().transpose(0, 2, 1)  # [B,T,32]
        B, T, D = z_np.shape
        zf_np = np.zeros_like(z_np)
        for b in range(B):
            zf_np[b] = SimpleKalman(D, 1e-3, 1e-2).filter(z_np[b])
        zf = torch.from_numpy(zf_np.transpose(0, 2, 1)).to(self.device)  # [B,32,T]

        z_vec = zf.mean(dim=-1)              # GAP over time -> [B,32]
        logit = self.clf_head(z_vec)[:, 0]   # [B]
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob}


# ============================ UDP 수신 (정확히 40초 창) ============================

def parse_csv6(line: str):
    try:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 6:
            return None
        return [float(parts[i]) for i in range(6)]
    except Exception:
        return None


def recv_udp_40s(port: int, timeout: float = 1.0):
    """
    첫 유효 샘플을 받은 시점부터 40초간 수신.
    반환: times(초, 0 기준), data(N,6)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(timeout)

    buf = []
    t0 = None
    print(f"[INFO] Listening UDP :{port} ... (first valid sample starts the 40s window)")
    try:
        while True:
            try:
                pkt, _ = sock.recvfrom(4096)
            except socket.timeout:
                if t0 is None:
                    continue
                else:
                    continue

            line = pkt.decode("utf-8", errors="ignore")
            for raw in line.strip().splitlines():
                vals = parse_csv6(raw)
                if vals is None:
                    continue
                now = time.monotonic()
                if t0 is None:
                    t0 = now
                    print("[INFO] First sample received. Start 40s window.")
                t_rel = now - t0
                buf.append((t_rel, *vals))
                if t_rel >= 40.0:
                    times = np.array([b[0] for b in buf], dtype=np.float64)
                    data = np.array([b[1:] for b in buf], dtype=np.float32)  # (N,6)
                    order = np.argsort(times)
                    times = times[order] - times[order][0]
                    data = data[order]
                    print(f"[INFO] Collected {len(times)} samples over {times[-1]:.2f}s.")
                    return times, data
    finally:
        sock.close()


# ============================ 트리밍(5s/5s) + 균일 리샘플 ============================

def trim_and_resample(times, data, target_hz: int):
    """
    40s 중 앞뒤 5s 제거 → 30s 균일 시간축 생성 → 채널별 선형보간
    반환: x_trim (6, T_out), t_out
    """
    total = times[-1]
    t_start = 5.0
    t_end = max(5.0, total - 5.0)
    span = t_end - t_start
    if span < 10.0:
        raise RuntimeError(f"유효 구간이 너무 짧음: {span:.2f}s (입력 드랍/지연 확인)")

    # 중복 시간 제거
    t_in, idx = np.unique(times, return_index=True)
    x_in = data[idx]  # (N,6)

    T_out = int(round(30.0 * target_hz))  # 정확히 30초
    t_out = np.linspace(t_start, t_start + 30.0, num=T_out, endpoint=False)

    x_out = np.zeros((6, T_out), dtype=np.float32)
    for ch in range(6):
        x_out[ch] = np.interp(t_out, t_in, x_in[:, ch]).astype(np.float32)
    return x_out, t_out


# ============================ 정규화 ============================

def apply_norm_train_stats(x_6T, mean, std, eps=1e-6):
    """
    훈련 폴드에서 추정된 채널별 mean/std로 z-score (test 경로와 동일)
    x_6T: (6, T), mean/std: 길이 6 리스트
    """
    m = np.asarray(mean, dtype=np.float32).reshape(6, 1)
    s = np.asarray(std, dtype=np.float32).reshape(6, 1)
    return ((x_6T - m) / (s + eps)).astype(np.float32)


# ============================ 유틸: ckpt 확인 ============================

def resolve_ckpt_or_die(path_arg: str) -> str:
    p = Path(path_arg)
    if p.exists():
        return str(p)
    print(f"[ERR] checkpoint not found: {path_arg}")
    sys.exit(2)


# ============================ 메인 ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000, help="UDP 포트 (휴대폰이 이 포트로 송신)")
    ap.add_argument("--target-hz", type=int, default=50, help="리샘플 목표 Hz (기본 50)")
    ap.add_argument("--ckpt", type=str, default="fold3_ckpt.pt",
                    help="체크포인트 경로(기본: 현재 폴더의 fold3_ckpt.pt)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu 또는 cuda")

    # test 경로 동일화를 위해 mean/std 필수
    ap.add_argument("--mean", type=str, required=True, help="채널별 평균 6개, 예: 0,0,0,0,0,0 (ax,ay,az,gx,gy,gz)")
    ap.add_argument("--std", type=str, required=True, help="채널별 표준편차 6개, 예: 1,1,1,1,1,1 (ax,ay,az,gx,gy,gz)")

    args = ap.parse_args()

    # mean/std 파싱 (훈련 폴드 통계 그대로 넣어야 함)
    try:
        mean = [float(x) for x in args.mean.split(",")]
        std = [float(x) for x in args.std.split(",")]
        if len(mean) != 6 or len(std) != 6:
            print("[ERR] mean/std는 6개 값이어야 함. 순서: ax,ay,az,gx,gy,gz")
            sys.exit(2)
    except Exception:
        print("[ERR] mean/std 파싱 실패. 예: --mean 0,0,0,0,0,0 --std 1,1,1,1,1,1")
        sys.exit(2)

    # 체크포인트 확인
    ckpt_path = resolve_ckpt_or_die(args.ckpt)

    # 1) 수신(40초)
    times, data = recv_udp_40s(args.port)

    # 2) 트리밍 + 리샘플(30초)
    x_6T, _ = trim_and_resample(times, data, args.target_hz)

    # 3) (훈련 통계) 정규화 — test 경로와 동일
    x_6T = apply_norm_train_stats(x_6T, mean=mean, std=std)

    # 4) 모델 로드 & 추론 (latent Kalman 포함)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    model = InferenceModel(ckpt_path, device=args.device)
    with torch.no_grad():
        x = torch.from_numpy(x_6T[None, ...]).to(model.device)  # [1,6,T]
        out = model(x)
        logit = out["logit"].item()
        prob = out["prob"].item()

    # 5) 결과
    print("=== INFERENCE RESULT (30s window, TEST-MATCHED) ===")
    print(f"logit: {logit:.6f}")
    print(f"prob(sigmoid): {prob:.6f}")
    print(f"label(@0.5): {1 if prob >= 0.5 else 0}")


if __name__ == "__main__":
    main()


