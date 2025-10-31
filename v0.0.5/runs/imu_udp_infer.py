#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
휴대폰 IMU(UDP) → PC 수신 40초 → 앞/뒤 5초 컷(=30초) → 모델 추론
- 체크포인트: (기본) 현재 폴더의 'fold3_ckpt.pt'만 사용. 없으면 에러.
- 분류 경로: enc(x) -> z(32,T) -> 시간평균(GAP) -> 32벡터 -> clf_head -> 로짓/확률
- 입력 포맷: UDP로 "ax,ay,az,gx,gy,gz" 한 줄(6개 float)

실행 예:
  python imu_udp_infer.py --port 9000 --target-hz 50 --seq-norm
"""

import argparse
import socket
import time
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================ 모델 정의 (체크포인트와 키 구조 일치) ============================

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


class InferenceModel(nn.Module):
    def __init__(self, ckpt_path, device="cpu"):
        super().__init__()
        self.ae = AE(in_ch=6, out_ch=6)
        self.proj_head = ProjHead()
        self.clf_head = ClfHead()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 예상 키: 'ae' (enc.*, dec.*), 'proj_head' (net.*), 'clf_head' (net.*)
        self.ae.load_state_dict(ckpt["ae"], strict=True)
        self.proj_head.load_state_dict(ckpt["proj_head"], strict=True)
        self.clf_head.load_state_dict(ckpt["clf_head"], strict=True)

        self.to(device)
        self.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x):  # x: [B,6,T]
        z = self.ae.enc(x)                 # [B,32,T]
        z_vec = z.mean(dim=-1)             # GAP over time -> [B,32]
        logit = self.clf_head(z_vec)[:, 0] # [B]
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
                    continue  # 아직 시작 전이면 계속 대기
                else:
                    continue  # 창 진행 중이면 타임아웃 무시하고 재시도

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

def apply_norm(x_6T, mean=None, std=None, seq_norm=False, eps=1e-6):
    """
    x_6T: (6, T)
    - mean/std 지정 시 채널별 고정 정규화
    - seq_norm=True 시 현재 시퀀스 통계로 z-score
    - 둘 다 없으면 그대로 반환
    """
    x = x_6T.copy()
    if mean is not None and std is not None:
        m = np.asarray(mean, dtype=np.float32).reshape(6, 1)
        s = np.asarray(std, dtype=np.float32).reshape(6, 1)
        x = (x - m) / (s + eps)
    elif seq_norm:
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True)
        x = (x - m) / (s + eps)
    return x


# ============================ 유틸: ckpt 확인 ============================

def resolve_ckpt_or_die(path_arg: str) -> str:
    p = Path(path_arg)
    if p.exists():
        return str(p)
    print("[ERR] 'fold3_ckpt.pt' not found in current folder. Put it here or pass --ckpt PATH.")
    sys.exit(2)


# ============================ 메인 ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000, help="UDP 포트 (휴대폰이 이 포트로 송신)")
    ap.add_argument("--target-hz", type=int, default=50, help="리샘플 목표 Hz (기본 50)")
    ap.add_argument("--ckpt", type=str, default="fold3_ckpt.pt",
                    help="체크포인트 경로(기본: 현재 폴더의 fold3_ckpt.pt)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu 또는 cuda")
    ap.add_argument("--mean", type=str, default=None, help="채널별 평균 6개, 예: 0,0,0,0,0,0")
    ap.add_argument("--std", type=str, default=None, help="채널별 표준편차 6개, 예: 1,1,1,1,1,1")
    ap.add_argument("--seq-norm", action="store_true", help="시퀀스 z-score 정규화")
    args = ap.parse_args()

    # mean/std 파싱
    mean = std = None
    if args.mean and args.std:
        try:
            mean = [float(x) for x in args.mean.split(",")]
            std = [float(x) for x in args.std.split(",")]
            if len(mean) != 6 or len(std) != 6:
                print("[ERR] mean/std는 6개 값이어야 함.")
                sys.exit(2)
        except Exception:
            print("[ERR] mean/std 파싱 실패.")
            sys.exit(2)

    # 체크포인트 확인 (현재 폴더만)
    ckpt_path = resolve_ckpt_or_die(args.ckpt)

    # 1) 수신(40초)
    times, data = recv_udp_40s(args.port)

    # 2) 트리밍 + 리샘플(30초)
    x_6T, _ = trim_and_resample(times, data, args.target_hz)

    # 3) 정규화
    x_6T = apply_norm(x_6T, mean=mean, std=std, seq_norm=args.seq_norm).astype(np.float32)

    # 4) 모델 로드 & 추론
    model = InferenceModel(ckpt_path, device=args.device)
    with torch.no_grad():
        x = torch.from_numpy(x_6T[None, ...]).to(model.device)  # [1,6,T]
        out = model(x)
        logit = out["logit"].item()
        prob = out["prob"].item()

    # 5) 결과
    print("=== INFERENCE RESULT (30s window) ===")
    print(f"logit: {logit:.6f}")
    print(f"prob(sigmoid): {prob:.6f}")


if __name__ == "__main__":
    main()

