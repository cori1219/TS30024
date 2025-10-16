#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모바일 추론용 TorchScript Lite(.ptl) 내보내기 — '정규화는 앱에서 수행' 버전
- 입력: 정규화된 [1, 6, T] (채널: ax, ay, az, gx, gy, gz)
- 처리: AE 인코더 → latent 칼만 스무딩 → 평균풀링 → 분류헤드 → sigmoid
- 출력: (prob, flag) 튜플
- 기본 경로: ./runs (체크포인트), ./mobile_export (출력)
- 학습 코드 파일명: main.py (AE1D, LatentClassifier import)
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 네 학습 코드 파일명: main.py
from main import AE1D, LatentClassifier  # 변경 금지

# ---- 선택적 모바일 최적화 (버전에 따라 없을 수 있음) ----
def maybe_optimize_for_mobile(ts_module: torch.jit.ScriptModule):
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        return optimize_for_mobile(ts_module)
    except Exception:
        return ts_module  # 해당 버전에 없으면 패스

class DAEInfer(nn.Module):
    def __init__(self, ae: nn.Module, clf: nn.Module,
                 proc: float = 1e-3, meas: float = 1e-2, threshold: float = 0.5):
        super().__init__()
        self.ae = ae.eval()
        self.clf = clf.eval()
        # TorchScript/Mobile 친화: 상수는 버퍼로 등록
        self.register_buffer("proc_tensor", torch.tensor(proc, dtype=torch.float32))
        self.register_buffer("meas_tensor", torch.tensor(meas, dtype=torch.float32))
        self.register_buffer("threshold_tensor", torch.tensor(threshold, dtype=torch.float32))

    @torch.jit.export
    def kalman(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D, T]
        B, D, T = z.shape
        out = torch.zeros_like(z)
        x = torch.zeros(B, D, device=z.device, dtype=z.dtype)
        p = torch.ones(B, D, device=z.device, dtype=z.dtype)
        q = self.proc_tensor.to(z.device)
        r = self.meas_tensor.to(z.device)
        # for 루프는 TorchScript에서 OK
        for t in range(int(T)):
            x_pred = x
            p_pred = p + q
            zt = z[:, :, t]
            k = p_pred / (p_pred + r)
            x = x_pred + k * (zt - x_pred)
            p = (1 - k) * p_pred
            out[:, :, t] = x
        return out

    def forward(self, x: torch.Tensor):
        # x: [B,6,T] — 이미 (x-mean)/std로 정규화된 입력
        # ★ mobile-lite 제약 때문에 with torch.no_grad() 같은 컨텍스트 금지
        _, z = self.ae(x)                 # [B,D,T]
        z = self.kalman(z)
        z_pool = z.mean(dim=-1)           # [B,D]
        logits = self.clf(z_pool)         # [B]
        prob = torch.sigmoid(logits)      # [B]
        flag = (prob >= self.threshold_tensor).to(torch.int64)  # [B]
        return prob, flag

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default=str(THIS_DIR / "runs"),
                   help="체크포인트/정규화 통계 디렉토리 (기본: ./runs)")
    p.add_argument("--fold", type=int, default=3,
                   help="사용할 fold 번호 (기본: 3)")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="직접 체크포인트 경로 지정 (지정 시 fold/runs_dir 무시)")
    p.add_argument("--latent", type=int, default=32, help="latent dim (학습과 동일)")
    p.add_argument("--hidden", type=int, default=64, help="hidden dim (학습과 동일)")
    p.add_argument("--window_sec", type=int, default=30, help="윈도우 길이(초)")
    p.add_argument("--target_hz", type=int, default=50,  help="타깃 샘플레이트(Hz)")
    p.add_argument("--proc", type=float, default=1e-3,   help="Kalman process var")
    p.add_argument("--meas", type=float, default=1e-2,   help="Kalman measure var")
    p.add_argument("--threshold", type=float, default=0.5, help="이상 판단 임계값")
    p.add_argument("--out_dir", type=str, default=str(THIS_DIR / "mobile_export"),
                   help="모바일 .ptl 출력 디렉토리")
    p.add_argument("--out_name", type=str, default="dae_infer.ptl",
                   help="저장 파일명")
    return p.parse_args()

def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트 경로
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = runs_dir / f"fold{args.fold}_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {ckpt_path}")

    # 모델 구성
    C = 6
    T = args.window_sec * args.target_hz
    ae  = AE1D(in_ch=C, hidden=args.hidden, latent=args.latent).eval()
    clf = LatentClassifier(in_dim=args.latent, hidden=args.hidden).eval()

    # 가중치 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(ckpt["ae"], strict=True)
    clf.load_state_dict(ckpt["clf_head"], strict=True)

    # TorchScript 변환 (script 사용)
    wrapped = DAEInfer(ae, clf, proc=args.proc, meas=args.meas, threshold=args.threshold).eval()
    scripted = torch.jit.script(wrapped)

    # (선택) 모바일 최적화 — 버전에 없으면 그냥 패스
    scripted = maybe_optimize_for_mobile(scripted)

    # 저장 (lite 인터프리터)
    out_path = out_dir / args.out_name
    scripted._save_for_lite_interpreter(str(out_path))

    # 안내: norm.json (앱 정규화용)
    norm_json = runs_dir / f"fold{args.fold}" / "norm.json"
    print("\n[OK] TorchScript Lite saved:", out_path)
    print("[INFO] Put this into your app assets as well (for app-side normalization):")
    print("      ", norm_json if norm_json.exists() else "(norm.json not found; save it during training)")

if __name__ == "__main__":
    main()

