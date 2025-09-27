#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py에서 학습한 체크포인트(state_dict)를 읽어
1) 모바일용 TorchScript 모듈: android_export/dae_mobile.pt
2) 정규화 통계 JSON:        android_export/normalizer.json
을 생성합니다.

- main.py는 수정하지 않습니다.
- 현재 main.py가 AE1D + LatentClassifier(평균 풀링) 조합으로 동작한다는 가정입니다.
- 체크포인트에 channel_mean/std가 없으면 (0, 1) 기본값으로 저장합니다.

사용 예:
  python3 export_mobile.py
  python3 export_mobile.py --ckpt runs/fold3_ckpt.pt --target_hz 50 --win_sec 30
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn

# ---- main.py의 클래스 가져오기 (파일명은 main.py 그대로) ----
from main import AE1D, LatentClassifier  # 필요한 최소 구성만 import


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/fold3_ckpt.pt",
                   help="학습 체크포인트(.pt, state_dict 저장본)")
    p.add_argument("--outdir", type=str, default="android_export",
                   help="내보낼 디렉토리")
    p.add_argument("--latent", type=int, default=32,
                   help="main.py에서 사용한 latent 차원(분류기 입력과 일치해야 함)")
    p.add_argument("--hidden", type=int, default=64,
                   help="main.py에서 사용한 분류기 hidden 차원")
    p.add_argument("--target_hz", type=int, default=50,
                   help="앱에서 사용할 표본화 주파수(Hz)")
    p.add_argument("--win_sec", type=int, default=30,
                   help="앱에서 사용할 윈도우 길이(초)")
    p.add_argument("--in_channels", type=int, default=6,
                   help="채널 수 (ax,ay,az,gx,gy,gz = 6)")
    return p.parse_args()


def build_models(in_ch: int, latent: int, hidden: int):
    ae = AE1D(in_ch=in_ch, hidden=hidden, latent=latent)
    clf = LatentClassifier(in_dim=latent, hidden=hidden)
    return ae, clf


class MobileDAE(nn.Module):
    """
    모바일 추론용 래퍼
    입력 : x [B, C, T]  (정규화 이후)
    출력 : p [B]        (이상 확률)
    """
    def __init__(self, ae: AE1D, clf: LatentClassifier):
        super().__init__()
        self.ae = ae
        self.clf = clf

    def forward(self, x):
        _, z = self.ae(x)             # z: [B, D, T]
        z_pool = z.mean(dim=-1)       # mean pooling → [B, D]
        logits = self.clf(z_pool)     # [B]
        return torch.sigmoid(logits)  # [B]


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 모델 구성 & 가중치 로드
    ae, clf = build_models(args.in_channels, args.latent, args.hidden)
    state = torch.load(ckpt_path, map_location="cpu")

    # 가중치 키 확인 및 로드
    if "ae" in state:
        ae.load_state_dict(state["ae"], strict=True)
    else:
        raise RuntimeError("checkpoint에 'ae' 가 없습니다.")

    if "clf_head" in state:
        clf.load_state_dict(state["clf_head"], strict=True)
    else:
        raise RuntimeError("checkpoint에 'clf_head' 가 없습니다.")

    ae.eval(); clf.eval()

    # 2) TorchScript 변환
    T = args.target_hz * args.win_sec
    example = torch.randn(1, args.in_channels, T)
    module = MobileDAE(ae, clf)
    ts = torch.jit.trace(module, example)
    (outdir / "dae_mobile.pt").unlink(missing_ok=True)
    ts.save(outdir / "dae_mobile.pt")

    # 3) 정규화 통계 저장(없으면 기본값)
    mean = state.get("channel_mean", [0.0] * args.in_channels)
    std  = state.get("channel_std",  [1.0] * args.in_channels)
    stats = {
        "mean": mean,
        "std": std,
        "target_hz": args.target_hz,
        "win_sec": args.win_sec,
        "channels": ["ax","ay","az","gx","gy","gz"][:args.in_channels],
    }
    (outdir / "normalizer.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {outdir/'dae_mobile.pt'} , {outdir/'normalizer.json'}")
    print(f"[INFO] Input shape expected by app: [1, {args.in_channels}, {T}]")

if __name__ == "__main__":
    main()

