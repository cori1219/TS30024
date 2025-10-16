#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모바일 추론용 TorchScript Lite(.ptl) 내보내기 — 정규화는 앱에서 수행.
- 입력: 정규화된 [1, 6, 1500] 텐서 (채널 순서 ax, ay, az, gx, gy, gz)
- 처리: AE 인코더 → latent 칼만 스무딩 → 평균 풀링 → 분류헤드 → sigmoid
- 출력: (prob, flag)  튜플
"""

from pathlib import Path
import torch

# >>> 네 학습 스크립트 모듈명으로 변경하세요.
#    예: from train_imu_dae import AE1D, LatentClassifier
from train_imu_dae import AE1D, LatentClassifier

# 설정
SAVE_DIR = Path("mobile_export"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = Path("runs/fold3_ckpt.pt")   # 성능 좋은 fold 선택
WINDOW_SEC = 30
TARGET_HZ = 50
C = 6
T = WINDOW_SEC * TARGET_HZ

# 모델 로드
latent, hidden = 32, 64   # 학습에 사용한 값과 동일
ckpt = torch.load(CKPT_PATH, map_location="cpu")

ae = AE1D(in_ch=C, hidden=hidden, latent=latent).eval()
clf = LatentClassifier(in_dim=latent, hidden=hidden).eval()
ae.load_state_dict(ckpt["ae"], strict=True)
clf.load_state_dict(ckpt["clf_head"], strict=True)

# 추론 래퍼 (정규화 없음)
class DAEInfer(torch.nn.Module):
    def __init__(self, ae, clf, proc=1e-3, meas=1e-2, threshold=0.5):
        super().__init__()
        self.ae, self.clf = ae, clf
        self.proc = float(proc); self.meas = float(meas)
        self.threshold = float(threshold)

    @torch.jit.export
    def kalman(self, z: torch.Tensor) -> torch.Tensor:  # z:[B,D,T]
        B, D, T = z.shape
        out = torch.zeros_like(z)
        x = torch.zeros(B, D); p = torch.ones(B, D)
        q = torch.tensor(self.proc); r = torch.tensor(self.meas)
        for t in range(T):
            x_pred, p_pred = x, p + q
            zt = z[:, :, t]
            k = p_pred / (p_pred + r)
            x = x_pred + k * (zt - x_pred)
            p = (1 - k) * p_pred
            out[:, :, t] = x
        return out

    def forward(self, x):                 # ★ 이미 정규화된 [B,6,T]가 들어옴
        _, z = self.ae(x)                 # [B,D,T]
        z = self.kalman(z)
        z_pool = z.mean(dim=-1)           # [B,D]
        prob = torch.sigmoid(self.clf(z_pool))  # [B]
        return prob, prob.ge(self.threshold)

# 스크립팅 & Lite 최적화
wrapped = DAEInfer(ae, clf).eval()
example = torch.zeros(1, C, T)           # [1,6,1500]
ts = torch.jit.trace(wrapped, example)
ts = torch.utils.mobile_optimizer.optimize_for_mobile(ts)

# 저장 (.ptl)
OUT = SAVE_DIR / "dae_infer.ptl"
ts._save_for_lite_interpreter(str(OUT))
print(f"[OK] Saved: {OUT}")

# 참고: 앱에 같이 넣을 파일
print("Put this into your app assets:")
print(" -", OUT)
print(" -", Path('runs_global/fold3/norm.json'))  # 앱에서 정규화용(mean/std) 읽기

