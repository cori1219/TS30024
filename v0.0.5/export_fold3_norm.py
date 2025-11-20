#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fold3 학습 폴드의 채널별 정규화 통계(mean/std) 추출 후
./runs/fold3_norm.npz 로 저장.

※ main.py 는 수정하지 않고, import 만 해서 재사용.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from main import load_dataset, ChannelNormalizer  # 네가 준 main.py 기준


# --- 학습 때와 동일하게 맞춰야 하는 설정들 ---
DATA_ROOT   = Path("./raw_data")   # 너가 학습에 쓴 --data_root
TARGET_HZ   = 100                  # --target_hz
WINDOW_SEC  = 30                   # --window_sec
STRIDE_SEC  = 15                   # --stride_sec
TRIM_SEC    = 5                    # --trim_sec
FOLDS       = 4                    # --folds
FOLD        = 3                    # 우리가 export 하고 싶은 fold 번호

SAVE_DIR    = Path("./runs")       # --save_dir


def main():
    print("[INFO] Loading dataset (same as training)...")
    items = load_dataset(DATA_ROOT, TARGET_HZ, WINDOW_SEC, STRIDE_SEC, TRIM_SEC)
    if len(items) == 0:
        raise RuntimeError("No items loaded. Check DATA_ROOT and preprocessing settings.")

    all_idx = np.arange(len(items))
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), 1):
        if fold != FOLD:
            continue

        print(f"[INFO] Building ChannelNormalizer for fold {FOLD} (train only)...")
        train_items = [items[i] for i in train_idx]
        ch_norm = ChannelNormalizer()
        ch_norm.fit(train_items)

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_DIR / f"fold{FOLD}_norm.npz"
        np.savez(out_path, mean=ch_norm.mean, std=ch_norm.std)
        print(f"[DONE] Saved fold{FOLD} norm stats to: {out_path}")
        break
    else:
        raise RuntimeError(f"Fold {FOLD} not found in KFold split.")


if __name__ == "__main__":
    main()

