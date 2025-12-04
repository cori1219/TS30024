#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
잘라놓은(뒤 5초 버리고 앞 30초만 남긴) zip 데이터들을
Discriminative AE + Kalman + classifier (fold*_ckpt.pt) 로 분류하는 스크립트.

- 학습 코드(main.py)에 정의된 함수/클래스를 그대로 재사용
- threshold = 0.485 고정
- 각 윈도우 / 각 zip 파일별 결과를 CSV로 저장
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 🔽 학습 코드 파일명이 main.py 라고 알려줬으니 이렇게 import
from main import (
    read_all_series_from_zip,
    build_windows_from_series,
    WindowData,
    ChannelNormalizer,
    WindowDatasetTorch,
    collate_fn,
    AE1D,
    LatentClassifier,
    ProjectionHead,
    dae_logits_and_latent,
)

THRESHOLD = 0.485  # 요청한 threshold 값


def build_infer_items(
    data_root: Path,
    target_hz: int,
    window_sec: int,
    stride_sec: int,
    trim_sec: int,
    use_uncalibrated: bool,
    align_axes: bool,
    excluded_axes: List[str],
) -> List[WindowData]:
    """
    data_root 아래의 *.zip 파일들에서 WindowData 리스트를 만드는 함수.

    - label 은 0으로 넣지만, 실제로는 추론에서 사용하지 않음.
    - group 필드에 zip stem(파일 이름)을 넣어서 나중에 zip 단위로 aggregate 가능하게 함.
    """
    items: List[WindowData] = []
    data_root = Path(data_root)

    zip_paths: List[Path] = []

    # 1) data_root 바로 아래 *.zip
    zip_paths.extend(sorted(data_root.glob("*.zip")))

    # 2) data_root/x, data_root/o 안의 zip 도 같이 처리하고 싶으면 유지
    if (data_root / "x").exists():
        zip_paths.extend(sorted((data_root / "x").glob("*.zip")))
    if (data_root / "o").exists():
        zip_paths.extend(sorted((data_root / "o").glob("*.zip")))

    # 중복 제거
    uniq: List[Path] = []
    seen = set()
    for zp in zip_paths:
        rp = zp.resolve()
        if rp not in seen:
            uniq.append(zp)
            seen.add(rp)

    if not uniq:
        raise RuntimeError(f"'{data_root}' 아래에서 zip 파일을 찾지 못했습니다.")

    print(f"[INFO] Found {len(uniq)} zip files for inference.")
    for zp in uniq:
        dfs = read_all_series_from_zip(
            zp,
            target_hz=target_hz,
            use_uncalibrated=use_uncalibrated,
            align_axes=align_axes,
        )
        for df in dfs:
            wins = build_windows_from_series(
                df=df,
                win_sec=window_sec,
                stride_sec=stride_sec,
                label=0,               # 추론이므로 더미 라벨
                target_hz=target_hz,
                group=zp.stem,         # zip 파일 이름으로 group
                trim_sec=trim_sec,     # ⚠ 이미 바깥에서 잘랐다면 0 권장
                excluded_axes=excluded_axes,
            )
            items.extend(wins)

    if not items:
        raise RuntimeError("윈도우가 하나도 만들어지지 않았습니다. (데이터 길이/윈도우 설정/trim_sec 확인)")

    print(f"[INFO] Total windows for inference: {len(items)}")
    return items


def load_model_from_ckpt(
    ckpt_path: Path,
    in_ch: int,
    latent: int,
    hidden: int,
    device: str = "cpu",
):
    """
    학습 때와 동일한 구조의 AE1D + LatentClassifier + ProjectionHead 를 만들고
    fold*_ckpt.pt 를 로드.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    ae = AE1D(in_ch=in_ch, hidden=hidden, latent=latent).to(device)
    clf_head = LatentClassifier(in_dim=latent, hidden=hidden).to(device)
    proj_head = ProjectionHead(in_dim=latent, proj_dim=64).to(device)

    ae.load_state_dict(ckpt["ae"])
    clf_head.load_state_dict(ckpt["clf_head"])
    proj_head.load_state_dict(ckpt["proj_head"])

    ae.eval()
    clf_head.eval()
    return ae, clf_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="잘라놓은 zip 들이 있는 폴더")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="예: ./runs/fold3_ckpt.pt")
    parser.add_argument("--out_dir", type=str, default="./infer_out_trimmed",
                        help="결과 CSV 저장 폴더")

    # 🔧 학습 커맨드에 맞춘 기본값들
    parser.add_argument("--target_hz", type=int, default=100)
    parser.add_argument("--window_sec", type=int, default=30)
    parser.add_argument("--stride_sec", type=int, default=15)

    # ⚠ 이미 zip 자체를 30초만 남기도록 잘라뒀다면 보통 0으로 두는 게 맞음
    parser.add_argument("--trim_sec", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_uncalibrated", action="store_true",
                        help="Uncalibrated (raw) 센서 파일도 포함하여 로드")
    parser.add_argument("--align_axes", action="store_true",
                        help="PCA를 이용해 센서 축을 정렬")
    parser.add_argument("--exclude_axes", nargs="+", default=[],
                        help="사용하지 않을 축 이름 (예: az gz)")

    args = parser.parse_args()

    device = args.device
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    data_root = Path(args.data_root)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) 윈도우 생성
    items = build_infer_items(
        data_root=data_root,
        target_hz=args.target_hz,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        trim_sec=args.trim_sec,
        use_uncalibrated=args.use_uncalibrated,
        align_axes=args.align_axes,
        excluded_axes=args.exclude_axes,
    )

    # 2) 채널 수 / Normalizer
    n_input_ch = items[0].feats.shape[0]
    print(f"[INFO] Model input channels: {n_input_ch} (Excluded axes: {args.exclude_axes})")

    normalizer = ChannelNormalizer()
    normalizer.fit(items)

    dataset = WindowDatasetTorch(items, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,         # 순서 유지 (items 순서 그대로)
        drop_last=False,
        collate_fn=collate_fn,
    )

    # 3) 모델 로드
    ae, clf_head = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        in_ch=n_input_ch,
        latent=args.latent,
        hidden=args.hidden,
        device=device,
    )

    # 4) 예측 (Kalman + classifier 포함)
    probs, y_dummy, latents = dae_logits_and_latent(
        ae=ae,
        clf_head=clf_head,
        loader=loader,
        device=device,
    )

    if probs.size == 0:
        raise RuntimeError("예측 결과가 없습니다. (윈도우/데이터 확인 필요)")

    # probs 순서는 dataset / items 순서와 같음 (shuffle=False)
    preds = (probs >= THRESHOLD).astype(int)

    # 5) 윈도우 단위 결과 정리
    rows = []
    group_counter = {}  # zip 별 window index 부여
    for item, p, pred in zip(items, probs, preds):
        g = item.group
        win_idx = group_counter.get(g, 0)
        group_counter[g] = win_idx + 1
        rows.append(
            {
                "group": g,             # zip 파일 이름 (stem)
                "window_index": win_idx,
                "prob": float(p),
                "pred": int(pred),      # 0: normal, 1: abnormal
            }
        )

    df_win = pd.DataFrame(rows)
    df_win.to_csv(out_dir / "window_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved window-level predictions to {out_dir / 'window_predictions.csv'}")

    # 6) zip(녹화) 단위 aggregate (옵션)
    #    - prob_mean, prob_max
    #    - prob_max >= THRESHOLD 이면 abnormal 로 판단
    agg = df_win.groupby("group")["prob"].agg(["mean", "max", "count"])
    agg["pred_by_max"] = (agg["max"] >= THRESHOLD).astype(int)
    agg = agg.reset_index()
    agg.to_csv(out_dir / "recording_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved recording-level predictions to {out_dir / 'recording_predictions.csv'}")

    print("[DONE] Inference finished.")


if __name__ == "__main__":
    main()

