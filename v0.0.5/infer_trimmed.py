#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
여러 data_root(예: ./1, ./2, ./3)에 있는 '*_trimmed.zip' 들을
Discriminative AE + Kalman + classifier (fold*_ckpt.pt) 로 한 번에 분류하는 스크립트.

- 학습 코드(main.py)에 정의된 함수/클래스를 그대로 재사용
- threshold = 0.485 고정
- 각 윈도우 / 각 zip 파일별 결과를 CSV로 저장
- 컬럼에 root(1/2/3 …)를 붙여서 어느 폴더에서 온 데이터인지 구분
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# main.py 에 있는 것들 그대로 재사용
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

THRESHOLD = 0.485  # 요청한 threshold


def build_infer_items_for_root(
    root_name: str,
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
    특정 data_root(예: ./1) 하나에 대해 WindowData 리스트를 만드는 함수.

    🔹 이제 '*_trimmed.zip' 인 파일만 사용함.
    - WindowData.group 에는 "rootName/zipStem" 형태로 저장해서
      나중에 zip 구분 + root 구분 둘 다 할 수 있게 한다.
    """
    items: List[WindowData] = []
    data_root = Path(data_root)

    zip_paths: List[Path] = []

    # 1) data_root 바로 아래 *_trimmed.zip 만 사용
    zip_paths.extend(sorted(data_root.glob("*_trimmed.zip")))

    # 2) data_root/x, data_root/o 안에서도 *_trimmed.zip 만 사용
    if (data_root / "x").exists():
        zip_paths.extend(sorted((data_root / "x").glob("*_trimmed.zip")))
    if (data_root / "o").exists():
        zip_paths.extend(sorted((data_root / "o").glob("*_trimmed.zip")))

    # 중복 제거
    uniq: List[Path] = []
    seen = set()
    for zp in zip_paths:
        rp = zp.resolve()
        if rp not in seen:
            uniq.append(zp)
            seen.add(rp)

    if not uniq:
        print(f"[WARN] '{data_root}' 아래에서 *_trimmed.zip 파일을 찾지 못했습니다. (root={root_name})")
        return []

    print(f"[INFO] [{root_name}] Found {len(uniq)} *_trimmed.zip files for inference.")
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
                group=f"{root_name}/{zp.stem}",  # root/zipStem 형태
                trim_sec=trim_sec,     # 이미 잘린 zip이라면 0 권장
                excluded_axes=excluded_axes,
            )
            items.extend(wins)

    print(f"[INFO] [{root_name}] Total windows for inference: {len(items)}")
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

    # ✅ 여러 data_root 를 한 번에 받도록 설정 (예: ./1 ./2 ./3)
    parser.add_argument(
        "--data_root",
        nargs="+",
        required=True,
        help="여러 data_root 폴더 (예: --data_root ./1 ./2 ./3)",
    )

    parser.add_argument("--ckpt", type=str, required=True,
                        help="예: ./runs/fold3_ckpt.pt")
    parser.add_argument("--out_dir", type=str, default="./infer_out_multi",
                        help="결과 CSV 저장 폴더")

    # 🔧 학습 커맨드에 맞춘 기본값들
    parser.add_argument("--target_hz", type=int, default=100)
    parser.add_argument("--window_sec", type=int, default=30)
    parser.add_argument("--stride_sec", type=int, default=15)

    # ⚠ 이미 *_trimmed.zip 이 30초 구간만 남도록 잘려있으면 보통 0으로 두는 게 맞음
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

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) 여러 root 에서 윈도우 모으기
    all_items: List[WindowData] = []
    root_list: List[str] = []

    for dr in args.data_root:
        root_name = Path(dr).name  # 예: './1' -> '1'
        items = build_infer_items_for_root(
            root_name=root_name,
            data_root=Path(dr),
            target_hz=args.target_hz,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            trim_sec=args.trim_sec,
            use_uncalibrated=args.use_uncalibrated,
            align_axes=args.align_axes,
            excluded_axes=args.exclude_axes,
        )
        all_items.extend(items)
        root_list.append(root_name)

    if not all_items:
        raise RuntimeError("어느 data_root에서도 윈도우가 만들어지지 않았습니다.")

    # 2) 채널 수 / Normalizer
    n_input_ch = all_items[0].feats.shape[0]
    print(f"[INFO] Model input channels: {n_input_ch} (Excluded axes: {args.exclude_axes})")
    print(f"[INFO] Total windows across roots {root_list}: {len(all_items)}")

    normalizer = ChannelNormalizer()
    normalizer.fit(all_items)

    dataset = WindowDatasetTorch(all_items, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,         # 순서 유지
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

    # 4) 예측
    probs, y_dummy, latents = dae_logits_and_latent(
        ae=ae,
        clf_head=clf_head,
        loader=loader,
        device=device,
    )

    if probs.size == 0:
        raise RuntimeError("예측 결과가 없습니다.")

    preds = (probs >= THRESHOLD).astype(int)

    # 5) 윈도우 단위 결과 정리
    rows = []
    group_counter = {}  # group(root/zipStem) 별 window index
    for item, p, pred in zip(all_items, probs, preds):
        # item.group 은 "rootName/zipStem" 형태
        group = item.group
        root_name, zip_stem = group.split("/", 1)

        win_idx = group_counter.get(group, 0)
        group_counter[group] = win_idx + 1

        rows.append(
            {
                "root": root_name,      # 1, 2, 3 ...
                "zip_stem": zip_stem,   # zip 파일 이름(stem)
                "group": group,         # root/zipStem
                "window_index": win_idx,
                "prob": float(p),
                "pred": int(pred),      # 0: normal, 1: abnormal
            }
        )

    df_win = pd.DataFrame(rows)
    df_win.to_csv(out_dir / "window_predictions.csv",
                  index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved window-level predictions to {out_dir / 'window_predictions.csv'}")

    # 6) recording(zip) 단위 aggregate
    agg = df_win.groupby(["root", "zip_stem"])["prob"].agg(["mean", "max", "count"])
    agg["pred_by_max"] = (agg["max"] >= THRESHOLD).astype(int)
    agg = agg.reset_index()
    agg.to_csv(out_dir / "recording_predictions.csv",
               index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved recording-level predictions to {out_dir / 'recording_predictions.csv'}")

    print("[DONE] Inference finished for roots:", ", ".join(root_list))


if __name__ == "__main__":
    main()

