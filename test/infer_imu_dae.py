#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMU 균형 이상 감지 — Discriminative AE 추론 스크립트 (inference)

- 학습 시 사용한 AE1D + LatentClassifier 구조 재구성
- 학습 시 저장한 ckpt(.pt) 로드
- 새로 들어온 IMU ZIP 데이터를 window로 쪼개서 모델 추론
- 각 window에 대해 이상 확률 / 라벨(0=정상, 1=이상) 출력

※ 짧은 시퀀스(예: 29.99초처럼 아주 살짝 부족한 경우)는
   마지막 샘플을 반복해서 패딩해 30초 윈도우를 만들도록 함.
"""

import os
import io
import csv
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA


# ================== 공통 유틸 ==================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def parse_timestamp_auto(v):
    try:
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            if   iv >= 10**17: return pd.to_datetime(iv, unit='ns',  utc=True)
            elif iv >= 10**14: return pd.to_datetime(iv, unit='us',  utc=True)
            elif iv >= 10**11: return pd.to_datetime(iv, unit='ms',  utc=True)
            elif iv >= 10**9:  return pd.to_datetime(iv, unit='s',   utc=True)
            else:              return pd.to_datetime(iv, unit='s',   utc=True)
        s = str(v).strip()
        if s.isdigit():
            iv = int(s)
            if   iv >= 10**17: return pd.to_datetime(iv, unit='ns',  utc=True)
            elif iv >= 10**14: return pd.to_datetime(iv, unit='us',  utc=True)
            elif iv >= 10**11: return pd.to_datetime(iv, unit='ms',  utc=True)
            elif iv >= 10**9:  return pd.to_datetime(iv, unit='s',   utc=True)
            else:              return pd.to_datetime(iv, unit='s',   utc=True)
        try:
            fv = float(s)
            return pd.to_datetime(fv, unit='s', utc=True)
        except Exception:
            return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT


def window_stack(arr: np.ndarray, win_len: int, stride: int):
    """
    arr: [T, C]
    return:
      windows: [N, win_len, C]
      starts:  [N] (각 윈도우 시작 인덱스)
    """
    T = arr.shape[0]
    if T < win_len:
        return np.empty((0, win_len, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=int)
    starts = np.arange(0, T - win_len + 1, stride, dtype=int)
    windows = np.stack([arr[s:s + win_len] for s in starts], axis=0).astype(np.float32)
    return windows, starts


def resample_df(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    rule = pd.to_timedelta(1 / target_hz, unit="s")
    idx = pd.date_range(df.index.min(), df.index.max(), freq=rule)
    df = df.infer_objects(copy=False)
    df = df.reindex(df.index.union(idx)).interpolate(method='time').reindex(idx)
    return df


def _read_csv_robust(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    raw_bytes = raw_bytes.replace(b"\x00", b"")
    encodings = ['utf-8','utf-8-sig','cp949','euc-kr','utf-16','utf-16le','utf-16be','latin-1']

    def try_decode(enc):
        try:
            return raw_bytes.decode(enc)
        except Exception:
            return None

    def try_pd(txt, **kw):
        return pd.read_csv(io.StringIO(txt), engine='python', on_bad_lines='skip', **kw)

    # 1) 기본 시도
    for enc in encodings:
        txt = try_decode(enc)
        if txt is None:
            continue
        try:
            return try_pd(txt)
        except Exception:
            pass

    # 2) 헤더/구분자 추정 시도
    for enc in encodings:
        txt = try_decode(enc)
        if txt is None:
            continue
        try:
            sample = "\n".join(
                [ln for ln in txt.splitlines() if ln.strip()][:80]
            )
            sniff_sep = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except Exception:
            sniff_sep = None
        for sep in [sniff_sep, ',', '\t', ';', '|']:
            if not sep:
                continue
            try:
                return try_pd(txt, sep=sep)
            except Exception:
                pass

    return None


def _parse_sensor_csv(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    df = _read_csv_robust(raw_bytes)
    if df is None or df.empty:
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = None
    for cand in ['timestamp', 'time', 'datetime']:
        if cand in df.columns:
            tcol = cand
            break

    if tcol is None and 'seconds_elapsed' in df.columns:
        base = pd.Timestamp('1970-01-01', tz='UTC')
        df['timestamp'] = base + pd.to_timedelta(
            pd.to_numeric(df['seconds_elapsed'], errors='coerce'),
            unit='s'
        )
        tcol = 'timestamp'

    if tcol is None:
        return None

    def find_axis(cols, key):
        if key in cols:
            return key
        pat = re.compile(rf'(^|[^a-z]){key}([^a-z]|$)')
        for c in cols:
            if pat.search(c):
                return c
        return None

    cx = find_axis(df.columns, 'x')
    cy = find_axis(df.columns, 'y')
    cz = find_axis(df.columns, 'z')
    if not (cx and cy and cz):
        return None

    for c in [cx, cy, cz]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    out = df[[tcol, cx, cy, cz]].copy().rename(
        columns={tcol: 'timestamp', cx: 'x', cy: 'y', cz: 'z'}
    )
    out['timestamp'] = out['timestamp'].apply(parse_timestamp_auto)
    out = (
        out.dropna(subset=['timestamp'])
           .set_index('timestamp')
           .sort_index()
           .dropna(how='all')
    )
    return out if not out.empty else None


def align_axes_via_pca(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 10:
        return df

    acc_cols = ['ax', 'ay', 'az']
    if not all(c in df.columns for c in acc_cols):
        return df

    acc_data = df[acc_cols].values
    pca = PCA(n_components=3)
    pca.fit(acc_data)
    df[acc_cols] = pca.transform(acc_data)

    gyr_cols = ['gx', 'gy', 'gz']
    if all(c in df.columns for c in gyr_cols):
        R = pca.components_
        df[gyr_cols] = df[gyr_cols].values @ R.T

    return df


def read_all_series_from_zip(zip_path: Path, target_hz: int,
                             use_uncalibrated: bool, align_axes: bool):
    import zipfile
    acc_list, gyr_list = [], []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            name = info.filename
            base = os.path.basename(name)
            if info.is_dir() or base.startswith('.') or base.startswith('._'):
                continue
            low = name.lower()
            if not (low.endswith('.csv') or low.endswith('.tsv') or low.endswith('.txt')):
                continue
            if info.file_size == 0:
                print(f"[WARN] Empty file skipped: {zip_path.name}:{name}")
                continue

            is_uncalib = 'uncalibrated' in low
            if is_uncalib and not use_uncalibrated:
                continue

            is_acc = 'accelerometer' in low
            is_gyr = 'gyroscope' in low
            if not (is_acc or is_gyr):
                continue

            with z.open(name) as fbin:
                raw = fbin.read()
            df = _parse_sensor_csv(raw)
            if df is None or df.empty:
                print(f"[WARN] Failed to parse CSV: {zip_path.name}:{name}")
                continue

            if is_acc:
                acc_list.append(
                    df.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})[['ax', 'ay', 'az']]
                )
            else:
                gyr_list.append(
                    df.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'})[['gx', 'gy', 'gz']]
                )

    if not acc_list and not gyr_list:
        return None

    df_all = None
    if acc_list:
        df_all = pd.concat(acc_list).sort_index()
        df_all = df_all.groupby(level=0).mean()

    if gyr_list:
        g = pd.concat(gyr_list).sort_index()
        g = g.groupby(level=0).mean()
        df_all = g if df_all is None else df_all.join(g, how='outer')

    for c in SENSOR_COLS:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

    df_all = df_all.dropna(how='all')
    df_all = resample_df(df_all, target_hz).interpolate(limit_direction='both')

    if align_axes and not df_all.empty:
        df_all = align_axes_via_pca(df_all)

    return df_all


# ================== Dataset / Normalizer ==================

@dataclass
class WindowData:
    feats: np.ndarray   # [C, T]
    label: int          # dummy (0/1)
    group: str


class ChannelNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, items: List[WindowData]):
        if not items:
            return
        C = items[0].feats.shape[0]
        s  = np.zeros(C, dtype=np.float64)
        ss = np.zeros(C, dtype=np.float64)
        n  = 0
        for it in items:
            x = it.feats  # [C, T]
            s  += x.sum(axis=1)
            ss += (x ** 2).sum(axis=1)
            n  += x.shape[1]
        m = s / max(1, n)
        v = ss / max(1, n) - m ** 2
        self.mean = m.astype(np.float32)
        self.std  = np.sqrt(np.maximum(v, 1e-8)).astype(np.float32)


class WindowDatasetTorch(Dataset):
    def __init__(self, items: List[WindowData],
                 normalizer: Optional[ChannelNormalizer] = None):
        self.items = items
        self.norm  = normalizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        w = self.items[idx]
        x = torch.from_numpy(w.feats)
        if self.norm is not None and self.norm.mean is not None:
            mean = torch.from_numpy(self.norm.mean).view(-1, 1)
            std  = torch.from_numpy(self.norm.std ).view(-1, 1)
            x = (x - mean) / std
        # label은 추론에서는 의미 없지만, 기존 함수 재사용을 위해 0으로 유지
        return x, torch.tensor(w.label, dtype=torch.float32)


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_windows_from_series(df: pd.DataFrame, win_sec: int, stride_sec: int,
                              target_hz: int, group: str, trim_sec: int,
                              excluded_axes: List[str]):
    """
    학습 설정과 동일한 window_sec, stride_sec 사용.
    단, T가 win_len보다 아주 조금 부족할 경우(예: 2999 vs 3000)는
    마지막 샘플을 반복 패딩해 1개의 윈도우를 만들도록 함.
    """
    valid_cols = [c for c in SENSOR_COLS if c not in excluded_axes]
    for c in valid_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=valid_cols)
    if df.empty:
        return [], []

    # 앞뒤 trim (필요 없으면 trim_sec=0)
    if trim_sec and trim_sec > 0:
        n_trim = trim_sec * target_hz
        if len(df) > n_trim * 2:
            df = df.iloc[n_trim:-n_trim]
        else:
            # 너무 짧아지는 경우 윈도우 생성 불가
            return [], []

    arr = df[valid_cols].to_numpy(dtype=np.float32)  # [T, C]
    win_len = win_sec * target_hz
    stride  = stride_sec * target_hz

    # ---------- 짧은 시퀀스 패딩 허용 ----------
    T = arr.shape[0]
    print(f"[DEBUG] Effective length T={T}, win_len={win_len}")

    if T < win_len:
        # 예: 30초 윈도우인데 10초만 있는 경우는 버림
        min_ratio = 0.8  # 80% 이상이면 패딩 허용
        if T < int(min_ratio * win_len):
            print(f"[WARN] Sequence too short (T={T}) for window_len={win_len}")
            return [], []
        # 살짝 모자란 경우(예: 2999 vs 3000) → 마지막 샘플 반복 패딩
        pad_len = win_len - T
        pad = np.repeat(arr[-1:, :], pad_len, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
        T = arr.shape[0]
        print(f"[DEBUG] Padded length to T={T}")
    # -----------------------------------------

    ws, starts = window_stack(arr, win_len, stride)
    if ws.shape[0] == 0:
        print("[WARN] window_stack produced 0 windows")
        return [], []

    ws = np.transpose(ws, (0, 2, 1))  # [N, C, T]

    # 윈도우 시작/끝 timestamp 계산 (df.index 기준, 패딩 영역은 마지막 timestamp로 매핑)
    timestamps = df.index.to_numpy()
    win_meta = []
    for st in starts:
        st_idx = int(st)
        ed_idx = int(st + win_len - 1)
        st_idx = min(max(st_idx, 0), len(timestamps) - 1)
        ed_idx = min(max(ed_idx, 0), len(timestamps) - 1)
        win_meta.append((timestamps[st_idx], timestamps[ed_idx]))

    items = [
        WindowData(feats=ws[i], label=0, group=group)
        for i in range(ws.shape[0])
    ]
    return items, win_meta


# ================== Kalman + 모델 정의 ==================

class SimpleKalman:
    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var

    def filter(self, seq: np.ndarray) -> np.ndarray:
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


class Encoder1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, latent, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder1D(nn.Module):
    def __init__(self, latent=32, hidden=64, out_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent, hidden, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, out_ch, 7, padding=3),
        )

    def forward(self, z):
        return self.net(z)


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
        return z.mean(dim=-1)


class LatentClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def dae_logits_and_latent(ae: AE1D, clf_head: LatentClassifier,
                          loader: DataLoader, device: str,
                          kalman_proc=1e-3, kalman_meas=1e-2):
    ae.eval()
    clf_head.eval()
    probs_list = []

    for xb, _ in loader:
        xb = xb.to(device)
        xr, z = ae(xb)          # z: [B, D, T]
        z = z.cpu().numpy()
        B, D, T = z.shape
        z = np.transpose(z, (0, 2, 1))  # [B, T, D]

        zf = np.zeros_like(z)
        for b in range(B):
            zf[b] = SimpleKalman(D, kalman_proc, kalman_meas).filter(z[b])

        z_pool = zf.mean(axis=1)      # [B, D]
        logits = clf_head(
            torch.from_numpy(z_pool).float().to(device)
        ).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        probs_list.append(probs)

    if probs_list:
        return np.concatenate(probs_list)
    else:
        return np.array([])


# ================== 메인 추론 로직 ==================

@dataclass
class InferArgs:
    zip_path: str
    ckpt_path: str
    output_csv: str
    target_hz: int
    window_sec: int
    stride_sec: int
    trim_sec: int
    latent: int
    hidden: int
    device: str
    threshold: float
    use_uncalibrated: bool
    align_axes: bool
    exclude_axes: List[str]
    batch_size: int = 64


def run_inference(args: InferArgs):
    set_seed(42)
    device = (
        args.device
        if torch.cuda.is_available() and args.device.startswith('cuda')
        else 'cpu'
    )
    print(f"[INFO] Using device: {device}")

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    print("[INFO] Loading series from zip...")
    df = read_all_series_from_zip(
        zip_path,
        args.target_hz,
        use_uncalibrated=args.use_uncalibrated,
        align_axes=args.align_axes,
    )
    if df is None or df.empty:
        raise RuntimeError("Failed to build series from zip. Check file format.")

    print(
        f"[INFO] Data length (samples): {len(df)}, "
        f"time range: {df.index.min()} ~ {df.index.max()}"
    )

    print("[INFO] Building windows...")
    items, win_meta = build_windows_from_series(
        df,
        args.window_sec,
        args.stride_sec,
        target_hz=args.target_hz,
        group=zip_path.stem,
        trim_sec=args.trim_sec,
        excluded_axes=args.exclude_axes,
    )
    if not items:
        raise RuntimeError(
            "No windows created from series. Check window/stride/trim settings."
        )

    print(f"[INFO] Total windows: {len(items)}")

    n_input_ch = items[0].feats.shape[0]
    print(f"[INFO] Model input channels: {n_input_ch} (Excluded: {args.exclude_axes})")

    # 현재 데이터 기반으로 채널 통계 추정
    ch_norm = ChannelNormalizer()
    ch_norm.fit(items)

    ds = WindowDatasetTorch(items, ch_norm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    ae = AE1D(in_ch=n_input_ch, hidden=args.hidden, latent=args.latent).to(device)
    clf_head = LatentClassifier(in_dim=args.latent, hidden=args.hidden).to(device)

    ae.load_state_dict(ckpt["ae"])
    clf_head.load_state_dict(ckpt["clf_head"])

    print("[INFO] Running model inference...")
    probs = dae_logits_and_latent(ae, clf_head, loader, device)
    if probs.size == 0:
        raise RuntimeError("Inference produced no outputs.")

    preds = (probs >= args.threshold).astype(int)

    # 결과 DataFrame: 각 윈도우 시작/끝 시간, 확률, 라벨
    rows = []
    for i, (p, yhat) in enumerate(zip(probs, preds)):
        st, ed = win_meta[i]
        rows.append(
            {
                "window_idx": i,
                "start_time": st.isoformat(),
                "end_time": ed.isoformat(),
                "prob_abnormal": float(p),
                "label": int(yhat),  # 0=정상, 1=이상
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved inference results to: {out_path}")
    print(
        f"[INFO] Abnormal windows (label=1): {out_df['label'].sum()} / {len(out_df)} "
        f"(threshold={args.threshold})"
    )


# ================== CLI ==================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zip_path",
        type=str,
        required=True,
        help="추론할 IMU ZIP 파일 경로 (예: ./o_trimmed.zip)",
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="학습된 ckpt(.pt) 경로 (예: ./fold3_ckpt.pt)",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="./inference_result.csv",
        help="윈도우별 결과를 저장할 csv 경로",
    )

    # 학습 커맨드에 맞춘 기본값들
    p.add_argument("--target_hz", type=int, default=100)
    p.add_argument("--window_sec", type=int, default=30)
    p.add_argument("--stride_sec", type=int, default=15)
    # 추론에서는 기본적으로 trim 안 하는 쪽이 더 안전해서 0으로 설정
    p.add_argument("--trim_sec", type=int, default=0)

    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--threshold",
        type=float,
        default=0.355,
        help="이상 판정 기준 threshold (기본=0.355)",
    )

    p.add_argument(
        "--use_uncalibrated",
        action="store_true",
        help="Uncalibrated 센서 파일도 포함",
    )
    p.add_argument(
        "--align_axes",
        action="store_true",
        help="PCA로 축 정렬",
    )
    p.add_argument(
        "--exclude_axes",
        nargs="+",
        default=[],
        help="제외할 축 이름 리스트 (예: az gz)",
    )

    args_ns = p.parse_args()
    args = InferArgs(
        zip_path=args_ns.zip_path,
        ckpt_path=args_ns.ckpt_path,
        output_csv=args_ns.output_csv,
        target_hz=args_ns.target_hz,
        window_sec=args_ns.window_sec,
        stride_sec=args_ns.stride_sec,
        trim_sec=args_ns.trim_sec,
        latent=args_ns.latent,
        hidden=args_ns.hidden,
        device=args_ns.device,
        threshold=args_ns.threshold,
        use_uncalibrated=args_ns.use_uncalibrated,
        align_axes=args_ns.align_axes,
        exclude_axes=args_ns.exclude_axes,
        batch_size=args_ns.batch_size,
    )
    run_inference(args)

