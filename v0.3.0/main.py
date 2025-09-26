#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMU 균형 이상 감지 — Discriminative AE (Recon + SupCon + BCE) + Kalman + K-Fold CV
Global 학습 + 채널 표준화 + PCA 2D 시각화 + 곡선 경계(RBF-SVM/LogReg)
Encoders: Conv1D / LSTM / TimeMixer / TSMixer
Pooling: mean / Attention(dot) / Multi-Head Attention(MHA)
"""

import os, io, csv, re, math, json, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score
)
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------ Repro ------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------ Time parser ------------------
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


SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def window_stack(arr: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    T = arr.shape[0]
    if T < win_len:
        return np.empty((0, win_len, arr.shape[1]), dtype=np.float32)
    starts = np.arange(0, T - win_len + 1, stride)
    return np.stack([arr[s:s+win_len] for s in starts], axis=0).astype(np.float32)


# ------------------ Robust CSV loader ------------------
FAILED_DIR = Path("./runs/failed_samples")

def _dump_head(raw_bytes: bytes, zip_name: str, inner_name: str):
    try:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        for enc in ['utf-8','utf-8-sig','cp949','euc-kr','utf-16','utf-16le','utf-16be','latin-1']:
            try:
                head = "\n".join(raw_bytes.decode(enc, errors="ignore").splitlines()[:50])
                (FAILED_DIR / f"{zip_name}__{os.path.basename(inner_name)}.head.txt").write_text(head, encoding="utf-8")
                break
            except: continue
    except: pass

def _read_csv_robust(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    raw_bytes = raw_bytes.replace(b"\x00", b"")
    encodings = ['utf-8','utf-8-sig','cp949','euc-kr','utf-16','utf-16le','utf-16be','latin-1']

    def try_decode(enc):
        try: return raw_bytes.decode(enc)
        except: return None

    def try_pd(txt, **kw):
        return pd.read_csv(io.StringIO(txt), engine='python', on_bad_lines='skip', **kw)

    for enc in encodings:
        txt = try_decode(enc)
        if txt is None: continue
        try: return try_pd(txt)
        except: pass

    for enc in encodings:
        txt = try_decode(enc)
        if txt is None: continue
        try:
            sample = "\n".join([ln for ln in txt.splitlines() if ln.strip()][:80])
            sniff_sep = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except: sniff_sep = None
        for sep in [sniff_sep, ',', '\t', ';', '|']:
            if not sep: continue
            try: return try_pd(txt, sep=sep)
            except: pass

    return None


# ------------------ Sensor CSV parser ------------------
def resample_df(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    rule = pd.to_timedelta(1/target_hz, unit="s")
    idx = pd.date_range(df.index.min(), df.index.max(), freq=rule)
    df = df.infer_objects(copy=False)
    df = df.reindex(df.index.union(idx)).interpolate(method='time').reindex(idx)
    return df

def _parse_sensor_csv(raw_bytes: bytes, zip_name: str, inner_name: str) -> Optional[pd.DataFrame]:
    df = _read_csv_robust(raw_bytes)
    if df is None or df.empty:
        _dump_head(raw_bytes, zip_name, inner_name); return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = None
    for cand in ['timestamp','time','datetime']:
        if cand in df.columns: tcol = cand; break
    if tcol is None and 'seconds_elapsed' in df.columns:
        base = pd.Timestamp('1970-01-01', tz='UTC')
        df['timestamp'] = base + pd.to_timedelta(pd.to_numeric(df['seconds_elapsed'], errors='coerce'), unit='s')
        tcol = 'timestamp'
    if tcol is None:
        _dump_head(raw_bytes, zip_name, inner_name); return None

    def find_axis(cols, key):
        if key in cols: return key
        pat = re.compile(rf'(^|[^a-z]){key}([^a-z]|$)')
        for c in cols:
            if pat.search(c): return c
        return None

    cx = find_axis(df.columns, 'x'); cy = find_axis(df.columns, 'y'); cz = find_axis(df.columns, 'z')
    if not (cx and cy and cz):
        _dump_head(raw_bytes, zip_name, inner_name); return None

    for c in [cx, cy, cz]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    out = df[[tcol, cx, cy, cz]].copy().rename(columns={tcol: 'timestamp', cx: 'x', cy: 'y', cz: 'z'})
    out['timestamp'] = out['timestamp'].apply(parse_timestamp_auto)
    out = out.dropna(subset=['timestamp']).set_index('timestamp').sort_index().dropna(how='all')
    return out if not out.empty else None


# ------------------ Read from zip & merge ------------------
def read_all_series_from_zip(zip_path: Path, target_hz: int):
    import zipfile
    acc_list, gyr_list = [], []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            name = info.filename; base = os.path.basename(name)
            if info.is_dir() or base.startswith('.') or base.startswith('._'): continue
            low = name.lower()
            if not (low.endswith('.csv') or low.endswith('.tsv') or low.endswith('.txt')): continue
            if info.file_size == 0:
                print(f"[WARN] Empty file skipped: {zip_path.name}:{name}"); continue
            is_acc = ('accelerometer' in low) and ('uncalibrated' not in low)
            is_gyr = ('gyroscope' in low) and ('uncalibrated' not in low)
            if not (is_acc or is_gyr): continue
            with z.open(name) as fbin:
                raw = fbin.read()
            df = _parse_sensor_csv(raw, zip_path.name, name)
            if df is None or df.empty:
                print(f"[WARN] Failed to parse CSV: {zip_path.name}:{name}"); continue
            if is_acc:
                acc_list.append(df.rename(columns={'x':'ax','y':'ay','z':'az'})[['ax','ay','az']])
            else:
                gyr_list.append(df.rename(columns={'x':'gx','y':'gy','z':'gz'})[['gx','gy','gz']])

    if not acc_list and not gyr_list: return []
    df_all = None
    if acc_list: df_all = pd.concat(acc_list).sort_index()
    if gyr_list:
        g = pd.concat(gyr_list).sort_index()
        df_all = g if df_all is None else df_all.join(g, how='outer')

    for c in SENSOR_COLS:
        if c in df_all.columns: df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
    df_all = df_all.dropna(how='all')
    df_all = resample_df(df_all, target_hz).interpolate(limit_direction='both')
    return [df_all]


# ------------------ Dataset ------------------
@dataclass
class WindowData:
    feats: np.ndarray   # [C, T]
    label: int          # 0/1
    group: str

class ChannelNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None
    def fit(self, items: List[WindowData]):
        C = items[0].feats.shape[0]
        s = np.zeros(C, dtype=np.float64); ss = np.zeros(C, dtype=np.float64); n = 0
        for it in items:
            x = it.feats; s += x.sum(axis=1); ss += (x**2).sum(axis=1); n += x.shape[1]
        m = s / max(1, n); v = ss / max(1, n) - m**2
        self.mean = m.astype(np.float32)
        self.std  = np.sqrt(np.maximum(v, 1e-8)).astype(np.float32)

class WindowDatasetTorch(Dataset):
    def __init__(self, items: List[WindowData], normalizer: Optional[ChannelNormalizer] = None):
        self.items = items; self.norm = normalizer
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        w = self.items[idx]
        x = torch.from_numpy(w.feats)
        if self.norm and self.norm.mean is not None:
            mean = torch.from_numpy(self.norm.mean).view(-1,1); std = torch.from_numpy(self.norm.std).view(-1,1)
            x = (x - mean) / std
        return x, torch.tensor(w.label, dtype=torch.float32)

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

def build_windows_from_series(df: pd.DataFrame, win_sec: int, stride_sec: int, label: int, target_hz: int, group: str, trim_sec: int = 5):
    for c in SENSOR_COLS:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=SENSOR_COLS)
    if df.empty: return []
    if trim_sec and trim_sec > 0:
        n_trim = trim_sec * target_hz
        if len(df) > n_trim * 2: df = df.iloc[n_trim:-n_trim]
        else: return []
    arr = df[SENSOR_COLS].to_numpy(dtype=np.float32)
    win_len = win_sec * target_hz; stride = stride_sec * target_hz
    ws = window_stack(arr, win_len, stride)
    if ws.shape[0] == 0: return []
    ws = np.transpose(ws, (0, 2, 1))  # [N,C,T]
    return [WindowData(feats=ws[i], label=label, group=group) for i in range(ws.shape[0])]

def load_dataset(data_root: Path, target_hz: int, win_sec: int, stride_sec: int, trim_sec: int):
    items: List[WindowData] = []
    for lbl_name, lbl_val in [("o",1),("x",0)]:
        for zp in sorted((data_root / lbl_name).glob("*.zip")):
            for df in read_all_series_from_zip(zp, target_hz):
                items.extend(build_windows_from_series(df, win_sec, stride_sec, lbl_val, target_hz, group=zp.stem, trim_sec=trim_sec))
    return items


# ------------------ Simple Kalman ------------------
class SimpleKalman:
    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var; self.r = measure_var
    def filter(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape; out = np.zeros_like(seq)
        x = np.zeros(D); p = np.ones(D)
        for t in range(T):
            x_pred = x; p_pred = p + self.q
            z = seq[t]; k = p_pred / (p_pred + self.r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred
            out[t] = x
        return out


# ------------------ Models ------------------
# Conv1D AE
class Encoder1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.ReLU(),
        )
    def forward(self, x):  # [B,C,T]
        return self.net(x)  # [B,D,T]

class Decoder1D(nn.Module):
    def __init__(self, latent=32, hidden=64, out_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent, hidden, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, out_ch, 7, padding=3),
        )
    def forward(self, z): return self.net(z)

class AE1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.latent_dim = latent
        self.enc = Encoder1D(in_ch, hidden, latent)
        self.dec = Decoder1D(latent, hidden, in_ch)
    def forward(self, x):
        z = self.enc(x); xr = self.dec(z)
        return xr, z
    def pooled_latent(self, z):  # fallback
        return z.mean(dim=-1)

# LSTM AE
class EncoderLSTM(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        dir_mul = 2 if bidirectional else 1
        self.to_latent = nn.Linear(hidden*dir_mul, latent)

    def forward(self, x):  # x: [B,C,T]
        x_seq = x.transpose(1, 2)        # [B,T,C]
        h, _ = self.lstm(x_seq)          # [B,T,H*dir]
        z_seq = self.to_latent(h)        # [B,T,D]
        z = z_seq.transpose(1, 2)        # [B,D,T]
        return z, z_seq

class DecoderLSTM(nn.Module):
    def __init__(self, latent=32, hidden=64, out_ch=6, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.to_output = nn.Linear(hidden, out_ch)

    def forward(self, z_seq):           # z_seq: [B,T,D]
        y, _ = self.lstm(z_seq)         # [B,T,H]
        out_seq = self.to_output(y)     # [B,T,C]
        out = out_seq.transpose(1, 2)   # [B,C,T]
        return out

class AELSTM(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.latent_dim = latent
        self.enc = EncoderLSTM(in_ch, hidden, latent, num_layers, bidirectional, dropout)
        self.dec = DecoderLSTM(latent, hidden, in_ch, num_layers, dropout)

    def forward(self, x):
        z, z_seq = self.enc(x)     # z:[B,D,T], z_seq:[B,T,D]
        xr = self.dec(z_seq)       # [B,C,T]
        return xr, z

    def pooled_latent(self, z):    # fallback
        return z.mean(dim=-1)

# ===== TimeMixer Encoder =====
class MixerBlock(nn.Module):
    def __init__(self, dim, token_kernel=7, ch_hidden=128, dropout=0.0):
        super().__init__()
        pad = token_kernel // 2
        self.token_norm = nn.LayerNorm(dim)
        self.token_dw = nn.Conv1d(dim, dim, kernel_size=token_kernel, padding=pad, groups=dim)
        self.token_pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.token_dropout = nn.Dropout(dropout)
        self.ch_norm = nn.LayerNorm(dim)
        self.ch_mlp = nn.Sequential(
            nn.Conv1d(dim, ch_hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch_hidden, dim, 1), nn.Dropout(dropout),
        )
    def forward(self, x):  # x: [B, D, T]
        xt = x.transpose(1,2)
        xt = self.token_norm(xt).transpose(1,2)
        t = self.token_dw(xt); t = self.token_pw(t); t = self.token_dropout(t)
        x = x + t
        xc = x.transpose(1,2); xc = self.ch_norm(xc).transpose(1,2)
        c = self.ch_mlp(xc); x = x + c
        return x

class EncoderTimeMixer(nn.Module):
    def __init__(self, in_ch=6, latent=32, depth=4, token_kernel=7, ch_hidden=128, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, latent, kernel_size=1)
        self.blocks = nn.ModuleList([
            MixerBlock(latent, token_kernel=token_kernel, ch_hidden=ch_hidden, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.BatchNorm1d(latent)
    def forward(self, x):
        z = self.proj(x)
        for blk in self.blocks: z = blk(z)
        return self.norm_out(z)

class AETIMEMIX(nn.Module):
    def __init__(self, in_ch=6, latent=32, depth=4, token_kernel=7, ch_hidden=128, dropout=0.0, hidden=64):
        super().__init__()
        self.latent_dim = latent
        self.enc = EncoderTimeMixer(in_ch=in_ch, latent=latent, depth=depth,
                                    token_kernel=token_kernel, ch_hidden=ch_hidden, dropout=dropout)
        self.dec = Decoder1D(latent=latent, hidden=hidden, out_ch=in_ch)
    def forward(self, x):
        z = self.enc(x); xr = self.dec(z); return xr, z
    def pooled_latent(self, z): return z.mean(dim=-1)

# ===== TSMixer Encoder (All-MLP for Time Series) =====
class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class TSMixerBlock(nn.Module):
    """
    입력 z: [B, D, T]
    - Token Mixing: 시간축(T) MLP (채널별 독립)
    - Channel Mixing: 채널축(D) MLP (시점별 독립)
    """
    def __init__(self, dim, token_hidden_ratio=2.0, channel_hidden_ratio=2.0, dropout=0.0):
        super().__init__()
        self.token_hidden_ratio = token_hidden_ratio
        self.dropout = nn.Dropout(dropout)
        self.ln_channel = nn.LayerNorm(dim)
        ch_hidden = int(channel_hidden_ratio * dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, ch_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ch_hidden, dim), nn.Dropout(dropout),
        )

    def _token_mlp(self, x):            # x: [B, D, T]
        B, D, T = x.shape
        # ✅ LN over T (마지막 축이 T이므로 transpose 불필요)
        xt = nn.functional.layer_norm(x, (T,))   # [B, D, T]
        h = max(1, int(self.token_hidden_ratio * T))
        ff = FeedForward(T, h, T, dropout=self.dropout.p if hasattr(self.dropout, "p") else 0.0).to(x.device)
        y = xt.reshape(B*D, T)
        y = ff(y).reshape(B, D, T)
        return y

    def forward(self, z):               # [B, D, T]
        z = z + self._token_mlp(z)
        x = z.transpose(1, 2)           # [B, T, D]
        x = self.ln_channel(x)
        x = self.channel_mlp(x)         # [B, T, D]
        x = x.transpose(1, 2)           # [B, D, T]
        z = z + x
        return z

class EncoderTSMixer(nn.Module):
    def __init__(self, in_ch=6, latent=32, depth=4, token_hidden_ratio=2.0,
                 channel_hidden_ratio=2.0, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, latent, kernel_size=1)
        self.blocks = nn.ModuleList([
            TSMixerBlock(dim=latent, token_hidden_ratio=token_hidden_ratio,
                         channel_hidden_ratio=channel_hidden_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.BatchNorm1d(latent)
    def forward(self, x):
        z = self.proj(x)
        for blk in self.blocks: z = blk(z)
        return self.norm_out(z)

class AETSMIX(nn.Module):
    def __init__(self, in_ch=6, latent=32, depth=4,
                 token_hidden_ratio=2.0, channel_hidden_ratio=2.0,
                 dropout=0.0, hidden=64):
        super().__init__()
        self.latent_dim = latent
        self.enc = EncoderTSMixer(in_ch=in_ch, latent=latent, depth=depth,
                                  token_hidden_ratio=token_hidden_ratio,
                                  channel_hidden_ratio=channel_hidden_ratio,
                                  dropout=dropout)
        self.dec = Decoder1D(latent=latent, hidden=hidden, out_ch=in_ch)
    def forward(self, x):
        z = self.enc(x); xr = self.dec(z); return xr, z
    def pooled_latent(self, z): return z.mean(dim=-1)


# -------- Attention Poolers --------
class AttnPoolDot(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5
    def forward(self, z):  # z: [B, D, T]
        B, D, T = z.shape
        q = self.q.view(1, D, 1).expand(B, -1, -1)
        scores = (z * q).sum(1) * self.scale     # [B, T]
        w = torch.softmax(scores, dim=-1)        # [B, T]
        w = self.dropout(w)
        ctx = (z * w.unsqueeze(1)).sum(-1)       # [B, D]
        return ctx

class AttnPoolMHA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, z):  # z: [B, D, T]
        x = z.transpose(1, 2)                # [B, T, D]
        B, T, D = x.shape
        cls = self.cls.expand(B, -1, -1)     # [B,1,D]
        seq = torch.cat([cls, x], dim=1)     # [B,1+T,D]
        out, _ = self.mha(seq, seq, seq)     # [B,1+T,D]
        out = self.ln(out)
        cls_out = out[:, 0, :]               # [B,D]
        return cls_out


# -------- Heads & Loss --------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, features, labels):
        sim = torch.matmul(features, features.t()) / self.t
        y = labels.view(-1,1)
        mask_pos = (y == y.t()).float()
        mask_pos.fill_diagonal_(0)
        mask_no_self = torch.ones_like(sim); mask_no_self.fill_diagonal_(0)
        log_prob = sim - torch.log((torch.exp(sim) * mask_no_self).sum(dim=1, keepdim=True) + 1e-12)
        denom = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / torch.clamp(denom, min=1.0)
        loss[denom == 0] = 0.0
        return loss.mean()

class LatentClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------ Training ------------------
def train_discriminative_ae(
    ae, clf_head, proj_head, pooler, loader, valloader,
    epochs: int, lr: float, device: str,
    w_rec=1.0, w_con=0.5, w_cls=0.5, temperature=0.07
):
    ae.train(); clf_head.train(); proj_head.train(); pooler.train()
    opt = torch.optim.Adam(
        list(ae.parameters()) + list(clf_head.parameters()) +
        list(proj_head.parameters()) + list(pooler.parameters()),
        lr=lr
    )
    recon = nn.SmoothL1Loss()
    bce   = nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature)

    for ep in range(1, epochs+1):
        tr = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            xr, z = ae(xb)                 # z:[B,D,T]
            z_pool = pooler(z)             # [B,D]
            logits = clf_head(z_pool)      # [B]
            z_proj = proj_head(z_pool)

            loss = w_rec*recon(xr, xb) + w_con*supcon(z_proj, yb) + w_cls*bce(logits, yb)
            loss.backward(); opt.step()
            tr += loss.item() * xb.size(0); n += xb.size(0)

        val = math.nan
        if valloader is not None and len(valloader.dataset) > 0:
            ae.eval(); clf_head.eval(); proj_head.eval(); pooler.eval()
            s = 0.0; m = 0
            with torch.no_grad():
                for xb, yb in valloader:
                    xb = xb.to(device); yb = yb.to(device)
                    xr, z = ae(xb)
                    z_pool = pooler(z)
                    logits = clf_head(z_pool)
                    z_proj = proj_head(z_pool)
                    s += (w_rec*recon(xr, xb) + w_con*supcon(z_proj, yb) + w_cls*bce(logits, yb)).item() * xb.size(0); m += xb.size(0)
            val = s / max(1,m)
            ae.train(); clf_head.train(); proj_head.train(); pooler.train()
        print(f"[DAE] Epoch {ep}/{epochs} train={tr/max(1,n):.4f} val={val:.4f}")


# ------------------ Inference ------------------
@torch.no_grad()
def dae_logits_and_latent(ae, clf_head, pooler, loader, device, kalman_proc=1e-3, kalman_meas=1e-2):
    ae.eval(); clf_head.eval(); pooler.eval()
    probs_list=[]; y_list=[]; latent_list=[]
    for xb, yb in loader:
        xb = xb.to(device)
        xr, z = ae(xb)              # z:[B,D,T]
        # Kalman over time
        z_np = z.cpu().numpy().transpose(0,2,1)  # [B,T,D]
        zf = np.zeros_like(z_np)
        for b in range(z_np.shape[0]):
            zf[b] = SimpleKalman(z_np.shape[2], kalman_proc, kalman_meas).filter(z_np[b])
        z_smooth = torch.from_numpy(zf.transpose(0,2,1)).to(device)  # [B,D,T]
        z_pool = pooler(z_smooth).cpu()       # [B,D]
        logits = clf_head(z_pool).cpu().numpy()
        probs = 1/(1+np.exp(-logits))
        probs_list.append(probs); y_list.append(yb.numpy()); latent_list.append(z_pool.numpy())
    if probs_list:
        return np.concatenate(probs_list), np.concatenate(y_list), np.concatenate(latent_list)
    else:
        return np.array([]), np.array([]), np.zeros((0, ae.latent_dim), dtype=np.float32)


# ------------------ PCA + Boundary ------------------
def _plot_latent_pca_with_dae_boundary(
    X_latent_tr, y_true_tr, dae_pred_tr,
    X_latent_te, y_true_te, dae_pred_te,
    out_tr, out_te, title_tr, title_te,
    boundary_mode='rbf_svm', svm_c=1.0, svm_gamma='scale'
):
    if X_latent_tr.shape[0] == 0: return

    pca = PCA(n_components=2, random_state=42)
    Ztr = pca.fit_transform(X_latent_tr)
    Zte = pca.transform(X_latent_te) if X_latent_te.shape[0] else np.zeros((0,2))

    scaler = StandardScaler()
    Ztr_s = scaler.fit_transform(Ztr)
    Zte_s = scaler.transform(Zte) if Zte.shape[0] else np.zeros((0,2))

    lab_tr = dae_pred_tr.astype(int); source = "DAE"
    if np.unique(lab_tr).size < 2:
        if np.unique(y_true_tr.astype(int)).size >= 2:
            lab_tr = y_true_tr.astype(int); source = "TRUE"
        else:
            lab_tr = None

    clf = None; tag = None
    if lab_tr is not None:
        if boundary_mode == 'rbf_svm':
            clf = SVC(kernel='rbf', C=svm_c, gamma=svm_gamma, class_weight='balanced').fit(Ztr_s, lab_tr)
            tag = f"SVM({source}, C={svm_c}, gamma={svm_gamma})"
        else:
            clf = LogisticRegression(class_weight='balanced', max_iter=200).fit(Ztr_s, lab_tr)
            tag = f"LogReg({source})"

    def plot_one(Zs, y_true, fname, title):
        if Zs.shape[0]==0: return
        plt.figure()
        plt.scatter(Zs[y_true==0,0], Zs[y_true==0,1], s=12, label='normal (true)')
        plt.scatter(Zs[y_true==1,0], Zs[y_true==1,1], s=12, label='abnormal (true)')
        if clf is not None and Zs.shape[0]>5:
            x_min, x_max = np.percentile(Zs[:,0], 1), np.percentile(Zs[:,0], 99)
            y_min, y_max = np.percentile(Zs[:,1], 1), np.percentile(Zs[:,1], 99)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                                 np.linspace(y_min, y_max, 400))
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = clf.decision_function(grid).reshape(xx.shape)
            plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 20), alpha=0.25)
            cs = plt.contour(xx, yy, zz, levels=[0.0], linewidths=2)
            if cs.collections and tag: cs.collections[0].set_label(tag)
        plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

    plot_one(Ztr_s, y_true_tr, out_tr, title_tr)
    plot_one(Zte_s, y_true_te, out_te, title_te)


# ------------------ Confusion plot ------------------
def _plot_confusion(y_true, y_pred, outpath, title):
    if y_true.size==0 or y_pred.size==0: return
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title(title)
    plt.xticks([0,1], ['normal','abnormal']); plt.yticks([0,1], ['normal','abnormal'])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i,j]), ha='center', va='center')
    plt.xlabel('Predicted (DAE)'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()


# ------------------ Args ------------------
@dataclass
class Args:
    data_root: str; target_hz: int; window_sec: int; stride_sec: int; trim_sec: int
    epochs_ae: int; batch_size: int; lr: float; folds: int; latent: int; hidden: int
    device: str; save_dir: str
    boundary_mode: str; svm_c: float; svm_gamma: Union[str,float]
    encoder: str; lstm_layers: int; lstm_bidir: bool; lstm_dropout: float
    attn_pool: str; attn_heads: int; attn_dropout: float
    tm_depth: int; tm_kernel: int; tm_ch_hidden: int; tm_dropout: float
    tsm_depth: int; tsm_token_ratio: float; tsm_channel_ratio: float; tsm_dropout: float

def _parse_gamma(g: str) -> Union[str,float]:
    try: return float(g)
    except: return g


# ------------------ Run ------------------
def run(args: Args):
    set_seed(42)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    items = load_dataset(Path(args.data_root), args.target_hz, args.window_sec, args.stride_sec, args.trim_sec)
    if len(items)==0: raise RuntimeError("No usable windows built. Check data paths and CSV format.")
    print(f"[INFO] Total windows: {len(items)} (pos={sum(i.label for i in items)}, neg={len(items)-sum(i.label for i in items)})")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    os.makedirs(args.save_dir, exist_ok=True)
    svm_gamma_final = _parse_gamma(str(args.svm_gamma))
    cv_reports = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(items))), 1):
        print(f"\n======= Fold {fold}/{args.folds} (GLOBAL) =======")
        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        ch_norm = ChannelNormalizer(); ch_norm.fit(train_items)

        # --- 모델 선택 ---
        if args.encoder == 'lstm':
            ae = AELSTM(in_ch=len(SENSOR_COLS), hidden=args.hidden, latent=args.latent,
                        num_layers=args.lstm_layers, bidirectional=args.lstm_bidir,
                        dropout=args.lstm_dropout).to(device)
        elif args.encoder == 'timemixer':
            ae = AETIMEMIX(in_ch=len(SENSOR_COLS), latent=args.latent,
                           depth=args.tm_depth, token_kernel=args.tm_kernel,
                           ch_hidden=args.tm_ch_hidden, dropout=args.tm_dropout,
                           hidden=args.hidden).to(device)
        elif args.encoder == 'tsmixer':
            ae = AETSMIX(in_ch=len(SENSOR_COLS), latent=args.latent,
                         depth=args.tsm_depth,
                         token_hidden_ratio=args.tsm_token_ratio,
                         channel_hidden_ratio=args.tsm_channel_ratio,
                         dropout=args.tsm_dropout,
                         hidden=args.hidden).to(device)
        else:
            ae = AE1D(in_ch=len(SENSOR_COLS), hidden=args.hidden, latent=args.latent).to(device)

        # --- Attention Pooler ---
        if args.attn_pool == 'mha':
            pooler = AttnPoolMHA(dim=args.latent, num_heads=args.attn_heads, dropout=args.attn_dropout).to(device)
        elif args.attn_pool == 'dot':
            pooler = AttnPoolDot(dim=args.latent, dropout=args.attn_dropout).to(device)
        else:
            class MeanPool(nn.Module):
                def forward(self, z): return z.mean(dim=-1)
            pooler = MeanPool().to(device)

        clf_head = LatentClassifier(in_dim=args.latent, hidden=args.hidden).to(device)
        proj_head = ProjectionHead(in_dim=args.latent, proj_dim=64).to(device)

        tr_loader = DataLoader(WindowDatasetTorch(train_items, ch_norm), batch_size=args.batch_size,
                               shuffle=True, drop_last=False, collate_fn=collate_fn)
        va_loader = DataLoader(WindowDatasetTorch(val_items, ch_norm),   batch_size=args.batch_size,
                               shuffle=False, drop_last=False, collate_fn=collate_fn)

        print("[INFO] Train Discriminative AE (recon + supcon + bce) + Pooling...")
        train_discriminative_ae(
            ae, clf_head, proj_head, pooler,
            tr_loader, va_loader,
            epochs=args.epochs_ae, lr=args.lr, device=device,
            w_rec=1.0, w_con=0.5, w_cls=0.5, temperature=0.07
        )

        # --- 평가 ---
        print("[INFO] Evaluate & collect latent...")
        p_tr, y_tr, Ztr = dae_logits_and_latent(ae, clf_head, pooler, tr_loader, device)
        p_te, y_te, Zte = dae_logits_and_latent(ae, clf_head, pooler, va_loader, device)

        def metrics(p, y):
            if y.size==0: return (math.nan, math.nan, math.nan, {}), np.array([], int)
            yhat = (p>=0.5).astype(int)
            try: roc = roc_auc_score(y, p)
            except: roc = float('nan')
            try: pr  = average_precision_score(y, p)
            except: pr = float('nan')
            f1 = f1_score(y.astype(int), yhat.astype(int), zero_division=0)
            rep = classification_report(y, yhat, output_dict=True, zero_division=0)
            return (roc, pr, f1, rep), yhat

        (roc_tr, pr_tr, f1_tr, rep_tr), y_tr_hat = metrics(p_tr, y_tr)
        (roc_te, pr_te, f1_te, rep_te), y_te_hat = metrics(p_te, y_te)

        print(f"[FOLD {fold}] ROC-AUC train/test = {roc_tr:.4f} / {roc_te:.4f}")
        print(f"[FOLD {fold}] PR-AUC  train/test = {pr_tr:.4f} / {pr_te:.4f}")
        print(f"[FOLD {fold}] F1      train/test = {f1_tr:.4f} / {f1_te:.4f}")
        enc_tag = f"{args.encoder.upper()} + {args.attn_pool.upper() if args.attn_pool!='none' else 'MEAN'}"

        fold_report = {
            "fold": fold,
            "encoder": enc_tag,
            "train": {"roc_auc": roc_tr, "pr_auc": pr_tr, "f1": f1_tr, "report": rep_tr},
            "test":  {"roc_auc": roc_te, "pr_auc": pr_te, "f1": f1_te, "report": rep_te},
        }
        with open(Path(args.save_dir)/f"fold{fold}_report.json",'w') as f: json.dump(fold_report, f, indent=2)
        torch.save({"ae": ae.state_dict(), "clf_head": clf_head.state_dict(),
                    "proj_head": proj_head.state_dict(), "pooler": pooler.state_dict()},
                   Path(args.save_dir)/f"fold{fold}_ckpt.pt")
        cv_reports.append(fold_report)

        # --- 시각화 ---
        figs_dir = Path(args.save_dir)/f"figs_fold{fold}"
        figs_dir.mkdir(parents=True, exist_ok=True)
        _plot_latent_pca_with_dae_boundary(
            Ztr, y_tr, (p_tr>=0.5).astype(int),
            Zte, y_te, (p_te>=0.5).astype(int),
            figs_dir/"train_latent_pca_boundary.png",
            figs_dir/"test_latent_pca_boundary.png",
            f"Fold {fold} — Train Latent (PCA+Std) + boundary [{enc_tag}]",
            f"Fold {fold} — Test Latent (PCA+Std) + boundary [{enc_tag}]",
            boundary_mode=args.boundary_mode, svm_c=args.svm_c, svm_gamma=svm_gamma_final
        )
        _plot_confusion(y_tr.astype(int), (p_tr>=0.5).astype(int),
                        figs_dir/"train_confusion_dae.png",
                        f"Fold {fold} — Train Confusion (DAE) [{enc_tag}]")
        _plot_confusion(y_te.astype(int), (p_te>=0.5).astype(int),
                        figs_dir/"test_confusion_dae.png",
                        f"Fold {fold} — Test Confusion (DAE) [{enc_tag}]")

    with open(Path(args.save_dir)/"cv_summary.json",'w') as f: json.dump(cv_reports, f, indent=2)
    print("[DONE] Global CV reports & figures saved to", args.save_dir)


# ------------------ CLI ------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--target_hz', type=int, default=50)
    p.add_argument('--window_sec', type=int, default=30)
    p.add_argument('--stride_sec', type=int, default=15)
    p.add_argument('--trim_sec', type=int, default=5)
    p.add_argument('--epochs_ae', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--latent', type=int, default=32)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./runs')

    # Boundary
    p.add_argument('--boundary_mode', type=str, default='rbf_svm', choices=['rbf_svm','logreg'])
    p.add_argument('--svm_c', type=float, default=2.0)
    p.add_argument('--svm_gamma', type=str, default='scale')

    # Encoder 선택
    p.add_argument('--encoder', type=str, default='tsmixer', choices=['lstm','conv','timemixer','tsmixer'])

    # LSTM 전용
    p.add_argument('--lstm_layers', type=int, default=1)
    p.add_argument('--lstm_bidir', action='store_true')
    p.add_argument('--lstm_dropout', type=float, default=0.0)

    # TimeMixer 전용
    p.add_argument('--tm_depth', type=int, default=4)
    p.add_argument('--tm_kernel', type=int, default=7)
    p.add_argument('--tm_ch_hidden', type=int, default=128)
    p.add_argument('--tm_dropout', type=float, default=0.0)

    # TSMixer 전용
    p.add_argument('--tsm_depth', type=int, default=4)
    p.add_argument('--tsm_token_ratio', type=float, default=2.0)
    p.add_argument('--tsm_channel_ratio', type=float, default=2.0)
    p.add_argument('--tsm_dropout', type=float, default=0.0)

    # Attention Pooling
    p.add_argument('--attn_pool', type=str, default='dot', choices=['none','dot','mha'],
                   help='시간축 풀링: none=mean, dot=dot-product attention, mha=multi-head attention')
    p.add_argument('--attn_heads', type=int, default=4, help='MHA head 수')
    p.add_argument('--attn_dropout', type=float, default=0.0)

    args_ns = p.parse_args()
    def _parse_gamma(g: str) -> Union[str,float]:
        try: return float(g)
        except: return g
    gamma_val = _parse_gamma(args_ns.svm_gamma)

    args = Args(
        data_root=args_ns.data_root, target_hz=args_ns.target_hz,
        window_sec=args_ns.window_sec, stride_sec=args_ns.stride_sec, trim_sec=args_ns.trim_sec,
        epochs_ae=args_ns.epochs_ae, batch_size=args_ns.batch_size, lr=args_ns.lr,
        folds=args_ns.folds, latent=args_ns.latent, hidden=args_ns.hidden,
        device=args_ns.device, save_dir=args_ns.save_dir,
        boundary_mode=args_ns.boundary_mode, svm_c=args_ns.svm_c, svm_gamma=gamma_val,
        encoder=args_ns.encoder, lstm_layers=args_ns.lstm_layers,
        lstm_bidir=args_ns.lstm_bidir, lstm_dropout=args_ns.lstm_dropout,
        attn_pool=args_ns.attn_pool, attn_heads=args_ns.attn_heads, attn_dropout=args_ns.attn_dropout,
        tm_depth=args_ns.tm_depth, tm_kernel=args_ns.tm_kernel,
        tm_ch_hidden=args_ns.tm_ch_hidden, tm_dropout=args_ns.tm_dropout,
        tsm_depth=args_ns.tsm_depth, tsm_token_ratio=args_ns.tsm_token_ratio,
        tsm_channel_ratio=args_ns.tsm_channel_ratio, tsm_dropout=args_ns.tsm_dropout
    )
    run(args)

