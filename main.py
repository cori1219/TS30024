#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMU 균형 이상 감지 — Discriminative AE (Recon + SupCon + BCE) + Latent Kalman + K-Fold CV
[수정 반영 완료]
1. --use_uncalibrated: Uncalibrated csv 파일 로드 허용 (중복 타임스탬프는 평균값 병합으로 해결)
2. --align_axes: PCA를 통해 데이터의 축을 신호의 분산 방향으로 정렬 (센서 부착 방향 정보 제거)
3. --exclude_axes: 특정 축(예: az gz)을 입력 피처에서 제거
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

    # 1) 기본 시도
    for enc in encodings:
        txt = try_decode(enc)
        if txt is None: continue
        try: return try_pd(txt)
        except: pass

    # 2) 헤더/구분자 추정 시도
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
    # reindex 전에 중복 인덱스가 없음을 보장해야 함 (호출하는 쪽에서 처리)
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

# [MODIFIED] Helper: PCA-based rotation to align axes
def align_axes_via_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    가속도(ax, ay, az)의 주성분을 계산하여 데이터 전체를 회전시킵니다.
    이를 통해 센서 부착 방향에 따른 편향을 제거하고,
    데이터를 '신호의 분산이 큰 순서'로 정렬합니다.
    """
    if len(df) < 10: return df
    
    # 가속도 데이터 추출
    acc_cols = ['ax', 'ay', 'az']
    if not all(c in df.columns for c in acc_cols): return df
    
    acc_data = df[acc_cols].values
    
    # PCA 계산 (3축)
    pca = PCA(n_components=3)
    pca.fit(acc_data)
    
    # 가속도 회전 적용
    df[acc_cols] = pca.transform(acc_data)
    
    # 자이로도 동일한 회전 행렬 적용 (축의 방향이 같아야 하므로)
    gyr_cols = ['gx', 'gy', 'gz']
    if all(c in df.columns for c in gyr_cols):
        # Rotation Matrix = pca.components_
        R = pca.components_
        df[gyr_cols] = df[gyr_cols].values @ R.T

    return df


# ------------------ Read from zip & merge ------------------
# [MODIFIED] Added arguments: use_uncalibrated, align_axes, and duplicate fix
def read_all_series_from_zip(zip_path: Path, target_hz: int, use_uncalibrated: bool, align_axes: bool):
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
            
            # [MODIFIED] Uncalibrated logic
            is_uncalib = 'uncalibrated' in low
            if is_uncalib and not use_uncalibrated:
                continue # Skip if uncalibrated not allowed
            
            is_acc = 'accelerometer' in low
            is_gyr = 'gyroscope' in low
            
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
    
    # [FIX] Duplicate timestamp handling via groupby mean
    if acc_list: 
        df_all = pd.concat(acc_list).sort_index()
        # 중복된 인덱스(시간)가 있으면 평균을 내서 하나로 합침
        df_all = df_all.groupby(level=0).mean()

    if gyr_list:
        g = pd.concat(gyr_list).sort_index()
        g = g.groupby(level=0).mean()
        
        df_all = g if df_all is None else df_all.join(g, how='outer')

    for c in SENSOR_COLS:
        if c in df_all.columns: df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
    df_all = df_all.dropna(how='all')
    
    # 중복이 제거되었으므로 리샘플링 안전
    df_all = resample_df(df_all, target_hz).interpolate(limit_direction='both')

    # [MODIFIED] Apply Axis Alignment (Normalization) if requested
    if align_axes and not df_all.empty:
        df_all = align_axes_via_pca(df_all)

    return [df_all]


# ------------------ Dataset ------------------
@dataclass
class WindowData:
    feats: np.ndarray   # [C, T]
    label: int          # 0/1
    group: str          # 원 zip stem (참고용)

class ChannelNormalizer:
    """채널별 z-score 표준화: 훈련 폴드로 fit, (x-mean)/std"""
    def __init__(self):
        self.mean = None  # [C]
        self.std  = None  # [C]

    def fit(self, items: List[WindowData]):
        if not items: return
        C = items[0].feats.shape[0]
        s = np.zeros(C, dtype=np.float64)
        ss = np.zeros(C, dtype=np.float64)
        n = 0
        for it in items:
            x = it.feats  # [C,T]
            s  += x.sum(axis=1)
            ss += (x**2).sum(axis=1)
            n  += x.shape[1]
        m = s / max(1, n)
        v = ss / max(1, n) - m**2
        self.mean = m.astype(np.float32)
        self.std  = np.sqrt(np.maximum(v, 1e-8)).astype(np.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        if self.mean is None: return x
        mean = torch.from_numpy(self.mean).to(x.device).view(1, -1, 1)
        std  = torch.from_numpy(self.std ).to(x.device).view(1, -1, 1)
        return (x - mean) / std

class WindowDatasetTorch(Dataset):
    def __init__(self, items: List[WindowData], normalizer: Optional[ChannelNormalizer] = None):
        self.items = items; self.norm = normalizer
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        w = self.items[idx]
        x = torch.from_numpy(w.feats)  # [C,T]
        if self.norm is not None and self.norm.mean is not None:
            x = (x - torch.from_numpy(self.norm.mean).view(-1,1)) / torch.from_numpy(self.norm.std).view(-1,1)
        return x, torch.tensor(w.label, dtype=torch.float32)

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

# [MODIFIED] Added excluded_axes to remove specific channels
def build_windows_from_series(
    df: pd.DataFrame, win_sec: int, stride_sec: int, label: int, 
    target_hz: int, group: str, trim_sec: int, excluded_axes: List[str]
):
    # Determine valid columns (filtering out excluded axes)
    valid_cols = [c for c in SENSOR_COLS if c not in excluded_axes]

    for c in valid_cols:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=valid_cols)
    if df.empty: return []
    
    if trim_sec and trim_sec > 0:
        n_trim = trim_sec * target_hz
        if len(df) > n_trim * 2: df = df.iloc[n_trim:-n_trim]
        else: return []
        
    arr = df[valid_cols].to_numpy(dtype=np.float32)
    win_len = win_sec * target_hz; stride = stride_sec * target_hz
    ws = window_stack(arr, win_len, stride)
    if ws.shape[0] == 0: return []
    ws = np.transpose(ws, (0, 2, 1))  # [N,C,T]
    return [WindowData(feats=ws[i], label=label, group=group) for i in range(ws.shape[0])]

# [MODIFIED] Pass new arguments down
def load_dataset(data_root: Path, target_hz: int, win_sec: int, stride_sec: int, trim_sec: int,
                 use_uncalibrated: bool, align_axes: bool, excluded_axes: List[str]):
    items: List[WindowData] = []
    for lbl_name, lbl_val in [("o",1),("x",0)]:
        # 해당 폴더가 없으면 스킵
        if not (data_root / lbl_name).exists(): continue
        
        for zp in sorted((data_root / lbl_name).glob("*.zip")):
            # 1. Load & Align (with duplicate fix)
            dfs = read_all_series_from_zip(zp, target_hz, use_uncalibrated, align_axes)
            for df in dfs:
                # 2. Windowing & Column Filtering
                wins = build_windows_from_series(
                    df, win_sec, stride_sec, lbl_val, target_hz, 
                    group=zp.stem, trim_sec=trim_sec, excluded_axes=excluded_axes
                )
                items.extend(wins)
    return items


# ------------------ Simple Kalman (for latent smoothing) ------------------
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
class Encoder1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.ReLU(),
        )
    def forward(self, x):  # [B,C,T]
        return self.net(x)  # [B,latent,T]

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
    def pooled_latent(self, z):
        return z.mean(dim=-1)  # [B,latent]

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
        device = features.device
        sim = torch.matmul(features, features.t()) / self.t  # [B,B]
        y = labels.view(-1,1)
        mask_pos = (y == y.t()).float().to(device)
        mask_pos.fill_diagonal_(0)
        mask_no_self = torch.ones_like(sim, device=device)
        mask_no_self.fill_diagonal_(0)
        log_prob = sim - torch.log((torch.exp(sim) * mask_no_self).sum(dim=1, keepdim=True) + 1e-12)
        denom = mask_pos.sum(dim=1)
        loss = -(mask_pos * log_prob).sum(dim=1) / torch.clamp(denom, min=1.0)
        loss[denom == 0] = 0.0
        return loss.mean()

class LatentClassifier(nn.Module):
    """Latent → 이진 로짓. Dropout 추가로 약간의 regularization."""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # [B,in_dim]
        return self.net(x).squeeze(-1)


# ------------------ Training (Discriminative AE) ------------------
def train_discriminative_ae(
    ae: AE1D,
    clf_head: LatentClassifier,
    proj_head: ProjectionHead,
    loader: DataLoader,
    valloader: Optional[DataLoader],
    epochs: int, lr: float, device: str,
    # loss 비율: 분류쪽 가중치를 조금 더 키움
    w_rec: float = 0.5, w_con: float = 0.5, w_cls: float = 1.0,
    temperature: float = 0.07,
    # BCE class weight
    cls_w_pos: float = 1.0, cls_w_neg: float = 1.0
):
    ae.train(); clf_head.train(); proj_head.train()
    opt = torch.optim.Adam(list(ae.parameters()) + list(clf_head.parameters()) + list(proj_head.parameters()), lr=lr)
    recon = nn.SmoothL1Loss()
    supcon = SupConLoss(temperature)

    for ep in range(1, epochs+1):
        tr = 0.0; n = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            xr, z = ae(xb)               # xr:[B,C,T], z:[B,D,T]
            z_pool = ae.pooled_latent(z) # [B,D]
            logits = clf_head(z_pool)    # [B]
            z_proj = proj_head(z_pool)   # [B,P]

            loss_rec = recon(xr, xb)
            loss_con = supcon(z_proj, yb)

            # class-weighted BCE
            bce_raw = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none'
            )
            cls_weights = torch.where(
                yb > 0.5,
                torch.tensor(cls_w_pos, device=yb.device),
                torch.tensor(cls_w_neg, device=yb.device),
            )
            loss_cls = (bce_raw * cls_weights).mean()

            loss = w_rec*loss_rec + w_con*loss_con + w_cls*loss_cls
            loss.backward(); opt.step()
            tr += loss.item() * xb.size(0); n += xb.size(0)

        val = math.nan
        if valloader is not None and len(valloader.dataset) > 0:
            ae.eval(); clf_head.eval(); proj_head.eval()
            s = 0.0; m = 0
            with torch.no_grad():
                for xb, yb in valloader:
                    xb = xb.to(device); yb = yb.to(device)
                    xr, z = ae(xb)
                    z_pool = ae.pooled_latent(z)
                    logits = clf_head(z_pool)
                    z_proj = proj_head(z_pool)

                    loss_rec = recon(xr, xb)
                    loss_con = supcon(z_proj, yb)

                    bce_raw = nn.functional.binary_cross_entropy_with_logits(
                        logits, yb, reduction='none'
                    )
                    cls_weights = torch.where(
                        yb > 0.5,
                        torch.tensor(cls_w_pos, device=yb.device),
                        torch.tensor(cls_w_neg, device=yb.device),
                    )
                    loss_cls = (bce_raw * cls_weights).mean()

                    s += (w_rec*loss_rec + w_con*loss_con + w_cls*loss_cls).item() * xb.size(0); m += xb.size(0)
            val = s / max(1,m)
            ae.train(); clf_head.train(); proj_head.train()
        print(f"[DAE] Epoch {ep}/{epochs} train={tr/max(1,n):.4f} val={val:.4f}")


# ------------------ Inference helpers (DAE logits + latent) ------------------
@torch.no_grad()
def dae_logits_and_latent(ae: AE1D, clf_head: LatentClassifier, loader: DataLoader, device: str, kalman_proc=1e-3, kalman_meas=1e-2):
    ae.eval(); clf_head.eval()
    probs_list=[]; y_list=[]; latent_list=[]
    for xb, yb in loader:
        xb = xb.to(device)
        xr, z = ae(xb)        # z:[B,D,T]
        z = z.cpu().numpy()
        B,D,T = z.shape
        z = np.transpose(z, (0,2,1))  # [B,T,D]
        zf = np.zeros_like(z)
        for b in range(B):
            zf[b] = SimpleKalman(D, kalman_proc, kalman_meas).filter(z[b])
        z_pool = zf.mean(axis=1)              # [B,D]
        logits = clf_head(torch.from_numpy(z_pool).float().to(device)).cpu().numpy()
        probs = 1/(1+np.exp(-logits))
        probs_list.append(probs); y_list.append(yb.numpy()); latent_list.append(z_pool)
    if probs_list:
        return np.concatenate(probs_list), np.concatenate(y_list), np.concatenate(latent_list)
    else:
        return np.array([]), np.array([]), np.zeros((0, ae.latent_dim), dtype=np.float32)


# ------------------ PCA + Curved Boundary (with Standardization) ------------------
def _plot_latent_pca_with_dae_boundary(
    X_latent_tr: np.ndarray, y_true_tr: np.ndarray, dae_pred_tr: np.ndarray,
    X_latent_te: np.ndarray, y_true_te: np.ndarray, dae_pred_te: np.ndarray,
    out_tr: Path, out_te: Path, title_tr: str, title_te: str,
    boundary_mode: str = 'rbf_svm', svm_c: float = 1.0, svm_gamma: Union[str,float] = 'scale'
):
    if X_latent_tr.shape[0] == 0:
        return

    pca = PCA(n_components=2, random_state=42)
    Ztr = pca.fit_transform(X_latent_tr)
    Zte = pca.transform(X_latent_te) if X_latent_te.shape[0] else np.zeros((0, 2))

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
    mode = boundary_mode
    if lab_tr is not None:
        if mode in ('rbf_svm', 'auto'):
            try:
                clf = SVC(kernel='rbf', C=svm_c, gamma=svm_gamma, class_weight='balanced')
                clf.fit(Ztr_s, lab_tr); tag = f"SVM({source}, C={svm_c}, gamma={svm_gamma})"
            except Exception:
                clf = None
                if mode == 'auto': mode = 'logreg'
        if (mode == 'logreg') and (clf is None):
            try:
                clf = LogisticRegression(class_weight='balanced', max_iter=200)
                clf.fit(Ztr_s, lab_tr); tag = f"LogReg({source})"
            except Exception:
                clf = None

    def plot_one(Zs, y_true, fname, title):
        if Zs.shape[0] == 0: return
        plt.figure()
        idx0 = (y_true == 0); idx1 = (y_true == 1)
        plt.scatter(Zs[idx0,0], Zs[idx0,1], s=12, label='normal (true)')
        plt.scatter(Zs[idx1,0], Zs[idx1,1], s=12, label='abnormal (true)')
        if clf is not None and Zs.shape[0] > 5:
            x_min, x_max = np.percentile(Zs[:,0], 1), np.percentile(Zs[:,0], 99)
            y_min, y_max = np.percentile(Zs[:,1], 1), np.percentile(Zs[:,1], 99)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                                 np.linspace(y_min, y_max, 400))
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = clf.decision_function(grid).reshape(xx.shape)
            plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), zz.max(), 20), alpha=0.25)
            cs = plt.contour(xx, yy, zz, levels=[0.0], linewidths=2)
            if cs.collections and tag: cs.collections[0].set_label(tag)
        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(fname, dpi=150); plt.close()

    plot_one(Ztr_s, y_true_tr, out_tr, title_tr)
    plot_one(Zte_s, y_true_te, out_te, title_te)


# ------------------ Confusion matrix plot ------------------
def _plot_confusion(y_true, y_pred, outpath, title):
    if y_true.size == 0 or y_pred.size == 0: return
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xticks([0, 1], ['normal', 'abnormal'])
    plt.yticks([0, 1], ['normal', 'abnormal'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xlabel('Predicted (DAE)'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()


# ------------------ Args ------------------
@dataclass
class Args:
    data_root: str
    target_hz: int
    window_sec: int
    stride_sec: int
    trim_sec: int
    epochs_ae: int
    batch_size: int
    lr: float
    folds: int
    latent: int
    hidden: int
    device: str
    save_dir: str
    boundary_mode: str   # 'rbf_svm' | 'logreg' | 'auto'
    svm_c: float
    svm_gamma: Union[str, float]  # 'scale'|'auto'|float
    
    # [MODIFIED] New args
    use_uncalibrated: bool
    align_axes: bool
    exclude_axes: List[str]


def _parse_gamma(g: str) -> Union[str, float]:
    try: return float(g)
    except: return g


# ------------------ Run (Global CV with channel normalization) ------------------
def run(args: Args):
    set_seed(42)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    data_root = Path(args.data_root)
    print("[INFO] Loading dataset...")
    
    # [MODIFIED] Pass new arguments
    items = load_dataset(
        data_root, args.target_hz, args.window_sec, args.stride_sec, args.trim_sec,
        use_uncalibrated=args.use_uncalibrated,
        align_axes=args.align_axes,
        excluded_axes=args.exclude_axes
    )
    
    if len(items) == 0:
        raise RuntimeError("No usable windows built. Check data paths and CSV format.")
    print(f"[INFO] Total windows: {len(items)} (pos={sum(i.label for i in items)}, neg={len(items)-sum(i.label for i in items)})")

    # [MODIFIED] Calculate actual input channels
    n_input_ch = items[0].feats.shape[0]
    print(f"[INFO] Model input channels: {n_input_ch} (Excluded: {args.exclude_axes})")

    all_idx = np.arange(len(items))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    os.makedirs(args.save_dir, exist_ok=True)
    cv_reports = []

    svm_gamma_final = _parse_gamma(str(args.svm_gamma))

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), 1):
        print(f"\n======= Fold {fold}/{args.folds} (GLOBAL) =======")
        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        # --- 채널별 표준화 통계 (train fold) ---
        ch_norm = ChannelNormalizer(); ch_norm.fit(train_items)

        # --- class 비율로 BCE weight 계산 ---
        num_pos = sum(it.label for it in train_items)
        num_neg = len(train_items) - num_pos
        print(f"[FOLD {fold}] pos={num_pos}, neg={num_neg}")
        cls_w_pos = num_neg / (num_pos + 1e-8)
        cls_w_neg = num_pos / (num_neg + 1e-8)

        # --- 모델 (input_ch 반영) ---
        ae = AE1D(in_ch=n_input_ch, hidden=args.hidden, latent=args.latent).to(device)
        clf_head = LatentClassifier(in_dim=args.latent, hidden=args.hidden).to(device)
        proj_head = ProjectionHead(in_dim=args.latent, proj_dim=64).to(device)

        tr_loader = DataLoader(WindowDatasetTorch(train_items, ch_norm), batch_size=args.batch_size,
                               shuffle=True, drop_last=False, collate_fn=collate_fn)
        va_loader = DataLoader(WindowDatasetTorch(val_items, ch_norm),   batch_size=args.batch_size,
                               shuffle=False, drop_last=False, collate_fn=collate_fn)

        print("[INFO] Train Discriminative AE (recon + supcon + class-weighted bce)...")
        if len(tr_loader.dataset)>0:
            train_discriminative_ae(
                ae, clf_head, proj_head,
                tr_loader, va_loader,
                epochs=args.epochs_ae, lr=args.lr, device=device,
                w_rec=0.5, w_con=0.5, w_cls=1.0, temperature=0.07,
                cls_w_pos=cls_w_pos, cls_w_neg=cls_w_neg
            )
        else:
            print("[WARN] Empty training set for DAE.")

        # --- 평가 & 메트릭 ---
        print("[INFO] Evaluate & collect latent...")
        p_tr, y_tr, Ztr = dae_logits_and_latent(ae, clf_head, tr_loader, device)
        p_te, y_te, Zte = dae_logits_and_latent(ae, clf_head, va_loader, device)

        def to_metrics(p, y, thr: float = 0.5):
            if y.size==0:
                return (float('nan'), float('nan'), float('nan'), {}), np.array([]).astype(int)
            yhat = (p>=thr).astype(int)
            try: roc = roc_auc_score(y, p)
            except: roc = float('nan')
            try: pr  = average_precision_score(y, p)
            except: pr = float('nan')
            f1 = f1_score(y.astype(int), yhat.astype(int), zero_division=0)
            rep = classification_report(y, yhat, output_dict=True, zero_division=0)
            return (roc, pr, f1, rep), yhat

        # --- Train F1 기준 best threshold 탐색 ---
        best_thr = 0.5
        best_f1  = float('nan')
        if y_tr.size > 0:
            best_f1 = -1.0
            for thr in np.linspace(0.05, 0.95, 181):  # 0.05 ~ 0.95, step=0.005
                yhat_tmp = (p_tr >= thr).astype(int)
                f1_tmp = f1_score(y_tr.astype(int), yhat_tmp.astype(int), zero_division=0)
                if f1_tmp > best_f1:
                    best_f1, best_thr = f1_tmp, thr
            print(f"[FOLD {fold}] Best train F1={best_f1:.4f} at threshold={best_thr:.3f}")
        else:
            print(f"[FOLD {fold}] No train samples, using default threshold={best_thr:.3f}")

        (roc_tr, pr_tr, f1_tr, rep_tr), y_tr_hat = to_metrics(p_tr, y_tr, thr=best_thr)
        (roc_te, pr_te, f1_te, rep_te), y_te_hat = to_metrics(p_te, y_te, thr=best_thr)

        print(f"[FOLD {fold}] ROC-AUC train/test = {roc_tr:.4f} / {roc_te:.4f}")
        print(f"[FOLD {fold}] PR-AUC  train/test = {pr_tr:.4f} / {pr_te:.4f}")
        print(f"[FOLD {fold}] F1      train/test = {f1_tr:.4f} / {f1_te:.4f}")
        print(f"[FOLD {fold}] Used threshold = {best_thr:.3f}")

        # 저장
        fold_report = {
            "fold": fold,
            "threshold": float(best_thr),
            "train": {"roc_auc": roc_tr, "pr_auc": pr_tr, "f1": f1_tr, "report": rep_tr},
            "test":  {"roc_auc": roc_te, "pr_auc": pr_te, "f1": f1_te, "report": rep_te},
        }
        cv_reports.append(fold_report)
        with open(Path(args.save_dir)/f"fold{fold}_report.json",'w') as f: json.dump(fold_report, f, indent=2)
        torch.save({"ae": ae.state_dict(), "clf_head": clf_head.state_dict(), "proj_head": proj_head.state_dict()},
                   Path(args.save_dir)/f"fold{fold}_ckpt.pt")

        # ---------- 시각화 ----------
        figs_dir = Path(args.save_dir)/f"figs_fold{fold}"
        figs_dir.mkdir(parents=True, exist_ok=True)

        _plot_latent_pca_with_dae_boundary(
            X_latent_tr=Ztr, y_true_tr=y_tr, dae_pred_tr=(p_tr>=best_thr).astype(int),
            X_latent_te=Zte, y_true_te=y_te, dae_pred_te=(p_te>=best_thr).astype(int),
            out_tr=figs_dir/"train_latent_pca_boundary.png",
            out_te=figs_dir/"test_latent_pca_boundary.png",
            title_tr=f"Fold {fold} — Train Latent (PCA+Std) + boundary",
            title_te=f"Fold {fold} — Test Latent (PCA+Std) + boundary",
            boundary_mode=args.boundary_mode, svm_c=args.svm_c, svm_gamma=svm_gamma_final
        )

        _plot_confusion(y_tr.astype(int), y_tr_hat, figs_dir/"train_confusion_dae.png",
                        f"Fold {fold} — Train Confusion (DAE)")
        _plot_confusion(y_te.astype(int), y_te_hat, figs_dir/"test_confusion_dae.png",
                        f"Fold {fold} — Test Confusion (DAE)")

    with open(Path(args.save_dir)/"cv_summary.json",'w') as f: json.dump(cv_reports, f, indent=2)
    print("[DONE] Global CV reports & figures saved to", args.save_dir)


# ------------------ CLI ------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--target_hz', type=int, default=50)
    p.add_argument('--window_sec', type=int, default=30)
    p.add_argument('--stride_sec', type=int, default=15)
    p.add_argument('--trim_sec', type=int, default=5, help='앞뒤로 잘라낼 초(second)')
    p.add_argument('--epochs_ae', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--latent', type=int, default=32)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./runs_global')
    # curved boundary options
    p.add_argument('--boundary_mode', type=str, default='rbf_svm',
                   choices=['rbf_svm','logreg','auto'],
                   help="경계 근사 방식: rbf_svm(곡선, 기본), logreg(직선), auto(가능하면 SVM)")
    p.add_argument('--svm_c', type=float, default=1.0, help='RBF-SVM C')
    p.add_argument('--svm_gamma', type=str, default='scale',
                   help="RBF-SVM gamma: 'scale'/'auto' 또는 숫자 문자열(예: '2.0')")
    
    # [MODIFIED] New CLI arguments
    p.add_argument('--use_uncalibrated', action='store_true',
                   help="Uncalibrated (raw) 센서 파일도 포함하여 로드")
    p.add_argument('--align_axes', action='store_true',
                   help="PCA를 이용해 센서 축을 신호 분산 방향으로 정렬 (Orientation Normalization)")
    p.add_argument('--exclude_axes', nargs='+', default=[],
                   help="사용하지 않을 축 이름 (예: az gz)")

    args_ns = p.parse_args()

    gamma_val: Union[str, float] = _parse_gamma(args_ns.svm_gamma)
    args = Args(
        data_root=args_ns.data_root, target_hz=args_ns.target_hz,
        window_sec=args_ns.window_sec, stride_sec=args_ns.stride_sec, trim_sec=args_ns.trim_sec,
        epochs_ae=args_ns.epochs_ae, batch_size=args_ns.batch_size, lr=args_ns.lr,
        folds=args_ns.folds, latent=args_ns.latent, hidden=args_ns.hidden,
        device=args_ns.device, save_dir=args_ns.save_dir,
        boundary_mode=args_ns.boundary_mode, svm_c=args_ns.svm_c, svm_gamma=gamma_val,
        # [MODIFIED]
        use_uncalibrated=args_ns.use_uncalibrated,
        align_axes=args_ns.align_axes,
        exclude_axes=args_ns.exclude_axes
    )
    run(args)
