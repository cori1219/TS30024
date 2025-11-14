#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMU 균형 이상 감지 — Discriminative AE (Recon + SupCon + BCE) + Latent Kalman + 4-Fold CV

출력(DAE 전용):
- figs_fold{K}/train_latent_pca_boundary.png : Train 잠재 2D(PCA) 산점도 + DAE 결정경계
- figs_fold{K}/test_latent_pca_boundary.png  : Test  잠재 2D(PCA) 산점도 + DAE 결정경계
- figs_fold{K}/train_confusion_dae.png       : Train 혼동행렬 (DAE)
- figs_fold{K}/test_confusion_dae.png        : Test  혼동행렬 (DAE)
"""

import os, io, csv, re, math, json, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
    decimals = ['.', ',']; seps = [',','\t',';','|']

    def try_decode(enc):
        try: return raw_bytes.decode(enc)
        except: return None
    def try_pd(txt, **kw):
        return pd.read_csv(io.StringIO(txt), engine='python', on_bad_lines='skip', **kw)

    for enc in encodings:
        txt = try_decode(enc)
        if txt is None: continue
        for dec in decimals:
            try: return try_pd(txt, sep=None, decimal=dec)
            except: pass

    def find_header_line(text):
        lines = text.splitlines()
        for i, line in enumerate(lines[:300]):
            low = line.lower()
            if ('x' in low and 'y' in low and 'z' in low) and ('time' in low or 'timestamp' in low or 'seconds_elapsed' in low):
                return i
        return None

    for enc in encodings:
        txt = try_decode(enc); 
        if txt is None: continue
        hdr = find_header_line(txt)
        try:
            sample = "\n".join([ln for ln in txt.splitlines() if ln.strip()][:80])
            sniff_sep = csv.Sniffer().sniff(sample, delimiters=",".join(seps)).delimiter
        except: sniff_sep = None
        for dec in decimals:
            if hdr is not None:
                for sep in [sniff_sep] + [s for s in seps if s and s != sniff_sep]:
                    try: return try_pd(txt, sep=sep if sep else None, decimal=dec, skiprows=hdr)
                    except: pass
            for sep in ([sniff_sep] if sniff_sep else []) + seps:
                try: return try_pd(txt, sep=sep, decimal=dec)
                except: pass

    for enc in encodings:
        txt = try_decode(enc); 
        if txt is None: continue
        for dec in decimals:
            try: return try_pd(txt, header=None, decimal=dec)
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
    cols_low = df.columns.tolist()

    # 특수 패턴: time,seconds_elapsed,z,y,x → x,y,z 보정
    if len(cols_low) >= 5 and cols_low[:5] == ['time','seconds_elapsed','z','y','x']:
        df.columns = ['time','seconds_elapsed','z','y','x'] + [f'extra_{i}' for i in range(len(df.columns)-5)]

    # timestamp 확정
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

class WindowDatasetTorch(Dataset):
    def __init__(self, items: List[WindowData]): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        w = self.items[idx]
        return torch.from_numpy(w.feats), torch.tensor(w.label, dtype=torch.float32)

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
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
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
    w_rec=1.0, w_con=0.5, w_cls=0.5, temperature=0.07
):
    ae.train(); clf_head.train(); proj_head.train()
    opt = torch.optim.Adam(list(ae.parameters()) + list(clf_head.parameters()) + list(proj_head.parameters()), lr=lr)
    recon = nn.SmoothL1Loss()
    bce   = nn.BCEWithLogitsLoss()
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
            loss_cls = bce(logits, yb)
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
                    loss_cls = bce(logits, yb)
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
        return np.array([]), np.array([]), np.zeros((0, ae.enc.net[-2].num_features), dtype=np.float32)


# ------------------ Visualization: PCA 2D + DAE boundary ------------------
def _plot_latent_pca_with_dae_boundary(X_latent_tr, y_true_tr, dae_pred_tr,
                                       X_latent_te, y_true_te, dae_pred_te,
                                       out_tr, out_te, title_tr, title_te):
    """
    - PCA는 train latent에 맞춰 학습 → train/test 동일 변환
    - 결정경계는 PCA 2D에서 DAE의 예측(0/1)을 타깃으로 하는 로지스틱 회귀로 근사
    """
    if X_latent_tr.shape[0] == 0: return
    pca = PCA(n_components=2, random_state=42)
    Ztr = pca.fit_transform(X_latent_tr)
    Zte = pca.transform(X_latent_te) if X_latent_te.shape[0] else np.zeros((0,2))

    # 경계 근사: DAE 예측값을 레이블로 사용
    proxy = LogisticRegression(class_weight='balanced', max_iter=200)
    proxy.fit(Ztr, dae_pred_tr.astype(int))

    def plot_one(Z, y_true, fname, title):
        if Z.shape[0]==0: return
        plt.figure()
        idx0=(y_true==0); idx1=(y_true==1)
        plt.scatter(Z[idx0,0], Z[idx0,1], s=12, label='normal (true)')
        plt.scatter(Z[idx1,0], Z[idx1,1], s=12, label='abnormal (true)')

        # 결정경계
        x_min, x_max = np.percentile(Z[:,0], 1), np.percentile(Z[:,0], 99)
        y_min, y_max = np.percentile(Z[:,1], 1), np.percentile(Z[:,1], 99)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = proxy.decision_function(grid).reshape(xx.shape)
        cs = plt.contour(xx, yy, zz, levels=[0.0])
        if cs.collections: cs.collections[0].set_label('DAE decision boundary')

        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(fname, dpi=150); plt.close()

    plot_one(Ztr, y_true_tr, out_tr, title_tr)
    plot_one(Zte, y_true_te, out_te, title_te)


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


# ------------------ Main CV pipeline ------------------
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

def run(args: Args):
    set_seed(42)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    data_root = Path(args.data_root)
    print("[INFO] Loading dataset...")
    items = load_dataset(data_root, args.target_hz, args.window_sec, args.stride_sec, args.trim_sec)
    if len(items) == 0:
        raise RuntimeError("No usable windows built. Check data paths and CSV format.")
    print(f"[INFO] Total windows: {len(items)} (pos={sum(i.label for i in items)}, neg={len(items)-sum(i.label for i in items)})")

    all_idx = np.arange(len(items))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_reports = []
    os.makedirs(args.save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_idx), 1):
        print(f"\n======= Fold {fold}/{args.folds} =======")
        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        # --- D-AE 구성 ---
        ae = AE1D(in_ch=len(SENSOR_COLS), hidden=args.hidden, latent=args.latent).to(device)
        clf_head = LatentClassifier(in_dim=args.latent, hidden=args.hidden).to(device)
        proj_head = ProjectionHead(in_dim=args.latent, proj_dim=64).to(device)

        tr_loader = DataLoader(WindowDatasetTorch(train_items), batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        va_loader = DataLoader(WindowDatasetTorch(val_items),   batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

        print("[INFO] Train Discriminative AE (recon + supcon + bce)...")
        if len(tr_loader.dataset)>0:
            train_discriminative_ae(
                ae, clf_head, proj_head,
                tr_loader, va_loader,
                epochs=args.epochs_ae, lr=args.lr, device=device,
                w_rec=1.0, w_con=0.5, w_cls=0.5, temperature=0.07
            )
        else:
            print("[WARN] Empty training set for DAE.")

        # --- 평가 (DAE 확률/예측 + 잠재 수집) ---
        print("[INFO] Evaluate & collect latent...")
        p_tr, y_tr, Ztr = dae_logits_and_latent(ae, clf_head, tr_loader, device)
        p_te, y_te, Zte = dae_logits_and_latent(ae, clf_head, va_loader, device)

        def to_metrics(p, y):
            if y.size==0: return (float('nan'), float('nan'), {}), np.array([]).astype(int)
            yhat = (p>=0.5).astype(int)
            try: roc = roc_auc_score(y, p)
            except: roc = float('nan')
            try: pr  = average_precision_score(y, p)
            except: pr = float('nan')
            rep = classification_report(y, yhat, output_dict=True)
            return (roc, pr, rep), yhat

        (roc_tr, pr_tr, rep_tr), y_tr_hat = to_metrics(p_tr, y_tr)
        (roc_te, pr_te, rep_te), y_te_hat = to_metrics(p_te, y_te)

        print(f"[FOLD {fold}] ROC-AUC (train)={roc_tr:.4f} (test)={roc_te:.4f} | PR-AUC (train)={pr_tr:.4f} (test)={pr_te:.4f}")

        fold_reports.append({
            "fold": fold,
            "train": {"roc_auc": roc_tr, "pr_auc": pr_tr, "report": rep_tr},
            "test":  {"roc_auc": roc_te, "pr_auc": pr_te, "report": rep_te},
        })
        with open(Path(args.save_dir)/f"fold{fold}_report.json",'w') as f: json.dump(fold_reports[-1], f, indent=2)
        torch.save({"ae": ae.state_dict(), "clf_head": clf_head.state_dict(), "proj_head": proj_head.state_dict()}, Path(args.save_dir)/f"fold{fold}_ckpt.pt")

        # ---------- DAE 전용 시각화 ----------
        figs_dir = Path(args.save_dir)/f"figs_fold{fold}"
        figs_dir.mkdir(parents=True, exist_ok=True)

        # (1) Latent PCA 2D + DAE 경계(로지스틱으로 근사)
        _plot_latent_pca_with_dae_boundary(
            X_latent_tr=Ztr, y_true_tr=y_tr, dae_pred_tr=(p_tr>=0.5).astype(int),
            X_latent_te=Zte, y_true_te=y_te, dae_pred_te=(p_te>=0.5).astype(int),
            out_tr=figs_dir/"train_latent_pca_boundary.png",
            out_te=figs_dir/"test_latent_pca_boundary.png",
            title_tr=f"Fold {fold} — Train Latent (PCA) + DAE boundary",
            title_te=f"Fold {fold} — Test Latent (PCA) + DAE boundary"
        )

        # (2) 혼동행렬 (DAE 기준)
        _plot_confusion(y_tr.astype(int), (p_tr>=0.5).astype(int), figs_dir/"train_confusion_dae.png",
                        f"Fold {fold} — Train Confusion (DAE)")
        _plot_confusion(y_te.astype(int), (p_te>=0.5).astype(int), figs_dir/"test_confusion_dae.png",
                        f"Fold {fold} — Test Confusion (DAE)")

    with open(Path(args.save_dir)/"cv_summary.json",'w') as f: json.dump(fold_reports, f, indent=2)
    print("[DONE] CV reports & figures saved to", args.save_dir)


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
    p.add_argument('--save_dir', type=str, default='./runs')
    args_ns = p.parse_args()
    run(Args(**vars(args_ns)))

