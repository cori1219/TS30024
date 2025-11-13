#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP IMU → (고정 개수) 수집: N = round(40s * target_hz) → 앞/뒤 5s 컷(=30s) → 50Hz 리샘플
→ (train-fold 통계) z-score → Encoder → Latent Kalman → GAP → MLP(sigmoid)

핵심 변경점:
- 시간 기준(t_rel ≥ 40s) 종료가 아니라, '샘플 개수' 기준(N개 수집 시 종료)

정규화 통계(ckpt에 없음 전제):
1) 권장: --train-py, --data-root, --folds, --fold, --window-sec, --stride-sec, --trim-sec
   (학습 스크립트 동적 import → 해당 폴드 train 윈도우들로 mean/std 계산)
2) 또는 수동: --mean, --std  (ax,ay,az,gx,gy,gz)

체크포인트: 'fold3_ckpt.pt' 자동 사용(CWD or script dir)
"""

import argparse
import socket
import sys
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================ 상수: ckpt 파일명 자동 고정 ============================
CKPT_NAME = "fold3_ckpt.pt"  # --ckpt 없이 자동 사용

# ============================ 모델 ============================

class Encoder(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):  # [B,6,T]
        return self.net(x)  # [B,32,T]

class Decoder(nn.Module):
    def __init__(self, out_ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, out_ch, kernel_size=7, padding=3),
        )
    def forward(self, z):
        return self.net(z)

class AE(nn.Module):
    def __init__(self, in_ch=6, out_ch=6):
        super().__init__()
        self.enc = Encoder(in_ch)
        self.dec = Decoder(out_ch)
    def forward(self, x):
        z = self.enc(x)
        xr = self.dec(z)
        return xr, z

class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
        )
    def forward(self, z_vec32):
        return self.net(z_vec32)

class ClfHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
    def forward(self, z_vec32):
        return self.net(z_vec32)  # [B,1]

# -------- Kalman --------
class SimpleKalman:
    def __init__(self, dim: int, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var
    def filter(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape
        out = np.zeros_like(seq)
        x = np.zeros(D, dtype=seq.dtype)
        p = np.ones(D, dtype=seq.dtype)
        for t in range(T):
            x_pred = x
            p_pred = p + self.q
            z = seq[t]
            k = p_pred / (p_pred + self.r)
            x = x_pred + k * (z - x_pred)
            p = (1 - k) * p_pred
            out[t] = x
        return out

class InferenceModel(nn.Module):
    def __init__(self, ckpt_path: Path, device="cpu"):
        super().__init__()
        self.ae = AE(in_ch=6, out_ch=6)
        self.proj_head = ProjHead()
        self.clf_head = ClfHead()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        self.ae.load_state_dict(ckpt["ae"], strict=True)
        self.proj_head.load_state_dict(ckpt["proj_head"], strict=True)
        self.clf_head.load_state_dict(ckpt["clf_head"], strict=True)
        self.to(device)
        self.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, x):  # x: [B,6,T]
        z = self.ae.enc(x)  # [B,32,T]
        # Kalman smoothing → GAP
        z_np = z.detach().cpu().numpy().transpose(0, 2, 1)  # [B,T,32]
        B, T, D = z_np.shape
        zf_np = np.zeros_like(z_np)
        for b in range(B):
            zf_np[b] = SimpleKalman(D, 1e-3, 1e-2).filter(z_np[b])
        zf = torch.from_numpy(zf_np.transpose(0, 2, 1)).to(self.device)  # [B,32,T]
        z_vec = zf.mean(dim=-1)              # [B,32]
        logit = self.clf_head(z_vec)[:, 0]   # [B]
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob}

# ============================ UDP 수신 (고정 개수) ============================

def parse_csv6(line: str):
    try:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 6:
            return None
        return [float(parts[i]) for i in range(6)]
    except Exception:
        return None

def recv_udp_fixed_count(port: int, target_hz: int, timeout: float = 1.0):
    """
    첫 유효 샘플을 받은 시점부터 '개수' 기준으로 수신.
    N_required = round(40.0 * target_hz) 샘플을 모을 때까지 계속 받는다.
    반환: times_uniform(0..40s 등간격), data(N,6)
    """
    N_required = int(round(40.0 * target_hz))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(timeout)

    buf_vals = []  # list of [ax,ay,az,gx,gy,gz]
    started = False
    print(f"[INFO] Listening UDP :{port} ... (start when first valid sample arrives; target N={N_required})")
    try:
        while len(buf_vals) < N_required:
            try:
                pkt, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            line = pkt.decode("utf-8", errors="ignore")
            for raw in line.strip().splitlines():
                vals = parse_csv6(raw)
                if vals is None:
                    continue
                if not started:
                    started = True
                    print("[INFO] First sample received. Start fixed-count capture.")
                buf_vals.append(vals)
                if len(buf_vals) >= N_required:
                    break
        N = len(buf_vals)
        data = np.array(buf_vals[:N_required], dtype=np.float32)  # (N_required,6)
        times_uniform = np.linspace(0.0, 40.0, num=N_required, endpoint=False)
        print(f"[INFO] Collected {N}/{N_required} samples (fixed-count).")
        return times_uniform, data
    finally:
        sock.close()

# ============================ 트리밍 + 리샘플 ============================

def trim_and_resample(times, data, target_hz: int):
    """
    40s 중 앞뒤 5s 제거 → 30s 균일 시간축 생성 → 채널별 선형보간
    반환: x_trim (6, T_out), t_out
    """
    total = times[-1] if times.size else 40.0
    t_start = 5.0
    t_end = max(5.0, total - 5.0)
    span = t_end - t_start
    if span < 10.0:
        raise RuntimeError(f"유효 구간이 너무 짧음: {span:.2f}s (입력 드랍/지연 확인)")

    # 중복 시간 제거(이론상 없음)
    t_in, idx = np.unique(times, return_index=True)
    x_in = data[idx]  # (N,6)

    T_out = int(round(30.0 * target_hz))  # 정확히 30초
    t_out = np.linspace(t_start, t_start + 30.0, num=T_out, endpoint=False)

    x_out = np.zeros((6, T_out), dtype=np.float32)
    for ch in range(6):
        x_out[ch] = np.interp(t_out, t_in, x_in[:, ch]).astype(np.float32)
    return x_out, t_out

# ============================ 정규화 통계(훈련 모듈) ============================

def apply_norm_train_stats(x_6T, mean, std, eps=1e-6):
    m = np.asarray(mean, dtype=np.float32).reshape(6, 1)
    s = np.asarray(std, dtype=np.float32).reshape(6, 1)
    return ((x_6T - m) / (s + eps)).astype(np.float32)

def parse_fold_from_ckpt(ckpt_path: Path):
    m = re.search(r"fold(\d+)", ckpt_path.name.lower())
    return int(m.group(1)) if m else None

def compute_norm_via_train_module(train_py: Path, data_root: Path,
                                  target_hz: int, window_sec: int, stride_sec: int, trim_sec: int,
                                  folds: int, fold: int):
    """
    학습 스크립트를 동적 import → 해당 폴드의 train 윈도우들로 ChannelNormalizer.fit → mean/std 계산
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_mod", str(train_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_dataset") or not hasattr(mod, "ChannelNormalizer"):
        raise RuntimeError("train-py에 load_dataset 또는 ChannelNormalizer가 없습니다.")
    items = mod.load_dataset(Path(data_root), int(target_hz), int(window_sec), int(stride_sec), int(trim_sec))
    if len(items) == 0:
        raise RuntimeError("load_dataset 결과가 비어 있습니다.")
    from sklearn.model_selection import KFold
    import numpy as _np
    idx = _np.arange(len(items))
    kf = KFold(n_splits=int(folds), shuffle=True, random_state=42)
    mean = std = None
    for i, (tr, va) in enumerate(kf.split(idx), 1):
        if i == int(fold):
            ch = mod.ChannelNormalizer()
            ch.fit([items[j] for j in tr])   # train 윈도우들로만 통계
            mean = ch.mean.tolist()
            std = ch.std.tolist()
            break
    if mean is None or std is None:
        raise RuntimeError("해당 fold를 찾지 못했습니다. --fold 값을 확인하세요.")
    print(f"[INFO] Computed train-fold normalization via {train_py.name} (fold={fold}/{folds})")
    return mean, std

# ============================ 유틸 ============================

def resolve_ckpt_or_die() -> Path:
    script_dir = Path(__file__).resolve().parent
    cand = [Path.cwd() / CKPT_NAME, script_dir / CKPT_NAME]
    for p in cand:
        if p.exists():
            print(f"[INFO] Using checkpoint (auto): {p.name if p.parent==Path.cwd() else p}")
            return p
    print(f"[ERR] '{CKPT_NAME}' not found in either CWD({Path.cwd()}) or script dir({script_dir}).")
    sys.exit(2)

# ============================ 메인 ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000, help="UDP 포트")
    ap.add_argument("--target-hz", type=int, default=50, help="리샘플 목표 Hz 및 수집 개수 계산")
    ap.add_argument("--device", type=str, default="cpu", help="cpu 또는 cuda")

    # 정규화 통계 방법 (둘 중 하나)
    ap.add_argument("--mean", type=str, default=None, help="채널별 평균 6개 ax,ay,az,gx,gy,gz")
    ap.add_argument("--std",  type=str, default=None, help="채널별 표준편차 6개 ax,ay,az,gx,gy,gz")

    # 또는 학습 스크립트에서 즉석 계산
    ap.add_argument("--train-py", type=str, default=None, help="학습 스크립트 경로(.py)")
    ap.add_argument("--data-root", type=str, default=None, help="데이터 루트(o/x)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0, help="1-based. 0이면 ckpt 이름에서 foldN 자동 또는 기본 3")
    ap.add_argument("--window-sec", type=int, default=30)
    ap.add_argument("--stride-sec", type=int, default=15)
    ap.add_argument("--trim-sec", type=int, default=5)

    args = ap.parse_args()

    ckpt_path = resolve_ckpt_or_die()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    # 1) 수신(고정 개수 = 40s * target_hz)
    times, data = recv_udp_fixed_count(args.port, args.target_hz)

    # 2) 트리밍 + 리샘플(30s)
    x_6T, _ = trim_and_resample(times, data, args.target_hz)

    # 3) 정규화 통계 확보: (a) train 모듈 즉석 계산 → (b) 수동 mean/std → (c) 에러
    mean = std = None
    if args.train_py and args.data_root:
        fold = args.fold if args.fold > 0 else (parse_fold_from_ckpt(ckpt_path) or 3)
        try:
            mean, std = compute_norm_via_train_module(
                train_py=Path(args.train_py),
                data_root=Path(args.data_root),
                target_hz=args.target_hz,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                trim_sec=args.trim_sec,
                folds=args.folds,
                fold=fold,
            )
        except Exception as e:
            print(f"[ERR] 학습 모듈을 통한 통계 계산 실패: {e}")
            sys.exit(2)
    elif args.mean and args.std:
        try:
            mean = [float(x) for x in args.mean.split(",")]
            std  = [float(x) for x in args.std.split(",")]
            assert len(mean) == 6 and len(std) == 6
            print("[INFO] Using manual normalization from CLI.")
        except Exception:
            print("[ERR] mean/std 파싱 실패. 예: --mean 0,0,0,0,0,0 --std 1,1,1,1,1,1")
            sys.exit(2)
    else:
        print("[ERR] 정규화 통계를 제공해야 합니다. "
              "옵션 중 하나를 사용하세요: (1) --train-py + --data-root (+folds/fold/window/stride/trim), (2) --mean/--std")
        sys.exit(2)

    # 4) 정규화 적용
    x_6T = apply_norm_train_stats(x_6T, mean=mean, std=std)

    # 5) 모델 로드 & 추론 (Kalman 포함)
    model = InferenceModel(ckpt_path, device=args.device)
    with torch.no_grad():
        x = torch.from_numpy(x_6T[None, ...]).to(model.device)  # [1,6,T]
        out = model(x)
        logit = out["logit"].item()
        prob = out["prob"].item()

    # 6) 결과
    print("=== INFERENCE RESULT (30s window, TEST-MATCHED with train-fold stats, fixed-count capture) ===")
    print(f"logit: {logit:.6f}")
    print(f"prob(sigmoid): {prob:.6f}")
    print(f"label(@0.5): {1 if prob >= 0.5 else 0}")

if __name__ == "__main__":
    main()

