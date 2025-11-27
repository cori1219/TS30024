import eventlet
eventlet.monkey_patch()

import sys, json, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA  # [NEW] PCA for alignment

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
PORT = 5000
WINDOW_SEC = 30
TARGET_HZ = 50  # 100Hz로 학습했으면 100으로 변경 필요 (여기선 기존 50 유지)
WINDOW_LEN = WINDOW_SEC * TARGET_HZ  # 30 * 50 = 1500 samples
BUFFER_Limit = WINDOW_LEN + 200      # 여유 버퍼

# [MODIFIED] Threshold 및 전처리 옵션 설정
THRESHOLD = 0.485
ALIGN_AXES = True               # 학습 시 --align_axes를 썼다면 True
EXCLUDE_AXES = ['az', 'gz']     # 학습 시 --exclude_axes로 뺀 축들 (예시)

# 입력으로 들어오는 전체 센서 키 (순서 중요)
ALL_SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

# 실제 모델에 들어가는 채널 계산
MODEL_INPUT_COLS = [c for c in ALL_SENSOR_COLS if c not in EXCLUDE_AXES]
N_INPUT_CH = len(MODEL_INPUT_COLS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 저장된 모델 경로 (main.py에서 저장한 fold3_ckpt.pt 등)
CKPT_PATH = "./runs_global/fold3_ckpt.pt" 
# 정규화 통계 (학습 데이터에서 구한 값을 하드코딩하거나 별도 파일 로드 필요)
# 여기서는 예시로 0.0, 1.0 사용 (실제로는 fold_report나 별도 파일에서 로드 권장)
NORM_MEAN = np.zeros(N_INPUT_CH, dtype=np.float32)
NORM_STD  = np.ones(N_INPUT_CH, dtype=np.float32)


# ==============================================================================
# 2. Model Architecture (Must match main.py)
# ==============================================================================
class Encoder1D(nn.Module):
    def __init__(self, in_ch=6, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

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
        return z.mean(dim=-1)

class LatentClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

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


# ==============================================================================
# 3. Global State
# ==============================================================================
app = Flask(__name__)
model_ae = None
model_clf = None
buffer_data = []  # List of dicts: {'ax':..., 'gz':...}
buffer_lock = eventlet.semaphore.Semaphore(1)


# ==============================================================================
# 4. Helpers
# ==============================================================================
def load_resources():
    global model_ae, model_clf
    print(f"[INFO] Loading model from {CKPT_PATH}...")
    print(f"[INFO] Config: Align={ALIGN_AXES}, Exclude={EXCLUDE_AXES}, InputCh={N_INPUT_CH}, Thr={THRESHOLD}")
    
    # 모델 초기화 (입력 채널 수 반영)
    ae = AE1D(in_ch=N_INPUT_CH, hidden=64, latent=32).to(DEVICE)
    clf = LatentClassifier(in_dim=32, hidden=64).to(DEVICE)
    
    # 체크포인트 로드
    if torch.cuda.is_available():
        ckpt = torch.load(CKPT_PATH)
    else:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        
    ae.load_state_dict(ckpt['ae'])
    clf.load_state_dict(ckpt['clf_head'])
    
    ae.eval()
    clf.eval()
    model_ae = ae
    model_clf = clf
    print("[INFO] Model loaded successfully.")

# [NEW] PCA Alignment Helper
def apply_pca_alignment(df_window: pd.DataFrame):
    """
    현재 윈도우 데이터에 대해 PCA 회전을 적용하여 축 정렬
    """
    if len(df_window) < 10: return df_window
    
    # 가속도 데이터 추출
    acc_cols = ['ax', 'ay', 'az']
    # 데이터프레임에 해당 컬럼들이 다 있는지 확인
    if not all(c in df_window.columns for c in acc_cols): return df_window
    
    acc_data = df_window[acc_cols].values
    
    # PCA 계산 (3축)
    pca = PCA(n_components=3)
    pca.fit(acc_data)
    
    # 가속도 회전 적용
    df_window[acc_cols] = pca.transform(acc_data)
    
    # 자이로 회전 적용 (가속도의 회전 매트릭스 사용)
    gyr_cols = ['gx', 'gy', 'gz']
    if all(c in df_window.columns for c in gyr_cols):
        R = pca.components_
        df_window[gyr_cols] = df_window[gyr_cols].values @ R.T
        
    return df_window

def process_and_infer(current_buffer):
    """
    버퍼 데이터를 DataFrame으로 변환 -> (PCA) -> (Exclude) -> (Normalize) -> Inference
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame(current_buffer)
    
    # 필요한 컬럼이 다 있는지 확인 (없으면 0으로 채움)
    for c in ALL_SENSOR_COLS:
        if c not in df.columns: df[c] = 0.0
    
    # 2. [OPTIONAL] Apply PCA Alignment
    # 주의: Exclude하기 '전'에 전체 6축(또는 3축)이 살아있을 때 수행해야 함
    if ALIGN_AXES:
        df = apply_pca_alignment(df)
        
    # 3. Exclude Axes & Select Columns
    # 실제 모델 입력에 필요한 컬럼만 추출
    feats = df[MODEL_INPUT_COLS].values.astype(np.float32) # [T, C]
    
    # 4. Channel Normalization (Z-score)
    # (T, C) - (C,) / (C,)
    feats = (feats - NORM_MEAN) / NORM_STD
    
    # 5. Tensor Conversion [Batch=1, C, T]
    feats_t = np.transpose(feats, (1, 0)) # [C, T]
    x_tensor = torch.from_numpy(feats_t).unsqueeze(0).to(DEVICE) # [1, C, T]
    
    # 6. Inference
    with torch.no_grad():
        _, z = model_ae(x_tensor)       # z: [1, Latent, T]
        
        # Latent Kalman Smoothing
        z_np = z.cpu().numpy()          # [1, D, T]
        z_seq = np.transpose(z_np[0], (1, 0)) # [T, D]
        
        kf = SimpleKalman(z_seq.shape[1])
        z_filtered = kf.filter(z_seq)   # [T, D]
        
        z_pool = z_filtered.mean(axis=0, keepdims=True) # [1, D]
        
        # Classifier
        logits = model_clf(torch.from_numpy(z_pool).float().to(DEVICE))
        prob = torch.sigmoid(logits).item()
        
    return prob


# ==============================================================================
# 5. API Endpoints
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    global buffer_data
    
    try:
        data = request.json  # Expect list of dicts or single dict
        if isinstance(data, dict):
            data = [data]
            
        with buffer_lock:
            buffer_data.extend(data)
            # 버퍼 크기 유지
            if len(buffer_data) > BUFFER_Limit:
                buffer_data = buffer_data[-BUFFER_Limit:]
            
            curr_len = len(buffer_data)
        
        # 윈도우 크기만큼 데이터가 모였는지 확인
        if curr_len >= WINDOW_LEN:
            # 추론용 데이터 스냅샷 (마지막 WINDOW_LEN개)
            snapshot = buffer_data[-WINDOW_LEN:]
            
            prob = process_and_infer(snapshot)
            status = "FALL" if prob >= THRESHOLD else "NORMAL"
            
            return jsonify({
                "status": "ok",
                "prediction": status,
                "probability": round(prob, 4),
                "threshold": THRESHOLD,
                "window_size": len(snapshot)
            })
        else:
            return jsonify({
                "status": "buffering",
                "current_length": curr_len,
                "required": WINDOW_LEN
            })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_buffer():
    global buffer_data
    with buffer_lock:
        buffer_data = []
    return jsonify({"status": "reset_complete"})


# ==============================================================================
# 6. Main Entry
# ==============================================================================
if __name__ == '__main__':
    load_resources()
    print(f"[START] Server running on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, threaded=True)

