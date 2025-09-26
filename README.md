# Balance Anomaly Detection AI Model

This project aims to develop an AI model that detects balance anomalies
in real-time using IMU sensors. It is designed to prevent accidents
related to impaired balance, such as drunk driving and elderly falls.

------------------------------------------------------------------------

## Table of Contents

1.  [Introduction](#introduction)
2.  [Vision and Goals](#vision-and-goals)
3.  [Key Features](#key-features)
4.  [Target Users](#target-users)
5.  [Installation](#installation)
6.  [Usage](#usage)
7.  [Tech Stack](#tech-stack)
8.  [Roadmap](#roadmap)
9.  [Update Log](#Update-Log)

------------------------------------------------------------------------

## Introduction

Using the built-in IMU sensors in smartphones, this solution collects
gait data and uses AI models to detect balance anomalies in real-time.
The focus is on enabling **self-check** for users, improving safety, and
ensuring accessibility without requiring additional wearables.

------------------------------------------------------------------------

## Vision and Goals

-   **Enhance User Safety:** Achieve over 90% detection accuracy
-   **Expand Self-Check Accessibility:** 50+ pilot users, 80%+ check
    completion rate
-   **Data-Driven Service Enhancement:** Collect over 2,000 gait data
    samples
-   **Early Detection for High-Risk Users:** Monthly self-check for
    high-risk groups

------------------------------------------------------------------------

## Key Features

### Must-Have

-   Real-time IMU data collection (accelerometer, gyroscope)
-   Binary classification of balance status (Normal / Abnormal)
-   Real-time alerts
-   Privacy protection with anonymized data processing

### Should-Have

-   Automated time-series data preprocessing
-   Local storage of user check history
-   Context-specific guidance based on results

### Could-Have

-   Multi-sensor data enhancement
-   Error detection and retry mechanism

------------------------------------------------------------------------

## Target Users

-   **Drivers:** Check balance before driving when tired or after
    drinking
-   **Post-Drinking Users:** Self-assessment after alcohol consumption
-   **Everyday Health Monitors:** Regular balance tracking for
    health-conscious users

------------------------------------------------------------------------

## Installation

``` bash
# Clone repository
git clone https://github.com/cori1219/TS30024.git

# Move to project directory
cd TS30024

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

``` bash
# Run the model v0.0.2 ~ v0.0.5
python main.py \
  --data_root ./raw_data \
  --target_hz 50 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
  --boundary_mode rbf_svm --svm_c 5.0 --svm_gamma 2.0 \
  --save_dir ./runs
```

``` bash
# Run the model v0.1.0
python main.py \
  --data_root ./raw_data \
  --target_hz 50 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
  --encoder lstm --lstm_layers 1 --lstm_bidir \
  --boundary_mode rbf_svm --svm_c 2.0 --svm_gamma scale \
  --save_dir ./runs
```

``` bash
# Run the model v0.1.1
python main.py \
  --data_root ./raw_data \
  --target_hz 50 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
  --encoder lstm --lstm_layers 1 --lstm_bidir \
  --attn_pool dot --attn_dropout 0.1 \
  --boundary_mode rbf_svm --svm_c 2.0 --svm_gamma scale \
  --save_dir ./runs
```

------------------------------------------------------------------------

## Tech Stack

-   **Language:** Python
-   **AI Models:** LSTM, TSMixer, ETC.
-   **Data Processing:** Numpy, Pandas, ETC.
-   **Visualization/UI:** TBD

------------------------------------------------------------------------

## Roadmap

1.  **Data Collection:** 2025.07 \~ 2025.09
2.  **AI Model Development:** 2025.09 \~ 2025.10
3.  **Test Tool Implementation:** 2025.11
4.  **Pilot Testing & Results:** 2025.12

------------------------------------------------------------------------

## Update Log

**v0.1.1 — 2025-09-26**  
- **Added**  
  - **Attention Pooling** for time-series latent aggregation:
    - `--attn_pool dot` (Dot-Product Attention, default)
    - `--attn_pool mha` (1-layer Multi-Head Attention + CLS)
    - `--attn_pool none` (mean pooling; 이전 동작)  
  - 관련 하이퍼파라미터: `--attn_dropout`, `--attn_heads`(MHA 전용).  
- **Changed**  
  - 분류 입력을 **Attention으로 가중합**한 latent로 교체(시간적 중요도 반영).  
  - 학습/평가 로그에 **Encoder+Pooling 표기**(예: `LSTM + DOT`).  
- **Fixed**  
  - Kalman 이후 텐서 전환 시 드문 디바이스/형 변환 경계 케이스 보완.

**v0.1.0 — 2025-09-25**  
- **Added**  
  - **LSTM 인코더** 지원(`--encoder lstm`), 시계열 의존성 학습 강화.  
  - Kalman smoothing과 결합한 시계열 잠재 안정화.  
- **Changed**  
  - 분류 입력을 LSTM 기반 latent로 전환(시계열 정보 보존).  
- **Fixed**  
  - LSTM 출력/헤드 차원 불일치 및 시퀀스 길이 처리 개선.

**v0.0.5 — 2025-09-25**  
- **Added**  
  - **F1-score** 계산/로그 저장.  
- **Changed**  
  - 전처리/리포팅 안정화, 지표 출력 정리.

**v0.0.4 — 2025-09-25**  
- **Added**  
  - 전 피험자 **정규화 후 글로벌 학습** 옵션(데이터 결합).  
- **Changed**  
  - 표준화/시각화 파이프라인 정리.

**v0.0.3 — 2025-09-25**  
- **Added**  
  - **RBF-SVM 경계 근사**, `--svm_gamma` 숫자/문자 옵션 지원.  
- **Changed**  
  - PCA 후 **StandardScaler** 적용, 경계 시각화 강화.  
- **Fixed**  
  - 단일 클래스 경계 학습 시 안전 폴백 처리.

**v0.0.2 — 2025-09-25**  
- **Added** 사람별 K-Fold CV, Kalman, PCA 2D + 경계 근사, 결과 디렉토리 구조.  
- **Changed** DataLoader/AE 보완.  
- **Fixed** 단일 클래스/리포트 경고 대응.

**v0.0.1:** First version

