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
  --save_dir ./runs_lstm
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

**v0.0.5 — 2025-09-25**
- **Added**
  - **F1-score 산출/저장**: Train/Test 모두 F1-score 계산.
  - 콘솔 로그에 `F1 train/test` 출력.
  - 각 폴드 리포트(`fold{K}_report.json`)와 요약(`cv_summary.json`)에 `"f1"` 필드 추가.
- **Changed**
  - 메트릭 계산 유틸을 통합해 ROC-AUC/PR-AUC/F1/리포트를 한 번에 반환.
- **Fixed**
  - 소수 클래스 미예측 시 경고 억제: `zero_division=0`로 안정화.

**v0.0.4 — 2025-09-25**
- **Added**
  - **글로벌 학습 파이프라인**: 모든 피험자 데이터를 합쳐 K-Fold CV.
  - **채널별 표준화(z-score)**: Train fold 통계로 fit → Train/Val에 동일 적용(데이터 누수 방지).
- **Changed**
  - 사람별 루프 제거, 전 윈도우 단위 CV로 단순화.
  - 저장 경로 정리: `runs_global*/fold{K}_report.json`, `figs_fold{K}/*`, `cv_summary.json`.
- **Fixed**
  - 피험자 간 스케일 불일치 완화, fold 간 통계 누출 제거.

**v0.0.3 — 2025-09-25**
- **Added**
  - **RBF-SVM 곡선 경계**(PCA→표준화 후 학습) 및 시각화 강화: 결정함수 `contourf` + 0-레벨 경계선.
  - CLI: `--svm_gamma`에 **숫자 문자열**(예: `2.0`, `5.0`) 지원 (`scale`/`auto` 유지).
- **Changed**
  - PCA 2D 후 **StandardScaler** 적용으로 곡률 표현력 개선.
  - 범례/타이틀에 경계 메타 정보 표기: `SVM(DAE, C=..., gamma=...)` 또는 `LogReg(TRUE)`.
- **Fixed**
  - 경계 학습 라벨 단일 클래스 시 `DAE→TRUE` 폴백, 그래도 단일이면 경계 생략.
  - 스케일 불균형으로 직선처럼 보이던 현상 완화.

**v0.0.2 — 2025-09-25**
- **Added**
  - **사람별 학습 파이프라인**: zip 스템(`o_n`/`x_n`)으로 subject 추출, 주체별 K-Fold CV.
  - 결과 구조: `runs/subject_{ID}/fold{K}_report.json`, `fold{K}_ckpt.pt`, `figs_fold{K}/*`, `all_subjects_summary.json`.
  - Kalman smoothing(잠재 시퀀스), PCA 2D + 로지스틱 경계 근사.
- **Changed**
  - DataLoader `drop_last=False`, `AE1D.latent_dim` 보관.
- **Fixed**
  - 단일 클래스 라벨 시 LogReg 예외 처리 및 `classification_report(..., zero_division=0)` 적용.

**v0.0.1:** First version

