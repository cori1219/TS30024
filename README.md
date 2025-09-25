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
9.  [License](#license)

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
# Run the model
python main.py \
  --data_root ./raw_data \
  --target_hz 50 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
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

**v0.0.2 — 2025-09-25**
- **Added**
  - 사람별 학습 파이프라인: zip 스템에서 subject 추출하여(패턴 `o_n`/`x_n`) **동일인 묶음** 후 주체별 K-Fold CV 수행.
  - 결과 구조 개선: `runs/subject_{ID}/fold{K}_report.json`, `fold{K}_ckpt.pt`, `figs_fold{K}/*` 및 전체 요약 `all_subjects_summary.json` 저장.
  - Latent smoothing 유지(Kalman), Latent PCA 2D 시각화 + **의사 결정경계**(로지스틱) 생성.

- **Changed**
  - 학습 데이터로더 `drop_last=False` (소량 데이터에서도 배치가 사라지지 않도록).
  - 모델 `AE1D`에 `latent_dim` 보관 → 빈 로더/빈 배치 시에도 안전한 반환.

- **Fixed**
  - 결정경계 학습 라벨이 단일 클래스일 때 `LogisticRegression` 예외 발생 문제:
    - 우선 **DAE 예측**을 사용, 단일 클래스면 **실제 라벨**로 대체 시도, 그것도 단일이면 **경계 생략** 후 산점도만 저장.
  - `classification_report`에서 클래스 부재 시 경고 발생 → `zero_division=0`로 안전 처리.
  - 파일명 규칙 보강: `o_1.zip`/`x_1.zip`처럼 접두형 패턴도 동일인으로 인식.

- **Notes**
  - 데이터 폴더 구조는 동일(`data_root/o/*.zip`, `data_root/x/*.zip`).
  - 실패한 CSV 헤더/앞부분은 `runs/failed_samples/*.head.txt`로 덤프.

**v0.0.1:** First version
