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
# Run the model training v0.0.1 ~ v0.0.2
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
  --save_dir ./runs
```

``` bash
# Run the model training v0.0.3 ~ v0.0.5
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 32 --hidden 64 \
  --boundary_mode rbf_svm --svm_c 5.0 --svm_gamma 2.0 \
  --save_dir ./runs \
  --use_uncalibrated \
  --align_axes \
  --exclude_axes az gz
```

``` bash
# Run the model training v0.1.0
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
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
# Run the model training v0.1.1
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
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

``` bash
# Run the model training v0.2.0
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 48 --hidden 64 \
  --encoder timemixer --tm_depth 4 --tm_kernel 7 --tm_ch_hidden 128 --tm_dropout 0.1 \
  --attn_pool dot --attn_dropout 0.1 \
  --boundary_mode rbf_svm --svm_c 2.0 --svm_gamma scale \
  --save_dir ./runs
```

``` bash
# Run the model training v0.3.0
python main.py \
  --data_root ./raw_data \
  --target_hz 100 \
  --window_sec 30 --stride_sec 15 \
  --trim_sec 5 \
  --epochs_ae 20 \
  --batch_size 64 --lr 1e-3 \
  --folds 4 \
  --latent 48 --hidden 64 \
  --encoder tsmixer \
  --tsm_depth 4 --tsm_token_ratio 2.0 --tsm_channel_ratio 2.0 --tsm_dropout 0.1 \
  --attn_pool dot --attn_dropout 0.1 \
  --boundary_mode rbf_svm --svm_c 2.0 --svm_gamma scale \
  --save_dir ./runs
```

``` bash
# Run the model test v0.0.5 with real-time data
python imu_udp_infer.py --target-hz 100 \
  --train-py ../main.py \
  --data-root ../../raw_data \
  --folds 4 --fold 3 \
  --window-sec 30 --stride-sec 15 --trim-sec 5

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

**v0.3.0 — 2025-09-26**  
- **Added**  
  - Added **TSMixer encoder** (`--encoder tsmixer`): extends time-series representation power with all-MLP token/channel mixing.  
  - Hyperparameters: `--tsm_depth`, `--tsm_token_ratio`, `--tsm_channel_ratio`, `--tsm_dropout`.  
- **Changed**  
  - Kept dot-product attention (`--attn_pool dot`) as the default pooling and verified compatibility with TSMixer.  
  - Cleaned up per-fold channel normalization path (fit on train statistics → apply consistently to train/val).  
- **Fixed**  
  - Fixed **TSMixer token-mixing LayerNorm axis bug**: LayerNorm is now applied along the time axis (T), resolving runtime shape errors.  
  - Stabilized boundary visualization (heatmap + 0-level contour).  
- **Recommendation**  
  - Fast/stable baseline: **v0.0.x** (Conv1D DAE + classical boundary approximation)  
  - Global normalization + curved boundary experiments: **v0.2.x** (RBF-SVM, standardized pipeline)  
  - This version (**v0.3.0**) is an experimental release focused on **TSMixer exploration**.

**v0.2.0 — 2025-09-26**  
- **Added**  
  - Introduced **TimeMixer encoder** (`--encoder timemixer`): MLP-Mixer–style blocks for parallel learning of temporal patterns.  
  - Hyperparameters: `--tm_depth`, `--tm_kernel`, `--tm_ch_hidden`, `--tm_dropout`.  
- **Changed**  
  - Unified model selection under `--encoder {lstm,conv,timemixer}` (drop-in replacement).  
  - Added **Encoder + Pooling tags** in visualization/logs (e.g., `TIMEMIXER + DOT`).  
- **Fixed**  
  - Improved robustness for tensor/device conversion after Kalman smoothing and for boundary-approximation input scaling edge cases.  
- **Perf/Deploy**  
  - TimeMixer parallelization improves training/inference speed and is friendly to TorchScript / TFLite.

**v0.1.1 — 2025-09-26**  
- **Added**  
  - **Attention pooling** for time-series latent aggregation:
    - `--attn_pool dot` (dot-product attention, default)
    - `--attn_pool mha` (1-layer multi-head attention + CLS)
    - `--attn_pool none` (mean pooling; previous behavior)  
  - Related hyperparameters: `--attn_dropout`, `--attn_heads` (for MHA).  
- **Changed**  
  - Replaced classifier input with **attention-weighted latent** (reflects temporal importance).  
  - Added **Encoder + Pooling tag** to training/eval logs (e.g., `LSTM + DOT`).  
- **Fixed**  
  - Made Kalman-to-tensor conversion more robust for rare device/dtype boundary cases.

**v0.1.0 — 2025-09-25**  
- **Added**  
  - Added **LSTM encoder** support (`--encoder lstm`) to better capture temporal dependencies.  
  - Combined LSTM latent with Kalman smoothing for more stable time-series representations.  
- **Changed**  
  - Switched classifier input to LSTM-based latent (preserves sequence information).  
- **Fixed**  
  - Resolved LSTM output/head dimension mismatch and sequence length handling issues.

**v0.0.5 — 2025-09-25**  
- **Added**  
  - **F1-score** computation and logging.  
- **Changed**  
  - Stabilized preprocessing/reporting and cleaned up metric outputs.

**v0.0.4 — 2025-09-25**  
- **Added**  
  - Option for **global training after per-subject normalization** (merged data).  
- **Changed**  
  - Refined standardization/visualization pipeline.

**v0.0.3 — 2025-09-25**  
- **Added**  
  - **RBF-SVM boundary approximation** and flexible `--svm_gamma` options (numeric/string).  
- **Changed**  
  - Applied **StandardScaler after PCA** and enhanced boundary visualization.  
- **Fixed**  
  - Added safe fallback behavior when training a boundary with a single class.

**v0.0.2 — 2025-09-25**  
- **Added**  
  - Per-subject K-Fold CV, Kalman, PCA 2D + boundary approximation, and result directory layout.  
- **Changed**  
  - Improved DataLoader/AE.  
- **Fixed**  
  - Handled single-class/report warnings.

**v0.0.1:** First version


