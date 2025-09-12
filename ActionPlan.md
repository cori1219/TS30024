# IMU-Based Balance Anomaly Detection AI Model Development Project Plan

## 1. Project Overview
- **Project Name**: IMU-Based Balance Anomaly Detection AI Model Development  
- **Duration**: July 2025 ~ December 2025  
- **Objective**: Develop and validate a lightweight AI model that can detect users' balance anomalies in real time using smartphone IMU sensor data, helping to prevent accidents and ensure safety.  

---

## 2. Project Phases

### **Phase 1: Data Collection (2025.07 ~ 2025.09)**
- **Goal**: Collect at least 800 samples of normal and abnormal gait data  
- **Key Tasks**:
  - Conduct gait experiments simulating impaired balance using special goggles  
  - Recruit over 40 participants for experiments  
  - Collect and preprocess IMU sensor data (accelerometer, gyroscope)  
  - Establish data quality checks and labeling standards  
- **Deliverables**:
  - Dataset (normal/abnormal gait data)  
  - Data preprocessing pipeline  

---

### **Phase 2: AI Model Development & Training (2025.09 ~ 2025.10)**
- **Goal**: Achieve accuracy ≥ 90%, F1-Score ≥ 0.90  
- **Key Tasks**:
  - Compare performance of LSTM and TSMixer models, select the optimal one  
  - Apply mobile optimization techniques (Quantization, Pruning)  
  - Pre-train with early data, retrain with full dataset for best performance  
- **Deliverables**:
  - Optimized AI model for mobile devices  
  - Model performance evaluation report  

---

### **Phase 3: Test Tool & Feedback Environment (2025.11)**
- **Goal**: Build a real-time detection tool with user feedback functionality  
- **Key Tasks**:
  - Develop a test app/tool integrated with the AI model  
  - Implement UI for detection results visualization and log storage  
  - Create a simple interface for collecting user feedback  
- **Deliverables**:
  - Test app or prototype  
  - User feedback dataset  

---

### **Phase 4: Pilot Test & Final Analysis (2025.12)**
- **Goal**: Evaluate user acceptance and derive final improvements  
- **Key Tasks**:
  - Recruit pilot test participants and conduct real-world testing  
  - Evaluate model accuracy, latency, and user satisfaction  
  - Summarize improvements and propose next research steps  
- **Deliverables**:
  - Pilot test results report  
  - Final improvement plan and next-phase proposal  

---

## 3. Project Timeline (Gantt Chart Style)

| Phase                       | 2025.07 | 2025.08 | 2025.09 | 2025.10 | 2025.11 | 2025.12 |
|-----------------------------|---------|---------|---------|---------|---------|---------|
| Data Collection              | ●       | ●       | ●       |         |         |         |
| Model Development & Training |         |         | ●       | ●       |         |         |
| Test Tool Development        |         |         |         |         | ●       |         |
| Pilot Test & Analysis         |         |         |         |         |         | ●       |

---

## 4. Key Performance Indicators (KPIs)

- **Model Performance**: Accuracy ≥ 90%, F1-Score ≥ 0.90  
- **Real-Time Detection**: Result returned within 1 second after data collection  
- **User Acceptance**: ≥ 80% of pilot users respond "helpful"  
- **Data Volume**: ≥ 800 gait samples with balanced normal/abnormal data  

---

## 5. Future Expansion Directions

- Expand data collection to diverse environments (e.g., outdoor, uneven terrain)  
- Enhance functionality for high-risk groups (elderly, patients with neurological disorders)  
- Explore integration with real-time risk alert systems  
