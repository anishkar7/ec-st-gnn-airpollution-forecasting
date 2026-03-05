# 📘 Project Progress Log  
**Project Title:** Explainable Causal ST-GNN Digital Twin for Air Pollution Forecasting  
**Team:** [Anish, Ashish, Rohan, Abhijit]  
**Start Date:** [05-03-2026]  
**Duration:** 10 Weeks  

---

## 🗓 Week 1 – Project Initialization

### 📅 Day 1 – Environment & Repository Setup
**Date:**  

**Tasks Completed:**
- Installed Anaconda
- Created dedicated conda environment: `ecstgnn`
- Installed PyTorch with CUDA (GPU enabled)
- Verified RTX 3080 working
- Installed Git and configured PATH
- Initialized private GitHub repository
- Created project folder structure
- Connected VS Code to conda environment
- Pushed initial project structure to `dev`
- Merged `dev` into `main`

**Challenges Faced:**
- PowerShell execution policy blocking conda
- Git not recognized in terminal
- VS Code interpreter detection issues

**Solutions Implemented:**
- Set execution policy to RemoteSigned
- Added Git to system PATH
- Ran `conda init powershell`
- Restarted system to apply changes

**Status:** ✅ Setup Complete

---

## 📅 Day 2 – Problem Definition & Design (Planned)

**Goals:**
- Finalize target variable
- Select forecast horizon
- Select city dataset
- Define graph structure
- Plan data acquisition pipeline

---

# 📊 Dataset Decisions

| Parameter | Selected Option | Reason |
|------------|-----------------|--------|
| City |  |  |
| Forecast Horizon |  |  |
| Target Variable | PM2.5 | Primary pollutant indicator |
| Graph Type | Distance-based adjacency | Initial simple approach |

---

# 🧠 Model Design Notes

- Nodes = Monitoring stations
- Edges = Spatial relationships
- Features = Pollution + Meteorological variables
- Output = Multi-step PM2.5 forecast
- Explainability = Attention weights + feature attribution
- Causality = Intervention-based analysis (Phase 2)

---

# 📈 Weekly Milestones

| Week | Milestone |
|------|-----------|
| 1 | Environment setup + Dataset selection |
| 2 | Data preprocessing pipeline |
| 3 | Graph construction + adjacency matrix |
| 4 | Baseline model implementation |
| 5 | ST-GNN model implementation |
| 6 | Explainability module |
| 7 | Causal analysis module |
| 8 | Evaluation + Comparison |
| 9 | Paper drafting |
| 10 | Final polishing + submission |

---

# 🔍 Experimental Tracking

## Baseline Results

| Model | MAE | RMSE | R² | Notes |
|--------|------|------|-----|------|
| ARIMA |  |  |  |  |
| LSTM |  |  |  |  |
| GRU |  |  |  |  |
| ST-GNN |  |  |  |  |

---

# 🧾 Notes & Observations

-  
-  
-  

---

# 🚀 Next Immediate Action

[tomorrow’s specific task here]