# 📘 Project Progress Log  
**Project Title:** Explainable Causal ST-GNN Digital Twin for Air Pollution Forecasting  
**Team:** Anish, Ashish, Rohan, Abhijit  
**Start Date:** 05-03-2026  
**Duration:** 10 Weeks  

---

## 🗓 Week 1 & 2 – Project Initialization & Phase 1 (Data Engineering)

### 📅 Environment & Repository Setup
**Tasks Completed:**
- Installed Anaconda and created dedicated conda environment: `ecstgnn`
- Installed PyTorch with CUDA (GPU enabled) and verified RTX 3080 working
- Initialized private GitHub repository and established folder structure (`app/`, `notebooks/`, `src/`, `data/`)
- Resolved PowerShell execution policy and VS Code interpreter detection issues

### 📅 Data Preprocessing Pipeline
**Tasks Completed:**
- Finalized dataset selection: Delhi-NCR region pollution and meteorological data.
- Created `data_preprocessing.ipynb`.
- Cleaned missing values, standardized timestamps to 6-hour intervals, and generated time-cyclical features (`hour_sin`, `hour_cos`).
- Mapped 23 physical monitoring stations to unique integer IDs.
- Exported clean baseline data: `processed_delhi_data_new.csv` and `node_metadata.csv`.

---

## 🗓 Week 3 – Phase 1 (Graph Construction)

### 📅 Spatial Network Initialization
**Tasks Completed:**
- Created `build_graph.ipynb`.
- Computed geographical distance matrices between the 23 Delhi-NCR stations.
- Generated the static spatial network representing wind/pollution transfer pathways.
- Exported `adjacency_matrix.npy` for the Graph Neural Network layer.

---

## 🗓 Week 4 – Phase 2 (Model Architecture & UI Integration) *[Current]*

### 📅 PyTorch ST-GNN Development
**Tasks Completed:**
- Built 3D sliding-window tensor pipeline in `01_create_dataloader.ipynb`.
- Configured PyTorch DataLoader: `[Batch, 16 Time Steps, 23 Nodes, 14 Features]`.
- Implemented `CausalSTGNN` class (GRU for temporal trends + Linear/Matmul for spatial message passing).
- Scaled data using `StandardScaler` and saved `scaler.pkl`.
- Successfully trained the baseline model over 10 epochs (MSE reduced from 0.0814 to 0.0397).
- Exported trained model weights: `stgnn_weights.pth`.

### 📅 Streamlit Dashboard Integration
**Tasks Completed:**
- Built the production inference engine (`app/models/inference.py`).
- Wired the AI model to fetch *real* historical data from the CSV, scale it, run a forward pass, and inverse-transform the predictions.
- Refactored `app/utils/data_loader.py` and `app/components/forecast.py` to natively ingest PyTorch tensors.
- Successfully verified the frontend UI dynamically updating the chart and metric cards with live ST-GNN forecasts.
- Synced all changes to GitHub via `main` branch.

**Status:** ✅ Phase 2 Complete

---

# 📊 Dataset Decisions

| Parameter | Selected Option | Reason |
|------------|-----------------|--------|
| City | Delhi-NCR | High baseline pollution levels, dense sensor network, severe seasonal variations. |
| Forecast Horizon | 24 Hours (4 steps of 6h) | Optimal balance between actionable policy lead-time and model accuracy. |
| Target Variable | PM2.5 | Primary pollutant indicator posing highest health risk in the region. |
| Graph Type | Distance-based adjacency | Initial approach to map geographical proximity and base wind dispersion. |

---

# 🧠 Model Design Notes

- **Nodes:** 23 physical air quality monitoring stations in Delhi.
- **Edges:** Spatial relationships (Adjacency Matrix).
- **Features:** 14 total (PM2.5, PM10, NO2, SO2, CO, O3, Temp, Humidity, Wind Speed, Visibility, AQI, Is_Weekend, Hour_Sin, Hour_Cos).
- **Architecture:** GRU extracts temporal memory (past 4 days) -> GNN applies spatial adjacency weights -> Linear layer outputs 24-hour forecast.
- **Explainability:** Feature attribution (SHAP/Integrated Gradients) to determine pollutant source.
- **Causality:** Spatial Granger Causality / DAGs to determine node-to-node pollution transfer (Phase 3).

---

# 📈 Weekly Milestones

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Environment setup + Dataset selection | ✅ Done |
| 2 | Data preprocessing pipeline | ✅ Done |
| 3 | Graph construction + adjacency matrix | ✅ Done |
| 4 | ST-GNN architecture + Dashboard Integration | ✅ Done |
| 5 | Explainability module (Feature Attribution) | ⏳ Next |
| 6 | Causal analysis module (Spatial Graph) | ⏳ Pending |
| 7 | Counterfactuals & Policy Simulation Engine | ⏳ Pending |
| 8 | Evaluation + Comparison against Baselines | ⏳ Pending |
| 9 | Paper drafting | ⏳ Pending |
| 10 | Final polishing + submission | ⏳ Pending |

---

# 🧾 Notes & Observations

- **Data Quirk Discovered:** AQI readings across northern stations (e.g., Wazirpur, Jahangirpuri) frequently flatline at exactly **500**. This is not a code error, but rather the official Indian National Air Quality Index scale cap, combined with standard dataset clipping for Severe+ conditions.
- **Data Loader Optimization:** Converting the raw pandas dataframe into a flattened 2D array for the `StandardScaler` and reshaping back to a `[16, 23, 14]` 3D PyTorch tensor proved critical for maintaining feature alignment across the 23 isolated nodes.

---

# 🚀 Next Immediate Action

**Phase 3 Initialization (Explainability Engine):** Determine the mathematical approach to open the model's "black box." Decide whether to implement SHAP values first (for the "Source Attribution" tab) or Granger Causality logic (for the "Causal Graph" tab).