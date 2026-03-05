# 🏙️ Explainable Causal ST-GNN Digital Twin for Real-Time Air Pollution Forecasting and Source Attribution

[![Status](https://img.shields.io/badge/Status-Research_Project-blue.svg)](#) 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#) 
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-PyTorch-red)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

Current Air Quality Index (AQI) systems are largely reactive—they display current pollution levels but fail to predict future spikes, identify root causes, or simulate mitigation policies. Authorities are left without the intelligent tools necessary for effective, proactive intervention.

This project introduces a **City Digital Twin for Air Pollution**, a dynamic, virtual replica of a real-world urban environment. By modeling the city as a dynamic graph and leveraging **Spatio-Temporal Graph Neural Networks (ST-GNN)** alongside **Causal AI** and **Reinforcement Learning**, this system not only predicts future AQI with quantified confidence but also learns the cause-and-effect relationships driving pollution. 

---

## 🎯 Main Objectives

* **Accurate Forecasting:** Predict AQI at multiple horizons (1h, 3h, 6h ahead).
* **Source Attribution:** Pinpoint exact pollution source contributions (e.g., traffic vs. weather).
* **Causal Discovery:** Learn complex, real-world causal relationships between environmental variables.
* **Counterfactual Simulation:** Run "what-if" scenarios for proposed policies.
* **Policy Recommendation:** Recommend optimal mitigation strategies using Reinforcement Learning.
* **Uncertainty Estimation:** Provide confidence intervals for all predictions.

---

## 🕸️ How We Model the City (Graph Representation)

We map the physical city into a computational graph that continuously learns from real-time data:

* **Nodes:** Air Quality Monitoring Stations
* **Edges:** Geographic proximity, wind-direction flow, and road connectivity
* **Node Features:** * **Pollutants:** PM2.5, PM10, NO₂
  * **Meteorology:** Temperature, Humidity, Wind speed, Wind direction
  * **Urban Data:** Traffic intensity proxy

---

## 🏗️ System Architecture

Our Digital Twin is powered by six interconnected modules:

| Module | Core Technology | Function & Output |
| :--- | :--- | :--- |
| **1. Spatio-Temporal Forecasting** | Graph Attention Network (GAT) + Temporal Attention | Models spatial dependencies and time-series data to output **future AQI values**. |
| **2. Source Attribution** | Attention Weights Mapping | Groups features (traffic, weather) and outputs **percentage contribution** (e.g., Traffic → 45%, Wind → 30%). |
| **3. Causal Discovery** | Causal Inference (NOTEARS / PCMCI) | Discovers cause-effect links between variables, outputting a **Causal Graph** explaining pollution dynamics. |
| **4. Counterfactual Simulation** | Causal Modeling | Simulates "what-if" interventions (e.g., 30% traffic reduction) to predict **how AQI would change**. |
| **5. RL Policy Layer** | Reinforcement Learning Agent | Learns optimal pollution mitigation strategies, rewarding AQI reduction with minimal disruption. Outputs **Recommended Policy**. |
| **6. Uncertainty Estimation** | Monte Carlo Dropout | Provides a confidence interval for predictions (e.g., **AQI = 170 ± 12**). |

---

## 📊 Evaluation & Metrics

The system's performance is rigorously tested across three primary domains:

### Forecasting Metrics
* **RMSE** (Root Mean Square Error)
* **MAE** (Mean Absolute Error)
* **R²** (Coefficient of Determination)

### Causal & Explainability Metrics
* **Counterfactual Consistency**
* **Causal Discovery Accuracy**

### Policy Simulation Metrics
* **AQI Reduction Percentage**
* **Stability** (Robustness under weather variations)

---

## 🚀 Why This Is Strong

This project represents a **research-level contribution** to smart city governance. By utilizing real-world Indian dataset contexts, it moves beyond theoretical models into practical applicability. 

**Key Innovations:**
* **Convergence of AI Fields:** Seamlessly merges GNNs, Causal AI, Explainable AI (XAI), and RL.
* **Actionable Intelligence:** Doesn't just predict a number; it tells policymakers *why* it's happening and *what* to do about it.

---

## 🌍 Social Impact

* **Pollution Control Boards:** Empowers authorities with proactive, predictive dashboards.
* **Traffic Management:** Enables data-driven, dynamic traffic routing policies to mitigate localized smog events.
* **Urban Planning:** Supports sustainable infrastructure development.
* **Public Health:** Protects vulnerable populations by enabling early-warning systems and targeted interventions.

---

## 🛠️ Getting Started

*(Instructions on how to clone the repository, install dependencies, and run the model will be added here as the codebase is finalized.)*

```bash
# Clone the repository
git clone [https://github.com/yourusername/Causal-STGNN-Digital-Twin.git](https://github.com/yourusername/Causal-STGNN-Digital-Twin.git)

# Navigate to the directory
cd Causal-STGNN-Digital-Twin

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python train.py
