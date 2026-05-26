import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.stattools import grangercausalitytests

# ==========================================
# Global configuration and file paths
# ==========================================
# Compute these file locations once when the module loads.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

MODEL_PATH = os.path.join(BASE_DIR, "stgnn_weights.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ADJ_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "adjacency_matrix.npy")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_delhi_data.csv")
NODES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "node_metadata.csv")

# Features used by the STGNN model
STGNN_FEATURES = [
    'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 
    'humidity', 'wind_speed', 'visibility', 'aqi', 
    'is_weekend', 'hour_sin', 'hour_cos'
]
CAUSAL_HUBS = ["Anand Vihar, Delhi", "Punjabi Bagh, Delhi", "Siri Fort, Delhi"]


# ==========================================
# Part 1 — Spatio-temporal GNN model
# ==========================================
class CausalSTGNN(nn.Module):
    def __init__(self, num_nodes=23, num_features=14, hidden_dim=64, forecast_horizon=4):
        super(CausalSTGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
        self.spatial_linear = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, forecast_horizon)
        
    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x_reshaped = x.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, num_features)
        gru_out, hidden = self.gru(x_reshaped) 
        last_hidden = hidden[-1].view(batch_size, num_nodes, self.hidden_dim)
        spatial_out = torch.matmul(adj, last_hidden) 
        spatial_out = F.relu(self.spatial_linear(spatial_out))
        predictions = self.predictor(spatial_out)
        return predictions.transpose(1, 2)


# ==========================================
# Helper — load dataset used by the dashboard
# ==========================================
def fetch_live_data():
    """Fetches the dataset once to prevent I/O bottlenecks in the dashboard."""
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# ==========================================
# Part 2 — Forecasting and inference helpers
# ==========================================
def get_pm25_forecast(df=None):
    """Generates predictions. Accepts pre-loaded DataFrame for massive speed gains."""
    if df is None:
        df = fetch_live_data()
        
    # Step 1 — load model weights and the spatial adjacency matrix
    model = CausalSTGNN(num_nodes=23, num_features=len(STGNN_FEATURES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    model.eval()
    
    adj_tensor = torch.tensor(np.load(ADJ_PATH), dtype=torch.float32)
    
    # Step 2 — prepare the most recent real-world observations
    nodes = pd.read_csv(NODES_PATH)
    station_to_id = dict(zip(nodes['station'], nodes['station_id']))
    
    # Make an explicit copy to avoid pandas' SettingWithCopyWarning
    df_live = df.copy()
    df_live['station_id'] = df_live['station'].map(station_to_id)
    df_live = df_live.sort_values(by=['datetime', 'station_id'])
    
    # Select the most recent 16 time steps
    latest_times = df_live['datetime'].unique()[-16:]
    df_recent = df_live[df_live['datetime'].isin(latest_times)]
    
    num_nodes, num_features = len(nodes), len(STGNN_FEATURES)
    reshaped_data = df_recent[STGNN_FEATURES].values.reshape(16, num_nodes, num_features)
    
    # Step 3 — scale features using the saved scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    scaled_data_2d = scaler.transform(reshaped_data.reshape(-1, num_features))
    scaled_data = scaled_data_2d.reshape(1, 16, num_nodes, num_features)
    recent_history = torch.tensor(scaled_data, dtype=torch.float32)
    
    # Step 4 — run model inference without tracking gradients
    with torch.no_grad():
        scaled_prediction = model(recent_history, adj_tensor)
    pred_array = scaled_prediction.numpy()[0] 
    
    # Step 5 — inverse-transform the predicted PM2.5 values back to original units
    dummy_array = np.zeros((4 * num_nodes, num_features))
    dummy_array[:, 0] = pred_array.flatten()
    real_pm25 = scaler.inverse_transform(dummy_array)[:, 0].reshape(4, num_nodes)
    
    return real_pm25


# ==========================================
# Part 3 — Explainability and causal analysis
# ==========================================
def source_attribution(latest_row: pd.Series) -> pd.DataFrame:
    """Calculates feature importance based on known atmospheric chemistry proxies."""
    no2 = float(latest_row.get("no2", 0))       
    so2 = float(latest_row.get("so2", 0))       
    pm10 = float(latest_row.get("pm10", 0))     
    wind = float(latest_row.get("wind_speed", 0))
    
    traffic_impact = max(5.0, no2 * 1.5)
    industry_impact = max(5.0, so2 * 2.0)
    dust_impact = max(5.0, (pm10 - latest_row.get("pm25", 0)) * 0.5)
    regional_impact = max(5.0, 50.0 - (wind * 2)) 
    
    total = traffic_impact + industry_impact + dust_impact + regional_impact
    
    return pd.DataFrame({
        "Source": ["Traffic Exhaust", "Industrial Emissions", "Construction/Dust", "Regional Transport"],
        "Contribution (%)": [
            round((traffic_impact / total) * 100, 1),
            round((industry_impact / total) * 100, 1),
            round((dust_impact / total) * 100, 1),
            round((regional_impact / total) * 100, 1),
        ],
    }).sort_values(by="Contribution (%)", ascending=False)

def causal_graph_dot(df=None) -> str:
    """Generates a Granger Causality DAG. Accepts pre-loaded DataFrame for speed."""
    try:
        if df is None:
            df = fetch_live_data()
            
        latest_times = df['datetime'].unique()[-100:]
        df_recent = df[df['datetime'].isin(latest_times)]
        
        # Pivot the data for the causal hubs and fill forward/backward to handle gaps
        pivot_df = df_recent[df_recent['station'].isin(CAUSAL_HUBS)].pivot_table(
            index='datetime', columns='station', values='pm25'
        )
        pivot_df = pivot_df.ffill().bfill() 
        
        if len(pivot_df) < 10:
            raise ValueError(f"Not enough overlapping data points. Rows: {len(pivot_df)}")
            
        # Maximum lag (in timesteps) to test for Granger causality
        causal_edges = []
        max_lag = 3
        
        for source in CAUSAL_HUBS:
            for target in CAUSAL_HUBS:
                if source != target and source in pivot_df.columns and target in pivot_df.columns:
                    test_data = pivot_df[[target, source]]
                    gc_res = grangercausalitytests(test_data, maxlag=[max_lag], verbose=False)
                    p_value = gc_res[max_lag][0]['ssr_ftest'][1]
                    
                    if p_value < 0.05:
                        src_clean = source.split(',')[0].replace(" ", "_")
                        tgt_clean = target.split(',')[0].replace(" ", "_")
                        causal_edges.append(f'{src_clean} -> {tgt_clean} [color="red", label=" causes ", penwidth=2.0];')
        
        if not causal_edges:
             causal_edges.append('No_Strong_Causality_Found -> In_This_Time_Window [style="dashed"];')
             
        edges_str = "\n        ".join(causal_edges)
        
        return f"""
        digraph CausalGraph {{
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#E3F2FD", fontname="Helvetica", color="#1E88E5"];
            edge [fontname="Helvetica", fontsize=10];
            {edges_str}
        }}
        """

    except Exception as e:
        print(f"🔥 DEBUG - CAUSAL GRAPH CRASHED: {e}")
        return f"""
        digraph ErrorGraph {{
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#FFCDD2", color="red", fontname="Helvetica"];
            Math_Error [label="Causal Engine Offline\\nCheck Terminal"];
        }}
        """

def run_counterfactual(latest_pm25: float, traffic_cut: int, industry_cut: int, df=None) -> tuple[float, float]:
    """Calculates dynamic policy impacts based on live feature attribution."""
    try:
        if df is None:
            df = fetch_live_data()
            
        latest_row = df.iloc[-1]
        
        no2 = float(latest_row.get("no2", 0))       
        so2 = float(latest_row.get("so2", 0))       
        pm10 = float(latest_row.get("pm10", 0))     
        wind = float(latest_row.get("wind_speed", 0))
        
        traffic_impact = max(5.0, no2 * 1.5)
        industry_impact = max(5.0, so2 * 2.0)
        dust_impact = max(5.0, (pm10 - latest_row.get("pm25", 0)) * 0.5)
        regional_impact = max(5.0, 50.0 - (wind * 2)) 
        
        total = traffic_impact + industry_impact + dust_impact + regional_impact
        
        traffic_pct = traffic_impact / total
        industry_pct = industry_impact / total
        
        actual_traffic_drop_pct = traffic_pct * (traffic_cut / 100.0)
        actual_industry_drop_pct = industry_pct * (industry_cut / 100.0)
        total_drop_pct = actual_traffic_drop_pct + actual_industry_drop_pct
        
        projected_pm25 = max(0.0, latest_pm25 * (1 - total_drop_pct))
        
        return projected_pm25, (total_drop_pct * 100)
        
    except Exception:
        # Fallback: use a simple heuristic when live data isn't available
        projected_drop = (traffic_cut * 0.35) + (industry_cut * 0.25)
        projected_pm25 = max(0.0, latest_pm25 * (1 - projected_drop / 100))
        return projected_pm25, projected_drop

def policy_recommendation(latest_pm25: float) -> tuple[str, str]:
    """
    Dynamic policy generator based on Delhi's official GRAP guidelines.
    Outputs formatted Markdown lists for clean UI rendering in Streamlit.
    """
    if latest_pm25 >= 450:
        actions = (
            "🚨 **SEVERE+ EMERGENCY (GRAP Stage IV)**\n"
            "- **Traffic:** Ban entry of non-essential diesel trucks into Delhi.\n"
            "- **Industry & Construction:** Halt all construction and demolition (C&D) activities.\n"
            "- **Public Operations:** Mandate 50% Work-From-Home for corporate and municipal offices.\n"
            "- **Education:** Suspend physical classes for schools and shift to online mode."
        )
        return (actions, "Critical")
        
    elif latest_pm25 >= 300:
        actions = (
            "🔴 **SEVERE (GRAP Stage III)**\n"
            "- **Traffic:** Ban BS-III petrol and BS-IV diesel vehicles.\n"
            "- **Industry:** Suspend operations of stone crushers and brick kilns.\n"
            "- **Public Transport:** Intensify public transport frequency (Metro/Buses).\n"
            "- **Maintenance:** Mandate daily mechanized sweeping and water sprinkling."
        )
        return (actions, "High")
        
    elif latest_pm25 >= 200:
        actions = (
            "🟠 **VERY POOR (GRAP Stage II)**\n"
            "- **Traffic:** Increase parking fees drastically to discourage private transport.\n"
            "- **Intervention:** Deploy anti-smog guns at identified pollution hotspots.\n"
            "- **Commercial:** Strictly ban the use of coal and firewood in restaurants/tandoors.\n"
            "- **Power:** Ensure uninterrupted power supply to deter diesel generator use."
        )
        return (actions, "Medium")
        
    elif latest_pm25 >= 100:
        actions = (
            "🟡 **POOR (GRAP Stage I)**\n"
            "- **Enforcement:** Enforce strict fines for open waste and biomass burning.\n"
            "- **Traffic:** Synchronize traffic grids to reduce vehicle idling emissions.\n"
            "- **Compliance:** Enforce PUC (Pollution Under Control) norm compliance strictly.\n"
            "- **Construction:** Halt road construction activities generating heavy dust."
        )
        return (actions, "Moderate")
        
    else:
        actions = (
            "🟢 **MODERATE / SATISFACTORY**\n"
            "- **Monitoring:** Maintain standard regional air quality monitoring.\n"
            "- **Maintenance:** Continue routine mechanized road sweeping.\n"
            "- **Awareness:** Promote citizen awareness on public transport usage."
        )
        return (actions, "Low")