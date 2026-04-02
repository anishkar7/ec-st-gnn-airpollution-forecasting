import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🧠 PART 1: THE DEEP LEARNING BRAIN (ST-GNN)
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

def get_pm25_forecast():
    """Loads the trained ST-GNN model, pulls the last 16 timesteps of real data, makes a prediction, and returns un-scaled PM2.5 values."""
    
    # Dynamic absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    
    model_path = os.path.join(base_dir, "stgnn_weights.pth")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    adj_path = os.path.join(project_root, "data", "processed", "adjacency_matrix.npy")
    data_path = os.path.join(project_root, "data", "processed", "processed_delhi_data.csv")
    nodes_path = os.path.join(project_root, "data", "processed", "node_metadata.csv")
    
    # 1. Load the Model & Adjacency Matrix
    model = CausalSTGNN()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    
    adj_tensor = torch.tensor(np.load(adj_path), dtype=torch.float32)
    
    # 2. Fetch the REAL recent history from the dataset
    df = pd.read_csv(data_path)
    nodes = pd.read_csv(nodes_path)
    
    # Ensure correct station ordering
    station_to_id = dict(zip(nodes['station'], nodes['station_id']))
    df['station_id'] = df['station'].map(station_to_id)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by=['datetime', 'station_id'])
    
    # Grab the last 16 unique timestamps
    latest_times = df['datetime'].unique()[-16:]
    df_recent = df[df['datetime'].isin(latest_times)]
    
    features = [
        'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 
        'humidity', 'wind_speed', 'visibility', 'aqi', 
        'is_weekend', 'hour_sin', 'hour_cos'
    ]
    
    num_nodes = len(nodes)
    num_features = len(features)
    
    # Reshape to [Time, Nodes, Features]
    raw_data = df_recent[features].values
    reshaped_data = raw_data.reshape(16, num_nodes, num_features)
    
    # 3. Scale the real data using our saved scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    # Flatten, transform, then reshape back to a 4D Tensor: [Batch=1, Time=16, Nodes=23, Features=14]
    scaled_data_2d = scaler.transform(reshaped_data.reshape(-1, num_features))
    scaled_data = scaled_data_2d.reshape(1, 16, num_nodes, num_features)
    recent_history = torch.tensor(scaled_data, dtype=torch.float32)
    
    # 4. Make the Prediction
    with torch.no_grad():
        scaled_prediction = model(recent_history, adj_tensor)
        
    pred_array = scaled_prediction.numpy()[0] # Shape: [4, 23]
    
    # 5. Un-scale back to real PM2.5 numbers
    dummy_array = np.zeros((4 * 23, 14))
    dummy_array[:, 0] = pred_array.flatten()
    
    real_pm25_flat = scaler.inverse_transform(dummy_array)[:, 0]
    real_pm25 = real_pm25_flat.reshape(4, 23)
    
    return real_pm25

# ==========================================
# 📊 PART 2: CAUSAL & POLICY LOGIC (FRONTEND HOOKS)
# ==========================================

def source_attribution(latest_row: pd.Series) -> pd.DataFrame:
    # Deterministic placeholder split for demo stability.
    traffic = min(55.0, max(25.0, 25.0 + float(latest_row.get("no2", 80.0)) / 6.0))
    weather = min(45.0, max(20.0, 35.0 + (50.0 - float(latest_row.get("humidity", 50.0))) / 5.0))
    regional = max(10.0, 100.0 - traffic - weather)

    return pd.DataFrame(
        {
            "Source": ["Traffic", "Meteorology", "Regional Transport"],
            "Contribution (%)": [round(traffic, 1), round(weather, 1), round(regional, 1)],
        }
    )

def causal_graph_dot() -> str:
    return """
    digraph {
        rankdir=LR;
        Traffic -> PM25 [label=" + "];
        WindSpeed -> PM25 [label=" - "];
        Humidity -> PM25 [label=" + "];
        Temperature -> Ozone [label=" + "];
        Ozone -> PM25 [label=" + "];
    }
    """

def run_counterfactual(latest_pm25: float, traffic_cut: int, industry_cut: int) -> tuple[float, float]:
    projected_drop = (traffic_cut * 0.35) + (industry_cut * 0.25)
    projected_pm25 = max(0.0, latest_pm25 * (1 - projected_drop / 100))
    return projected_pm25, projected_drop

def policy_recommendation(latest_pm25: float) -> tuple[str, str]:
    if latest_pm25 >= 300:
        return (
            "Activate graded response: traffic rationing + construction curbs + emergency advisories",
            "High",
        )
    if latest_pm25 >= 200:
        return (
            "Target heavy-duty vehicle restrictions and dust suppression in critical corridors",
            "Medium",
        )
    return ("Maintain monitoring mode with preventive control actions", "Medium")