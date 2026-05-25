import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from app.models.inference import CausalSTGNN, STGNN_FEATURES

print("🏋️ Starting Genuine ST-GNN Training Run...")

# 1. Setup Paths & Load Data
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_delhi_data.csv")
ADJ_PATH = os.path.join(BASE_DIR, "data", "processed", "adjacency_matrix.npy")
NODES_PATH = os.path.join(BASE_DIR, "data", "processed", "node_metadata.csv")
SCALER_PATH = os.path.join(BASE_DIR, "app", "models", "scaler.pkl")

df = pd.read_csv(DATA_PATH)
adj_matrix = torch.tensor(np.load(ADJ_PATH), dtype=torch.float32)
nodes = pd.read_csv(NODES_PATH)

station_to_id = dict(zip(nodes['station'], nodes['station_id']))
df['station_id'] = df['station'].map(station_to_id)
df = df.sort_values(by=['datetime', 'station_id'])

# 2. Reshape into Spatiotemporal Tensor [Time steps, Nodes, Features]
unique_times = df['datetime'].unique()
num_nodes = len(nodes)
num_features = len(STGNN_FEATURES)
total_timesteps = len(unique_times)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

raw_data = df[STGNN_FEATURES].values.reshape(total_timesteps, num_nodes, num_features)
scaled_data = scaler.transform(raw_data.reshape(-1, num_features)).reshape(total_timesteps, num_nodes, num_features)

# 3. Create Sliding Window Sequences (Lookback=16, Horizon=4)
X, Y = [], []
for i in range(total_timesteps - 16 - 4 + 1):
    X.append(scaled_data[i : i + 16])
    # Target is the PM2.5 values (feature index 0) for the next 4 steps
    Y.append(scaled_data[i + 16 : i + 16 + 4, :, 0])

X = torch.tensor(np.array(X), dtype=torch.float32)
Y = torch.tensor(np.array(Y), dtype=torch.float32)

# Chronological 80/20 Train/Val Split
train_idx = int(len(X) * 0.8)
X_train, Y_train = X[:train_idx], Y[:train_idx]
X_val, Y_val = X[train_idx:], Y[train_idx:]

# 4. Initialize Model & Training Parameters
model = CausalSTGNN(num_nodes=23, num_features=14, hidden_dim=64, forecast_horizon=4)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
epochs = 20

# 5. Live Training Execution Loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Process in standard optimization passes
    output = model(X_train, adj_matrix)
    loss = criterion(output, Y_train)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate Validation Metrics
    model.eval()
    with torch.no_grad():
        val_output = model(X_val, adj_matrix)
        val_loss = criterion(val_output, Y_val)
        
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")

# 6. Plot & Save Authentic Visual Asset
plt.style.use('seaborn-v0_8-white')
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss (Huber)', color='#1E3A8A', linewidth=2.5)
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss (Huber)', color='#EF4444', linewidth=2.0, linestyle='--')
plt.title('Figure 4.1: ST-GNN Training & Validation Convergence', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Loss Value', fontweight='bold')
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.legend(frameon=True)
sns.despine()
plt.tight_layout()

save_path = os.path.join(BASE_DIR, "thesis_plots", "fig4_1_loss_curves.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🏁 Real loss curves generated and saved to: {save_path}")