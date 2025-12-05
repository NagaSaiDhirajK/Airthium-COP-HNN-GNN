
# AIRTHIUM Optimal Dataset Usage Example

import h5py
import torch
import numpy as np

# Load optimal dataset
with h5py.File('airthium_optimal_dataset.h5', 'r') as f:
    # HNN data (fixed trajectory length)
    hnn_trajectories = torch.tensor(f['hnn/trajectories'][:], dtype=torch.float32)
    hnn_targets = torch.tensor(f['hnn/targets'][:], dtype=torch.float32)
    
    # GNN data (graph structure)
    node_features = torch.tensor(f['gnn/node_features'][:], dtype=torch.float32)
    edge_index = torch.tensor(f['gnn/edge_index'][:], dtype=torch.long)
    edge_features = torch.tensor(f['gnn/edge_features'][:], dtype=torch.float32)
    graph_targets = torch.tensor(f['gnn/graph_targets'][:], dtype=torch.float32)

print(f"HNN trajectories: {hnn_trajectories.shape}")  # [100, 100, 7]
print(f"GNN node features: {node_features.shape}")    # [100, 5, 8]
print("Ready for GPU training on Kaggle T4 x2!")

# Example: Create DataLoader for training
from torch.utils.data import TensorDataset, DataLoader

hnn_dataset = TensorDataset(hnn_trajectories, hnn_targets)
hnn_loader = DataLoader(hnn_dataset, batch_size=32, shuffle=True)

print("Optimal data structure ready for HNN-GNN training!")
