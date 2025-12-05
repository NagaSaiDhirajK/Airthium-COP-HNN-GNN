
import torch
import h5py
import json
from torch_geometric.data import Data, DataLoader
import numpy as np

def load_gnn_dataset(file_path):
    """Load GNN dataset from HDF5 file."""
    
    with h5py.File(file_path, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        
        # Load graph structure
        edge_index = torch.tensor(f['graph_structure/edge_index'][:], dtype=torch.long)
        adjacency_matrix = torch.tensor(f['graph_structure/adjacency_matrix'][:], dtype=torch.float)
        
        # Load features
        node_features = torch.tensor(f['features/node_features'][:], dtype=torch.float)
        edge_features = torch.tensor(f['features/edge_features'][:], dtype=torch.float)
        graph_features = torch.tensor(f['features/graph_features'][:], dtype=torch.float)
        
        # Load targets
        targets = torch.tensor(f['targets'][:], dtype=torch.float)
        
        return {
            'metadata': metadata,
            'edge_index': edge_index,
            'adjacency_matrix': adjacency_matrix,
            'node_features': node_features,
            'edge_features': edge_features,
            'graph_features': graph_features,
            'targets': targets
        }

def create_pyg_data_list(dataset):
    """Convert dataset to PyTorch Geometric Data objects."""
    
    data_list = []
    n_cycles, trajectory_length = dataset['node_features'].shape[:2]
    
    for cycle in range(n_cycles):
        for t in range(trajectory_length):
            data = Data(
                x=dataset['node_features'][cycle, t],  # Node features
                edge_index=dataset['edge_index'],       # Edge connectivity
                edge_attr=dataset['edge_features'][cycle, t],  # Edge features
                y=dataset['targets'][cycle, t],         # Target COP
                graph_attr=dataset['graph_features'][cycle, t]  # Graph features
            )
            data_list.append(data)
    
    return data_list

def create_dataloader(data_list, batch_size=32, shuffle=True):
    """Create PyTorch Geometric DataLoader."""
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

# Example usage:
# dataset = load_gnn_dataset('gnn_dataset.h5')
# data_list = create_pyg_data_list(dataset)
# train_loader = create_dataloader(data_list, batch_size=32)
