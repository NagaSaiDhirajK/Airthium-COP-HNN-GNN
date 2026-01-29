#!/usr/bin/env python3
"""
Comprehensive Data Loader for HNN-GNN Heat Pump COP Prediction
===============================================================

This data loader loads the comprehensive physics dataset and prepares
it for training with the Hybrid HNN-GNN model.

Dataset Structure (from comprehensive_physics_generator_v2.py):
- HNN temporal: 100 timesteps × 14 channels
- GNN nodes: 10 nodes × 35 features
- GNN edges: 11 edges × 12 features
- GNN graph: 50 features
- Targets: COP_heating, COP_cooling

Author: Comprehensive physics-based data loader
"""

import sys
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


class ComprehensiveDataset(Dataset):
    """
    PyTorch Dataset for comprehensive physics HNN-GNN data.
    
    Loads HDF5 data and prepares it for the Hybrid HNN-GNN model.
    """
    
    # Heat pump graph topology (10 nodes, 11 edges)
    EDGE_INDEX = torch.tensor([
        # Source nodes
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        # Target nodes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5]
    ], dtype=torch.long)
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 normalize: bool = True,
                 random_seed: int = 42):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing physics_data.h5
            split: 'train', 'val', or 'test'
            train_ratio: Fraction for training
            normalize: Whether to normalize features
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        
        # Load data
        self._load_data()
        
        # Split data
        self._split_data(train_ratio, random_seed)
        
        # Compute normalization stats if needed
        if normalize:
            self._compute_normalization()
    
    def _load_data(self):
        """Load data from HDF5 file."""
        h5_path = self.data_dir / 'physics_data.h5'
        info_path = self.data_dir / 'dataset_info.json'
        
        with h5py.File(h5_path, 'r') as f:
            # Try to get n_cycles from HDF5 attributes first, then JSON, then infer from data
            if 'n_cycles' in f.attrs:
                n_cycles = f.attrs['n_cycles']
            elif info_path.exists():
                with open(info_path, 'r') as jf:
                    info = json.load(jf)
                    n_cycles = info['n_cycles']
            else:
                # Infer from dataset shape
                n_cycles = f['targets/COP_heating'].shape[0]
            
            # Load targets
            self.cop_heating = f['targets/COP_heating'][:]
            self.cop_cooling = f['targets/COP_cooling'][:]
            
            # Load temporal HNN data (trajectories for physics losses)
            self.T_evap_traj = f['temporal_hnn/T_evap_traj'][:]
            self.T_cond_traj = f['temporal_hnn/T_cond_traj'][:]
            self.P_evap_traj = f['temporal_hnn/P_evap_traj'][:]
            self.P_cond_traj = f['temporal_hnn/P_cond_traj'][:]
            self.mass_flow_traj = f['temporal_hnn/mass_flow_traj'][:]
            self.W_comp_traj = f['temporal_hnn/W_comp_traj'][:]  # For energy balance loss
            self.Q_evap_traj = f['temporal_hnn/Q_evap_traj'][:]  # For energy balance loss
            self.Q_cond_traj = f['temporal_hnn/Q_cond_traj'][:]  # For energy balance loss
            self.superheat_traj = f['temporal_hnn/superheat_traj'][:]
            self.subcooling_traj = f['temporal_hnn/subcooling_traj'][:]
            self.hamiltonian_q = f['temporal_hnn/hamiltonian_q'][:]
            self.hamiltonian_p = f['temporal_hnn/hamiltonian_p'][:]
            self.hamiltonian_H = f['temporal_hnn/hamiltonian_H'][:]  # For Hamiltonian conservation loss
            self.control_signals = f['temporal_hnn/control_signals'][:]
            
            # Load GNN data
            self.node_features = f['gnn/node_features'][:]
            self.edge_features = f['gnn/edge_features'][:]
            self.graph_features = f['gnn/graph_features'][:]
            
            # Load additional physics for loss functions
            self.T_evap = f['cycle_parameters/T_evap'][:]
            self.T_cond = f['cycle_parameters/T_cond'][:]
            self.carnot_cop = f['validation/carnot_cop'][:]
            self.carnot_efficiency = f['validation/carnot_efficiency'][:]
        
        self.n_cycles = n_cycles
        print(f"[LOAD] Loaded {n_cycles} cycles from {h5_path}")
    
    def _split_data(self, train_ratio: float, random_seed: int):
        """Split data into train/val/test sets."""
        np.random.seed(random_seed)
        indices = np.random.permutation(self.n_cycles)
        
        train_size = int(train_ratio * self.n_cycles)
        val_size = int((1 - train_ratio) / 2 * self.n_cycles)
        
        if self.split == 'train':
            self.indices = indices[:train_size]
        elif self.split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
        
        print(f"[SPLIT] {self.split}: {len(self.indices)} samples")
    
    def _compute_normalization(self):
        """Compute normalization statistics from training data."""
        # Only compute on training split
        train_indices = self.indices if self.split == 'train' else np.arange(int(0.8 * self.n_cycles))
        
        # Compute data-driven normalization from actual data ranges
        # Using min-max style normalization parameters
        self.temporal_mean = {
            'T_evap': np.mean(self.T_evap_traj[train_indices]),
            'T_cond': np.mean(self.T_cond_traj[train_indices]),
            'P_evap': np.mean(self.P_evap_traj[train_indices]),
            'P_cond': np.mean(self.P_cond_traj[train_indices]),
            'mass_flow': np.mean(self.mass_flow_traj[train_indices]),
            'W_comp': np.mean(self.W_comp_traj[train_indices]),
            'Q_evap': np.mean(self.Q_evap_traj[train_indices]),
            'Q_cond': np.mean(self.Q_cond_traj[train_indices]),
            'superheat': np.mean(self.superheat_traj[train_indices]),
            'subcooling': np.mean(self.subcooling_traj[train_indices])
        }
        self.temporal_std = {
            'T_evap': np.std(self.T_evap_traj[train_indices]) + 1e-6,
            'T_cond': np.std(self.T_cond_traj[train_indices]) + 1e-6,
            'P_evap': np.std(self.P_evap_traj[train_indices]) + 1e-6,
            'P_cond': np.std(self.P_cond_traj[train_indices]) + 1e-6,
            'mass_flow': np.std(self.mass_flow_traj[train_indices]) + 1e-6,
            'W_comp': np.std(self.W_comp_traj[train_indices]) + 1e-6,
            'Q_evap': np.std(self.Q_evap_traj[train_indices]) + 1e-6,
            'Q_cond': np.std(self.Q_cond_traj[train_indices]) + 1e-6,
            'superheat': np.std(self.superheat_traj[train_indices]) + 1e-6,
            'subcooling': np.std(self.subcooling_traj[train_indices]) + 1e-6
        }
        
        # Hamiltonian normalization
        self.hamiltonian_mean = np.mean(self.hamiltonian_q[train_indices])
        self.hamiltonian_std = np.std(self.hamiltonian_q[train_indices]) + 1e-6
        
        # COP normalization
        self.cop_mean = np.mean(self.cop_heating[train_indices])
        self.cop_std = np.std(self.cop_heating[train_indices])
        
        print(f"[NORM] Computed normalization from {len(train_indices)} training samples")
        print(f"       COP: mean={self.cop_mean:.3f}, std={self.cop_std:.3f}")
    
    def _build_hnn_input(self, idx: int) -> torch.Tensor:
        """
        Build HNN input tensor from temporal features.
        
        Output shape: [100, 14] (14 channels for HNN)
        Uses data-driven z-score normalization.
        """
        # Stack temporal features with data-driven normalization
        features = [
            (self.T_evap_traj[idx] - self.temporal_mean['T_evap']) / self.temporal_std['T_evap'],
            (self.T_cond_traj[idx] - self.temporal_mean['T_cond']) / self.temporal_std['T_cond'],
            (self.P_evap_traj[idx] - self.temporal_mean['P_evap']) / self.temporal_std['P_evap'],
            (self.P_cond_traj[idx] - self.temporal_mean['P_cond']) / self.temporal_std['P_cond'],
            (self.mass_flow_traj[idx] - self.temporal_mean['mass_flow']) / self.temporal_std['mass_flow'],
            (self.W_comp_traj[idx] - self.temporal_mean['W_comp']) / self.temporal_std['W_comp'],
            (self.Q_evap_traj[idx] - self.temporal_mean['Q_evap']) / self.temporal_std['Q_evap'],
            (self.Q_cond_traj[idx] - self.temporal_mean['Q_cond']) / self.temporal_std['Q_cond'],
            (self.superheat_traj[idx] - self.temporal_mean['superheat']) / self.temporal_std['superheat'],
            (self.subcooling_traj[idx] - self.temporal_mean['subcooling']) / self.temporal_std['subcooling'],
        ]
        
        # Stack to [100, 10]
        temporal = np.stack(features, axis=-1)
        
        # Add Hamiltonian features [100, 4] - normalized
        hamiltonian_features = (self.hamiltonian_q[idx] - self.hamiltonian_mean) / self.hamiltonian_std
        
        # Combine: [100, 14]
        hnn_input = np.concatenate([temporal, hamiltonian_features], axis=-1)
        
        # Clip extreme values to prevent NaN
        hnn_input = np.clip(hnn_input, -10, 10)
        
        return torch.tensor(hnn_input, dtype=torch.float32)
    
    def _build_gnn_data(self, idx: int) -> Data:
        """
        Build PyTorch Geometric Data object for GNN.
        
        - node_features: [10, 35]
        - edge_features: [11, 12]
        - graph_features: [50]
        """
        node_feat = torch.tensor(self.node_features[idx], dtype=torch.float32)
        edge_feat = torch.tensor(self.edge_features[idx], dtype=torch.float32)
        graph_feat = torch.tensor(self.graph_features[idx], dtype=torch.float32)
        
        return Data(
            x=node_feat,
            edge_index=self.EDGE_INDEX,
            edge_attr=edge_feat,
            graph_attr=graph_feat
        )
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Data]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - hnn_input: [100, 14]
            - gnn_data: PyG Data object
            - cop_target: [1]
            - carnot_cop: [1]
            - T_evap: [1]
            - T_cond: [1]
            - hamiltonian_H: [100] - for Hamiltonian conservation loss
            - Q_cond_traj: [100] - for energy balance loss
            - Q_evap_traj: [100] - for energy balance loss
            - W_comp_traj: [100] - for energy balance loss
        """
        actual_idx = self.indices[idx]
        
        # Build inputs
        hnn_input = self._build_hnn_input(actual_idx)
        gnn_data = self._build_gnn_data(actual_idx)
        
        # Build targets
        cop_target = torch.tensor([self.cop_heating[actual_idx]], dtype=torch.float32)
        carnot_cop = torch.tensor([self.carnot_cop[actual_idx]], dtype=torch.float32)
        T_evap = torch.tensor([self.T_evap[actual_idx]], dtype=torch.float32)
        T_cond = torch.tensor([self.T_cond[actual_idx]], dtype=torch.float32)
        carnot_eff = torch.tensor([self.carnot_efficiency[actual_idx]], dtype=torch.float32)
        
        # Physics data for loss functions
        hamiltonian_H = torch.tensor(self.hamiltonian_H[actual_idx], dtype=torch.float32)
        Q_cond_traj = torch.tensor(self.Q_cond_traj[actual_idx], dtype=torch.float32)
        Q_evap_traj = torch.tensor(self.Q_evap_traj[actual_idx], dtype=torch.float32)
        W_comp_traj = torch.tensor(self.W_comp_traj[actual_idx], dtype=torch.float32)
        
        return {
            'hnn_input': hnn_input,
            'gnn_data': gnn_data,
            'cop_target': cop_target,
            'carnot_cop': carnot_cop,
            'T_evap': T_evap,
            'T_cond': T_cond,
            'carnot_efficiency': carnot_eff,
            'hamiltonian_H': hamiltonian_H,
            'Q_cond_traj': Q_cond_traj,
            'Q_evap_traj': Q_evap_traj,
            'W_comp_traj': W_comp_traj
        }


def collate_comprehensive(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, Batch]]:
    """
    Custom collate function for comprehensive dataset.
    
    Combines individual samples into batched tensors and batched graph.
    """
    # Stack HNN inputs: [batch_size, 100, 14]
    hnn_inputs = torch.stack([item['hnn_input'] for item in batch], dim=0)
    
    # Batch GNN data using PyG Batch
    gnn_data_list = [item['gnn_data'] for item in batch]
    gnn_batch = Batch.from_data_list(gnn_data_list)
    
    # Stack scalar targets
    cop_targets = torch.stack([item['cop_target'] for item in batch], dim=0)
    carnot_cops = torch.stack([item['carnot_cop'] for item in batch], dim=0)
    T_evaps = torch.stack([item['T_evap'] for item in batch], dim=0)
    T_conds = torch.stack([item['T_cond'] for item in batch], dim=0)
    carnot_effs = torch.stack([item['carnot_efficiency'] for item in batch], dim=0)
    
    # Stack physics trajectories for loss functions
    hamiltonian_H = torch.stack([item['hamiltonian_H'] for item in batch], dim=0)
    Q_cond_traj = torch.stack([item['Q_cond_traj'] for item in batch], dim=0)
    Q_evap_traj = torch.stack([item['Q_evap_traj'] for item in batch], dim=0)
    W_comp_traj = torch.stack([item['W_comp_traj'] for item in batch], dim=0)
    
    return {
        'hnn_input': hnn_inputs,
        'gnn_data': gnn_batch,
        'cop_target': cop_targets,
        'carnot_cop': carnot_cops,
        'T_evap': T_evaps,
        'T_cond': T_conds,
        'carnot_efficiency': carnot_effs,
        'hamiltonian_H': hamiltonian_H,
        'Q_cond_traj': Q_cond_traj,
        'Q_evap_traj': Q_evap_traj,
        'W_comp_traj': W_comp_traj
    }


def create_data_loaders(data_dir: str,
                        batch_size: int = 16,
                        num_workers: int = 0,
                        train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Fraction of data for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ComprehensiveDataset(data_dir, split='train', train_ratio=train_ratio)
    val_dataset = ComprehensiveDataset(data_dir, split='val', train_ratio=train_ratio)
    test_dataset = ComprehensiveDataset(data_dir, split='test', train_ratio=train_ratio)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_comprehensive,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_comprehensive,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_comprehensive,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def test_data_loader():
    """Test the comprehensive data loader."""
    print("=" * 70)
    print("TESTING COMPREHENSIVE DATA LOADER")
    print("=" * 70)
    
    # Load dataset
    data_dir = "comprehensive_dataset_test"
    
    print("\n1. Creating datasets...")
    train_dataset = ComprehensiveDataset(data_dir, split='train')
    val_dataset = ComprehensiveDataset(data_dir, split='val')
    test_dataset = ComprehensiveDataset(data_dir, split='test')
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    print("\n2. Testing single sample...")
    sample = train_dataset[0]
    print(f"   HNN input shape: {sample['hnn_input'].shape}")
    print(f"   GNN node features: {sample['gnn_data'].x.shape}")
    print(f"   GNN edge features: {sample['gnn_data'].edge_attr.shape}")
    print(f"   GNN graph features: {sample['gnn_data'].graph_attr.shape}")
    print(f"   COP target: {sample['cop_target'].item():.3f}")
    print(f"   Carnot COP: {sample['carnot_cop'].item():.3f}")
    
    print("\n3. Testing data loader...")
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=8)
    
    batch = next(iter(train_loader))
    print(f"   Batch HNN input shape: {batch['hnn_input'].shape}")
    print(f"   Batch GNN data: {batch['gnn_data']}")
    print(f"   Batch COP targets shape: {batch['cop_target'].shape}")
    
    print("\n4. Checking data values...")
    cop_targets = batch['cop_target']
    print(f"   COP range: [{cop_targets.min().item():.2f}, {cop_targets.max().item():.2f}]")
    print(f"   COP mean: {cop_targets.mean().item():.2f}")
    
    hnn_input = batch['hnn_input']
    print(f"   HNN input range: [{hnn_input.min().item():.2f}, {hnn_input.max().item():.2f}]")
    print(f"   HNN has NaN: {torch.isnan(hnn_input).any().item()}")
    
    print("\n[SUCCESS] Data loader test passed!")
    print("=" * 70)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    test_data_loader()
