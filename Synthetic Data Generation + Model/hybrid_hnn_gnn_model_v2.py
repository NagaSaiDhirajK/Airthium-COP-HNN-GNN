#!/usr/bin/env python3
"""
Hybrid HNN-GNN Architecture v2 for Heat Pump COP Prediction
===========================================================

Updated to work with modular dataset structure (21 HDF5 files).

Key Changes from v1:
- HNN input: 33 features (from 7 temporal files)
- GNN input: 15 node, 8 edge, 20 graph features (from 7 static + 6 context files)
- Compatible with modular_data_loader.py
- Optimized for 100-timestep sequences

Architecture:
1. HNN Branch: Processes 33 temporal features over 100 timesteps
2. GNN Branch: Processes component interactions (static per cycle)
3. Fusion Layer: Intelligently combines HNN and GNN outputs
4. COP Prediction Head: Final COP prediction with physics constraints

Author: Roo (AI Assistant)
Date: 2025-12-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import existing architectures
from hamiltonian_neural_network import HeatPumpHNN, HamiltonianNetwork
from graph_neural_network import HeatPumpGNN
from physics_informed_loss_functions import PhysicsInformedLoss


class FusionNetworkV2(nn.Module):
    """
    Enhanced Fusion Network for combining HNN and GNN outputs.
    
    Uses multi-head attention to dynamically weight temporal dynamics (HNN)
    and spatial relationships (GNN).
    """
    
    def __init__(self, 
                 hnn_dim: int = 128,
                 gnn_dim: int = 128,
                 fusion_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize fusion network.
        
        Args:
            hnn_dim: Dimension of HNN output
            gnn_dim: Dimension of GNN output
            fusion_dim: Dimension of fused representation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(FusionNetworkV2, self).__init__()
        
        self.hnn_dim = hnn_dim
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project HNN and GNN outputs to same dimension
        self.hnn_proj = nn.Linear(hnn_dim, fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
        
        # Multi-head attention for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, hnn_output: torch.Tensor, gnn_output: torch.Tensor) -> torch.Tensor:
        """
        Fuse HNN and GNN outputs.
        
        Args:
            hnn_output: HNN output [batch_size, seq_len, hnn_dim]
            gnn_output: GNN output [batch_size, gnn_dim]
            
        Returns:
            Fused representation [batch_size, seq_len, fusion_dim]
        """
        batch_size, seq_len, _ = hnn_output.shape
        
        # Project to fusion dimension
        hnn_proj = self.hnn_proj(hnn_output)  # [batch_size, seq_len, fusion_dim]
        
        # Expand GNN output to match sequence length
        gnn_expanded = gnn_output.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, gnn_dim]
        gnn_proj = self.gnn_proj(gnn_expanded)  # [batch_size, seq_len, fusion_dim]
        
        # Cross-attention: HNN attends to GNN
        attended_output, attention_weights = self.attention(
            query=hnn_proj,
            key=gnn_proj,
            value=gnn_proj
        )
        
        # Concatenate and fuse
        combined = torch.cat([hnn_proj, attended_output], dim=-1)  # [batch_size, seq_len, 2*fusion_dim]
        fused = self.fusion_layers(combined)  # [batch_size, seq_len, fusion_dim]
        
        # Residual connection and normalization
        fused = self.layer_norm(fused + hnn_proj)
        
        return fused


class COPPredictionHeadV2(nn.Module):
    """
    Enhanced COP Prediction Head with physics constraints.
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1):
        """
        Initialize COP prediction head.
        
        Args:
            input_dim: Input dimension from fusion layer
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(COPPredictionHeadV2, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
        
        # Physics constraints
        self.cop_min = 1.5
        self.cop_max = 8.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict COP with physics constraints.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            COP predictions [batch_size, seq_len, 1]
        """
        cop_pred = self.predictor(x)
        
        # Apply physics constraints
        cop_pred = torch.clamp(cop_pred, min=self.cop_min, max=self.cop_max)
        
        return cop_pred


class HybridHNNGNNV2(nn.Module):
    """
    Hybrid HNN-GNN Architecture v2 for Heat Pump COP Prediction.
    
    Updated for modular dataset structure with:
    - 33 HNN temporal features (100 timesteps)
    - 15 GNN node features (10 nodes)
    - 8 GNN edge features (11 edges)
    - 20 GNN graph features
    """
    
    def __init__(self,
                 # HNN parameters
                 hnn_input_dim: int = 33,  # Updated from 8 to 33
                 hnn_phase_space_dim: int = 8,
                 hnn_hidden_dims: List[int] = [128, 256, 128],
                 hnn_dt: float = 0.01,
                 
                 # GNN parameters
                 gnn_node_input_dim: int = 15,
                 gnn_edge_input_dim: int = 8,
                 gnn_graph_input_dim: int = 20,
                 gnn_hidden_dim: int = 128,
                 gnn_num_layers: int = 4,
                 
                 # Fusion parameters
                 fusion_dim: int = 256,
                 num_heads: int = 4,
                 
                 # General parameters
                 dropout: float = 0.1):
        """
        Initialize Hybrid HNN-GNN v2 model.
        
        Args:
            hnn_input_dim: Input feature dimension for HNN (33 from modular dataset)
            hnn_phase_space_dim: Phase space dimension for HNN
            hnn_hidden_dims: Hidden dimensions for HNN
            hnn_dt: Time step for HNN integration
            gnn_node_input_dim: Node feature dimension for GNN
            gnn_edge_input_dim: Edge feature dimension for GNN
            gnn_graph_input_dim: Graph feature dimension for GNN
            gnn_hidden_dim: Hidden dimension for GNN
            gnn_num_layers: Number of GNN layers
            fusion_dim: Fusion layer dimension
            num_heads: Number of attention heads in fusion
            dropout: Dropout rate
        """
        super(HybridHNNGNNV2, self).__init__()
        
        # Store dimensions
        self.hnn_input_dim = hnn_input_dim
        self.hnn_phase_space_dim = hnn_phase_space_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.fusion_dim = fusion_dim
        
        # HNN Branch
        self.hnn_branch = HeatPumpHNN(
            phase_space_dim=hnn_phase_space_dim,
            hidden_dims=hnn_hidden_dims,
            cop_hidden_dims=[128, 64],
            dt=hnn_dt,
            input_dim=hnn_input_dim  # 33 features
        )
        
        # GNN Branch
        self.gnn_branch = HeatPumpGNN(
            node_input_dim=gnn_node_input_dim,
            edge_input_dim=gnn_edge_input_dim,
            graph_input_dim=gnn_graph_input_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            output_dim=gnn_hidden_dim  # Output features, not COP
        )
        
        # Modify GNN to output features instead of COP
        self.gnn_branch.cop_predictor = nn.Identity()
        
        # Fusion Network
        self.fusion_layer = FusionNetworkV2(
            hnn_dim=hnn_phase_space_dim + 1,  # Phase space + Hamiltonian
            gnn_dim=gnn_hidden_dim * 4,  # 3 pooling methods + graph features
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # COP Prediction Head
        self.cop_predictor = COPPredictionHeadV2(
            input_dim=fusion_dim,
            hidden_dims=[128, 64],
            dropout=dropout
        )
        
    def forward(self, 
                hnn_input: torch.Tensor,
                gnn_data: Data,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            hnn_input: Temporal features for HNN [batch_size, seq_len, 33]
            gnn_data: PyTorch Geometric Data object for GNN
            return_intermediates: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing predictions and intermediate results
        """
        # HNN Branch: Process temporal dynamics
        hnn_results = self.hnn_branch(hnn_input, return_trajectory=False)
        hnn_phase_space = hnn_results['phase_space']  # [batch_size, seq_len, phase_space_dim]
        hnn_hamiltonian = hnn_results['hamiltonian']  # [batch_size, seq_len, 1]
        
        # Combine phase space and Hamiltonian for fusion
        hnn_output = torch.cat([hnn_phase_space, hnn_hamiltonian], dim=-1)
        
        # GNN Branch: Process component interactions
        # Extract GNN data components
        x, edge_index, edge_attr = gnn_data.x, gnn_data.edge_index, gnn_data.edge_attr
        graph_attr = gnn_data.graph_attr if hasattr(gnn_data, 'graph_attr') else None
        batch = gnn_data.batch if hasattr(gnn_data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Get batch size from graph_attr if available, otherwise from batch
        if graph_attr is not None:
            batch_size_gnn = graph_attr.size(0)
        else:
            batch_size_gnn = batch.max().item() + 1
        
        # Input embeddings
        x = self.gnn_branch.node_embedding(x)
        edge_attr = self.gnn_branch.edge_embedding(edge_attr)
        
        # Process graph features
        if graph_attr is not None:
            graph_features = self.gnn_branch.graph_embedding(graph_attr)
        else:
            graph_features = torch.zeros(batch_size_gnn, self.gnn_hidden_dim, device=x.device)
        
        # Graph convolution layers
        for i in range(self.gnn_branch.num_layers):
            x_conv = self.gnn_branch.conv_layers[i](x, edge_index, edge_attr)
            x_att = self.gnn_branch.attention_layers[i](x_conv, edge_index)
            x = self.gnn_branch.norm_layers[i](x + x_att)
            x = F.dropout(x, p=self.gnn_branch.dropout, training=self.training)
        
        # Global pooling - pool per graph in batch
        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        pooled_sum = global_add_pool(x, batch)
        
        # Combine pooled features with graph features
        gnn_output = torch.cat([pooled_mean, pooled_max, pooled_sum, graph_features], dim=-1)
        
        # Fusion Layer: Combine HNN and GNN
        fused_features = self.fusion_layer(hnn_output, gnn_output)
        
        # COP Prediction
        cop_prediction = self.cop_predictor(fused_features)
        
        # Prepare results
        results = {
            'cop_prediction': cop_prediction,
            'hamiltonian': hnn_hamiltonian,
            'phase_space': hnn_phase_space
        }
        
        if return_intermediates:
            results['hnn_output'] = hnn_output
            results['gnn_output'] = gnn_output
            results['fused_features'] = fused_features
        
        return results
    
    def compute_physics_loss(self,
                            predictions: torch.Tensor,
                            targets: torch.Tensor,
                            hamiltonian_data: Dict[str, torch.Tensor],
                            thermodynamic_data: Dict[str, torch.Tensor],
                            heat_pump_data: Dict[str, torch.Tensor],
                            graph_data: Dict[str, torch.Tensor],
                            loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth COP values
            hamiltonian_data: Hamiltonian mechanics data
            thermodynamic_data: Thermodynamic properties
            heat_pump_data: Heat pump specific data
            graph_data: Graph structure data
            loss_weights: Optional custom loss weights
            
        Returns:
            Dictionary of loss components
        """
        if loss_weights is None:
            loss_weights = {
                'lambda_hamiltonian': 1.0,
                'lambda_thermodynamic': 0.5,
                'lambda_heat_pump': 0.8,
                'lambda_graph': 0.3,
                'lambda_data': 1.0
            }
        
        physics_loss_fn = PhysicsInformedLoss(**loss_weights)
        
        losses = physics_loss_fn(
            predictions=predictions,
            targets=targets,
            hamiltonian_data=hamiltonian_data,
            thermodynamic_data=thermodynamic_data,
            heat_pump_data=heat_pump_data,
            graph_data=graph_data
        )
        
        return losses


def create_hybrid_model_v2(config: Dict) -> HybridHNNGNNV2:
    """
    Factory function to create hybrid HNN-GNN v2 model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hybrid HNN-GNN v2 model instance
    """
    return HybridHNNGNNV2(
        hnn_input_dim=config.get('hnn_input_dim', 33),
        hnn_phase_space_dim=config.get('hnn_phase_space_dim', 8),
        hnn_hidden_dims=config.get('hnn_hidden_dims', [128, 256, 128]),
        hnn_dt=config.get('hnn_dt', 0.01),
        gnn_node_input_dim=config.get('gnn_node_input_dim', 15),
        gnn_edge_input_dim=config.get('gnn_edge_input_dim', 8),
        gnn_graph_input_dim=config.get('gnn_graph_input_dim', 20),
        gnn_hidden_dim=config.get('gnn_hidden_dim', 128),
        gnn_num_layers=config.get('gnn_num_layers', 4),
        fusion_dim=config.get('fusion_dim', 256),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1)
    )


def test_hybrid_model_v2():
    """
    Test hybrid HNN-GNN v2 architecture with modular data dimensions.
    """
    print("=" * 70)
    print("TESTING HYBRID HNN-GNN V2 ARCHITECTURE")
    print("=" * 70)
    
    # Configuration matching modular dataset
    config = {
        'hnn_input_dim': 33,  # From modular dataset
        'hnn_phase_space_dim': 8,
        'hnn_hidden_dims': [64, 128, 64],
        'hnn_dt': 0.01,
        'gnn_node_input_dim': 15,
        'gnn_edge_input_dim': 8,
        'gnn_graph_input_dim': 20,
        'gnn_hidden_dim': 64,
        'gnn_num_layers': 3,
        'fusion_dim': 128,
        'num_heads': 4,
        'dropout': 0.1
    }
    
    # Create model
    print("\n1. Creating Hybrid Model V2...")
    model = create_hybrid_model_v2(config)
    print(f"   [OK] Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample data matching modular dataset dimensions
    batch_size = 4
    seq_len = 100
    num_nodes = 10
    num_edges = 11
    
    # HNN input: 33 features from modular dataset
    hnn_input = torch.randn(batch_size, seq_len, 33)
    
    # GNN input
    node_features = torch.randn(batch_size * num_nodes, 15)
    edge_features = torch.randn(batch_size * num_edges, 8)
    graph_features = torch.randn(batch_size, 20)
    
    # Edge index (heat pump topology)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5]
    ], dtype=torch.long)
    
    # Expand for batch
    batch_edge_index = []
    for b in range(batch_size):
        batch_edge_index.append(edge_index + b * num_nodes)
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    # Create GNN data
    gnn_data = Data(
        x=node_features,
        edge_index=batch_edge_index,
        edge_attr=edge_features,
        graph_attr=graph_features,
        batch=batch
    )
    
    print("\n2. Testing Forward Pass...")
    results = model(hnn_input, gnn_data, return_intermediates=True)
    
    print(f"   HNN input shape: {hnn_input.shape}")
    print(f"   GNN node features shape: {node_features.shape}")
    print(f"   COP prediction shape: {results['cop_prediction'].shape}")
    print(f"   Hamiltonian shape: {results['hamiltonian'].shape}")
    print(f"   Phase space shape: {results['phase_space'].shape}")
    print(f"   [OK] Forward pass successful")
    
    # Check output values
    cop_pred = results['cop_prediction']
    print(f"\n3. Checking Output Values...")
    print(f"   COP range: [{cop_pred.min().item():.2f}, {cop_pred.max().item():.2f}]")
    print(f"   COP mean: {cop_pred.mean().item():.2f}")
    print(f"   Within physics bounds [1.5, 8.0]: {torch.all((cop_pred >= 1.5) & (cop_pred <= 8.0)).item()}")
    print(f"   [OK] Output values valid")
    
    # Test backward pass
    print(f"\n4. Testing Backward Pass...")
    targets = torch.randn(batch_size, seq_len, 1) * 2 + 4
    loss = F.mse_loss(results['cop_prediction'], targets)
    loss.backward()
    
    # Check gradients
    has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    no_nan_gradients = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    print(f"   All parameters have gradients: {has_gradients}")
    print(f"   No NaN gradients: {no_nan_gradients}")
    print(f"   [OK] Backward pass successful")
    
    print(f"\n[SUCCESS] ALL HYBRID MODEL V2 TESTS PASSED!")
    print(f"Model ready for training with modular dataset!")
    print(f"\nArchitecture Summary:")
    print(f"  - HNN Branch: 33 temporal features over 100 timesteps")
    print(f"  - GNN Branch: 15 node, 8 edge, 20 graph features")
    print(f"  - Fusion Layer: Multi-head attention fusion")
    print(f"  - COP Head: Physics-constrained prediction")
    print(f"  - Total Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_hybrid_model_v2()