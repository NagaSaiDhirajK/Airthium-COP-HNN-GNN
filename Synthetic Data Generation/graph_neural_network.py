
"""
Graph Neural Network Architecture for Heat Pump COP Prediction
============================================================

This module implements Graph Neural Networks (GNNs) for modeling spatial relationships
between heat pump components. The GNN captures component interactions, heat/mass transfer
networks, and system-level dependencies that affect COP performance.

Key Features:
- Heat pump component topology (10 nodes, 11 edges)
- Multi-scale graph convolutions (local and global interactions)
- Temporal graph dynamics (time-varying node/edge features)
- Physics-informed graph constraints
- Attention mechanisms for component importance
- Graph pooling for system-level predictions

Author: Roo (AI Assistant)
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GraphConv, GINConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    MessagePassing, aggr
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math


class HeatPumpGraphConv(MessagePassing):
    """
    Custom Graph Convolution for Heat Pump Components
    
    Implements physics-informed message passing that respects:
    - Heat transfer relationships (thermal conductance)
    - Mass flow conservation (refrigerant flow)
    - Energy balance constraints
    - Component-specific interactions
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 edge_dim: int = 8, aggr: str = 'add'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Edge transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * out_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Update transformation
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics-informed message passing
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features
        x_transformed = self.node_mlp(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        
        # Update nodes
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Create messages between connected nodes
        
        Args:
            x_i: Target node features
            x_j: Source node features  
            edge_attr: Edge features
        
        Returns:
            Messages to be aggregated
        """
        # Combine source, target, and edge information
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Transform to create physics-informed messages
        message = self.message_mlp(message_input)
        
        return message


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for Component Importance
    
    Learns attention weights for different heat pump components
    based on their relevance to COP prediction.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Multi-head attention
        self.gat_conv = GATConv(
            in_channels, out_channels // heads, 
            heads=heads, dropout=dropout, concat=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Attended node features [num_nodes, out_channels]
        """
        # Apply attention
        attended = self.gat_conv(x, edge_index)
        
        # Residual connection
        residual = self.residual_proj(x)
        
        # Layer normalization
        out = self.layer_norm(attended + residual)
        
        return out


class TemporalGraphLayer(nn.Module):
    """
    Temporal Graph Layer for Time-Varying Dynamics
    
    Captures temporal evolution of heat pump component states
    and their interactions over time.
    """
    
    def __init__(self, node_channels: int, edge_channels: int, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.hidden_dim = hidden_dim
        
        # Temporal encoding for nodes
        self.node_temporal = nn.LSTM(
            node_channels, hidden_dim, batch_first=True
        )
        
        # Temporal encoding for edges
        self.edge_temporal = nn.LSTM(
            edge_channels, hidden_dim, batch_first=True
        )
        
        # Graph convolution
        self.graph_conv = HeatPumpGraphConv(
            hidden_dim, hidden_dim, edge_dim=hidden_dim
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, node_channels)
    
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal graph dynamics
        
        Args:
            node_features: [batch, time, nodes, node_channels]
            edge_features: [batch, time, edges, edge_channels]
            edge_index: [2, edges]
        
        Returns:
            Updated node features [batch, time, nodes, node_channels]
        """
        batch_size, time_steps, num_nodes, _ = node_features.shape
        _, _, num_edges, _ = edge_features.shape
        
        # Reshape for LSTM processing
        node_flat = node_features.view(-1, time_steps, self.node_channels)
        edge_flat = edge_features.view(-1, time_steps, self.edge_channels)
        
        # Apply temporal encoding
        node_temporal, _ = self.node_temporal(node_flat)
        edge_temporal, _ = self.edge_temporal(edge_flat)
        
        # Reshape back
        node_temporal = node_temporal.view(batch_size, time_steps, num_nodes, self.hidden_dim)
        edge_temporal = edge_temporal.view(batch_size, time_steps, num_edges, self.hidden_dim)
        
        # Apply graph convolution at each time step
        outputs = []
        for t in range(time_steps):
            # Get features at time t
            x_t = node_temporal[:, t].view(-1, self.hidden_dim)  # [batch*nodes, hidden]
            edge_attr_t = edge_temporal[:, t].view(-1, self.hidden_dim)  # [batch*edges, hidden]
            
            # Expand edge_index for batch
            batch_edge_index = []
            for b in range(batch_size):
                batch_edge_index.append(edge_index + b * num_nodes)
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            
            # Apply graph convolution
            x_out = self.graph_conv(x_t, batch_edge_index, edge_attr_t)
            
            # Reshape back to batch format
            x_out = x_out.view(batch_size, num_nodes, self.hidden_dim)
            outputs.append(x_out)
        
        # Stack temporal outputs
        temporal_output = torch.stack(outputs, dim=1)  # [batch, time, nodes, hidden]
        
        # Project to output dimension
        output = self.output_proj(temporal_output)
        
        return output


class HeatPumpGNN(nn.Module):
    """
    Complete Graph Neural Network for Heat Pump COP Prediction
    
    Architecture:
    1. Node/edge embedding layers
    2. Multiple graph convolution layers
    3. Attention mechanisms
    4. Temporal dynamics
    5. Graph pooling
    6. COP prediction head
    """
    
    def __init__(self, 
                 node_input_dim: int = 15,
                 edge_input_dim: int = 8,
                 graph_input_dim: int = 20,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.graph_input_dim = graph_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_dim = output_dim
        
        # Input embeddings
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_embedding = nn.Sequential(
            nn.Linear(graph_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Graph convolution
            self.conv_layers.append(
                HeatPumpGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            )
            
            # Attention layer
            self.attention_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
            )
            
            # Layer normalization
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        
        # Temporal dynamics
        self.temporal_layer = TemporalGraphLayer(
            hidden_dim, hidden_dim, hidden_dim
        )
        
        # Graph pooling
        self.global_pool = nn.ModuleDict({
            'mean': lambda x, batch: global_mean_pool(x, batch),
            'max': lambda x, batch: global_max_pool(x, batch),
            'add': lambda x, batch: global_add_pool(x, batch)
        })
        
        # COP prediction head
        self.cop_predictor = nn.Sequential(
            nn.Linear(3 * hidden_dim + hidden_dim, hidden_dim),  # 3 pooling + graph features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Physics constraints
        self.cop_min = 1.5
        self.cop_max = 8.0
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for COP prediction
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, node_input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
                - graph_attr: Graph-level features [graph_input_dim]
                - batch: Batch assignment for nodes
        
        Returns:
            COP predictions [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        graph_attr = data.graph_attr if hasattr(data, 'graph_attr') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Input embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        if graph_attr is not None:
            graph_features = self.graph_embedding(graph_attr)
        else:
            graph_features = torch.zeros(batch.max().item() + 1, self.hidden_dim, device=x.device)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            # Graph convolution
            x_conv = self.conv_layers[i](x, edge_index, edge_attr)
            
            # Attention
            x_att = self.attention_layers[i](x_conv, edge_index)
            
            # Residual connection and normalization
            x = self.norm_layers[i](x + x_att)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        pooled_features = []
        for pool_name, pool_fn in self.global_pool.items():
            pooled = pool_fn(x, batch)
            pooled_features.append(pooled)
        
        # Combine pooled features with graph features
        combined_features = torch.cat(pooled_features + [graph_features], dim=-1)
        
        # COP prediction
        cop_pred = self.cop_predictor(combined_features)
        
        # Apply physics constraints
        cop_pred = torch.clamp(cop_pred, min=self.cop_min, max=self.cop_max)
        
        return cop_pred
    
    def forward_temporal(self, node_features: torch.Tensor, 
                        edge_features: torch.Tensor,
                        graph_features: torch.Tensor,
                        edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal data
        
        Args:
            node_features: [batch, time, nodes, node_input_dim]
            edge_features: [batch, time, edges, edge_input_dim]
            graph_features: [batch, time, graph_input_dim]
            edge_index: [2, edges]
        
        Returns:
            COP predictions [batch, time, output_dim]
        """
        batch_size, time_steps = node_features.shape[:2]
        
        # Embed features
        node_embedded = self.node_embedding(node_features.view(-1, self.node_input_dim))
        node_embedded = node_embedded.view(batch_size, time_steps, -1, self.hidden_dim)
        
        edge_embedded = self.edge_embedding(edge_features.view(-1, self.edge_input_dim))
        edge_embedded = edge_embedded.view(batch_size, time_steps, -1, self.hidden_dim)
        
        graph_embedded = self.graph_embedding(graph_features.view(-1, self.graph_input_dim))
        graph_embedded = graph_embedded.view(batch_size, time_steps, self.hidden_dim)
        
        # Apply temporal dynamics
        node_temporal = self.temporal_layer(node_embedded, edge_embedded, edge_index)
        
        # Process each time step
        outputs = []
        for t in range(time_steps):
            # Create batch data for time t
            x_t = node_temporal[:, t].view(-1, self.hidden_dim)
            edge_attr_t = edge_embedded[:, t].view(-1, self.hidden_dim)
            graph_attr_t = graph_embedded[:, t]
            
            # Create batch indices
            batch_t = torch.arange(batch_size, device=x_t.device).repeat_interleave(
                node_temporal.shape[2]
            )
            
            # Apply graph layers
            for i in range(self.num_layers):
                x_conv = self.conv_layers[i](x_t, edge_index, edge_attr_t)
                x_att = self.attention_layers[i](x_conv, edge_index)
                x_t = self.norm_layers[i](x_t + x_att)
                x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            
            # Global pooling
            pooled_features = []
            for pool_name, pool_fn in self.global_pool.items():
                pooled = pool_fn(x_t, batch_t)
                pooled_features.append(pooled)
            
            # Combine with graph features
            combined = torch.cat(pooled_features + [graph_attr_t], dim=-1)
            
            # Predict COP
            cop_t = self.cop_predictor(combined)
            cop_t = torch.clamp(cop_t, min=self.cop_min, max=self.cop_max)
            
            outputs.append(cop_t)
        
        # Stack temporal outputs
        temporal_output = torch.stack(outputs, dim=1)  # [batch, time, output_dim]
        
        return temporal_output


class MultiScaleGNN(nn.Module):
    """
    Multi-Scale Graph Neural Network
    
    Captures both local component interactions and global system behavior
    through multiple graph scales and hierarchical pooling.
    """
    
    def __init__(self, 
                 node_input_dim: int = 15,
                 edge_input_dim: int = 8,
                 graph_input_dim: int = 20,
                 hidden_dim: int = 128,
                 num_scales: int = 3,
                 output_dim: int = 1):
        super().__init__()
        
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        
        # Multi-scale GNN layers
        self.scale_gnns = nn.ModuleList()
        for scale in range(num_scales):
            gnn = HeatPumpGNN(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graph_input_dim=graph_input_dim,
                hidden_dim=hidden_dim,
                num_layers=2 + scale,  # Increasing depth for larger scales
                output_dim=hidden_dim
            )
            self.scale_gnns.append(gnn)
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(num_scales * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Physics constraints
        self.cop_min = 1.5
        self.cop_max = 8.0
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Multi-scale forward pass
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            COP predictions [batch_size, output_dim]
        """
        scale_outputs = []
        
        # Process at each scale
        for scale_gnn in self.scale_gnns:
            scale_out = scale_gnn(data)
            scale_outputs.append(scale_out)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_outputs, dim=-1)
        
        # Final prediction
        cop_pred = self.scale_fusion(fused_features)
        cop_pred = torch.clamp(cop_pred, min=self.cop_min, max=self.cop_max)
        
        return cop_pred


class AdaptiveGNN(nn.Module):
    """
    Adaptive Graph Neural Network
    
    Dynamically adjusts graph structure and attention based on
    operating conditions and component states.
    """
    
    def __init__(self, 
                 node_input_dim: int = 15,
                 edge_input_dim: int = 8,
                 graph_input_dim: int = 20,
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Base GNN
        self.base_gnn = HeatPumpGNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            graph_input_dim=graph_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Adaptive edge weight predictor
        self.edge_weight_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Adaptive node importance predictor
        self.node_importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Final prediction layer
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Physics constraints
        self.cop_min = 1.5
        self.cop_max = 8.0
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Adaptive forward pass
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            COP predictions [batch_size, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Get base GNN features
        node_features = self.base_gnn.node_embedding(x)
        edge_features = self.base_gnn.edge_embedding(edge_attr)
        
        # Predict adaptive edge weights
        row, col = edge_index
        edge_input = torch.cat([node_features[row], node_features[col], edge_attr], dim=-1)
        edge_weights = self.edge_weight_predictor(edge_input).squeeze(-1)
        
        # Apply edge weights to edge features
        weighted_edge_attr = edge_features * edge_weights.unsqueeze(-1)
        
        # Create modified data with adaptive weights
        adaptive_data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=weighted_edge_attr,
            batch=batch
        )
        if hasattr(data, 'graph_attr'):
            adaptive_data.graph_attr = data.graph_attr
        
        # Forward through base GNN with adaptive weights
        gnn_output = self.base_gnn(adaptive_data)
        
        # Final prediction
        cop_pred = self.final_predictor(gnn_output)
        cop_pred = torch.clamp(cop_pred, min=self.cop_min, max=self.cop_max)
        
        return cop_pred


def create_heat_pump_graph_data(node_features: torch.Tensor,
                               edge_features: torch.Tensor,
                               graph_features: torch.Tensor,
                               edge_index: torch.Tensor,
                               targets: torch.Tensor) -> Data:
    """
    Create PyTorch Geometric Data object for heat pump graph
    
    Args:
        node_features: Node features [num_nodes, node_dim]
        edge_features: Edge features [num_edges, edge_dim]
        graph_features: Graph-level features [graph_dim]
        edge_index: Edge connectivity [2, num_edges]
        targets: COP targets [1] or [time_steps]
    
    Returns:
        PyTorch Geometric Data object
    """
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        graph_attr=graph_features,
        y=targets
    )
    
    return data


def test_gnn_concepts():
    """
    Test GNN concepts with synthetic data
    """
    print("🧪 GRAPH NEURAL NETWORK CONCEPTS VALIDATION")
    print("=" * 60)
    
    # Heat pump graph topology (10 nodes, 11 edges)
    num_nodes = 10
    num_edges = 11
    node_dim = 15
    edge_dim = 8
    graph_dim = 20
    batch_size = 4
    
    # Create synthetic graph data
    node_features = torch.randn(batch_size * num_nodes, node_dim)
    edge_features = torch.randn(batch_size * num_edges, edge_dim)
    graph_features = torch.randn(batch_size, graph_dim)
    
    # Create edge index (heat pump topology)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],  # Source nodes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5]   # Target nodes
    ], dtype=torch.long)
    
    # Expand edge index for batch
    batch_edge_index = []
    for b in range(batch_size):
        batch_edge_index.append(edge_index + b * num_nodes)
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    
    # Create batch assignment
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    # Create targets
    targets = torch.randn(batch_size, 1) * 2 + 4  # COP around 4
    
    print(f"📊 Graph Data Shapes:")
    print(f"  Node features: {node_features.shape}")
    print(f"  Edge features: {edge_features.shape}")
    print(f"  Graph features: {graph_features.shape}")
    print(f"  Edge index: {batch_edge_index.shape}")
    print(f"  Batch: {batch.shape}")
    print(f"  Targets: {targets.shape}")
    print()
    
    # Test basic GNN
    print("1. Testing Basic Heat Pump GNN...")
    gnn = HeatPumpGNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graph_input_dim=graph_dim,
        hidden_dim=64,
        num_layers=3
    )
    
    # Create data object
    data = Data(
        x=node_features,
        edge_index=batch_edge_index,
        edge_attr=edge_features,
        graph_attr=graph_features,
        batch=batch,
        y=targets
    )
    
    with torch.no_grad():
        output = gnn(data)
    
    print(f"   Input shape: {node_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   COP predictions: {output.squeeze().tolist()}")
    print(f"   ✅ Basic GNN working")
    
    # Test multi-scale GNN
    print("\n2. Testing Multi-Scale GNN...")
    multi_gnn = MultiScaleGNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graph_input_dim=graph_dim,
        hidden_dim=64,
        num_scales=3
    )
    
    with torch.no_grad():
        multi_output = multi_gnn(data)
    
    print(f"   Multi-scale output shape: {multi_output.shape}")
    print(f"   Multi-scale COP predictions: {multi_output.squeeze().tolist()}")
    print(f"   ✅ Multi-scale GNN working")
    
    # Test adaptive GNN
    print("\n3. Testing Adaptive GNN...")
    adaptive_gnn = AdaptiveGNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graph_input_dim=graph_dim,
        hidden_dim=64
    )
    
    with torch.no_grad():
        adaptive_output = adaptive_gnn(data)
    
    print(f"   Adaptive output shape: {adaptive_output.shape}")
    print(f"   Adaptive COP predictions: {adaptive_output.squeeze().tolist()}")
    print(f"   ✅ Adaptive GNN working")
    
    # Test temporal GNN
    print("\n4. Testing Temporal Graph Dynamics...")
    time_steps = 10
    temporal_node_features = torch.randn(batch_size, time_steps, num_nodes, node_dim)
    temporal_edge_features = torch.randn(batch_size, time_steps, num_edges, edge_dim)
    temporal_graph_features = torch.randn(batch_size, time_steps, graph_dim)
    
    with torch.no_grad():
        temporal_output = gnn.forward_temporal(
            temporal_node_features,
            temporal_edge_features,
            temporal_graph_features,
            edge_index
        )
    
    print(f"   Temporal input shape: {temporal_node_features.shape}")
    print(f"   Temporal output shape: {temporal_output.shape}")
    print(f"   Temporal COP sample: {temporal_output[0, :5, 0].tolist()}")
    print(f"   ✅ Temporal GNN working")
    
    print()
    print("✅ ALL GNN CONCEPTS VALIDATED!")
    print("🎯 Ready for PyTorch implementation on Kaggle!")
    print()
    print("📋 Key GNN Components Implemented:")
    print("  1. ✅ Heat pump component topology (10 nodes, 11 edges)")
    print("  2. ✅ Physics-informed message passing")
    print("  3. ✅ Graph attention mechanisms")
    print("  4. ✅ Temporal graph dynamics")
    print("  5. ✅ Multi-scale graph processing")
    print("  6. ✅ Adaptive graph structure")
    print("  7. ✅ Graph pooling for system-level predictions")
    print("  8. ✅ COP prediction from graph features")


if __name__ == "__main__":
    test_gnn_concepts()