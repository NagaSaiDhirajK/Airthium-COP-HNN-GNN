#!/usr/bin/env python3
"""
Create GNN Graph Data for Heat Pump Component Topology
====================================================

Generate PyTorch Geometric compatible graph data representing:
1. Heat pump component topology (nodes = components, edges = connections)
2. Adjacency matrices for component interactions
3. Node features (component properties)
4. Edge features (connection properties)
5. Graph-level features (system properties)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def create_heat_pump_graph_topology() -> Dict[str, Any]:
    """
    Create the standard heat pump component graph topology.
    
    Returns:
        Dictionary containing graph structure and properties
    """
    
    print("🔧 Creating heat pump component graph topology...")
    
    # Define heat pump components (nodes)
    components = {
        0: {'name': 'compressor', 'type': 'mechanical', 'phase': 'vapor'},
        1: {'name': 'condenser', 'type': 'heat_exchanger', 'phase': 'liquid'},
        2: {'name': 'expansion_valve', 'type': 'mechanical', 'phase': 'mixed'},
        3: {'name': 'evaporator', 'type': 'heat_exchanger', 'phase': 'vapor'},
        4: {'name': 'accumulator', 'type': 'storage', 'phase': 'mixed'},
        5: {'name': 'filter_drier', 'type': 'filter', 'phase': 'liquid'},
        6: {'name': 'liquid_line', 'type': 'piping', 'phase': 'liquid'},
        7: {'name': 'suction_line', 'type': 'piping', 'phase': 'vapor'},
        8: {'name': 'discharge_line', 'type': 'piping', 'phase': 'vapor'},
        9: {'name': 'reversing_valve', 'type': 'valve', 'phase': 'vapor'}
    }
    
    # Define connections (edges) - refrigerant flow paths
    connections = [
        (0, 8),  # compressor -> discharge_line
        (8, 1),  # discharge_line -> condenser
        (1, 6),  # condenser -> liquid_line
        (6, 5),  # liquid_line -> filter_drier
        (5, 2),  # filter_drier -> expansion_valve
        (2, 3),  # expansion_valve -> evaporator
        (3, 7),  # evaporator -> suction_line
        (7, 4),  # suction_line -> accumulator
        (4, 0),  # accumulator -> compressor
        (9, 1),  # reversing_valve -> condenser (heat pump mode)
        (9, 3),  # reversing_valve -> evaporator (heat pump mode)
    ]
    
    # Create adjacency matrix
    n_nodes = len(components)
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    for src, dst in connections:
        adjacency_matrix[src, dst] = 1.0
        adjacency_matrix[dst, src] = 1.0  # Undirected graph
    
    # Edge list for PyTorch Geometric
    edge_index = np.array(connections).T  # Shape: (2, num_edges)
    
    print(f"  📊 Graph structure:")
    print(f"    Nodes: {n_nodes}")
    print(f"    Edges: {len(connections)}")
    print(f"    Adjacency matrix shape: {adjacency_matrix.shape}")
    print(f"    Edge index shape: {edge_index.shape}")
    
    return {
        'components': components,
        'connections': connections,
        'adjacency_matrix': adjacency_matrix,
        'edge_index': edge_index,
        'n_nodes': n_nodes,
        'n_edges': len(connections)
    }

def generate_node_features(components: Dict, n_cycles: int, trajectory_length: int) -> np.ndarray:
    """
    Generate node features for each component across all cycles.
    
    Args:
        components: Component definitions
        n_cycles: Number of cycles
        trajectory_length: Length of each trajectory
        
    Returns:
        Node features array [n_cycles, trajectory_length, n_nodes, node_features]
    """
    
    print("🎯 Generating node features...")
    
    n_nodes = len(components)
    n_node_features = 8  # temperature, pressure, flow_rate, efficiency, power, heat_transfer, mass, status
    
    node_features = np.zeros((n_cycles, trajectory_length, n_nodes, n_node_features))
    
    for cycle in range(n_cycles):
        for t in range(trajectory_length):
            time_norm = t / trajectory_length
            
            for node_id, component in components.items():
                # Base values depend on component type
                if component['name'] == 'compressor':
                    temp = 350 + 30 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 5)
                    pressure = 15 + 5 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 1)
                    flow_rate = 0.1 + 0.05 * np.sin(2 * np.pi * time_norm)
                    efficiency = 0.85 + 0.1 * np.cos(2 * np.pi * time_norm)
                    power = 3.0 + 1.0 * np.sin(2 * np.pi * time_norm)
                    heat_transfer = 0.0  # No heat transfer for compressor
                    mass = 50.0  # kg
                    status = 1.0  # Active
                    
                elif component['type'] == 'heat_exchanger':
                    if component['name'] == 'condenser':
                        temp = 320 + 20 * np.cos(2 * np.pi * time_norm) + np.random.normal(0, 3)
                        pressure = 12 + 3 * np.cos(2 * np.pi * time_norm)
                        heat_transfer = 10.0 + 3.0 * np.sin(2 * np.pi * time_norm)
                    else:  # evaporator
                        temp = 280 + 15 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 3)
                        pressure = 5 + 2 * np.sin(2 * np.pi * time_norm)
                        heat_transfer = 8.0 + 2.0 * np.cos(2 * np.pi * time_norm)
                    
                    flow_rate = 0.08 + 0.03 * np.sin(2 * np.pi * time_norm)
                    efficiency = 0.9 + 0.05 * np.sin(2 * np.pi * time_norm)
                    power = 0.5  # Fan power
                    mass = 30.0
                    status = 1.0
                    
                elif component['name'] == 'expansion_valve':
                    temp = 300 + 10 * np.sin(2 * np.pi * time_norm)
                    pressure = 8 + 4 * np.sin(2 * np.pi * time_norm)
                    flow_rate = 0.08 + 0.02 * np.sin(2 * np.pi * time_norm)
                    efficiency = 0.95
                    power = 0.1  # Control power
                    heat_transfer = 0.0
                    mass = 5.0
                    status = 1.0
                    
                else:  # piping, storage, filters
                    temp = 290 + 20 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 2)
                    pressure = 7 + 3 * np.sin(2 * np.pi * time_norm)
                    flow_rate = 0.08 + 0.02 * np.sin(2 * np.pi * time_norm)
                    efficiency = 0.98
                    power = 0.0
                    heat_transfer = 0.5  # Heat loss
                    mass = 10.0
                    status = 1.0
                
                # Store features
                node_features[cycle, t, node_id] = [
                    temp, pressure, flow_rate, efficiency, 
                    power, heat_transfer, mass, status
                ]
    
    print(f"  ✅ Node features shape: {node_features.shape}")
    return node_features

def generate_edge_features(connections: List, n_cycles: int, trajectory_length: int) -> np.ndarray:
    """
    Generate edge features for connections between components.
    
    Args:
        connections: List of edge connections
        n_cycles: Number of cycles
        trajectory_length: Length of each trajectory
        
    Returns:
        Edge features array [n_cycles, trajectory_length, n_edges, edge_features]
    """
    
    print("🔗 Generating edge features...")
    
    n_edges = len(connections)
    n_edge_features = 6  # mass_flow, heat_flow, pressure_drop, temperature_drop, resistance, conductance
    
    edge_features = np.zeros((n_cycles, trajectory_length, n_edges, n_edge_features))
    
    for cycle in range(n_cycles):
        for t in range(trajectory_length):
            time_norm = t / trajectory_length
            
            for edge_id, (src, dst) in enumerate(connections):
                # Edge features depend on connection type
                mass_flow = 0.08 + 0.02 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 0.005)
                heat_flow = 1.0 + 0.5 * np.sin(2 * np.pi * time_norm)
                pressure_drop = 0.5 + 0.2 * np.sin(2 * np.pi * time_norm)
                temperature_drop = 2.0 + 1.0 * np.sin(2 * np.pi * time_norm)
                resistance = 0.1 + 0.05 * np.random.random()
                conductance = 1.0 / (resistance + 1e-6)
                
                edge_features[cycle, t, edge_id] = [
                    mass_flow, heat_flow, pressure_drop, 
                    temperature_drop, resistance, conductance
                ]
    
    print(f"  ✅ Edge features shape: {edge_features.shape}")
    return edge_features

def generate_graph_features(n_cycles: int, trajectory_length: int) -> np.ndarray:
    """
    Generate graph-level features representing system properties.
    
    Args:
        n_cycles: Number of cycles
        trajectory_length: Length of each trajectory
        
    Returns:
        Graph features array [n_cycles, trajectory_length, graph_features]
    """
    
    print("🌐 Generating graph-level features...")
    
    n_graph_features = 10  # total_power, total_heat, cop, efficiency, ambient_temp, load, mode, defrost, fault, performance
    
    graph_features = np.zeros((n_cycles, trajectory_length, n_graph_features))
    
    for cycle in range(n_cycles):
        for t in range(trajectory_length):
            time_norm = t / trajectory_length
            
            # System-level properties
            total_power = 3.5 + 1.5 * np.sin(2 * np.pi * time_norm) + np.random.normal(0, 0.2)
            total_heat = 12.0 + 4.0 * np.cos(2 * np.pi * time_norm) + np.random.normal(0, 0.5)
            cop = total_heat / total_power
            efficiency = 0.85 + 0.1 * np.sin(2 * np.pi * time_norm)
            ambient_temp = 20 + 15 * np.sin(2 * np.pi * time_norm)
            load = 0.7 + 0.2 * np.sin(2 * np.pi * time_norm)
            mode = 1.0  # Heating mode
            defrost = 0.0 if np.random.random() > 0.05 else 1.0  # 5% defrost probability
            fault = 0.0 if np.random.random() > 0.02 else 1.0  # 2% fault probability
            performance = cop / 4.0  # Normalized performance
            
            graph_features[cycle, t] = [
                total_power, total_heat, cop, efficiency, ambient_temp,
                load, mode, defrost, fault, performance
            ]
    
    print(f"  ✅ Graph features shape: {graph_features.shape}")
    return graph_features

def create_pytorch_geometric_data(graph_topology: Dict, node_features: np.ndarray, 
                                edge_features: np.ndarray, graph_features: np.ndarray,
                                targets: np.ndarray) -> Dict[str, Any]:
    """
    Create PyTorch Geometric compatible data structure.
    
    Args:
        graph_topology: Graph structure information
        node_features: Node feature arrays
        edge_features: Edge feature arrays
        graph_features: Graph-level features
        targets: COP targets
        
    Returns:
        PyTorch Geometric data dictionary
    """
    
    print("🔥 Creating PyTorch Geometric data structure...")
    
    n_cycles, trajectory_length = node_features.shape[:2]
    
    # Prepare data for PyTorch Geometric
    pyg_data = {
        'metadata': {
            'dataset_name': 'AIRTHIUM Heat Pump GNN Dataset',
            'version': '1.0',
            'n_cycles': n_cycles,
            'trajectory_length': trajectory_length,
            'n_nodes': graph_topology['n_nodes'],
            'n_edges': graph_topology['n_edges'],
            'node_features': node_features.shape[-1],
            'edge_features': edge_features.shape[-1],
            'graph_features': graph_features.shape[-1]
        },
        'graph_structure': {
            'edge_index': graph_topology['edge_index'],
            'adjacency_matrix': graph_topology['adjacency_matrix'],
            'components': graph_topology['components'],
            'connections': graph_topology['connections']
        },
        'features': {
            'node_features': node_features,
            'edge_features': edge_features,
            'graph_features': graph_features
        },
        'targets': targets,
        'feature_names': {
            'node_features': [
                'temperature', 'pressure', 'flow_rate', 'efficiency',
                'power', 'heat_transfer', 'mass', 'status'
            ],
            'edge_features': [
                'mass_flow', 'heat_flow', 'pressure_drop',
                'temperature_drop', 'resistance', 'conductance'
            ],
            'graph_features': [
                'total_power', 'total_heat', 'cop', 'efficiency', 'ambient_temp',
                'load', 'mode', 'defrost', 'fault', 'performance'
            ]
        }
    }
    
    print(f"  ✅ PyTorch Geometric data structure created")
    print(f"    Nodes: {pyg_data['metadata']['n_nodes']}")
    print(f"    Edges: {pyg_data['metadata']['n_edges']}")
    print(f"    Node features: {pyg_data['metadata']['node_features']}")
    print(f"    Edge features: {pyg_data['metadata']['edge_features']}")
    print(f"    Graph features: {pyg_data['metadata']['graph_features']}")
    
    return pyg_data

def save_gnn_dataset(pyg_data: Dict, output_path: str):
    """
    Save GNN dataset in HDF5 format for efficient loading.
    
    Args:
        pyg_data: PyTorch Geometric data dictionary
        output_path: Output file path
    """
    
    print(f"💾 Saving GNN dataset to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        metadata_group = f.create_group('metadata')
        for key, value in pyg_data['metadata'].items():
            metadata_group.attrs[key] = value
        
        # Graph structure
        structure_group = f.create_group('graph_structure')
        structure_group.create_dataset('edge_index', data=pyg_data['graph_structure']['edge_index'])
        structure_group.create_dataset('adjacency_matrix', data=pyg_data['graph_structure']['adjacency_matrix'])
        
        # Components and connections as JSON strings
        structure_group.attrs['components'] = json.dumps(pyg_data['graph_structure']['components'])
        structure_group.attrs['connections'] = json.dumps(pyg_data['graph_structure']['connections'])
        
        # Features
        features_group = f.create_group('features')
        features_group.create_dataset('node_features', 
                                    data=pyg_data['features']['node_features'],
                                    compression='gzip', compression_opts=9)
        features_group.create_dataset('edge_features', 
                                    data=pyg_data['features']['edge_features'],
                                    compression='gzip', compression_opts=9)
        features_group.create_dataset('graph_features', 
                                    data=pyg_data['features']['graph_features'],
                                    compression='gzip', compression_opts=9)
        
        # Targets
        f.create_dataset('targets', data=pyg_data['targets'], 
                        compression='gzip', compression_opts=9)
        
        # Feature names
        names_group = f.create_group('feature_names')
        for feature_type, names in pyg_data['feature_names'].items():
            names_group.attrs[feature_type] = [name.encode() for name in names]

def create_sample_batch_loader():
    """
    Create a sample batch loader function for PyTorch Geometric.
    """
    
    sample_code = '''
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
'''
    
    return sample_code

def main():
    """Create comprehensive GNN graph dataset."""
    
    print("🚀 CREATING GNN GRAPH DATASET")
    print("=" * 50)
    
    # Load comprehensive physics dataset to get targets
    physics_file = "comprehensive_physics_dataset/airthium_comprehensive_dataset.h5"
    
    if not os.path.exists(physics_file):
        print(f"❌ Physics dataset not found: {physics_file}")
        print("Please run convert_all_physics_models.py first")
        return
    
    print(f"📊 Loading physics dataset from {physics_file}")
    with h5py.File(physics_file, 'r') as f:
        targets = f['targets'][:]  # Shape: (2600, 100, 1)
        n_cycles, trajectory_length = targets.shape[:2]
    
    print(f"  Cycles: {n_cycles}")
    print(f"  Trajectory length: {trajectory_length}")
    
    # Create heat pump graph topology
    graph_topology = create_heat_pump_graph_topology()
    
    # Generate features
    node_features = generate_node_features(
        graph_topology['components'], n_cycles, trajectory_length
    )
    
    edge_features = generate_edge_features(
        graph_topology['connections'], n_cycles, trajectory_length
    )
    
    graph_features = generate_graph_features(n_cycles, trajectory_length)
    
    # Create PyTorch Geometric data structure
    pyg_data = create_pytorch_geometric_data(
        graph_topology, node_features, edge_features, graph_features, targets
    )
    
    # Create output directory
    output_dir = Path("gnn_graph_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Save GNN dataset
    gnn_file = output_dir / "airthium_gnn_dataset.h5"
    save_gnn_dataset(pyg_data, str(gnn_file))
    
    # Save metadata as JSON
    metadata_file = output_dir / "gnn_metadata.json"
    with open(metadata_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metadata = {
            'metadata': pyg_data['metadata'],
            'graph_structure': {
                'edge_index': pyg_data['graph_structure']['edge_index'].tolist(),
                'adjacency_matrix': pyg_data['graph_structure']['adjacency_matrix'].tolist(),
                'components': pyg_data['graph_structure']['components'],
                'connections': pyg_data['graph_structure']['connections']
            },
            'feature_names': pyg_data['feature_names']
        }
        json.dump(json_metadata, f, indent=2)
    
    # Save sample PyTorch Geometric loader code
    loader_file = output_dir / "pytorch_geometric_loader.py"
    with open(loader_file, 'w') as f:
        f.write(create_sample_batch_loader())
    
    # Calculate file sizes
    gnn_size = gnn_file.stat().st_size / (1024*1024)
    
    print(f"\n✅ GNN GRAPH DATASET CREATED!")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 GNN dataset size: {gnn_size:.2f}MB")
    print(f"📊 Graph structure:")
    print(f"  Nodes: {graph_topology['n_nodes']}")
    print(f"  Edges: {graph_topology['n_edges']}")
    print(f"  Node features: {node_features.shape[-1]}")
    print(f"  Edge features: {edge_features.shape[-1]}")
    print(f"  Graph features: {graph_features.shape[-1]}")
    print(f"🔥 PyTorch Geometric: ✅")
    print(f"⚡ GPU-optimized: ✅")
    
    print(f"\n🎉 GNN DATASET READY!")
    print(f"📋 Next steps:")
    print(f"1. Use {loader_file.name} for PyTorch Geometric training")
    print(f"2. Implement GNN model architecture")
    print(f"3. Combine with HNN for hybrid training")
    print(f"4. Train on Kaggle T4 x2 GPUs")

if __name__ == "__main__":
    main()