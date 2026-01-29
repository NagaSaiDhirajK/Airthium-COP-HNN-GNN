#!/usr/bin/env python3
"""Analyze the generated dataset for feature counts and variation."""

import h5py
import numpy as np
from pathlib import Path

def analyze_dataset(h5_path: str):
    """Analyze the HDF5 dataset."""
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        print(f"Dataset: {h5_path}")
        print(f"Total cycles: {f.attrs['n_cycles']}")
        print(f"Timesteps: {f.attrs['n_timesteps']}")
        print()
        
        # Analyze each group
        for group_name in f.keys():
            print(f"--- {group_name.upper()} ---")
            group = f[group_name]
            for key in group.keys():
                data = group[key][:]
                shape = data.shape
                std = np.std(data)
                status = "VARYING" if std > 1e-8 else "CONSTANT"
                if len(shape) == 1:
                    print(f"  {key}: shape={shape}, range=[{np.min(data):.3f}, {np.max(data):.3f}], std={std:.4f} [{status}]")
                else:
                    print(f"  {key}: shape={shape}, std={std:.4f} [{status}]")
            print()
        
        # Feature summary
        print("=" * 70)
        print("FEATURE SUMMARY FOR MODEL")
        print("=" * 70)
        
        # HNN features
        hnn = f['temporal_hnn']
        print("\nHNN Temporal Features (100 timesteps x channels):")
        total_hnn = 0
        for key in sorted(hnn.keys()):
            shape = hnn[key].shape
            if len(shape) == 2:  # (samples, timesteps)
                total_hnn += 1
                print(f"  {key}: {shape} -> 1 channel")
            elif len(shape) == 3:  # (samples, timesteps, features)
                total_hnn += shape[2]
                print(f"  {key}: {shape} -> {shape[2]} channels")
        
        print(f"\nTotal HNN channels: {total_hnn}")
        print(f"HNN input shape: (batch, 100 timesteps, {total_hnn} channels)")
        
        # GNN features
        gnn = f['gnn']
        node_shape = gnn['node_features'].shape
        edge_shape = gnn['edge_features'].shape
        graph_shape = gnn['graph_features'].shape
        
        print("\nGNN Features:")
        print(f"  Node features: {node_shape} = {node_shape[1]} nodes x {node_shape[2]} features")
        print(f"  Edge features: {edge_shape} = {edge_shape[1]} edges x {edge_shape[2]} features")
        print(f"  Graph features: {graph_shape} = {graph_shape[1]} features")
        
        # Check GNN variation
        print("\n  Node feature variation (first 10 features):")
        node_data = gnn['node_features'][:]
        for i in range(min(10, node_shape[2])):
            col = node_data[:, :, i].flatten()
            print(f"    Feature {i}: std={np.std(col):.6f}, range=[{np.min(col):.4f}, {np.max(col):.4f}]")
        
        print("\n  Graph feature variation (first 10 features):")
        graph_data = gnn['graph_features'][:]
        for i in range(min(10, graph_shape[1])):
            col = graph_data[:, i]
            status = "VARYING" if np.std(col) > 1e-8 else "CONSTANT"
            print(f"    Feature {i}: std={np.std(col):.6f}, range=[{np.min(col):.4f}, {np.max(col):.4f}] [{status}]")
        
        # Target
        cop = f['targets/COP_heating'][:]
        print("\n" + "=" * 70)
        print("TARGET (COP_heating):")
        print("=" * 70)
        print(f"  Min: {np.min(cop):.3f}")
        print(f"  Max: {np.max(cop):.3f}")
        print(f"  Mean: {np.mean(cop):.3f}")
        print(f"  Std: {np.std(cop):.3f}")
        print(f"  Median: {np.median(cop):.3f}")
        
        # Histogram
        hist, edges = np.histogram(cop, bins=10)
        print("\n  Histogram:")
        for i in range(len(hist)):
            bar = "#" * int(hist[i] / max(hist) * 30)
            print(f"    {edges[i]:.1f}-{edges[i+1]:.1f}: {bar} ({hist[i]})")
        
        # Total feature count
        total_input = 100 * total_hnn + node_shape[1] * node_shape[2] + edge_shape[1] * edge_shape[2] + graph_shape[1]
        
        print("\n" + "=" * 70)
        print("TOTAL INPUT FEATURES")
        print("=" * 70)
        print(f"  HNN temporal: 100 x {total_hnn} = {100 * total_hnn}")
        print(f"  GNN nodes: {node_shape[1]} x {node_shape[2]} = {node_shape[1] * node_shape[2]}")
        print(f"  GNN edges: {edge_shape[1]} x {edge_shape[2]} = {edge_shape[1] * edge_shape[2]}")
        print(f"  GNN graph: {graph_shape[1]}")
        print(f"")
        print(f"  TOTAL: {total_input} features per cycle")
        print(f"  TARGET: 1 (COP_heating)")
        print("=" * 70)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "filtered_dataset_500/physics_data.h5"
    analyze_dataset(path)
