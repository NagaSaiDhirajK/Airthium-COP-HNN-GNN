#!/usr/bin/env python3
"""
Test Complete Pipeline: Data Loading + Model Forward Pass
==========================================================

Tests the full pipeline:
1. Load modular dataset
2. Create dataloaders
3. Load hybrid model v2
4. Run forward pass
5. Validate outputs
"""

import torch
import numpy as np
from pathlib import Path

# Import our modules
from modular_data_loader import create_modular_dataloaders
from hybrid_hnn_gnn_model_v2 import create_hybrid_model_v2


def test_complete_pipeline():
    """Test the complete training pipeline."""
    
    print("=" * 70)
    print("TESTING COMPLETE PIPELINE")
    print("=" * 70)
    
    # 1. Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    try:
        train_loader, val_loader = create_modular_dataloaders(
            data_dir="modular_dataset_500",
            batch_size=4,
            train_split=0.8,
            num_workers=0
        )
        print(f"   [OK] Train: {len(train_loader)} batches")
        print(f"   [OK] Val: {len(val_loader)} batches")
    except Exception as e:
        print(f"   [FAIL] Error creating dataloaders: {e}")
        return False
    
    # 2. Create model
    print("\n[2/4] Creating hybrid model v2...")
    try:
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
        
        model = create_hybrid_model_v2(config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   [OK] Model created: {n_params:,} parameters")
    except Exception as e:
        print(f"   [FAIL] Error creating model: {e}")
        return False
    
    # 3. Test forward pass
    print("\n[3/4] Testing forward pass...")
    try:
        # Get a batch
        batch = next(iter(train_loader))
        
        hnn_input = batch['hnn_input']
        gnn_data = batch['gnn_data']
        targets = batch['target']
        
        print(f"   Batch shapes:")
        print(f"     HNN input: {hnn_input.shape}")
        print(f"     GNN nodes: {gnn_data.x.shape}")
        print(f"     GNN edges: {gnn_data.edge_attr.shape}")
        print(f"     Targets: {targets.shape}")
        
        # Forward pass
        with torch.no_grad():
            results = model(hnn_input, gnn_data, return_intermediates=True)
        
        cop_pred = results['cop_prediction']
        hamiltonian = results['hamiltonian']
        phase_space = results['phase_space']
        
        print(f"\n   Output shapes:")
        print(f"     COP prediction: {cop_pred.shape}")
        print(f"     Hamiltonian: {hamiltonian.shape}")
        print(f"     Phase space: {phase_space.shape}")
        
        print(f"\n   Output values:")
        print(f"     COP range: [{cop_pred.min().item():.2f}, {cop_pred.max().item():.2f}]")
        print(f"     COP mean: {cop_pred.mean().item():.2f}")
        print(f"     Target COP: {targets.mean().item():.2f}")
        
        # Check for NaNs
        has_nans = (
            torch.isnan(cop_pred).any() or
            torch.isnan(hamiltonian).any() or
            torch.isnan(phase_space).any()
        )
        
        if has_nans:
            print(f"   [FAIL] Model outputs contain NaNs!")
            return False
        else:
            print(f"   [OK] No NaNs in outputs")
        
        # Check physics constraints
        in_range = torch.all((cop_pred >= 1.5) & (cop_pred <= 8.0))
        if in_range:
            print(f"   [OK] COP within physics bounds [1.5, 8.0]")
        else:
            print(f"   [WARN] Some COP values outside physics bounds")
        
    except Exception as e:
        print(f"   [FAIL] Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test backward pass
    print("\n[4/4] Testing backward pass...")
    try:
        # Forward pass with gradients
        results = model(hnn_input, gnn_data)
        cop_pred = results['cop_prediction']
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(cop_pred, targets.unsqueeze(1).expand_as(cop_pred))
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Parameters with gradients: {has_grads}/{total_params}")
        
        # Check for NaN gradients
        nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        
        if nan_grads > 0:
            print(f"   [FAIL] {nan_grads} parameters have NaN gradients!")
            return False
        else:
            print(f"   [OK] No NaN gradients")
        
    except Exception as e:
        print(f"   [FAIL] Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("[SUCCESS] COMPLETE PIPELINE TEST PASSED!")
    print("=" * 70)
    print("\nPipeline components working:")
    print("  [OK] Modular dataset loading (21 HDF5 files)")
    print("  [OK] Data batching (HNN + GNN)")
    print("  [OK] Hybrid model v2 (33 HNN features)")
    print("  [OK] Forward pass (COP prediction)")
    print("  [OK] Backward pass (gradient computation)")
    print("\nReady for training!")
    
    return True


if __name__ == "__main__":
    success = test_complete_pipeline()
    exit(0 if success else 1)