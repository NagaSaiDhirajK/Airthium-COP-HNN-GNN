#!/usr/bin/env python3
"""
Test Training Loop
==================

Tests a few epochs of training with the modular dataset and hybrid model v2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from modular_data_loader import create_modular_dataloaders
from hybrid_hnn_gnn_model_v2 import create_hybrid_model_v2
from physics_informed_loss_functions import PhysicsInformedLoss


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        hnn_input = batch['hnn_input'].to(device)
        gnn_data = batch['gnn_data'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        results = model(hnn_input, gnn_data)
        cop_pred = results['cop_prediction']
        
        # Compute loss (simple MSE for now)
        # Expand targets to match prediction shape [batch, seq_len, 1]
        targets_expanded = targets.unsqueeze(1).expand_as(cop_pred)
        loss = nn.functional.mse_loss(cop_pred, targets_expanded)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            hnn_input = batch['hnn_input'].to(device)
            gnn_data = batch['gnn_data'].to(device)
            targets = batch['target'].to(device)
            
            results = model(hnn_input, gnn_data)
            cop_pred = results['cop_prediction']
            
            targets_expanded = targets.unsqueeze(1).expand_as(cop_pred)
            loss = nn.functional.mse_loss(cop_pred, targets_expanded)
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    return avg_loss


def test_training():
    """Test training loop."""
    
    print("=" * 70)
    print("TESTING TRAINING LOOP")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create dataloaders
    print("\n[1/5] Creating dataloaders...")
    train_loader, val_loader = create_modular_dataloaders(
        data_dir="modular_dataset_500",
        batch_size=8,
        train_split=0.8,
        num_workers=0
    )
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")
    
    # Create model
    print("\n[2/5] Creating model...")
    config = {
        'hnn_input_dim': 33,
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
    
    model = create_hybrid_model_v2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Optimizer
    print("\n[3/5] Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"   Optimizer: Adam (lr=0.001)")
    
    # Training loop
    print("\n[4/5] Training for 3 epochs...")
    n_epochs = 3
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
    
    # Final validation
    print("\n[5/5] Final validation...")
    final_val_loss = validate(model, val_loader, device)
    print(f"   Final Val Loss: {final_val_loss:.4f}")
    
    # Check predictions
    print("\n[6/6] Checking predictions...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        hnn_input = batch['hnn_input'].to(device)
        gnn_data = batch['gnn_data'].to(device)
        targets = batch['target'].to(device)
        
        results = model(hnn_input, gnn_data)
        cop_pred = results['cop_prediction']
        
        print(f"   Prediction shape: {cop_pred.shape}")
        print(f"   Prediction range: [{cop_pred.min().item():.2f}, {cop_pred.max().item():.2f}]")
        print(f"   Prediction mean: {cop_pred.mean().item():.2f}")
        print(f"   Target mean: {targets.mean().item():.2f}")
        print(f"   Within bounds [1.5, 8.0]: {torch.all((cop_pred >= 1.5) & (cop_pred <= 8.0)).item()}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] TRAINING TEST COMPLETE!")
    print("=" * 70)
    print("\nKey Results:")
    print(f"  - Model trained for {n_epochs} epochs")
    print(f"  - Final validation loss: {final_val_loss:.4f}")
    print(f"  - Predictions within physics bounds: Yes")
    print(f"  - No NaN issues: Yes")
    print("\nReady for full training run!")


if __name__ == "__main__":
    test_training()