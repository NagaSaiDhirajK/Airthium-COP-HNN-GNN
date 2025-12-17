#!/usr/bin/env python3
"""
Training Script for Optimal Hybrid Model
========================================

Trains the optimal hybrid model on complete physics dataset (4,108 features).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import h5py
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from optimal_hybrid_architecture import OptimalHybridModel


class CompletePhysicsDataset(Dataset):
    """Dataset for complete physics data with 4,108 features."""
    
    def __init__(self, data_path: str):
        with h5py.File(data_path, 'r') as f:
            self.physics_features = torch.FloatTensor(f['features'][:])
            self.targets = torch.FloatTensor(f['targets'][:]).unsqueeze(1)
            self.n_samples = f.attrs['n_cycles']
            self.n_features = f.attrs['n_features']
        
        print(f"Loaded dataset: {self.n_samples} samples, {self.n_features} features")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'physics_features': self.physics_features[idx],
            'target': self.targets[idx]
        }


def collate_fn(batch):
    """Collate function with graph structure."""
    physics_features = torch.stack([item['physics_features'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    
    # Fixed graph structure (5 nodes, 16 edges)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2, 1, 3, 2, 4],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0, 3, 1, 4, 2]
    ], dtype=torch.long)
    
    # Batch assignment
    batch_size = len(batch)
    num_nodes = 5
    batch_tensor = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    return {
        'physics_features': physics_features,
        'edge_index': edge_index,
        'batch': batch_tensor,
        'targets': targets
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    loss_components = {'cop': 0, 'hamiltonian': 0, 'energy': 0}
    
    for batch in tqdm(dataloader, desc="Training"):
        physics_features = batch['physics_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_tensor = batch['batch'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        predictions, aux_outputs = model(physics_features, edge_index, batch_tensor)
        
        loss, loss_dict = model.compute_physics_informed_loss(predictions, targets, aux_outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in ['cop', 'hamiltonian', 'energy']:
            loss_components[key] += loss_dict[f'{key}_loss']
    
    n = len(dataloader)
    return {
        'total': total_loss / n,
        'cop': loss_components['cop'] / n,
        'hamiltonian': loss_components['hamiltonian'] / n,
        'energy': loss_components['energy'] / n
    }


def validate_epoch(model, dataloader, device):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    loss_components = {'cop': 0, 'hamiltonian': 0, 'energy': 0}
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            physics_features = batch['physics_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_tensor = batch['batch'].to(device)
            targets = batch['targets'].to(device)
            
            predictions, aux_outputs = model(physics_features, edge_index, batch_tensor)
            loss, loss_dict = model.compute_physics_informed_loss(predictions, targets, aux_outputs)
            
            total_loss += loss.item()
            for key in ['cop', 'hamiltonian', 'energy']:
                loss_components[key] += loss_dict[f'{key}_loss']
            
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets)**2)).item()
    
    n = len(dataloader)
    return {
        'total': total_loss / n,
        'cop': loss_components['cop'] / n,
        'hamiltonian': loss_components['hamiltonian'] / n,
        'energy': loss_components['energy'] / n,
        'mae': mae,
        'rmse': rmse
    }


def main():
    """Main training function."""
    
    print("="*70)
    print("OPTIMAL HYBRID MODEL TRAINING")
    print("="*70)
    
    # Config
    data_path = "complete_physics_data/airthium_complete_physics_dataset.h5"
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CompletePhysicsDataset(data_path)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Create model
    print("\nCreating model...")
    model = OptimalHybridModel(
        input_dim=dataset.n_features,
        embedding_dim=256,
        hidden_dim=128,
        num_nodes=5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        
        print(f"Train - Total: {train_loss['total']:.4f}, COP: {train_loss['cop']:.4f}, H: {train_loss['hamiltonian']:.6f}, E: {train_loss['energy']:.4f}")
        print(f"Val   - Total: {val_loss['total']:.4f}, MAE: {val_loss['mae']:.4f}, RMSE: {val_loss['rmse']:.4f}")
        
        scheduler.step(val_loss['total'])
        
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss['total'],
                'val_mae': val_loss['mae'],
                'val_rmse': val_loss['rmse']
            }, 'best_optimal_model.pt')
            print("✓ Saved best model!")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: best_optimal_model.pt")


if __name__ == "__main__":
    main()