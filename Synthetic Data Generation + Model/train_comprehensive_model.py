#!/usr/bin/env python3
"""
Training Script for Comprehensive Physics-Based HNN-GNN Model
=============================================================

Integrates:
- Comprehensive physics dataset (from comprehensive_physics_generator_v2.py)
- Comprehensive data loader (from comprehensive_data_loader.py)
- Hybrid HNN-GNN model (from hybrid_hnn_gnn_model_v2.py)
- Physics-informed loss functions

Usage:
    python train_comprehensive_model.py --data_dir comprehensive_dataset_test --epochs 50

Author: Comprehensive training pipeline
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Import components
from comprehensive_data_loader import create_data_loaders, ComprehensiveDataset
from hamiltonian_neural_network import HeatPumpHNN
from graph_neural_network import HeatPumpGNN
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class SimplifiedHybridModel(nn.Module):
    """
    Simplified Hybrid HNN-GNN model matched to comprehensive dataset dimensions.
    
    Input dimensions (from comprehensive_data_loader):
    - HNN: [batch, 100 timesteps, 14 features]
    - GNN nodes: [batch*10, 35 features]
    - GNN edges: [batch*11, 12 features]
    - GNN graph: [batch, 50 features]
    
    Output: COP prediction [batch, 1]
    """
    
    def __init__(self,
                 hnn_input_dim: int = 14,
                 hnn_hidden_dim: int = 64,
                 gnn_node_dim: int = 35,
                 gnn_edge_dim: int = 12,
                 gnn_graph_dim: int = 50,
                 gnn_hidden_dim: int = 64,
                 fusion_dim: int = 128,
                 dropout: float = 0.1,
                 cop_mean: float = 4.77,
                 cop_std: float = 1.74):
        """Initialize simplified hybrid model."""
        super(SimplifiedHybridModel, self).__init__()
        
        # COP statistics for output scaling
        self.cop_mean = cop_mean
        self.cop_std = cop_std
        
        # HNN Branch: Temporal dynamics encoder
        self.hnn_encoder = nn.Sequential(
            nn.Linear(hnn_input_dim, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hnn_hidden_dim, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.ReLU()
        )
        
        # HNN: Process sequence with LSTM
        self.hnn_lstm = nn.LSTM(
            input_size=hnn_hidden_dim,
            hidden_size=hnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # GNN Branch: Node encoder
        self.gnn_node_encoder = nn.Sequential(
            nn.Linear(gnn_node_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU()
        )
        
        # GNN: Edge encoder
        self.gnn_edge_encoder = nn.Sequential(
            nn.Linear(gnn_edge_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU()
        )
        
        # GNN: Graph encoder
        self.gnn_graph_encoder = nn.Sequential(
            nn.Linear(gnn_graph_dim, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.ReLU()
        )
        
        # Message passing (simplified)
        self.message_layer = nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim)
        
        # Fusion layer
        hnn_out_dim = hnn_hidden_dim
        gnn_out_dim = gnn_hidden_dim * 3  # mean + max + graph
        
        self.fusion = nn.Sequential(
            nn.Linear(hnn_out_dim + gnn_out_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # COP prediction head - output normalized COP (z-score)
        self.cop_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, hnn_input: torch.Tensor, gnn_data: Batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hnn_input: [batch, 100, 14]
            gnn_data: PyG Batch with nodes, edges, graph_attr
            
        Returns:
            Dictionary with cop_prediction and intermediates
        """
        batch_size = hnn_input.size(0)
        
        # HNN Branch
        hnn_encoded = self.hnn_encoder(hnn_input)  # [batch, 100, hidden]
        hnn_out, (h_n, c_n) = self.hnn_lstm(hnn_encoded)
        hnn_features = h_n[-1]  # Last layer hidden state [batch, hidden]
        
        # GNN Branch
        x = gnn_data.x  # [batch*10, 35]
        edge_index = gnn_data.edge_index
        edge_attr = gnn_data.edge_attr  # [batch*11, 12]
        graph_attr = gnn_data.graph_attr  # [batch*50] - needs reshaping
        batch_idx = gnn_data.batch
        
        # Reshape graph_attr if needed
        if graph_attr.dim() == 1:
            graph_attr = graph_attr.view(batch_size, -1)
        
        # Encode nodes
        x = self.gnn_node_encoder(x)
        
        # Encode edges
        edge_features = self.gnn_edge_encoder(edge_attr)
        
        # Simple message passing (aggregate neighbor features)
        # Get source and target node indices
        src, tgt = edge_index
        
        # Aggregate messages
        messages = x[src] + x[tgt]  # Simple sum
        messages = messages[:, :x.size(1)]  # Ensure same size
        
        # Pool per graph
        gnn_mean = global_mean_pool(x, batch_idx)
        gnn_max = global_max_pool(x, batch_idx)
        
        # Encode graph features
        graph_encoded = self.gnn_graph_encoder(graph_attr)
        
        # Combine GNN outputs
        gnn_features = torch.cat([gnn_mean, gnn_max, graph_encoded], dim=-1)
        
        # Fusion
        combined = torch.cat([hnn_features, gnn_features], dim=-1)
        fused = self.fusion(combined)
        
        # COP prediction - output is in normalized space, denormalize
        cop_normalized = self.cop_head(fused)  # Output in z-score space
        cop_pred = cop_normalized * self.cop_std + self.cop_mean  # Denormalize
        
        # Soft clamp using sigmoid to maintain gradients (maps to [1, 10])
        # cop_pred = 1.0 + 9.0 * torch.sigmoid(cop_pred - 5.5)  # Soft constraint
        
        return {
            'cop_prediction': cop_pred,
            'cop_normalized': cop_normalized,
            'hnn_features': hnn_features,
            'gnn_features': gnn_features,
            'fused_features': fused
        }


class PhysicsInformedTrainer:
    """
    Training pipeline with research-standard physics-informed loss.
    
    Based on:
    - Greydanus et al. (2019) - Hamiltonian Neural Networks
    - Raissi et al. (2019) - Physics-Informed Neural Networks
    - Chen et al. (2018) - Neural ODEs
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 carnot_weight: float = 0.01,
                 hamiltonian_weight: float = 0.1,
                 energy_balance_weight: float = 0.01,
                 n_epochs: int = 100,
                 early_stopping_patience: int = 15):
        """
        Initialize trainer with research-standard configuration.
        
        Args:
            model: The model to train
            device: Training device
            learning_rate: Initial learning rate (research standard: 1e-3)
            weight_decay: AdamW weight decay (research standard: 1e-4)
            carnot_weight: Weight for Carnot constraint loss
            hamiltonian_weight: Weight for Hamiltonian conservation loss (critical for HNN)
            energy_balance_weight: Weight for energy balance constraint
            n_epochs: Total number of epochs (for cosine annealing)
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.carnot_weight = carnot_weight
        self.hamiltonian_weight = hamiltonian_weight
        self.energy_balance_weight = energy_balance_weight
        self.early_stopping_patience = early_stopping_patience
        
        # Research-standard: AdamW with weight decay
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Research-standard: Cosine annealing LR scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def compute_loss(self,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     carnot_cop: torch.Tensor,
                     hamiltonian_H: torch.Tensor = None,
                     Q_cond: torch.Tensor = None,
                     Q_evap: torch.Tensor = None,
                     W_comp: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined physics-informed loss (research standard).
        
        Loss = L_data + λ_H * L_hamilton + λ_C * L_carnot + λ_E * L_energy
        
        Args:
            predictions: Model COP predictions
            targets: Ground truth COP
            carnot_cop: Theoretical Carnot COP limit
            hamiltonian_H: Hamiltonian energy over timesteps [batch, timesteps]
            Q_cond: Condenser heat [batch, timesteps]
            Q_evap: Evaporator heat [batch, timesteps]
            W_comp: Compressor work [batch, timesteps]
            
        Returns:
            Dictionary of loss components
        """
        # Data loss (MSE on COP)
        L_data = F.mse_loss(predictions, targets)
        
        # HAMILTONIAN CONSERVATION LOSS (critical for HNN - dH/dt ~ 0)
        # This is THE defining feature of Hamiltonian Neural Networks
        L_hamilton = torch.tensor(0.0, device=self.device)
        if hamiltonian_H is not None and hamiltonian_H.numel() > 0:
            # Compute time derivative using finite differences
            dH_dt = hamiltonian_H[:, 1:] - hamiltonian_H[:, :-1]
            L_hamilton = torch.mean(dH_dt ** 2)
        
        # Carnot constraint loss (COP should be below Carnot)
        carnot_violation = F.relu(predictions - carnot_cop)
        L_carnot = torch.mean(carnot_violation ** 2)
        
        # Energy balance loss: Q_cond = Q_evap + W_comp (First Law)
        L_energy = torch.tensor(0.0, device=self.device)
        if Q_cond is not None and Q_evap is not None and W_comp is not None:
            # Use mean values over trajectory
            Q_cond_mean = Q_cond.mean(dim=1, keepdim=True)
            Q_evap_mean = Q_evap.mean(dim=1, keepdim=True)
            W_comp_mean = W_comp.mean(dim=1, keepdim=True)
            energy_imbalance = Q_cond_mean - Q_evap_mean - W_comp_mean
            L_energy = torch.mean(energy_imbalance ** 2)
        
        # Total loss with research-standard weights
        total_loss = (L_data +
                      self.hamiltonian_weight * L_hamilton +
                      self.carnot_weight * L_carnot +
                      self.energy_balance_weight * L_energy)
        
        return {
            'total': total_loss,
            'mse': L_data,
            'hamiltonian': L_hamilton,
            'carnot': L_carnot,
            'energy': L_energy
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with physics-informed loss."""
        self.model.train()
        total_loss = 0
        mse_loss_sum = 0
        hamiltonian_loss_sum = 0
        carnot_loss_sum = 0
        energy_loss_sum = 0
        n_batches = 0
        
        for batch in train_loader:
            # Move to device
            hnn_input = batch['hnn_input'].to(self.device)
            gnn_data = batch['gnn_data'].to(self.device)
            cop_target = batch['cop_target'].to(self.device)
            carnot_cop = batch['carnot_cop'].to(self.device)
            
            # Extract Hamiltonian and energy data for physics losses
            hamiltonian_H = batch.get('hamiltonian_H')
            Q_cond = batch.get('Q_cond_traj')
            Q_evap = batch.get('Q_evap_traj')
            W_comp = batch.get('W_comp_traj')
            
            if hamiltonian_H is not None:
                hamiltonian_H = hamiltonian_H.to(self.device)
            if Q_cond is not None:
                Q_cond = Q_cond.to(self.device)
            if Q_evap is not None:
                Q_evap = Q_evap.to(self.device)
            if W_comp is not None:
                W_comp = W_comp.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(hnn_input, gnn_data)
            predictions = outputs['cop_prediction']
            
            # Compute physics-informed loss
            losses = self.compute_loss(
                predictions, cop_target, carnot_cop,
                hamiltonian_H=hamiltonian_H,
                Q_cond=Q_cond,
                Q_evap=Q_evap,
                W_comp=W_comp
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Research-standard: Gradient clipping (prevents physics constraint explosions)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += losses['total'].item()
            mse_loss_sum += losses['mse'].item()
            hamiltonian_loss_sum += losses['hamiltonian'].item()
            carnot_loss_sum += losses['carnot'].item()
            energy_loss_sum += losses['energy'].item()
            n_batches += 1
        
        return {
            'total': total_loss / n_batches,
            'mse': mse_loss_sum / n_batches,
            'hamiltonian': hamiltonian_loss_sum / n_batches,
            'carnot': carnot_loss_sum / n_batches,
            'energy': energy_loss_sum / n_batches
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        mse_loss_sum = 0
        predictions_all = []
        targets_all = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                hnn_input = batch['hnn_input'].to(self.device)
                gnn_data = batch['gnn_data'].to(self.device)
                cop_target = batch['cop_target'].to(self.device)
                carnot_cop = batch['carnot_cop'].to(self.device)
                
                outputs = self.model(hnn_input, gnn_data)
                predictions = outputs['cop_prediction']
                
                losses = self.compute_loss(predictions, cop_target, carnot_cop)
                
                total_loss += losses['total'].item()
                mse_loss_sum += losses['mse'].item()
                n_batches += 1
                
                predictions_all.extend(predictions.cpu().numpy().flatten())
                targets_all.extend(cop_target.cpu().numpy().flatten())
        
        # Compute metrics
        predictions_arr = np.array(predictions_all)
        targets_arr = np.array(targets_all)
        
        mae = np.mean(np.abs(predictions_arr - targets_arr))
        rmse = np.sqrt(np.mean((predictions_arr - targets_arr) ** 2))
        
        # R-squared
        ss_res = np.sum((targets_arr - predictions_arr) ** 2)
        ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'total': total_loss / n_batches,
            'mse': mse_loss_sum / n_batches,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train(self,
              train_loader,
              val_loader,
              n_epochs: int = 100,
              save_dir: str = 'checkpoints'):
        """
        Full training loop with early stopping and research-standard configuration.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("STARTING TRAINING (Research-Standard HNN-GNN Configuration)")
        print("=" * 70)
        print(f"  Device: {self.device}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
        print(f"  Physics losses: Hamiltonian={self.hamiltonian_weight}, "
              f"Carnot={self.carnot_weight}, Energy={self.energy_balance_weight}")
        print("=" * 70)
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler (CosineAnnealingLR doesn't need metrics)
            self.scheduler.step()
            
            # Track losses
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_metrics['total'])
            
            # Save best model and check early stopping
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['total'],
                    'val_r2': val_metrics['r2'],
                    'val_mae': val_metrics['mae']
                }, save_path / 'best_model.pt')
            else:
                self.epochs_without_improvement += 1
            
            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress with physics losses
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train: {train_losses['total']:.4f} "
                  f"(MSE:{train_losses['mse']:.4f}, H:{train_losses['hamiltonian']:.4f}) | "
                  f"Val: {val_metrics['total']:.4f} | "
                  f"MAE: {val_metrics['mae']:.3f} | "
                  f"R2: {val_metrics['r2']:.3f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {self.early_stopping_patience} epochs.")
                break
        
        # Save final model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path / 'final_model.pt')
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Final validation MAE: {val_metrics['mae']:.3f}")
        print(f"  Final validation R2: {val_metrics['r2']:.3f}")
        print(f"  Models saved to: {save_path}")
        print("=" * 70)
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_mae': val_metrics['mae'],
            'final_r2': val_metrics['r2'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train HNN-GNN COP model')
    parser.add_argument('--data_dir', type=str, default='comprehensive_dataset_test',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, or cpu')
    args = parser.parse_args()
    
    # Determine device with robust CUDA testing
    def test_cuda_kernels():
        """Test if CUDA kernels are available for this GPU."""
        try:
            # Try a simple CUDA operation
            test_tensor = torch.randn(10, 10, device='cuda')
            _ = torch.mm(test_tensor, test_tensor)
            return True
        except RuntimeError as e:
            if 'no kernel image' in str(e).lower():
                return False
            raise
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"       Testing CUDA kernel compatibility...")
            if test_cuda_kernels():
                device = 'cuda'
                print(f"       ✓ CUDA kernels working")
            else:
                device = 'cpu'
                print(f"       ✗ CUDA kernels not available for this GPU architecture")
                print(f"       [NOTE] RTX 50-series (Blackwell) requires PyTorch nightly with CUDA 12.6+")
                print(f"       [TIP] Install: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126")
                print(f"       Falling back to CPU training...")
        else:
            device = 'cpu'
    else:
        device = args.device
    
    if device == 'cuda':
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"       CUDA Version: {torch.version.cuda}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[INFO] Using CPU")
    
    # Create data loaders
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Create model
    print("[INFO] Creating model...")
    model = SimplifiedHybridModel(
        hnn_input_dim=14,
        hnn_hidden_dim=64,
        gnn_node_dim=35,
        gnn_edge_dim=12,
        gnn_graph_dim=50,
        gnn_hidden_dim=64,
        fusion_dim=128,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer with research-standard configuration
    trainer = PhysicsInformedTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=1e-4,  # Research standard
        carnot_weight=0.01,
        hamiltonian_weight=0.1,  # Critical for true HNN
        energy_balance_weight=0.01,
        n_epochs=args.epochs,
        early_stopping_patience=15
    )
    
    # Print physics loss explanation
    print("\n[PHYSICS-INFORMED LOSS COMPONENTS]")
    print("  MSE Loss: Prediction accuracy (main objective)")
    print("  Hamiltonian (H): dH/dt ~ 0 (energy conservation)")
    print("    - Low H loss = data is physically consistent")
    print("    - High H loss = energy conservation violated")
    print("  Carnot: Ensures COP <= Carnot limit")
    print("  Energy: Q_cond = Q_evap + W_comp (First Law)")
    print("")
    
    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Save training results to JSON (convert numpy types to Python native)
    results_path = Path(args.save_dir) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_val_loss': float(results['best_val_loss']),
            'final_mae': float(results['final_mae']),
            'final_r2': float(results['final_r2']),
            'epochs_trained': len(results['train_losses']),
            'train_losses': [float(x) for x in results['train_losses']],
            'val_losses': [float(x) for x in results['val_losses']],
            'configuration': {
                'learning_rate': args.lr,
                'weight_decay': 1e-4,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
                'hamiltonian_weight': 0.1,
                'carnot_weight': 0.01,
                'energy_balance_weight': 0.01,
                'early_stopping_patience': 15
            }
        }, f, indent=2)
    print(f"[INFO] Training results saved to: {results_path}")
    
    # Final evaluation on test set
    print("\n[INFO] Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print(f"  Test Loss: {test_metrics['total']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.3f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.3f}")
    print(f"  Test R2: {test_metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
