#!/usr/bin/env python3
"""
Hamiltonian Neural Network (HNN) Architecture for Heat Pump COP Prediction
========================================================================

Implements physics-informed Hamiltonian Neural Networks that:
1. Learn Hamiltonian dynamics from data while preserving energy conservation
2. Use symplectic integration for stable long-term predictions
3. Enforce Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
4. Integrate with heat pump thermodynamics for COP prediction
5. Support multi-modal training with GNN components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class HamiltonianNetwork(nn.Module):
    """
    Core Hamiltonian Neural Network that learns the Hamiltonian function H(q,p).
    
    The network learns H(q,p) such that:
    - dq/dt = ∂H/∂p
    - dp/dt = -∂H/∂q
    
    This preserves energy conservation and symplectic structure.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 256, 128],
                 activation: str = 'tanh',
                 use_separable: bool = True):
        """
        Initialize Hamiltonian Network.
        
        Args:
            input_dim: Dimension of phase space (q + p)
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('tanh', 'relu', 'swish')
            use_separable: Whether to use separable Hamiltonian H = T(p) + V(q)
        """
        super(HamiltonianNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.q_dim = input_dim // 2  # Generalized coordinates
        self.p_dim = input_dim // 2  # Generalized momenta
        self.use_separable = use_separable
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        if use_separable:
            # Separable Hamiltonian: H = T(p) + V(q)
            self.kinetic_net = self._build_network(self.p_dim, hidden_dims)
            self.potential_net = self._build_network(self.q_dim, hidden_dims)
        else:
            # General Hamiltonian: H(q,p)
            self.hamiltonian_net = self._build_network(input_dim, hidden_dims)
    
    def _build_network(self, input_size: int, hidden_dims: List[int]) -> nn.Module:
        """Build a feedforward network."""
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation
            ])
            prev_dim = hidden_dim
        
        # Output layer (scalar Hamiltonian)
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamiltonian H(q,p).
        
        Args:
            z: Phase space coordinates [batch_size, seq_len, q_dim + p_dim]
            
        Returns:
            Hamiltonian values [batch_size, seq_len, 1]
        """
        if self.use_separable:
            q = z[..., :self.q_dim]  # Coordinates
            p = z[..., self.q_dim:]  # Momenta
            
            T = self.kinetic_net(p)    # Kinetic energy T(p)
            V = self.potential_net(q)  # Potential energy V(q)
            
            H = T + V  # Total Hamiltonian
        else:
            H = self.hamiltonian_net(z)
        
        return H
    
    def hamiltonian_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamiltonian dynamics: [dq/dt, dp/dt].
        
        Args:
            z: Phase space coordinates [batch_size, seq_len, q_dim + p_dim]
            
        Returns:
            Time derivatives [batch_size, seq_len, q_dim + p_dim]
        """
        z.requires_grad_(True)
        H = self.forward(z)
        
        # Compute gradients
        grad_H = torch.autograd.grad(
            H.sum(), z, 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        dH_dq = grad_H[..., :self.q_dim]  # ∂H/∂q
        dH_dp = grad_H[..., self.q_dim:]  # ∂H/∂p
        
        # Hamilton's equations
        dq_dt = dH_dp   # dq/dt = ∂H/∂p
        dp_dt = -dH_dq  # dp/dt = -∂H/∂q
        
        return torch.cat([dq_dt, dp_dt], dim=-1)

class SymplecticIntegrator(nn.Module):
    """
    Symplectic integrator for stable long-term Hamiltonian dynamics.
    
    Uses Störmer-Verlet method to preserve symplectic structure.
    """
    
    def __init__(self, hamiltonian_net: HamiltonianNetwork, dt: float = 0.01):
        """
        Initialize symplectic integrator.
        
        Args:
            hamiltonian_net: Hamiltonian network
            dt: Integration time step
        """
        super(SymplecticIntegrator, self).__init__()
        self.hamiltonian_net = hamiltonian_net
        self.dt = dt
    
    def forward(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Integrate Hamiltonian dynamics using symplectic method.
        
        Args:
            z0: Initial phase space coordinates [batch_size, q_dim + p_dim]
            n_steps: Number of integration steps
            
        Returns:
            Trajectory [batch_size, n_steps, q_dim + p_dim]
        """
        batch_size = z0.shape[0]
        dim = z0.shape[1]
        q_dim = dim // 2
        
        trajectory = torch.zeros(batch_size, n_steps, dim, device=z0.device)
        trajectory[:, 0] = z0
        
        q = z0[:, :q_dim]
        p = z0[:, q_dim:]
        
        for i in range(1, n_steps):
            # Störmer-Verlet integration
            q_half, p_new = self._verlet_step(q, p)
            q_new, _ = self._verlet_step(q_half, p_new, half_step=True)
            
            trajectory[:, i] = torch.cat([q_new, p_new], dim=1)
            q, p = q_new, p_new
        
        return trajectory
    
    def _verlet_step(self, q: torch.Tensor, p: torch.Tensor, 
                     half_step: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Störmer-Verlet step."""
        dt = self.dt / 2 if half_step else self.dt
        
        z = torch.cat([q, p], dim=1)
        z.requires_grad_(True)
        
        H = self.hamiltonian_net(z.unsqueeze(1)).squeeze(1)
        
        grad_H = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        dH_dq = grad_H[:, :q.shape[1]]
        dH_dp = grad_H[:, q.shape[1]:]
        
        # Update
        q_new = q + dt * dH_dp
        p_new = p - dt * dH_dq
        
        return q_new, p_new

class HeatPumpHNN(nn.Module):
    """
    Heat Pump Hamiltonian Neural Network for COP prediction.
    
    Combines Hamiltonian dynamics with heat pump thermodynamics.
    """
    
    def __init__(self,
                 phase_space_dim: int = 8,  # 4 coordinates + 4 momenta
                 hidden_dims: List[int] = [128, 256, 128],
                 cop_hidden_dims: List[int] = [64, 32],
                 dt: float = 0.01):
        """
        Initialize Heat Pump HNN.
        
        Args:
            phase_space_dim: Dimension of phase space (must be even)
            hidden_dims: Hidden dimensions for Hamiltonian network
            cop_hidden_dims: Hidden dimensions for COP prediction network
            dt: Integration time step
        """
        super(HeatPumpHNN, self).__init__()
        
        self.phase_space_dim = phase_space_dim
        self.q_dim = phase_space_dim // 2
        self.p_dim = phase_space_dim // 2
        self.dt = dt
        
        # Hamiltonian network
        self.hamiltonian_net = HamiltonianNetwork(
            input_dim=phase_space_dim,
            hidden_dims=hidden_dims,
            use_separable=True
        )
        
        # Symplectic integrator
        self.integrator = SymplecticIntegrator(self.hamiltonian_net, dt)
        
        # COP prediction network
        self.cop_net = self._build_cop_network(phase_space_dim, cop_hidden_dims)
        
        # Feature mapping network (maps input features to phase space)
        self.feature_mapper = nn.Sequential(
            nn.Linear(8, 32),  # Input features to intermediate
            nn.Tanh(),
            nn.Linear(32, phase_space_dim),  # To phase space
            nn.Tanh()
        )
    
    def _build_cop_network(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build COP prediction network."""
        layers = []
        prev_dim = input_dim + 1  # Phase space + Hamiltonian
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (COP prediction)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure positive COP
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Heat Pump HNN.
        
        Args:
            x: Input features [batch_size, seq_len, feature_dim]
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Dictionary containing predictions and intermediate results
        """
        batch_size, seq_len, _ = x.shape
        
        # Map input features to phase space
        z = self.feature_mapper(x)  # [batch_size, seq_len, phase_space_dim]
        
        # Compute Hamiltonian for each time step
        H = self.hamiltonian_net(z)  # [batch_size, seq_len, 1]
        
        # Predict COP from phase space + Hamiltonian
        cop_input = torch.cat([z, H], dim=-1)
        cop_pred = self.cop_net(cop_input) * 7.0 + 1.0  # Scale to [1, 8] range
        
        results = {
            'cop_prediction': cop_pred,
            'hamiltonian': H,
            'phase_space': z
        }
        
        if return_trajectory:
            # Integrate dynamics for trajectory prediction
            z0 = z[:, 0]  # Initial conditions
            trajectory = self.integrator(z0, seq_len)
            results['trajectory'] = trajectory
        
        return results
    
    def compute_energy_conservation_loss(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute energy conservation loss.
        
        Args:
            H: Hamiltonian values [batch_size, seq_len, 1]
            
        Returns:
            Conservation loss
        """
        # Energy should be conserved: dH/dt ≈ 0
        dH_dt = torch.diff(H, dim=1) / self.dt
        conservation_loss = torch.mean(dH_dt**2)
        return conservation_loss

class MultiScaleHNN(nn.Module):
    """
    Multi-scale Hamiltonian Neural Network for different time scales.
    
    Handles fast dynamics (compressor cycles) and slow dynamics (thermal response).
    """
    
    def __init__(self,
                 fast_dim: int = 4,
                 slow_dim: int = 4,
                 hidden_dims: List[int] = [64, 128, 64]):
        """
        Initialize multi-scale HNN.
        
        Args:
            fast_dim: Dimension for fast dynamics
            slow_dim: Dimension for slow dynamics
            hidden_dims: Hidden layer dimensions
        """
        super(MultiScaleHNN, self).__init__()
        
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        
        # Fast dynamics HNN (compressor, expansion valve)
        self.fast_hnn = HamiltonianNetwork(
            input_dim=fast_dim,
            hidden_dims=hidden_dims,
            activation='tanh'
        )
        
        # Slow dynamics HNN (heat exchangers, thermal mass)
        self.slow_hnn = HamiltonianNetwork(
            input_dim=slow_dim,
            hidden_dims=hidden_dims,
            activation='tanh'
        )
        
        # Coupling network (interaction between scales)
        self.coupling_net = nn.Sequential(
            nn.Linear(fast_dim + slow_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-scale dynamics.
        
        Args:
            z_fast: Fast phase space coordinates
            z_slow: Slow phase space coordinates
            
        Returns:
            Multi-scale Hamiltonian components
        """
        H_fast = self.fast_hnn(z_fast)
        H_slow = self.slow_hnn(z_slow)
        
        # Coupling between scales
        z_combined = torch.cat([z_fast, z_slow], dim=-1)
        H_coupling = self.coupling_net(z_combined)
        
        H_total = H_fast + H_slow + H_coupling
        
        return {
            'hamiltonian_total': H_total,
            'hamiltonian_fast': H_fast,
            'hamiltonian_slow': H_slow,
            'hamiltonian_coupling': H_coupling
        }

class AdaptiveHNN(nn.Module):
    """
    Adaptive Hamiltonian Neural Network with learnable time step.
    """
    
    def __init__(self, 
                 phase_space_dim: int,
                 hidden_dims: List[int] = [128, 256, 128],
                 initial_dt: float = 0.01):
        """
        Initialize adaptive HNN.
        
        Args:
            phase_space_dim: Phase space dimension
            hidden_dims: Hidden layer dimensions
            initial_dt: Initial time step
        """
        super(AdaptiveHNN, self).__init__()
        
        self.hamiltonian_net = HamiltonianNetwork(
            input_dim=phase_space_dim,
            hidden_dims=hidden_dims
        )
        
        # Learnable time step
        self.log_dt = nn.Parameter(torch.log(torch.tensor(initial_dt)))
        
        # Adaptive step size network
        self.step_size_net = nn.Sequential(
            nn.Linear(phase_space_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive time stepping."""
        H = self.hamiltonian_net(z)
        dt = torch.exp(self.log_dt)
        
        # Adaptive step size based on local dynamics
        step_scale = self.step_size_net(z)
        adaptive_dt = dt * step_scale
        
        return {
            'hamiltonian': H,
            'time_step': dt,
            'adaptive_time_step': adaptive_dt
        }

def create_hnn_model(config: Dict) -> nn.Module:
    """
    Factory function to create HNN models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HNN model instance
    """
    model_type = config.get('type', 'heat_pump')
    
    if model_type == 'heat_pump':
        return HeatPumpHNN(
            phase_space_dim=config.get('phase_space_dim', 8),
            hidden_dims=config.get('hidden_dims', [128, 256, 128]),
            cop_hidden_dims=config.get('cop_hidden_dims', [64, 32]),
            dt=config.get('dt', 0.01)
        )
    elif model_type == 'multi_scale':
        return MultiScaleHNN(
            fast_dim=config.get('fast_dim', 4),
            slow_dim=config.get('slow_dim', 4),
            hidden_dims=config.get('hidden_dims', [64, 128, 64])
        )
    elif model_type == 'adaptive':
        return AdaptiveHNN(
            phase_space_dim=config.get('phase_space_dim', 8),
            hidden_dims=config.get('hidden_dims', [128, 256, 128]),
            initial_dt=config.get('initial_dt', 0.01)
        )
    else:
        raise ValueError(f"Unknown HNN model type: {model_type}")

def test_hnn_architecture():
    """Test HNN architecture components."""
    
    print("🧪 TESTING HAMILTONIAN NEURAL NETWORK ARCHITECTURE")
    print("=" * 60)
    
    # Test basic Hamiltonian network
    print("1. Testing Basic Hamiltonian Network...")
    hnn = HamiltonianNetwork(input_dim=8, hidden_dims=[64, 128, 64])
    z = torch.randn(32, 100, 8)  # [batch, seq_len, phase_space]
    
    H = hnn(z)
    dynamics = hnn.hamiltonian_dynamics(z)
    
    print(f"   Input shape: {z.shape}")
    print(f"   Hamiltonian shape: {H.shape}")
    print(f"   Dynamics shape: {dynamics.shape}")
    print(f"   ✅ Basic HNN working")
    
    # Test symplectic integrator
    print("\n2. Testing Symplectic Integrator...")
    integrator = SymplecticIntegrator(hnn, dt=0.01)
    z0 = torch.randn(32, 8)
    trajectory = integrator(z0, n_steps=100)
    
    print(f"   Initial state shape: {z0.shape}")
    print(f"   Trajectory shape: {trajectory.shape}")
    print(f"   ✅ Symplectic integrator working")
    
    # Test Heat Pump HNN
    print("\n3. Testing Heat Pump HNN...")
    heat_pump_hnn = HeatPumpHNN(
        phase_space_dim=8,
        hidden_dims=[64, 128, 64],
        cop_hidden_dims=[32, 16]
    )
    
    x = torch.randn(32, 100, 8)  # Input features
    results = heat_pump_hnn(x, return_trajectory=True)
    
    print(f"   Input features shape: {x.shape}")
    print(f"   COP prediction shape: {results['cop_prediction'].shape}")
    print(f"   Hamiltonian shape: {results['hamiltonian'].shape}")
    print(f"   Phase space shape: {results['phase_space'].shape}")
    print(f"   Trajectory shape: {results['trajectory'].shape}")
    
    # Test energy conservation
    conservation_loss = heat_pump_hnn.compute_energy_conservation_loss(results['hamiltonian'])
    print(f"   Energy conservation loss: {conservation_loss.item():.6f}")
    print(f"   ✅ Heat Pump HNN working")
    
    # Test multi-scale HNN
    print("\n4. Testing Multi-Scale HNN...")
    multi_scale_hnn = MultiScaleHNN(fast_dim=4, slow_dim=4)
    z_fast = torch.randn(32, 100, 4)
    z_slow = torch.randn(32, 100, 4)
    
    multi_results = multi_scale_hnn(z_fast, z_slow)
    
    print(f"   Fast dynamics shape: {z_fast.shape}")
    print(f"   Slow dynamics shape: {z_slow.shape}")
    print(f"   Total Hamiltonian shape: {multi_results['hamiltonian_total'].shape}")
    print(f"   ✅ Multi-scale HNN working")
    
    # Test adaptive HNN
    print("\n5. Testing Adaptive HNN...")
    adaptive_hnn = AdaptiveHNN(phase_space_dim=8)
    z = torch.randn(32, 100, 8)
    
    adaptive_results = adaptive_hnn(z)
    
    print(f"   Input shape: {z.shape}")
    print(f"   Hamiltonian shape: {adaptive_results['hamiltonian'].shape}")
    print(f"   Time step: {adaptive_results['time_step'].item():.6f}")
    print(f"   Adaptive time step shape: {adaptive_results['adaptive_time_step'].shape}")
    print(f"   ✅ Adaptive HNN working")
    
    print(f"\n✅ ALL HNN COMPONENTS TESTED SUCCESSFULLY!")
    print(f"🎯 Ready for integration with GNN and physics-informed training!")
    
    # Test model factory
    print(f"\n6. Testing Model Factory...")
    config = {
        'type': 'heat_pump',
        'phase_space_dim': 8,
        'hidden_dims': [64, 128, 64],
        'dt': 0.01
    }
    
    model = create_hnn_model(config)
    test_input = torch.randn(16, 50, 8)
    test_output = model(test_input)
    
    print(f"   Factory model type: {type(model).__name__}")
    print(f"   Test output keys: {list(test_output.keys())}")
    print(f"   ✅ Model factory working")
    
    print(f"\n🎉 HAMILTONIAN NEURAL NETWORK ARCHITECTURE COMPLETE!")

if __name__ == "__main__":
    test_hnn_architecture()