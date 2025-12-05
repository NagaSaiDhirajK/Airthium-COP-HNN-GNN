#!/usr/bin/env python3
"""
Physics-Informed Loss Functions for HNN-GNN Training
==================================================

Implement physics-informed loss functions that enforce:
1. Hamiltonian energy conservation (dH/dt = 0)
2. Thermodynamic constraints (entropy, energy balance)
3. Heat pump cycle constraints (COP bounds, efficiency)
4. Component interaction physics (mass/energy conservation)
5. Graph structure consistency (node/edge relationships)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PhysicsInformedLoss(nn.Module):
    """
    Comprehensive physics-informed loss function for HNN-GNN training.
    
    Combines multiple physics constraints:
    - Hamiltonian conservation
    - Thermodynamic laws
    - Heat pump physics
    - Graph consistency
    """
    
    def __init__(self, 
                 lambda_hamiltonian: float = 1.0,
                 lambda_thermodynamic: float = 0.5,
                 lambda_heat_pump: float = 0.8,
                 lambda_graph: float = 0.3,
                 lambda_data: float = 1.0):
        """
        Initialize physics-informed loss function.
        
        Args:
            lambda_hamiltonian: Weight for Hamiltonian conservation loss
            lambda_thermodynamic: Weight for thermodynamic constraints
            lambda_heat_pump: Weight for heat pump physics
            lambda_graph: Weight for graph consistency
            lambda_data: Weight for data fitting loss
        """
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_hamiltonian = lambda_hamiltonian
        self.lambda_thermodynamic = lambda_thermodynamic
        self.lambda_heat_pump = lambda_heat_pump
        self.lambda_graph = lambda_graph
        self.lambda_data = lambda_data
        
        # Base loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                hamiltonian_data: Dict[str, torch.Tensor],
                thermodynamic_data: Dict[str, torch.Tensor],
                heat_pump_data: Dict[str, torch.Tensor],
                graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive physics-informed loss.
        
        Args:
            predictions: Model COP predictions [batch_size, seq_len, 1]
            targets: True COP values [batch_size, seq_len, 1]
            hamiltonian_data: Hamiltonian mechanics data
            thermodynamic_data: Thermodynamic properties
            heat_pump_data: Heat pump specific data
            graph_data: Graph structure data
            
        Returns:
            Dictionary of loss components
        """
        
        # Data fitting loss
        data_loss = self.mse_loss(predictions, targets)
        
        # Physics constraint losses
        hamiltonian_loss = self.hamiltonian_conservation_loss(hamiltonian_data)
        thermodynamic_loss = self.thermodynamic_constraints_loss(thermodynamic_data)
        heat_pump_loss = self.heat_pump_physics_loss(heat_pump_data, predictions)
        graph_loss = self.graph_consistency_loss(graph_data)
        
        # Total physics-informed loss
        total_loss = (self.lambda_data * data_loss +
                     self.lambda_hamiltonian * hamiltonian_loss +
                     self.lambda_thermodynamic * thermodynamic_loss +
                     self.lambda_heat_pump * heat_pump_loss +
                     self.lambda_graph * graph_loss)
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'hamiltonian_loss': hamiltonian_loss,
            'thermodynamic_loss': thermodynamic_loss,
            'heat_pump_loss': heat_pump_loss,
            'graph_loss': graph_loss
        }

class HamiltonianConservationLoss(nn.Module):
    """
    Enforce Hamiltonian energy conservation: dH/dt = 0
    """
    
    def __init__(self):
        super(HamiltonianConservationLoss, self).__init__()
        
    def forward(self, hamiltonian_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Hamiltonian conservation loss.
        
        Args:
            hamiltonian_data: Dictionary containing:
                - 'hamiltonian': H values [batch_size, seq_len]
                - 'q': Generalized coordinates [batch_size, seq_len, n_q]
                - 'p': Generalized momenta [batch_size, seq_len, n_p]
                - 'dt': Time step
                
        Returns:
            Conservation loss
        """
        
        H = hamiltonian_data['hamiltonian']  # [batch_size, seq_len]
        dt = hamiltonian_data.get('dt', 0.01)
        
        # Compute dH/dt using finite differences
        dH_dt = torch.diff(H, dim=1) / dt  # [batch_size, seq_len-1]
        
        # Energy should be conserved: dH/dt ≈ 0
        conservation_loss = torch.mean(dH_dt**2)
        
        return conservation_loss

class ThermodynamicConstraintsLoss(nn.Module):
    """
    Enforce thermodynamic laws and constraints.
    """
    
    def __init__(self):
        super(ThermodynamicConstraintsLoss, self).__init__()
        
    def forward(self, thermo_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute thermodynamic constraints loss.
        
        Args:
            thermo_data: Dictionary containing:
                - 'temperature': T [batch_size, seq_len, n_components]
                - 'pressure': P [batch_size, seq_len, n_components]
                - 'enthalpy': h [batch_size, seq_len, n_components]
                - 'entropy': s [batch_size, seq_len, n_components]
                - 'mass_flow': m_dot [batch_size, seq_len, n_edges]
                - 'heat_flow': Q_dot [batch_size, seq_len, n_edges]
                
        Returns:
            Thermodynamic constraints loss
        """
        
        T = thermo_data['temperature']
        P = thermo_data['pressure']
        h = thermo_data['enthalpy']
        s = thermo_data['entropy']
        m_dot = thermo_data['mass_flow']
        Q_dot = thermo_data['heat_flow']
        
        total_loss = 0.0
        
        # 1. Second law of thermodynamics: dS/dt >= 0
        ds_dt = torch.diff(s, dim=1)  # [batch_size, seq_len-1, n_components]
        entropy_violation = torch.relu(-ds_dt)  # Penalize entropy decrease
        entropy_loss = torch.mean(entropy_violation**2)
        total_loss += entropy_loss
        
        # 2. Energy conservation: ΣQ_in = ΣQ_out + ΣW
        energy_balance = torch.sum(Q_dot, dim=-1)  # [batch_size, seq_len]
        energy_conservation_loss = torch.mean(energy_balance**2)
        total_loss += energy_conservation_loss
        
        # 3. Mass conservation: Σm_in = Σm_out
        mass_balance = torch.sum(m_dot, dim=-1)  # [batch_size, seq_len]
        mass_conservation_loss = torch.mean(mass_balance**2)
        total_loss += mass_conservation_loss
        
        # 4. Temperature bounds (physical limits)
        T_min, T_max = 200.0, 500.0  # Kelvin
        temp_violation = torch.relu(T_min - T) + torch.relu(T - T_max)
        temp_bounds_loss = torch.mean(temp_violation**2)
        total_loss += temp_bounds_loss
        
        # 5. Pressure bounds
        P_min, P_max = 0.1, 50.0  # bar
        pressure_violation = torch.relu(P_min - P) + torch.relu(P - P_max)
        pressure_bounds_loss = torch.mean(pressure_violation**2)
        total_loss += pressure_bounds_loss
        
        return total_loss

class HeatPumpPhysicsLoss(nn.Module):
    """
    Enforce heat pump specific physics constraints.
    """
    
    def __init__(self):
        super(HeatPumpPhysicsLoss, self).__init__()
        
    def forward(self, heat_pump_data: Dict[str, torch.Tensor], 
                cop_predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute heat pump physics loss.
        
        Args:
            heat_pump_data: Dictionary containing:
                - 'Q_evap': Evaporator heat [batch_size, seq_len]
                - 'Q_cond': Condenser heat [batch_size, seq_len]
                - 'W_comp': Compressor work [batch_size, seq_len]
                - 'T_evap': Evaporator temperature [batch_size, seq_len]
                - 'T_cond': Condenser temperature [batch_size, seq_len]
            cop_predictions: Predicted COP values [batch_size, seq_len, 1]
                
        Returns:
            Heat pump physics loss
        """
        
        Q_evap = heat_pump_data['Q_evap']
        Q_cond = heat_pump_data['Q_cond']
        W_comp = heat_pump_data['W_comp']
        T_evap = heat_pump_data['T_evap']
        T_cond = heat_pump_data['T_cond']
        
        cop_pred = cop_predictions.squeeze(-1)  # [batch_size, seq_len]
        
        total_loss = 0.0
        
        # 1. COP definition: COP = Q_cond / W_comp
        cop_physics = Q_cond / (W_comp + 1e-6)  # Avoid division by zero
        cop_definition_loss = torch.mean((cop_pred - cop_physics)**2)
        total_loss += cop_definition_loss
        
        # 2. Energy balance: Q_cond = Q_evap + W_comp
        energy_balance = Q_cond - (Q_evap + W_comp)
        energy_balance_loss = torch.mean(energy_balance**2)
        total_loss += energy_balance_loss
        
        # 3. Carnot efficiency bound: COP <= T_cond / (T_cond - T_evap)
        carnot_cop = T_cond / (T_cond - T_evap + 1e-6)
        carnot_violation = torch.relu(cop_pred - 0.6 * carnot_cop)  # 60% of Carnot
        carnot_loss = torch.mean(carnot_violation**2)
        total_loss += carnot_loss
        
        # 4. COP bounds (realistic for heat pumps)
        cop_min, cop_max = 1.5, 8.0
        cop_violation = torch.relu(cop_min - cop_pred) + torch.relu(cop_pred - cop_max)
        cop_bounds_loss = torch.mean(cop_violation**2)
        total_loss += cop_bounds_loss
        
        # 5. Temperature difference constraint
        delta_T = T_cond - T_evap
        delta_T_min = 10.0  # Minimum temperature difference
        delta_T_violation = torch.relu(delta_T_min - delta_T)
        delta_T_loss = torch.mean(delta_T_violation**2)
        total_loss += delta_T_loss
        
        return total_loss

class GraphConsistencyLoss(nn.Module):
    """
    Enforce graph structure consistency and component relationships.
    """
    
    def __init__(self):
        super(GraphConsistencyLoss, self).__init__()
        
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute graph consistency loss.
        
        Args:
            graph_data: Dictionary containing:
                - 'node_features': Node features [batch_size, seq_len, n_nodes, n_node_features]
                - 'edge_features': Edge features [batch_size, seq_len, n_edges, n_edge_features]
                - 'edge_index': Edge connectivity [2, n_edges]
                - 'adjacency_matrix': Adjacency matrix [n_nodes, n_nodes]
                
        Returns:
            Graph consistency loss
        """
        
        node_features = graph_data['node_features']
        edge_features = graph_data['edge_features']
        edge_index = graph_data['edge_index']
        
        total_loss = 0.0
        
        # 1. Node-edge consistency: connected nodes should have similar properties
        src_nodes = edge_index[0]  # Source nodes
        dst_nodes = edge_index[1]  # Destination nodes
        
        # Extract temperature and pressure for connected nodes
        node_temp = node_features[:, :, :, 0]  # [batch_size, seq_len, n_nodes]
        node_pressure = node_features[:, :, :, 1]  # [batch_size, seq_len, n_nodes]
        
        # Temperature consistency along edges
        temp_src = node_temp[:, :, src_nodes]  # [batch_size, seq_len, n_edges]
        temp_dst = node_temp[:, :, dst_nodes]  # [batch_size, seq_len, n_edges]
        edge_temp_drop = edge_features[:, :, :, 3]  # Temperature drop along edge
        
        temp_consistency = (temp_src - temp_dst) - edge_temp_drop
        temp_consistency_loss = torch.mean(temp_consistency**2)
        total_loss += temp_consistency_loss
        
        # 2. Mass flow conservation at nodes
        # For each node, sum of incoming flows = sum of outgoing flows
        n_nodes = node_features.shape[2]
        mass_flow = edge_features[:, :, :, 0]  # [batch_size, seq_len, n_edges]
        
        for node_id in range(n_nodes):
            # Incoming edges
            incoming_mask = (dst_nodes == node_id)
            incoming_flow = torch.sum(mass_flow[:, :, incoming_mask], dim=-1)
            
            # Outgoing edges
            outgoing_mask = (src_nodes == node_id)
            outgoing_flow = torch.sum(mass_flow[:, :, outgoing_mask], dim=-1)
            
            # Conservation: incoming = outgoing
            flow_balance = incoming_flow - outgoing_flow
            flow_conservation_loss = torch.mean(flow_balance**2)
            total_loss += flow_conservation_loss
        
        # 3. Physical bounds on edge properties
        mass_flow_min, mass_flow_max = 0.0, 1.0  # kg/s
        mass_flow_violation = torch.relu(mass_flow_min - mass_flow) + torch.relu(mass_flow - mass_flow_max)
        mass_flow_bounds_loss = torch.mean(mass_flow_violation**2)
        total_loss += mass_flow_bounds_loss
        
        return total_loss

class AdaptivePhysicsLoss(nn.Module):
    """
    Adaptive physics-informed loss that adjusts weights during training.
    """
    
    def __init__(self, initial_weights: Dict[str, float]):
        super(AdaptivePhysicsLoss, self).__init__()
        
        self.physics_loss = PhysicsInformedLoss(**initial_weights)
        self.loss_history = {
            'data_loss': [],
            'hamiltonian_loss': [],
            'thermodynamic_loss': [],
            'heat_pump_loss': [],
            'graph_loss': []
        }
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive physics-informed loss.
        """
        
        losses = self.physics_loss(*args, **kwargs)
        
        # Store loss history for adaptation
        for key in self.loss_history:
            if key in losses:
                self.loss_history[key].append(losses[key].item())
        
        # Adapt weights based on loss magnitudes
        self.adapt_weights()
        
        return losses
    
    def adapt_weights(self):
        """
        Adapt loss weights based on relative magnitudes.
        """
        
        if len(self.loss_history['data_loss']) < 10:
            return  # Need some history
        
        # Get recent average losses
        recent_losses = {}
        for key in self.loss_history:
            recent_losses[key] = np.mean(self.loss_history[key][-10:])
        
        # Normalize weights to balance loss contributions
        total_loss = sum(recent_losses.values())
        if total_loss > 0:
            for key in recent_losses:
                if key == 'data_loss':
                    continue  # Keep data loss weight fixed
                
                # Increase weight for smaller losses to balance contributions
                target_contribution = 0.2  # Each physics loss should be ~20% of total
                current_contribution = recent_losses[key] / total_loss
                
                if current_contribution < target_contribution:
                    # Increase weight
                    weight_attr = f'lambda_{key.replace("_loss", "")}'
                    if hasattr(self.physics_loss, weight_attr):
                        current_weight = getattr(self.physics_loss, weight_attr)
                        new_weight = min(current_weight * 1.1, 2.0)  # Cap at 2.0
                        setattr(self.physics_loss, weight_attr, new_weight)

def create_physics_loss_function(config: Dict[str, float]) -> nn.Module:
    """
    Factory function to create physics-informed loss function.
    
    Args:
        config: Configuration dictionary with loss weights
        
    Returns:
        Physics-informed loss function
    """
    
    if config.get('adaptive', False):
        return AdaptivePhysicsLoss(config)
    else:
        return PhysicsInformedLoss(**config)

def validate_physics_constraints(predictions: torch.Tensor,
                                data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Validate physics constraints on predictions.
    
    Args:
        predictions: Model predictions
        data: Input data with physics information
        
    Returns:
        Dictionary of constraint violations
    """
    
    violations = {}
    
    # COP bounds
    cop_min, cop_max = 1.5, 8.0
    cop_violations = torch.sum(torch.relu(cop_min - predictions) + torch.relu(predictions - cop_max))
    violations['cop_bounds'] = cop_violations.item()
    
    # Energy conservation (if Hamiltonian data available)
    if 'hamiltonian' in data:
        H = data['hamiltonian']
        dH_dt = torch.diff(H, dim=1)
        energy_violation = torch.mean(dH_dt**2)
        violations['energy_conservation'] = energy_violation.item()
    
    # Thermodynamic bounds
    if 'temperature' in data:
        T = data['temperature']
        temp_violations = torch.sum(torch.relu(200 - T) + torch.relu(T - 500))
        violations['temperature_bounds'] = temp_violations.item()
    
    return violations

# Example usage and testing
def test_physics_loss():
    """
    Test physics-informed loss functions.
    """
    
    print("🧪 Testing Physics-Informed Loss Functions")
    print("=" * 50)
    
    # Create sample data
    batch_size, seq_len = 32, 100
    n_nodes, n_edges = 10, 11
    
    # Sample predictions and targets
    predictions = torch.randn(batch_size, seq_len, 1) * 2 + 4  # COP around 4
    targets = torch.randn(batch_size, seq_len, 1) * 2 + 4
    
    # Sample Hamiltonian data
    hamiltonian_data = {
        'hamiltonian': torch.randn(batch_size, seq_len) * 100 + 1000,
        'q': torch.randn(batch_size, seq_len, 4),
        'p': torch.randn(batch_size, seq_len, 4),
        'dt': 0.01
    }
    
    # Sample thermodynamic data
    thermodynamic_data = {
        'temperature': torch.randn(batch_size, seq_len, n_nodes) * 50 + 300,
        'pressure': torch.randn(batch_size, seq_len, n_nodes) * 5 + 10,
        'enthalpy': torch.randn(batch_size, seq_len, n_nodes) * 100 + 300,
        'entropy': torch.randn(batch_size, seq_len, n_nodes) * 0.5 + 1.0,
        'mass_flow': torch.randn(batch_size, seq_len, n_edges) * 0.05 + 0.1,
        'heat_flow': torch.randn(batch_size, seq_len, n_edges) * 2 + 5
    }
    
    # Sample heat pump data
    heat_pump_data = {
        'Q_evap': torch.randn(batch_size, seq_len) * 2 + 8,
        'Q_cond': torch.randn(batch_size, seq_len) * 2 + 12,
        'W_comp': torch.randn(batch_size, seq_len) * 1 + 3,
        'T_evap': torch.randn(batch_size, seq_len) * 20 + 280,
        'T_cond': torch.randn(batch_size, seq_len) * 20 + 320
    }
    
    # Sample graph data
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    graph_data = {
        'node_features': torch.randn(batch_size, seq_len, n_nodes, 8),
        'edge_features': torch.randn(batch_size, seq_len, n_edges, 6),
        'edge_index': edge_index,
        'adjacency_matrix': torch.eye(n_nodes)
    }
    
    # Test physics loss
    config = {
        'lambda_hamiltonian': 1.0,
        'lambda_thermodynamic': 0.5,
        'lambda_heat_pump': 0.8,
        'lambda_graph': 0.3,
        'lambda_data': 1.0
    }
    
    physics_loss = create_physics_loss_function(config)
    
    losses = physics_loss(predictions, targets, hamiltonian_data, 
                         thermodynamic_data, heat_pump_data, graph_data)
    
    print("Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test constraint validation
    violations = validate_physics_constraints(predictions, {
        'hamiltonian': hamiltonian_data['hamiltonian'],
        'temperature': thermodynamic_data['temperature']
    })
    
    print("\nConstraint Violations:")
    for key, value in violations.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✅ Physics loss functions tested successfully!")

if __name__ == "__main__":
    test_physics_loss()