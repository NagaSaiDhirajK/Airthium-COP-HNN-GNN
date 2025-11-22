#!/usr/bin/env python3
"""
Hamiltonian Mechanics for Energy Conservation in AIRTHIUM Heat Pumps
===================================================================

This module implements Hamiltonian mechanics principles for modeling energy conservation
and thermodynamic processes in heat pump systems. This is essential for training
Hamiltonian Neural Networks (HNNs) that respect physical conservation laws.

Key Features:
- Hamiltonian formulation of thermodynamic systems
- Energy conservation constraints
- Phase space coordinates (position, momentum)
- Canonical equations of motion
- Symplectic integration methods
- Conservation law verification

Sources:
- Classical Mechanics (Goldstein, Poole & Safko, 2014)
- Hamiltonian Mechanics in Thermodynamics (Callen, 1985)
- Physics-Informed Neural Networks (Raissi et al., 2019)
- Hamiltonian Neural Networks (Greydanus et al., 2019)
- Thermodynamic Hamiltonian Formalism (Eu, 1992)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class HamiltonianState:
    """Represents a state in Hamiltonian phase space."""
    state_id: str
    time_s: float
    
    # Generalized coordinates (positions)
    q_temperature_K: np.ndarray  # Temperature coordinates
    q_pressure_Pa: np.ndarray    # Pressure coordinates
    q_volume_m3: np.ndarray      # Volume coordinates
    q_entropy_J_per_K: np.ndarray # Entropy coordinates
    
    # Generalized momenta
    p_thermal_J_s_per_K: np.ndarray    # Thermal momentum
    p_mechanical_Pa_s: np.ndarray      # Mechanical momentum
    p_volumetric_Pa_m3_s: np.ndarray   # Volumetric momentum
    p_entropic_J_s_per_K2: np.ndarray  # Entropic momentum
    
    # System energy components
    kinetic_energy_J: float      # Kinetic energy
    potential_energy_J: float    # Potential energy
    internal_energy_J: float     # Internal energy
    total_hamiltonian_J: float   # Total Hamiltonian
    
    # Conservation quantities
    total_energy_J: float        # Total energy
    total_entropy_J_per_K: float # Total entropy
    total_mass_kg: float         # Total mass

@dataclass
class HamiltonianSystem:
    """Defines a Hamiltonian system for heat pump components."""
    system_id: str
    component_type: str
    
    # System parameters
    mass_kg: float
    heat_capacity_J_per_kg_K: float
    thermal_conductivity_W_per_m_K: float
    
    # Hamiltonian function coefficients
    kinetic_coefficients: np.ndarray
    potential_coefficients: np.ndarray
    coupling_matrix: np.ndarray
    
    # Conservation constraints
    energy_conservation_tolerance: float
    entropy_conservation_tolerance: float
    mass_conservation_tolerance: float

class HamiltonianMechanicsModel:
    """Model thermodynamic systems using Hamiltonian mechanics."""
    
    def __init__(self):
        """Initialize Hamiltonian mechanics model."""
        
        # Physical constants
        self.R_universal = 8.314  # J/(mol·K)
        self.k_boltzmann = 1.381e-23  # J/K
        self.avogadro = 6.022e23  # mol^-1
        
        # Component-specific Hamiltonian parameters
        self.component_hamiltonians = {
            'compressor': {
                'mass_kg': 25.0,
                'heat_capacity_J_per_kg_K': 500.0,
                'thermal_conductivity_W_per_m_K': 50.0,
                'kinetic_coefficients': np.array([1.0, 0.5, 0.3, 0.2]),
                'potential_coefficients': np.array([0.8, 1.2, 0.6, 0.4]),
                'coupling_strength': 0.15
            },
            'evaporator': {
                'mass_kg': 15.0,
                'heat_capacity_J_per_kg_K': 900.0,
                'thermal_conductivity_W_per_m_K': 200.0,
                'kinetic_coefficients': np.array([0.8, 1.0, 0.4, 0.3]),
                'potential_coefficients': np.array([1.2, 0.9, 0.8, 0.5]),
                'coupling_strength': 0.25
            },
            'condenser': {
                'mass_kg': 18.0,
                'heat_capacity_J_per_kg_K': 850.0,
                'thermal_conductivity_W_per_m_K': 180.0,
                'kinetic_coefficients': np.array([0.9, 0.8, 1.0, 0.4]),
                'potential_coefficients': np.array([1.1, 1.0, 0.7, 0.6]),
                'coupling_strength': 0.20
            },
            'expansion_valve': {
                'mass_kg': 2.0,
                'heat_capacity_J_per_kg_K': 400.0,
                'thermal_conductivity_W_per_m_K': 15.0,
                'kinetic_coefficients': np.array([0.5, 0.3, 0.4, 1.0]),
                'potential_coefficients': np.array([0.6, 0.4, 0.5, 0.9]),
                'coupling_strength': 0.10
            }
        }
        
        # Conservation tolerances
        self.conservation_tolerances = {
            'energy': 1e-6,    # Relative energy conservation
            'entropy': 1e-5,   # Relative entropy conservation
            'mass': 1e-8       # Relative mass conservation
        }
        
        # Integration parameters
        self.integration_method = 'RK45'  # Runge-Kutta method
        self.max_step = 0.1  # Maximum integration step size
        self.rtol = 1e-8     # Relative tolerance
        self.atol = 1e-10    # Absolute tolerance
    
    def create_hamiltonian_system(self, component_type: str, 
                                initial_state: Dict[str, float]) -> HamiltonianSystem:
        """Create a Hamiltonian system for a component."""
        
        if component_type not in self.component_hamiltonians:
            raise ValueError(f"Unknown component type: {component_type}")
        
        params = self.component_hamiltonians[component_type]
        
        # Create coupling matrix (4x4 for 4 coordinates)
        n_coords = 4
        coupling_matrix = np.eye(n_coords) * (1.0 - params['coupling_strength'])
        
        # Add off-diagonal coupling terms
        for i in range(n_coords):
            for j in range(n_coords):
                if i != j:
                    coupling_matrix[i, j] = params['coupling_strength'] / (n_coords - 1)
        
        return HamiltonianSystem(
            system_id=f"{component_type}_hamiltonian",
            component_type=component_type,
            mass_kg=params['mass_kg'],
            heat_capacity_J_per_kg_K=params['heat_capacity_J_per_kg_K'],
            thermal_conductivity_W_per_m_K=params['thermal_conductivity_W_per_m_K'],
            kinetic_coefficients=params['kinetic_coefficients'],
            potential_coefficients=params['potential_coefficients'],
            coupling_matrix=coupling_matrix,
            energy_conservation_tolerance=self.conservation_tolerances['energy'],
            entropy_conservation_tolerance=self.conservation_tolerances['entropy'],
            mass_conservation_tolerance=self.conservation_tolerances['mass']
        )
    
    def initialize_hamiltonian_state(self, system: HamiltonianSystem,
                                   initial_conditions: Dict[str, float]) -> HamiltonianState:
        """Initialize Hamiltonian state from thermodynamic conditions."""
        
        # Extract initial conditions
        T = initial_conditions.get('temperature_K', 298.15)
        P = initial_conditions.get('pressure_Pa', 101325.0)
        V = initial_conditions.get('volume_m3', 0.001)
        S = initial_conditions.get('entropy_J_per_K', 1000.0)
        
        # Initialize generalized coordinates (positions)
        q_temp = np.array([T])
        q_press = np.array([P])
        q_vol = np.array([V])
        q_entropy = np.array([S])
        
        # Initialize generalized momenta (derived from thermodynamic relations)
        # p = ∂L/∂q̇ where L is the Lagrangian
        p_thermal = np.array([system.mass_kg * system.heat_capacity_J_per_kg_K])
        p_mechanical = np.array([system.mass_kg * 1.0])  # Simplified
        p_volumetric = np.array([P])
        p_entropic = np.array([T])
        
        # Calculate energy components
        kinetic_energy = self._calculate_kinetic_energy(
            system, q_temp, q_press, q_vol, q_entropy,
            p_thermal, p_mechanical, p_volumetric, p_entropic
        )
        
        potential_energy = self._calculate_potential_energy(
            system, q_temp, q_press, q_vol, q_entropy
        )
        
        internal_energy = system.mass_kg * system.heat_capacity_J_per_kg_K * T
        total_hamiltonian = kinetic_energy + potential_energy
        
        # Conservation quantities
        total_energy = internal_energy + kinetic_energy + potential_energy
        total_entropy = S
        total_mass = system.mass_kg
        
        return HamiltonianState(
            state_id=f"{system.system_id}_state_0",
            time_s=0.0,
            q_temperature_K=q_temp,
            q_pressure_Pa=q_press,
            q_volume_m3=q_vol,
            q_entropy_J_per_K=q_entropy,
            p_thermal_J_s_per_K=p_thermal,
            p_mechanical_Pa_s=p_mechanical,
            p_volumetric_Pa_m3_s=p_volumetric,
            p_entropic_J_s_per_K2=p_entropic,
            kinetic_energy_J=kinetic_energy,
            potential_energy_J=potential_energy,
            internal_energy_J=internal_energy,
            total_hamiltonian_J=total_hamiltonian,
            total_energy_J=total_energy,
            total_entropy_J_per_K=total_entropy,
            total_mass_kg=total_mass
        )
    
    def hamiltonian_equations_of_motion(self, t: float, y: np.ndarray, 
                                      system: HamiltonianSystem) -> np.ndarray:
        """Compute Hamilton's equations of motion: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q."""
        
        n_coords = len(system.kinetic_coefficients)
        
        # Extract coordinates and momenta
        q = y[:n_coords]
        p = y[n_coords:]
        
        # Compute Hamiltonian gradients
        dH_dp = self._compute_hamiltonian_gradient_p(system, q, p)
        dH_dq = self._compute_hamiltonian_gradient_q(system, q, p)
        
        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        return np.concatenate([dq_dt, dp_dt])
    
    def integrate_hamiltonian_system(self, system: HamiltonianSystem,
                                   initial_state: HamiltonianState,
                                   time_span: Tuple[float, float],
                                   external_forces: Optional[Dict[str, Callable]] = None) -> List[HamiltonianState]:
        """Integrate Hamiltonian system using symplectic methods."""
        
        t_start, t_end = time_span
        n_coords = len(system.kinetic_coefficients)
        
        # Initial conditions
        q0 = np.concatenate([
            initial_state.q_temperature_K,
            initial_state.q_pressure_Pa,
            initial_state.q_volume_m3,
            initial_state.q_entropy_J_per_K
        ])
        p0 = np.concatenate([
            initial_state.p_thermal_J_s_per_K,
            initial_state.p_mechanical_Pa_s,
            initial_state.p_volumetric_Pa_m3_s,
            initial_state.p_entropic_J_s_per_K2
        ])
        
        y0 = np.concatenate([q0, p0])
        
        # Define equations of motion with external forces
        def equations_with_forces(t, y):
            dydt = self.hamiltonian_equations_of_motion(t, y, system)
            
            # Add external forces if provided
            if external_forces:
                force_contributions = np.zeros_like(dydt)
                for force_name, force_func in external_forces.items():
                    force_contributions += force_func(t, y)
                dydt += force_contributions
            
            return dydt
        
        # Integrate using scipy
        solution = solve_ivp(
            equations_with_forces,
            time_span,
            y0,
            method=self.integration_method,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True
        )
        
        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")
        
        # Convert solution to HamiltonianState objects
        states = []
        for i, t in enumerate(solution.t):
            y = solution.y[:, i]
            q = y[:n_coords]
            p = y[n_coords:]
            
            # Calculate energies
            kinetic_energy = self._calculate_kinetic_energy(
                system, q[:1], q[1:2], q[2:3], q[3:4],
                p[:1], p[1:2], p[2:3], p[3:4]
            )
            
            potential_energy = self._calculate_potential_energy(
                system, q[:1], q[1:2], q[2:3], q[3:4]
            )
            
            internal_energy = system.mass_kg * system.heat_capacity_J_per_kg_K * q[0]
            total_hamiltonian = kinetic_energy + potential_energy
            total_energy = internal_energy + kinetic_energy + potential_energy
            
            state = HamiltonianState(
                state_id=f"{system.system_id}_state_{i}",
                time_s=t,
                q_temperature_K=q[:1],
                q_pressure_Pa=q[1:2],
                q_volume_m3=q[2:3],
                q_entropy_J_per_K=q[3:4],
                p_thermal_J_s_per_K=p[:1],
                p_mechanical_Pa_s=p[1:2],
                p_volumetric_Pa_m3_s=p[2:3],
                p_entropic_J_s_per_K2=p[3:4],
                kinetic_energy_J=kinetic_energy,
                potential_energy_J=potential_energy,
                internal_energy_J=internal_energy,
                total_hamiltonian_J=total_hamiltonian,
                total_energy_J=total_energy,
                total_entropy_J_per_K=q[3],
                total_mass_kg=system.mass_kg
            )
            states.append(state)
        
        return states
    
    def verify_conservation_laws(self, states: List[HamiltonianState]) -> Dict[str, float]:
        """Verify conservation of energy, entropy, and mass."""
        
        if len(states) < 2:
            return {'energy_drift': 0.0, 'entropy_drift': 0.0, 'mass_drift': 0.0}
        
        # Initial values
        initial_energy = states[0].total_energy_J
        initial_entropy = states[0].total_entropy_J_per_K
        initial_mass = states[0].total_mass_kg
        
        # Calculate drifts
        energy_values = [state.total_energy_J for state in states]
        entropy_values = [state.total_entropy_J_per_K for state in states]
        mass_values = [state.total_mass_kg for state in states]
        
        energy_drift = (max(energy_values) - min(energy_values)) / abs(initial_energy)
        entropy_drift = (max(entropy_values) - min(entropy_values)) / abs(initial_entropy)
        mass_drift = (max(mass_values) - min(mass_values)) / abs(initial_mass)
        
        return {
            'energy_drift': energy_drift,
            'entropy_drift': entropy_drift,
            'mass_drift': mass_drift,
            'energy_conservation_satisfied': energy_drift < self.conservation_tolerances['energy'],
            'entropy_conservation_satisfied': entropy_drift < self.conservation_tolerances['entropy'],
            'mass_conservation_satisfied': mass_drift < self.conservation_tolerances['mass']
        }
    
    def _calculate_kinetic_energy(self, system: HamiltonianSystem,
                                q_temp: np.ndarray, q_press: np.ndarray,
                                q_vol: np.ndarray, q_entropy: np.ndarray,
                                p_temp: np.ndarray, p_press: np.ndarray,
                                p_vol: np.ndarray, p_entropy: np.ndarray) -> float:
        """Calculate kinetic energy from generalized momenta."""
        
        # Kinetic energy: T = (1/2) * p^T * M^(-1) * p
        # where M is the mass matrix (generalized inertia)
        
        p = np.array([p_temp[0], p_press[0], p_vol[0], p_entropy[0]])
        
        # Mass matrix (diagonal for simplicity)
        M_inv = np.diag(1.0 / (system.kinetic_coefficients * system.mass_kg))
        
        kinetic_energy = 0.5 * np.dot(p, np.dot(M_inv, p))
        
        return kinetic_energy
    
    def _calculate_potential_energy(self, system: HamiltonianSystem,
                                  q_temp: np.ndarray, q_press: np.ndarray,
                                  q_vol: np.ndarray, q_entropy: np.ndarray) -> float:
        """Calculate potential energy from generalized coordinates."""
        
        # Potential energy: V = (1/2) * q^T * K * q
        # where K is the stiffness matrix
        
        q = np.array([q_temp[0], q_press[0], q_vol[0], q_entropy[0]])
        
        # Normalize coordinates for numerical stability
        q_norm = q / np.array([300.0, 1e6, 0.001, 1000.0])  # Typical scales
        
        # Stiffness matrix
        K = np.diag(system.potential_coefficients) + system.coupling_matrix
        
        potential_energy = 0.5 * np.dot(q_norm, np.dot(K, q_norm))
        
        return potential_energy
    
    def _compute_hamiltonian_gradient_p(self, system: HamiltonianSystem,
                                      q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute ∂H/∂p for Hamilton's equations."""
        
        # For kinetic energy T = (1/2) * p^T * M^(-1) * p
        # ∂T/∂p = M^(-1) * p
        
        M_inv = np.diag(1.0 / (system.kinetic_coefficients * system.mass_kg))
        dH_dp = np.dot(M_inv, p)
        
        return dH_dp
    
    def _compute_hamiltonian_gradient_q(self, system: HamiltonianSystem,
                                      q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute ∂H/∂q for Hamilton's equations."""
        
        # For potential energy V = (1/2) * q^T * K * q
        # ∂V/∂q = K * q
        
        # Normalize coordinates
        q_norm = q / np.array([300.0, 1e6, 0.001, 1000.0])
        normalization = np.array([300.0, 1e6, 0.001, 1000.0])
        
        K = np.diag(system.potential_coefficients) + system.coupling_matrix
        dH_dq = np.dot(K, q_norm) / normalization
        
        return dH_dq
    
    def generate_hamiltonian_scenarios(self, num_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Generate diverse Hamiltonian mechanics scenarios."""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Random component type
            component_type = np.random.choice(list(self.component_hamiltonians.keys()))
            
            # Create Hamiltonian system
            system = self.create_hamiltonian_system(component_type, {})
            
            # Random initial conditions
            initial_conditions = {
                'temperature_K': np.random.uniform(250.0, 400.0),
                'pressure_Pa': np.random.uniform(50000, 2000000),
                'volume_m3': np.random.uniform(0.0001, 0.01),
                'entropy_J_per_K': np.random.uniform(500.0, 2000.0)
            }
            
            # Initialize state
            initial_state = self.initialize_hamiltonian_state(system, initial_conditions)
            
            # Random external forces
            external_forces = None
            if np.random.random() > 0.5:
                def heat_input(t, y):
                    force = np.zeros_like(y)
                    force[0] = 100.0 * np.sin(0.1 * t)  # Sinusoidal heat input
                    return force
                external_forces = {'heat_input': heat_input}
            
            # Integrate system
            time_span = (0.0, np.random.uniform(100.0, 1000.0))
            states = self.integrate_hamiltonian_system(
                system, initial_state, time_span, external_forces
            )
            
            # Verify conservation laws
            conservation_check = self.verify_conservation_laws(states)
            
            scenarios.append({
                'scenario_id': i,
                'component_type': component_type,
                'system_parameters': asdict(system),
                'initial_conditions': initial_conditions,
                'time_span': time_span,
                'has_external_forces': external_forces is not None,
                'trajectory': [asdict(state) for state in states[::10]],  # Subsample
                'conservation_check': conservation_check,
                'final_hamiltonian': states[-1].total_hamiltonian_J,
                'energy_drift': conservation_check['energy_drift']
            })
        
        return scenarios
    
    def save_hamiltonian_data(self, scenarios: List[Dict[str, Any]], output_path: str) -> None:
        """Save Hamiltonian mechanics data to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            return obj
        
        # Convert scenarios
        serializable_scenarios = []
        for scenario in scenarios:
            serializable_scenario = {}
            for key, value in scenario.items():
                serializable_scenario[key] = convert_numpy(value)
            serializable_scenarios.append(serializable_scenario)
        
        data = {
            'hamiltonian_scenarios': serializable_scenarios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'component_hamiltonians': self.component_hamiltonians,
                'conservation_tolerances': self.conservation_tolerances,
                'integration_method': self.integration_method,
                'physical_constants': {
                    'R_universal': self.R_universal,
                    'k_boltzmann': self.k_boltzmann,
                    'avogadro': self.avogadro
                }
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved Hamiltonian mechanics data to {output_path}")


if __name__ == "__main__":
    # Test Hamiltonian mechanics models
    print("🔄 Testing Hamiltonian Mechanics Models")
    
    # Initialize model
    hamiltonian_model = HamiltonianMechanicsModel()
    
    # Create test system
    system = hamiltonian_model.create_hamiltonian_system('compressor', {})
    
    print(f"\n⚙️ Hamiltonian System Created:")
    print(f"Component: {system.component_type}")
    print(f"Mass: {system.mass_kg} kg")
    print(f"Heat capacity: {system.heat_capacity_J_per_kg_K} J/(kg·K)")
    
    # Initialize state
    initial_conditions = {
        'temperature_K': 353.15,
        'pressure_Pa': 1800000,
        'volume_m3': 0.001,
        'entropy_J_per_K': 1200.0
    }
    
    initial_state = hamiltonian_model.initialize_hamiltonian_state(system, initial_conditions)
    
    print(f"\n📊 Initial Hamiltonian State:")
    print(f"Temperature: {initial_state.q_temperature_K[0]:.1f} K")
    print(f"Pressure: {initial_state.q_pressure_Pa[0]:.0f} Pa")
    print(f"Total Hamiltonian: {initial_state.total_hamiltonian_J:.2f} J")
    
    # Integrate system
    time_span = (0.0, 300.0)
    states = hamiltonian_model.integrate_hamiltonian_system(system, initial_state, time_span)
    
    print(f"\n🔄 Integration Results:")
    print(f"Integrated {len(states)} time steps")
    print(f"Final temperature: {states[-1].q_temperature_K[0]:.1f} K")
    print(f"Final Hamiltonian: {states[-1].total_hamiltonian_J:.2f} J")
    
    # Check conservation laws
    conservation = hamiltonian_model.verify_conservation_laws(states)
    
    print(f"\n⚖️ Conservation Laws:")
    print(f"Energy drift: {conservation['energy_drift']:.2e}")
    print(f"Energy conserved: {conservation['energy_conservation_satisfied']}")
    print(f"Entropy drift: {conservation['entropy_drift']:.2e}")
    print(f"Mass drift: {conservation['mass_drift']:.2e}")
    
    # Generate scenarios
    print(f"\n🎲 Generating Hamiltonian scenarios...")
    scenarios = hamiltonian_model.generate_hamiltonian_scenarios(3)
    print(f"Generated {len(scenarios)} Hamiltonian scenarios")
    
    # Save test data
    hamiltonian_model.save_hamiltonian_data(scenarios, "hamiltonian_mechanics_data.json")
    
    print("\n✅ Hamiltonian mechanics models testing complete!")