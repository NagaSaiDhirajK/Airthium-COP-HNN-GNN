#!/usr/bin/env python3
"""
Phase-Space Trajectories for Thermodynamic Processes in AIRTHIUM Heat Pumps
===========================================================================

This module implements phase-space trajectory modeling for thermodynamic processes
in heat pump systems. Phase-space trajectories are essential for training
Hamiltonian Neural Networks and understanding system dynamics.

Key Features:
- Phase-space coordinate systems
- Thermodynamic trajectory generation
- Poincaré sections and maps
- Lyapunov exponent calculation
- Attractor identification
- Bifurcation analysis

Sources:
- Nonlinear Dynamics and Chaos (Strogatz, 2014)
- Phase Space Methods in Thermodynamics (Grmela, 1997)
- Dynamical Systems in Thermodynamics (Eu, 1992)
- Chaos in Heat Transfer (Vadasz, 2005)
- Phase Space Reconstruction (Takens, 1981)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from datetime import datetime

@dataclass
class PhaseSpacePoint:
    """Represents a point in thermodynamic phase space."""
    point_id: str
    time_s: float
    
    # Thermodynamic coordinates
    temperature_K: float
    pressure_Pa: float
    volume_m3: float
    entropy_J_per_K: float
    enthalpy_J: float
    
    # Velocity coordinates (time derivatives)
    dT_dt_K_per_s: float
    dP_dt_Pa_per_s: float
    dV_dt_m3_per_s: float
    dS_dt_J_per_K_per_s: float
    dH_dt_J_per_s: float
    
    # System state
    component_type: str
    operating_mode: str
    
    # Derived quantities
    kinetic_energy_density: float
    potential_energy_density: float
    phase_space_distance: float

@dataclass
class PhaseSpaceTrajectory:
    """Represents a complete trajectory in phase space."""
    trajectory_id: str
    component_type: str
    process_type: str  # 'compression', 'expansion', 'heating', 'cooling', 'mixed'
    
    # Trajectory data
    points: List[PhaseSpacePoint]
    time_span: Tuple[float, float]
    
    # Trajectory characteristics
    is_periodic: bool
    period_s: Optional[float]
    is_chaotic: bool
    lyapunov_exponent: float
    
    # Geometric properties
    trajectory_length: float
    bounding_box: Dict[str, Tuple[float, float]]
    centroid: Dict[str, float]
    
    # Thermodynamic properties
    entropy_production_rate: float
    energy_dissipation_rate: float
    irreversibility_measure: float

@dataclass
class PoincareSection:
    """Represents a Poincaré section of phase space."""
    section_id: str
    section_plane: Dict[str, float]  # Defines the cutting plane
    intersection_points: List[PhaseSpacePoint]
    
    # Section analysis
    return_map: List[Tuple[float, float]]
    fixed_points: List[PhaseSpacePoint]
    stability_analysis: Dict[str, float]

class PhaseSpaceTrajectoriesModel:
    """Model phase-space trajectories for thermodynamic processes."""
    
    def __init__(self):
        """Initialize phase-space trajectories model."""
        
        # Physical constants
        self.R_universal = 8.314  # J/(mol·K)
        self.k_boltzmann = 1.381e-23  # J/K
        
        # Component-specific dynamics parameters
        self.component_dynamics = {
            'compressor': {
                'characteristic_time_s': 10.0,
                'nonlinearity_strength': 0.3,
                'damping_coefficient': 0.1,
                'forcing_amplitude': 0.2,
                'natural_frequency_Hz': 0.1,
                'chaos_parameter': 0.15
            },
            'evaporator': {
                'characteristic_time_s': 30.0,
                'nonlinearity_strength': 0.2,
                'damping_coefficient': 0.05,
                'forcing_amplitude': 0.15,
                'natural_frequency_Hz': 0.05,
                'chaos_parameter': 0.08
            },
            'condenser': {
                'characteristic_time_s': 25.0,
                'nonlinearity_strength': 0.25,
                'damping_coefficient': 0.08,
                'forcing_amplitude': 0.18,
                'natural_frequency_Hz': 0.06,
                'chaos_parameter': 0.12
            },
            'expansion_valve': {
                'characteristic_time_s': 5.0,
                'nonlinearity_strength': 0.4,
                'damping_coefficient': 0.2,
                'forcing_amplitude': 0.3,
                'natural_frequency_Hz': 0.2,
                'chaos_parameter': 0.25
            }
        }
        
        # Phase space dimensions and scales
        self.phase_space_scales = {
            'temperature_K': (200.0, 400.0),
            'pressure_Pa': (50000, 2000000),
            'volume_m3': (0.0001, 0.01),
            'entropy_J_per_K': (500.0, 3000.0),
            'enthalpy_J': (100000, 500000)
        }
        
        # Trajectory analysis parameters
        self.analysis_params = {
            'min_trajectory_length': 100,
            'periodicity_tolerance': 0.05,
            'chaos_threshold': 0.01,
            'lyapunov_time_horizon': 1000.0,
            'embedding_dimension': 3,
            'time_delay': 1
        }
    
    def generate_thermodynamic_trajectory(self, component_type: str,
                                        initial_conditions: Dict[str, float],
                                        process_parameters: Dict[str, float],
                                        time_span: Tuple[float, float],
                                        external_forcing: Optional[Callable] = None) -> PhaseSpaceTrajectory:
        """Generate a phase-space trajectory for a thermodynamic process."""
        
        if component_type not in self.component_dynamics:
            raise ValueError(f"Unknown component type: {component_type}")
        
        dynamics = self.component_dynamics[component_type]
        t_start, t_end = time_span
        
        # Define the dynamical system
        def thermodynamic_dynamics(t, y):
            T, P, V, S, H, dT_dt, dP_dt, dV_dt, dS_dt, dH_dt = y
            
            # Nonlinear thermodynamic equations
            # Based on modified van der Pol oscillator with thermodynamic coupling
            
            # Temperature dynamics
            d2T_dt2 = (dynamics['nonlinearity_strength'] * (1 - T**2/300**2) * dT_dt 
                      - (2*np.pi*dynamics['natural_frequency_Hz'])**2 * (T - 298.15)
                      + dynamics['forcing_amplitude'] * np.sin(2*np.pi*dynamics['natural_frequency_Hz']*t))
            
            # Pressure dynamics (coupled to temperature)
            d2P_dt2 = (dynamics['nonlinearity_strength'] * (1 - P**2/1e12) * dP_dt
                      - (2*np.pi*dynamics['natural_frequency_Hz'])**2 * (P - 101325)
                      + 0.1 * dT_dt)  # Coupling term
            
            # Volume dynamics (inverse coupling to pressure)
            d2V_dt2 = (-dynamics['nonlinearity_strength'] * dV_dt
                      - (2*np.pi*dynamics['natural_frequency_Hz'])**2 * (V - 0.001)
                      - 0.001 * dP_dt)  # Inverse coupling
            
            # Entropy dynamics (always increasing for irreversible processes)
            d2S_dt2 = (0.1 * abs(dT_dt) + 0.01 * abs(dP_dt) + 0.001 * abs(dV_dt))
            
            # Enthalpy dynamics (derived from other variables)
            d2H_dt2 = (T * d2S_dt2 + dT_dt * dS_dt + V * d2P_dt2 + dV_dt * dP_dt)
            
            # Add external forcing if provided
            if external_forcing:
                force = external_forcing(t, y)
                d2T_dt2 += force.get('temperature', 0.0)
                d2P_dt2 += force.get('pressure', 0.0)
                d2V_dt2 += force.get('volume', 0.0)
                d2S_dt2 += force.get('entropy', 0.0)
                d2H_dt2 += force.get('enthalpy', 0.0)
            
            return [dT_dt, dP_dt, dV_dt, dS_dt, dH_dt,
                   d2T_dt2, d2P_dt2, d2V_dt2, d2S_dt2, d2H_dt2]
        
        # Initial conditions
        T0 = initial_conditions.get('temperature_K', 298.15)
        P0 = initial_conditions.get('pressure_Pa', 101325.0)
        V0 = initial_conditions.get('volume_m3', 0.001)
        S0 = initial_conditions.get('entropy_J_per_K', 1000.0)
        H0 = initial_conditions.get('enthalpy_J', 300000.0)
        
        # Initial velocities (small random perturbations)
        dT0 = np.random.normal(0, 0.1)
        dP0 = np.random.normal(0, 100)
        dV0 = np.random.normal(0, 1e-6)
        dS0 = np.random.normal(0, 0.01)
        dH0 = np.random.normal(0, 10)
        
        y0 = [T0, P0, V0, S0, H0, dT0, dP0, dV0, dS0, dH0]
        
        # Integrate the system
        solution = solve_ivp(
            thermodynamic_dynamics,
            time_span,
            y0,
            method='RK45',
            max_step=1.0,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True
        )
        
        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")
        
        # Convert solution to PhaseSpacePoint objects
        points = []
        for i, t in enumerate(solution.t):
            y = solution.y[:, i]
            T, P, V, S, H, dT_dt, dP_dt, dV_dt, dS_dt, dH_dt = y
            
            # Calculate derived quantities
            kinetic_energy_density = 0.5 * (dT_dt**2 + dP_dt**2 + dV_dt**2 + dS_dt**2 + dH_dt**2)
            potential_energy_density = 0.5 * ((T-298.15)**2 + (P-101325)**2 + (V-0.001)**2)
            
            # Phase space distance from origin
            phase_space_distance = np.sqrt(T**2 + P**2 + V**2 + S**2 + H**2)
            
            point = PhaseSpacePoint(
                point_id=f"point_{i}",
                time_s=t,
                temperature_K=T,
                pressure_Pa=P,
                volume_m3=V,
                entropy_J_per_K=S,
                enthalpy_J=H,
                dT_dt_K_per_s=dT_dt,
                dP_dt_Pa_per_s=dP_dt,
                dV_dt_m3_per_s=dV_dt,
                dS_dt_J_per_K_per_s=dS_dt,
                dH_dt_J_per_s=dH_dt,
                component_type=component_type,
                operating_mode=process_parameters.get('mode', 'normal'),
                kinetic_energy_density=kinetic_energy_density,
                potential_energy_density=potential_energy_density,
                phase_space_distance=phase_space_distance
            )
            points.append(point)
        
        # Analyze trajectory properties
        trajectory_analysis = self._analyze_trajectory(points)
        
        return PhaseSpaceTrajectory(
            trajectory_id=f"{component_type}_trajectory_{len(points)}",
            component_type=component_type,
            process_type=process_parameters.get('process_type', 'mixed'),
            points=points,
            time_span=time_span,
            is_periodic=trajectory_analysis['is_periodic'],
            period_s=trajectory_analysis['period_s'],
            is_chaotic=trajectory_analysis['is_chaotic'],
            lyapunov_exponent=trajectory_analysis['lyapunov_exponent'],
            trajectory_length=trajectory_analysis['trajectory_length'],
            bounding_box=trajectory_analysis['bounding_box'],
            centroid=trajectory_analysis['centroid'],
            entropy_production_rate=trajectory_analysis['entropy_production_rate'],
            energy_dissipation_rate=trajectory_analysis['energy_dissipation_rate'],
            irreversibility_measure=trajectory_analysis['irreversibility_measure']
        )
    
    def create_poincare_section(self, trajectory: PhaseSpaceTrajectory,
                              section_variable: str,
                              section_value: float,
                              tolerance: float = 0.01) -> PoincareSection:
        """Create a Poincaré section of the trajectory."""
        
        intersection_points = []
        
        # Find intersections with the section plane
        for i in range(len(trajectory.points) - 1):
            point1 = trajectory.points[i]
            point2 = trajectory.points[i + 1]
            
            # Get variable values
            val1 = getattr(point1, section_variable)
            val2 = getattr(point2, section_variable)
            
            # Check for intersection
            if (val1 <= section_value <= val2) or (val2 <= section_value <= val1):
                # Linear interpolation to find exact intersection
                if abs(val2 - val1) > 1e-10:
                    alpha = (section_value - val1) / (val2 - val1)
                    
                    # Interpolate all coordinates
                    interp_point = PhaseSpacePoint(
                        point_id=f"poincare_{len(intersection_points)}",
                        time_s=point1.time_s + alpha * (point2.time_s - point1.time_s),
                        temperature_K=point1.temperature_K + alpha * (point2.temperature_K - point1.temperature_K),
                        pressure_Pa=point1.pressure_Pa + alpha * (point2.pressure_Pa - point1.pressure_Pa),
                        volume_m3=point1.volume_m3 + alpha * (point2.volume_m3 - point1.volume_m3),
                        entropy_J_per_K=point1.entropy_J_per_K + alpha * (point2.entropy_J_per_K - point1.entropy_J_per_K),
                        enthalpy_J=point1.enthalpy_J + alpha * (point2.enthalpy_J - point1.enthalpy_J),
                        dT_dt_K_per_s=point1.dT_dt_K_per_s + alpha * (point2.dT_dt_K_per_s - point1.dT_dt_K_per_s),
                        dP_dt_Pa_per_s=point1.dP_dt_Pa_per_s + alpha * (point2.dP_dt_Pa_per_s - point1.dP_dt_Pa_per_s),
                        dV_dt_m3_per_s=point1.dV_dt_m3_per_s + alpha * (point2.dV_dt_m3_per_s - point1.dV_dt_m3_per_s),
                        dS_dt_J_per_K_per_s=point1.dS_dt_J_per_K_per_s + alpha * (point2.dS_dt_J_per_K_per_s - point1.dS_dt_J_per_K_per_s),
                        dH_dt_J_per_s=point1.dH_dt_J_per_s + alpha * (point2.dH_dt_J_per_s - point1.dH_dt_J_per_s),
                        component_type=point1.component_type,
                        operating_mode=point1.operating_mode,
                        kinetic_energy_density=point1.kinetic_energy_density + alpha * (point2.kinetic_energy_density - point1.kinetic_energy_density),
                        potential_energy_density=point1.potential_energy_density + alpha * (point2.potential_energy_density - point1.potential_energy_density),
                        phase_space_distance=point1.phase_space_distance + alpha * (point2.phase_space_distance - point1.phase_space_distance)
                    )
                    
                    intersection_points.append(interp_point)
        
        # Create return map
        return_map = []
        if len(intersection_points) > 1:
            for i in range(len(intersection_points) - 1):
                x_n = intersection_points[i].temperature_K  # Use temperature as return map variable
                x_n1 = intersection_points[i + 1].temperature_K
                return_map.append((x_n, x_n1))
        
        # Find fixed points (approximate)
        fixed_points = []
        for point in intersection_points:
            # A fixed point satisfies f(x) = x in the return map
            # Here we use a simple criterion: points where velocity is small
            if abs(point.dT_dt_K_per_s) < 0.1:
                fixed_points.append(point)
        
        # Stability analysis
        stability_analysis = self._analyze_poincare_stability(return_map)
        
        return PoincareSection(
            section_id=f"poincare_{section_variable}_{section_value}",
            section_plane={section_variable: section_value},
            intersection_points=intersection_points,
            return_map=return_map,
            fixed_points=fixed_points,
            stability_analysis=stability_analysis
        )
    
    def calculate_lyapunov_exponent(self, trajectory: PhaseSpaceTrajectory,
                                  embedding_dim: int = 3) -> float:
        """Calculate the largest Lyapunov exponent of the trajectory."""
        
        if len(trajectory.points) < 100:
            return 0.0
        
        # Extract time series (use temperature as primary variable)
        time_series = np.array([point.temperature_K for point in trajectory.points])
        dt = trajectory.points[1].time_s - trajectory.points[0].time_s
        
        # Time delay embedding
        tau = self.analysis_params['time_delay']
        embedded_data = self._time_delay_embedding(time_series, embedding_dim, tau)
        
        if embedded_data.shape[0] < 50:
            return 0.0
        
        # Calculate Lyapunov exponent using the method of Wolf et al.
        lyapunov_sum = 0.0
        num_iterations = 0
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embedded_data)
        
        for i in range(0, len(embedded_data) - 10, 10):
            # Find nearest neighbor
            distances, indices = nbrs.kneighbors([embedded_data[i]])
            
            if len(indices[0]) > 1:
                nearest_idx = indices[0][1]  # Skip self (index 0)
                
                # Calculate initial separation
                initial_separation = distances[0][1]
                
                if initial_separation > 0 and i + 10 < len(embedded_data) and nearest_idx + 10 < len(embedded_data):
                    # Calculate separation after time evolution
                    final_separation = np.linalg.norm(
                        embedded_data[i + 10] - embedded_data[nearest_idx + 10]
                    )
                    
                    if final_separation > 0:
                        lyapunov_sum += np.log(final_separation / initial_separation)
                        num_iterations += 1
        
        if num_iterations > 0:
            lyapunov_exponent = lyapunov_sum / (num_iterations * 10 * dt)
        else:
            lyapunov_exponent = 0.0
        
        return lyapunov_exponent
    
    def _analyze_trajectory(self, points: List[PhaseSpacePoint]) -> Dict[str, Any]:
        """Analyze trajectory properties."""
        
        if len(points) < 10:
            return {
                'is_periodic': False,
                'period_s': None,
                'is_chaotic': False,
                'lyapunov_exponent': 0.0,
                'trajectory_length': 0.0,
                'bounding_box': {},
                'centroid': {},
                'entropy_production_rate': 0.0,
                'energy_dissipation_rate': 0.0,
                'irreversibility_measure': 0.0
            }
        
        # Extract coordinates
        temperatures = [p.temperature_K for p in points]
        pressures = [p.pressure_Pa for p in points]
        volumes = [p.volume_m3 for p in points]
        entropies = [p.entropy_J_per_K for p in points]
        times = [p.time_s for p in points]
        
        # Check periodicity
        is_periodic, period_s = self._detect_periodicity(temperatures, times)
        
        # Calculate trajectory length
        trajectory_length = 0.0
        for i in range(len(points) - 1):
            dx = points[i+1].temperature_K - points[i].temperature_K
            dy = points[i+1].pressure_Pa - points[i].pressure_Pa
            dz = points[i+1].volume_m3 - points[i].volume_m3
            trajectory_length += np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Bounding box
        bounding_box = {
            'temperature_K': (min(temperatures), max(temperatures)),
            'pressure_Pa': (min(pressures), max(pressures)),
            'volume_m3': (min(volumes), max(volumes)),
            'entropy_J_per_K': (min(entropies), max(entropies))
        }
        
        # Centroid
        centroid = {
            'temperature_K': np.mean(temperatures),
            'pressure_Pa': np.mean(pressures),
            'volume_m3': np.mean(volumes),
            'entropy_J_per_K': np.mean(entropies)
        }
        
        # Entropy production rate
        entropy_production_rate = (entropies[-1] - entropies[0]) / (times[-1] - times[0])
        
        # Energy dissipation rate (simplified)
        kinetic_energies = [p.kinetic_energy_density for p in points]
        energy_dissipation_rate = abs(kinetic_energies[-1] - kinetic_energies[0]) / (times[-1] - times[0])
        
        # Irreversibility measure
        irreversibility_measure = entropy_production_rate / max(abs(np.mean(entropies)), 1.0)
        
        # Simple chaos detection (based on sensitivity to initial conditions)
        lyapunov_exponent = self._estimate_lyapunov_simple(temperatures, times)
        is_chaotic = lyapunov_exponent > self.analysis_params['chaos_threshold']
        
        return {
            'is_periodic': is_periodic,
            'period_s': period_s,
            'is_chaotic': is_chaotic,
            'lyapunov_exponent': lyapunov_exponent,
            'trajectory_length': trajectory_length,
            'bounding_box': bounding_box,
            'centroid': centroid,
            'entropy_production_rate': entropy_production_rate,
            'energy_dissipation_rate': energy_dissipation_rate,
            'irreversibility_measure': irreversibility_measure
        }
    
    def _detect_periodicity(self, signal: List[float], times: List[float]) -> Tuple[bool, Optional[float]]:
        """Detect if the signal is periodic."""
        
        if len(signal) < 20:
            return False, None
        
        # Use autocorrelation to detect periodicity
        signal_array = np.array(signal)
        signal_centered = signal_array - np.mean(signal_array)
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr[1:], height=self.analysis_params['periodicity_tolerance'])
        
        if len(peaks) > 0:
            # First significant peak indicates period
            period_samples = peaks[0] + 1
            dt = times[1] - times[0]
            period_s = period_samples * dt
            return True, period_s
        
        return False, None
    
    def _estimate_lyapunov_simple(self, signal: List[float], times: List[float]) -> float:
        """Simple estimation of Lyapunov exponent."""
        
        if len(signal) < 50:
            return 0.0
        
        signal_array = np.array(signal)
        dt = times[1] - times[0]
        
        # Calculate local divergence rates
        divergence_rates = []
        
        for i in range(10, len(signal_array) - 10):
            # Local derivative
            local_derivative = (signal_array[i+1] - signal_array[i-1]) / (2 * dt)
            
            if abs(local_derivative) > 1e-10:
                # Estimate local stretching
                stretching = abs(local_derivative)
                divergence_rates.append(np.log(stretching))
        
        if len(divergence_rates) > 0:
            return np.mean(divergence_rates) / dt
        else:
            return 0.0
    
    def _time_delay_embedding(self, time_series: np.ndarray, 
                            embedding_dim: int, tau: int) -> np.ndarray:
        """Create time delay embedding of the time series."""
        
        N = len(time_series)
        M = N - (embedding_dim - 1) * tau
        
        if M <= 0:
            return np.array([])
        
        embedded = np.zeros((M, embedding_dim))
        
        for i in range(M):
            for j in range(embedding_dim):
                embedded[i, j] = time_series[i + j * tau]
        
        return embedded
    
    def _analyze_poincare_stability(self, return_map: List[Tuple[float, float]]) -> Dict[str, float]:
        """Analyze stability of Poincaré return map."""
        
        if len(return_map) < 5:
            return {'stability_measure': 0.0, 'map_dimension': 0.0}
        
        # Calculate correlation dimension of return map
        points = np.array(return_map)
        
        if points.shape[0] < 10:
            return {'stability_measure': 0.0, 'map_dimension': 0.0}
        
        # Simple stability measure: variance of return map
        x_values = points[:, 0]
        y_values = points[:, 1]
        
        stability_measure = np.var(y_values - x_values)  # Deviation from identity map
        
        # Estimate fractal dimension (simplified)
        distances = pdist(points)
        if len(distances) > 0:
            map_dimension = -np.polyfit(np.log(np.sort(distances)[1:10]), 
                                      np.log(np.arange(1, 10)), 1)[0]
        else:
            map_dimension = 0.0
        
        return {
            'stability_measure': stability_measure,
            'map_dimension': map_dimension
        }
    
    def generate_phase_space_scenarios(self, num_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Generate diverse phase-space trajectory scenarios."""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Random component type
            component_type = np.random.choice(list(self.component_dynamics.keys()))
            
            # Random initial conditions
            initial_conditions = {
                'temperature_K': np.random.uniform(250.0, 400.0),
                'pressure_Pa': np.random.uniform(50000, 2000000),
                'volume_m3': np.random.uniform(0.0001, 0.01),
                'entropy_J_per_K': np.random.uniform(500.0, 3000.0),
                'enthalpy_J': np.random.uniform(100000, 500000)
            }
            
            # Random process parameters
            process_parameters = {
                'process_type': np.random.choice(['compression', 'expansion', 'heating', 'cooling', 'mixed']),
                'mode': np.random.choice(['normal', 'transient', 'fault'])
            }
            
            # Random time span
            time_span = (0.0, np.random.uniform(100.0, 1000.0))
            
            # Random external forcing
            external_forcing = None
            if np.random.random() > 0.7:
                def random_forcing(t, y):
                    return {
                        'temperature': 10.0 * np.sin(0.1 * t),
                        'pressure': 1000.0 * np.cos(0.05 * t),
                        'volume': 0.0001 * np.sin(0.2 * t),
                        'entropy': 1.0 * np.sin(0.01 * t),
                        'enthalpy': 100.0 * np.cos(0.02 * t)
                    }
                external_forcing = random_forcing
            
            # Generate trajectory
            try:
                trajectory = self.generate_thermodynamic_trajectory(
                    component_type, initial_conditions, process_parameters, time_span, external_forcing
                )
                
                # Create Poincaré section
                poincare_section = self.create_poincare_section(
                    trajectory, 'temperature_K', np.mean([p.temperature_K for p in trajectory.points])
                )
                
                # Calculate Lyapunov exponent
                lyapunov_exp = self.calculate_lyapunov_exponent(trajectory)
                
                scenarios.append({
                    'scenario_id': i,
                    'component_type': component_type,
                    'initial_conditions': initial_conditions,
                    'process_parameters': process_parameters,
                    'time_span': time_span,
                    'has_external_forcing': external_forcing is not None,
                    'trajectory_summary': {
                        'num_points': len(trajectory.points),
                        'is_periodic': trajectory.is_periodic,
                        'period_s': trajectory.period_s,
                        'is_chaotic': trajectory.is_chaotic,
                        'lyapunov_exponent': trajectory.lyapunov_exponent,
                        'trajectory_length': trajectory.trajectory_length,
                        'entropy_production_rate': trajectory.entropy_production_rate
                    },
                    'poincare_section': {
                        'num_intersections': len(poincare_section.intersection_points),
                        'num_fixed_points': len(poincare_section.fixed_points),
                        'stability_measure': poincare_section.stability_analysis.get('stability_measure', 0.0)
                    },
                    'sample_trajectory_points': [asdict(p) for p in trajectory.points[::max(1, len(trajectory.points)//20)]],
                    'calculated_lyapunov': lyapunov_exp
                })
                
            except Exception as e:
                warnings.warn(f"Failed to generate scenario {i}: {str(e)}")
                continue
        
        return scenarios
    
    def save_phase_space_data(self, scenarios: List[Dict[str, Any]], output_path: str) -> None:
        """Save phase-space trajectory data to file."""
        
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
            'phase_space_scenarios': serializable_scenarios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'component_dynamics': self.component_dynamics,
                'phase_space_scales': self.phase_space_scales,
                'analysis_params': self.analysis_params,
                'physical_constants': {
                    'R_universal': self.R_universal,
                    'k_boltzmann': self.k_boltzmann
                }
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved phase-space trajectory data to {output_path}")


if __name__ == "__main__":
    # Test phase-space trajectory models
    print("🔄 Testing Phase-Space Trajectory Models")
    
    # Initialize model
    phase_model = PhaseSpaceTrajectoriesModel()
    
    # Test trajectory generation
    initial_conditions = {
        'temperature_K': 353.15,
        'pressure_Pa': 1800000,
        'volume_m3': 0.001,
        'entropy_J_per_K': 1200.0,
        'enthalpy_J': 300000.0
    }
    
    process_parameters = {
        'process_type': 'compression',
        'mode': 'normal'
    }
    
    time_span = (0.0, 100.0)
    
    print(f"\n🚀 Generating trajectory...")
    trajectory = phase_model.generate_thermodynamic_trajectory(
        'compressor', initial_conditions, process_parameters, time_span
    )
    
    print(f"✅ Trajectory generated: {len(trajectory.points)} points")
    print(f"✅ Is periodic: {trajectory.is_periodic}")
    print(f"✅ Is chaotic: {trajectory.is_chaotic}")
    print(f"✅ Lyapunov exponent: {trajectory.lyapunov_exponent:.4f}")
    
    # Test Poincaré section
    print(f"\n📊 Creating Poincaré section...")
    avg_temp = np.mean([p.temperature_K for p in trajectory.points])
    poincare = phase_model.create_poincare_section(trajectory, 'temperature_K', avg_temp)
    
    print(f"✅ Poincaré section: {len(poincare.intersection_points)} intersections")
    print(f"✅ Fixed points: {len(poincare.fixed_points)}")
    
    # Test Lyapunov calculation
    print(f"\n📈 Calculating Lyapunov exponent...")
    lyapunov = phase_model.calculate_lyapunov_exponent(trajectory)
    print(f"✅ Calculated Lyapunov: {lyapunov:.4f}")
    
    # Generate scenarios
    print(f"\n🎲 Generating phase-space scenarios...")
    scenarios = phase_model.generate_phase_space_scenarios(3)
    print(f"Generated {len(scenarios)} phase-space scenarios")
    
    # Save test data
    phase_model.save_phase_space_data(scenarios, "phase_space_trajectories_data.json")
    
    print("\n✅ Phase-space trajectory models testing complete!")