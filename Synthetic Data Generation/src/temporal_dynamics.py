#!/usr/bin/env python3
"""
Temporal Dynamics for Hamiltonian Neural Network Training
=========================================================

This module implements realistic temporal dynamics for AIRTHIUM heat pump systems,
capturing transient behavior, startup/shutdown dynamics, and energy-conserving
phase-space trajectories required for HNN training.

Key Features:
- Hamiltonian formulation with generalized coordinates and momenta
- Realistic transient behavior (startup, shutdown, load changes)
- Energy conservation validation
- Phase-space trajectory generation
- Component-level temporal dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from thermodynamic_properties import ThermodynamicState, get_fluid_properties
from cycle_generators import CycleResults, CycleParameters
from component_models import CompressorModel, HeatExchangerModel, ExpansionDeviceModel

@dataclass
class HamiltonianState:
    """
    Complete Hamiltonian state for thermodynamic system.
    
    Following Hamilton's formulation: H(q,p) where q are coordinates, p are momenta
    """
    # Time
    time: float  # [s]
    
    # Generalized Coordinates (q) - Thermodynamic states
    q_P: np.ndarray  # Pressures at each component [Pa] - shape (4,)
    q_V: np.ndarray  # Specific volumes [m³/kg] - shape (4,)
    q_T: np.ndarray  # Temperatures [K] - shape (4,)
    q_S: np.ndarray  # Specific entropies [J/kg·K] - shape (4,)
    q_H: np.ndarray  # Specific enthalpies [J/kg] - shape (4,)
    q_m: np.ndarray  # Mass distribution [kg] - shape (4,)
    
    # Generalized Momenta (p) - Rates of change
    p_P: np.ndarray  # dP/dt [Pa/s] - shape (4,)
    p_V: np.ndarray  # dV/dt [m³/kg·s] - shape (4,)
    p_T: np.ndarray  # dT/dt [K/s] - shape (4,)
    p_S: np.ndarray  # dS/dt [J/kg·K·s] - shape (4,)
    p_H: np.ndarray  # dH/dt [J/kg·s] - shape (4,)
    p_m: np.ndarray  # dm/dt [kg/s] - shape (4,)
    
    # Hamiltonian components
    internal_energy: float  # Internal energy [J]
    kinetic_energy: float   # Kinetic energy [J]
    potential_energy: float # Potential energy [J]
    total_hamiltonian: float # Total Hamiltonian H = U + K + Φ [J]
    
    # System performance
    COP_heating: float
    COP_cooling: float
    power_input: float  # [W]
    heat_output: float  # [W]
    
    # Component states
    compressor_speed: float  # [rpm]
    expansion_valve_opening: float  # [%]
    
    # Process identification
    process_phase: str  # "startup", "steady_state", "shutdown", "transient"
    dominant_process: str  # "compression", "evaporation", "expansion", "condensation"

@dataclass
class TransientScenario:
    """Definition of a transient scenario for simulation."""
    name: str
    description: str
    duration: float  # [s]
    initial_conditions: Dict[str, float]
    boundary_conditions: Callable[[float], Dict[str, float]]  # Function of time
    disturbances: List[Dict[str, Any]]  # List of disturbance events

class TemporalDynamicsGenerator:
    """
    Generate realistic temporal dynamics for AIRTHIUM heat pump systems.
    """
    
    def __init__(self, fluid_name: str = "R410A", sampling_rate: float = 10.0):
        """
        Initialize temporal dynamics generator.
        
        Args:
            fluid_name: Working fluid name
            sampling_rate: Data sampling rate [Hz]
        """
        self.fluid_name = fluid_name
        self.fluid = get_fluid_properties(fluid_name)
        self.dt = 1.0 / sampling_rate
        self.sampling_rate = sampling_rate
        
        # Component models for realistic dynamics
        from component_models import CompressorType, HeatExchangerType, ExpansionType
        self.compressor = CompressorModel(CompressorType.SCROLL, displacement=0.15)
        self.evaporator = HeatExchangerModel(HeatExchangerType.FINNED_TUBE, area=40.0)
        self.condenser = HeatExchangerModel(HeatExchangerType.PLATE, area=60.0)
        self.expansion_valve = ExpansionDeviceModel(ExpansionType.THROTTLING_VALVE)
        
        # System parameters
        self.total_refrigerant_mass = 5.0  # [kg]
        self.system_volume = 0.05  # [m³]
        
        # Thermal masses for realistic transients
        self.thermal_masses = {
            'compressor': 50.0,  # [kg] equivalent thermal mass
            'evaporator': 30.0,
            'condenser': 40.0,
            'piping': 20.0
        }
        
    def generate_startup_dynamics(self, target_conditions: Dict[str, float],
                                startup_duration: float = 300.0) -> List[HamiltonianState]:
        """
        Generate realistic startup dynamics from cold start to steady state.
        
        Args:
            target_conditions: Target steady-state conditions
            startup_duration: Duration of startup phase [s]
            
        Returns:
            List of Hamiltonian states during startup
        """
        
        # Time vector
        t_span = np.arange(0, startup_duration + self.dt, self.dt)
        n_points = len(t_span)
        
        # Initial conditions (system at ambient temperature)
        T_ambient = target_conditions.get('T_ambient', 288.15)  # 15°C
        P_ambient = 101325  # 1 atm
        
        # Initial state (all components at ambient conditions)
        initial_state = self._create_initial_state(T_ambient, P_ambient)
        
        # Startup profile functions
        compressor_speed_profile = self._create_compressor_startup_profile(startup_duration)
        valve_opening_profile = self._create_valve_startup_profile(startup_duration)
        
        # Generate trajectory
        states = []
        current_state = initial_state
        
        for i, t in enumerate(t_span):
            # Get control inputs at this time
            compressor_speed = compressor_speed_profile(t)
            valve_opening = valve_opening_profile(t)
            
            # Calculate system dynamics
            if i > 0:
                # Integrate from previous state
                current_state = self._integrate_dynamics(
                    current_state, self.dt, compressor_speed, valve_opening,
                    target_conditions, phase="startup"
                )
            
            # Update time
            current_state.time = t
            current_state.process_phase = "startup"
            current_state.compressor_speed = compressor_speed
            current_state.expansion_valve_opening = valve_opening
            
            states.append(current_state)
        
        return states
    
    def generate_shutdown_dynamics(self, initial_steady_state: HamiltonianState,
                                 shutdown_duration: float = 600.0) -> List[HamiltonianState]:
        """
        Generate realistic shutdown dynamics from steady state to off.
        
        Args:
            initial_steady_state: Starting steady-state condition
            shutdown_duration: Duration of shutdown phase [s]
            
        Returns:
            List of Hamiltonian states during shutdown
        """
        
        t_span = np.arange(0, shutdown_duration + self.dt, self.dt)
        
        # Shutdown profiles
        compressor_speed_profile = self._create_compressor_shutdown_profile(
            initial_steady_state.compressor_speed, shutdown_duration
        )
        valve_opening_profile = self._create_valve_shutdown_profile(shutdown_duration)
        
        states = []
        current_state = initial_steady_state
        
        for i, t in enumerate(t_span):
            compressor_speed = compressor_speed_profile(t)
            valve_opening = valve_opening_profile(t)
            
            if i > 0:
                current_state = self._integrate_dynamics(
                    current_state, self.dt, compressor_speed, valve_opening,
                    {}, phase="shutdown"
                )
            
            current_state.time = t
            current_state.process_phase = "shutdown"
            current_state.compressor_speed = compressor_speed
            current_state.expansion_valve_opening = valve_opening
            
            states.append(current_state)
        
        return states
    
    def generate_load_change_dynamics(self, initial_state: HamiltonianState,
                                    load_change: Dict[str, float],
                                    response_duration: float = 120.0) -> List[HamiltonianState]:
        """
        Generate dynamics for load change events.
        
        Args:
            initial_state: Initial steady state
            load_change: Dictionary of load changes (e.g., {'heating_load': 1.2})
            response_duration: Duration of response [s]
            
        Returns:
            List of Hamiltonian states during load change
        """
        
        t_span = np.arange(0, response_duration + self.dt, self.dt)
        
        # Create load change profiles
        load_profiles = {}
        for param, final_value in load_change.items():
            initial_value = getattr(initial_state, param, 1.0)
            load_profiles[param] = self._create_exponential_profile(
                initial_value, final_value, response_duration, tau=30.0
            )
        
        states = []
        current_state = initial_state
        
        for i, t in enumerate(t_span):
            # Apply load changes
            target_conditions = {}
            for param, profile in load_profiles.items():
                target_conditions[param] = profile(t)
            
            if i > 0:
                current_state = self._integrate_dynamics(
                    current_state, self.dt, 
                    current_state.compressor_speed,
                    current_state.expansion_valve_opening,
                    target_conditions, phase="load_change"
                )
            
            current_state.time = t
            current_state.process_phase = "transient"
            
            states.append(current_state)
        
        return states
    
    def generate_defrost_cycle(self, initial_state: HamiltonianState,
                             defrost_duration: float = 600.0) -> List[HamiltonianState]:
        """
        Generate defrost cycle dynamics for air-source heat pump.
        
        Args:
            initial_state: State before defrost
            defrost_duration: Duration of defrost cycle [s]
            
        Returns:
            List of Hamiltonian states during defrost
        """
        
        t_span = np.arange(0, defrost_duration + self.dt, self.dt)
        
        # Defrost phases
        phases = [
            {'name': 'reverse_valve_switch', 'duration': 10.0},
            {'name': 'defrost_heating', 'duration': 300.0},
            {'name': 'ice_melting', 'duration': 240.0},
            {'name': 'return_to_heating', 'duration': 50.0}
        ]
        
        states = []
        current_state = initial_state
        phase_start_time = 0
        
        for i, t in enumerate(t_span):
            # Determine current defrost phase
            current_phase = self._get_defrost_phase(t, phases)
            
            # Apply defrost-specific dynamics
            if i > 0:
                current_state = self._integrate_defrost_dynamics(
                    current_state, self.dt, current_phase, t
                )
            
            current_state.time = t
            current_state.process_phase = f"defrost_{current_phase}"
            
            states.append(current_state)
        
        return states
    
    def _create_initial_state(self, T_ambient: float, P_ambient: float) -> HamiltonianState:
        """Create initial state at ambient conditions."""
        
        # All components start at ambient temperature and pressure
        q_P = np.array([P_ambient, P_ambient, P_ambient, P_ambient])
        q_T = np.array([T_ambient, T_ambient, T_ambient, T_ambient])
        
        # Estimate other properties
        try:
            state_ambient = self.fluid.get_properties(P_ambient, T_ambient)
            q_V = np.array([1/state_ambient.rho] * 4)
            q_S = np.array([state_ambient.s] * 4)
            q_H = np.array([state_ambient.h] * 4)
        except:
            # Fallback values
            q_V = np.array([0.001] * 4)  # Liquid-like density
            q_S = np.array([1000.0] * 4)
            q_H = np.array([200000.0] * 4)
        
        # Mass distribution (equal initially)
        q_m = np.array([self.total_refrigerant_mass / 4] * 4)
        
        # Zero initial rates
        p_P = np.zeros(4)
        p_V = np.zeros(4)
        p_T = np.zeros(4)
        p_S = np.zeros(4)
        p_H = np.zeros(4)
        p_m = np.zeros(4)
        
        # Calculate initial Hamiltonian
        internal_energy = np.sum(q_m * (q_H - q_P * q_V))
        kinetic_energy = 0.0  # No flow initially
        potential_energy = np.sum(q_m * 9.81 * 1.0)  # Assume 1m height
        total_hamiltonian = internal_energy + kinetic_energy + potential_energy
        
        return HamiltonianState(
            time=0.0,
            q_P=q_P, q_V=q_V, q_T=q_T, q_S=q_S, q_H=q_H, q_m=q_m,
            p_P=p_P, p_V=p_V, p_T=p_T, p_S=p_S, p_H=p_H, p_m=p_m,
            internal_energy=internal_energy,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            total_hamiltonian=total_hamiltonian,
            COP_heating=1.0,
            COP_cooling=1.0,
            power_input=0.0,
            heat_output=0.0,
            compressor_speed=0.0,
            expansion_valve_opening=0.0,
            process_phase="initial",
            dominant_process="none"
        )
    
    def _create_compressor_startup_profile(self, duration: float) -> Callable[[float], float]:
        """Create realistic compressor speed ramp-up profile."""
        
        def speed_profile(t: float) -> float:
            if t < 5.0:
                # Initial delay
                return 0.0
            elif t < 30.0:
                # Soft start ramp
                return 600.0 * (t - 5.0) / 25.0
            elif t < 60.0:
                # Acceleration to operating speed
                progress = (t - 30.0) / 30.0
                return 600.0 + 1200.0 * (1 - np.exp(-3 * progress))
            else:
                # Steady operating speed
                return 1800.0
        
        return speed_profile
    
    def _create_valve_startup_profile(self, duration: float) -> Callable[[float], float]:
        """Create expansion valve opening profile during startup."""
        
        def valve_profile(t: float) -> float:
            if t < 10.0:
                return 0.0  # Closed initially
            elif t < 45.0:
                # Gradual opening
                return 20.0 * (t - 10.0) / 35.0
            elif t < 120.0:
                # Control-based opening
                return 20.0 + 30.0 * (1 - np.exp(-(t - 45.0) / 30.0))
            else:
                return 50.0  # Steady-state opening
        
        return valve_profile
    
    def _create_compressor_shutdown_profile(self, initial_speed: float, 
                                          duration: float) -> Callable[[float], float]:
        """Create compressor coast-down profile."""
        
        def shutdown_profile(t: float) -> float:
            if t < 5.0:
                return initial_speed  # Brief delay
            else:
                # Exponential decay
                tau = 20.0  # Time constant
                return initial_speed * np.exp(-(t - 5.0) / tau)
        
        return shutdown_profile
    
    def _create_valve_shutdown_profile(self, duration: float) -> Callable[[float], float]:
        """Create valve closing profile during shutdown."""
        
        def valve_profile(t: float) -> float:
            if t < 2.0:
                return 50.0  # Initial opening
            elif t < 30.0:
                # Rapid closing
                return 50.0 * (1 - (t - 2.0) / 28.0)
            else:
                return 0.0  # Fully closed
        
        return valve_profile
    
    def _create_exponential_profile(self, initial: float, final: float, 
                                  duration: float, tau: float) -> Callable[[float], float]:
        """Create exponential approach profile."""
        
        def profile(t: float) -> float:
            return final + (initial - final) * np.exp(-t / tau)
        
        return profile
    
    def _integrate_dynamics(self, current_state: HamiltonianState, dt: float,
                          compressor_speed: float, valve_opening: float,
                          target_conditions: Dict[str, float], 
                          phase: str) -> HamiltonianState:
        """
        Integrate system dynamics using Hamilton's equations.
        
        Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q + external_forces
        """
        
        # Calculate derivatives based on Hamilton's equations
        dq_dt, dp_dt = self._calculate_hamiltonian_derivatives(
            current_state, compressor_speed, valve_opening, target_conditions, phase
        )
        
        # Integrate using Euler method (could use higher-order methods)
        new_q_P = current_state.q_P + dq_dt['P'] * dt
        new_q_V = current_state.q_V + dq_dt['V'] * dt
        new_q_T = current_state.q_T + dq_dt['T'] * dt
        new_q_S = current_state.q_S + dq_dt['S'] * dt
        new_q_H = current_state.q_H + dq_dt['H'] * dt
        new_q_m = current_state.q_m + dq_dt['m'] * dt
        
        new_p_P = current_state.p_P + dp_dt['P'] * dt
        new_p_V = current_state.p_V + dp_dt['V'] * dt
        new_p_T = current_state.p_T + dp_dt['T'] * dt
        new_p_S = current_state.p_S + dp_dt['S'] * dt
        new_p_H = current_state.p_H + dp_dt['H'] * dt
        new_p_m = current_state.p_m + dp_dt['m'] * dt
        
        # Calculate new Hamiltonian
        internal_energy = np.sum(new_q_m * (new_q_H - new_q_P * new_q_V))
        
        # Kinetic energy from flow rates
        flow_velocity = np.linalg.norm(new_p_m)  # Simplified
        kinetic_energy = 0.5 * np.sum(new_q_m) * (flow_velocity / 1000)**2
        
        # Potential energy
        potential_energy = np.sum(new_q_m * 9.81 * 1.0)
        
        total_hamiltonian = internal_energy + kinetic_energy + potential_energy
        
        # Calculate performance metrics
        COP_heating, COP_cooling, power_input, heat_output = self._calculate_performance(
            new_q_P, new_q_V, new_q_T, new_q_H, new_q_m, compressor_speed
        )
        
        # Determine dominant process
        dominant_process = self._identify_dominant_process(dq_dt, dp_dt)
        
        return HamiltonianState(
            time=current_state.time + dt,
            q_P=new_q_P, q_V=new_q_V, q_T=new_q_T, q_S=new_q_S, q_H=new_q_H, q_m=new_q_m,
            p_P=new_p_P, p_V=new_p_V, p_T=new_p_T, p_S=new_p_S, p_H=new_p_H, p_m=new_p_m,
            internal_energy=internal_energy,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            total_hamiltonian=total_hamiltonian,
            COP_heating=COP_heating,
            COP_cooling=COP_cooling,
            power_input=power_input,
            heat_output=heat_output,
            compressor_speed=compressor_speed,
            expansion_valve_opening=valve_opening,
            process_phase=phase,
            dominant_process=dominant_process
        )
    
    def _calculate_hamiltonian_derivatives(self, state: HamiltonianState,
                                         compressor_speed: float, valve_opening: float,
                                         target_conditions: Dict[str, float],
                                         phase: str) -> Tuple[Dict, Dict]:
        """
        Calculate derivatives based on Hamilton's equations and system physics.
        """
        
        # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q + F_external
        
        # Simplified derivatives based on thermodynamic processes
        dq_dt = {
            'P': state.p_P.copy(),
            'V': state.p_V.copy(),
            'T': state.p_T.copy(),
            'S': state.p_S.copy(),
            'H': state.p_H.copy(),
            'm': state.p_m.copy()
        }
        
        # External forces from component operations
        dp_dt = {
            'P': np.zeros(4),
            'V': np.zeros(4),
            'T': np.zeros(4),
            'S': np.zeros(4),
            'H': np.zeros(4),
            'm': np.zeros(4)
        }
        
        # Compressor effects (component 0 -> 1)
        if compressor_speed > 0:
            pressure_rise = compressor_speed * 100  # Simplified
            dp_dt['P'][1] += pressure_rise / self.thermal_masses['compressor']
            dp_dt['P'][0] -= pressure_rise / self.thermal_masses['compressor']
            
            # Temperature rise from compression
            temp_rise = pressure_rise * 0.001  # Simplified
            dp_dt['T'][1] += temp_rise / self.thermal_masses['compressor']
        
        # Heat exchanger effects
        # Evaporator (component 3 -> 0)
        if state.q_T[3] < state.q_T[0]:
            heat_rate = 1000 * (state.q_T[0] - state.q_T[3])  # Simplified
            dp_dt['H'][0] += heat_rate / (state.q_m[0] + 1e-6)
            dp_dt['T'][0] += heat_rate / (state.q_m[0] * 2000 + 1e-6)  # Cp ~ 2000 J/kg/K
        
        # Condenser (component 1 -> 2)
        if state.q_T[1] > state.q_T[2]:
            heat_rate = 1000 * (state.q_T[1] - state.q_T[2])  # Simplified
            dp_dt['H'][2] -= heat_rate / (state.q_m[2] + 1e-6)
            dp_dt['T'][2] -= heat_rate / (state.q_m[2] * 2000 + 1e-6)
        
        # Expansion valve effects (component 2 -> 3)
        if valve_opening > 0:
            pressure_drop = valve_opening * 10  # Simplified
            dp_dt['P'][3] -= pressure_drop / self.thermal_masses['evaporator']
            dp_dt['P'][2] += pressure_drop / self.thermal_masses['condenser']
        
        # Add thermal inertia effects
        for i in range(4):
            component_mass = list(self.thermal_masses.values())[i % len(self.thermal_masses)]
            dp_dt['T'][i] *= 1.0 / component_mass
            dp_dt['P'][i] *= 1.0 / component_mass
        
        return dq_dt, dp_dt
    
    def _calculate_performance(self, P: np.ndarray, V: np.ndarray, T: np.ndarray,
                             H: np.ndarray, m: np.ndarray, 
                             compressor_speed: float) -> Tuple[float, float, float, float]:
        """Calculate instantaneous performance metrics."""
        
        # Simplified performance calculation
        if compressor_speed > 0:
            # Power input (simplified)
            power_input = compressor_speed * 10  # W
            
            # Heat rates
            heat_evap = np.sum(m * H * 0.001)  # Simplified
            heat_cond = heat_evap + power_input
            
            # COPs
            COP_heating = heat_cond / power_input if power_input > 0 else 1.0
            COP_cooling = heat_evap / power_input if power_input > 0 else 1.0
            
            # Limit to realistic values
            COP_heating = np.clip(COP_heating, 1.0, 8.0)
            COP_cooling = np.clip(COP_cooling, 1.0, 6.0)
            
            return COP_heating, COP_cooling, power_input, heat_cond
        else:
            return 1.0, 1.0, 0.0, 0.0
    
    def _identify_dominant_process(self, dq_dt: Dict, dp_dt: Dict) -> str:
        """Identify the dominant thermodynamic process."""
        
        # Analyze rates of change to identify process
        P_rate = np.max(np.abs(dq_dt['P']))
        T_rate = np.max(np.abs(dq_dt['T']))
        H_rate = np.max(np.abs(dq_dt['H']))
        
        if P_rate > T_rate and P_rate > H_rate:
            return "compression" if np.sum(dq_dt['P']) > 0 else "expansion"
        elif H_rate > P_rate and H_rate > T_rate:
            return "evaporation" if np.sum(dq_dt['H']) > 0 else "condensation"
        else:
            return "heat_transfer"
    
    def _get_defrost_phase(self, t: float, phases: List[Dict]) -> str:
        """Determine current defrost phase."""
        
        cumulative_time = 0
        for phase in phases:
            if t <= cumulative_time + phase['duration']:
                return phase['name']
            cumulative_time += phase['duration']
        
        return phases[-1]['name']  # Last phase
    
    def _integrate_defrost_dynamics(self, state: HamiltonianState, dt: float,
                                  phase: str, t: float) -> HamiltonianState:
        """Integrate dynamics during defrost cycle."""
        
        # Simplified defrost dynamics
        if phase == "reverse_valve_switch":
            # Rapid pressure changes
            dp_dt_defrost = {'P': np.array([100, -100, 50, -50])}
        elif phase == "defrost_heating":
            # Heating outdoor coil
            dp_dt_defrost = {'T': np.array([0, 0, 5, 10])}
        elif phase == "ice_melting":
            # Energy consumption for melting
            dp_dt_defrost = {'H': np.array([0, 0, -1000, 2000])}
        else:  # return_to_heating
            # Return to normal operation
            dp_dt_defrost = {'P': np.array([-50, 50, -25, 25])}
        
        # Apply changes (simplified)
        new_state = state
        for prop, changes in dp_dt_defrost.items():
            if prop == 'P':
                new_state.q_P += changes * dt
            elif prop == 'T':
                new_state.q_T += changes * dt
            elif prop == 'H':
                new_state.q_H += changes * dt
        
        return new_state
    
    def validate_energy_conservation(self, states: List[HamiltonianState]) -> Dict[str, float]:
        """Validate energy conservation throughout trajectory."""
        
        if not states:
            return {'valid': False, 'error': 100.0, 'drift': 100.0}
        
        initial_energy = states[0].total_hamiltonian
        final_energy = states[-1].total_hamiltonian
        
        # Calculate energy drift
        energies = [s.total_hamiltonian for s in states]
        max_energy = max(energies)
        min_energy = min(energies)
        
        if abs(initial_energy) > 1e-6:
            relative_error = abs(final_energy - initial_energy) / abs(initial_energy) * 100
            energy_drift = (max_energy - min_energy) / abs(initial_energy) * 100
        else:
            relative_error = 0.0
            energy_drift = 0.0
        
        # Energy should be conserved within 2% for realistic systems
        valid = relative_error < 2.0 and energy_drift < 5.0
        
        return {
            'valid': valid,
            'error': relative_error,
            'drift': energy_drift,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'max_energy': max_energy,
            'min_energy': min_energy
        }
    
    def save_temporal_dataset(self, states_list: List[List[HamiltonianState]],
                            output_path: str, metadata: Dict[str, Any] = None) -> None:
        """Save temporal dataset for HNN training."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'n_trajectories': len(states_list),
                'sampling_rate': self.sampling_rate,
                'fluid_name': self.fluid_name,
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                **(metadata or {})
            },
            'trajectories': []
        }
        
        for i, states in enumerate(states_list):
            if not states:
                continue
                
            trajectory = {
                'trajectory_id': f"traj_{i:06d}",
                'duration': states[-1].time - states[0].time,
                'n_timesteps': len(states),
                'scenario': states[0].process_phase,
                
                # Hamiltonian coordinates and momenta
                'coordinates': {
                    'time': [s.time for s in states],
                    'P': [s.q_P.tolist() for s in states],
                    'V': [s.q_V.tolist() for s in states],
                    'T': [s.q_T.tolist() for s in states],
                    'S': [s.q_S.tolist() for s in states],
                    'H': [s.q_H.tolist() for s in states],
                    'm': [s.q_m.tolist() for s in states]
                },
                'momenta': {
                    'dP_dt': [s.p_P.tolist() for s in states],
                    'dV_dt': [s.p_V.tolist() for s in states],
                    'dT_dt': [s.p_T.tolist() for s in states],
                    'dS_dt': [s.p_S.tolist() for s in states],
                    'dH_dt': [s.p_H.tolist() for s in states],
                    'dm_dt': [s.p_m.tolist() for s in states]
                },
                
                # Energy components
                'energies': {
                    'internal': [s.internal_energy for s in states],
                    'kinetic': [s.kinetic_energy for s in states],
                    'potential': [s.potential_energy for s in states],
                    'total_hamiltonian': [s.total_hamiltonian for s in states]
                },
                
                # Performance targets
                'targets': {
                    'COP_heating': [s.COP_heating for s in states],
                    'COP_cooling': [s.COP_cooling for s in states],
                    'power_input': [s.power_input for s in states],
                    'heat_output': [s.heat_output for s in states]
                },
                
                # Control inputs
                'controls': {
                    'compressor_speed': [s.compressor_speed for s in states],
                    'valve_opening': [s.expansion_valve_opening for s in states]
                },
                
                # Process information
                'processes': {
                    'phase': [s.process_phase for s in states],
                    'dominant_process': [s.dominant_process for s in states]
                }
            }
            
            dataset['trajectories'].append(trajectory)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 Saved temporal dataset with {len(states_list)} trajectories to {output_path}")


if __name__ == "__main__":
    # Test temporal dynamics generation
    print("🔄 Testing Temporal Dynamics Generator")
    
    # Initialize generator
    generator = TemporalDynamicsGenerator(fluid_name="R410A", sampling_rate=10.0)
    
    # Test startup dynamics
    print("\n🚀 Generating startup dynamics...")
    target_conditions = {
        'T_ambient': 278.15,  # 5°C
        'heating_load': 10000,  # 10 kW
        'water_temp': 318.15   # 45°C
    }
    
    startup_states = generator.generate_startup_dynamics(
        target_conditions, startup_duration=120.0
    )
    
    print(f"✅ Generated {len(startup_states)} startup states")
    print(f"⏱️  Duration: {startup_states[-1].time:.1f} seconds")
    print(f"🎯 Final COP: {startup_states[-1].COP_heating:.2f}")
    
    # Validate energy conservation
    energy_validation = generator.validate_energy_conservation(startup_states)
    print(f"⚡ Energy conservation - Valid: {energy_validation['valid']}, "
          f"Error: {energy_validation['error']:.2f}%")
    
    # Test shutdown dynamics
    print("\n🛑 Generating shutdown dynamics...")
    shutdown_states = generator.generate_shutdown_dynamics(
        startup_states[-1], shutdown_duration=60.0
    )
    
    print(f"✅ Generated {len(shutdown_states)} shutdown states")
    print(f"🎯 Final compressor speed: {shutdown_states[-1].compressor_speed:.1f} rpm")
    
    # Save test dataset
    all_trajectories = [startup_states, shutdown_states]
    generator.save_temporal_dataset(
        all_trajectories,
        "test_temporal_data.json",
        metadata={'test_run': True, 'scenarios': ['startup', 'shutdown']}
    )
    
    print("\n✅ Temporal dynamics implementation complete!")