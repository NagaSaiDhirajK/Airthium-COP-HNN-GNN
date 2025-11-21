#!/usr/bin/env python3
"""
Part-Load Operation and Cycling Behavior Models for AIRTHIUM Heat Pumps
======================================================================

This module implements realistic part-load operation and cycling behavior
for heat pump systems, including variable capacity control, start-stop cycling,
and modulation strategies.

Key Features:
- Variable capacity compressor control
- Start-stop cycling behavior
- Modulation strategies (hot gas bypass, variable speed)
- Part-load efficiency curves
- Cycling losses and penalties
- Building load matching algorithms

Sources:
- ASHRAE Handbook - HVAC Systems and Equipment (2020)
- AHRI Standard 210/240 Performance Rating
- DOE Building Technologies Office Studies
- Carrier Variable Capacity Systems Manual
- Daikin VRV Technical Data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

@dataclass
class PartLoadState:
    """Current part-load operating state of the heat pump."""
    system_id: str
    timestamp: str
    
    # Load conditions
    building_load_kW: float           # Current building heating/cooling load
    design_load_kW: float             # Design load at rating conditions
    load_factor: float                # Actual load / Design load (0-1)
    
    # Capacity control
    compressor_capacity_percent: float # Compressor capacity (0-100%)
    modulation_type: str              # 'on_off', 'hot_gas_bypass', 'variable_speed'
    modulation_level: float           # Modulation level (0-1)
    
    # Cycling behavior
    is_cycling: bool                  # True if system is cycling
    cycle_on_time_minutes: float      # Time compressor has been on
    cycle_off_time_minutes: float     # Time compressor has been off
    cycles_per_hour: float            # Cycling frequency
    
    # Performance at part load
    part_load_cop: float              # COP at current part load
    part_load_capacity_kW: float      # Actual capacity delivered
    part_load_power_kW: float         # Power consumption at part load
    
    # Efficiency factors
    part_load_ratio: float            # PLR = Load / Available Capacity
    energy_input_ratio: float         # EIR = Power / Capacity
    cycling_degradation_factor: float # Degradation due to cycling
    
    # Control parameters
    thermostat_deadband_K: float      # Thermostat deadband
    minimum_on_time_minutes: float    # Minimum compressor on time
    minimum_off_time_minutes: float   # Minimum compressor off time
    
    # System state
    compressor_running: bool          # Current compressor state
    defrost_active: bool              # Defrost cycle active
    startup_transient: bool           # In startup transient

@dataclass
class CyclingParameters:
    """Parameters for cycling behavior modeling."""
    system_type: str
    capacity_control_type: str
    cycling_characteristics: Dict[str, float]
    part_load_curves: Dict[str, Callable]

class PartLoadCyclingModel:
    """Model part-load operation and cycling behavior."""
    
    def __init__(self):
        """Initialize part-load cycling model with verified parameters."""
        
        # Part-load performance curves from AHRI testing
        self.part_load_curves = {
            'single_speed': {
                # COP vs Part Load Ratio (PLR)
                'cop_curve': lambda plr: np.maximum(0.1, 
                    -0.8*plr**3 + 1.2*plr**2 + 0.6*plr),
                # Capacity vs PLR (for cycling systems)
                'capacity_curve': lambda plr: np.where(plr > 0.1, 1.0, 0.0),
                # Energy Input Ratio vs PLR
                'eir_curve': lambda plr: np.maximum(0.5,
                    0.3 + 0.7/np.maximum(0.1, plr))
            },
            'two_speed': {
                'cop_curve': lambda plr: np.maximum(0.1,
                    np.where(plr < 0.5, 
                        -0.5*plr**2 + 1.0*plr + 0.2,
                        -0.3*plr**2 + 0.8*plr + 0.4)),
                'capacity_curve': lambda plr: np.where(plr > 0.1,
                    np.where(plr < 0.5, 0.5, 1.0), 0.0),
                'eir_curve': lambda plr: np.maximum(0.4,
                    0.25 + 0.75/np.maximum(0.1, plr))
            },
            'variable_speed': {
                'cop_curve': lambda plr: np.maximum(0.2,
                    -0.2*plr**3 + 0.3*plr**2 + 0.9*plr),
                'capacity_curve': lambda plr: np.maximum(0.2, plr),
                'eir_curve': lambda plr: np.maximum(0.3,
                    0.2 + 0.8*plr**(-0.5))
            },
            'hot_gas_bypass': {
                'cop_curve': lambda plr: np.maximum(0.1,
                    -1.2*plr**3 + 1.8*plr**2 + 0.4*plr),
                'capacity_curve': lambda plr: np.maximum(0.3, plr),
                'eir_curve': lambda plr: np.maximum(0.6,
                    0.4 + 0.6/np.maximum(0.3, plr))
            }
        }
        
        # Cycling degradation factors from DOE studies
        self.cycling_degradation = {
            'single_speed': {
                'degradation_coefficient': 0.25,  # Cd coefficient
                'minimum_plr': 0.1,               # Below this, cycling occurs
                'cycling_loss_factor': 0.05,     # 5% loss per cycle
                'startup_energy_penalty': 0.02   # 2% energy penalty per start
            },
            'two_speed': {
                'degradation_coefficient': 0.15,
                'minimum_plr': 0.05,
                'cycling_loss_factor': 0.03,
                'startup_energy_penalty': 0.015
            },
            'variable_speed': {
                'degradation_coefficient': 0.05,
                'minimum_plr': 0.2,
                'cycling_loss_factor': 0.01,
                'startup_energy_penalty': 0.005
            },
            'hot_gas_bypass': {
                'degradation_coefficient': 0.35,
                'minimum_plr': 0.3,
                'cycling_loss_factor': 0.02,
                'startup_energy_penalty': 0.01
            }
        }
        
        # Thermostat and control parameters
        self.control_parameters = {
            'residential': {
                'deadband_K': 1.0,              # ±0.5°C deadband
                'minimum_on_time_min': 3.0,     # 3 minutes minimum on
                'minimum_off_time_min': 3.0,    # 3 minutes minimum off
                'anticipator_factor': 0.1       # Thermostat anticipator
            },
            'commercial': {
                'deadband_K': 2.0,              # ±1°C deadband
                'minimum_on_time_min': 5.0,     # 5 minutes minimum on
                'minimum_off_time_min': 5.0,    # 5 minutes minimum off
                'anticipator_factor': 0.05
            },
            'industrial': {
                'deadband_K': 3.0,              # ±1.5°C deadband
                'minimum_on_time_min': 10.0,    # 10 minutes minimum on
                'minimum_off_time_min': 10.0,   # 10 minutes minimum off
                'anticipator_factor': 0.02
            }
        }
        
        # Building thermal mass effects
        self.thermal_mass_effects = {
            'light': {
                'time_constant_hours': 2.0,     # Light construction
                'load_smoothing_factor': 0.1,
                'temperature_swing_K': 2.0
            },
            'medium': {
                'time_constant_hours': 6.0,     # Medium construction
                'load_smoothing_factor': 0.3,
                'temperature_swing_K': 1.0
            },
            'heavy': {
                'time_constant_hours': 12.0,    # Heavy construction
                'load_smoothing_factor': 0.6,
                'temperature_swing_K': 0.5
            }
        }
    
    def initialize_part_load_state(self, system_id: str, system_type: str,
                                  design_capacity_kW: float, current_load_kW: float,
                                  control_type: str = 'residential',
                                  modulation_type: str = 'single_speed') -> PartLoadState:
        """Initialize part-load operating state."""
        
        timestamp = datetime.now().isoformat()
        
        # Calculate load factor
        load_factor = current_load_kW / design_capacity_kW if design_capacity_kW > 0 else 0
        load_factor = np.clip(load_factor, 0, 1.5)  # Allow 50% overload
        
        # Get control parameters
        control_params = self.control_parameters.get(control_type, 
                                                   self.control_parameters['residential'])
        
        # Determine if cycling is needed
        cycling_params = self.cycling_degradation.get(modulation_type,
                                                    self.cycling_degradation['single_speed'])
        
        is_cycling = load_factor < cycling_params['minimum_plr']
        
        # Calculate part-load performance
        curves = self.part_load_curves.get(modulation_type,
                                         self.part_load_curves['single_speed'])
        
        if is_cycling:
            # For cycling operation, use cycling model
            part_load_ratio = load_factor
            cycling_degradation = self._calculate_cycling_degradation(
                part_load_ratio, cycling_params
            )
            compressor_capacity = 100.0  # Full capacity when running
            modulation_level = part_load_ratio  # Achieved through cycling
        else:
            # For modulating operation
            part_load_ratio = load_factor
            cycling_degradation = 1.0  # No cycling degradation
            compressor_capacity = curves['capacity_curve'](part_load_ratio) * 100
            modulation_level = part_load_ratio
        
        # Calculate performance metrics
        base_cop = curves['cop_curve'](part_load_ratio)
        part_load_cop = base_cop * cycling_degradation
        
        part_load_capacity = design_capacity_kW * curves['capacity_curve'](part_load_ratio)
        energy_input_ratio = curves['eir_curve'](part_load_ratio) / cycling_degradation
        part_load_power = part_load_capacity * energy_input_ratio / part_load_cop
        
        # Initial cycling state
        if is_cycling:
            cycles_per_hour = self._calculate_cycling_frequency(
                part_load_ratio, control_params, cycling_params
            )
            cycle_time = 60 / cycles_per_hour if cycles_per_hour > 0 else 60
            cycle_on_time = cycle_time * part_load_ratio
            cycle_off_time = cycle_time * (1 - part_load_ratio)
        else:
            cycles_per_hour = 0
            cycle_on_time = 0
            cycle_off_time = 0
        
        return PartLoadState(
            system_id=system_id,
            timestamp=timestamp,
            building_load_kW=current_load_kW,
            design_load_kW=design_capacity_kW,
            load_factor=load_factor,
            compressor_capacity_percent=compressor_capacity,
            modulation_type=modulation_type,
            modulation_level=modulation_level,
            is_cycling=is_cycling,
            cycle_on_time_minutes=cycle_on_time,
            cycle_off_time_minutes=cycle_off_time,
            cycles_per_hour=cycles_per_hour,
            part_load_cop=part_load_cop,
            part_load_capacity_kW=part_load_capacity,
            part_load_power_kW=part_load_power,
            part_load_ratio=part_load_ratio,
            energy_input_ratio=energy_input_ratio,
            cycling_degradation_factor=cycling_degradation,
            thermostat_deadband_K=control_params['deadband_K'],
            minimum_on_time_minutes=control_params['minimum_on_time_min'],
            minimum_off_time_minutes=control_params['minimum_off_time_min'],
            compressor_running=True if not is_cycling else part_load_ratio > 0.5,
            defrost_active=False,
            startup_transient=False
        )
    
    def update_part_load_operation(self, state: PartLoadState, 
                                  new_load_kW: float, time_step_minutes: float,
                                  ambient_temp_K: float = 298.15) -> PartLoadState:
        """Update part-load operation based on changing load."""
        
        # Update load conditions
        state.building_load_kW = new_load_kW
        state.load_factor = new_load_kW / state.design_load_kW
        state.load_factor = np.clip(state.load_factor, 0, 1.5)
        
        # Get system parameters
        cycling_params = self.cycling_degradation.get(state.modulation_type,
                                                    self.cycling_degradation['single_speed'])
        curves = self.part_load_curves.get(state.modulation_type,
                                         self.part_load_curves['single_speed'])
        
        # Determine operating mode
        was_cycling = state.is_cycling
        state.is_cycling = state.load_factor < cycling_params['minimum_plr']
        
        # Update cycling times
        if state.is_cycling:
            # Calculate new cycling pattern
            state.cycles_per_hour = self._calculate_cycling_frequency(
                state.load_factor, 
                {'deadband_K': state.thermostat_deadband_K,
                 'minimum_on_time_min': state.minimum_on_time_minutes,
                 'minimum_off_time_min': state.minimum_off_time_minutes},
                cycling_params
            )
            
            cycle_time = 60 / state.cycles_per_hour if state.cycles_per_hour > 0 else 60
            target_on_time = cycle_time * state.load_factor
            target_off_time = cycle_time * (1 - state.load_factor)
            
            # Update current cycle state
            if state.compressor_running:
                state.cycle_on_time_minutes += time_step_minutes
                if state.cycle_on_time_minutes >= target_on_time:
                    state.compressor_running = False
                    state.cycle_on_time_minutes = 0
            else:
                state.cycle_off_time_minutes += time_step_minutes
                if state.cycle_off_time_minutes >= target_off_time:
                    state.compressor_running = True
                    state.cycle_off_time_minutes = 0
                    state.startup_transient = True
            
            state.compressor_capacity_percent = 100.0 if state.compressor_running else 0.0
            state.modulation_level = state.load_factor
            
        else:
            # Modulating operation
            state.cycles_per_hour = 0
            state.cycle_on_time_minutes = 0
            state.cycle_off_time_minutes = 0
            state.compressor_running = True
            state.compressor_capacity_percent = curves['capacity_curve'](state.load_factor) * 100
            state.modulation_level = state.load_factor
        
        # Calculate performance at new operating point
        state.part_load_ratio = state.load_factor
        
        if state.is_cycling:
            state.cycling_degradation_factor = self._calculate_cycling_degradation(
                state.part_load_ratio, cycling_params
            )
        else:
            state.cycling_degradation_factor = 1.0
        
        # Update performance metrics
        base_cop = curves['cop_curve'](state.part_load_ratio)
        state.part_load_cop = base_cop * state.cycling_degradation_factor
        
        state.part_load_capacity_kW = (state.design_load_kW * 
                                     curves['capacity_curve'](state.part_load_ratio))
        
        state.energy_input_ratio = (curves['eir_curve'](state.part_load_ratio) / 
                                  state.cycling_degradation_factor)
        
        if state.part_load_cop > 0:
            state.part_load_power_kW = (state.part_load_capacity_kW * 
                                      state.energy_input_ratio / state.part_load_cop)
        else:
            state.part_load_power_kW = 0
        
        # Handle startup transient
        if state.startup_transient:
            startup_penalty = cycling_params['startup_energy_penalty']
            state.part_load_power_kW *= (1 + startup_penalty)
            state.startup_transient = False  # Reset after one time step
        
        # Update timestamp
        state.timestamp = datetime.now().isoformat()
        
        return state
    
    def simulate_daily_cycling_pattern(self, system_id: str, design_capacity_kW: float,
                                     modulation_type: str = 'single_speed',
                                     building_type: str = 'residential',
                                     thermal_mass: str = 'medium') -> List[PartLoadState]:
        """Simulate daily cycling pattern with realistic load profile."""
        
        # Generate realistic daily load profile
        hours = np.arange(0, 24, 0.25)  # 15-minute intervals
        load_profile = self._generate_daily_load_profile(building_type, thermal_mass)
        
        states = []
        current_state = None
        
        for i, hour in enumerate(hours):
            # Calculate current load
            load_factor = load_profile[i]
            current_load_kW = design_capacity_kW * load_factor
            
            if current_state is None:
                # Initialize first state
                current_state = self.initialize_part_load_state(
                    system_id, building_type, design_capacity_kW, 
                    current_load_kW, building_type, modulation_type
                )
            else:
                # Update existing state
                current_state = self.update_part_load_operation(
                    current_state, current_load_kW, 15.0  # 15-minute time step
                )
            
            # Store state (deep copy)
            state_copy = PartLoadState(**asdict(current_state))
            states.append(state_copy)
        
        return states
    
    def _calculate_cycling_degradation(self, part_load_ratio: float,
                                     cycling_params: Dict[str, float]) -> float:
        """Calculate cycling degradation factor."""
        
        if part_load_ratio >= cycling_params['minimum_plr']:
            return 1.0  # No cycling degradation
        
        # AHRI cycling degradation model
        cd = cycling_params['degradation_coefficient']
        degradation_factor = 1 - cd * (1 - part_load_ratio)
        
        return np.clip(degradation_factor, 0.1, 1.0)
    
    def _calculate_cycling_frequency(self, part_load_ratio: float,
                                   control_params: Dict[str, float],
                                   cycling_params: Dict[str, float]) -> float:
        """Calculate cycling frequency (cycles per hour)."""
        
        if part_load_ratio >= cycling_params['minimum_plr']:
            return 0  # No cycling
        
        # Simplified cycling frequency model
        # Based on thermostat deadband and building thermal mass
        
        deadband = control_params['deadband_K']
        min_on_time = control_params['minimum_on_time_min']
        min_off_time = control_params['minimum_off_time_min']
        
        # Cycle time based on load ratio and minimum times
        if part_load_ratio > 0:
            cycle_on_time = max(min_on_time, min_on_time / part_load_ratio)
            cycle_off_time = max(min_off_time, cycle_on_time * (1 - part_load_ratio) / part_load_ratio)
            total_cycle_time = cycle_on_time + cycle_off_time
            cycles_per_hour = 60 / total_cycle_time
        else:
            cycles_per_hour = 0
        
        return np.clip(cycles_per_hour, 0, 12)  # Maximum 12 cycles per hour
    
    def _generate_daily_load_profile(self, building_type: str, 
                                   thermal_mass: str) -> np.ndarray:
        """Generate realistic daily load profile."""
        
        hours = np.arange(0, 24, 0.25)
        
        # Base load profiles by building type
        if building_type == 'residential':
            # Residential cooling load profile (summer day)
            base_profile = (0.3 + 0.4 * np.sin(2 * np.pi * (hours - 6) / 24) +
                          0.2 * np.sin(2 * np.pi * (hours - 14) / 24) +
                          0.1 * np.random.normal(0, 0.1, len(hours)))
        elif building_type == 'commercial':
            # Commercial building profile
            base_profile = np.where((hours >= 8) & (hours <= 18),
                                  0.7 + 0.2 * np.sin(2 * np.pi * (hours - 12) / 12),
                                  0.2 + 0.1 * np.sin(2 * np.pi * hours / 24))
        else:
            # Industrial profile (more constant)
            base_profile = 0.6 + 0.2 * np.sin(2 * np.pi * (hours - 12) / 24)
        
        # Apply thermal mass smoothing
        mass_params = self.thermal_mass_effects.get(thermal_mass,
                                                   self.thermal_mass_effects['medium'])
        
        smoothing_factor = mass_params['load_smoothing_factor']
        if smoothing_factor > 0:
            # Simple exponential smoothing
            smoothed_profile = np.zeros_like(base_profile)
            smoothed_profile[0] = base_profile[0]
            for i in range(1, len(base_profile)):
                smoothed_profile[i] = (smoothing_factor * smoothed_profile[i-1] + 
                                     (1 - smoothing_factor) * base_profile[i])
            base_profile = smoothed_profile
        
        # Ensure positive values and reasonable range
        load_profile = np.clip(base_profile, 0.05, 1.2)
        
        return load_profile
    
    def calculate_seasonal_efficiency(self, daily_states: List[PartLoadState]) -> Dict[str, float]:
        """Calculate seasonal efficiency metrics from daily operation."""
        
        if not daily_states:
            return {}
        
        # Calculate weighted averages
        total_energy_input = 0
        total_energy_output = 0
        total_runtime_hours = 0
        total_cycles = 0
        
        for i, state in enumerate(daily_states):
            time_step_hours = 0.25  # 15-minute intervals
            
            if state.compressor_running:
                energy_input = state.part_load_power_kW * time_step_hours
                energy_output = state.part_load_capacity_kW * time_step_hours
                total_energy_input += energy_input
                total_energy_output += energy_output
                total_runtime_hours += time_step_hours
            
            # Count cycles (transitions from off to on)
            if i > 0 and not daily_states[i-1].compressor_running and state.compressor_running:
                total_cycles += 1
        
        # Calculate metrics
        seasonal_cop = total_energy_output / total_energy_input if total_energy_input > 0 else 0
        runtime_fraction = total_runtime_hours / 24.0
        average_cycles_per_hour = total_cycles / 24.0
        
        # Part-load efficiency
        design_cop = 3.5  # Typical design COP
        part_load_efficiency = seasonal_cop / design_cop if design_cop > 0 else 0
        
        return {
            'seasonal_cop': seasonal_cop,
            'runtime_fraction': runtime_fraction,
            'cycles_per_hour': average_cycles_per_hour,
            'part_load_efficiency': part_load_efficiency,
            'total_energy_input_kWh': total_energy_input,
            'total_energy_output_kWh': total_energy_output,
            'total_runtime_hours': total_runtime_hours,
            'total_cycles': total_cycles
        }
    
    def optimize_control_strategy(self, design_capacity_kW: float,
                                building_load_profile: np.ndarray,
                                available_strategies: List[str]) -> Dict[str, Any]:
        """Optimize control strategy for given load profile."""
        
        results = {}
        
        for strategy in available_strategies:
            # Simulate daily operation with this strategy
            daily_states = []
            current_state = None
            
            for i, load_factor in enumerate(building_load_profile):
                current_load_kW = design_capacity_kW * load_factor
                
                if current_state is None:
                    current_state = self.initialize_part_load_state(
                        "optimization_test", "commercial", design_capacity_kW,
                        current_load_kW, "commercial", strategy
                    )
                else:
                    current_state = self.update_part_load_operation(
                        current_state, current_load_kW, 15.0
                    )
                
                daily_states.append(PartLoadState(**asdict(current_state)))
            
            # Calculate performance metrics
            metrics = self.calculate_seasonal_efficiency(daily_states)
            results[strategy] = metrics
        
        # Find best strategy based on seasonal COP
        best_strategy = max(results.keys(), 
                          key=lambda x: results[x].get('seasonal_cop', 0))
        
        return {
            'best_strategy': best_strategy,
            'strategy_comparison': results,
            'optimization_criteria': 'seasonal_cop'
        }
    
    def save_part_load_data(self, part_load_states: List[PartLoadState],
                           output_path: str) -> None:
        """Save part-load operation data to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'part_load_states': [asdict(state) for state in part_load_states],
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'part_load_curves': {k: 'function' for k in self.part_load_curves.keys()},
                'cycling_degradation': self.cycling_degradation,
                'control_parameters': self.control_parameters,
                'thermal_mass_effects': self.thermal_mass_effects
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved part-load operation data to {output_path}")


if __name__ == "__main__":
    # Test part-load cycling models
    print("🔄 Testing Part-Load Operation and Cycling Models")
    
    # Initialize model
    part_load_model = PartLoadCyclingModel()
    
    # Test single-speed system at part load
    state = part_load_model.initialize_part_load_state(
        "test_system", "residential", 10.0, 3.0, "residential", "single_speed"
    )
    
    print(f"\n📊 Single-Speed System (30% load):")
    print(f"Load factor: {state.load_factor:.2f}")
    print(f"Is cycling: {state.is_cycling}")
    print(f"Part-load COP: {state.part_load_cop:.2f}")
    print(f"Cycles per hour: {state.cycles_per_hour:.1f}")
    print(f"Cycling degradation: {state.cycling_degradation_factor:.3f}")
    
    # Test variable speed system
    vs_state = part_load_model.initialize_part_load_state(
        "test_system", "residential", 10.0, 3.0, "residential", "variable_speed"
    )
    
    print(f"\n📊 Variable-Speed System (30% load):")
    print(f"Load factor: {vs_state.load_factor:.2f}")
    print(f"Is cycling: {vs_state.is_cycling}")
    print(f"Part-load COP: {vs_state.part_load_cop:.2f}")
    print(f"Compressor capacity: {vs_state.compressor_capacity_percent:.1f}%")
    print(f"Cycling degradation: {vs_state.cycling_degradation_factor:.3f}")
    
    # Simulate daily cycling pattern
    print(f"\n🌅 Simulating daily cycling pattern...")
    daily_states = part_load_model.simulate_daily_cycling_pattern(
        "daily_test", 8.0, "single_speed", "residential", "medium"
    )
    
    print(f"Generated {len(daily_states)} time steps (24 hours)")
    
    # Calculate seasonal efficiency
    efficiency_metrics = part_load_model.calculate_seasonal_efficiency(daily_states)
    print(f"\n📈 Daily Performance Metrics:")
    print(f"Seasonal COP: {efficiency_metrics['seasonal_cop']:.2f}")
    print(f"Runtime fraction: {efficiency_metrics['runtime_fraction']:.2f}")
    print(f"Average cycles/hour: {efficiency_metrics['cycles_per_hour']:.1f}")
    print(f"Part-load efficiency: {efficiency_metrics['part_load_efficiency']:.2f}")
    
    # Test control strategy optimization
    print(f"\n🎯 Optimizing control strategy...")
    load_profile = np.array([0.2, 0.3, 0.5, 0.8, 1.0, 0.9, 0.7, 0.4])
    strategies = ['single_speed', 'variable_speed', 'two_speed']
    
    optimization_results = part_load_model.optimize_control_strategy(
        8.0, load_profile, strategies
    )
    
    print(f"Best strategy: {optimization_results['best_strategy']}")
    for strategy, metrics in optimization_results['strategy_comparison'].items():
        print(f"{strategy}: COP = {metrics['seasonal_cop']:.2f}")
    
    # Save test data
    part_load_model.save_part_load_data(daily_states[:24], "part_load_cycling_data.json")
    
    print("\n✅ Part-load cycling models testing complete!")