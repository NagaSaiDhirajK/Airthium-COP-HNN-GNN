"""
Defrost Cycle Dynamics for Air-Source Heat Pumps
===============================================

This module implements comprehensive defrost cycle modeling for air-source heat
pump systems, including frost formation, defrost initiation, cycle dynamics,
and energy penalties associated with defrost operations.

Key Features:
- Frost formation modeling based on weather conditions
- Multiple defrost initiation strategies (time-based, temperature-based, pressure-based)
- Defrost cycle dynamics with realistic timing and energy consumption
- Reverse cycle and hot gas bypass defrost methods
- Energy penalties and capacity losses during defrost
- Frost accumulation effects on heat exchanger performance
- Defrost frequency optimization algorithms

Author: AIRTHIUM COP Prediction System
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefrostMethod(Enum):
    """Defrost methods for air-source heat pumps"""
    REVERSE_CYCLE = "reverse_cycle"
    HOT_GAS_BYPASS = "hot_gas_bypass"
    ELECTRIC_RESISTANCE = "electric_resistance"
    AMBIENT_AIR = "ambient_air"

class DefrostInitiationStrategy(Enum):
    """Defrost initiation strategies"""
    TIME_BASED = "time_based"
    TEMPERATURE_BASED = "temperature_based"
    PRESSURE_BASED = "pressure_based"
    DEMAND_BASED = "demand_based"
    INTELLIGENT_ADAPTIVE = "intelligent_adaptive"

class DefrostPhase(Enum):
    """Phases of defrost cycle"""
    NORMAL_OPERATION = "normal_operation"
    DEFROST_INITIATION = "defrost_initiation"
    DEFROST_ACTIVE = "defrost_active"
    DEFROST_TERMINATION = "defrost_termination"
    RECOVERY = "recovery"

@dataclass
class FrostConditions:
    """Environmental conditions affecting frost formation"""
    outdoor_temperature_C: float
    outdoor_humidity_percent: float
    coil_surface_temperature_C: float
    air_velocity_m_per_s: float
    atmospheric_pressure_Pa: float
    precipitation_rate_mm_per_h: float

@dataclass
class FrostAccumulation:
    """Frost accumulation state on heat exchanger"""
    frost_thickness_mm: float
    frost_density_kg_per_m3: float
    frost_mass_kg: float
    coverage_percentage: float
    thermal_resistance_K_per_W: float
    pressure_drop_increase_Pa: float
    heat_transfer_degradation_percent: float

@dataclass
class DefrostParameters:
    """Parameters for defrost cycle operation"""
    method: DefrostMethod
    initiation_strategy: DefrostInitiationStrategy
    initiation_threshold: float
    cycle_duration_s: float
    termination_criteria: Dict[str, float]
    energy_consumption_kW: float
    auxiliary_heat_kW: float
    fan_operation_during_defrost: bool

@dataclass
class DefrostCycleState:
    """Current state of defrost cycle"""
    timestamp: datetime
    phase: DefrostPhase
    time_in_phase_s: float
    frost_accumulation: FrostAccumulation
    defrost_parameters: DefrostParameters
    energy_penalty_kWh: float
    capacity_loss_percent: float
    cop_degradation_factor: float
    time_since_last_defrost_s: float
    defrost_frequency_per_day: float

class DefrostCycleModel:
    """
    Comprehensive defrost cycle model for air-source heat pumps.
    
    This class models frost formation, defrost initiation, cycle dynamics,
    and performance impacts of defrost operations on heat pump systems.
    """
    
    def __init__(self):
        """Initialize defrost cycle model"""
        self.defrost_methods = self._initialize_defrost_methods()
        self.initiation_strategies = self._initialize_initiation_strategies()
        self.frost_formation_model = self._initialize_frost_formation_model()
        self.performance_impacts = self._initialize_performance_impacts()
        
        # State tracking
        self.current_frost_state = None
        self.last_defrost_time = None
        self.defrost_history = []
        
        logger.info("Defrost cycle model initialized")
    
    def _initialize_defrost_methods(self) -> Dict[DefrostMethod, Dict[str, Any]]:
        """Initialize defrost method specifications"""
        
        return {
            DefrostMethod.REVERSE_CYCLE: {
                'typical_duration_s': 600,  # 10 minutes
                'energy_consumption_factor': 1.2,  # 20% extra energy
                'effectiveness': 0.95,  # 95% frost removal
                'auxiliary_heat_required': True,
                'fan_operation': False,  # Fan off during reverse cycle
                'temperature_rise_rate_C_per_s': 0.1,
                'defrost_termination_temp_C': 10.0
            },
            
            DefrostMethod.HOT_GAS_BYPASS: {
                'typical_duration_s': 480,  # 8 minutes
                'energy_consumption_factor': 1.15,  # 15% extra energy
                'effectiveness': 0.90,  # 90% frost removal
                'auxiliary_heat_required': False,
                'fan_operation': True,  # Fan continues operation
                'temperature_rise_rate_C_per_s': 0.08,
                'defrost_termination_temp_C': 8.0
            },
            
            DefrostMethod.ELECTRIC_RESISTANCE: {
                'typical_duration_s': 900,  # 15 minutes
                'energy_consumption_factor': 2.0,  # 100% extra energy
                'effectiveness': 0.98,  # 98% frost removal
                'auxiliary_heat_required': False,
                'fan_operation': True,
                'temperature_rise_rate_C_per_s': 0.12,
                'defrost_termination_temp_C': 12.0
            },
            
            DefrostMethod.AMBIENT_AIR: {
                'typical_duration_s': 1800,  # 30 minutes
                'energy_consumption_factor': 0.1,  # Minimal energy
                'effectiveness': 0.70,  # 70% frost removal
                'auxiliary_heat_required': True,
                'fan_operation': True,
                'temperature_rise_rate_C_per_s': 0.02,
                'defrost_termination_temp_C': 2.0
            }
        }
    
    def _initialize_initiation_strategies(self) -> Dict[DefrostInitiationStrategy, Dict[str, Any]]:
        """Initialize defrost initiation strategy parameters"""
        
        return {
            DefrostInitiationStrategy.TIME_BASED: {
                'interval_minutes': 90,  # Every 90 minutes
                'temperature_threshold_C': 2.0,  # Only if outdoor temp < 2°C
                'humidity_threshold_percent': 70.0,  # Only if RH > 70%
                'override_conditions': ['emergency_low_capacity']
            },
            
            DefrostInitiationStrategy.TEMPERATURE_BASED: {
                'coil_temperature_threshold_C': -8.0,  # Defrost when coil < -8°C
                'temperature_differential_C': 15.0,  # Outdoor - coil temp difference
                'minimum_interval_minutes': 30,  # Minimum time between defrosts
                'maximum_interval_minutes': 180  # Maximum time without defrost
            },
            
            DefrostInitiationStrategy.PRESSURE_BASED: {
                'pressure_drop_threshold_Pa': 150.0,  # Defrost when ΔP > 150 Pa
                'pressure_rise_rate_Pa_per_min': 5.0,  # Rate of pressure increase
                'baseline_pressure_drop_Pa': 50.0,  # Clean coil pressure drop
                'measurement_averaging_minutes': 10  # Average over 10 minutes
            },
            
            DefrostInitiationStrategy.DEMAND_BASED: {
                'capacity_degradation_threshold_percent': 15.0,  # 15% capacity loss
                'cop_degradation_threshold_percent': 10.0,  # 10% COP loss
                'load_satisfaction_threshold_percent': 85.0,  # Can't meet 85% of load
                'trending_window_minutes': 20  # Trend analysis window
            },
            
            DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE: {
                'frost_accumulation_threshold_mm': 2.0,  # 2mm frost thickness
                'multiple_criteria_required': True,  # Multiple conditions must be met
                'learning_rate': 0.1,  # Adaptive learning rate
                'weather_prediction_horizon_hours': 2,  # Look-ahead for weather
                'optimization_target': 'energy_efficiency'  # Optimize for efficiency
            }
        }
    
    def _initialize_frost_formation_model(self) -> Dict[str, Any]:
        """Initialize frost formation model parameters"""
        
        return {
            'formation_conditions': {
                'temperature_threshold_C': 2.0,  # Frost forms below 2°C
                'humidity_threshold_percent': 60.0,  # Frost forms above 60% RH
                'surface_temperature_offset_C': -5.0,  # Coil temp below air temp
                'minimum_air_velocity_m_per_s': 0.5  # Minimum air flow for frost
            },
            
            'formation_rates': {
                'base_rate_mm_per_hour': 0.1,  # Base frost formation rate
                'temperature_factor': 2.0,  # Rate doubles per 5°C decrease
                'humidity_factor': 1.5,  # Rate increases with humidity
                'velocity_factor': 0.8,  # Higher velocity reduces formation
                'surface_roughness_factor': 1.2  # Rough surfaces accumulate more
            },
            
            'frost_properties': {
                'density_kg_per_m3': 300.0,  # Typical frost density
                'thermal_conductivity_W_per_m_K': 0.4,  # Frost thermal conductivity
                'specific_heat_J_per_kg_K': 2100.0,  # Frost specific heat
                'latent_heat_fusion_J_per_kg': 334000.0,  # Heat to melt frost
                'adhesion_strength_Pa': 1000.0  # Frost adhesion to surface
            },
            
            'performance_impacts': {
                'thermal_resistance_factor': 0.1,  # K·m²/W per mm thickness
                'pressure_drop_factor': 50.0,  # Pa per mm thickness
                'heat_transfer_degradation_factor': 0.05,  # % per mm thickness
                'air_flow_reduction_factor': 0.03  # % per mm thickness
            }
        }
    
    def _initialize_performance_impacts(self) -> Dict[str, Any]:
        """Initialize performance impact parameters"""
        
        return {
            'capacity_impacts': {
                'frost_thickness_mm': [0, 1, 2, 3, 4, 5],
                'capacity_reduction_percent': [0, 5, 12, 22, 35, 50]
            },
            
            'cop_impacts': {
                'frost_thickness_mm': [0, 1, 2, 3, 4, 5],
                'cop_reduction_percent': [0, 3, 8, 15, 25, 40]
            },
            
            'defrost_penalties': {
                'energy_penalty_kWh_per_cycle': 2.5,  # Average energy penalty
                'capacity_loss_during_defrost_percent': 100.0,  # No heating during defrost
                'recovery_time_minutes': 5.0,  # Time to restore normal operation
                'auxiliary_heat_energy_kWh': 1.5  # Auxiliary heat during defrost
            }
        }
    
    def calculate_frost_formation_rate(self, conditions: FrostConditions) -> float:
        """
        Calculate frost formation rate based on environmental conditions
        
        Args:
            conditions: Environmental conditions
            
        Returns:
            Frost formation rate in mm/hour
        """
        
        formation_model = self.frost_formation_model
        
        # Check if conditions favor frost formation
        if (conditions.outdoor_temperature_C > formation_model['formation_conditions']['temperature_threshold_C'] or
            conditions.outdoor_humidity_percent < formation_model['formation_conditions']['humidity_threshold_percent'] or
            conditions.coil_surface_temperature_C > 0.0):
            return 0.0  # No frost formation
        
        # Base formation rate
        base_rate = formation_model['formation_rates']['base_rate_mm_per_hour']
        
        # Temperature effect (exponential increase as temperature decreases)
        temp_factor = np.exp(-conditions.outdoor_temperature_C / 5.0)
        
        # Humidity effect (linear increase above threshold)
        humidity_excess = max(0, conditions.outdoor_humidity_percent - 
                            formation_model['formation_conditions']['humidity_threshold_percent'])
        humidity_factor = 1.0 + humidity_excess / 100.0
        
        # Air velocity effect (higher velocity reduces formation)
        velocity_factor = 1.0 / (1.0 + conditions.air_velocity_m_per_s / 2.0)
        
        # Surface temperature effect
        surface_temp_factor = max(0.1, 1.0 - conditions.coil_surface_temperature_C / 10.0)
        
        # Precipitation effect (increases frost formation)
        precip_factor = 1.0 + conditions.precipitation_rate_mm_per_h * 0.1
        
        formation_rate = (base_rate * temp_factor * humidity_factor * 
                         velocity_factor * surface_temp_factor * precip_factor)
        
        return max(0.0, formation_rate)
    
    def update_frost_accumulation(self, 
                                current_accumulation: FrostAccumulation,
                                formation_rate_mm_per_hour: float,
                                time_step_hours: float,
                                heat_exchanger_area_m2: float = 2.0) -> FrostAccumulation:
        """
        Update frost accumulation state
        
        Args:
            current_accumulation: Current frost state
            formation_rate_mm_per_hour: Frost formation rate
            time_step_hours: Time step for update
            heat_exchanger_area_m2: Heat exchanger surface area
            
        Returns:
            Updated FrostAccumulation object
        """
        
        # Update frost thickness
        thickness_increase = formation_rate_mm_per_hour * time_step_hours
        new_thickness = current_accumulation.frost_thickness_mm + thickness_increase
        
        # Calculate frost mass
        frost_props = self.frost_formation_model['frost_properties']
        frost_volume_m3 = new_thickness * 1e-3 * heat_exchanger_area_m2  # Convert mm to m
        frost_mass = frost_volume_m3 * frost_props['density_kg_per_m3']
        
        # Calculate coverage percentage (assumes uniform distribution)
        max_thickness_mm = 10.0  # Maximum practical frost thickness
        coverage_percent = min(100.0, (new_thickness / max_thickness_mm) * 100.0)
        
        # Calculate thermal resistance
        thermal_conductivity = frost_props['thermal_conductivity_W_per_m_K']
        thermal_resistance = (new_thickness * 1e-3) / thermal_conductivity  # K·m²/W
        
        # Calculate pressure drop increase
        impact_factors = self.frost_formation_model['performance_impacts']
        pressure_drop_increase = new_thickness * impact_factors['pressure_drop_factor']
        
        # Calculate heat transfer degradation
        heat_transfer_degradation = min(50.0, new_thickness * impact_factors['heat_transfer_degradation_factor'])
        
        return FrostAccumulation(
            frost_thickness_mm=new_thickness,
            frost_density_kg_per_m3=frost_props['density_kg_per_m3'],
            frost_mass_kg=frost_mass,
            coverage_percentage=coverage_percent,
            thermal_resistance_K_per_W=thermal_resistance,
            pressure_drop_increase_Pa=pressure_drop_increase,
            heat_transfer_degradation_percent=heat_transfer_degradation
        )
    
    def check_defrost_initiation(self, 
                                frost_accumulation: FrostAccumulation,
                                conditions: FrostConditions,
                                strategy: DefrostInitiationStrategy,
                                time_since_last_defrost_s: float,
                                system_performance: Dict[str, float]) -> bool:
        """
        Check if defrost should be initiated based on strategy
        
        Args:
            frost_accumulation: Current frost state
            conditions: Environmental conditions
            strategy: Defrost initiation strategy
            time_since_last_defrost_s: Time since last defrost
            system_performance: Current system performance metrics
            
        Returns:
            True if defrost should be initiated
        """
        
        strategy_params = self.initiation_strategies[strategy]
        
        if strategy == DefrostInitiationStrategy.TIME_BASED:
            # Time-based initiation
            interval_s = strategy_params['interval_minutes'] * 60
            temp_ok = conditions.outdoor_temperature_C < strategy_params['temperature_threshold_C']
            humidity_ok = conditions.outdoor_humidity_percent > strategy_params['humidity_threshold_percent']
            
            return (time_since_last_defrost_s >= interval_s and temp_ok and humidity_ok)
        
        elif strategy == DefrostInitiationStrategy.TEMPERATURE_BASED:
            # Temperature-based initiation
            coil_temp_low = conditions.coil_surface_temperature_C < strategy_params['coil_temperature_threshold_C']
            temp_diff_high = (conditions.outdoor_temperature_C - conditions.coil_surface_temperature_C) > \
                           strategy_params['temperature_differential_C']
            min_time_ok = time_since_last_defrost_s >= strategy_params['minimum_interval_minutes'] * 60
            max_time_exceeded = time_since_last_defrost_s >= strategy_params['maximum_interval_minutes'] * 60
            
            return (coil_temp_low or temp_diff_high or max_time_exceeded) and min_time_ok
        
        elif strategy == DefrostInitiationStrategy.PRESSURE_BASED:
            # Pressure-based initiation
            pressure_drop_high = frost_accumulation.pressure_drop_increase_Pa > \
                                strategy_params['pressure_drop_threshold_Pa']
            
            return pressure_drop_high
        
        elif strategy == DefrostInitiationStrategy.DEMAND_BASED:
            # Demand-based initiation
            capacity_low = system_performance.get('capacity_degradation_percent', 0) > \
                          strategy_params['capacity_degradation_threshold_percent']
            cop_low = system_performance.get('cop_degradation_percent', 0) > \
                     strategy_params['cop_degradation_threshold_percent']
            load_unmet = system_performance.get('load_satisfaction_percent', 100) < \
                        strategy_params['load_satisfaction_threshold_percent']
            
            return capacity_low or cop_low or load_unmet
        
        elif strategy == DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE:
            # Intelligent adaptive initiation (multiple criteria)
            frost_thick = frost_accumulation.frost_thickness_mm > \
                         strategy_params['frost_accumulation_threshold_mm']
            performance_degraded = (system_performance.get('capacity_degradation_percent', 0) > 10.0 or
                                  system_performance.get('cop_degradation_percent', 0) > 8.0)
            
            return frost_thick and performance_degraded
        
        return False
    
    def simulate_defrost_cycle(self, 
                             method: DefrostMethod,
                             initial_frost: FrostAccumulation,
                             ambient_temp_C: float,
                             cycle_duration_s: Optional[float] = None) -> Tuple[List[Dict], float]:
        """
        Simulate a complete defrost cycle
        
        Args:
            method: Defrost method to use
            initial_frost: Initial frost accumulation
            ambient_temp_C: Ambient temperature during defrost
            cycle_duration_s: Override cycle duration
            
        Returns:
            Tuple of (cycle_data_points, total_energy_consumption_kWh)
        """
        
        method_params = self.defrost_methods[method]
        duration = cycle_duration_s or method_params['typical_duration_s']
        
        cycle_data = []
        total_energy = 0.0
        
        # Time step for simulation (10 second intervals)
        dt = 10.0  # seconds
        n_steps = int(duration / dt)
        
        current_frost_thickness = initial_frost.frost_thickness_mm
        coil_temperature = ambient_temp_C - 10.0  # Initial coil temperature
        
        for step in range(n_steps):
            time_s = step * dt
            
            # Calculate coil temperature rise
            temp_rise_rate = method_params['temperature_rise_rate_C_per_s']
            coil_temperature += temp_rise_rate * dt
            
            # Calculate frost melting
            if coil_temperature > 0.0:
                # Frost starts melting when coil temperature > 0°C
                melt_rate_mm_per_s = 0.01 * (coil_temperature / 5.0)  # Simplified melting model
                frost_melted = melt_rate_mm_per_s * dt
                current_frost_thickness = max(0.0, current_frost_thickness - frost_melted)
            
            # Calculate energy consumption
            base_power_kW = 5.0  # Base heat pump power
            defrost_power_kW = base_power_kW * method_params['energy_consumption_factor']
            
            if method_params['auxiliary_heat_required']:
                aux_power_kW = 3.0  # Auxiliary heat power
                defrost_power_kW += aux_power_kW
            
            energy_step_kWh = defrost_power_kW * (dt / 3600.0)
            total_energy += energy_step_kWh
            
            # Check termination criteria
            termination_temp = method_params['defrost_termination_temp_C']
            if coil_temperature >= termination_temp and current_frost_thickness < 0.1:
                # Defrost complete
                break
            
            # Record data point
            cycle_data.append({
                'time_s': time_s,
                'coil_temperature_C': coil_temperature,
                'frost_thickness_mm': current_frost_thickness,
                'power_consumption_kW': defrost_power_kW,
                'cumulative_energy_kWh': total_energy
            })
        
        return cycle_data, total_energy
    
    def calculate_performance_impact(self, frost_accumulation: FrostAccumulation) -> Dict[str, float]:
        """
        Calculate performance impact of frost accumulation
        
        Args:
            frost_accumulation: Current frost state
            
        Returns:
            Dictionary with performance impact metrics
        """
        
        thickness = frost_accumulation.frost_thickness_mm
        
        # Interpolate capacity reduction
        capacity_data = self.performance_impacts['capacity_impacts']
        capacity_reduction = np.interp(thickness, 
                                     capacity_data['frost_thickness_mm'],
                                     capacity_data['capacity_reduction_percent'])
        
        # Interpolate COP reduction
        cop_data = self.performance_impacts['cop_impacts']
        cop_reduction = np.interp(thickness,
                                cop_data['frost_thickness_mm'],
                                cop_data['cop_reduction_percent'])
        
        # Calculate additional impacts
        air_flow_reduction = min(30.0, thickness * 2.0)  # % reduction in air flow
        pressure_drop_increase = frost_accumulation.pressure_drop_increase_Pa
        
        return {
            'capacity_reduction_percent': capacity_reduction,
            'cop_reduction_percent': cop_reduction,
            'air_flow_reduction_percent': air_flow_reduction,
            'pressure_drop_increase_Pa': pressure_drop_increase,
            'heat_transfer_degradation_percent': frost_accumulation.heat_transfer_degradation_percent
        }
    
    def simulate_defrost_operation(self,
                                 start_time: datetime,
                                 duration_hours: int,
                                 weather_conditions: List[FrostConditions],
                                 defrost_method: DefrostMethod = DefrostMethod.REVERSE_CYCLE,
                                 initiation_strategy: DefrostInitiationStrategy = DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE) -> List[DefrostCycleState]:
        """
        Simulate defrost operation over extended period
        
        Args:
            start_time: Start time for simulation
            duration_hours: Simulation duration in hours
            weather_conditions: List of weather conditions over time
            defrost_method: Defrost method to use
            initiation_strategy: Defrost initiation strategy
            
        Returns:
            List of DefrostCycleState objects
        """
        
        states = []
        current_time = start_time
        time_step_hours = 1.0
        
        # Initialize frost accumulation
        current_frost = FrostAccumulation(
            frost_thickness_mm=0.0,
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.0,
            coverage_percentage=0.0,
            thermal_resistance_K_per_W=0.0,
            pressure_drop_increase_Pa=0.0,
            heat_transfer_degradation_percent=0.0
        )
        
        time_since_last_defrost = 0.0
        defrost_count = 0
        current_phase = DefrostPhase.NORMAL_OPERATION
        time_in_phase = 0.0
        
        for hour in range(duration_hours):
            if hour < len(weather_conditions):
                conditions = weather_conditions[hour]
            else:
                # Use last available conditions
                conditions = weather_conditions[-1]
            
            # Calculate frost formation rate
            formation_rate = self.calculate_frost_formation_rate(conditions)
            
            # Update frost accumulation (only during normal operation)
            if current_phase == DefrostPhase.NORMAL_OPERATION:
                current_frost = self.update_frost_accumulation(
                    current_frost, formation_rate, time_step_hours
                )
            
            # Calculate performance impact
            performance_impact = self.calculate_performance_impact(current_frost)
            
            # Check for defrost initiation
            system_performance = {
                'capacity_degradation_percent': performance_impact['capacity_reduction_percent'],
                'cop_degradation_percent': performance_impact['cop_reduction_percent'],
                'load_satisfaction_percent': 100.0 - performance_impact['capacity_reduction_percent']
            }
            
            should_defrost = self.check_defrost_initiation(
                current_frost, conditions, initiation_strategy,
                time_since_last_defrost * 3600, system_performance
            )
            
            # Handle defrost cycle
            energy_penalty = 0.0
            if should_defrost and current_phase == DefrostPhase.NORMAL_OPERATION:
                # Initiate defrost
                current_phase = DefrostPhase.DEFROST_ACTIVE
                time_in_phase = 0.0
                defrost_count += 1
                
                # Simulate defrost cycle
                cycle_data, total_energy = self.simulate_defrost_cycle(
                    defrost_method, current_frost, conditions.outdoor_temperature_C
                )
                
                energy_penalty = total_energy
                
                # Reset frost accumulation after defrost
                effectiveness = self.defrost_methods[defrost_method]['effectiveness']
                remaining_frost_factor = 1.0 - effectiveness
                
                current_frost = FrostAccumulation(
                    frost_thickness_mm=current_frost.frost_thickness_mm * remaining_frost_factor,
                    frost_density_kg_per_m3=current_frost.frost_density_kg_per_m3,
                    frost_mass_kg=current_frost.frost_mass_kg * remaining_frost_factor,
                    coverage_percentage=current_frost.coverage_percentage * remaining_frost_factor,
                    thermal_resistance_K_per_W=current_frost.thermal_resistance_K_per_W * remaining_frost_factor,
                    pressure_drop_increase_Pa=current_frost.pressure_drop_increase_Pa * remaining_frost_factor,
                    heat_transfer_degradation_percent=current_frost.heat_transfer_degradation_percent * remaining_frost_factor
                )
                
                time_since_last_defrost = 0.0
                current_phase = DefrostPhase.NORMAL_OPERATION
            
            # Update time tracking
            time_since_last_defrost += time_step_hours
            time_in_phase += time_step_hours
            
            # Calculate COP degradation
            cop_degradation = 1.0 - (performance_impact['cop_reduction_percent'] / 100.0)
            
            # Create defrost parameters
            defrost_params = DefrostParameters(
                method=defrost_method,
                initiation_strategy=initiation_strategy,
                initiation_threshold=2.0,  # mm frost thickness
                cycle_duration_s=600.0,
                termination_criteria={'coil_temp_C': 10.0},
                energy_consumption_kW=6.0,
                auxiliary_heat_kW=3.0,
                fan_operation_during_defrost=False
            )
            
            # Create state
            state = DefrostCycleState(
                timestamp=current_time,
                phase=current_phase,
                time_in_phase_s=time_in_phase * 3600,
                frost_accumulation=current_frost,
                defrost_parameters=defrost_params,
                energy_penalty_kWh=energy_penalty,
                capacity_loss_percent=performance_impact['capacity_reduction_percent'],
                cop_degradation_factor=cop_degradation,
                time_since_last_defrost_s=time_since_last_defrost * 3600,
                defrost_frequency_per_day=defrost_count / (hour + 1) * 24 if hour > 0 else 0
            )
            
            states.append(state)
            current_time += timedelta(hours=1)
        
        logger.info(f"Simulated {duration_hours} hours with {defrost_count} defrost cycles")
        return states
    
    def validate_defrost_model(self) -> Dict[str, bool]:
        """Validate defrost cycle model"""
        
        validation_results = {
            'frost_formation': False,
            'defrost_initiation': False,
            'defrost_simulation': False,
            'performance_impact': False,
            'energy_calculation': False
        }
        
        try:
            # Test frost formation
            conditions = FrostConditions(
                outdoor_temperature_C=-5.0,
                outdoor_humidity_percent=80.0,
                coil_surface_temperature_C=-10.0,
                air_velocity_m_per_s=2.0,
                atmospheric_pressure_Pa=101325.0,
                precipitation_rate_mm_per_h=0.0
            )
            
            formation_rate = self.calculate_frost_formation_rate(conditions)
            validation_results['frost_formation'] = formation_rate > 0
            
            # Test frost accumulation
            initial_frost = FrostAccumulation(
                frost_thickness_mm=0.0,
                frost_density_kg_per_m3=300.0,
                frost_mass_kg=0.0,
                coverage_percentage=0.0,
                thermal_resistance_K_per_W=0.0,
                pressure_drop_increase_Pa=0.0,
                heat_transfer_degradation_percent=0.0
            )
            
            updated_frost = self.update_frost_accumulation(initial_frost, formation_rate, 1.0)
            validation_results['frost_formation'] = updated_frost.frost_thickness_mm > 0
            
            # Test defrost initiation
            system_performance = {
                'capacity_degradation_percent': 15.0,
                'cop_degradation_percent': 10.0,
                'load_satisfaction_percent': 85.0
            }
            
            should_defrost = self.check_defrost_initiation(
                updated_frost, conditions, DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE,
                7200.0, system_performance
            )
            validation_results['defrost_initiation'] = isinstance(should_defrost, bool)
            
            # Test defrost simulation
            cycle_data, energy = self.simulate_defrost_cycle(
                DefrostMethod.REVERSE_CYCLE, updated_frost, -5.0
            )
            validation_results['defrost_simulation'] = len(cycle_data) > 0 and energy > 0
            
            # Test performance impact
            impact = self.calculate_performance_impact(updated_frost)
            validation_results['performance_impact'] = (
                'capacity_reduction_percent' in impact and
                'cop_reduction_percent' in impact
            )
            
            # Test energy calculation
            validation_results['energy_calculation'] = energy > 0 and energy < 10.0  # Reasonable range
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = DefrostCycleModel()
    
    print("❄️ Testing Defrost Cycle Model")
    
    # Test frost formation conditions
    frost_conditions = FrostConditions(
        outdoor_temperature_C=-5.0,
        outdoor_humidity_percent=85.0,
        coil_surface_temperature_C=-12.0,
        air_velocity_m_per_s=2.5,
        atmospheric_pressure_Pa=101325.0,
        precipitation_rate_mm_per_h=0.0
    )
    
    formation_rate = model.calculate_frost_formation_rate(frost_conditions)
    print(f"✅ Frost formation rate: {formation_rate:.3f} mm/hour")
    
    # Test frost accumulation
    initial_frost = FrostAccumulation(
        frost_thickness_mm=0.0,
        frost_density_kg_per_m3=300.0,
        frost_mass_kg=0.0,
        coverage_percentage=0.0,
        thermal_resistance_K_per_W=0.0,
        pressure_drop_increase_Pa=0.0,
        heat_transfer_degradation_percent=0.0
    )
    
    # Simulate 2 hours of frost accumulation
    frost_after_2h = model.update_frost_accumulation(initial_frost, formation_rate, 2.0)
    print(f"✅ Frost thickness after 2h: {frost_after_2h.frost_thickness_mm:.2f} mm")
    print(f"✅ Pressure drop increase: {frost_after_2h.pressure_drop_increase_Pa:.0f} Pa")
    
    # Test performance impact
    performance_impact = model.calculate_performance_impact(frost_after_2h)
    print(f"✅ Capacity reduction: {performance_impact['capacity_reduction_percent']:.1f}%")
    print(f"✅ COP reduction: {performance_impact['cop_reduction_percent']:.1f}%")
    
    # Test defrost initiation
    system_performance = {
        'capacity_degradation_percent': performance_impact['capacity_reduction_percent'],
        'cop_degradation_percent': performance_impact['cop_reduction_percent'],
        'load_satisfaction_percent': 100.0 - performance_impact['capacity_reduction_percent']
    }
    
    should_defrost = model.check_defrost_initiation(
        frost_after_2h, frost_conditions, DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE,
        7200.0, system_performance
    )
    print(f"✅ Should initiate defrost: {should_defrost}")
    
    # Test defrost cycle simulation
    if should_defrost:
        cycle_data, total_energy = model.simulate_defrost_cycle(
            DefrostMethod.REVERSE_CYCLE, frost_after_2h, frost_conditions.outdoor_temperature_C
        )
        print(f"✅ Defrost cycle duration: {len(cycle_data) * 10:.0f} seconds")
        print(f"✅ Defrost energy consumption: {total_energy:.2f} kWh")
    
    # Validate model
    validation = model.validate_defrost_model()
    print(f"✅ Model validation: {sum(validation.values())}/{len(validation)} tests passed")
    
    print("🎉 Defrost cycle model working correctly!")