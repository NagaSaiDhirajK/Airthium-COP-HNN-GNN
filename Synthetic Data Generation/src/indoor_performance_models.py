
"""
Indoor Performance Variation Models for Heat Engine Systems
===========================================================

This module implements performance modeling for indoor heat engine/heat pump systems,
focusing on internal operational variations rather than seasonal outdoor effects.
Designed for systems like AIRTHIUM's Stirling engine-based heat pumps that operate
in controlled indoor environments.

Key Features:
- Indoor ambient temperature variations (room temperature fluctuations)
- Equipment aging and wear patterns over time
- Load demand variations throughout day/week/year
- Maintenance cycle effects on performance
- Internal component temperature variations
- Power supply quality variations
- Operational mode transitions (startup, steady-state, shutdown)

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
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OperationalMode(Enum):
    """Operational modes for indoor equipment"""
    STARTUP = "startup"
    STEADY_STATE = "steady_state"
    PART_LOAD = "part_load"
    SHUTDOWN = "shutdown"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"

class LoadPattern(Enum):
    """Load demand patterns"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    LABORATORY = "laboratory"

@dataclass
class IndoorConditions:
    """Indoor environmental conditions"""
    timestamp: datetime
    ambient_temperature_C: float
    relative_humidity_percent: float
    atmospheric_pressure_Pa: float
    air_circulation_rate_m3_per_h: float
    equipment_room_temperature_C: float
    power_supply_voltage_V: float
    power_supply_frequency_Hz: float
    vibration_level_mm_per_s: float

@dataclass
class EquipmentState:
    """Current equipment operational state"""
    operational_mode: OperationalMode
    load_demand_percent: float
    operating_hours_total: float
    cycles_completed: int
    last_maintenance_hours: float
    component_temperatures_C: Dict[str, float]
    internal_pressures_Pa: Dict[str, float]
    flow_rates_kg_per_s: Dict[str, float]

@dataclass
class PerformanceFactors:
    """Performance degradation and enhancement factors"""
    aging_factor: float
    fouling_factor: float
    wear_factor: float
    maintenance_factor: float
    load_efficiency_factor: float
    temperature_factor: float
    power_quality_factor: float

@dataclass
class IndoorPerformanceData:
    """Indoor performance data point"""
    timestamp: datetime
    indoor_conditions: IndoorConditions
    equipment_state: EquipmentState
    performance_factors: PerformanceFactors
    heat_engine_cop: float
    thermal_efficiency_percent: float
    energy_consumption_kWh: float
    heat_output_kW: float
    maintenance_indicator: float

class IndoorPerformanceModel:
    """
    Indoor performance model for heat engine systems.
    
    This class models performance variations due to indoor operational
    conditions, equipment aging, and load patterns rather than outdoor
    seasonal effects.
    """
    
    def __init__(self, equipment_type: str = "stirling_heat_engine"):
        """Initialize indoor performance model"""
        self.equipment_type = equipment_type
        self.operational_parameters = self._initialize_operational_parameters()
        self.aging_models = self._initialize_aging_models()
        self.load_patterns = self._initialize_load_patterns()
        
        logger.info(f"Indoor performance model initialized for {equipment_type}")
    
    def _initialize_operational_parameters(self) -> Dict[str, Any]:
        """Initialize operational parameters for indoor equipment"""
        
        return {
            'nominal_conditions': {
                'ambient_temperature_C': 20.0,
                'relative_humidity_percent': 50.0,
                'atmospheric_pressure_Pa': 101325.0,
                'power_supply_voltage_V': 230.0,
                'power_supply_frequency_Hz': 50.0
            },
            
            'operating_ranges': {
                'ambient_temp_range_C': (15.0, 30.0),
                'humidity_range_percent': (30.0, 70.0),
                'pressure_range_Pa': (98000.0, 105000.0),
                'voltage_range_V': (207.0, 253.0),  # ±10% of nominal
                'frequency_range_Hz': (49.5, 50.5)
            },
            
            'performance_curves': {
                'temperature_coefficient': -0.002,  # COP change per °C
                'humidity_coefficient': -0.0005,   # COP change per % RH
                'voltage_coefficient': 0.001,      # COP change per V
                'frequency_coefficient': 0.01      # COP change per Hz
            },
            
            'component_temperatures': {
                'hot_side_nominal_C': 200.0,
                'cold_side_nominal_C': 40.0,
                'regenerator_nominal_C': 120.0,
                'displacer_nominal_C': 80.0,
                'power_piston_nominal_C': 60.0
            }
        }
    
    def _initialize_aging_models(self) -> Dict[str, Any]:
        """Initialize equipment aging and degradation models"""
        
        return {
            'wear_rates': {
                'seals_degradation_per_1000h': 0.001,
                'bearing_wear_per_1000h': 0.0005,
                'heat_exchanger_fouling_per_1000h': 0.002,
                'regenerator_degradation_per_1000h': 0.0008,
                'insulation_degradation_per_1000h': 0.0003
            },
            
            'maintenance_cycles': {
                'minor_maintenance_hours': 1000.0,
                'major_maintenance_hours': 8760.0,  # Annual
                'overhaul_hours': 26280.0,          # 3 years
                'performance_recovery_minor': 0.95,
                'performance_recovery_major': 0.98,
                'performance_recovery_overhaul': 1.0
            }
        }
    
    def _initialize_load_patterns(self) -> Dict[str, Any]:
        """Initialize load demand patterns"""
        
        return {
            'residential': {
                'daily_profile': [0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.5, 0.4, 0.4,
                                0.5, 0.4, 0.4, 0.5, 0.7, 0.9, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4],
                'weekly_variation': 0.2,
                'seasonal_variation': 0.3
            },
            
            'commercial': {
                'daily_profile': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1],
                'weekly_variation': 0.3,
                'seasonal_variation': 0.2
            },
            
            'industrial': {
                'daily_profile': [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8],
                'weekly_variation': 0.1,
                'seasonal_variation': 0.15
            },
            
            'laboratory': {
                'daily_profile': [0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4],
                'weekly_variation': 0.25,
                'seasonal_variation': 0.1
            }
        }
    
    def generate_indoor_conditions(self, 
                                 timestamp: datetime,
                                 facility_params: Optional[Dict] = None) -> IndoorConditions:
        """
        Generate indoor environmental conditions
        
        Args:
            timestamp: Current timestamp
            facility_params: Facility-specific parameters
            
        Returns:
            IndoorConditions object
        """
        
        if facility_params is None:
            facility_params = {
                'hvac_control_quality': 'good',  # poor, average, good, excellent
                'power_supply_stability': 'stable',  # unstable, variable, stable, regulated
                'vibration_environment': 'low'  # low, medium, high
            }
        
        nominal = self.operational_parameters['nominal_conditions']
        ranges = self.operational_parameters['operating_ranges']
        
        # Indoor temperature variations (much smaller than outdoor)
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        day_of_year = timestamp.timetuple().tm_yday
        
        # HVAC control quality affects temperature stability
        hvac_quality_factors = {
            'poor': {'temp_variation': 3.0, 'humidity_variation': 15.0},
            'average': {'temp_variation': 2.0, 'humidity_variation': 10.0},
            'good': {'temp_variation': 1.0, 'humidity_variation': 5.0},
            'excellent': {'temp_variation': 0.5, 'humidity_variation': 3.0}
        }
        
        hvac_factor = hvac_quality_factors[facility_params['hvac_control_quality']]
        
        # Daily temperature cycle (much smaller than outdoor)
        daily_temp_variation = hvac_factor['temp_variation'] * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)
        
        # Annual temperature drift (building heating/cooling efficiency)
        annual_temp_drift = 1.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        
        # Random temperature fluctuations
        temp_noise = np.random.normal(0, hvac_factor['temp_variation'] * 0.3)
        
        ambient_temperature = (nominal['ambient_temperature_C'] + 
                             daily_temp_variation + 
                             annual_temp_drift + 
                             temp_noise)
        
        # Equipment room temperature (slightly higher due to equipment heat)
        equipment_room_temp = ambient_temperature + np.random.uniform(2.0, 8.0)
        
        # Humidity variations
        humidity_daily = hvac_factor['humidity_variation'] * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
        humidity_noise = np.random.normal(0, hvac_factor['humidity_variation'] * 0.2)
        
        relative_humidity = np.clip(
            nominal['relative_humidity_percent'] + humidity_daily + humidity_noise,
            ranges['humidity_range_percent'][0],
            ranges['humidity_range_percent'][1]
        )
        
        # Atmospheric pressure (indoor variations are minimal)
        pressure_variation = np.random.normal(0, 200.0)  # Small variations
        atmospheric_pressure = nominal['atmospheric_pressure_Pa'] + pressure_variation
        
        # Air circulation rate (depends on HVAC system)
        base_circulation = 500.0  # m³/h
        circulation_variation = np.random.uniform(0.8, 1.2)
        air_circulation = base_circulation * circulation_variation
        
        # Power supply quality
        power_stability_factors = {
            'unstable': {'voltage_variation': 15.0, 'frequency_variation': 0.5},
            'variable': {'voltage_variation': 8.0, 'frequency_variation': 0.2},
            'stable': {'voltage_variation': 3.0, 'frequency_variation': 0.1},
            'regulated': {'voltage_variation': 1.0, 'frequency_variation': 0.05}
        }
        
        power_factor = power_stability_factors[facility_params['power_supply_stability']]
        
        voltage_variation = np.random.normal(0, power_factor['voltage_variation'])
        power_voltage = np.clip(
            nominal['power_supply_voltage_V'] + voltage_variation,
            ranges['voltage_range_V'][0],
            ranges['voltage_range_V'][1]
        )
        
        frequency_variation = np.random.normal(0, power_factor['frequency_variation'])
        power_frequency = np.clip(
            nominal['power_supply_frequency_Hz'] + frequency_variation,
            ranges['frequency_range_Hz'][0],
            ranges['frequency_range_Hz'][1]
        )
        
        # Vibration levels
        vibration_levels = {
            'low': 0.5,     # mm/s
            'medium': 2.0,
            'high': 5.0
        }
        
        base_vibration = vibration_levels[facility_params['vibration_environment']]
        vibration_variation = np.random.exponential(0.3)
        vibration_level = base_vibration * (1.0 + vibration_variation)
        
        return IndoorConditions(
            timestamp=timestamp,
            ambient_temperature_C=ambient_temperature,
            relative_humidity_percent=relative_humidity,
            atmospheric_pressure_Pa=atmospheric_pressure,
            air_circulation_rate_m3_per_h=air_circulation,
            equipment_room_temperature_C=equipment_room_temp,
            power_supply_voltage_V=power_voltage,
            power_supply_frequency_Hz=power_frequency,
            vibration_level_mm_per_s=vibration_level
        )
    
    def calculate_equipment_state(self,
                                timestamp: datetime,
                                operating_hours: float,
                                load_pattern: LoadPattern = LoadPattern.COMMERCIAL) -> EquipmentState:
        """
        Calculate current equipment operational state
        
        Args:
            timestamp: Current timestamp
            operating_hours: Total operating hours
            load_pattern: Load demand pattern type
            
        Returns:
            EquipmentState object
        """
        
        # Determine operational mode based on load and time
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        day_of_week = timestamp.weekday()  # 0 = Monday
        
        # Get load demand based on pattern
        load_demand = self._calculate_load_demand(timestamp, load_pattern)
        
        # Determine operational mode
        if load_demand < 0.1:
            operational_mode = OperationalMode.STANDBY
        elif load_demand < 0.3:
            operational_mode = OperationalMode.PART_LOAD
        elif 0.3 <= load_demand <= 0.9:
            operational_mode = OperationalMode.STEADY_STATE
        else:
            operational_mode = OperationalMode.STEADY_STATE
        
        # Check for maintenance periods
        hours_since_minor = operating_hours % self.aging_models['maintenance_cycles']['minor_maintenance_hours']
        if hours_since_minor < 2.0:  # 2-hour maintenance window
            operational_mode = OperationalMode.MAINTENANCE
        
        # Calculate cycles completed (assume 8-hour average cycle)
        cycles_completed = int(operating_hours / 8.0)
        
        # Last maintenance hours
        last_maintenance = hours_since_minor
        
        # Component temperatures (based on load and ambient)
        nominal_temps = self.operational_parameters['component_temperatures']
        load_factor = max(0.3, load_demand)  # Minimum load factor
        
        component_temperatures = {}
        for component, nominal_temp in nominal_temps.items():
            temp_variation = np.random.normal(0, 5.0)  # ±5°C variation
            load_effect = (load_factor - 0.5) * 20.0  # Load effect on temperature
            component_temperatures[component] = nominal_temp + load_effect + temp_variation
        
        # Internal pressures (Stirling engine specific)
        internal_pressures = {
            'working_gas_pressure_Pa': 1500000 + np.random.normal(0, 50000),  # 15 bar ± 0.5 bar
            'cooling_circuit_pressure_Pa': 200000 + np.random.normal(0, 10000),  # 2 bar ± 0.1 bar
            'heating_circuit_pressure_Pa': 300000 + np.random.normal(0, 15000)   # 3 bar ± 0.15 bar
        }
        
        # Flow rates
        flow_rates = {
            'cooling_water_kg_per_s': 0.5 * load_factor + np.random.normal(0, 0.05),
            'heating_fluid_kg_per_s': 0.3 * load_factor + np.random.normal(0, 0.03),
            'working_gas_kg_per_s': 0.01 * load_factor + np.random.normal(0, 0.001)
        }
        
        return EquipmentState(
            operational_mode=operational_mode,
            load_demand_percent=load_demand * 100,
            operating_hours_total=operating_hours,
            cycles_completed=cycles_completed,
            last_maintenance_hours=last_maintenance,
            component_temperatures_C=component_temperatures,
            internal_pressures_Pa=internal_pressures,
            flow_rates_kg_per_s=flow_rates
        )
    
    def _calculate_load_demand(self, timestamp: datetime, load_pattern: LoadPattern) -> float:
        """Calculate load demand based on time and pattern"""
        
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        day_of_year = timestamp.timetuple().tm_yday
        
        pattern_data = self.load_patterns[load_pattern.value]
        
        # Base daily profile
        daily_load = pattern_data['daily_profile'][hour_of_day]
        
        # Weekly variation (weekends different for some patterns)
        if load_pattern in [LoadPattern.COMMERCIAL, LoadPattern.LABORATORY]:
            if day_of_week >= 5:  # Weekend
                daily_load *= 0.3  # Reduced weekend operation
        
        # Weekly random variation
        weekly_variation = np.random.normal(1.0, pattern_data['weekly_variation'])
        daily_load *= weekly_variation
        
        # Annual variation (seasonal demand changes)
        annual_factor = 1.0 + pattern_data['seasonal_variation'] * np.sin(2 * np.pi * day_of_year / 365.25)
        daily_load *= annual_factor
        
        # Random short-term variations
        short_term_variation = np.random.normal(1.0, 0.1)
        daily_load *= short_term_variation
        
        return np.clip(daily_load, 0.0, 1.2)  # Allow 20% overload
    
    def calculate_performance_factors(self,
                                    indoor_conditions: IndoorConditions,
                                    equipment_state: EquipmentState) -> PerformanceFactors:
        """
        Calculate performance factors affecting COP
        
        Args:
            indoor_conditions: Current indoor conditions
            equipment_state: Current equipment state
            
        Returns:
            PerformanceFactors object
        """
        
        # Aging factor (equipment degradation over time)
        operating_hours = equipment_state.operating_hours_total
        wear_rates = self.aging_models['wear_rates']
        
        aging_factor = 1.0
        aging_factor -= wear_rates['seals_degradation_per_1000h'] * (operating_hours / 1000.0)
        aging_factor -= wear_rates['bearing_wear_per_1000h'] * (operating_hours / 1000.0)
        aging_factor -= wear_rates['insulation_degradation_per_1000h'] * (operating_hours / 1000.0)
        aging_factor = max(0.7, aging_factor)  # Minimum 70% performance
        
        # Fouling factor (heat exchanger degradation)
        fouling_factor = 1.0 - wear_rates['heat_exchanger_fouling_per_1000h'] * (operating_hours / 1000.0)
        fouling_factor = max(0.8, fouling_factor)  # Minimum 80% heat transfer
        
        # Wear factor (mechanical component wear)
        wear_factor = 1.0 - wear_rates['regenerator_degradation_per_1000h'] * (operating_hours / 1000.0)
        wear_factor = max(0.85, wear_factor)  # Minimum 85% regenerator effectiveness
        
        # Maintenance factor (performance recovery after maintenance)
        maintenance_cycles = self.aging_models['maintenance_cycles']
        hours_since_maintenance = equipment_state.last_maintenance_hours
        
        if hours_since_maintenance < 100:  # Recent maintenance
            maintenance_factor = 1.02  # 2% performance boost
        elif hours_since_maintenance < 500:
            maintenance_factor = 1.0
        else:
            maintenance_factor = 0.98  # Slight degradation before maintenance
        
        # Load efficiency factor (part-load performance)
        load_ratio = equipment_state.load_demand_percent / 100.0
        
        if load_ratio < 0.2:
            load_efficiency_factor = 0.6  # Poor efficiency at very low loads
        elif load_ratio < 0.5:
            load_efficiency_factor = 0.8 + 0.4 * (load_ratio - 0.2) / 0.3
        else:
            load_efficiency_factor = 1.0  # Optimal efficiency at higher loads
        
        # Temperature factor (ambient temperature effects)
        nominal_temp = self.operational_parameters['nominal_conditions']['ambient_temperature_C']
        temp_coeff = self.operational_parameters['performance_curves']['temperature_coefficient']
        temp_diff = indoor_conditions.ambient_temperature_C - nominal_temp
        temperature_factor = 1.0 + temp_coeff * temp_diff
        temperature_factor = max(0.9, min(1.1, temperature_factor))
        
        # Power quality factor
        nominal_voltage = self.operational_parameters['nominal_conditions']['power_supply_voltage_V']
        voltage_coeff = self.operational_parameters['performance_curves']['voltage_coefficient']
        voltage_diff = indoor_conditions.power_supply_voltage_V - nominal_voltage
        power_quality_factor = 1.0 + voltage_coeff * voltage_diff
        power_quality_factor = max(0.95, min(1.05, power_quality_factor))
        
        return PerformanceFactors(
            aging_factor=aging_factor,
            fouling_factor=fouling_factor,
            wear_factor=wear_factor,
            maintenance_factor=maintenance_factor,
            load_efficiency_factor=load_efficiency_factor,
            temperature_factor=temperature_factor,
            power_quality_factor=power_quality_factor
        )
    
    def calculate_indoor_cop(self,
                           indoor_conditions: IndoorConditions,
                           equipment_state: EquipmentState,
                           performance_factors: PerformanceFactors,
                           base_cop: float = 4.5) -> float:
        """
        Calculate COP considering indoor operational factors
        
        Args:
            indoor_conditions: Current indoor conditions
            equipment_state: Current equipment state
            performance_factors: Performance factors
            base_cop: Base COP at nominal conditions
            
        Returns:
            Adjusted COP value
        """
        
        # Apply all performance factors
        adjusted_cop = base_cop
        adjusted_cop *= performance_factors.aging_factor
        adjusted_cop *= performance_factors.fouling_factor
        adjusted_cop *= performance_factors.wear_factor
        adjusted_cop *= performance_factors.maintenance_factor
        adjusted_cop *= performance_factors.load_efficiency_factor
        adjusted_cop *= performance_factors.temperature_factor
        adjusted_cop *= performance_factors.power_quality_factor
        
        # Operational mode effects
        mode_factors = {
            OperationalMode.STARTUP: 0.7,      # Reduced efficiency during startup
            OperationalMode.STEADY_STATE: 1.0,  # Nominal efficiency
            OperationalMode.PART_LOAD: 0.9,     # Slightly reduced at part load
            OperationalMode.SHUTDOWN: 0.5,      # Poor efficiency during shutdown
            OperationalMode.MAINTENANCE: 0.0,   # No operation during maintenance
            OperationalMode.STANDBY: 0.1        # Minimal operation in standby
        }
        
        mode_factor = mode_factors[equipment_state.operational_mode]
        adjusted_cop *= mode_factor
        
        return max(1.0, adjusted_cop)  # Minimum COP of 1.0
    
    def simulate_indoor_performance(self,
                                  start_date: datetime,
                                  duration_days: int = 30,
                                  time_step_hours: int = 1,
                                  facility_params: Optional[Dict] = None,
                                  load_pattern: LoadPattern = LoadPattern.COMMERCIAL,
                                  base_cop: float = 4.5) -> List[IndoorPerformanceData]:
        """
        Simulate indoor equipment performance over time
        
        Args:
            start_date: Starting date for simulation
            duration_days: Simulation duration in days
            time_step_hours: Time step in hours
            facility_params: Facility-specific parameters
            load_pattern: Load demand pattern
            base_cop: Base COP at nominal conditions
            
        Returns:
            List of IndoorPerformanceData objects
        """
        
        performance_data = []
        current_time = start_date
        time_delta = timedelta(hours=time_step_hours)
        
        # Initialize cumulative variables
        operating_hours = 0.0
        
        for _ in range(int(duration_days * 24 / time_step_hours)):
            
            # Generate indoor conditions
            indoor_conditions = self.generate_indoor_conditions(current_time, facility_params)
            
            # Calculate equipment state
            equipment_state = self.calculate_equipment_state(current_time, operating_hours, load_pattern)
            
            # Calculate performance factors
            performance_factors = self.calculate_performance_factors(indoor_conditions, equipment_state)
            
            # Calculate COP
            cop = self.calculate_indoor_cop(indoor_conditions, equipment_state, performance_factors, base_cop)
            
            # Calculate thermal efficiency (for heat engines)
            thermal_efficiency = cop * 25.0  # Approximate conversion for Stirling engines
            thermal_efficiency = min(thermal_efficiency, 45.0)  # Maximum theoretical efficiency
            
            # Calculate energy consumption and heat output
            load_demand_kW = equipment_state.load_demand_percent / 100.0 * 20.0  # Assume 20 kW max load
            
            if equipment_state.operational_mode != OperationalMode.MAINTENANCE:
                if cop > 0:
                    energy_consumption = load_demand_kW * time_step_hours / cop
                    heat_output = load_demand_kW
                else:
                    energy_consumption = 0.5 * time_step_hours  # Standby consumption
                    heat_output = 0.0
                
                operating_hours += time_step_hours
            else:
                energy_consumption = 0.1 * time_step_hours  # Maintenance power
                heat_output = 0.0
            
            # Calculate maintenance indicator (0-1, higher means maintenance needed soon)
            maintenance_indicator = min(1.0, equipment_state.last_maintenance_hours / 
                                      self.aging_models['maintenance_cycles']['minor_maintenance_hours'])
            
            # Create performance data point
            performance_point = IndoorPerformanceData(
                timestamp=current_time,
                indoor_conditions=indoor_conditions,
                equipment_state=equipment_state,
                performance_factors=performance_factors,
                heat_engine_cop=cop,
                thermal_efficiency_percent=thermal_efficiency,
                energy_consumption_kWh=energy_consumption,
                heat_output_kW=heat_output,
                maintenance_indicator=maintenance_indicator
            )
            
            performance_data.append(performance_point)
            current_time += time_delta
        
        logger.info(f"Simulated {len(performance_data)} indoor performance data points over {duration_days} days")
        return performance_data
    
    def validate_indoor_model(self) -> Dict[str, bool]:
        """Validate indoor performance model"""
        
        validation_results = {
            'indoor_conditions_generation': False,
            'equipment_state_calculation': False,
            'performance_factors_calculation': False,
            'cop_calculation': False,
            'simulation_execution': False
        }
        
        try:
            # Test indoor conditions generation
            test_date = datetime(2024, 6, 15, 12, 0)
            indoor_conditions = self.generate_indoor_conditions(test_date)
            validation_results['indoor_conditions_generation'] = (
                15 <= indoor_conditions.ambient_temperature_C <= 30 and
                30 <= indoor_conditions.relative_humidity_percent <= 70 and
                indoor_conditions.power_supply_voltage_V > 200
            )
            
            # Test equipment state calculation
            equipment_state = self.calculate_equipment_state(test_date, 1000.0)
            validation_results['equipment_state_calculation'] = (
                equipment_state.operating_hours_total == 1000.0 and
                0 <= equipment_state.load_demand_percent <= 120 and
                len(equipment_state.component_temperatures_C) > 0
            )
            
            # Test performance factors calculation
            performance_factors = self.calculate_performance_factors(indoor_conditions, equipment_state)
            validation_results['performance_factors_calculation'] = (
                0.5 <= performance_factors.aging_factor <= 1.1 and
                0.5 <= performance_factors.load_efficiency_factor <= 1.1 and
                0.9 <= performance_factors.temperature_factor <= 1.1
            )
            
            # Test COP calculation
            cop = self.calculate_indoor_cop(indoor_conditions, equipment_state, performance_factors)
            validation_results['cop_calculation'] = 1.0 <= cop <= 8.0
            
            # Test simulation execution
            performance_data = self.simulate_indoor_performance(
                datetime(2024, 1, 1), duration_days=2, time_step_hours=6
            )
            validation_results['simulation_execution'] = len(performance_data) == 8
            
        except Exception as e:
            logger.error(f"Indoor model validation failed: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = IndoorPerformanceModel()
    
    print("🏭 Testing Indoor Performance Model")
    
    # Test indoor conditions generation
    test_date = datetime(2024, 6, 15, 12, 0)
    indoor_conditions = model.generate_indoor_conditions(test_date)
    
    print(f"✅ Indoor conditions:")
    print(f"  - Ambient temperature: {indoor_conditions.ambient_temperature_C:.1f}°C")
    print(f"  - Equipment room temperature: {indoor_conditions.equipment_room_temperature_C:.1f}°C")
    print(f"  - Power supply voltage: {indoor_conditions.power_supply_voltage_V:.1f}V")
    print(f"  - Vibration level: {indoor_conditions.vibration_level_mm_per_s:.2f} mm/s")
    
    # Test equipment state
    equipment_state = model.calculate_equipment_state(test_date, 1000.0)
    
    print(f"✅ Equipment state:")
    print(f"  - Operational mode: {equipment_state.operational_mode.value}")
    print(f"  - Load demand: {equipment_state.load_demand_percent:.1f}%")
    print(f"  - Operating hours: {equipment_state.operating_hours_total:.0f}h")
    print(f"  - Hot side temperature: {equipment_state.component_temperatures_C.get('hot_side_nominal_C', 0):.1f}°C")
    
    # Test performance factors
    performance_factors = model.calculate_performance_factors(indoor_conditions, equipment_state)
    
    print(f"✅ Performance factors:")
    print(f"  - Aging factor: {performance_factors.aging_factor:.3f}")
    print(f"  - Load efficiency factor: {performance_factors.load_efficiency_factor:.3f}")
    print(f"  - Temperature factor: {performance_factors.temperature_factor:.3f}")
    print(f"  - Maintenance factor: {performance_factors.maintenance_factor:.3f}")
    
    # Test COP calculation
    cop = model.calculate_indoor_cop(indoor_conditions, equipment_state, performance_factors)
    
    print(f"✅ Indoor COP: {cop:.2f}")
    
    # Test short simulation
    performance_data = model.simulate_indoor_performance(
        datetime(2024, 1, 1), duration_days=7, time_step_hours=6
    )
    
    print(f"✅ Generated {len(performance_data)} performance data points")
    
    # Calculate average performance
    avg_cop = sum(p.heat_engine_cop for p in performance_data) / len(performance_data)
    avg_efficiency = sum(p.thermal_efficiency_percent for p in performance_data) / len(performance_data)
    
    print(f"✅ Average COP: {avg_cop:.2f}")
    print(f"✅ Average thermal efficiency: {avg_efficiency:.1f}%")
    
    # Validate model
    validation = model.validate_indoor_model()
    print(f"✅ Model validation: {sum(validation.values())}/{len(validation)} tests passed")
    
    print("🎉 Indoor performance model working correctly!")
