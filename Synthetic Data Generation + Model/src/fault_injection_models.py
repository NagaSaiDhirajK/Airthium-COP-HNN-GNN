#!/usr/bin/env python3
"""
Fault Injection Models for AIRTHIUM Heat Pumps
==============================================

This module implements realistic fault injection models for heat pump systems,
including refrigerant leaks, sensor drift, component failures, and system faults.

Key Features:
- Refrigerant leak modeling with progressive charge loss
- Sensor drift and calibration errors
- Component failure modes (compressor, valves, fans)
- Electrical faults and control system failures
- Fouling acceleration and blockage faults
- Fault detection and diagnostic indicators

Sources:
- ASHRAE Handbook - HVAC Systems and Equipment (2020)
- NIST Fault Detection and Diagnostics Database
- DOE Building Technologies Office Studies
- IEEE Reliability Engineering Standards
- Carrier Fault Diagnostic Manual
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from enum import Enum

class FaultType(Enum):
    """Types of faults that can be injected."""
    REFRIGERANT_LEAK = "refrigerant_leak"
    SENSOR_DRIFT = "sensor_drift"
    COMPRESSOR_FAILURE = "compressor_failure"
    FAN_FAILURE = "fan_failure"
    VALVE_STUCK = "valve_stuck"
    FOULING_ACCELERATED = "fouling_accelerated"
    ELECTRICAL_FAULT = "electrical_fault"
    CONTROL_FAULT = "control_fault"
    BLOCKAGE = "blockage"
    INSULATION_DEGRADATION = "insulation_degradation"

class FaultSeverity(Enum):
    """Severity levels for faults."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class FaultState:
    """Current fault state of a system component."""
    fault_id: str
    component_id: str
    fault_type: FaultType
    severity: FaultSeverity
    onset_time: str
    progression_rate: float
    current_magnitude: float
    
    # Fault-specific parameters
    fault_parameters: Dict[str, float]
    
    # Performance impacts
    capacity_impact_factor: float      # Multiplier for capacity (0-1)
    cop_impact_factor: float           # Multiplier for COP (0-1)
    power_impact_factor: float         # Multiplier for power (1+)
    
    # Detection indicators
    detectable: bool                   # Can this fault be detected?
    detection_confidence: float        # Confidence in fault detection (0-1)
    false_positive_rate: float         # Probability of false positive
    
    # Maintenance requirements
    repair_urgency: str               # 'low', 'medium', 'high', 'critical'
    estimated_repair_cost: float      # USD
    estimated_repair_time_hours: float

@dataclass
class SensorFault:
    """Specific sensor fault parameters."""
    sensor_id: str
    sensor_type: str
    fault_type: str
    drift_rate: float                 # Units per hour
    noise_increase_factor: float      # Multiplier for sensor noise
    bias_offset: float                # Constant offset
    scale_factor_error: float         # Multiplicative error
    intermittent_failure_rate: float  # Failures per hour

class FaultInjectionModel:
    """Model fault injection and progression in heat pump systems."""
    
    def __init__(self):
        """Initialize fault injection model with realistic parameters."""
        
        # Fault progression models from reliability studies
        self.fault_progression_models = {
            FaultType.REFRIGERANT_LEAK: {
                'initial_rate_percent_per_year': [0.5, 2.0, 5.0, 15.0],  # By severity
                'acceleration_factor': 1.2,
                'detection_threshold_percent': 10.0,
                'critical_threshold_percent': 50.0,
                'repair_cost_base': 300,
                'repair_time_hours': 4
            },
            FaultType.SENSOR_DRIFT: {
                'drift_rates': {
                    'temperature': [0.1, 0.5, 1.0, 2.0],    # K per year by severity
                    'pressure': [0.01, 0.05, 0.1, 0.2],     # bar per year
                    'flow': [0.02, 0.1, 0.2, 0.5],          # % per year
                    'power': [0.01, 0.05, 0.1, 0.2]         # % per year
                },
                'detection_threshold_factor': 2.0,
                'repair_cost_base': 150,
                'repair_time_hours': 2
            },
            FaultType.COMPRESSOR_FAILURE: {
                'failure_modes': {
                    'bearing_wear': {'rate': [0.1, 0.5, 1.0, 2.0], 'impact': [0.95, 0.85, 0.7, 0.4]},
                    'valve_leakage': {'rate': [0.2, 0.8, 1.5, 3.0], 'impact': [0.9, 0.75, 0.5, 0.2]},
                    'motor_degradation': {'rate': [0.05, 0.2, 0.5, 1.0], 'impact': [0.98, 0.9, 0.8, 0.6]}
                },
                'detection_threshold': 0.9,
                'repair_cost_base': 2000,
                'repair_time_hours': 8
            },
            FaultType.FAN_FAILURE: {
                'failure_modes': {
                    'motor_failure': {'rate': [0.1, 0.3, 0.8, 2.0], 'impact': [0.8, 0.6, 0.3, 0.0]},
                    'blade_damage': {'rate': [0.05, 0.2, 0.5, 1.0], 'impact': [0.9, 0.7, 0.5, 0.2]},
                    'bearing_failure': {'rate': [0.2, 0.5, 1.0, 2.0], 'impact': [0.85, 0.65, 0.4, 0.1]}
                },
                'detection_threshold': 0.7,
                'repair_cost_base': 400,
                'repair_time_hours': 3
            },
            FaultType.VALVE_STUCK: {
                'stuck_positions': [0.0, 0.25, 0.5, 0.75, 1.0],  # Fraction open
                'progression_rate': [0.1, 0.3, 0.8, 2.0],
                'detection_threshold': 0.1,  # 10% deviation from setpoint
                'repair_cost_base': 250,
                'repair_time_hours': 2
            },
            FaultType.FOULING_ACCELERATED: {
                'acceleration_factors': [2.0, 5.0, 10.0, 20.0],  # By severity
                'base_fouling_rate': 0.0001,  # m²K/W per year
                'detection_threshold': 0.0005,
                'repair_cost_base': 500,
                'repair_time_hours': 6
            }
        }
        
        # Sensor fault models
        self.sensor_fault_models = {
            'temperature_sensor': {
                'drift_rate_K_per_year': [0.1, 0.5, 1.0, 2.0],
                'noise_increase_factor': [1.2, 1.5, 2.0, 3.0],
                'bias_offset_K': [0.5, 1.0, 2.0, 5.0],
                'scale_factor_error': [0.01, 0.02, 0.05, 0.1],
                'intermittent_failure_rate': [0.001, 0.01, 0.1, 1.0]  # per hour
            },
            'pressure_sensor': {
                'drift_rate_bar_per_year': [0.01, 0.05, 0.1, 0.2],
                'noise_increase_factor': [1.2, 1.5, 2.0, 3.0],
                'bias_offset_bar': [0.1, 0.2, 0.5, 1.0],
                'scale_factor_error': [0.01, 0.02, 0.05, 0.1],
                'intermittent_failure_rate': [0.001, 0.01, 0.1, 1.0]
            },
            'flow_sensor': {
                'drift_rate_percent_per_year': [1.0, 2.0, 5.0, 10.0],
                'noise_increase_factor': [1.2, 1.5, 2.0, 3.0],
                'bias_offset_percent': [2.0, 5.0, 10.0, 20.0],
                'scale_factor_error': [0.02, 0.05, 0.1, 0.2],
                'intermittent_failure_rate': [0.001, 0.01, 0.1, 1.0]
            },
            'power_sensor': {
                'drift_rate_percent_per_year': [0.5, 1.0, 2.0, 5.0],
                'noise_increase_factor': [1.1, 1.3, 1.8, 2.5],
                'bias_offset_percent': [1.0, 2.0, 5.0, 10.0],
                'scale_factor_error': [0.01, 0.02, 0.05, 0.1],
                'intermittent_failure_rate': [0.0005, 0.005, 0.05, 0.5]
            }
        }
        
        # Fault interaction models
        self.fault_interactions = {
            (FaultType.REFRIGERANT_LEAK, FaultType.COMPRESSOR_FAILURE): 1.5,  # Leak accelerates compressor wear
            (FaultType.FOULING_ACCELERATED, FaultType.FAN_FAILURE): 2.0,     # Fouling worsens with poor airflow
            (FaultType.SENSOR_DRIFT, FaultType.CONTROL_FAULT): 1.3,          # Sensor drift causes control issues
            (FaultType.VALVE_STUCK, FaultType.REFRIGERANT_LEAK): 1.2         # Stuck valve can cause pressure issues
        }
        
        # Detection algorithms and thresholds
        self.detection_algorithms = {
            'statistical_process_control': {
                'sensitivity': 0.8,
                'false_positive_rate': 0.05,
                'detection_delay_hours': 2.0
            },
            'model_based_detection': {
                'sensitivity': 0.9,
                'false_positive_rate': 0.02,
                'detection_delay_hours': 1.0
            },
            'machine_learning_detection': {
                'sensitivity': 0.95,
                'false_positive_rate': 0.01,
                'detection_delay_hours': 0.5
            }
        }
    
    def inject_fault(self, system_id: str, component_id: str, 
                    fault_type: FaultType, severity: FaultSeverity,
                    onset_time: str = None) -> FaultState:
        """Inject a fault into the system."""
        
        if onset_time is None:
            onset_time = datetime.now().isoformat()
        
        fault_id = f"{system_id}_{component_id}_{fault_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get fault parameters
        fault_params = self.fault_progression_models.get(fault_type, {})
        severity_index = list(FaultSeverity).index(severity)
        
        # Initialize fault-specific parameters
        fault_parameters = {}
        initial_magnitude = 0.0
        progression_rate = 0.0
        
        if fault_type == FaultType.REFRIGERANT_LEAK:
            progression_rate = fault_params['initial_rate_percent_per_year'][severity_index] / 8760  # per hour
            fault_parameters = {
                'leak_rate_kg_per_hour': progression_rate * 0.001,  # Convert to kg/hr
                'detection_threshold': fault_params['detection_threshold_percent'],
                'critical_threshold': fault_params['critical_threshold_percent']
            }
            
        elif fault_type == FaultType.SENSOR_DRIFT:
            sensor_type = self._determine_sensor_type(component_id)
            sensor_params = self.sensor_fault_models.get(sensor_type, 
                                                       self.sensor_fault_models['temperature_sensor'])
            
            if 'temperature' in sensor_type:
                progression_rate = sensor_params['drift_rate_K_per_year'][severity_index] / 8760
                fault_parameters = {
                    'drift_rate': progression_rate,
                    'noise_factor': sensor_params['noise_increase_factor'][severity_index],
                    'bias_offset': sensor_params['bias_offset_K'][severity_index],
                    'scale_error': sensor_params['scale_factor_error'][severity_index]
                }
            elif 'pressure' in sensor_type:
                progression_rate = sensor_params['drift_rate_bar_per_year'][severity_index] / 8760
                fault_parameters = {
                    'drift_rate': progression_rate,
                    'noise_factor': sensor_params['noise_increase_factor'][severity_index],
                    'bias_offset': sensor_params['bias_offset_bar'][severity_index],
                    'scale_error': sensor_params['scale_factor_error'][severity_index]
                }
            
        elif fault_type == FaultType.COMPRESSOR_FAILURE:
            failure_mode = np.random.choice(['bearing_wear', 'valve_leakage', 'motor_degradation'])
            mode_params = fault_params['failure_modes'][failure_mode]
            progression_rate = mode_params['rate'][severity_index] / 8760  # per hour
            initial_magnitude = 1.0 - mode_params['impact'][severity_index]
            
            fault_parameters = {
                'failure_mode': failure_mode,
                'progression_rate': progression_rate,
                'capacity_impact': mode_params['impact'][severity_index],
                'detection_threshold': fault_params['detection_threshold']
            }
            
        elif fault_type == FaultType.FAN_FAILURE:
            failure_mode = np.random.choice(['motor_failure', 'blade_damage', 'bearing_failure'])
            mode_params = fault_params['failure_modes'][failure_mode]
            progression_rate = mode_params['rate'][severity_index] / 8760
            initial_magnitude = 1.0 - mode_params['impact'][severity_index]
            
            fault_parameters = {
                'failure_mode': failure_mode,
                'progression_rate': progression_rate,
                'airflow_impact': mode_params['impact'][severity_index],
                'detection_threshold': fault_params['detection_threshold']
            }
            
        elif fault_type == FaultType.FOULING_ACCELERATED:
            acceleration_factor = fault_params['acceleration_factors'][severity_index]
            progression_rate = fault_params['base_fouling_rate'] * acceleration_factor / 8760
            
            fault_parameters = {
                'acceleration_factor': acceleration_factor,
                'base_fouling_rate': fault_params['base_fouling_rate'],
                'detection_threshold': fault_params['detection_threshold']
            }
        
        # Calculate performance impacts
        capacity_impact, cop_impact, power_impact = self._calculate_performance_impacts(
            fault_type, severity, initial_magnitude, fault_parameters
        )
        
        # Determine detectability
        detectable, detection_confidence, false_positive_rate = self._calculate_detectability(
            fault_type, severity, initial_magnitude
        )
        
        # Estimate repair requirements
        repair_urgency = self._determine_repair_urgency(fault_type, severity)
        repair_cost = fault_params.get('repair_cost_base', 200) * (1 + severity_index * 0.5)
        repair_time = fault_params.get('repair_time_hours', 2) * (1 + severity_index * 0.3)
        
        return FaultState(
            fault_id=fault_id,
            component_id=component_id,
            fault_type=fault_type,
            severity=severity,
            onset_time=onset_time,
            progression_rate=progression_rate,
            current_magnitude=initial_magnitude,
            fault_parameters=fault_parameters,
            capacity_impact_factor=capacity_impact,
            cop_impact_factor=cop_impact,
            power_impact_factor=power_impact,
            detectable=detectable,
            detection_confidence=detection_confidence,
            false_positive_rate=false_positive_rate,
            repair_urgency=repair_urgency,
            estimated_repair_cost=repair_cost,
            estimated_repair_time_hours=repair_time
        )
    
    def progress_fault(self, fault_state: FaultState, time_hours: float,
                      operating_conditions: Dict[str, float] = None) -> FaultState:
        """Progress fault over time based on operating conditions."""
        
        if operating_conditions is None:
            operating_conditions = {}
        
        # Calculate progression acceleration factors
        acceleration_factor = 1.0
        
        # Temperature acceleration
        operating_temp = operating_conditions.get('average_temperature', 298.15)
        if operating_temp > 323.15:  # Above 50°C
            acceleration_factor *= 1.5
        elif operating_temp < 263.15:  # Below -10°C
            acceleration_factor *= 1.3
        
        # Load factor acceleration
        load_factor = operating_conditions.get('load_factor', 1.0)
        if load_factor > 1.1:  # Overload
            acceleration_factor *= 1.4
        
        # Cycling acceleration
        cycles_per_hour = operating_conditions.get('cycles_per_hour', 0)
        if cycles_per_hour > 6:  # Excessive cycling
            acceleration_factor *= 1.2
        
        # Progress the fault
        progression = fault_state.progression_rate * time_hours * acceleration_factor
        fault_state.current_magnitude = min(1.0, fault_state.current_magnitude + progression)
        
        # Update performance impacts based on current magnitude
        capacity_impact, cop_impact, power_impact = self._calculate_performance_impacts(
            fault_state.fault_type, fault_state.severity, 
            fault_state.current_magnitude, fault_state.fault_parameters
        )
        
        fault_state.capacity_impact_factor = capacity_impact
        fault_state.cop_impact_factor = cop_impact
        fault_state.power_impact_factor = power_impact
        
        # Update detectability
        detectable, detection_confidence, false_positive_rate = self._calculate_detectability(
            fault_state.fault_type, fault_state.severity, fault_state.current_magnitude
        )
        
        fault_state.detectable = detectable
        fault_state.detection_confidence = detection_confidence
        fault_state.false_positive_rate = false_positive_rate
        
        # Update repair urgency
        fault_state.repair_urgency = self._determine_repair_urgency(
            fault_state.fault_type, fault_state.severity, fault_state.current_magnitude
        )
        
        return fault_state
    
    def inject_sensor_fault(self, sensor_id: str, sensor_type: str,
                           fault_severity: FaultSeverity) -> SensorFault:
        """Inject specific sensor fault."""
        
        severity_index = list(FaultSeverity).index(fault_severity)
        sensor_params = self.sensor_fault_models.get(sensor_type,
                                                   self.sensor_fault_models['temperature_sensor'])
        
        # Determine fault type
        fault_types = ['drift', 'bias', 'noise', 'scale_error', 'intermittent']
        fault_type = np.random.choice(fault_types)
        
        # Set fault parameters based on type
        if fault_type == 'drift':
            if 'temperature' in sensor_type:
                drift_rate = sensor_params['drift_rate_K_per_year'][severity_index] / 8760
            elif 'pressure' in sensor_type:
                drift_rate = sensor_params['drift_rate_bar_per_year'][severity_index] / 8760
            else:
                drift_rate = sensor_params.get('drift_rate_percent_per_year', [1.0, 2.0, 5.0, 10.0])[severity_index] / 8760
        else:
            drift_rate = 0.0
        
        return SensorFault(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            fault_type=fault_type,
            drift_rate=drift_rate,
            noise_increase_factor=sensor_params['noise_increase_factor'][severity_index],
            bias_offset=sensor_params.get('bias_offset_K', sensor_params.get('bias_offset_bar', [1.0, 2.0, 5.0, 10.0]))[severity_index],
            scale_factor_error=sensor_params['scale_factor_error'][severity_index],
            intermittent_failure_rate=sensor_params['intermittent_failure_rate'][severity_index]
        )
    
    def apply_sensor_fault(self, sensor_value: float, sensor_fault: SensorFault,
                          time_hours: float) -> float:
        """Apply sensor fault to a measurement."""
        
        faulty_value = sensor_value
        
        # Apply drift
        if sensor_fault.fault_type == 'drift':
            drift = sensor_fault.drift_rate * time_hours
            faulty_value += drift
        
        # Apply bias offset
        if sensor_fault.fault_type == 'bias':
            faulty_value += sensor_fault.bias_offset
        
        # Apply scale factor error
        if sensor_fault.fault_type == 'scale_error':
            faulty_value *= (1 + sensor_fault.scale_factor_error)
        
        # Apply noise increase
        if sensor_fault.fault_type == 'noise':
            base_noise = abs(sensor_value) * 0.01  # 1% base noise
            increased_noise = base_noise * sensor_fault.noise_increase_factor
            faulty_value += np.random.normal(0, increased_noise)
        
        # Apply intermittent failure
        if sensor_fault.fault_type == 'intermittent':
            failure_probability = sensor_fault.intermittent_failure_rate * time_hours
            if np.random.random() < failure_probability:
                faulty_value = np.nan  # Sensor reading failure
        
        return faulty_value
    
    def detect_fault(self, system_data: Dict[str, List[float]], 
                    detection_algorithm: str = 'statistical_process_control') -> Dict[str, Any]:
        """Detect faults using specified algorithm."""
        
        algorithm_params = self.detection_algorithms.get(detection_algorithm,
                                                       self.detection_algorithms['statistical_process_control'])
        
        detection_results = {
            'faults_detected': [],
            'detection_confidence': [],
            'false_alarms': [],
            'detection_delay_hours': algorithm_params['detection_delay_hours']
        }
        
        # Simple statistical process control detection
        if detection_algorithm == 'statistical_process_control':
            for parameter, values in system_data.items():
                if len(values) < 10:  # Need minimum data points
                    continue
                
                # Calculate control limits (3-sigma)
                mean_val = np.mean(values)
                std_val = np.std(values)
                upper_limit = mean_val + 3 * std_val
                lower_limit = mean_val - 3 * std_val
                
                # Check for out-of-control points
                outliers = [i for i, val in enumerate(values) 
                           if val > upper_limit or val < lower_limit]
                
                if len(outliers) > len(values) * 0.05:  # More than 5% outliers
                    detection_results['faults_detected'].append({
                        'parameter': parameter,
                        'fault_type': 'statistical_anomaly',
                        'confidence': algorithm_params['sensitivity'],
                        'outlier_count': len(outliers)
                    })
        
        # Add false alarms based on algorithm characteristics
        if np.random.random() < algorithm_params['false_positive_rate']:
            detection_results['false_alarms'].append({
                'parameter': np.random.choice(list(system_data.keys())),
                'false_alarm_type': 'statistical_noise',
                'timestamp': datetime.now().isoformat()
            })
        
        return detection_results
    
    def _determine_sensor_type(self, component_id: str) -> str:
        """Determine sensor type from component ID."""
        
        if 'temp' in component_id.lower() or 'temperature' in component_id.lower():
            return 'temperature_sensor'
        elif 'press' in component_id.lower() or 'pressure' in component_id.lower():
            return 'pressure_sensor'
        elif 'flow' in component_id.lower():
            return 'flow_sensor'
        elif 'power' in component_id.lower() or 'current' in component_id.lower():
            return 'power_sensor'
        else:
            return 'temperature_sensor'  # Default
    
    def _calculate_performance_impacts(self, fault_type: FaultType, severity: FaultSeverity,
                                     magnitude: float, fault_parameters: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate performance impacts of fault."""
        
        severity_factor = (list(FaultSeverity).index(severity) + 1) * 0.25
        
        if fault_type == FaultType.REFRIGERANT_LEAK:
            # Capacity and COP decrease with charge loss
            capacity_impact = 1.0 - magnitude * 0.8 * severity_factor
            cop_impact = 1.0 - magnitude * 0.6 * severity_factor
            power_impact = 1.0 + magnitude * 0.3 * severity_factor
            
        elif fault_type == FaultType.COMPRESSOR_FAILURE:
            capacity_impact = fault_parameters.get('capacity_impact', 1.0 - magnitude * severity_factor)
            cop_impact = 1.0 - magnitude * 0.7 * severity_factor
            power_impact = 1.0 + magnitude * 0.5 * severity_factor
            
        elif fault_type == FaultType.FAN_FAILURE:
            airflow_impact = fault_parameters.get('airflow_impact', 1.0 - magnitude * severity_factor)
            capacity_impact = airflow_impact
            cop_impact = airflow_impact * 0.8
            power_impact = 1.0 + (1 - airflow_impact) * 0.3
            
        elif fault_type == FaultType.FOULING_ACCELERATED:
            # Fouling reduces heat transfer effectiveness
            capacity_impact = 1.0 - magnitude * 0.4 * severity_factor
            cop_impact = 1.0 - magnitude * 0.3 * severity_factor
            power_impact = 1.0 + magnitude * 0.2 * severity_factor
            
        else:
            # Default impacts for other fault types
            capacity_impact = 1.0 - magnitude * 0.3 * severity_factor
            cop_impact = 1.0 - magnitude * 0.2 * severity_factor
            power_impact = 1.0 + magnitude * 0.1 * severity_factor
        
        return (np.clip(capacity_impact, 0.1, 1.0),
                np.clip(cop_impact, 0.1, 1.0),
                np.clip(power_impact, 1.0, 2.0))
    
    def _calculate_detectability(self, fault_type: FaultType, severity: FaultSeverity,
                               magnitude: float) -> Tuple[bool, float, float]:
        """Calculate fault detectability parameters."""
        
        severity_index = list(FaultSeverity).index(severity)
        
        # Base detectability increases with severity and magnitude
        base_detectability = 0.3 + severity_index * 0.2 + magnitude * 0.4
        
        # Fault-specific detectability
        if fault_type == FaultType.REFRIGERANT_LEAK:
            detectability = base_detectability * 0.8  # Harder to detect initially
            false_positive_rate = 0.02
        elif fault_type == FaultType.SENSOR_DRIFT:
            detectability = base_detectability * 1.2  # Easier to detect with data analysis
            false_positive_rate = 0.05
        elif fault_type == FaultType.COMPRESSOR_FAILURE:
            detectability = base_detectability * 1.1  # Moderate detectability
            false_positive_rate = 0.03
        elif fault_type == FaultType.FAN_FAILURE:
            detectability = base_detectability * 1.3  # Usually obvious
            false_positive_rate = 0.01
        else:
            detectability = base_detectability
            false_positive_rate = 0.03
        
        detectability = np.clip(detectability, 0.0, 0.95)
        detectable = detectability > 0.5
        
        return detectable, detectability, false_positive_rate
    
    def _determine_repair_urgency(self, fault_type: FaultType, severity: FaultSeverity,
                                magnitude: float = 0.0) -> str:
        """Determine repair urgency based on fault characteristics."""
        
        if severity == FaultSeverity.CRITICAL or magnitude > 0.8:
            return 'critical'
        elif severity == FaultSeverity.SEVERE or magnitude > 0.6:
            return 'high'
        elif severity == FaultSeverity.MODERATE or magnitude > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def generate_fault_scenario(self, system_id: str, duration_hours: float,
                              fault_probability_per_hour: float = 0.001) -> List[FaultState]:
        """Generate realistic fault scenario over time."""
        
        faults = []
        current_time = 0
        
        # Common components that can fail
        components = [
            'compressor_001', 'evaporator_fan_001', 'condenser_fan_001',
            'expansion_valve_001', 'temp_sensor_evap', 'temp_sensor_cond',
            'pressure_sensor_high', 'pressure_sensor_low', 'flow_sensor_001'
        ]
        
        while current_time < duration_hours:
            # Check if fault should occur
            if np.random.random() < fault_probability_per_hour:
                # Select random component and fault type
                component = np.random.choice(components)
                fault_type = np.random.choice(list(FaultType))
                severity = np.random.choice(list(FaultSeverity))
                
                # Create fault
                fault = self.inject_fault(system_id, component, fault_type, severity)
                faults.append(fault)
            
            current_time += 1.0  # Advance by 1 hour
        
        return faults
    
    def save_fault_data(self, fault_states: List[FaultState], output_path: str) -> None:
        """Save fault injection data to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'fault_states': [asdict(state) for state in fault_states],
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'fault_progression_models': {k.value: v for k, v in self.fault_progression_models.items()},
                'sensor_fault_models': self.sensor_fault_models,
                'detection_algorithms': self.detection_algorithms
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved fault injection data to {output_path}")


if __name__ == "__main__":
    # Test fault injection models
    print("🔄 Testing Fault Injection Models")
    
    # Initialize model
    fault_model = FaultInjectionModel()
    
    # Test refrigerant leak fault
    leak_fault = fault_model.inject_fault(
        "test_system", "compressor_001",
        FaultType.REFRIGERANT_LEAK, FaultSeverity.MODERATE
    )
    
    print(f"\n📊 Refrigerant Leak Fault:")
    print(f"Fault ID: {leak_fault.fault_id}")
    print(f"Severity: {leak_fault.severity.value}")
    print(f"Capacity impact: {leak_fault.capacity_impact_factor:.3f}")
    print(f"COP impact: {leak_fault.cop_impact_factor:.3f}")
    print(f"Detectable: {leak_fault.detectable}")
    print(f"Repair urgency: {leak_fault.repair_urgency}")
    
    # Test sensor fault
    sensor_fault = fault_model.inject_sensor_fault(
        "temp_sensor_001", "temperature_sensor", FaultSeverity.MINOR
    )
    
    print(f"\n📊 Sensor Fault:")
    print(f"Sensor ID: {sensor_fault.sensor_id}")
    print(f"Fault type: {sensor_fault.fault_type}")
    print(f"Drift rate: {sensor_fault.drift_rate:.6f} K/hour")
    print(f"Noise factor: {sensor_fault.noise_increase_factor:.2f}")
    
    # Test fault progression
    print(f"\n⏱️ Testing fault progression...")
    operating_conditions = {
        'average_temperature': 333.15,  # High temperature
        'load_factor': 1.2,             # Overload
        'cycles_per_hour': 8            # High cycling
    }
    
    progressed_fault = fault_model.progress_fault(leak_fault, 100.0, operating_conditions)
    print(f"After 100 hours:")
    print(f"Current magnitude: {progressed_fault.current_magnitude:.3f}")
    print(f"Capacity impact: {progressed_fault.capacity_impact_factor:.3f}")
    print(f"Detection confidence: {progressed_fault.detection_confidence:.3f}")
    
    # Test fault scenario generation
    print(f"\n🎲 Generating fault scenario...")
    fault_scenario = fault_model.generate_fault_scenario("test_system", 8760, 0.0001)
    print(f"Generated {len(fault_scenario)} faults over 1 year")
    
    if fault_scenario:
        fault_types = [f.fault_type.value for f in fault_scenario]
        print(f"Fault types: {set(fault_types)}")
    
    # Save test data
    fault_model.save_fault_data([leak_fault, progressed_fault], "fault_injection_data.json")
    
    print("\n✅ Fault injection models testing complete!")