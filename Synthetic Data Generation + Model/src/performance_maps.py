#!/usr/bin/env python3
"""
Component-Level Performance Maps and Degradation Models
======================================================

This module implements realistic performance maps and degradation models for
AIRTHIUM heat pump components, capturing aging effects, efficiency degradation,
and component health evolution over time.

Key Features:
- Compressor performance maps with wear patterns
- Heat exchanger fouling and effectiveness degradation
- Expansion valve flow characteristics and aging
- Time-dependent degradation models
- Maintenance cycle effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.interpolate import interp2d, griddata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from component_models import CompressorType, HeatExchangerType, ExpansionType

@dataclass
class ComponentHealth:
    """Component health and degradation state."""
    component_id: str
    component_type: str
    age_hours: float  # Operating hours
    health_index: float  # 0-1 (1 = new, 0 = failed)
    efficiency_factor: float  # Efficiency multiplier
    capacity_factor: float  # Capacity multiplier
    degradation_rate: float  # %/year
    maintenance_history: List[Dict[str, Any]]
    fault_indicators: Dict[str, float]

@dataclass
class PerformanceMap:
    """Performance map data structure."""
    component_id: str
    map_type: str  # 'efficiency', 'capacity', 'power'
    x_axis: np.ndarray  # Independent variable 1
    y_axis: np.ndarray  # Independent variable 2
    z_values: np.ndarray  # Dependent variable (performance metric)
    x_label: str
    y_label: str
    z_label: str
    operating_envelope: Dict[str, Tuple[float, float]]
    degradation_factors: Dict[str, float]

class ComponentPerformanceMaps:
    """Generate and manage component performance maps with degradation."""
    
    def __init__(self):
        """Initialize performance map generator."""
        self.component_specs = self._load_component_specifications()
        self.degradation_models = self._initialize_degradation_models()
        
    def _load_component_specifications(self) -> Dict[str, Dict]:
        """Load detailed component specifications."""
        return {
            'scroll_compressor': {
                'displacement': 0.15,  # m³/s
                'speed_range': [600, 6000],  # rpm
                'pressure_ratio_range': [1.5, 8.0],
                'efficiency_nominal': 0.85,
                'power_nominal': 5000,  # W
                'oil_flow_rate': 0.001,  # kg/s
                'bearing_life': 50000,  # hours
                'valve_life': 40000,   # hours
                'degradation_factors': {
                    'bearing_wear': 0.02,  # %/year
                    'valve_leakage': 0.015,  # %/year
                    'oil_degradation': 0.01,  # %/year
                    'refrigerant_contamination': 0.005  # %/year
                }
            },
            'finned_tube_evaporator': {
                'area': 40.0,  # m²
                'volume': 0.02,  # m³
                'air_flow_range': [2000, 8000],  # m³/h
                'effectiveness_nominal': 0.75,
                'fouling_resistance_clean': 0.0001,  # m²K/W
                'fouling_resistance_limit': 0.002,  # m²K/W
                'frost_accumulation_rate': 0.1,  # mm/day in winter
                'degradation_factors': {
                    'fouling_buildup': 0.05,  # %/year
                    'fin_corrosion': 0.02,  # %/year
                    'tube_scaling': 0.03,  # %/year
                    'fan_degradation': 0.01  # %/year
                }
            },
            'plate_condenser': {
                'area': 60.0,  # m²
                'volume': 0.015,  # m³
                'water_flow_range': [0.5, 2.0],  # m³/h
                'effectiveness_nominal': 0.85,
                'fouling_resistance_clean': 0.00005,  # m²K/W
                'fouling_resistance_limit': 0.001,  # m²K/W
                'plate_thickness': 0.0005,  # m
                'degradation_factors': {
                    'water_side_fouling': 0.08,  # %/year
                    'refrigerant_side_fouling': 0.03,  # %/year
                    'gasket_aging': 0.02,  # %/year
                    'plate_corrosion': 0.01  # %/year
                }
            },
            'electronic_expansion_valve': {
                'cv_range': [0.1, 2.0],
                'opening_range': [0, 100],  # %
                'response_time_new': 2.0,  # seconds
                'response_time_aged': 8.0,  # seconds
                'superheat_control_accuracy': 1.0,  # K
                'degradation_factors': {
                    'actuator_wear': 0.03,  # %/year
                    'seat_erosion': 0.02,  # %/year
                    'contamination': 0.04,  # %/year
                    'calibration_drift': 0.01  # %/year
                }
            }
        }
    
    def _initialize_degradation_models(self) -> Dict[str, Callable]:
        """Initialize degradation model functions."""
        return {
            'linear': lambda t, rate: max(0, 1 - rate * t / 8760),  # Linear degradation
            'exponential': lambda t, rate: np.exp(-rate * t / 8760),  # Exponential decay
            'weibull': lambda t, beta, eta: np.exp(-(t/eta)**beta),  # Weibull reliability
            'bathtub': lambda t, a, b, c: a * np.exp(-b*t) + c * (t**2),  # Bathtub curve
            'step': lambda t, intervals: self._step_degradation(t, intervals)  # Step degradation
        }
    
    def generate_compressor_performance_map(self, component_health: ComponentHealth) -> Dict[str, PerformanceMap]:
        """Generate compressor performance maps with degradation effects."""
        
        specs = self.component_specs['scroll_compressor']
        
        # Operating ranges
        speed_range = np.linspace(specs['speed_range'][0], specs['speed_range'][1], 20)
        pr_range = np.linspace(specs['pressure_ratio_range'][0], specs['pressure_ratio_range'][1], 15)
        
        # Create meshgrid
        Speed, PR = np.meshgrid(speed_range, pr_range)
        
        # Base efficiency map (new compressor)
        efficiency_base = self._calculate_base_compressor_efficiency(Speed, PR, specs)
        
        # Apply degradation effects
        efficiency_degraded = self._apply_compressor_degradation(
            efficiency_base, component_health, Speed, PR
        )
        
        # Power consumption map
        power_base = self._calculate_base_compressor_power(Speed, PR, specs)
        power_degraded = self._apply_power_degradation(power_base, component_health)
        
        # Volumetric efficiency map
        vol_efficiency_base = self._calculate_volumetric_efficiency(Speed, PR, specs)
        vol_efficiency_degraded = self._apply_volumetric_degradation(
            vol_efficiency_base, component_health
        )
        
        # Create performance maps
        maps = {
            'isentropic_efficiency': PerformanceMap(
                component_id=component_health.component_id,
                map_type='efficiency',
                x_axis=speed_range,
                y_axis=pr_range,
                z_values=efficiency_degraded,
                x_label='Compressor Speed [rpm]',
                y_label='Pressure Ratio [-]',
                z_label='Isentropic Efficiency [-]',
                operating_envelope={'speed': specs['speed_range'], 'pr': specs['pressure_ratio_range']},
                degradation_factors=specs['degradation_factors']
            ),
            'power_consumption': PerformanceMap(
                component_id=component_health.component_id,
                map_type='power',
                x_axis=speed_range,
                y_axis=pr_range,
                z_values=power_degraded,
                x_label='Compressor Speed [rpm]',
                y_label='Pressure Ratio [-]',
                z_label='Power Consumption [W]',
                operating_envelope={'speed': specs['speed_range'], 'pr': specs['pressure_ratio_range']},
                degradation_factors=specs['degradation_factors']
            ),
            'volumetric_efficiency': PerformanceMap(
                component_id=component_health.component_id,
                map_type='capacity',
                x_axis=speed_range,
                y_axis=pr_range,
                z_values=vol_efficiency_degraded,
                x_label='Compressor Speed [rpm]',
                y_label='Pressure Ratio [-]',
                z_label='Volumetric Efficiency [-]',
                operating_envelope={'speed': specs['speed_range'], 'pr': specs['pressure_ratio_range']},
                degradation_factors=specs['degradation_factors']
            )
        }
        
        return maps
    
    def generate_heat_exchanger_performance_map(self, component_health: ComponentHealth,
                                              hx_type: str = 'evaporator') -> Dict[str, PerformanceMap]:
        """Generate heat exchanger performance maps with fouling effects."""
        
        if hx_type == 'evaporator':
            specs = self.component_specs['finned_tube_evaporator']
            flow_range = np.linspace(specs['air_flow_range'][0], specs['air_flow_range'][1], 15)
            temp_diff_range = np.linspace(5, 25, 12)  # Temperature difference [K]
        else:  # condenser
            specs = self.component_specs['plate_condenser']
            flow_range = np.linspace(specs['water_flow_range'][0], specs['water_flow_range'][1], 15)
            temp_diff_range = np.linspace(3, 15, 12)  # Temperature difference [K]
        
        # Create meshgrid
        Flow, TempDiff = np.meshgrid(flow_range, temp_diff_range)
        
        # Base effectiveness map
        effectiveness_base = self._calculate_base_hx_effectiveness(Flow, TempDiff, specs, hx_type)
        
        # Apply fouling degradation
        effectiveness_degraded = self._apply_fouling_degradation(
            effectiveness_base, component_health, specs
        )
        
        # Heat transfer coefficient map
        htc_base = self._calculate_base_htc(Flow, TempDiff, specs, hx_type)
        htc_degraded = self._apply_htc_degradation(htc_base, component_health, specs)
        
        # Pressure drop map
        pressure_drop = self._calculate_pressure_drop_map(Flow, TempDiff, specs, component_health)
        
        # Create performance maps
        flow_label = 'Air Flow Rate [m³/h]' if hx_type == 'evaporator' else 'Water Flow Rate [m³/h]'
        
        maps = {
            'effectiveness': PerformanceMap(
                component_id=component_health.component_id,
                map_type='effectiveness',
                x_axis=flow_range,
                y_axis=temp_diff_range,
                z_values=effectiveness_degraded,
                x_label=flow_label,
                y_label='Temperature Difference [K]',
                z_label='Effectiveness [-]',
                operating_envelope={'flow': (flow_range[0], flow_range[-1]), 
                                  'temp_diff': (temp_diff_range[0], temp_diff_range[-1])},
                degradation_factors=specs['degradation_factors']
            ),
            'heat_transfer_coefficient': PerformanceMap(
                component_id=component_health.component_id,
                map_type='htc',
                x_axis=flow_range,
                y_axis=temp_diff_range,
                z_values=htc_degraded,
                x_label=flow_label,
                y_label='Temperature Difference [K]',
                z_label='Heat Transfer Coefficient [W/m²K]',
                operating_envelope={'flow': (flow_range[0], flow_range[-1]), 
                                  'temp_diff': (temp_diff_range[0], temp_diff_range[-1])},
                degradation_factors=specs['degradation_factors']
            ),
            'pressure_drop': PerformanceMap(
                component_id=component_health.component_id,
                map_type='pressure_drop',
                x_axis=flow_range,
                y_axis=temp_diff_range,
                z_values=pressure_drop,
                x_label=flow_label,
                y_label='Temperature Difference [K]',
                z_label='Pressure Drop [Pa]',
                operating_envelope={'flow': (flow_range[0], flow_range[-1]), 
                                  'temp_diff': (temp_diff_range[0], temp_diff_range[-1])},
                degradation_factors=specs['degradation_factors']
            )
        }
        
        return maps
    
    def generate_expansion_valve_performance_map(self, component_health: ComponentHealth) -> Dict[str, PerformanceMap]:
        """Generate expansion valve performance maps with aging effects."""
        
        specs = self.component_specs['electronic_expansion_valve']
        
        # Operating ranges
        opening_range = np.linspace(specs['opening_range'][0], specs['opening_range'][1], 20)
        pressure_diff_range = np.linspace(500000, 2000000, 15)  # Pa
        
        # Create meshgrid
        Opening, PressureDiff = np.meshgrid(opening_range, pressure_diff_range)
        
        # Flow coefficient map
        cv_base = self._calculate_base_flow_coefficient(Opening, PressureDiff, specs)
        cv_degraded = self._apply_valve_degradation(cv_base, component_health, specs)
        
        # Response time map
        response_time = self._calculate_response_time_map(Opening, component_health, specs)
        
        # Control accuracy map
        control_accuracy = self._calculate_control_accuracy_map(Opening, component_health, specs)
        
        # Create performance maps
        maps = {
            'flow_coefficient': PerformanceMap(
                component_id=component_health.component_id,
                map_type='flow',
                x_axis=opening_range,
                y_axis=pressure_diff_range,
                z_values=cv_degraded,
                x_label='Valve Opening [%]',
                y_label='Pressure Difference [Pa]',
                z_label='Flow Coefficient [-]',
                operating_envelope={'opening': specs['opening_range'], 
                                  'pressure_diff': (500000, 2000000)},
                degradation_factors=specs['degradation_factors']
            ),
            'response_time': PerformanceMap(
                component_id=component_health.component_id,
                map_type='response',
                x_axis=opening_range,
                y_axis=np.array([component_health.age_hours]),
                z_values=response_time,
                x_label='Valve Opening [%]',
                y_label='Operating Hours [h]',
                z_label='Response Time [s]',
                operating_envelope={'opening': specs['opening_range'], 
                                  'age': (0, 100000)},
                degradation_factors=specs['degradation_factors']
            )
        }
        
        return maps
    
    def _calculate_base_compressor_efficiency(self, Speed: np.ndarray, PR: np.ndarray, 
                                            specs: Dict) -> np.ndarray:
        """Calculate base compressor efficiency map."""
        
        # Normalized speed and pressure ratio
        speed_norm = (Speed - specs['speed_range'][0]) / (specs['speed_range'][1] - specs['speed_range'][0])
        pr_norm = (PR - specs['pressure_ratio_range'][0]) / (specs['pressure_ratio_range'][1] - specs['pressure_ratio_range'][0])
        
        # Efficiency model based on typical scroll compressor characteristics
        efficiency = specs['efficiency_nominal'] * (
            1.0 - 0.15 * (speed_norm - 0.7)**2 - 0.2 * (pr_norm - 0.5)**2 - 
            0.1 * speed_norm * pr_norm + 0.05 * np.sin(2 * np.pi * speed_norm)
        )
        
        # Ensure realistic bounds
        efficiency = np.clip(efficiency, 0.4, 0.95)
        
        return efficiency
    
    def _apply_compressor_degradation(self, efficiency_base: np.ndarray, 
                                    health: ComponentHealth, Speed: np.ndarray, 
                                    PR: np.ndarray) -> np.ndarray:
        """Apply degradation effects to compressor efficiency."""
        
        # Age-based degradation
        age_years = health.age_hours / 8760
        
        # Bearing wear effect (increases with speed)
        bearing_factor = 1 - 0.02 * age_years * (Speed / 6000)**1.5
        
        # Valve leakage effect (increases with pressure ratio)
        valve_factor = 1 - 0.015 * age_years * (PR / 8.0)**2
        
        # Oil degradation effect
        oil_factor = 1 - 0.01 * age_years
        
        # Combined degradation
        degradation_factor = bearing_factor * valve_factor * oil_factor * health.efficiency_factor
        
        return efficiency_base * np.clip(degradation_factor, 0.3, 1.0)
    
    def _calculate_base_compressor_power(self, Speed: np.ndarray, PR: np.ndarray, 
                                       specs: Dict) -> np.ndarray:
        """Calculate base compressor power consumption."""
        
        # Power model based on speed and pressure ratio
        power = specs['power_nominal'] * (Speed / 1800)**2.5 * (PR / 3.0)**1.2
        
        return np.clip(power, 500, 15000)  # Realistic power range
    
    def _apply_power_degradation(self, power_base: np.ndarray, 
                               health: ComponentHealth) -> np.ndarray:
        """Apply degradation effects to power consumption."""
        
        # Degraded components typically consume more power
        power_increase = 1 + (1 - health.efficiency_factor) * 0.3
        
        return power_base * power_increase
    
    def _calculate_volumetric_efficiency(self, Speed: np.ndarray, PR: np.ndarray, 
                                       specs: Dict) -> np.ndarray:
        """Calculate volumetric efficiency map."""
        
        # Volumetric efficiency decreases with pressure ratio and varies with speed
        vol_eff = 0.95 - 0.05 * (PR - 1.5) / 6.5 - 0.02 * np.abs(Speed - 1800) / 1800
        
        return np.clip(vol_eff, 0.6, 0.98)
    
    def _apply_volumetric_degradation(self, vol_eff_base: np.ndarray, 
                                    health: ComponentHealth) -> np.ndarray:
        """Apply degradation to volumetric efficiency."""
        
        # Valve leakage and wear reduce volumetric efficiency
        degradation = health.efficiency_factor * 0.95 + 0.05
        
        return vol_eff_base * degradation
    
    def _calculate_base_hx_effectiveness(self, Flow: np.ndarray, TempDiff: np.ndarray, 
                                       specs: Dict, hx_type: str) -> np.ndarray:
        """Calculate base heat exchanger effectiveness."""
        
        if hx_type == 'evaporator':
            # Air-side heat exchanger
            effectiveness = specs['effectiveness_nominal'] * (
                1 - 0.1 * np.exp(-Flow / 3000) - 0.05 * np.exp(-TempDiff / 10)
            )
        else:
            # Water-side heat exchanger
            effectiveness = specs['effectiveness_nominal'] * (
                1 - 0.08 * np.exp(-Flow / 1.0) - 0.03 * np.exp(-TempDiff / 8)
            )
        
        return np.clip(effectiveness, 0.4, 0.95)
    
    def _apply_fouling_degradation(self, effectiveness_base: np.ndarray, 
                                 health: ComponentHealth, specs: Dict) -> np.ndarray:
        """Apply fouling degradation to heat exchanger effectiveness."""
        
        # Fouling reduces effectiveness over time
        age_years = health.age_hours / 8760
        
        # Fouling resistance increases with age
        fouling_current = specs['fouling_resistance_clean'] + (
            specs['fouling_resistance_limit'] - specs['fouling_resistance_clean']
        ) * (1 - np.exp(-0.1 * age_years))
        
        # Effectiveness reduction due to fouling
        fouling_factor = specs['fouling_resistance_clean'] / fouling_current
        
        return effectiveness_base * fouling_factor * health.efficiency_factor
    
    def _calculate_base_htc(self, Flow: np.ndarray, TempDiff: np.ndarray, 
                          specs: Dict, hx_type: str) -> np.ndarray:
        """Calculate base heat transfer coefficient."""
        
        if hx_type == 'evaporator':
            # Air-side HTC
            htc = 50 * (Flow / 5000)**0.8 * (TempDiff / 15)**0.2
        else:
            # Water-side HTC
            htc = 2000 * (Flow / 1.0)**0.8 * (TempDiff / 10)**0.1
        
        return htc
    
    def _apply_htc_degradation(self, htc_base: np.ndarray, health: ComponentHealth, 
                             specs: Dict) -> np.ndarray:
        """Apply degradation to heat transfer coefficient."""
        
        return htc_base * health.efficiency_factor
    
    def _calculate_pressure_drop_map(self, Flow: np.ndarray, TempDiff: np.ndarray, 
                                   specs: Dict, health: ComponentHealth) -> np.ndarray:
        """Calculate pressure drop map with fouling effects."""
        
        # Base pressure drop increases with flow rate squared
        dp_base = 1000 * (Flow / np.mean(Flow))**2
        
        # Fouling increases pressure drop
        fouling_multiplier = 1 + (1 - health.efficiency_factor) * 2
        
        return dp_base * fouling_multiplier
    
    def _calculate_base_flow_coefficient(self, Opening: np.ndarray, PressureDiff: np.ndarray, 
                                       specs: Dict) -> np.ndarray:
        """Calculate base flow coefficient for expansion valve."""
        
        # Flow coefficient varies with opening percentage
        cv_max = specs['cv_range'][1]
        cv = cv_max * (Opening / 100)**0.8
        
        return cv
    
    def _apply_valve_degradation(self, cv_base: np.ndarray, health: ComponentHealth, 
                               specs: Dict) -> np.ndarray:
        """Apply degradation to valve flow coefficient."""
        
        # Seat erosion and contamination reduce flow coefficient
        return cv_base * health.efficiency_factor
    
    def _calculate_response_time_map(self, Opening: np.ndarray, health: ComponentHealth, 
                                   specs: Dict) -> np.ndarray:
        """Calculate valve response time with aging."""
        
        age_years = health.age_hours / 8760
        
        # Response time increases with age
        response_time = specs['response_time_new'] + (
            specs['response_time_aged'] - specs['response_time_new']
        ) * (1 - np.exp(-0.2 * age_years))
        
        # Response time may vary with opening position
        response_variation = 1 + 0.2 * np.abs(Opening - 50) / 50
        
        return response_time * response_variation
    
    def _calculate_control_accuracy_map(self, Opening: np.ndarray, health: ComponentHealth, 
                                      specs: Dict) -> np.ndarray:
        """Calculate control accuracy degradation."""
        
        age_years = health.age_hours / 8760
        
        # Control accuracy degrades with age
        accuracy = specs['superheat_control_accuracy'] * (1 + 0.5 * age_years)
        
        return np.full_like(Opening, accuracy)
    
    def _step_degradation(self, t: float, intervals: List[Tuple[float, float]]) -> float:
        """Step degradation function for maintenance cycles."""
        
        for time_threshold, degradation_level in intervals:
            if t < time_threshold:
                return degradation_level
        
        return intervals[-1][1]  # Return final degradation level
    
    def create_component_health(self, component_id: str, component_type: str, 
                              age_hours: float = 0, maintenance_history: List = None) -> ComponentHealth:
        """Create component health object with realistic degradation."""
        
        maintenance_history = maintenance_history or []
        
        # Calculate health index based on age and maintenance
        base_degradation_rate = 0.02  # 2% per year base rate
        age_years = age_hours / 8760
        
        # Health decreases with age but can be restored with maintenance
        health_index = max(0.1, 1.0 - base_degradation_rate * age_years)
        
        # Apply maintenance effects
        for maintenance in maintenance_history:
            if maintenance.get('type') == 'major_overhaul':
                health_index = min(1.0, health_index + 0.3)
            elif maintenance.get('type') == 'routine_maintenance':
                health_index = min(1.0, health_index + 0.1)
        
        # Efficiency and capacity factors
        efficiency_factor = 0.5 + 0.5 * health_index
        capacity_factor = 0.7 + 0.3 * health_index
        
        # Fault indicators
        fault_indicators = {
            'vibration_level': (1 - health_index) * 10,  # mm/s
            'noise_level': (1 - health_index) * 20,      # dB above normal
            'temperature_rise': (1 - health_index) * 15,  # K above normal
            'efficiency_loss': (1 - efficiency_factor) * 100  # %
        }
        
        return ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            age_hours=age_hours,
            health_index=health_index,
            efficiency_factor=efficiency_factor,
            capacity_factor=capacity_factor,
            degradation_rate=base_degradation_rate,
            maintenance_history=maintenance_history,
            fault_indicators=fault_indicators
        )
    
    def save_performance_maps(self, maps: Dict[str, PerformanceMap], 
                            output_path: str) -> None:
        """Save performance maps to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert maps to serializable format
        maps_data = {}
        for map_name, perf_map in maps.items():
            maps_data[map_name] = {
                'component_id': perf_map.component_id,
                'map_type': perf_map.map_type,
                'x_axis': perf_map.x_axis.tolist(),
                'y_axis': perf_map.y_axis.tolist(),
                'z_values': perf_map.z_values.tolist(),
                'x_label': perf_map.x_label,
                'y_label': perf_map.y_label,
                'z_label': perf_map.z_label,
                'operating_envelope': perf_map.operating_envelope,
                'degradation_factors': perf_map.degradation_factors
            }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(maps_data, f, indent=2)
        
        print(f"💾 Saved performance maps to {output_path}")


if __name__ == "__main__":
    # Test performance maps generation
    print("🔄 Testing Component Performance Maps")
    
    # Initialize generator
    perf_maps = ComponentPerformanceMaps()
    
    # Create component health objects
    compressor_health = perf_maps.create_component_health(
        "scroll_comp_001", "scroll_compressor", age_hours=15000
    )
    
    evaporator_health = perf_maps.create_component_health(
        "evap_001", "finned_tube_evaporator", age_hours=12000
    )
    
    valve_health = perf_maps.create_component_health(
        "exp_valve_001", "electronic_expansion_valve", age_hours=8000
    )
    
    print(f"\n📊 Component Health Status:")
    print(f"Compressor Health: {compressor_health.health_index:.2f}")
    print(f"Evaporator Health: {evaporator_health.health_index:.2f}")
    print(f"Valve Health: {valve_health.health_index:.2f}")
    
    # Generate performance maps
    print("\n🗺️ Generating performance maps...")
    
    comp_maps = perf_maps.generate_compressor_performance_map(compressor_health)
    evap_maps = perf_maps.generate_heat_exchanger_performance_map(evaporator_health, 'evaporator')
    valve_maps = perf_maps.generate_expansion_valve_performance_map(valve_health)
    
    print(f"✅ Generated {len(comp_maps)} compressor maps")
    print(f"✅ Generated {len(evap_maps)} evaporator maps")
    print(f"✅ Generated {len(valve_maps)} valve maps")
    
    # Save maps
    all_maps = {**comp_maps, **evap_maps, **valve_maps}
    perf_maps.save_performance_maps(all_maps, "component_performance_maps.json")
    
    print("\n✅ Component performance maps generation complete!")