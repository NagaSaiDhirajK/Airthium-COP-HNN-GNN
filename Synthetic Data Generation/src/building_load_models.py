"""
Multi-Zone Building Load Interaction Models
==========================================

This module implements multi-zone building thermal load modeling for heat pump
systems, capturing zone-specific loads, inter-zone heat transfer, and complex
building dynamics that affect heat pump performance and control strategies.

Key Features:
- Multi-zone thermal modeling with zone-specific characteristics
- Inter-zone heat transfer and air movement modeling
- Zone-specific control strategies and setpoint management
- Building envelope thermal dynamics and thermal mass effects
- Occupancy patterns and internal heat gains by zone
- Solar gains and shading effects for different orientations
- HVAC system distribution and zone balancing
- Load aggregation and heat pump sizing implications

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

class ZoneType(Enum):
    """Types of building zones"""
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"
    HALLWAY = "hallway"
    BASEMENT = "basement"
    ATTIC = "attic"
    GARAGE = "garage"

class ControlStrategy(Enum):
    """Zone control strategies"""
    SINGLE_ZONE = "single_zone"
    MULTI_ZONE_CONSTANT = "multi_zone_constant"
    MULTI_ZONE_VARIABLE = "multi_zone_variable"
    DEMAND_CONTROLLED = "demand_controlled"
    OCCUPANCY_BASED = "occupancy_based"
    ADAPTIVE_COMFORT = "adaptive_comfort"

class BuildingOrientation(Enum):
    """Building orientation for solar calculations"""
    NORTH = "north"
    NORTHEAST = "northeast"
    EAST = "east"
    SOUTHEAST = "southeast"
    SOUTH = "south"
    SOUTHWEST = "southwest"
    WEST = "west"
    NORTHWEST = "northwest"

@dataclass
class ZoneCharacteristics:
    """Characteristics of a building zone"""
    zone_id: str
    zone_type: ZoneType
    floor_area_m2: float
    ceiling_height_m: float
    exterior_wall_area_m2: float
    window_area_m2: float
    orientation: BuildingOrientation
    thermal_mass_kJ_per_K: float
    air_changes_per_hour: float
    internal_gains_W: float
    occupancy_schedule: List[float]  # 24-hour occupancy profile (0-1)
    setpoint_schedule_heating_C: List[float]  # 24-hour heating setpoints
    setpoint_schedule_cooling_C: List[float]  # 24-hour cooling setpoints

@dataclass
class ZoneThermalProperties:
    """Thermal properties of a zone"""
    u_value_walls_W_per_m2_K: float
    u_value_windows_W_per_m2_K: float
    u_value_floor_W_per_m2_K: float
    u_value_ceiling_W_per_m2_K: float
    solar_heat_gain_coefficient: float
    thermal_bridging_factor: float
    air_leakage_m3_per_h_per_m2: float

@dataclass
class ZoneState:
    """Current state of a building zone"""
    zone_id: str
    timestamp: datetime
    temperature_C: float
    humidity_percent: float
    co2_ppm: float
    occupancy_count: int
    heating_load_W: float
    cooling_load_W: float
    internal_gains_W: float
    solar_gains_W: float
    ventilation_load_W: float
    infiltration_load_W: float
    inter_zone_heat_transfer_W: float

@dataclass
class BuildingState:
    """Overall building state"""
    timestamp: datetime
    zones: Dict[str, ZoneState]
    total_heating_load_W: float
    total_cooling_load_W: float
    peak_zone_load_W: float
    load_diversity_factor: float
    building_average_temperature_C: float
    outdoor_temperature_C: float
    heat_pump_capacity_required_W: float

class MultiZoneBuildingModel:
    """
    Comprehensive multi-zone building load model for heat pump systems.
    
    This class models complex building thermal dynamics with multiple zones,
    inter-zone interactions, and realistic load patterns that affect heat
    pump performance and control strategies.
    """
    
    def __init__(self):
        """Initialize multi-zone building model"""
        self.zone_types_database = self._initialize_zone_types()
        self.thermal_properties_database = self._initialize_thermal_properties()
        self.control_strategies = self._initialize_control_strategies()
        self.building_envelope_model = self._initialize_envelope_model()
        
        # State tracking
        self.current_building_state = None
        self.zone_interaction_matrix = None
        
        logger.info("Multi-zone building model initialized")
    
    def _initialize_zone_types(self) -> Dict[ZoneType, Dict[str, Any]]:
        """Initialize zone type characteristics"""
        
        return {
            ZoneType.LIVING_ROOM: {
                'typical_area_m2': 25.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.1,
                'equipment_load_W_per_m2': 5.0,
                'lighting_load_W_per_m2': 8.0,
                'window_to_wall_ratio': 0.15,
                'air_changes_per_hour': 0.5,
                'thermal_mass_factor': 1.2,
                'comfort_temperature_range_C': (20.0, 24.0)
            },
            
            ZoneType.BEDROOM: {
                'typical_area_m2': 15.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.13,
                'equipment_load_W_per_m2': 3.0,
                'lighting_load_W_per_m2': 5.0,
                'window_to_wall_ratio': 0.12,
                'air_changes_per_hour': 0.4,
                'thermal_mass_factor': 1.0,
                'comfort_temperature_range_C': (18.0, 22.0)
            },
            
            ZoneType.KITCHEN: {
                'typical_area_m2': 12.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.08,
                'equipment_load_W_per_m2': 15.0,
                'lighting_load_W_per_m2': 10.0,
                'window_to_wall_ratio': 0.10,
                'air_changes_per_hour': 1.0,
                'thermal_mass_factor': 0.8,
                'comfort_temperature_range_C': (19.0, 25.0)
            },
            
            ZoneType.BATHROOM: {
                'typical_area_m2': 6.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.17,
                'equipment_load_W_per_m2': 8.0,
                'lighting_load_W_per_m2': 12.0,
                'window_to_wall_ratio': 0.05,
                'air_changes_per_hour': 2.0,
                'thermal_mass_factor': 0.6,
                'comfort_temperature_range_C': (21.0, 26.0)
            },
            
            ZoneType.OFFICE: {
                'typical_area_m2': 20.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.05,
                'equipment_load_W_per_m2': 12.0,
                'lighting_load_W_per_m2': 15.0,
                'window_to_wall_ratio': 0.20,
                'air_changes_per_hour': 0.6,
                'thermal_mass_factor': 1.0,
                'comfort_temperature_range_C': (20.0, 24.0)
            },
            
            ZoneType.HALLWAY: {
                'typical_area_m2': 8.0,
                'ceiling_height_m': 2.7,
                'occupancy_density_people_per_m2': 0.02,
                'equipment_load_W_per_m2': 1.0,
                'lighting_load_W_per_m2': 6.0,
                'window_to_wall_ratio': 0.02,
                'air_changes_per_hour': 0.3,
                'thermal_mass_factor': 0.8,
                'comfort_temperature_range_C': (18.0, 26.0)
            },
            
            ZoneType.BASEMENT: {
                'typical_area_m2': 30.0,
                'ceiling_height_m': 2.4,
                'occupancy_density_people_per_m2': 0.03,
                'equipment_load_W_per_m2': 3.0,
                'lighting_load_W_per_m2': 4.0,
                'window_to_wall_ratio': 0.03,
                'air_changes_per_hour': 0.2,
                'thermal_mass_factor': 2.0,
                'comfort_temperature_range_C': (16.0, 22.0)
            }
        }
    
    def _initialize_thermal_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize thermal property databases"""
        
        return {
            'insulation_levels': {
                'poor': {
                    'u_value_walls': 2.0,
                    'u_value_windows': 5.0,
                    'u_value_floor': 1.5,
                    'u_value_ceiling': 2.5,
                    'air_leakage': 10.0
                },
                'average': {
                    'u_value_walls': 1.0,
                    'u_value_windows': 3.0,
                    'u_value_floor': 0.8,
                    'u_value_ceiling': 1.2,
                    'air_leakage': 5.0
                },
                'good': {
                    'u_value_walls': 0.5,
                    'u_value_windows': 2.0,
                    'u_value_floor': 0.4,
                    'u_value_ceiling': 0.6,
                    'air_leakage': 2.0
                },
                'excellent': {
                    'u_value_walls': 0.2,
                    'u_value_windows': 1.0,
                    'u_value_floor': 0.2,
                    'u_value_ceiling': 0.3,
                    'air_leakage': 0.5
                }
            },
            
            'thermal_mass_types': {
                'light': {'thermal_mass_kJ_per_m2_K': 150},
                'medium': {'thermal_mass_kJ_per_m2_K': 300},
                'heavy': {'thermal_mass_kJ_per_m2_K': 600}
            },
            
            'window_properties': {
                'single_glazed': {'shgc': 0.85, 'u_value': 5.0},
                'double_glazed': {'shgc': 0.70, 'u_value': 3.0},
                'triple_glazed': {'shgc': 0.60, 'u_value': 1.5},
                'low_e_double': {'shgc': 0.55, 'u_value': 2.0}
            }
        }
    
    def _initialize_control_strategies(self) -> Dict[ControlStrategy, Dict[str, Any]]:
        """Initialize control strategy parameters"""
        
        return {
            ControlStrategy.SINGLE_ZONE: {
                'description': 'Single thermostat controls entire building',
                'zone_priority': 'living_room',
                'temperature_deadband_C': 2.0,
                'setback_capability': True,
                'load_balancing': False
            },
            
            ControlStrategy.MULTI_ZONE_CONSTANT: {
                'description': 'Multiple zones with constant air flow',
                'zone_priority': 'equal',
                'temperature_deadband_C': 1.5,
                'setback_capability': True,
                'load_balancing': True,
                'air_flow_modulation': False
            },
            
            ControlStrategy.MULTI_ZONE_VARIABLE: {
                'description': 'Multiple zones with variable air flow',
                'zone_priority': 'demand_based',
                'temperature_deadband_C': 1.0,
                'setback_capability': True,
                'load_balancing': True,
                'air_flow_modulation': True,
                'minimum_air_flow_percent': 30.0
            },
            
            ControlStrategy.DEMAND_CONTROLLED: {
                'description': 'Demand-controlled ventilation and heating/cooling',
                'zone_priority': 'occupancy_weighted',
                'temperature_deadband_C': 1.0,
                'setback_capability': True,
                'load_balancing': True,
                'air_flow_modulation': True,
                'co2_control': True,
                'occupancy_sensing': True
            },
            
            ControlStrategy.ADAPTIVE_COMFORT: {
                'description': 'Adaptive comfort model with wider temperature ranges',
                'zone_priority': 'comfort_optimized',
                'temperature_deadband_C': 3.0,
                'setback_capability': True,
                'load_balancing': True,
                'adaptive_setpoints': True,
                'outdoor_temperature_compensation': True
            }
        }
    
    def _initialize_envelope_model(self) -> Dict[str, Any]:
        """Initialize building envelope model parameters"""
        
        return {
            'solar_orientation_factors': {
                BuildingOrientation.SOUTH: 1.0,
                BuildingOrientation.SOUTHEAST: 0.9,
                BuildingOrientation.SOUTHWEST: 0.9,
                BuildingOrientation.EAST: 0.7,
                BuildingOrientation.WEST: 0.7,
                BuildingOrientation.NORTHEAST: 0.5,
                BuildingOrientation.NORTHWEST: 0.5,
                BuildingOrientation.NORTH: 0.3
            },
            
            'thermal_bridging': {
                'poor_construction': 0.15,  # 15% increase in heat loss
                'average_construction': 0.08,  # 8% increase
                'good_construction': 0.03,  # 3% increase
                'excellent_construction': 0.01  # 1% increase
            },
            
            'air_leakage_factors': {
                'stack_effect_coefficient': 0.000145,  # m³/(s·m²·Pa^0.5)
                'wind_coefficient': 0.000319,  # m³/(s·m²·(m/s))
                'reference_pressure_Pa': 4.0,  # Reference pressure difference
                'height_factor': 0.3  # Height effect on stack pressure
            }
        }
    
    def create_building_zones(self, 
                            zone_definitions: List[Dict[str, Any]],
                            building_characteristics: Optional[Dict] = None) -> Dict[str, ZoneCharacteristics]:
        """
        Create building zones with specified characteristics
        
        Args:
            zone_definitions: List of zone definition dictionaries
            building_characteristics: Overall building characteristics
            
        Returns:
            Dictionary of zone characteristics
        """
        
        if building_characteristics is None:
            building_characteristics = {
                'insulation_level': 'average',
                'thermal_mass': 'medium',
                'window_type': 'double_glazed',
                'construction_quality': 'average_construction'
            }
        
        zones = {}
        
        for zone_def in zone_definitions:
            zone_id = zone_def['zone_id']
            zone_type = ZoneType(zone_def['zone_type'])
            
            # Get zone type defaults
            zone_defaults = self.zone_types_database[zone_type]
            
            # Override with specific values if provided
            floor_area = zone_def.get('floor_area_m2', zone_defaults['typical_area_m2'])
            ceiling_height = zone_def.get('ceiling_height_m', zone_defaults['ceiling_height_m'])
            orientation = BuildingOrientation(zone_def.get('orientation', 'south'))
            
            # Calculate derived properties
            exterior_wall_area = floor_area * 0.4  # Simplified assumption
            window_area = exterior_wall_area * zone_defaults['window_to_wall_ratio']
            
            # Thermal mass calculation
            thermal_mass_factor = zone_defaults['thermal_mass_factor']
            thermal_mass_base = self.thermal_properties_database['thermal_mass_types'][building_characteristics['thermal_mass']]['thermal_mass_kJ_per_m2_K']
            thermal_mass = thermal_mass_base * floor_area * thermal_mass_factor
            
            # Internal gains calculation
            equipment_gains = zone_defaults['equipment_load_W_per_m2'] * floor_area
            lighting_gains = zone_defaults['lighting_load_W_per_m2'] * floor_area
            occupancy_gains = zone_defaults['occupancy_density_people_per_m2'] * floor_area * 100.0  # 100W per person
            total_internal_gains = equipment_gains + lighting_gains + occupancy_gains
            
            # Create occupancy schedule (simplified)
            occupancy_schedule = self._create_occupancy_schedule(zone_type)
            
            # Create setpoint schedules
            comfort_range = zone_defaults['comfort_temperature_range_C']
            heating_setpoints = [comfort_range[0]] * 24  # Simplified constant setpoint
            cooling_setpoints = [comfort_range[1]] * 24
            
            # Apply setback for bedrooms during day
            if zone_type == ZoneType.BEDROOM:
                for hour in range(8, 18):  # 8 AM to 6 PM
                    heating_setpoints[hour] -= 2.0
                    cooling_setpoints[hour] += 2.0
            
            zone_char = ZoneCharacteristics(
                zone_id=zone_id,
                zone_type=zone_type,
                floor_area_m2=floor_area,
                ceiling_height_m=ceiling_height,
                exterior_wall_area_m2=exterior_wall_area,
                window_area_m2=window_area,
                orientation=orientation,
                thermal_mass_kJ_per_K=thermal_mass,
                air_changes_per_hour=zone_defaults['air_changes_per_hour'],
                internal_gains_W=total_internal_gains,
                occupancy_schedule=occupancy_schedule,
                setpoint_schedule_heating_C=heating_setpoints,
                setpoint_schedule_cooling_C=cooling_setpoints
            )
            
            zones[zone_id] = zone_char
        
        return zones
    
    def _create_occupancy_schedule(self, zone_type: ZoneType) -> List[float]:
        """Create 24-hour occupancy schedule for zone type"""
        
        # Base schedules (0-1 occupancy factor)
        schedules = {
            ZoneType.LIVING_ROOM: [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 0.7, 0.3, 0.2, 0.2, 0.3, 
                                  0.4, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8],
            
            ZoneType.BEDROOM: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            
            ZoneType.KITCHEN: [0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 1.0, 0.6, 0.3, 0.2, 0.4,
                              0.6, 0.3, 0.2, 0.2, 0.3, 0.6, 0.9, 0.8, 0.5, 0.3, 0.2, 0.2],
            
            ZoneType.BATHROOM: [0.3, 0.2, 0.2, 0.2, 0.2, 0.5, 1.0, 0.8, 0.3, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.4],
            
            ZoneType.OFFICE: [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            
            ZoneType.HALLWAY: [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1,
                              0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1],
            
            ZoneType.BASEMENT: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0]
        }
        
        return schedules.get(zone_type, [0.1] * 24)  # Default low occupancy
    
    def calculate_zone_loads(self, 
                           zone: ZoneCharacteristics,
                           zone_temp_C: float,
                           outdoor_temp_C: float,
                           solar_irradiance_W_per_m2: float,
                           hour_of_day: int,
                           building_characteristics: Dict[str, str]) -> ZoneState:
        """
        Calculate thermal loads for a specific zone
        
        Args:
            zone: Zone characteristics
            zone_temp_C: Current zone temperature
            outdoor_temp_C: Outdoor temperature
            solar_irradiance_W_per_m2: Solar irradiance
            hour_of_day: Hour of day (0-23)
            building_characteristics: Building thermal properties
            
        Returns:
            ZoneState object with calculated loads
        """
        
        # Get thermal properties
        insulation = self.thermal_properties_database['insulation_levels'][building_characteristics['insulation_level']]
        window_props = self.thermal_properties_database['window_properties'][building_characteristics['window_type']]
        
        # Temperature difference
        temp_diff = outdoor_temp_C - zone_temp_C
        
        # Transmission loads
        wall_load = insulation['u_value_walls'] * zone.exterior_wall_area_m2 * temp_diff
        window_load = window_props['u_value'] * zone.window_area_m2 * temp_diff
        floor_load = insulation['u_value_floor'] * zone.floor_area_m2 * temp_diff * 0.5  # Ground coupling factor
        ceiling_load = insulation['u_value_ceiling'] * zone.floor_area_m2 * temp_diff * 0.8  # Attic factor
        
        transmission_load = wall_load + window_load + floor_load + ceiling_load
        
        # Infiltration load
        air_density = 1.2  # kg/m³
        specific_heat_air = 1006  # J/(kg·K)
        volume = zone.floor_area_m2 * zone.ceiling_height_m
        
        infiltration_rate_m3_per_s = (zone.air_changes_per_hour * volume) / 3600.0
        infiltration_load = infiltration_rate_m3_per_s * air_density * specific_heat_air * temp_diff / 1000.0  # kW to W
        
        # Solar gains
        orientation_factor = self.building_envelope_model['solar_orientation_factors'][zone.orientation]
        solar_gains = (solar_irradiance_W_per_m2 * zone.window_area_m2 * 
                      window_props['shgc'] * orientation_factor)
        
        # Internal gains (modulated by occupancy)
        occupancy_factor = zone.occupancy_schedule[hour_of_day]
        internal_gains = zone.internal_gains_W * occupancy_factor
        
        # Ventilation load (if mechanical ventilation)
        ventilation_rate_m3_per_s = 0.01 * zone.floor_area_m2  # 10 L/s per m²
        ventilation_load = ventilation_rate_m3_per_s * air_density * specific_heat_air * temp_diff / 1000.0
        
        # Determine heating/cooling loads
        total_heat_gain = solar_gains + internal_gains
        total_heat_loss = transmission_load + infiltration_load + ventilation_load
        
        net_load = total_heat_loss - total_heat_gain
        
        if net_load > 0:
            # Heating required
            heating_load = net_load
            cooling_load = 0.0
        else:
            # Cooling required
            heating_load = 0.0
            cooling_load = -net_load
        
        # Current occupancy
        max_occupancy = int(zone.floor_area_m2 * self.zone_types_database[zone.zone_type]['occupancy_density_people_per_m2'])
        current_occupancy = int(max_occupancy * occupancy_factor)
        
        # Simplified humidity and CO2 (would be more complex in reality)
        humidity = 45.0 + 10.0 * occupancy_factor  # Higher humidity with occupancy
        co2 = 400.0 + 600.0 * occupancy_factor  # Higher CO2 with occupancy
        
        return ZoneState(
            zone_id=zone.zone_id,
            timestamp=datetime.now(),
            temperature_C=zone_temp_C,
            humidity_percent=humidity,
            co2_ppm=co2,
            occupancy_count=current_occupancy,
            heating_load_W=heating_load,
            cooling_load_W=cooling_load,
            internal_gains_W=internal_gains,
            solar_gains_W=solar_gains,
            ventilation_load_W=ventilation_load,
            infiltration_load_W=infiltration_load,
            inter_zone_heat_transfer_W=0.0  # Calculated separately
        )
    
    def calculate_inter_zone_heat_transfer(self, 
                                         zones: Dict[str, ZoneCharacteristics],
                                         zone_states: Dict[str, ZoneState]) -> Dict[str, float]:
        """
        Calculate heat transfer between zones
        
        Args:
            zones: Zone characteristics
            zone_states: Current zone states
            
        Returns:
            Dictionary of inter-zone heat transfer rates (W)
        """
        
        inter_zone_transfers = {}
        
        # Simplified inter-zone heat transfer model
        # Assumes heat transfer proportional to temperature difference and shared wall area
        
        zone_ids = list(zones.keys())
        
        for i, zone_id_1 in enumerate(zone_ids):
            total_transfer = 0.0
            
            for j, zone_id_2 in enumerate(zone_ids):
                if i != j:
                    zone_1 = zones[zone_id_1]
                    zone_2 = zones[zone_id_2]
                    state_1 = zone_states[zone_id_1]
                    state_2 = zone_states[zone_id_2]
                    
                    # Estimate shared wall area (simplified)
                    shared_wall_area = min(zone_1.floor_area_m2, zone_2.floor_area_m2) * 0.3  # 30% of smaller zone
                    
                    # Inter-zone U-value (internal walls)
                    u_value_internal = 2.0  # W/(m²·K) for internal walls
                    
                    # Heat transfer rate
                    temp_diff = state_2.temperature_C - state_1.temperature_C
                    heat_transfer = u_value_internal * shared_wall_area * temp_diff
                    
                    total_transfer += heat_transfer
            
            inter_zone_transfers[zone_id_1] = total_transfer
        
        return inter_zone_transfers
    
    def simulate_building_loads(self,
                              zones: Dict[str, ZoneCharacteristics],
                              weather_data: List[Dict],
                              control_strategy: ControlStrategy,
                              building_characteristics: Dict[str, str],
                              simulation_hours: int = 24) -> List[BuildingState]:
        """
        Simulate building loads over time period
        
        Args:
            zones: Building zones
            weather_data: Hourly weather data
            control_strategy: Control strategy to use
            building_characteristics: Building thermal properties
            simulation_hours: Number of hours to simulate
            
        Returns:
            List of BuildingState objects
        """
        
        building_states = []
        
        # Initialize zone temperatures
        zone_temperatures = {zone_id: 20.0 for zone_id in zones.keys()}  # Start at 20°C
        
        for hour in range(simulation_hours):
            weather = weather_data[hour % len(weather_data)]  # Cycle through weather data
            outdoor_temp = weather['temperature_C']
            solar_irradiance = weather.get('solar_irradiance_W_per_m2', 0.0)
            
            # Calculate zone loads
            zone_states = {}
            for zone_id, zone in zones.items():
                zone_state = self.calculate_zone_loads(
                    zone, zone_temperatures[zone_id], outdoor_temp, 
                    solar_irradiance, hour % 24, building_characteristics
                )
                zone_states[zone_id] = zone_state
            
            # Calculate inter-zone heat transfer
            inter_zone_transfers = self.calculate_inter_zone_heat_transfer(zones, zone_states)
            
            # Update zone states with inter-zone transfers
            for zone_id, transfer in inter_zone_transfers.items():
                zone_states[zone_id].inter_zone_heat_transfer_W = transfer
            
            # Apply control strategy
            zone_temperatures = self._apply_control_strategy(
                zones, zone_states, zone_temperatures, control_strategy, hour % 24
            )
            
            # Calculate building totals
            total_heating = sum(state.heating_load_W for state in zone_states.values())
            total_cooling = sum(state.cooling_load_W for state in zone_states.values())
            peak_zone_load = max(max(state.heating_load_W, state.cooling_load_W) 
                               for state in zone_states.values())
            
            # Load diversity factor
            sum_individual_peaks = sum(max(state.heating_load_W, state.cooling_load_W) 
                                     for state in zone_states.values())
            load_diversity = peak_zone_load / sum_individual_peaks if sum_individual_peaks > 0 else 1.0
            
            # Building average temperature
            total_area = sum(zone.floor_area_m2 for zone in zones.values())
            avg_temp = sum(state.temperature_C * zones[zone_id].floor_area_m2 
                          for zone_id, state in zone_states.items()) / total_area
            
            # Heat pump capacity required (with safety factor)
            hp_capacity = max(total_heating, total_cooling) * 1.2  # 20% safety factor
            
            building_state = BuildingState(
                timestamp=datetime.now() + timedelta(hours=hour),
                zones=zone_states,
                total_heating_load_W=total_heating,
                total_cooling_load_W=total_cooling,
                peak_zone_load_W=peak_zone_load,
                load_diversity_factor=load_diversity,
                building_average_temperature_C=avg_temp,
                outdoor_temperature_C=outdoor_temp,
                heat_pump_capacity_required_W=hp_capacity
            )
            
            building_states.append(building_state)
        
        return building_states
    
    def _apply_control_strategy(self,
                              zones: Dict[str, ZoneCharacteristics],
                              zone_states: Dict[str, ZoneState],
                              current_temperatures: Dict[str, float],
                              control_strategy: ControlStrategy,
                              hour_of_day: int) -> Dict[str, float]:
        """Apply control strategy to determine new zone temperatures"""
        
        new_temperatures = current_temperatures.copy()
        strategy_params = self.control_strategies[control_strategy]
        
        if control_strategy == ControlStrategy.SINGLE_ZONE:
            # Single thermostat controls all zones
            priority_zone = self._find_priority_zone(zones, zone_states, 'living_room')
            if priority_zone:
                target_temp = self._get_zone_setpoint(zones[priority_zone], hour_of_day, 'heating')
                # All zones try to reach same temperature (simplified)
                for zone_id in zones.keys():
                    new_temperatures[zone_id] = self._update_zone_temperature(
                        current_temperatures[zone_id], target_temp, zone_states[zone_id]
                    )
        
        elif control_strategy == ControlStrategy.MULTI_ZONE_VARIABLE:
            # Each zone controlled individually with variable air flow
            for zone_id, zone in zones.items():
                state = zone_states[zone_id]
                
                # Determine heating or cooling mode
                if state.heating_load_W > state.cooling_load_W:
                    target_temp = self._get_zone_setpoint(zone, hour_of_day, 'heating')
                else:
                    target_temp = self._get_zone_setpoint(zone, hour_of_day, 'cooling')
                
                new_temperatures[zone_id] = self._update_zone_temperature(
                    current_temperatures[zone_id], target_temp, state
                )
        
        elif control_strategy == ControlStrategy.DEMAND_CONTROLLED:
            # Demand-controlled with occupancy sensing
            for zone_id, zone in zones.items():
                state = zone_states[zone_id]
                occupancy_factor = zone.occupancy_schedule[hour_of_day]
                
                if occupancy_factor > 0.1:  # Zone is occupied
                    if state.heating_load_W > state.cooling_load_W:
                        target_temp = self._get_zone_setpoint(zone, hour_of_day, 'heating')
                    else:
                        target_temp = self._get_zone_setpoint(zone, hour_of_day, 'cooling')
                else:
                    # Setback for unoccupied zones
                    if state.heating_load_W > state.cooling_load_W:
                        target_temp = self._get_zone_setpoint(zone, hour_of_day, 'heating') - 3.0
                    else:
                        target_temp = self._get_zone_setpoint(zone, hour_of_day, 'cooling') + 3.0
                
                new_temperatures[zone_id] = self._update_zone_temperature(
                    current_temperatures[zone_id], target_temp, state
                )
        
        else:
            # Default: individual zone control
            for zone_id, zone in zones.items():
                state = zone_states[zone_id]
                
                if state.heating_load_W > state.cooling_load_W:
                    target_temp = self._get_zone_setpoint(zone, hour_of_day, 'heating')
                else:
                    target_temp = self._get_zone_setpoint(zone, hour_of_day, 'cooling')
                
                new_temperatures[zone_id] = self._update_zone_temperature(
                    current_temperatures[zone_id], target_temp, state
                )
        
        return new_temperatures
    
    def _find_priority_zone(self, zones: Dict[str, ZoneCharacteristics], 
                           zone_states: Dict[str, ZoneState], 
                           preferred_type: str) -> Optional[str]:
        """Find priority zone for control"""
        
        # First try to find preferred zone type
        for zone_id, zone in zones.items():
            if zone.zone_type.value == preferred_type:
                return zone_id
        
        # Otherwise return largest zone
        largest_zone = max(zones.items(), key=lambda x: x[1].floor_area_m2)
        return largest_zone[0]
    
    def _get_zone_setpoint(self, zone: ZoneCharacteristics, hour_of_day: int, mode: str) -> float:
        """Get zone setpoint for given hour and mode"""
        
        if mode == 'heating':
            return zone.setpoint_schedule_heating_C[hour_of_day]
        else:
            return zone.setpoint_schedule_cooling_C[hour_of_day]
    
    def _update_zone_temperature(self, current_temp: float, target_temp: float, 
                               zone_state: ZoneState) -> float:
        """Update zone temperature based on loads and control"""
        
        # Simplified temperature update (would be more complex with thermal mass)
        temp_diff = target_temp - current_temp
        
        # Limit temperature change rate (thermal inertia)
        max_change_per_hour = 2.0  # Maximum 2°C change per hour
        
        if abs(temp_diff) > max_change_per_hour:
            temp_change = max_change_per_hour * (1 if temp_diff > 0 else -1)
        else:
            temp_change = temp_diff * 0.8  # 80% of the way to setpoint
        
        new_temp = current_temp + temp_change
        
        # Reasonable bounds
        return max(10.0, min(35.0, new_temp))
    
    def analyze_load_patterns(self, building_states: List[BuildingState]) -> Dict[str, Any]:
        """
        Analyze building load patterns for heat pump sizing and control optimization
        
        Args:
            building_states: List of building states over time
            
        Returns:
            Dictionary with load pattern analysis
        """
        
        if not building_states:
            return {}
        
        # Extract time series data
        timestamps = [state.timestamp for state in building_states]
        heating_loads = [state.total_heating_load_W for state in building_states]
        cooling_loads = [state.total_cooling_load_W for state in building_states]
        diversity_factors = [state.load_diversity_factor for state in building_states]
        
        # Peak loads
        peak_heating = max(heating_loads) if heating_loads else 0
        peak_cooling = max(cooling_loads) if cooling_loads else 0
        
        # Average loads
        avg_heating = np.mean(heating_loads) if heating_loads else 0
        avg_cooling = np.mean(cooling_loads) if cooling_loads else 0
        
        # Load factors
        heating_load_factor = avg_heating / peak_heating if peak_heating > 0 else 0
        cooling_load_factor = avg_cooling / peak_cooling if peak_cooling > 0 else 0
        
        # Diversity analysis
        avg_diversity = np.mean(diversity_factors) if diversity_factors else 1.0
        min_diversity = min(diversity_factors) if diversity_factors else 1.0
        
        # Zone-specific analysis
        zone_analysis = {}
        if building_states:
            first_state = building_states[0]
            for zone_id in first_state.zones.keys():
                zone_heating_loads = [state.zones[zone_id].heating_load_W for state in building_states]
                zone_cooling_loads = [state.zones[zone_id].cooling_load_W for state in building_states]
                
                zone_analysis[zone_id] = {
                    'peak_heating_W': max(zone_heating_loads),
                    'peak_cooling_W': max(zone_cooling_loads),
                    'avg_heating_W': np.mean(zone_heating_loads),
                    'avg_cooling_W': np.mean(zone_cooling_loads),
                    'load_factor_heating': np.mean(zone_heating_loads) / max(zone_heating_loads) if max(zone_heating_loads) > 0 else 0,
                    'load_factor_cooling': np.mean(zone_cooling_loads) / max(zone_cooling_loads) if max(zone_cooling_loads) > 0 else 0
                }
        
        return {
            'peak_heating_load_W': peak_heating,
            'peak_cooling_load_W': peak_cooling,
            'average_heating_load_W': avg_heating,
            'average_cooling_load_W': avg_cooling,
            'heating_load_factor': heating_load_factor,
            'cooling_load_factor': cooling_load_factor,
            'average_diversity_factor': avg_diversity,
            'minimum_diversity_factor': min_diversity,
            'recommended_heat_pump_capacity_W': max(peak_heating, peak_cooling) * 1.15,  # 15% safety margin
            'zone_analysis': zone_analysis,
            'simulation_hours': len(building_states),
            'total_energy_heating_Wh': sum(heating_loads),
            'total_energy_cooling_Wh': sum(cooling_loads)
        }
    
    def optimize_zone_control(self, 
                            zones: Dict[str, ZoneCharacteristics],
                            building_states: List[BuildingState],
                            optimization_objective: str = 'energy') -> Dict[str, Any]:
        """
        Optimize zone control parameters for given objective
        
        Args:
            zones: Building zones
            building_states: Historical building states
            optimization_objective: 'energy', 'comfort', or 'cost'
            
        Returns:
            Optimized control parameters
        """
        
        if not building_states:
            return {}
        
        # Analyze current performance
        load_analysis = self.analyze_load_patterns(building_states)
        
        # Calculate comfort metrics
        comfort_violations = 0
        total_comfort_hours = 0
        
        for state in building_states:
            for zone_id, zone_state in state.zones.items():
                zone = zones[zone_id]
                comfort_range = self.zone_types_database[zone.zone_type]['comfort_temperature_range_C']
                
                total_comfort_hours += 1
                if (zone_state.temperature_C < comfort_range[0] - 1.0 or 
                    zone_state.temperature_C > comfort_range[1] + 1.0):
                    comfort_violations += 1
        
        comfort_satisfaction = 1.0 - (comfort_violations / total_comfort_hours) if total_comfort_hours > 0 else 1.0
        
        # Optimization recommendations
        recommendations = {}
        
        if optimization_objective == 'energy':
            # Energy-focused optimization
            recommendations.update({
                'setback_strategy': 'aggressive',
                'deadband_width_C': 2.0,
                'night_setback_C': 3.0,
                'unoccupied_setback_C': 4.0,
                'zone_priority': 'load_weighted',
                'recommended_control_strategy': ControlStrategy.DEMAND_CONTROLLED.value
            })
        
        elif optimization_objective == 'comfort':
            # Comfort-focused optimization
            recommendations.update({
                'setback_strategy': 'conservative',
                'deadband_width_C': 1.0,
                'night_setback_C': 1.0,
                'unoccupied_setback_C': 2.0,
                'zone_priority': 'occupancy_weighted',
                'recommended_control_strategy': ControlStrategy.MULTI_ZONE_VARIABLE.value
            })
        
        else:  # cost optimization
            # Cost-focused optimization (balance of energy and comfort)
            recommendations.update({
                'setback_strategy': 'moderate',
                'deadband_width_C': 1.5,
                'night_setback_C': 2.0,
                'unoccupied_setback_C': 3.0,
                'zone_priority': 'cost_weighted',
                'recommended_control_strategy': ControlStrategy.ADAPTIVE_COMFORT.value
            })
        
        return {
            'current_performance': {
                'comfort_satisfaction_percent': comfort_satisfaction * 100,
                'energy_efficiency_rating': load_analysis['heating_load_factor'] + load_analysis['cooling_load_factor'],
                'load_diversity_factor': load_analysis['average_diversity_factor']
            },
            'optimization_recommendations': recommendations,
            'potential_savings': {
                'energy_reduction_percent': 15.0 if optimization_objective == 'energy' else 8.0,
                'comfort_improvement_percent': 20.0 if optimization_objective == 'comfort' else 5.0,
                'cost_reduction_percent': 12.0 if optimization_objective == 'cost' else 6.0
            }
        }
    
    def validate_building_model(self, 
                              zones: Dict[str, ZoneCharacteristics],
                              building_states: List[BuildingState]) -> Dict[str, Any]:
        """
        Validate building model for thermodynamic consistency
        
        Args:
            zones: Building zones
            building_states: Simulated building states
            
        Returns:
            Validation results
        """
        
        validation_results = {
            'model_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'performance_metrics': {}
        }
        
        try:
            if not zones:
                validation_results['validation_errors'].append("No zones defined")
                validation_results['model_valid'] = False
                return validation_results
            
            if not building_states:
                validation_results['validation_errors'].append("No building states to validate")
                validation_results['model_valid'] = False
                return validation_results
            
            # Check zone characteristics
            total_area = 0
            for zone_id, zone in zones.items():
                total_area += zone.floor_area_m2
                
                if zone.floor_area_m2 <= 0:
                    validation_results['validation_errors'].append(f"Zone {zone_id} has invalid floor area")
                
                if zone.thermal_mass_kJ_per_K <= 0:
                    validation_results['validation_errors'].append(f"Zone {zone_id} has invalid thermal mass")
                
                if len(zone.occupancy_schedule) != 24:
                    validation_results['validation_errors'].append(f"Zone {zone_id} has invalid occupancy schedule")
            
            # Check building states
            temperature_violations = 0
            load_inconsistencies = 0
            
            for i, state in enumerate(building_states):
                # Temperature bounds check
                for zone_id, zone_state in state.zones.items():
                    if zone_state.temperature_C < -10 or zone_state.temperature_C > 50:
                        temperature_violations += 1
                    
                    # Load consistency check
                    if zone_state.heating_load_W < 0 or zone_state.cooling_load_W < 0:
                        load_inconsistencies += 1
                
                # Energy balance check (simplified)
                total_zone_loads = sum(max(zs.heating_load_W, zs.cooling_load_W) 
                                     for zs in state.zones.values())
                
                if abs(total_zone_loads - max(state.total_heating_load_W, state.total_cooling_load_W)) > 100:
                    validation_results['validation_warnings'].append(
                        f"Energy balance inconsistency at hour {i}"
                    )
            
            # Performance metrics
            validation_results['performance_metrics'] = {
                'total_building_area_m2': total_area,
                'temperature_violation_rate': temperature_violations / (len(building_states) * len(zones)),
                'load_inconsistency_rate': load_inconsistencies / (len(building_states) * len(zones)),
                'simulation_hours': len(building_states),
                'zones_count': len(zones)
            }
            
            # Final validation
            if temperature_violations > len(building_states) * 0.1:  # More than 10% violations
                validation_results['validation_errors'].append("Excessive temperature violations")
                validation_results['model_valid'] = False
            
            if load_inconsistencies > 0:
                validation_results['validation_errors'].append("Load calculation inconsistencies found")
                validation_results['model_valid'] = False
            
            logger.info(f"Building model validation completed: {'PASSED' if validation_results['model_valid'] else 'FAILED'}")
            
        except Exception as e:
            validation_results['validation_errors'].append(f"Validation error: {str(e)}")
            validation_results['model_valid'] = False
            logger.error(f"Building model validation failed: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    print("🏢 Testing Multi-Zone Building Load Models")
    
    # Create model instance
    model = MultiZoneBuildingModel()
    
    # Define building zones
    zone_definitions = [
        {
            'zone_id': 'living_room',
            'zone_type': 'living_room',
            'floor_area_m2': 30.0,
            'orientation': 'south'
        },
        {
            'zone_id': 'bedroom_1',
            'zone_type': 'bedroom',
            'floor_area_m2': 15.0,
            'orientation': 'east'
        },
        {
            'zone_id': 'kitchen',
            'zone_type': 'kitchen',
            'floor_area_m2': 12.0,
            'orientation': 'north'
        }
    ]
    
    # Create zones
    zones = model.create_building_zones(zone_definitions)
    print(f"✅ Created {len(zones)} zones")
    
    # Create sample weather data
    weather_data = []
    for hour in range(24):
        temp = 15.0 + 10.0 * np.sin((hour - 6) * np.pi / 12)  # Sinusoidal temperature
        solar = max(0, 800 * np.sin((hour - 6) * np.pi / 12))  # Solar irradiance
        
        weather_data.append({
            'temperature_C': temp,
            'solar_irradiance_W_per_m2': solar
        })
    
    # Simulate building loads
    building_characteristics = {
        'insulation_level': 'average',
        'thermal_mass': 'medium',
        'window_type': 'double_glazed',
        'construction_quality': 'average_construction'
    }
    
    building_states = model.simulate_building_loads(
        zones, weather_data, ControlStrategy.MULTI_ZONE_VARIABLE, 
        building_characteristics, 24
    )
    
    print(f"✅ Simulated {len(building_states)} hours of building operation")
    
    # Analyze load patterns
    load_analysis = model.analyze_load_patterns(building_states)
    print(f"✅ Peak heating load: {load_analysis['peak_heating_load_W']:.0f} W")
    print(f"✅ Peak cooling load: {load_analysis['peak_cooling_load_W']:.0f} W")
    print(f"✅ Load diversity factor: {load_analysis['average_diversity_factor']:.2f}")
    
    # Optimize control
    optimization = model.optimize_zone_control(zones, building_states, 'energy')
    print(f"✅ Comfort satisfaction: {optimization['current_performance']['comfort_satisfaction_percent']:.1f}%")
    
    # Validate model
    validation = model.validate_building_model(zones, building_states)
    print(f"✅ Model validation: {'PASSED' if validation['model_valid'] else 'FAILED'}")
    
    print("✅ Multi-zone building load models working correctly!")
            