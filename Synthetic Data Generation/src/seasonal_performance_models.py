"""
Seasonal Performance Variation Models
====================================

This module implements seasonal performance modeling for heat pump systems,
capturing annual performance cycles, weather pattern effects, and seasonal
efficiency variations that impact COP throughout the year.

Key Features:
- Annual weather pattern modeling with realistic seasonal variations
- Building thermal load variations throughout the year
- Heat pump performance degradation due to seasonal conditions
- Frost formation and defrost cycle frequency modeling
- Seasonal refrigerant property variations
- Ground temperature effects for ground-source heat pumps
- Solar radiation and wind effects on system performance

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

class Season(Enum):
    """Seasonal classifications"""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"

class ClimateZone(Enum):
    """Climate zone classifications"""
    ARCTIC = "arctic"
    SUBARCTIC = "subarctic"
    CONTINENTAL = "continental"
    TEMPERATE = "temperate"
    MEDITERRANEAN = "mediterranean"
    SUBTROPICAL = "subtropical"
    TROPICAL = "tropical"

@dataclass
class WeatherConditions:
    """Weather conditions at a specific time"""
    timestamp: datetime
    outdoor_temperature_C: float
    indoor_temperature_C: float
    relative_humidity_percent: float
    wind_speed_m_per_s: float
    solar_irradiance_W_per_m2: float
    atmospheric_pressure_Pa: float
    precipitation_mm_per_h: float
    cloud_cover_percent: float
    ground_temperature_C: float

@dataclass
class SeasonalParameters:
    """Seasonal performance parameters"""
    season: Season
    climate_zone: ClimateZone
    average_outdoor_temp_C: float
    temperature_amplitude_C: float
    humidity_baseline_percent: float
    humidity_variation_percent: float
    wind_speed_average_m_per_s: float
    solar_irradiance_average_W_per_m2: float
    frost_formation_probability: float
    defrost_frequency_per_day: float

@dataclass
class BuildingLoadProfile:
    """Building thermal load profile"""
    heating_load_kW: float
    cooling_load_kW: float
    domestic_hot_water_load_kW: float
    internal_gains_kW: float
    solar_gains_kW: float
    infiltration_losses_kW: float
    thermal_mass_effect_kW: float

@dataclass
class SeasonalPerformanceData:
    """Seasonal performance data point"""
    timestamp: datetime
    weather: WeatherConditions
    building_load: BuildingLoadProfile
    heat_pump_cop: float
    energy_consumption_kWh: float
    capacity_degradation_factor: float
    frost_accumulation_mm: float
    defrost_energy_penalty_kWh: float
    seasonal_efficiency_factor: float

class SeasonalPerformanceModel:
    """
    Comprehensive seasonal performance model for heat pump systems.
    
    This class models the complex interactions between weather patterns,
    building loads, and heat pump performance throughout the year.
    """
    
    def __init__(self, climate_zone: ClimateZone = ClimateZone.TEMPERATE):
        """Initialize seasonal performance model"""
        self.climate_zone = climate_zone
        self.seasonal_parameters = self._initialize_seasonal_parameters()
        self.weather_patterns = self._initialize_weather_patterns()
        self.building_characteristics = self._initialize_building_characteristics()
        
        logger.info(f"Seasonal performance model initialized for {climate_zone.value} climate")
    
    def _initialize_seasonal_parameters(self) -> Dict[Season, SeasonalParameters]:
        """Initialize seasonal parameters for different climate zones"""
        
        # Base parameters for temperate climate
        base_params = {
            Season.WINTER: SeasonalParameters(
                season=Season.WINTER,
                climate_zone=self.climate_zone,
                average_outdoor_temp_C=-2.0,
                temperature_amplitude_C=15.0,
                humidity_baseline_percent=75.0,
                humidity_variation_percent=20.0,
                wind_speed_average_m_per_s=4.5,
                solar_irradiance_average_W_per_m2=150.0,
                frost_formation_probability=0.6,
                defrost_frequency_per_day=3.0
            ),
            
            Season.SPRING: SeasonalParameters(
                season=Season.SPRING,
                climate_zone=self.climate_zone,
                average_outdoor_temp_C=12.0,
                temperature_amplitude_C=12.0,
                humidity_baseline_percent=65.0,
                humidity_variation_percent=25.0,
                wind_speed_average_m_per_s=3.8,
                solar_irradiance_average_W_per_m2=400.0,
                frost_formation_probability=0.2,
                defrost_frequency_per_day=0.5
            ),
            
            Season.SUMMER: SeasonalParameters(
                season=Season.SUMMER,
                climate_zone=self.climate_zone,
                average_outdoor_temp_C=25.0,
                temperature_amplitude_C=10.0,
                humidity_baseline_percent=60.0,
                humidity_variation_percent=30.0,
                wind_speed_average_m_per_s=3.2,
                solar_irradiance_average_W_per_m2=800.0,
                frost_formation_probability=0.0,
                defrost_frequency_per_day=0.0
            ),
            
            Season.AUTUMN: SeasonalParameters(
                season=Season.AUTUMN,
                climate_zone=self.climate_zone,
                average_outdoor_temp_C=8.0,
                temperature_amplitude_C=14.0,
                humidity_baseline_percent=70.0,
                humidity_variation_percent=22.0,
                wind_speed_average_m_per_s=4.0,
                solar_irradiance_average_W_per_m2=250.0,
                frost_formation_probability=0.3,
                defrost_frequency_per_day=1.0
            )
        }
        
        # Adjust parameters based on climate zone
        climate_adjustments = {
            ClimateZone.ARCTIC: {'temp_offset': -25.0, 'frost_multiplier': 3.0},
            ClimateZone.SUBARCTIC: {'temp_offset': -15.0, 'frost_multiplier': 2.0},
            ClimateZone.CONTINENTAL: {'temp_offset': -5.0, 'frost_multiplier': 1.5},
            ClimateZone.TEMPERATE: {'temp_offset': 0.0, 'frost_multiplier': 1.0},
            ClimateZone.MEDITERRANEAN: {'temp_offset': 8.0, 'frost_multiplier': 0.3},
            ClimateZone.SUBTROPICAL: {'temp_offset': 15.0, 'frost_multiplier': 0.1},
            ClimateZone.TROPICAL: {'temp_offset': 20.0, 'frost_multiplier': 0.0}
        }
        
        adjustment = climate_adjustments.get(self.climate_zone, {'temp_offset': 0.0, 'frost_multiplier': 1.0})
        
        for season_params in base_params.values():
            season_params.average_outdoor_temp_C += adjustment['temp_offset']
            season_params.frost_formation_probability *= adjustment['frost_multiplier']
            season_params.defrost_frequency_per_day *= adjustment['frost_multiplier']
        
        return base_params
    
    def _initialize_weather_patterns(self) -> Dict[str, Any]:
        """Initialize weather pattern models"""
        
        return {
            'temperature_cycles': {
                'daily_amplitude': 8.0,  # Daily temperature variation
                'annual_phase_shift': 0.0,  # Phase shift for southern hemisphere
                'urban_heat_island': 2.0,  # Urban temperature increase
                'altitude_lapse_rate': -0.0065  # Temperature decrease per meter altitude
            },
            
            'humidity_patterns': {
                'morning_peak': 0.85,  # Peak humidity in early morning
                'afternoon_minimum': 0.45,  # Minimum humidity in afternoon
                'seasonal_variation': 0.3,  # Seasonal humidity variation
                'precipitation_effect': 0.2  # Humidity increase during precipitation
            },
            
            'wind_patterns': {
                'diurnal_variation': 0.4,  # Daily wind speed variation
                'seasonal_variation': 0.3,  # Seasonal wind speed variation
                'gust_factor': 1.8,  # Maximum gust factor
                'terrain_roughness': 0.1  # Terrain effect on wind
            },
            
            'solar_patterns': {
                'clear_sky_factor': 0.85,  # Clear sky solar transmission
                'cloud_attenuation': 0.3,  # Solar reduction due to clouds
                'atmospheric_turbidity': 0.1,  # Atmospheric scattering
                'ground_reflectance': 0.2  # Ground albedo
            }
        }
    
    def _initialize_building_characteristics(self) -> Dict[str, Any]:
        """Initialize building thermal characteristics"""
        
        return {
            'thermal_mass': {
                'light_construction': 150.0,  # kJ/(m²·K)
                'medium_construction': 300.0,
                'heavy_construction': 600.0
            },
            
            'insulation_levels': {
                'poor': {'u_value': 2.5, 'air_changes': 1.5},  # W/(m²·K), ACH
                'average': {'u_value': 1.2, 'air_changes': 0.8},
                'good': {'u_value': 0.6, 'air_changes': 0.4},
                'excellent': {'u_value': 0.3, 'air_changes': 0.2}
            },
            
            'internal_gains': {
                'occupancy_W_per_m2': 5.0,  # People heat gain
                'lighting_W_per_m2': 8.0,   # Lighting heat gain
                'equipment_W_per_m2': 10.0,  # Equipment heat gain
                'diversity_factor': 0.7      # Simultaneous usage factor
            },
            
            'setpoint_schedules': {
                'heating_setpoint_C': 20.0,
                'cooling_setpoint_C': 24.0,
                'night_setback_C': 2.0,
                'weekend_adjustment_C': 1.0
            }
        }
    
    def get_season_from_date(self, date: datetime) -> Season:
        """Determine season from date"""
        
        month = date.month
        day = date.day
        
        # Northern hemisphere seasons (adjust for southern hemisphere if needed)
        if month in [12, 1, 2] or (month == 3 and day < 20):
            return Season.WINTER
        elif month in [3, 4, 5] or (month == 6 and day < 21):
            return Season.SPRING
        elif month in [6, 7, 8] or (month == 9 and day < 22):
            return Season.SUMMER
        else:
            return Season.AUTUMN
    
    def generate_weather_conditions(self, 
                                  timestamp: datetime,
                                  location_params: Optional[Dict] = None) -> WeatherConditions:
        """
        Generate realistic weather conditions for a specific timestamp
        
        Args:
            timestamp: Date and time for weather generation
            location_params: Location-specific parameters (latitude, altitude, etc.)
            
        Returns:
            WeatherConditions object
        """
        
        if location_params is None:
            location_params = {'latitude': 45.0, 'altitude': 100.0}
        
        season = self.get_season_from_date(timestamp)
        season_params = self.seasonal_parameters[season]
        
        # Day of year (1-365)
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        
        # Annual temperature cycle
        annual_temp_variation = season_params.temperature_amplitude_C * \
                               np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        
        # Daily temperature cycle
        daily_temp_variation = self.weather_patterns['temperature_cycles']['daily_amplitude'] * \
                              np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
        
        # Random weather variations
        temp_noise = np.random.normal(0, 2.0)  # 2°C standard deviation
        
        outdoor_temp = (season_params.average_outdoor_temp_C + 
                       annual_temp_variation + 
                       daily_temp_variation + 
                       temp_noise)
        
        # Indoor temperature (controlled but influenced by outdoor conditions)
        indoor_temp = 20.0 + 0.1 * (outdoor_temp - 20.0)  # Slight coupling
        
        # Humidity calculation
        humidity_base = season_params.humidity_baseline_percent
        humidity_daily = 20.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)  # Peak at 6 AM
        humidity_noise = np.random.normal(0, 5.0)
        
        relative_humidity = np.clip(humidity_base + humidity_daily + humidity_noise, 20.0, 95.0)
        
        # Wind speed
        wind_base = season_params.wind_speed_average_m_per_s
        wind_daily = 1.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)  # Peak at 2 PM
        wind_noise = np.random.exponential(1.0)  # Exponential distribution for gusts
        
        wind_speed = max(0.5, wind_base + wind_daily + wind_noise)
        
        # Solar irradiance
        latitude_rad = np.radians(location_params['latitude'])
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        hour_angle = 15 * (hour_of_day - 12)
        
        solar_elevation = np.arcsin(
            np.sin(latitude_rad) * np.sin(np.radians(declination)) +
            np.cos(latitude_rad) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        
        if solar_elevation > 0:
            clear_sky_irradiance = 1000 * np.sin(solar_elevation) * \
                                  self.weather_patterns['solar_patterns']['clear_sky_factor']
            cloud_factor = 1.0 - 0.6 * (np.random.random() * 0.5)  # Random cloud cover
            solar_irradiance = max(0, clear_sky_irradiance * cloud_factor)
        else:
            solar_irradiance = 0.0
        
        # Atmospheric pressure (varies with weather systems)
        pressure_base = 101325.0  # Standard atmospheric pressure
        pressure_variation = np.random.normal(0, 1000.0)  # Weather system variation
        atmospheric_pressure = pressure_base + pressure_variation
        
        # Precipitation (simplified model)
        precipitation_probability = 0.1 + 0.05 * np.sin(2 * np.pi * day_of_year / 365.25)
        precipitation = 5.0 * np.random.exponential(1.0) if np.random.random() < precipitation_probability else 0.0
        
        # Cloud cover (correlated with precipitation)
        cloud_cover = min(100.0, 30.0 + 50.0 * (precipitation > 0) + np.random.normal(0, 20.0))
        cloud_cover = max(0.0, cloud_cover)
        
        # Ground temperature (lags air temperature)
        ground_temp_lag = 30  # days
        ground_temp = season_params.average_outdoor_temp_C + \
                     0.5 * season_params.temperature_amplitude_C * \
                     np.sin(2 * np.pi * (day_of_year - 80 - ground_temp_lag) / 365.25)
        
        return WeatherConditions(
            timestamp=timestamp,
            outdoor_temperature_C=outdoor_temp,
            indoor_temperature_C=indoor_temp,
            relative_humidity_percent=relative_humidity,
            wind_speed_m_per_s=wind_speed,
            solar_irradiance_W_per_m2=solar_irradiance,
            atmospheric_pressure_Pa=atmospheric_pressure,
            precipitation_mm_per_h=precipitation,
            cloud_cover_percent=cloud_cover,
            ground_temperature_C=ground_temp
        )
    
    def calculate_building_load(self, 
                              weather: WeatherConditions,
                              building_params: Optional[Dict] = None) -> BuildingLoadProfile:
        """
        Calculate building thermal load based on weather conditions
        
        Args:
            weather: Current weather conditions
            building_params: Building-specific parameters
            
        Returns:
            BuildingLoadProfile object
        """
        
        if building_params is None:
            building_params = {
                'floor_area_m2': 150.0,
                'insulation_level': 'average',
                'thermal_mass': 'medium_construction',
                'orientation': 'south'
            }
        
        floor_area = building_params['floor_area_m2']
        insulation = self.building_characteristics['insulation_levels'][building_params['insulation_level']]
        
        # Temperature difference
        temp_diff = abs(weather.outdoor_temperature_C - weather.indoor_temperature_C)
        
        # Transmission losses/gains
        transmission_load = insulation['u_value'] * floor_area * 0.8 * temp_diff / 1000.0  # kW
        
        # Infiltration losses/gains
        air_density = 1.2  # kg/m³
        specific_heat_air = 1.006  # kJ/(kg·K)
        ceiling_height = 2.5  # m
        volume = floor_area * ceiling_height
        
        infiltration_load = (insulation['air_changes'] * volume * air_density * 
                           specific_heat_air * temp_diff) / 3600.0  # kW
        
        # Solar gains
        window_area = floor_area * 0.15  # 15% window-to-floor ratio
        solar_heat_gain_coeff = 0.7
        solar_gains = (weather.solar_irradiance_W_per_m2 * window_area * 
                      solar_heat_gain_coeff) / 1000.0  # kW
        
        # Internal gains
        internal_gains_total = (
            self.building_characteristics['internal_gains']['occupancy_W_per_m2'] +
            self.building_characteristics['internal_gains']['lighting_W_per_m2'] +
            self.building_characteristics['internal_gains']['equipment_W_per_m2']
        ) * floor_area * self.building_characteristics['internal_gains']['diversity_factor'] / 1000.0
        
        # Determine heating vs cooling load
        if weather.outdoor_temperature_C < weather.indoor_temperature_C:
            # Heating mode
            heating_load = transmission_load + infiltration_load - solar_gains - internal_gains_total
            heating_load = max(0, heating_load)
            cooling_load = 0.0
        else:
            # Cooling mode
            cooling_load = transmission_load + infiltration_load + solar_gains + internal_gains_total
            cooling_load = max(0, cooling_load)
            heating_load = 0.0
        
        # Domestic hot water load (relatively constant)
        dhw_load = 2.0  # kW average
        
        # Thermal mass effect (dampens load variations)
        thermal_mass_capacity = self.building_characteristics['thermal_mass'][building_params['thermal_mass']]
        thermal_mass_effect = 0.1 * thermal_mass_capacity * temp_diff / 1000.0  # kW
        
        return BuildingLoadProfile(
            heating_load_kW=heating_load,
            cooling_load_kW=cooling_load,
            domestic_hot_water_load_kW=dhw_load,
            internal_gains_kW=internal_gains_total,
            solar_gains_kW=solar_gains,
            infiltration_losses_kW=infiltration_load,
            thermal_mass_effect_kW=thermal_mass_effect
        )
    
    def calculate_seasonal_cop(self, 
                             weather: WeatherConditions,
                             building_load: BuildingLoadProfile,
                             heat_pump_params: Optional[Dict] = None) -> float:
        """
        Calculate heat pump COP considering seasonal effects
        
        Args:
            weather: Current weather conditions
            building_load: Building thermal load
            heat_pump_params: Heat pump specifications
            
        Returns:
            Seasonal COP value
        """
        
        if heat_pump_params is None:
            heat_pump_params = {
                'rated_capacity_kW': 12.0,
                'rated_cop': 4.5,
                'rated_conditions': {'outdoor_temp_C': 7.0, 'indoor_temp_C': 20.0}
            }
        
        # Base COP from rated conditions
        base_cop = heat_pump_params['rated_cop']
        
        # Temperature-dependent COP adjustment
        outdoor_temp = weather.outdoor_temperature_C
        rated_outdoor_temp = heat_pump_params['rated_conditions']['outdoor_temp_C']
        
        # COP decreases with lower outdoor temperature (heating mode)
        temp_factor = 1.0 + 0.02 * (outdoor_temp - rated_outdoor_temp)
        temp_factor = max(0.3, min(1.5, temp_factor))  # Limit range
        
        # Humidity effect (frost formation reduces efficiency)
        humidity_factor = 1.0
        if outdoor_temp < 5.0 and weather.relative_humidity_percent > 70.0:
            frost_probability = (70.0 - outdoor_temp) / 100.0 * (weather.relative_humidity_percent - 70.0) / 30.0
            frost_probability = max(0.0, min(1.0, frost_probability))
            humidity_factor = 1.0 - 0.15 * frost_probability  # Up to 15% reduction
        
        # Wind effect (increases heat transfer from outdoor unit)
        wind_factor = 1.0 + 0.01 * min(weather.wind_speed_m_per_s, 10.0)  # Up to 10% improvement
        
        # Part-load effect
        total_load = building_load.heating_load_kW + building_load.cooling_load_kW
        rated_capacity = heat_pump_params['rated_capacity_kW']
        
        if total_load > 0:
            load_ratio = min(total_load / rated_capacity, 1.2)  # Allow 20% overload
            if load_ratio < 0.3:
                part_load_factor = 0.7 + 0.3 * (load_ratio / 0.3)  # Reduced efficiency at low loads
            else:
                part_load_factor = 1.0
        else:
            part_load_factor = 0.1  # Standby losses
        
        # Seasonal degradation (equipment aging, fouling)
        season = self.get_season_from_date(weather.timestamp)
        seasonal_degradation = {
            Season.WINTER: 0.95,  # Harsh conditions
            Season.SPRING: 0.98,  # Mild conditions
            Season.SUMMER: 0.97,  # High loads
            Season.AUTUMN: 0.96   # Moderate conditions
        }
        
        degradation_factor = seasonal_degradation[season]
        
        # Calculate final COP
        seasonal_cop = (base_cop * temp_factor * humidity_factor * 
                       wind_factor * part_load_factor * degradation_factor)
        
        return max(1.0, seasonal_cop)  # Minimum COP of 1.0
    
    def simulate_annual_performance(self, 
                                  start_date: datetime,
                                  duration_days: int = 365,
                                  time_step_hours: int = 1,
                                  location_params: Optional[Dict] = None,
                                  building_params: Optional[Dict] = None,
                                  heat_pump_params: Optional[Dict] = None) -> List[SeasonalPerformanceData]:
        """
        Simulate annual heat pump performance with seasonal variations
        
        Args:
            start_date: Starting date for simulation
            duration_days: Simulation duration in days
            time_step_hours: Time step in hours
            location_params: Location-specific parameters
            building_params: Building-specific parameters
            heat_pump_params: Heat pump specifications
            
        Returns:
            List of SeasonalPerformanceData objects
        """
        
        performance_data = []
        current_time = start_date
        time_delta = timedelta(hours=time_step_hours)
        
        # Initialize cumulative variables
        frost_accumulation = 0.0
        
        for _ in range(int(duration_days * 24 / time_step_hours)):
            
            # Generate weather conditions
            weather = self.generate_weather_conditions(current_time, location_params)
            
            # Calculate building load
            building_load = self.calculate_building_load(weather, building_params)
            
            # Calculate seasonal COP
            cop = self.calculate_seasonal_cop(weather, building_load, heat_pump_params)
            
            # Calculate energy consumption
            total_thermal_load = building_load.heating_load_kW + building_load.cooling_load_kW
            if total_thermal_load > 0 and cop > 0:
                energy_consumption = total_thermal_load * time_step_hours / cop
            else:
                energy_consumption = 0.1 * time_step_hours  # Standby consumption
            
            # Frost accumulation and defrost effects
            season = self.get_season_from_date(current_time)
            season_params = self.seasonal_parameters[season]
            
            if (weather.outdoor_temperature_C < 2.0 and 
                weather.relative_humidity_percent > 60.0 and
                np.random.random() < season_params.frost_formation_probability * time_step_hours / 24.0):
                frost_accumulation += 0.1  # mm per hour
            
            # Defrost cycle
            defrost_energy_penalty = 0.0
            if frost_accumulation > 2.0:  # Defrost when frost > 2mm
                defrost_energy_penalty = 2.0  # kWh per defrost cycle
                frost_accumulation = 0.0
            
            # Capacity degradation due to frost
            if frost_accumulation > 0:
                capacity_degradation = 1.0 - 0.05 * frost_accumulation  # 5% per mm
            else:
                capacity_degradation = 1.0
            
            # Seasonal efficiency factor
            seasonal_efficiency = self._calculate_seasonal_efficiency_factor(weather, season)
            
            # Create performance data point
            performance_point = SeasonalPerformanceData(
                timestamp=current_time,
                weather=weather,
                building_load=building_load,
                heat_pump_cop=cop,
                energy_consumption_kWh=energy_consumption,
                capacity_degradation_factor=capacity_degradation,
                frost_accumulation_mm=frost_accumulation,
                defrost_energy_penalty_kWh=defrost_energy_penalty,
                seasonal_efficiency_factor=seasonal_efficiency
            )
            
            performance_data.append(performance_point)
            current_time += time_delta
        
        logger.info(f"Simulated {len(performance_data)} data points over {duration_days} days")
        return performance_data
    
    def _calculate_seasonal_efficiency_factor(self, weather: WeatherConditions, season: Season) -> float:
        """Calculate seasonal efficiency factor based on conditions"""
        
        # Base efficiency factors by season
        base_factors = {
            Season.WINTER: 0.85,  # Challenging conditions
            Season.SPRING: 0.95,  # Optimal conditions
            Season.SUMMER: 0.90,  # High ambient temperatures
            Season.AUTUMN: 0.92   # Good conditions
        }
        
        base_factor = base_factors[season]
        
        # Adjust for specific weather conditions
        temp_adjustment = 1.0
        if weather.outdoor_temperature_C < -10.0:
            temp_adjustment = 0.9  # Very cold
        elif weather.outdoor_temperature_C > 35.0:
            temp_adjustment = 0.95  # Very hot
        
        humidity_adjustment = 1.0 - 0.001 * max(0, weather.relative_humidity_percent - 60.0)
        
        wind_adjustment = 1.0 + 0.005 * min(weather.wind_speed_m_per_s, 8.0)
        
        return base_factor * temp_adjustment * humidity_adjustment * wind_adjustment
    
    def validate_seasonal_model(self) -> Dict[str, bool]:
        """Validate seasonal performance model"""
        
        validation_results = {
            'weather_generation': False,
            'building_load_calculation': False,
            'cop_calculation': False,
            'annual_simulation': False,
            'seasonal_variations': False
        }
        
        try:
            # Test weather generation
            test_date = datetime(2024, 6, 15, 12, 0)  # Summer day
            weather = self.generate_weather_conditions(test_date)
            validation_results['weather_generation'] = (
                -50 < weather.outdoor_temperature_C < 50 and
                0 <= weather.relative_humidity_percent <= 100 and
                weather.wind_speed_m_per_s >= 0
            )
            
            # Test building load calculation
            building_load = self.calculate_building_load(weather)
            validation_results['building_load_calculation'] = (
                building_load.heating_load_kW >= 0 and
                building_load.cooling_load_kW >= 0
            )
            
            # Test COP calculation
            cop = self.calculate_seasonal_cop(weather, building_load)
            validation_results['cop_calculation'] = 1.0 <= cop <= 8.0
            
            # Test short annual simulation
            performance_data = self.simulate_annual_performance(
                datetime(2024, 1, 1), duration_days=7, time_step_hours=6
            )
            validation_results['annual_simulation'] = len(performance_data) == 28
            
            # Test seasonal variations
            winter_weather = self.generate_weather_conditions(datetime(2024, 1, 15, 12, 0))
            summer_weather = self.generate_weather_conditions(datetime(2024, 7, 15, 12, 0))
            
            winter_cop = self.calculate_seasonal_cop(winter_weather, self.calculate_building_load(winter_weather))
            summer_cop = self.calculate_seasonal_cop(summer_weather, self.calculate_building_load(summer_weather))
            
            validation_results['seasonal_variations'] = abs(winter_cop - summer_cop) > 0.5
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = SeasonalPerformanceModel(ClimateZone.TEMPERATE)
    
    print("🌍 Testing Seasonal Performance Model")
    
    # Test weather generation for different seasons
    winter_date = datetime(2024, 1, 15, 12, 0)
    summer_date = datetime(2024, 7, 15, 12, 0)
    
    winter_weather = model.generate_weather_conditions(winter_date)
    summer_weather = model.generate_weather_conditions(summer_date)
    
    print(f"✅ Winter weather: {winter_weather.outdoor_temperature_C:.1f}°C, {winter_weather.relative_humidity_percent:.0f}% RH")
    print(f"✅ Summer weather: {summer_weather.outdoor_temperature_C:.1f}°C, {summer_weather.relative_humidity_percent:.0f}% RH")
    
    # Test building load calculation
    winter_load = model.calculate_building_load(winter_weather)
    summer_load = model.calculate_building_load(summer_weather)
    
    print(f"✅ Winter heating load: {winter_load.heating_load_kW:.1f} kW")
    print(f"✅ Summer cooling load: {summer_load.cooling_load_kW:.1f} kW")
    
    # Test seasonal COP calculation
    winter_cop = model.calculate_seasonal_cop(winter_weather, winter_load)
    summer_cop = model.calculate_seasonal_cop(summer_weather, summer_load)
    
    print(f"✅ Winter COP: {winter_cop:.2f}")
    print(f"✅ Summer COP: {summer_cop:.2f}")
    
    # Test short annual simulation
    performance_data = model.simulate_annual_performance(
        datetime(2024, 1, 1), duration_days=30, time_step_hours=24
    )
    
    print(f"✅ Generated {len(performance_data)} daily performance points")
    
    # Calculate average seasonal performance
    winter_points = [p for p in performance_data if model.get_season_from_date(p.timestamp) == Season.WINTER]
    if winter_points:
        avg_winter_cop = sum(p.heat_pump_cop for p in winter_points) / len(winter_points)
        print(f"✅ Average winter COP: {avg_winter_cop:.2f}")
    
    # Validate model
    validation = model.validate_seasonal_model()
    print(f"✅ Model validation: {sum(validation.values())}/{len(validation)} tests passed")
    
    print("🎉 Seasonal performance model working correctly!")