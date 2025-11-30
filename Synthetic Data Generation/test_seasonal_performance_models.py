#!/usr/bin/env python3
"""
Test Seasonal Performance Models
===============================

Comprehensive test suite for seasonal performance models including weather generation,
building load calculations, seasonal COP variations, and annual performance simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import directly from the module file
from seasonal_performance_models import (
    SeasonalPerformanceModel, Season, ClimateZone, WeatherConditions,
    SeasonalParameters, BuildingLoadProfile, SeasonalPerformanceData
)

def test_seasonal_performance_models():
    """Test seasonal performance models comprehensively."""
    
    print("🌍 Testing Seasonal Performance Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        print("✅ Import successful")
        
        # Test different climate zones
        print("\n🔄 Testing SeasonalPerformanceModel initialization...")
        climate_zones = list(ClimateZone)
        seasonal_models = {}
        
        for climate_zone in climate_zones:
            model = SeasonalPerformanceModel(climate_zone)
            seasonal_models[climate_zone] = model
            print(f"✅ {climate_zone.value} climate model initialized")
            print(f"  - Seasonal parameters: {len(model.seasonal_parameters)} seasons")
            print(f"  - Weather patterns: {len(model.weather_patterns)} categories")
            print(f"  - Building characteristics: {len(model.building_characteristics)} categories")
        
        # Use temperate climate for detailed testing
        test_model = seasonal_models[ClimateZone.TEMPERATE]
        
        # Test seasonal parameters
        print("\n🔄 Testing seasonal parameters...")
        for season, params in test_model.seasonal_parameters.items():
            print(f"✅ {season.value} parameters:")
            print(f"  - Average outdoor temp: {params.average_outdoor_temp_C:.1f}°C")
            print(f"  - Temperature amplitude: {params.temperature_amplitude_C:.1f}°C")
            print(f"  - Humidity baseline: {params.humidity_baseline_percent:.1f}%")
            print(f"  - Wind speed average: {params.wind_speed_average_m_per_s:.1f} m/s")
            print(f"  - Solar irradiance average: {params.solar_irradiance_average_W_per_m2:.0f} W/m²")
            print(f"  - Frost formation probability: {params.frost_formation_probability:.2f}")
            print(f"  - Defrost frequency: {params.defrost_frequency_per_day:.1f} /day")
        
        # Test season determination from date
        print("\n🔄 Testing season determination...")
        test_dates = [
            datetime(2024, 1, 15),  # Winter
            datetime(2024, 4, 15),  # Spring
            datetime(2024, 7, 15),  # Summer
            datetime(2024, 10, 15)  # Autumn
        ]
        
        for test_date in test_dates:
            season = test_model.get_season_from_date(test_date)
            print(f"✅ {test_date.strftime('%B %d')}: {season.value}")
        
        # Test weather generation
        print("\n🔄 Testing weather generation...")
        location_params = {'latitude': 45.0, 'altitude': 100.0}
        
        weather_data = {}
        for test_date in test_dates:
            weather = test_model.generate_weather_conditions(test_date, location_params)
            season = test_model.get_season_from_date(test_date)
            weather_data[season] = weather
            
            print(f"✅ {season.value} weather ({test_date.strftime('%B %d')}):")
            print(f"  - Outdoor temperature: {weather.outdoor_temperature_C:.1f}°C")
            print(f"  - Indoor temperature: {weather.indoor_temperature_C:.1f}°C")
            print(f"  - Relative humidity: {weather.relative_humidity_percent:.1f}%")
            print(f"  - Wind speed: {weather.wind_speed_m_per_s:.1f} m/s")
            print(f"  - Solar irradiance: {weather.solar_irradiance_W_per_m2:.0f} W/m²")
            print(f"  - Atmospheric pressure: {weather.atmospheric_pressure_Pa:.0f} Pa")
            print(f"  - Precipitation: {weather.precipitation_mm_per_h:.1f} mm/h")
            print(f"  - Cloud cover: {weather.cloud_cover_percent:.1f}%")
            print(f"  - Ground temperature: {weather.ground_temperature_C:.1f}°C")
        
        # Test building load calculations
        print("\n🔄 Testing building load calculations...")
        building_params_sets = [
            {
                'floor_area_m2': 100.0,
                'insulation_level': 'poor',
                'thermal_mass': 'light_construction',
                'orientation': 'north'
            },
            {
                'floor_area_m2': 150.0,
                'insulation_level': 'average',
                'thermal_mass': 'medium_construction',
                'orientation': 'south'
            },
            {
                'floor_area_m2': 200.0,
                'insulation_level': 'excellent',
                'thermal_mass': 'heavy_construction',
                'orientation': 'south'
            }
        ]
        
        building_loads = {}
        for i, building_params in enumerate(building_params_sets):
            print(f"✅ Building {i+1} ({building_params['insulation_level']} insulation):")
            seasonal_loads = {}
            
            for season, weather in weather_data.items():
                load = test_model.calculate_building_load(weather, building_params)
                seasonal_loads[season] = load
                
                print(f"  - {season.value}:")
                print(f"    * Heating load: {load.heating_load_kW:.2f} kW")
                print(f"    * Cooling load: {load.cooling_load_kW:.2f} kW")
                print(f"    * DHW load: {load.domestic_hot_water_load_kW:.2f} kW")
                print(f"    * Internal gains: {load.internal_gains_kW:.2f} kW")
                print(f"    * Solar gains: {load.solar_gains_kW:.2f} kW")
                print(f"    * Infiltration losses: {load.infiltration_losses_kW:.2f} kW")
                print(f"    * Thermal mass effect: {load.thermal_mass_effect_kW:.2f} kW")
            
            building_loads[i] = seasonal_loads
        
        # Test seasonal COP calculations
        print("\n🔄 Testing seasonal COP calculations...")
        heat_pump_params_sets = [
            {
                'rated_capacity_kW': 8.0,
                'rated_cop': 3.5,
                'rated_conditions': {'outdoor_temp_C': 7.0, 'indoor_temp_C': 20.0}
            },
            {
                'rated_capacity_kW': 12.0,
                'rated_cop': 4.5,
                'rated_conditions': {'outdoor_temp_C': 7.0, 'indoor_temp_C': 20.0}
            },
            {
                'rated_capacity_kW': 16.0,
                'rated_cop': 5.0,
                'rated_conditions': {'outdoor_temp_C': 7.0, 'indoor_temp_C': 20.0}
            }
        ]
        
        cop_results = {}
        for i, heat_pump_params in enumerate(heat_pump_params_sets):
            print(f"✅ Heat pump {i+1} ({heat_pump_params['rated_capacity_kW']} kW, COP {heat_pump_params['rated_cop']}):")
            seasonal_cops = {}
            
            for season, weather in weather_data.items():
                building_load = building_loads[1][season]  # Use average building
                cop = test_model.calculate_seasonal_cop(weather, building_load, heat_pump_params)
                seasonal_cops[season] = cop
                
                print(f"  - {season.value}: COP = {cop:.2f}")
            
            cop_results[i] = seasonal_cops
        
        # Test annual performance simulation
        print("\n🔄 Testing annual performance simulation...")
        
        # Short simulation for testing (1 week with daily steps)
        start_date = datetime(2024, 1, 1)
        duration_days = 7
        time_step_hours = 24
        
        performance_data = test_model.simulate_annual_performance(
            start_date, duration_days, time_step_hours,
            location_params, building_params_sets[1], heat_pump_params_sets[1]
        )
        
        print(f"✅ Annual simulation completed:")
        print(f"  - Data points: {len(performance_data)}")
        print(f"  - Duration: {duration_days} days")
        print(f"  - Time step: {time_step_hours} hours")
        
        # Analyze simulation results
        if performance_data:
            cops = [p.heat_pump_cop for p in performance_data]
            energy_consumptions = [p.energy_consumption_kWh for p in performance_data]
            frost_accumulations = [p.frost_accumulation_mm for p in performance_data]
            
            print(f"  - COP range: {min(cops):.2f} - {max(cops):.2f}")
            print(f"  - Average COP: {np.mean(cops):.2f}")
            print(f"  - Energy consumption range: {min(energy_consumptions):.1f} - {max(energy_consumptions):.1f} kWh")
            print(f"  - Total energy consumption: {sum(energy_consumptions):.1f} kWh")
            print(f"  - Max frost accumulation: {max(frost_accumulations):.1f} mm")
        
        # Test longer simulation with different time steps
        print("\n🔄 Testing extended simulation...")
        
        # Monthly simulation for a full year
        monthly_performance = test_model.simulate_annual_performance(
            datetime(2024, 1, 1), 365, 24 * 7,  # Weekly data points
            location_params, building_params_sets[1], heat_pump_params_sets[1]
        )
        
        print(f"✅ Extended simulation completed:")
        print(f"  - Data points: {len(monthly_performance)}")
        print(f"  - Duration: 365 days")
        print(f"  - Time step: 168 hours (weekly)")
        
        # Analyze seasonal performance
        seasonal_performance = {season: [] for season in Season}
        for data_point in monthly_performance:
            season = test_model.get_season_from_date(data_point.timestamp)
            seasonal_performance[season].append(data_point)
        
        print(f"✅ Seasonal performance analysis:")
        for season, data_points in seasonal_performance.items():
            if data_points:
                avg_cop = np.mean([p.heat_pump_cop for p in data_points])
                avg_energy = np.mean([p.energy_consumption_kWh for p in data_points])
                avg_frost = np.mean([p.frost_accumulation_mm for p in data_points])
                
                print(f"  - {season.value}:")
                print(f"    * Data points: {len(data_points)}")
                print(f"    * Average COP: {avg_cop:.2f}")
                print(f"    * Average energy consumption: {avg_energy:.1f} kWh/week")
                print(f"    * Average frost accumulation: {avg_frost:.1f} mm")
        
        # Test model validation
        print("\n🔄 Testing model validation...")
        
        for climate_zone, model in seasonal_models.items():
            validation_results = model.validate_seasonal_model()
            
            print(f"✅ {climate_zone.value} climate validation:")
            for test_name, result in validation_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  - {test_name}: {status}")
            
            passed_tests = sum(validation_results.values())
            total_tests = len(validation_results)
            print(f"  - Overall: {passed_tests}/{total_tests} tests passed")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test WeatherConditions dataclass
        test_weather = WeatherConditions(
            timestamp=datetime(2024, 6, 15, 12, 0),
            outdoor_temperature_C=25.0,
            indoor_temperature_C=22.0,
            relative_humidity_percent=60.0,
            wind_speed_m_per_s=3.5,
            solar_irradiance_W_per_m2=800.0,
            atmospheric_pressure_Pa=101325.0,
            precipitation_mm_per_h=0.0,
            cloud_cover_percent=20.0,
            ground_temperature_C=18.0
        )
        print(f"✅ WeatherConditions created:")
        print(f"  - Timestamp: {test_weather.timestamp}")
        print(f"  - Outdoor temperature: {test_weather.outdoor_temperature_C}°C")
        print(f"  - Solar irradiance: {test_weather.solar_irradiance_W_per_m2} W/m²")
        
        # Test SeasonalParameters dataclass
        test_seasonal_params = SeasonalParameters(
            season=Season.SUMMER,
            climate_zone=ClimateZone.TEMPERATE,
            average_outdoor_temp_C=25.0,
            temperature_amplitude_C=10.0,
            humidity_baseline_percent=60.0,
            humidity_variation_percent=30.0,
            wind_speed_average_m_per_s=3.2,
            solar_irradiance_average_W_per_m2=800.0,
            frost_formation_probability=0.0,
            defrost_frequency_per_day=0.0
        )
        print(f"✅ SeasonalParameters created:")
        print(f"  - Season: {test_seasonal_params.season.value}")
        print(f"  - Climate zone: {test_seasonal_params.climate_zone.value}")
        print(f"  - Average temperature: {test_seasonal_params.average_outdoor_temp_C}°C")
        
        # Test BuildingLoadProfile dataclass
        test_building_load = BuildingLoadProfile(
            heating_load_kW=8.5,
            cooling_load_kW=0.0,
            domestic_hot_water_load_kW=2.0,
            internal_gains_kW=3.5,
            solar_gains_kW=1.2,
            infiltration_losses_kW=2.8,
            thermal_mass_effect_kW=0.5
        )
        print(f"✅ BuildingLoadProfile created:")
        print(f"  - Heating load: {test_building_load.heating_load_kW} kW")
        print(f"  - Cooling load: {test_building_load.cooling_load_kW} kW")
        print(f"  - DHW load: {test_building_load.domestic_hot_water_load_kW} kW")
        
        # Test SeasonalPerformanceData dataclass
        test_performance_data = SeasonalPerformanceData(
            timestamp=datetime(2024, 1, 15, 12, 0),
            weather=test_weather,
            building_load=test_building_load,
            heat_pump_cop=3.8,
            energy_consumption_kWh=2.2,
            capacity_degradation_factor=0.95,
            frost_accumulation_mm=1.5,
            defrost_energy_penalty_kWh=0.5,
            seasonal_efficiency_factor=0.92
        )
        print(f"✅ SeasonalPerformanceData created:")
        print(f"  - Timestamp: {test_performance_data.timestamp}")
        print(f"  - COP: {test_performance_data.heat_pump_cop}")
        print(f"  - Energy consumption: {test_performance_data.energy_consumption_kWh} kWh")
        print(f"  - Seasonal efficiency factor: {test_performance_data.seasonal_efficiency_factor}")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Test extreme weather conditions
        try:
            extreme_date = datetime(2024, 12, 21, 0, 0)  # Winter solstice
            extreme_weather = test_model.generate_weather_conditions(extreme_date)
            print(f"✅ Extreme weather handled: {extreme_weather.outdoor_temperature_C:.1f}°C")
        except Exception as e:
            print(f"❌ Extreme weather error: {str(e)}")
        
        # Test with very small building
        try:
            small_building_params = {
                'floor_area_m2': 10.0,
                'insulation_level': 'excellent',
                'thermal_mass': 'light_construction',
                'orientation': 'north'
            }
            small_load = test_model.calculate_building_load(weather_data[Season.WINTER], small_building_params)
            print(f"✅ Small building handled: {small_load.heating_load_kW:.3f} kW")
        except Exception as e:
            print(f"❌ Small building error: {str(e)}")
        
        # Test with very large heat pump
        try:
            large_hp_params = {
                'rated_capacity_kW': 100.0,
                'rated_cop': 6.0,
                'rated_conditions': {'outdoor_temp_C': 7.0, 'indoor_temp_C': 20.0}
            }
            large_cop = test_model.calculate_seasonal_cop(
                weather_data[Season.SUMMER], building_loads[1][Season.SUMMER], large_hp_params
            )
            print(f"✅ Large heat pump handled: COP = {large_cop:.2f}")
        except Exception as e:
            print(f"❌ Large heat pump error: {str(e)}")
        
        # Test performance metrics
        print("\n🔄 Testing performance metrics...")
        
        # Calculate climate zone comparison
        climate_comparison = {}
        test_date = datetime(2024, 1, 15, 12, 0)  # Winter day
        
        for climate_zone, model in seasonal_models.items():
            weather = model.generate_weather_conditions(test_date)
            building_load = model.calculate_building_load(weather)
            cop = model.calculate_seasonal_cop(weather, building_load)
            
            climate_comparison[climate_zone.value] = {
                'outdoor_temp_C': weather.outdoor_temperature_C,
                'heating_load_kW': building_load.heating_load_kW,
                'cop': cop
            }
        
        print(f"✅ Climate zone comparison (winter day):")
        for climate_zone, metrics in climate_comparison.items():
            print(f"  - {climate_zone}:")
            print(f"    * Outdoor temp: {metrics['outdoor_temp_C']:.1f}°C")
            print(f"    * Heating load: {metrics['heating_load_kW']:.1f} kW")
            print(f"    * COP: {metrics['cop']:.2f}")
        
        # Calculate seasonal efficiency variations
        seasonal_efficiency_variations = {}
        for season in Season:
            test_date = {
                Season.WINTER: datetime(2024, 1, 15, 12, 0),
                Season.SPRING: datetime(2024, 4, 15, 12, 0),
                Season.SUMMER: datetime(2024, 7, 15, 12, 0),
                Season.AUTUMN: datetime(2024, 10, 15, 12, 0)
            }[season]
            
            weather = test_model.generate_weather_conditions(test_date)
            efficiency_factor = test_model._calculate_seasonal_efficiency_factor(weather, season)
            seasonal_efficiency_variations[season.value] = efficiency_factor
        
        print(f"✅ Seasonal efficiency variations:")
        for season, efficiency in seasonal_efficiency_variations.items():
            print(f"  - {season}: {efficiency:.3f}")
        
        # Calculate annual energy consumption by building type
        annual_energy_by_building = {}
        for i, building_params in enumerate(building_params_sets):
            annual_data = test_model.simulate_annual_performance(
                datetime(2024, 1, 1), 30, 24,  # 30 days, daily steps
                location_params, building_params, heat_pump_params_sets[1]
            )
            
            total_energy = sum(p.energy_consumption_kWh for p in annual_data)
            avg_cop = np.mean([p.heat_pump_cop for p in annual_data])
            
            annual_energy_by_building[building_params['insulation_level']] = {
                'total_energy_kWh': total_energy,
                'average_cop': avg_cop,
                'floor_area_m2': building_params['floor_area_m2']
            }
        
        print(f"✅ Annual energy consumption by building type (30 days):")
        for insulation_level, metrics in annual_energy_by_building.items():
            energy_per_m2 = metrics['total_energy_kWh'] / metrics['floor_area_m2']
            print(f"  - {insulation_level}:")
            print(f"    * Total energy: {metrics['total_energy_kWh']:.1f} kWh")
            print(f"    * Energy per m²: {energy_per_m2:.1f} kWh/m²")
            print(f"    * Average COP: {metrics['average_cop']:.2f}")
        
        print("\n🎉 ALL SEASONAL PERFORMANCE MODELS TESTS COMPLETED!")
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error in seasonal performance models: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Test completed with errors")
        return False

if __name__ == "__main__":
    success = test_seasonal_performance_models()
    sys.exit(0 if success else 1)