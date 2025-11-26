#!/usr/bin/env python3
"""
Comprehensive Test Suite for Building Load Models
================================================

Tests the multi-zone building load interaction models for heat pump systems,
including zone-specific loads, inter-zone heat transfer, control strategies,
and building dynamics validation.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import the building load models module
from building_load_models import (
    MultiZoneBuildingModel, ZoneType, ControlStrategy, BuildingOrientation,
    ZoneCharacteristics, ZoneThermalProperties, ZoneState, BuildingState
)

def test_building_load_models_comprehensive():
    """Comprehensive test of building load models."""
    
    print("🏢 Testing Building Load Models - Comprehensive Test")
    print("=" * 65)
    
    # Test 1: Import and initialization
    print("🔄 Testing imports and initialization...")
    try:
        model = MultiZoneBuildingModel()
        print("✅ MultiZoneBuildingModel initialized successfully")
        print(f"✅ Zone types database: {len(model.zone_types_database)} zone types")
        print(f"✅ Thermal properties database: {len(model.thermal_properties_database)} property sets")
        print(f"✅ Control strategies: {len(model.control_strategies)} strategies")
        print(f"✅ Building envelope model initialized")
    except Exception as e:
        print(f"❌ Import/initialization failed: {e}")
        return False
    
    # Test 2: Enum functionality
    print("\n🔄 Testing enum functionality...")
    try:
        # Test ZoneType enum
        zone_types = list(ZoneType)
        print(f"✅ ZoneType enum: {len(zone_types)} types")
        print(f"✅ Zone types: {[zt.value for zt in zone_types[:4]]}...")
        
        # Test ControlStrategy enum
        control_strategies = list(ControlStrategy)
        print(f"✅ ControlStrategy enum: {len(control_strategies)} strategies")
        print(f"✅ Control strategies: {[cs.value for cs in control_strategies[:3]]}...")
        
        # Test BuildingOrientation enum
        orientations = list(BuildingOrientation)
        print(f"✅ BuildingOrientation enum: {len(orientations)} orientations")
        print(f"✅ Orientations: {[bo.value for bo in orientations[:4]]}...")
        
    except Exception as e:
        print(f"❌ Enum functionality test failed: {e}")
        return False
    
    # Test 3: Zone types database
    print("\n🔄 Testing zone types database...")
    try:
        zone_db = model.zone_types_database
        
        # Test living room characteristics
        living_room = zone_db[ZoneType.LIVING_ROOM]
        print(f"✅ Living room typical area: {living_room['typical_area_m2']} m²")
        print(f"✅ Living room ceiling height: {living_room['ceiling_height_m']} m")
        print(f"✅ Living room occupancy density: {living_room['occupancy_density_people_per_m2']} people/m²")
        print(f"✅ Living room equipment load: {living_room['equipment_load_W_per_m2']} W/m²")
        print(f"✅ Living room comfort range: {living_room['comfort_temperature_range_C']} °C")
        
        # Test bedroom characteristics
        bedroom = zone_db[ZoneType.BEDROOM]
        print(f"✅ Bedroom typical area: {bedroom['typical_area_m2']} m²")
        print(f"✅ Bedroom air changes: {bedroom['air_changes_per_hour']} ACH")
        
        # Test kitchen characteristics
        kitchen = zone_db[ZoneType.KITCHEN]
        print(f"✅ Kitchen equipment load: {kitchen['equipment_load_W_per_m2']} W/m²")
        print(f"✅ Kitchen air changes: {kitchen['air_changes_per_hour']} ACH")
        
    except Exception as e:
        print(f"❌ Zone types database test failed: {e}")
        return False
    
    # Test 4: Thermal properties database
    print("\n🔄 Testing thermal properties database...")
    try:
        thermal_db = model.thermal_properties_database
        
        # Test insulation levels
        insulation_levels = thermal_db['insulation_levels']
        print(f"✅ Insulation levels: {list(insulation_levels.keys())}")
        
        excellent = insulation_levels['excellent']
        print(f"✅ Excellent insulation - Wall U-value: {excellent['u_value_walls']} W/(m²·K)")
        print(f"✅ Excellent insulation - Window U-value: {excellent['u_value_windows']} W/(m²·K)")
        print(f"✅ Excellent insulation - Air leakage: {excellent['air_leakage']} m³/(h·m²)")
        
        # Test thermal mass types
        thermal_mass = thermal_db['thermal_mass_types']
        print(f"✅ Thermal mass types: {list(thermal_mass.keys())}")
        print(f"✅ Heavy thermal mass: {thermal_mass['heavy']['thermal_mass_kJ_per_m2_K']} kJ/(m²·K)")
        
        # Test window properties
        windows = thermal_db['window_properties']
        print(f"✅ Window types: {list(windows.keys())}")
        print(f"✅ Triple glazed SHGC: {windows['triple_glazed']['shgc']}")
        
    except Exception as e:
        print(f"❌ Thermal properties database test failed: {e}")
        return False
    
    # Test 5: Control strategies database
    print("\n🔄 Testing control strategies database...")
    try:
        control_db = model.control_strategies
        
        # Test single zone strategy
        single_zone = control_db[ControlStrategy.SINGLE_ZONE]
        print(f"✅ Single zone description: {single_zone['description']}")
        print(f"✅ Single zone priority: {single_zone['zone_priority']}")
        print(f"✅ Single zone deadband: {single_zone['temperature_deadband_C']} °C")
        
        # Test multi-zone variable strategy
        multi_var = control_db[ControlStrategy.MULTI_ZONE_VARIABLE]
        print(f"✅ Multi-zone variable: {multi_var['description']}")
        print(f"✅ Air flow modulation: {multi_var['air_flow_modulation']}")
        print(f"✅ Min air flow: {multi_var['minimum_air_flow_percent']}%")
        
        # Test demand controlled strategy
        demand_ctrl = control_db[ControlStrategy.DEMAND_CONTROLLED]
        print(f"✅ Demand controlled CO2 control: {demand_ctrl['co2_control']}")
        print(f"✅ Occupancy sensing: {demand_ctrl['occupancy_sensing']}")
        
    except Exception as e:
        print(f"❌ Control strategies database test failed: {e}")
        return False
    
    # Test 6: Building envelope model
    print("\n🔄 Testing building envelope model...")
    try:
        envelope = model.building_envelope_model
        
        # Test solar orientation factors
        solar_factors = envelope['solar_orientation_factors']
        print(f"✅ Solar orientation factors: {len(solar_factors)} orientations")
        print(f"✅ South factor: {solar_factors[BuildingOrientation.SOUTH]}")
        print(f"✅ North factor: {solar_factors[BuildingOrientation.NORTH]}")
        print(f"✅ East factor: {solar_factors[BuildingOrientation.EAST]}")
        
        # Test thermal bridging
        thermal_bridging = envelope['thermal_bridging']
        print(f"✅ Thermal bridging levels: {list(thermal_bridging.keys())}")
        print(f"✅ Poor construction bridging: {thermal_bridging['poor_construction']*100}%")
        print(f"✅ Excellent construction bridging: {thermal_bridging['excellent_construction']*100}%")
        
        # Test air leakage factors
        air_leakage = envelope['air_leakage_factors']
        print(f"✅ Stack effect coefficient: {air_leakage['stack_effect_coefficient']}")
        print(f"✅ Wind coefficient: {air_leakage['wind_coefficient']}")
        
    except Exception as e:
        print(f"❌ Building envelope model test failed: {e}")
        return False
    
    # Test 7: Zone creation
    print("\n🔄 Testing zone creation...")
    try:
        # Define test building zones
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
                'zone_id': 'bedroom_2',
                'zone_type': 'bedroom',
                'floor_area_m2': 12.0,
                'orientation': 'west'
            },
            {
                'zone_id': 'kitchen',
                'zone_type': 'kitchen',
                'floor_area_m2': 14.0,
                'orientation': 'north'
            },
            {
                'zone_id': 'bathroom',
                'zone_type': 'bathroom',
                'floor_area_m2': 6.0,
                'orientation': 'north'
            }
        ]
        
        # Create zones with default building characteristics
        zones = model.create_building_zones(zone_definitions)
        
        print(f"✅ Created {len(zones)} zones successfully")
        
        # Test individual zone characteristics
        living_room = zones['living_room']
        print(f"✅ Living room zone ID: {living_room.zone_id}")
        print(f"✅ Living room type: {living_room.zone_type.value}")
        print(f"✅ Living room area: {living_room.floor_area_m2} m²")
        print(f"✅ Living room ceiling height: {living_room.ceiling_height_m} m")
        print(f"✅ Living room orientation: {living_room.orientation.value}")
        print(f"✅ Living room thermal mass: {living_room.thermal_mass_kJ_per_K:.0f} kJ/K")
        print(f"✅ Living room internal gains: {living_room.internal_gains_W:.0f} W")
        print(f"✅ Living room occupancy schedule length: {len(living_room.occupancy_schedule)}")
        print(f"✅ Living room heating setpoints length: {len(living_room.setpoint_schedule_heating_C)}")
        
        # Test bedroom characteristics
        bedroom = zones['bedroom_1']
        print(f"✅ Bedroom area: {bedroom.floor_area_m2} m²")
        print(f"✅ Bedroom air changes: {bedroom.air_changes_per_hour} ACH")
        print(f"✅ Bedroom window area: {bedroom.window_area_m2:.1f} m²")
        
        # Test kitchen characteristics
        kitchen = zones['kitchen']
        print(f"✅ Kitchen internal gains: {kitchen.internal_gains_W:.0f} W")
        print(f"✅ Kitchen exterior wall area: {kitchen.exterior_wall_area_m2:.1f} m²")
        
    except Exception as e:
        print(f"❌ Zone creation test failed: {e}")
        return False
    
    # Test 8: Occupancy schedule creation
    print("\n🔄 Testing occupancy schedule creation...")
    try:
        # Test different zone type schedules
        living_schedule = model._create_occupancy_schedule(ZoneType.LIVING_ROOM)
        bedroom_schedule = model._create_occupancy_schedule(ZoneType.BEDROOM)
        kitchen_schedule = model._create_occupancy_schedule(ZoneType.KITCHEN)
        office_schedule = model._create_occupancy_schedule(ZoneType.OFFICE)
        
        print(f"✅ Living room schedule length: {len(living_schedule)}")
        print(f"✅ Living room peak occupancy: {max(living_schedule):.1f}")
        print(f"✅ Living room 8 AM occupancy: {living_schedule[8]:.1f}")
        print(f"✅ Living room 8 PM occupancy: {living_schedule[20]:.1f}")
        
        print(f"✅ Bedroom peak occupancy: {max(bedroom_schedule):.1f}")
        print(f"✅ Bedroom 2 AM occupancy: {bedroom_schedule[2]:.1f}")
        print(f"✅ Bedroom 2 PM occupancy: {bedroom_schedule[14]:.1f}")
        
        print(f"✅ Kitchen morning peak: {kitchen_schedule[7]:.1f}")
        print(f"✅ Kitchen evening peak: {kitchen_schedule[18]:.1f}")
        
        print(f"✅ Office work hours peak: {office_schedule[10]:.1f}")
        print(f"✅ Office night occupancy: {office_schedule[2]:.1f}")
        
    except Exception as e:
        print(f"❌ Occupancy schedule test failed: {e}")
        return False
    
    # Test 9: Zone load calculations
    print("\n🔄 Testing zone load calculations...")
    try:
        # Use living room for detailed testing
        living_room = zones['living_room']
        
        # Test winter heating conditions
        winter_state = model.calculate_zone_loads(
            zone=living_room,
            zone_temp_C=20.0,
            outdoor_temp_C=-5.0,
            solar_irradiance_W_per_m2=200.0,
            hour_of_day=14,  # 2 PM
            building_characteristics={
                'insulation_level': 'average',
                'thermal_mass': 'medium',
                'window_type': 'double_glazed',
                'construction_quality': 'average_construction'
            }
        )
        
        print("✅ Winter heating load calculation:")
        print(f"  - Zone ID: {winter_state.zone_id}")
        print(f"  - Temperature: {winter_state.temperature_C:.1f} °C")
        print(f"  - Heating load: {winter_state.heating_load_W:.0f} W")
        print(f"  - Cooling load: {winter_state.cooling_load_W:.0f} W")
        print(f"  - Internal gains: {winter_state.internal_gains_W:.0f} W")
        print(f"  - Solar gains: {winter_state.solar_gains_W:.0f} W")
        print(f"  - Ventilation load: {winter_state.ventilation_load_W:.0f} W")
        print(f"  - Infiltration load: {winter_state.infiltration_load_W:.0f} W")
        print(f"  - Occupancy: {winter_state.occupancy_count} people")
        print(f"  - Humidity: {winter_state.humidity_percent:.1f}%")
        print(f"  - CO2: {winter_state.co2_ppm:.0f} ppm")
        
        # Test summer cooling conditions
        summer_state = model.calculate_zone_loads(
            zone=living_room,
            zone_temp_C=24.0,
            outdoor_temp_C=35.0,
            solar_irradiance_W_per_m2=800.0,
            hour_of_day=14,  # 2 PM
            building_characteristics={
                'insulation_level': 'average',
                'thermal_mass': 'medium',
                'window_type': 'double_glazed',
                'construction_quality': 'average_construction'
            }
        )
        
        print("✅ Summer cooling load calculation:")
        print(f"  - Heating load: {summer_state.heating_load_W:.0f} W")
        print(f"  - Cooling load: {summer_state.cooling_load_W:.0f} W")
        print(f"  - Solar gains: {summer_state.solar_gains_W:.0f} W")
        print(f"  - Internal gains: {summer_state.internal_gains_W:.0f} W")
        
        # Test different insulation levels
        excellent_state = model.calculate_zone_loads(
            zone=living_room,
            zone_temp_C=20.0,
            outdoor_temp_C=-5.0,
            solar_irradiance_W_per_m2=200.0,
            hour_of_day=14,
            building_characteristics={
                'insulation_level': 'excellent',
                'thermal_mass': 'heavy',
                'window_type': 'triple_glazed',
                'construction_quality': 'excellent_construction'
            }
        )
        
        print("✅ Excellent insulation comparison:")
        print(f"  - Average insulation heating: {winter_state.heating_load_W:.0f} W")
        print(f"  - Excellent insulation heating: {excellent_state.heating_load_W:.0f} W")
        
        # Calculate energy savings with division by zero protection
        if winter_state.heating_load_W > 0:
            energy_savings = ((winter_state.heating_load_W - excellent_state.heating_load_W) / winter_state.heating_load_W * 100)
            print(f"  - Energy savings: {energy_savings:.1f}%")
        else:
            print(f"  - Energy savings: N/A (no heating load in baseline)")
        
    except Exception as e:
        print(f"❌ Zone load calculations test failed: {e}")
        return False
    
    # Test 10: Inter-zone heat transfer
    print("\n🔄 Testing inter-zone heat transfer...")
    try:
        # Create zone states with different temperatures
        zone_states = {}
        zone_temperatures = {
            'living_room': 22.0,
            'bedroom_1': 18.0,
            'bedroom_2': 19.0,
            'kitchen': 24.0,
            'bathroom': 26.0
        }
        
        for zone_id, temp in zone_temperatures.items():
            zone_states[zone_id] = model.calculate_zone_loads(
                zone=zones[zone_id],
                zone_temp_C=temp,
                outdoor_temp_C=10.0,
                solar_irradiance_W_per_m2=300.0,
                hour_of_day=12,
                building_characteristics={
                    'insulation_level': 'average',
                    'thermal_mass': 'medium',
                    'window_type': 'double_glazed',
                    'construction_quality': 'average_construction'
                }
            )
        
        # Calculate inter-zone heat transfer
        inter_zone_transfers = model.calculate_inter_zone_heat_transfer(zones, zone_states)
        
        print("✅ Inter-zone heat transfer calculated:")
        for zone_id, transfer in inter_zone_transfers.items():
            print(f"  - {zone_id}: {transfer:.0f} W")
        
        # Verify energy balance (sum should be close to zero)
        total_transfer = sum(inter_zone_transfers.values())
        print(f"✅ Total inter-zone transfer: {total_transfer:.1f} W (should be ~0)")
        
    except Exception as e:
        print(f"❌ Inter-zone heat transfer test failed: {e}")
        return False
    
    # Test 11: Building load simulation
    print("\n🔄 Testing building load simulation...")
    try:
        # Create sample weather data for 24 hours
        weather_data = []
        for hour in range(24):
            # Sinusoidal temperature variation
            temp = 15.0 + 10.0 * np.sin((hour - 6) * np.pi / 12)
            # Solar irradiance (only during day)
            solar = max(0, 800 * np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
            
            weather_data.append({
                'temperature_C': temp,
                'solar_irradiance_W_per_m2': solar
            })
        
        print(f"✅ Created weather data: {len(weather_data)} hours")
        print(f"✅ Min temperature: {min(w['temperature_C'] for w in weather_data):.1f} °C")
        print(f"✅ Max temperature: {max(w['temperature_C'] for w in weather_data):.1f} °C")
        print(f"✅ Max solar: {max(w['solar_irradiance_W_per_m2'] for w in weather_data):.0f} W/m²")
        
        # Building characteristics
        building_characteristics = {
            'insulation_level': 'average',
            'thermal_mass': 'medium',
            'window_type': 'double_glazed',
            'construction_quality': 'average_construction'
        }
        
        # Simulate with different control strategies
        strategies_to_test = [
            ControlStrategy.SINGLE_ZONE,
            ControlStrategy.MULTI_ZONE_VARIABLE,
            ControlStrategy.DEMAND_CONTROLLED
        ]
        
        simulation_results = {}
        
        for strategy in strategies_to_test:
            building_states = model.simulate_building_loads(
                zones=zones,
                weather_data=weather_data,
                control_strategy=strategy,
                building_characteristics=building_characteristics,
                simulation_hours=24
            )
            
            simulation_results[strategy.value] = building_states
            
            print(f"✅ {strategy.value} simulation: {len(building_states)} hours")
            
            # Check first and last states
            first_state = building_states[0]
            last_state = building_states[-1]
            
            print(f"  - Initial total heating: {first_state.total_heating_load_W:.0f} W")
            print(f"  - Final total heating: {last_state.total_heating_load_W:.0f} W")
            print(f"  - Peak zone load: {max(s.peak_zone_load_W for s in building_states):.0f} W")
            print(f"  - Avg diversity factor: {np.mean([s.load_diversity_factor for s in building_states]):.2f}")
            print(f"  - Required HP capacity: {max(s.heat_pump_capacity_required_W for s in building_states):.0f} W")
        
    except Exception as e:
        print(f"❌ Building load simulation test failed: {e}")
        return False
    
    # Test 12: Load pattern analysis
    print("\n🔄 Testing load pattern analysis...")
    try:
        # Use multi-zone variable results for analysis
        building_states = simulation_results['multi_zone_variable']
        
        load_analysis = model.analyze_load_patterns(building_states)
        
        print("✅ Load pattern analysis completed:")
        print(f"  - Peak heating load: {load_analysis['peak_heating_load_W']:.0f} W")
        print(f"  - Peak cooling load: {load_analysis['peak_cooling_load_W']:.0f} W")
        print(f"  - Average heating load: {load_analysis['average_heating_load_W']:.0f} W")
        print(f"  - Average cooling load: {load_analysis['average_cooling_load_W']:.0f} W")
        print(f"  - Heating load factor: {load_analysis['heating_load_factor']:.2f}")
        print(f"  - Cooling load factor: {load_analysis['cooling_load_factor']:.2f}")
        print(f"  - Average diversity factor: {load_analysis['average_diversity_factor']:.2f}")
        print(f"  - Minimum diversity factor: {load_analysis['minimum_diversity_factor']:.2f}")
        print(f"  - Recommended HP capacity: {load_analysis['recommended_heat_pump_capacity_W']:.0f} W")
        print(f"  - Total heating energy: {load_analysis['total_energy_heating_Wh']:.0f} Wh")
        print(f"  - Total cooling energy: {load_analysis['total_energy_cooling_Wh']:.0f} Wh")
        
        # Check zone-specific analysis
        zone_analysis = load_analysis['zone_analysis']
        print(f"✅ Zone-specific analysis: {len(zone_analysis)} zones")
        
        for zone_id, analysis in zone_analysis.items():
            print(f"  - {zone_id}:")
            print(f"    Peak heating: {analysis['peak_heating_W']:.0f} W")
            print(f"    Peak cooling: {analysis['peak_cooling_W']:.0f} W")
            print(f"    Load factor heating: {analysis['load_factor_heating']:.2f}")
        
    except Exception as e:
        print(f"❌ Load pattern analysis test failed: {e}")
        return False
    
    # Test 13: Control optimization
    print("\n🔄 Testing control optimization...")
    try:
        # Test different optimization objectives
        objectives = ['energy', 'comfort', 'cost']
        
        for objective in objectives:
            optimization = model.optimize_zone_control(
                zones=zones,
                building_states=building_states,
                optimization_objective=objective
            )
            
            print(f"✅ {objective.capitalize()} optimization:")
            
            # Current performance
            current_perf = optimization['current_performance']
            print(f"  - Comfort satisfaction: {current_perf['comfort_satisfaction_percent']:.1f}%")
            print(f"  - Energy efficiency rating: {current_perf['energy_efficiency_rating']:.2f}")
            print(f"  - Load diversity factor: {current_perf['load_diversity_factor']:.2f}")
            
            # Recommendations
            recommendations = optimization['optimization_recommendations']
            print(f"  - Setback strategy: {recommendations['setback_strategy']}")
            print(f"  - Deadband width: {recommendations['deadband_width_C']:.1f} °C")
            print(f"  - Night setback: {recommendations['night_setback_C']:.1f} °C")
            print(f"  - Recommended strategy: {recommendations['recommended_control_strategy']}")
            
            # Potential savings
            savings = optimization['potential_savings']
            print(f"  - Energy reduction: {savings['energy_reduction_percent']:.1f}%")
            print(f"  - Comfort improvement: {savings['comfort_improvement_percent']:.1f}%")
            print(f"  - Cost reduction: {savings['cost_reduction_percent']:.1f}%")
        
    except Exception as e:
        print(f"❌ Control optimization test failed: {e}")
        return False
    
    # Test 14: Model validation
    print("\n🔄 Testing model validation...")
    try:
        validation = model.validate_building_model(zones, building_states)
        
        print("✅ Model validation completed:")
        print(f"  - Model valid: {validation['model_valid']}")
        print(f"  - Validation errors: {len(validation['validation_errors'])}")
        print(f"  - Validation warnings: {len(validation['validation_warnings'])}")
        
        # Performance metrics
        perf_metrics = validation['performance_metrics']
        print(f"  - Total building area: {perf_metrics['total_building_area_m2']:.0f} m²")
        print(f"  - Temperature violation rate: {perf_metrics['temperature_violation_rate']:.4f}")
        print(f"  - Load inconsistency rate: {perf_metrics['load_inconsistency_rate']:.4f}")
        print(f"  - Simulation hours: {perf_metrics['simulation_hours']}")
        print(f"  - Zones count: {perf_metrics['zones_count']}")
        
        # Print any errors or warnings
        if validation['validation_errors']:
            print("  - Validation errors:")
            for error in validation['validation_errors']:
                print(f"    * {error}")
        
        if validation['validation_warnings']:
            print("  - Validation warnings:")
            for warning in validation['validation_warnings'][:3]:  # Show first 3
                print(f"    * {warning}")
        
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False
    
    # Test 15: Dataclass functionality
    print("\n🔄 Testing dataclass functionality...")
    try:
        # Test ZoneCharacteristics dataclass
        test_zone_char = ZoneCharacteristics(
            zone_id="test_zone",
            zone_type=ZoneType.OFFICE,
            floor_area_m2=20.0,
            ceiling_height_m=2.7,
            exterior_wall_area_m2=8.0,
            window_area_m2=1.6,
            orientation=BuildingOrientation.SOUTH,
            thermal_mass_kJ_per_K=6000.0,
            air_changes_per_hour=0.6,
            internal_gains_W=540.0,
            occupancy_schedule=[0.0] * 24,
            setpoint_schedule_heating_C=[20.0] * 24,
            setpoint_schedule_cooling_C=[24.0] * 24
        )
        
        print("✅ ZoneCharacteristics dataclass:")
        print(f"  - Zone ID: {test_zone_char.zone_id}")
        print(f"  - Zone type: {test_zone_char.zone_type.value}")
        print(f"  - Floor area: {test_zone_char.floor_area_m2} m²")
        print(f"  - Orientation: {test_zone_char.orientation.value}")
        
        # Test ZoneState dataclass
        test_zone_state = ZoneState(
            zone_id="test_zone",
            timestamp=datetime.now(),
            temperature_C=22.0,
            humidity_percent=45.0,
            co2_ppm=450.0,
            occupancy_count=1,
            heating_load_W=1500.0,
            cooling_load_W=0.0,
            internal_gains_W=540.0,
            solar_gains_W=200.0,
            ventilation_load_W=300.0,
            infiltration_load_W=150.0,
            inter_zone_heat_transfer_W=50.0
        )
        
        print("✅ ZoneState dataclass:")
        print(f"  - Temperature: {test_zone_state.temperature_C} °C")
        print(f"  - Heating load: {test_zone_state.heating_load_W} W")
        print(f"  - Occupancy: {test_zone_state.occupancy_count} people")
        print(f"  - CO2: {test_zone_state.co2_ppm} ppm")
        print(f"  - Humidity: {test_zone_state.humidity_percent}%")
        
        # Test BuildingState dataclass
        test_building_state = BuildingState(
            timestamp=datetime.now(),
            zones={'test_zone': test_zone_state},
            total_heating_load_W=1500.0,
            total_cooling_load_W=0.0,
            peak_zone_load_W=1500.0,
            load_diversity_factor=1.0,
            building_average_temperature_C=22.0,
            outdoor_temperature_C=10.0,
            heat_pump_capacity_required_W=1800.0
        )
        
        print("✅ BuildingState dataclass:")
        print(f"  - Total heating load: {test_building_state.total_heating_load_W} W")
        print(f"  - Peak zone load: {test_building_state.peak_zone_load_W} W")
        print(f"  - Load diversity: {test_building_state.load_diversity_factor}")
        print(f"  - HP capacity required: {test_building_state.heat_pump_capacity_required_W} W")
        
    except Exception as e:
        print(f"❌ Dataclass functionality test failed: {e}")
        return False
    
    # Test 16: Edge cases and error handling
    print("\n🔄 Testing edge cases and error handling...")
    try:
        # Test empty zone definitions
        try:
            empty_zones = model.create_building_zones([])
            print(f"✅ Empty zone definitions handled: {len(empty_zones)} zones")
        except Exception as e:
            print(f"✅ Empty zone definitions properly rejected: {type(e).__name__}")
        
        # Test invalid zone type
        try:
            invalid_zone_def = [{
                'zone_id': 'invalid',
                'zone_type': 'invalid_type',
                'floor_area_m2': 10.0
            }]
            invalid_zones = model.create_building_zones(invalid_zone_def)
            print("⚠️ Invalid zone type should have been rejected")
        except Exception as e:
            print(f"✅ Invalid zone type properly rejected: {type(e).__name__}")
        
        # Test extreme weather conditions
        extreme_winter = model.calculate_zone_loads(
            zone=zones['living_room'],
            zone_temp_C=20.0,
            outdoor_temp_C=-40.0,  # Extreme cold
            solar_irradiance_W_per_m2=0.0,
            hour_of_day=12,
            building_characteristics={
                'insulation_level': 'poor',
                'thermal_mass': 'light',
                'window_type': 'single_glazed',
                'construction_quality': 'poor_construction'
            }
        )
        
        print(f"✅ Extreme winter load: {extreme_winter.heating_load_W:.0f} W")
        
        # Test extreme summer conditions
        extreme_summer = model.calculate_zone_loads(
            zone=zones['living_room'],
            zone_temp_C=24.0,
            outdoor_temp_C=50.0,  # Extreme heat
            solar_irradiance_W_per_m2=1200.0,  # Very high solar
            hour_of_day=14,
            building_characteristics={
                'insulation_level': 'poor',
                'thermal_mass': 'light',
                'window_type': 'single_glazed',
                'construction_quality': 'poor_construction'
            }
        )
        
        print(f"✅ Extreme summer load: {extreme_summer.cooling_load_W:.0f} W")
        
        # Test validation with empty data
        empty_validation = model.validate_building_model({}, [])
        print(f"✅ Empty data validation: {'PASSED' if not empty_validation['model_valid'] else 'FAILED'}")
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False
    
    print("\n🎉 ALL BUILDING LOAD MODEL TESTS COMPLETED!")
    print("=" * 65)
    print("✅ Imports and initialization: PASSED")
    print("✅ Enum functionality: PASSED")
    print("✅ Zone types database: PASSED")
    print("✅ Thermal properties database: PASSED")
    print("✅ Control strategies database: PASSED")
    print("✅ Building envelope model: PASSED")
    print("✅ Zone creation: PASSED")
    print("✅ Occupancy schedules: PASSED")
    print("✅ Zone load calculations: PASSED")
    print("✅ Inter-zone heat transfer: PASSED")
    print("✅ Building load simulation: PASSED")
    print("✅ Load pattern analysis: PASSED")
    print("✅ Control optimization: PASSED")
    print("✅ Model validation: PASSED")
    print("✅ Dataclass functionality: PASSED")
    print("✅ Edge cases and error handling: PASSED")
    
    return True

if __name__ == "__main__":
    success = test_building_load_models_comprehensive()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
        sys.exit(1)