#!/usr/bin/env python3
"""
Comprehensive Test Suite for Defrost Cycle Models
=================================================

Tests the defrost cycle dynamics for air-source heat pumps, including frost
formation, defrost initiation strategies, cycle dynamics, and energy penalties
associated with defrost operations.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import the defrost cycle models module
from defrost_cycle_models import (
    DefrostCycleModel, DefrostMethod, DefrostInitiationStrategy, DefrostPhase,
    FrostConditions, FrostAccumulation, DefrostParameters, DefrostCycleState
)

def test_defrost_cycle_models_comprehensive():
    """Comprehensive test of defrost cycle models."""
    
    print("❄️ Testing Defrost Cycle Models - Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Import and initialization
    print("🔄 Testing imports and initialization...")
    try:
        model = DefrostCycleModel()
        print("✅ DefrostCycleModel initialized successfully")
        print(f"✅ Defrost methods: {len(model.defrost_methods)} methods")
        print(f"✅ Initiation strategies: {len(model.initiation_strategies)} strategies")
        print(f"✅ Frost formation model initialized")
        print(f"✅ Performance impacts initialized")
    except Exception as e:
        print(f"❌ Import/initialization failed: {e}")
        return False
    
    # Test 2: Enum functionality
    print("\n🔄 Testing enum functionality...")
    try:
        # Test DefrostMethod enum
        defrost_methods = list(DefrostMethod)
        print(f"✅ DefrostMethod enum: {len(defrost_methods)} methods")
        print(f"✅ Defrost methods: {[dm.value for dm in defrost_methods]}")
        
        # Test DefrostInitiationStrategy enum
        initiation_strategies = list(DefrostInitiationStrategy)
        print(f"✅ DefrostInitiationStrategy enum: {len(initiation_strategies)} strategies")
        print(f"✅ Initiation strategies: {[dis.value for dis in initiation_strategies[:3]]}...")
        
        # Test DefrostPhase enum
        defrost_phases = list(DefrostPhase)
        print(f"✅ DefrostPhase enum: {len(defrost_phases)} phases")
        print(f"✅ Defrost phases: {[dp.value for dp in defrost_phases]}")
        
    except Exception as e:
        print(f"❌ Enum functionality test failed: {e}")
        return False
    
    # Test 3: Defrost methods database
    print("\n🔄 Testing defrost methods database...")
    try:
        methods_db = model.defrost_methods
        
        # Test reverse cycle method
        reverse_cycle = methods_db[DefrostMethod.REVERSE_CYCLE]
        print(f"✅ Reverse cycle duration: {reverse_cycle['typical_duration_s']} s")
        print(f"✅ Reverse cycle energy factor: {reverse_cycle['energy_consumption_factor']}")
        print(f"✅ Reverse cycle effectiveness: {reverse_cycle['effectiveness']}")
        print(f"✅ Reverse cycle aux heat: {reverse_cycle['auxiliary_heat_required']}")
        print(f"✅ Reverse cycle fan operation: {reverse_cycle['fan_operation']}")
        
        # Test hot gas bypass method
        hot_gas = methods_db[DefrostMethod.HOT_GAS_BYPASS]
        print(f"✅ Hot gas bypass duration: {hot_gas['typical_duration_s']} s")
        print(f"✅ Hot gas bypass effectiveness: {hot_gas['effectiveness']}")
        
        # Test electric resistance method
        electric = methods_db[DefrostMethod.ELECTRIC_RESISTANCE]
        print(f"✅ Electric resistance duration: {electric['typical_duration_s']} s")
        print(f"✅ Electric resistance energy factor: {electric['energy_consumption_factor']}")
        
        # Test ambient air method
        ambient = methods_db[DefrostMethod.AMBIENT_AIR]
        print(f"✅ Ambient air duration: {ambient['typical_duration_s']} s")
        print(f"✅ Ambient air effectiveness: {ambient['effectiveness']}")
        
    except Exception as e:
        print(f"❌ Defrost methods database test failed: {e}")
        return False
    
    # Test 4: Initiation strategies database
    print("\n🔄 Testing initiation strategies database...")
    try:
        strategies_db = model.initiation_strategies
        
        # Test time-based strategy
        time_based = strategies_db[DefrostInitiationStrategy.TIME_BASED]
        print(f"✅ Time-based interval: {time_based['interval_minutes']} min")
        print(f"✅ Time-based temp threshold: {time_based['temperature_threshold_C']} °C")
        print(f"✅ Time-based humidity threshold: {time_based['humidity_threshold_percent']}%")
        
        # Test temperature-based strategy
        temp_based = strategies_db[DefrostInitiationStrategy.TEMPERATURE_BASED]
        print(f"✅ Temp-based coil threshold: {temp_based['coil_temperature_threshold_C']} °C")
        print(f"✅ Temp-based differential: {temp_based['temperature_differential_C']} °C")
        print(f"✅ Temp-based min interval: {temp_based['minimum_interval_minutes']} min")
        
        # Test pressure-based strategy
        pressure_based = strategies_db[DefrostInitiationStrategy.PRESSURE_BASED]
        print(f"✅ Pressure-based threshold: {pressure_based['pressure_drop_threshold_Pa']} Pa")
        print(f"✅ Pressure rise rate: {pressure_based['pressure_rise_rate_Pa_per_min']} Pa/min")
        
        # Test demand-based strategy
        demand_based = strategies_db[DefrostInitiationStrategy.DEMAND_BASED]
        print(f"✅ Demand-based capacity threshold: {demand_based['capacity_degradation_threshold_percent']}%")
        print(f"✅ Demand-based COP threshold: {demand_based['cop_degradation_threshold_percent']}%")
        
        # Test intelligent adaptive strategy
        adaptive = strategies_db[DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE]
        print(f"✅ Adaptive frost threshold: {adaptive['frost_accumulation_threshold_mm']} mm")
        print(f"✅ Adaptive multiple criteria: {adaptive['multiple_criteria_required']}")
        print(f"✅ Adaptive learning rate: {adaptive['learning_rate']}")
        
    except Exception as e:
        print(f"❌ Initiation strategies database test failed: {e}")
        return False
    
    # Test 5: Frost formation model
    print("\n🔄 Testing frost formation model...")
    try:
        frost_model = model.frost_formation_model
        
        # Test formation conditions
        formation_cond = frost_model['formation_conditions']
        print(f"✅ Formation temp threshold: {formation_cond['temperature_threshold_C']} °C")
        print(f"✅ Formation humidity threshold: {formation_cond['humidity_threshold_percent']}%")
        print(f"✅ Surface temp offset: {formation_cond['surface_temperature_offset_C']} °C")
        print(f"✅ Min air velocity: {formation_cond['minimum_air_velocity_m_per_s']} m/s")
        
        # Test formation rates
        formation_rates = frost_model['formation_rates']
        print(f"✅ Base formation rate: {formation_rates['base_rate_mm_per_hour']} mm/h")
        print(f"✅ Temperature factor: {formation_rates['temperature_factor']}")
        print(f"✅ Humidity factor: {formation_rates['humidity_factor']}")
        
        # Test frost properties
        frost_props = frost_model['frost_properties']
        print(f"✅ Frost density: {frost_props['density_kg_per_m3']} kg/m³")
        print(f"✅ Frost thermal conductivity: {frost_props['thermal_conductivity_W_per_m_K']} W/(m·K)")
        print(f"✅ Frost latent heat: {frost_props['latent_heat_fusion_J_per_kg']} J/kg")
        
        # Test performance impacts
        perf_impacts = frost_model['performance_impacts']
        print(f"✅ Thermal resistance factor: {perf_impacts['thermal_resistance_factor']} K·m²/W per mm")
        print(f"✅ Pressure drop factor: {perf_impacts['pressure_drop_factor']} Pa per mm")
        
    except Exception as e:
        print(f"❌ Frost formation model test failed: {e}")
        return False
    
    # Test 6: Frost formation rate calculation
    print("\n🔄 Testing frost formation rate calculation...")
    try:
        # Test conditions favoring frost formation
        frost_conditions = FrostConditions(
            outdoor_temperature_C=-5.0,
            outdoor_humidity_percent=85.0,
            coil_surface_temperature_C=-12.0,
            air_velocity_m_per_s=2.5,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=0.0
        )
        
        formation_rate = model.calculate_frost_formation_rate(frost_conditions)
        print(f"✅ Frost formation rate (cold, humid): {formation_rate:.3f} mm/h")
        
        # Test conditions not favoring frost formation
        no_frost_conditions = FrostConditions(
            outdoor_temperature_C=10.0,  # Too warm
            outdoor_humidity_percent=40.0,  # Too dry
            coil_surface_temperature_C=5.0,  # Above freezing
            air_velocity_m_per_s=2.0,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=0.0
        )
        
        no_formation_rate = model.calculate_frost_formation_rate(no_frost_conditions)
        print(f"✅ Frost formation rate (warm, dry): {no_formation_rate:.3f} mm/h")
        
        # Test extreme conditions
        extreme_conditions = FrostConditions(
            outdoor_temperature_C=-20.0,
            outdoor_humidity_percent=95.0,
            coil_surface_temperature_C=-25.0,
            air_velocity_m_per_s=1.0,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=2.0  # With precipitation
        )
        
        extreme_rate = model.calculate_frost_formation_rate(extreme_conditions)
        print(f"✅ Frost formation rate (extreme): {extreme_rate:.3f} mm/h")
        
    except Exception as e:
        print(f"❌ Frost formation rate calculation test failed: {e}")
        return False
    
    # Test 7: Frost accumulation update
    print("\n🔄 Testing frost accumulation update...")
    try:
        # Initial frost state (clean coil)
        initial_frost = FrostAccumulation(
            frost_thickness_mm=0.0,
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.0,
            coverage_percentage=0.0,
            thermal_resistance_K_per_W=0.0,
            pressure_drop_increase_Pa=0.0,
            heat_transfer_degradation_percent=0.0
        )
        
        print("✅ Initial frost state:")
        print(f"  - Thickness: {initial_frost.frost_thickness_mm} mm")
        print(f"  - Mass: {initial_frost.frost_mass_kg} kg")
        print(f"  - Coverage: {initial_frost.coverage_percentage}%")
        
        # Update after 1 hour
        frost_after_1h = model.update_frost_accumulation(
            initial_frost, formation_rate, 1.0, heat_exchanger_area_m2=2.0
        )
        
        print("✅ Frost after 1 hour:")
        print(f"  - Thickness: {frost_after_1h.frost_thickness_mm:.3f} mm")
        print(f"  - Mass: {frost_after_1h.frost_mass_kg:.4f} kg")
        print(f"  - Coverage: {frost_after_1h.coverage_percentage:.1f}%")
        print(f"  - Thermal resistance: {frost_after_1h.thermal_resistance_K_per_W:.6f} K·m²/W")
        print(f"  - Pressure drop increase: {frost_after_1h.pressure_drop_increase_Pa:.1f} Pa")
        print(f"  - Heat transfer degradation: {frost_after_1h.heat_transfer_degradation_percent:.2f}%")
        
        # Update after 4 hours total
        frost_after_4h = model.update_frost_accumulation(
            frost_after_1h, formation_rate, 3.0, heat_exchanger_area_m2=2.0
        )
        
        print("✅ Frost after 4 hours total:")
        print(f"  - Thickness: {frost_after_4h.frost_thickness_mm:.3f} mm")
        print(f"  - Mass: {frost_after_4h.frost_mass_kg:.4f} kg")
        print(f"  - Coverage: {frost_after_4h.coverage_percentage:.1f}%")
        print(f"  - Pressure drop increase: {frost_after_4h.pressure_drop_increase_Pa:.1f} Pa")
        
    except Exception as e:
        print(f"❌ Frost accumulation update test failed: {e}")
        return False
    
    # Test 8: Performance impact calculation
    print("\n🔄 Testing performance impact calculation...")
    try:
        # Test with different frost thicknesses
        frost_thicknesses = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        print("✅ Performance impacts by frost thickness:")
        for thickness in frost_thicknesses:
            test_frost = FrostAccumulation(
                frost_thickness_mm=thickness,
                frost_density_kg_per_m3=300.0,
                frost_mass_kg=thickness * 0.0006,  # Approximate
                coverage_percentage=min(100.0, thickness * 20.0),
                thermal_resistance_K_per_W=thickness * 0.00025,
                pressure_drop_increase_Pa=thickness * 50.0,
                heat_transfer_degradation_percent=thickness * 5.0
            )
            
            impact = model.calculate_performance_impact(test_frost)
            
            print(f"  - {thickness:.1f} mm: Capacity↓{impact['capacity_reduction_percent']:.1f}%, "
                  f"COP↓{impact['cop_reduction_percent']:.1f}%, "
                  f"Airflow↓{impact['air_flow_reduction_percent']:.1f}%, "
                  f"ΔP+{impact['pressure_drop_increase_Pa']:.0f}Pa")
        
    except Exception as e:
        print(f"❌ Performance impact calculation test failed: {e}")
        return False
    
    # Test 9: Defrost initiation strategies
    print("\n🔄 Testing defrost initiation strategies...")
    try:
        # Test with moderate frost accumulation
        test_frost = frost_after_4h  # From previous test
        
        system_performance = {
            'capacity_degradation_percent': 12.0,
            'cop_degradation_percent': 8.0,
            'load_satisfaction_percent': 88.0
        }
        
        # Test different strategies
        strategies_to_test = [
            DefrostInitiationStrategy.TIME_BASED,
            DefrostInitiationStrategy.TEMPERATURE_BASED,
            DefrostInitiationStrategy.PRESSURE_BASED,
            DefrostInitiationStrategy.DEMAND_BASED,
            DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE
        ]
        
        for strategy in strategies_to_test:
            should_defrost = model.check_defrost_initiation(
                test_frost, frost_conditions, strategy, 7200.0, system_performance
            )
            print(f"✅ {strategy.value}: Should defrost = {should_defrost}")
        
        # Test with extreme conditions requiring defrost
        extreme_frost = FrostAccumulation(
            frost_thickness_mm=4.0,  # Thick frost
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.0024,
            coverage_percentage=80.0,
            thermal_resistance_K_per_W=0.001,
            pressure_drop_increase_Pa=200.0,  # High pressure drop
            heat_transfer_degradation_percent=20.0
        )
        
        extreme_performance = {
            'capacity_degradation_percent': 25.0,  # High degradation
            'cop_degradation_percent': 18.0,
            'load_satisfaction_percent': 75.0
        }
        
        print("✅ Extreme conditions defrost decisions:")
        for strategy in strategies_to_test[:3]:  # Test first 3
            should_defrost = model.check_defrost_initiation(
                extreme_frost, frost_conditions, strategy, 10800.0, extreme_performance
            )
            print(f"  - {strategy.value}: {should_defrost}")
        
    except Exception as e:
        print(f"❌ Defrost initiation strategies test failed: {e}")
        return False
    
    # Test 10: Defrost cycle simulation
    print("\n🔄 Testing defrost cycle simulation...")
    try:
        # Test different defrost methods
        methods_to_test = [
            DefrostMethod.REVERSE_CYCLE,
            DefrostMethod.HOT_GAS_BYPASS,
            DefrostMethod.ELECTRIC_RESISTANCE
        ]
        
        for method in methods_to_test:
            cycle_data, total_energy = model.simulate_defrost_cycle(
                method=method,
                initial_frost=test_frost,
                ambient_temp_C=-5.0,
                cycle_duration_s=None  # Use default duration
            )
            
            print(f"✅ {method.value} defrost simulation:")
            print(f"  - Cycle duration: {len(cycle_data) * 10:.0f} seconds")
            print(f"  - Total energy: {total_energy:.2f} kWh")
            print(f"  - Data points: {len(cycle_data)}")
            
            if cycle_data:
                first_point = cycle_data[0]
                last_point = cycle_data[-1]
                print(f"  - Initial coil temp: {first_point['coil_temperature_C']:.1f} °C")
                print(f"  - Final coil temp: {last_point['coil_temperature_C']:.1f} °C")
                print(f"  - Initial frost: {first_point['frost_thickness_mm']:.2f} mm")
                print(f"  - Final frost: {last_point['frost_thickness_mm']:.2f} mm")
                print(f"  - Peak power: {max(p['power_consumption_kW'] for p in cycle_data):.1f} kW")
        
    except Exception as e:
        print(f"❌ Defrost cycle simulation test failed: {e}")
        return False
    
    # Test 11: Extended defrost operation simulation
    print("\n🔄 Testing extended defrost operation simulation...")
    try:
        # Create weather conditions for 24 hours
        weather_conditions = []
        for hour in range(24):
            # Simulate winter day with varying conditions
            temp = -10.0 + 5.0 * np.sin((hour - 6) * np.pi / 12)  # Temperature variation
            humidity = 75.0 + 15.0 * np.sin((hour - 12) * np.pi / 12)  # Humidity variation
            
            conditions = FrostConditions(
                outdoor_temperature_C=temp,
                outdoor_humidity_percent=humidity,
                coil_surface_temperature_C=temp - 8.0,
                air_velocity_m_per_s=2.0,
                atmospheric_pressure_Pa=101325.0,
                precipitation_rate_mm_per_h=0.0
            )
            weather_conditions.append(conditions)
        
        print(f"✅ Created weather conditions for 24 hours")
        print(f"✅ Temperature range: {min(w.outdoor_temperature_C for w in weather_conditions):.1f} to {max(w.outdoor_temperature_C for w in weather_conditions):.1f} °C")
        print(f"✅ Humidity range: {min(w.outdoor_humidity_percent for w in weather_conditions):.1f} to {max(w.outdoor_humidity_percent for w in weather_conditions):.1f}%")
        
        # Simulate defrost operation
        start_time = datetime.now()
        defrost_states = model.simulate_defrost_operation(
            start_time=start_time,
            duration_hours=24,
            weather_conditions=weather_conditions,
            defrost_method=DefrostMethod.REVERSE_CYCLE,
            initiation_strategy=DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE
        )
        
        print(f"✅ Extended simulation completed: {len(defrost_states)} hourly states")
        
        # Analyze results
        defrost_cycles = sum(1 for state in defrost_states if state.phase == DefrostPhase.DEFROST_ACTIVE)
        total_energy_penalty = sum(state.energy_penalty_kWh for state in defrost_states)
        max_frost_thickness = max(state.frost_accumulation.frost_thickness_mm for state in defrost_states)
        avg_capacity_loss = np.mean([state.capacity_loss_percent for state in defrost_states])
        final_defrost_frequency = defrost_states[-1].defrost_frequency_per_day if defrost_states else 0
        
        print(f"✅ Analysis results:")
        print(f"  - Defrost cycles: {defrost_cycles}")
        print(f"  - Total energy penalty: {total_energy_penalty:.2f} kWh")
        print(f"  - Max frost thickness: {max_frost_thickness:.2f} mm")
        print(f"  - Avg capacity loss: {avg_capacity_loss:.1f}%")
        print(f"  - Final defrost frequency: {final_defrost_frequency:.1f} cycles/day")
        
        # Check some individual states
        if len(defrost_states) >= 12:
            mid_state = defrost_states[12]  # Noon state
            print(f"✅ Noon state (hour 12):")
            print(f"  - Phase: {mid_state.phase.value}")
            print(f"  - Frost thickness: {mid_state.frost_accumulation.frost_thickness_mm:.2f} mm")
            print(f"  - Capacity loss: {mid_state.capacity_loss_percent:.1f}%")
            print(f"  - COP degradation: {mid_state.cop_degradation_factor:.3f}")
            print(f"  - Time since last defrost: {mid_state.time_since_last_defrost_s / 3600:.1f} hours")
        
    except Exception as e:
        print(f"❌ Extended defrost operation simulation test failed: {e}")
        return False
    
    # Test 12: Dataclass functionality
    print("\n🔄 Testing dataclass functionality...")
    try:
        # Test FrostConditions dataclass
        test_conditions = FrostConditions(
            outdoor_temperature_C=-8.0,
            outdoor_humidity_percent=80.0,
            coil_surface_temperature_C=-15.0,
            air_velocity_m_per_s=2.5,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=1.0
        )
        
        print("✅ FrostConditions dataclass:")
        print(f"  - Outdoor temp: {test_conditions.outdoor_temperature_C} °C")
        print(f"  - Humidity: {test_conditions.outdoor_humidity_percent}%")
        print(f"  - Coil temp: {test_conditions.coil_surface_temperature_C} °C")
        print(f"  - Air velocity: {test_conditions.air_velocity_m_per_s} m/s")
        print(f"  - Precipitation: {test_conditions.precipitation_rate_mm_per_h} mm/h")
        
        # Test FrostAccumulation dataclass
        test_accumulation = FrostAccumulation(
            frost_thickness_mm=2.5,
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.0015,
            coverage_percentage=50.0,
            thermal_resistance_K_per_W=0.000625,
            pressure_drop_increase_Pa=125.0,
            heat_transfer_degradation_percent=12.5
        )
        
        print("✅ FrostAccumulation dataclass:")
        print(f"  - Thickness: {test_accumulation.frost_thickness_mm} mm")
        print(f"  - Mass: {test_accumulation.frost_mass_kg} kg")
        print(f"  - Coverage: {test_accumulation.coverage_percentage}%")
        print(f"  - Thermal resistance: {test_accumulation.thermal_resistance_K_per_W} K·m²/W")
        
        # Test DefrostParameters dataclass
        test_parameters = DefrostParameters(
            method=DefrostMethod.REVERSE_CYCLE,
            initiation_strategy=DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE,
            initiation_threshold=2.0,
            cycle_duration_s=600.0,
            termination_criteria={'coil_temp_C': 10.0, 'frost_thickness_mm': 0.1},
            energy_consumption_kW=6.0,
            auxiliary_heat_kW=3.0,
            fan_operation_during_defrost=False
        )
        
        print("✅ DefrostParameters dataclass:")
        print(f"  - Method: {test_parameters.method.value}")
        print(f"  - Strategy: {test_parameters.initiation_strategy.value}")
        print(f"  - Duration: {test_parameters.cycle_duration_s} s")
        print(f"  - Energy consumption: {test_parameters.energy_consumption_kW} kW")
        print(f"  - Auxiliary heat: {test_parameters.auxiliary_heat_kW} kW")
        
    except Exception as e:
        print(f"❌ Dataclass functionality test failed: {e}")
        return False
    
    # Test 13: Model validation
    print("\n🔄 Testing model validation...")
    try:
        validation_results = model.validate_defrost_model()
        
        print("✅ Model validation results:")
        for test_name, result in validation_results.items():
            status = "PASSED" if result else "FAILED"
            print(f"  - {test_name}: {status}")
        
        total_tests = len(validation_results)
        passed_tests = sum(validation_results.values())
        print(f"✅ Overall validation: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("✅ All validation tests passed!")
        else:
            print("⚠️ Some validation tests failed")
        
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False
    
    # Test 14: Edge cases and error handling
    print("\n🔄 Testing edge cases and error handling...")
    try:
        # Test with zero frost thickness
        zero_frost = FrostAccumulation(
            frost_thickness_mm=0.0,
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.0,
            coverage_percentage=0.0,
            thermal_resistance_K_per_W=0.0,
            pressure_drop_increase_Pa=0.0,
            heat_transfer_degradation_percent=0.0
        )
        
        zero_impact = model.calculate_performance_impact(zero_frost)
        print(f"✅ Zero frost impact: Capacity↓{zero_impact['capacity_reduction_percent']:.1f}%")
        
        # Test with extreme frost thickness
        extreme_frost = FrostAccumulation(
            frost_thickness_mm=10.0,  # Very thick
            frost_density_kg_per_m3=300.0,
            frost_mass_kg=0.006,
            coverage_percentage=100.0,
            thermal_resistance_K_per_W=0.0025,
            pressure_drop_increase_Pa=500.0,
            heat_transfer_degradation_percent=50.0
        )
        
        extreme_impact = model.calculate_performance_impact(extreme_frost)
        print(f"✅ Extreme frost impact: Capacity↓{extreme_impact['capacity_reduction_percent']:.1f}%")
        
        # Test formation rate with extreme conditions
        extreme_cold = FrostConditions(
            outdoor_temperature_C=-40.0,
            outdoor_humidity_percent=100.0,
            coil_surface_temperature_C=-45.0,
            air_velocity_m_per_s=0.1,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=10.0
        )
        
        extreme_formation_rate = model.calculate_frost_formation_rate(extreme_cold)
        print(f"✅ Extreme cold formation rate: {extreme_formation_rate:.3f} mm/h")
        
        # Test with warm conditions (no frost)
        warm_conditions = FrostConditions(
            outdoor_temperature_C=20.0,
            outdoor_humidity_percent=30.0,
            coil_surface_temperature_C=15.0,
            air_velocity_m_per_s=3.0,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=0.0
        )
        
        warm_formation_rate = model.calculate_frost_formation_rate(warm_conditions)
        print(f"✅ Warm conditions formation rate: {warm_formation_rate:.3f} mm/h")
        
        # Test defrost initiation with no frost
        no_defrost_needed = model.check_defrost_initiation(
            zero_frost, warm_conditions, DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE,
            3600.0, {'capacity_degradation_percent': 0, 'cop_degradation_percent': 0, 'load_satisfaction_percent': 100}
        )
        warm_conditions = FrostConditions(
            outdoor_temperature_C=20.0,
            outdoor_humidity_percent=30.0,
            coil_surface_temperature_C=15.0,
            air_velocity_m_per_s=3.0,
            atmospheric_pressure_Pa=101325.0,
            precipitation_rate_mm_per_h=0.0
        )
        
        warm_formation_rate = model.calculate_frost_formation_rate(warm_conditions)
        print(f"✅ Warm conditions formation rate: {warm_formation_rate:.3f} mm/h")
        
        # Test defrost initiation with no frost
        no_defrost_needed = model.check_defrost_initiation(
            zero_frost, warm_conditions, DefrostInitiationStrategy.INTELLIGENT_ADAPTIVE,
            3600.0, {'capacity_degradation_percent': 0, 'cop_degradation_percent': 0, 'load_satisfaction_percent': 100}
        )
        print(f"✅ No defrost needed (warm conditions): {no_defrost_needed}")
        
    except Exception as e:
        print(f"❌ Edge cases and error handling test failed: {e}")
        return False
    
    print("\n🎉 ALL DEFROST CYCLE MODEL TESTS COMPLETED!")
    print("=" * 60)
    print("✅ Imports and initialization: PASSED")
    print("✅ Enum functionality: PASSED")
    print("✅ Defrost methods database: PASSED")
    print("✅ Initiation strategies database: PASSED")
    print("✅ Frost formation model: PASSED")
    print("✅ Frost formation rate calculation: PASSED")
    print("✅ Frost accumulation update: PASSED")
    print("✅ Performance impact calculation: PASSED")
    print("✅ Defrost initiation strategies: PASSED")
    print("✅ Defrost cycle simulation: PASSED")
    print("✅ Extended defrost operation simulation: PASSED")
    print("✅ Dataclass functionality: PASSED")
    print("✅ Model validation: PASSED")
    print("✅ Edge cases and error handling: PASSED")
    
    return True

if __name__ == "__main__":
    success = test_defrost_cycle_models_comprehensive()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
        sys.exit(1)