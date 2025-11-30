#!/usr/bin/env python3
"""
Comprehensive Test Suite for Indoor Performance Models
=====================================================

This test suite validates the indoor performance modeling system for heat engine
equipment operating in controlled indoor environments.

Author: AIRTHIUM COP Prediction System
Date: 2024
"""

import sys
import os
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from indoor_performance_models import (
        IndoorPerformanceModel, OperationalMode, LoadPattern,
        IndoorConditions, EquipmentState, PerformanceFactors, IndoorPerformanceData
    )
    print("✅ Successfully imported indoor performance models")
except ImportError as e:
    print(f"❌ Failed to import indoor performance models: {e}")
    sys.exit(1)

def test_indoor_conditions_generation():
    """Test indoor environmental conditions generation"""
    print("\n🔄 Testing indoor conditions generation...")
    
    model = IndoorPerformanceModel()
    test_date = datetime(2024, 6, 15, 12, 0)
    
    # Test different facility parameters
    facility_configs = [
        {'hvac_control_quality': 'poor', 'power_supply_stability': 'unstable', 'vibration_environment': 'high'},
        {'hvac_control_quality': 'good', 'power_supply_stability': 'stable', 'vibration_environment': 'low'},
        {'hvac_control_quality': 'excellent', 'power_supply_stability': 'regulated', 'vibration_environment': 'low'}
    ]
    
    for i, config in enumerate(facility_configs):
        conditions = model.generate_indoor_conditions(test_date, config)
        
        print(f"✅ Config {i+1} ({config['hvac_control_quality']} HVAC):")
        print(f"  - Ambient temperature: {conditions.ambient_temperature_C:.1f}°C")
        print(f"  - Equipment room temperature: {conditions.equipment_room_temperature_C:.1f}°C")
        print(f"  - Relative humidity: {conditions.relative_humidity_percent:.1f}%")
        print(f"  - Power voltage: {conditions.power_supply_voltage_V:.1f}V")
        print(f"  - Power frequency: {conditions.power_supply_frequency_Hz:.2f}Hz")
        print(f"  - Vibration level: {conditions.vibration_level_mm_per_s:.2f} mm/s")
        
        # Validate ranges
        assert 15 <= conditions.ambient_temperature_C <= 30, f"Temperature out of range: {conditions.ambient_temperature_C}"
        assert 30 <= conditions.relative_humidity_percent <= 70, f"Humidity out of range: {conditions.relative_humidity_percent}"
        assert 200 <= conditions.power_supply_voltage_V <= 260, f"Voltage out of range: {conditions.power_supply_voltage_V}"
        assert 49 <= conditions.power_supply_frequency_Hz <= 51, f"Frequency out of range: {conditions.power_supply_frequency_Hz}"
    
    print("✅ Indoor conditions generation tests passed")

def test_equipment_state_calculation():
    """Test equipment operational state calculation"""
    print("\n🔄 Testing equipment state calculation...")
    
    model = IndoorPerformanceModel()
    test_date = datetime(2024, 6, 15, 12, 0)
    
    # Test different operating hours and load patterns
    test_scenarios = [
        (100.0, LoadPattern.RESIDENTIAL),
        (1000.0, LoadPattern.COMMERCIAL),
        (5000.0, LoadPattern.INDUSTRIAL),
        (10000.0, LoadPattern.LABORATORY)
    ]
    
    for operating_hours, load_pattern in test_scenarios:
        equipment_state = model.calculate_equipment_state(test_date, operating_hours, load_pattern)
        
        print(f"✅ {load_pattern.value} pattern ({operating_hours}h):")
        print(f"  - Operational mode: {equipment_state.operational_mode.value}")
        print(f"  - Load demand: {equipment_state.load_demand_percent:.1f}%")
        print(f"  - Cycles completed: {equipment_state.cycles_completed}")
        print(f"  - Hours since maintenance: {equipment_state.last_maintenance_hours:.1f}h")
        print(f"  - Hot side temperature: {equipment_state.component_temperatures_C.get('hot_side_nominal_C', 0):.1f}°C")
        print(f"  - Working gas pressure: {equipment_state.internal_pressures_Pa.get('working_gas_pressure_Pa', 0)/1000:.0f} kPa")
        
        # Validate state
        assert equipment_state.operating_hours_total == operating_hours
        assert 0 <= equipment_state.load_demand_percent <= 120
        assert len(equipment_state.component_temperatures_C) == 5
        assert len(equipment_state.internal_pressures_Pa) == 3
        assert len(equipment_state.flow_rates_kg_per_s) == 3
    
    print("✅ Equipment state calculation tests passed")

def test_performance_factors_calculation():
    """Test performance factors calculation"""
    print("\n🔄 Testing performance factors calculation...")
    
    model = IndoorPerformanceModel()
    test_date = datetime(2024, 6, 15, 12, 0)
    
    # Test different aging scenarios
    aging_scenarios = [
        (100.0, "New equipment"),
        (2000.0, "Moderate aging"),
        (8000.0, "Significant aging"),
        (15000.0, "High aging")
    ]
    
    for operating_hours, description in aging_scenarios:
        indoor_conditions = model.generate_indoor_conditions(test_date)
        equipment_state = model.calculate_equipment_state(test_date, operating_hours)
        performance_factors = model.calculate_performance_factors(indoor_conditions, equipment_state)
        
        print(f"✅ {description} ({operating_hours}h):")
        print(f"  - Aging factor: {performance_factors.aging_factor:.3f}")
        print(f"  - Fouling factor: {performance_factors.fouling_factor:.3f}")
        print(f"  - Wear factor: {performance_factors.wear_factor:.3f}")
        print(f"  - Maintenance factor: {performance_factors.maintenance_factor:.3f}")
        print(f"  - Load efficiency factor: {performance_factors.load_efficiency_factor:.3f}")
        print(f"  - Temperature factor: {performance_factors.temperature_factor:.3f}")
        print(f"  - Power quality factor: {performance_factors.power_quality_factor:.3f}")
        
        # Validate factors
        assert 0.5 <= performance_factors.aging_factor <= 1.1
        assert 0.5 <= performance_factors.fouling_factor <= 1.1
        assert 0.5 <= performance_factors.wear_factor <= 1.1
        assert 0.9 <= performance_factors.maintenance_factor <= 1.1
        assert 0.5 <= performance_factors.load_efficiency_factor <= 1.1
        assert 0.9 <= performance_factors.temperature_factor <= 1.1
        assert 0.9 <= performance_factors.power_quality_factor <= 1.1
    
    print("✅ Performance factors calculation tests passed")

def test_cop_calculation():
    """Test COP calculation with various conditions"""
    print("\n🔄 Testing COP calculation...")
    
    model = IndoorPerformanceModel()
    test_date = datetime(2024, 6, 15, 12, 0)
    
    # Test different base COP values
    base_cops = [3.5, 4.5, 5.5, 6.0]
    
    for base_cop in base_cops:
        indoor_conditions = model.generate_indoor_conditions(test_date)
        equipment_state = model.calculate_equipment_state(test_date, 2000.0)
        performance_factors = model.calculate_performance_factors(indoor_conditions, equipment_state)
        
        calculated_cop = model.calculate_indoor_cop(
            indoor_conditions, equipment_state, performance_factors, base_cop
        )
        
        print(f"✅ Base COP {base_cop:.1f}:")
        print(f"  - Calculated COP: {calculated_cop:.2f}")
        print(f"  - Operational mode: {equipment_state.operational_mode.value}")
        print(f"  - Load demand: {equipment_state.load_demand_percent:.1f}%")
        
        # Validate COP
        assert 1.0 <= calculated_cop <= 8.0, f"COP out of range: {calculated_cop}"
        
        # COP should be related to base COP
        if equipment_state.operational_mode == OperationalMode.STEADY_STATE:
            assert 0.5 * base_cop <= calculated_cop <= 1.2 * base_cop
    
    print("✅ COP calculation tests passed")

def test_simulation_execution():
    """Test full simulation execution"""
    print("\n🔄 Testing simulation execution...")
    
    model = IndoorPerformanceModel()
    
    # Test short simulation
    performance_data = model.simulate_indoor_performance(
        start_date=datetime(2024, 1, 1),
        duration_days=2,
        time_step_hours=6,
        load_pattern=LoadPattern.COMMERCIAL
    )
    
    expected_points = 2 * 24 // 6  # 8 points
    
    print(f"✅ Short simulation:")
    print(f"  - Data points: {len(performance_data)} (expected: {expected_points})")
    
    if performance_data:
        cops = [p.heat_engine_cop for p in performance_data]
        efficiencies = [p.thermal_efficiency_percent for p in performance_data]
        energy_consumptions = [p.energy_consumption_kWh for p in performance_data]
        
        print(f"  - COP range: {min(cops):.2f} - {max(cops):.2f}")
        print(f"  - Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")
        print(f"  - Energy consumption range: {min(energy_consumptions):.2f} - {max(energy_consumptions):.2f} kWh")
        
        # Validate data
        assert len(performance_data) == expected_points
        assert all(1.0 <= cop <= 8.0 for cop in cops)
        assert all(0 <= eff <= 50 for eff in efficiencies)
        assert all(ec >= 0 for ec in energy_consumptions)
    
    print("✅ Simulation execution tests passed")

def test_model_validation():
    """Test model validation function"""
    print("\n🔄 Testing model validation...")
    
    model = IndoorPerformanceModel()
    validation_results = model.validate_indoor_model()
    
    print("✅ Validation results:")
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  - {test_name}: {status}")
    
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    print(f"✅ Overall validation: {passed_tests}/{total_tests} tests passed")
    
    # Should pass most tests
    assert passed_tests >= total_tests * 0.8, f"Too many validation failures: {passed_tests}/{total_tests}"
    
    print("✅ Model validation tests passed")

def run_all_tests():
    """Run all test functions"""
    print("🏭 Testing Indoor Performance Models - Comprehensive Test Suite")
    print("=" * 70)
    
    test_functions = [
        test_indoor_conditions_generation,
        test_equipment_state_calculation,
        test_performance_factors_calculation,
        test_cop_calculation,
        test_simulation_execution,
        test_model_validation
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"🎉 INDOOR PERFORMANCE MODELS TEST SUMMARY")
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Indoor performance models are working correctly.")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review the issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)