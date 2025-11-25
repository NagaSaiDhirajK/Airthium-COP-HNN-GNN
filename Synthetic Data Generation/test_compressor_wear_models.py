#!/usr/bin/env python3
"""
Comprehensive test script for Compressor Wear Models
Tests all major compressor wear and degradation functionality
"""

import sys
sys.path.append('src')

def test_compressor_wear_models():
    """Test all compressor wear model functionality"""
    print("🔧 Testing Compressor Wear Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from compressor_wear_models import (
            CompressorWearModel, CompressorWearState, WearProgression
        )
        import numpy as np
        import pandas as pd
        print("✅ Import successful")
        
        # Test CompressorWearModel initialization
        print("\n🔄 Testing CompressorWearModel initialization...")
        wear_model = CompressorWearModel()
        print("✅ CompressorWearModel initialized successfully")
        
        # Test wear parameters
        print("\n🔄 Testing wear parameters...")
        wear_params = wear_model.wear_parameters
        print(f"✅ Loaded wear parameters for {len(wear_params)} compressor types:")
        for comp_type, params in wear_params.items():
            print(f"  - {comp_type}: {len(params)} wear categories")
            for wear_type, values in params.items():
                print(f"    * {wear_type}: initial rate = {values['initial_rate']:.1f}%/1000h")
        
        # Test performance impact parameters
        print("\n🔄 Testing performance impact parameters...")
        perf_impacts = wear_model.performance_impacts
        print(f"✅ Loaded performance impacts for {len(perf_impacts)} categories:")
        for impact_type, coeffs in perf_impacts.items():
            print(f"  - {impact_type}: bearing coeff = {coeffs['bearing_coefficient']:.1f}")
        
        # Test condition thresholds
        print("\n🔄 Testing condition monitoring thresholds...")
        thresholds = wear_model.condition_thresholds
        print(f"✅ Loaded condition thresholds for {len(thresholds)} parameters:")
        for param, levels in thresholds.items():
            print(f"  - {param}: good < {levels['good']:.1f}, unacceptable > {levels['unacceptable']:.1f}")
        
        # Test wear state initialization - Scroll compressor
        print("\n🔄 Testing scroll compressor wear state initialization...")
        scroll_wear = wear_model.initialize_wear_state(
            component_id="test_scroll_001",
            compressor_type="scroll_compressor",
            age_hours=0
        )
        print(f"✅ Created scroll compressor wear state: {type(scroll_wear)}")
        print(f"✅ Initial bearing wear: {scroll_wear.bearing_wear_percentage:.2f}%")
        print(f"✅ Initial valve leakage: {scroll_wear.valve_leakage_percentage:.2f}%")
        print(f"✅ Initial oil quality: {scroll_wear.oil_quality_index:.3f}")
        print(f"✅ Initial efficiency degradation: {scroll_wear.efficiency_degradation:.2f}%")
        print(f"✅ Initial vibration level: {scroll_wear.vibration_level:.2f} mm/s")
        print(f"✅ Initial noise level: {scroll_wear.noise_level:.2f} dB")
        
        # Test wear state initialization - Reciprocating compressor
        print("\n🔄 Testing reciprocating compressor wear state initialization...")
        recip_wear = wear_model.initialize_wear_state(
            component_id="test_recip_001",
            compressor_type="reciprocating_compressor",
            age_hours=5000  # 5000 hours old
        )
        print(f"✅ Created reciprocating compressor wear state: {type(recip_wear)}")
        print(f"✅ Age: {recip_wear.age_hours} hours")
        print(f"✅ Bearing wear: {recip_wear.bearing_wear_percentage:.2f}%")
        print(f"✅ Valve leakage: {recip_wear.valve_leakage_percentage:.2f}%")
        print(f"✅ Oil quality: {recip_wear.oil_quality_index:.3f}")
        print(f"✅ Bearing clearance: {recip_wear.bearing_clearance:.3f} mm")
        print(f"✅ Suction valve condition: {recip_wear.suction_valve_condition:.3f}")
        print(f"✅ Discharge valve condition: {recip_wear.discharge_valve_condition:.3f}")
        
        # Test wear state updates
        print("\n🔄 Testing wear state updates...")
        
        # Define operating conditions
        operating_conditions = {
            'discharge_temperature': 363.15,  # 90°C (high temperature)
            'load_factor': 1.2,              # 20% overload
            'pressure_ratio': 4.5,           # High pressure ratio
            'start_stop_cycles': 100         # 100 cycles
        }
        
        # Update scroll compressor after 1000 hours
        scroll_wear_updated = wear_model.update_wear_state(
            scroll_wear, 1000, operating_conditions
        )
        print(f"✅ Updated scroll compressor after 1000 hours")
        print(f"✅ New age: {scroll_wear_updated.age_hours} hours")
        print(f"✅ New bearing wear: {scroll_wear_updated.bearing_wear_percentage:.2f}%")
        print(f"✅ New valve leakage: {scroll_wear_updated.valve_leakage_percentage:.2f}%")
        print(f"✅ New oil quality: {scroll_wear_updated.oil_quality_index:.3f}")
        print(f"✅ New efficiency degradation: {scroll_wear_updated.efficiency_degradation:.2f}%")
        print(f"✅ New capacity degradation: {scroll_wear_updated.capacity_degradation:.2f}%")
        print(f"✅ New power increase: {scroll_wear_updated.power_increase:.2f}%")
        print(f"✅ New vibration level: {scroll_wear_updated.vibration_level:.2f} mm/s")
        print(f"✅ Fault indicators: {len(scroll_wear_updated.fault_indicators)} items")
        
        # Update reciprocating compressor
        recip_wear_updated = wear_model.update_wear_state(
            recip_wear, 2000, operating_conditions
        )
        print(f"✅ Updated reciprocating compressor after 2000 hours")
        print(f"✅ Total age: {recip_wear_updated.age_hours} hours")
        print(f"✅ Bearing wear: {recip_wear_updated.bearing_wear_percentage:.2f}%")
        print(f"✅ Oil contamination: {recip_wear_updated.oil_contamination_level:.1f} ppm")
        
        # Test maintenance event simulation
        print("\n🔄 Testing maintenance event simulation...")
        
        # Test different maintenance types
        maintenance_types = ['oil_change', 'filter_replacement', 'valve_replacement', 
                           'bearing_replacement', 'major_overhaul', 'routine_maintenance']
        
        for maintenance_type in maintenance_types:
            # Create a copy for testing
            test_wear = wear_model.initialize_wear_state(
                "test_maintenance", "scroll_compressor", 10000
            )
            
            # Add some wear first
            test_wear = wear_model.update_wear_state(test_wear, 5000, operating_conditions)
            
            # Record pre-maintenance state
            pre_bearing = test_wear.bearing_wear_percentage
            pre_valve = test_wear.valve_leakage_percentage
            pre_oil = test_wear.oil_quality_index
            pre_efficiency = test_wear.efficiency_degradation
            
            # Simulate maintenance
            maintained_wear = wear_model.simulate_maintenance_event(test_wear, maintenance_type)
            
            print(f"✅ {maintenance_type}:")
            print(f"  - Bearing wear: {pre_bearing:.2f}% → {maintained_wear.bearing_wear_percentage:.2f}%")
            print(f"  - Valve leakage: {pre_valve:.2f}% → {maintained_wear.valve_leakage_percentage:.2f}%")
            print(f"  - Oil quality: {pre_oil:.3f} → {maintained_wear.oil_quality_index:.3f}")
            print(f"  - Efficiency degradation: {pre_efficiency:.2f}% → {maintained_wear.efficiency_degradation:.2f}%")
            print(f"  - Maintenance events: {len(maintained_wear.maintenance_events)}")
            if maintained_wear.maintenance_events:
                cost = maintained_wear.maintenance_events[-1]['cost_estimate']
                print(f"  - Estimated cost: ${cost:.0f}")
        
        # Test private calculation methods
        print("\n🔄 Testing calculation methods...")
        
        # Test temperature wear factor
        temp_factor_normal = wear_model._calculate_temperature_wear_factor(353.15)  # 80°C
        temp_factor_high = wear_model._calculate_temperature_wear_factor(373.15)    # 100°C
        print(f"✅ Temperature wear factor (80°C): {temp_factor_normal:.3f}")
        print(f"✅ Temperature wear factor (100°C): {temp_factor_high:.3f}")
        
        # Test load wear factor
        load_factor_normal = wear_model._calculate_load_wear_factor(1.0)
        load_factor_overload = wear_model._calculate_load_wear_factor(1.3)
        load_factor_partload = wear_model._calculate_load_wear_factor(0.7)
        print(f"✅ Load wear factor (100%): {load_factor_normal:.3f}")
        print(f"✅ Load wear factor (130%): {load_factor_overload:.3f}")
        print(f"✅ Load wear factor (70%): {load_factor_partload:.3f}")
        
        # Test pressure wear factor
        pressure_factor_low = wear_model._calculate_pressure_wear_factor(2.5)
        pressure_factor_high = wear_model._calculate_pressure_wear_factor(5.0)
        print(f"✅ Pressure wear factor (2.5): {pressure_factor_low:.3f}")
        print(f"✅ Pressure wear factor (5.0): {pressure_factor_high:.3f}")
        
        # Test cycling wear factor
        cycling_factor_low = wear_model._calculate_cycling_wear_factor(50)
        cycling_factor_high = wear_model._calculate_cycling_wear_factor(500)
        print(f"✅ Cycling wear factor (50 cycles): {cycling_factor_low:.3f}")
        print(f"✅ Cycling wear factor (500 cycles): {cycling_factor_high:.3f}")
        
        # Test performance degradation calculations
        bearing_wear_test = 20.0  # 20% bearing wear
        valve_leakage_test = 15.0  # 15% valve leakage
        oil_degradation_test = 30.0  # 30% oil degradation
        
        efficiency_deg = wear_model._calculate_efficiency_degradation(
            bearing_wear_test, valve_leakage_test, oil_degradation_test
        )
        capacity_deg = wear_model._calculate_capacity_degradation(
            bearing_wear_test, valve_leakage_test, oil_degradation_test
        )
        power_inc = wear_model._calculate_power_increase(
            bearing_wear_test, valve_leakage_test, oil_degradation_test
        )
        
        print(f"✅ Efficiency degradation (20%/15%/30% wear): {efficiency_deg:.2f}%")
        print(f"✅ Capacity degradation (20%/15%/30% wear): {capacity_deg:.2f}%")
        print(f"✅ Power increase (20%/15%/30% wear): {power_inc:.2f}%")
        
        # Test condition indicator calculations
        vibration = wear_model._calculate_vibration_level(bearing_wear_test, valve_leakage_test)
        noise = wear_model._calculate_noise_level(bearing_wear_test, valve_leakage_test)
        temp_rise = wear_model._calculate_temperature_rise(bearing_wear_test, 0.7)  # 70% oil quality
        
        print(f"✅ Vibration level (20%/15% wear): {vibration:.2f} mm/s")
        print(f"✅ Noise level (20%/15% wear): {noise:.2f} dB")
        print(f"✅ Temperature rise (20% wear, 70% oil): {temp_rise:.2f} K")
        
        # Test fault indicators calculation
        fault_indicators = wear_model._calculate_fault_indicators(scroll_wear_updated)
        print(f"✅ Fault indicators calculated: {len(fault_indicators)} items")
        for indicator, value in fault_indicators.items():
            if 'probability' in indicator:
                print(f"  - {indicator}: {value:.3f}")
            else:
                print(f"  - {indicator}: {value:.0f} hours")
        
        # Test remaining life estimation
        remaining_life = wear_model._estimate_remaining_life(scroll_wear_updated)
        print(f"✅ Remaining useful life: {remaining_life:.0f} hours")
        
        # Test maintenance cost estimation
        for maint_type in maintenance_types:
            cost = wear_model._estimate_maintenance_cost(maint_type)
            print(f"✅ {maint_type} cost: ${cost:.0f}")
        
        # Test wear schedule generation
        print("\n🔄 Testing wear schedule generation...")
        
        # Generate schedule for scroll compressor
        scroll_schedule = wear_model.generate_wear_schedule(
            component_id="test_scroll_001",
            compressor_type="scroll_compressor",
            simulation_hours=43800  # 5 years
        )
        print(f"✅ Generated scroll compressor schedule: {len(scroll_schedule)} events")
        if scroll_schedule:
            print(f"✅ First event at: {scroll_schedule[0]['time_hours']} hours")
            print(f"✅ Event type: {scroll_schedule[0]['maintenance_type']}")
        
        # Generate schedule for reciprocating compressor
        recip_schedule = wear_model.generate_wear_schedule(
            component_id="test_recip_001",
            compressor_type="reciprocating_compressor",
            simulation_hours=35040  # 4 years
        )
        print(f"✅ Generated reciprocating compressor schedule: {len(recip_schedule)} events")
        
        # Test data saving
        print("\n🔄 Testing data saving...")
        
        wear_states = [scroll_wear_updated, recip_wear_updated]
        wear_model.save_wear_data(wear_states, "test_compressor_wear_data.json")
        print("✅ Compressor wear data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test CompressorWearState
        test_wear_state = CompressorWearState(
            component_id="test_wear",
            compressor_type="scroll_compressor",
            age_hours=15000,
            bearing_wear_percentage=25.0,
            bearing_clearance=0.075,
            bearing_temperature_rise=12.0,
            valve_leakage_percentage=18.0,
            suction_valve_condition=0.64,
            discharge_valve_condition=0.70,
            oil_quality_index=0.75,
            oil_viscosity_change=-5.0,
            oil_contamination_level=25.0,
            efficiency_degradation=8.5,
            capacity_degradation=4.2,
            power_increase=6.1,
            vibration_level=8.5,
            noise_level=12.0,
            discharge_temperature_rise=15.0,
            maintenance_events=[],
            fault_indicators={}
        )
        print(f"✅ CompressorWearState created: {test_wear_state.component_id}")
        print(f"✅ Age: {test_wear_state.age_hours} hours")
        print(f"✅ Bearing wear: {test_wear_state.bearing_wear_percentage}%")
        
        # Test WearProgression
        test_wear_progression = WearProgression(
            component="bearing",
            wear_type="mechanical",
            initial_rate=0.5,
            acceleration_factor=1.2,
            failure_threshold=80.0,
            maintenance_effectiveness=0.3
        )
        print(f"✅ WearProgression created: {test_wear_progression.component}")
        print(f"✅ Wear type: {test_wear_progression.wear_type}")
        print(f"✅ Initial rate: {test_wear_progression.initial_rate}%/1000h")
        
        print("\n🎉 ALL COMPRESSOR WEAR TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in compressor wear models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_compressor_wear_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)