#!/usr/bin/env python3
"""
Comprehensive test script for Fault Injection Models
Tests all major fault injection and progression functionality
"""

import sys
sys.path.append('src')

def test_fault_injection_models():
    """Test all fault injection model functionality"""
    print("🚨 Testing Fault Injection Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from fault_injection_models import (
            FaultInjectionModel, FaultState, SensorFault, FaultType, FaultSeverity
        )
        import numpy as np
        import pandas as pd
        from datetime import datetime
        print("✅ Import successful")
        
        # Test FaultInjectionModel initialization
        print("\n🔄 Testing FaultInjectionModel initialization...")
        fault_model = FaultInjectionModel()
        print("✅ FaultInjectionModel initialized successfully")
        
        # Test fault progression models
        print("\n🔄 Testing fault progression models...")
        fault_progression = fault_model.fault_progression_models
        print(f"✅ Loaded fault progression models for {len(fault_progression)} fault types:")
        for fault_type, params in fault_progression.items():
            print(f"  - {fault_type.value}: repair cost base = ${params.get('repair_cost_base', 'N/A')}")
            if 'initial_rate_percent_per_year' in params:
                rates = params['initial_rate_percent_per_year']
                print(f"    * Initial rates: {rates[0]:.1f}% - {rates[-1]:.1f}% per year")
        
        # Test sensor fault models
        print("\n🔄 Testing sensor fault models...")
        sensor_fault_models = fault_model.sensor_fault_models
        print(f"✅ Loaded sensor fault models for {len(sensor_fault_models)} sensor types:")
        for sensor_type, params in sensor_fault_models.items():
            print(f"  - {sensor_type}: {len(params)} parameter types")
        
        # Test fault interactions
        print("\n🔄 Testing fault interactions...")
        fault_interactions = fault_model.fault_interactions
        print(f"✅ Loaded fault interactions: {len(fault_interactions)} interaction pairs")
        for (fault1, fault2), factor in fault_interactions.items():
            print(f"  - {fault1.value} + {fault2.value}: acceleration factor = {factor:.1f}")
        
        # Test detection algorithms
        print("\n🔄 Testing detection algorithms...")
        detection_algorithms = fault_model.detection_algorithms
        print(f"✅ Loaded detection algorithms: {len(detection_algorithms)} algorithms")
        for algorithm, params in detection_algorithms.items():
            print(f"  - {algorithm}: sensitivity = {params['sensitivity']:.2f}, FPR = {params['false_positive_rate']:.3f}")
        
        # Test fault injection - Refrigerant leak
        print("\n🔄 Testing refrigerant leak fault injection...")
        leak_fault = fault_model.inject_fault(
            system_id="test_system_001",
            component_id="compressor_001",
            fault_type=FaultType.REFRIGERANT_LEAK,
            severity=FaultSeverity.MODERATE
        )
        print(f"✅ Created refrigerant leak fault: {type(leak_fault)}")
        print(f"✅ Fault ID: {leak_fault.fault_id}")
        print(f"✅ Component: {leak_fault.component_id}")
        print(f"✅ Fault type: {leak_fault.fault_type.value}")
        print(f"✅ Severity: {leak_fault.severity.value}")
        print(f"✅ Progression rate: {leak_fault.progression_rate:.6f}")
        print(f"✅ Current magnitude: {leak_fault.current_magnitude:.3f}")
        print(f"✅ Capacity impact: {leak_fault.capacity_impact_factor:.3f}")
        print(f"✅ COP impact: {leak_fault.cop_impact_factor:.3f}")
        print(f"✅ Power impact: {leak_fault.power_impact_factor:.3f}")
        print(f"✅ Detectable: {leak_fault.detectable}")
        print(f"✅ Detection confidence: {leak_fault.detection_confidence:.3f}")
        print(f"✅ Repair urgency: {leak_fault.repair_urgency}")
        print(f"✅ Repair cost: ${leak_fault.estimated_repair_cost:.0f}")
        print(f"✅ Repair time: {leak_fault.estimated_repair_time_hours:.1f} hours")
        
        # Test fault injection - Sensor drift
        print("\n🔄 Testing sensor drift fault injection...")
        sensor_drift_fault = fault_model.inject_fault(
            system_id="test_system_001",
            component_id="temp_sensor_evap",
            fault_type=FaultType.SENSOR_DRIFT,
            severity=FaultSeverity.MINOR
        )
        print(f"✅ Created sensor drift fault: {type(sensor_drift_fault)}")
        print(f"✅ Fault type: {sensor_drift_fault.fault_type.value}")
        print(f"✅ Severity: {sensor_drift_fault.severity.value}")
        print(f"✅ Fault parameters: {len(sensor_drift_fault.fault_parameters)} items")
        for key, value in sensor_drift_fault.fault_parameters.items():
            print(f"  - {key}: {value}")
        
        # Test fault injection - Compressor failure
        print("\n🔄 Testing compressor failure fault injection...")
        compressor_fault = fault_model.inject_fault(
            system_id="test_system_001",
            component_id="compressor_001",
            fault_type=FaultType.COMPRESSOR_FAILURE,
            severity=FaultSeverity.SEVERE
        )
        print(f"✅ Created compressor failure fault: {type(compressor_fault)}")
        print(f"✅ Severity: {compressor_fault.severity.value}")
        print(f"✅ Current magnitude: {compressor_fault.current_magnitude:.3f}")
        print(f"✅ Capacity impact: {compressor_fault.capacity_impact_factor:.3f}")
        print(f"✅ Repair urgency: {compressor_fault.repair_urgency}")
        
        # Test fault injection - Fan failure
        print("\n🔄 Testing fan failure fault injection...")
        fan_fault = fault_model.inject_fault(
            system_id="test_system_001",
            component_id="evaporator_fan_001",
            fault_type=FaultType.FAN_FAILURE,
            severity=FaultSeverity.MODERATE
        )
        print(f"✅ Created fan failure fault: {type(fan_fault)}")
        print(f"✅ Fault parameters: {fan_fault.fault_parameters}")
        
        # Test fault injection - Fouling accelerated
        print("\n🔄 Testing accelerated fouling fault injection...")
        fouling_fault = fault_model.inject_fault(
            system_id="test_system_001",
            component_id="evaporator_001",
            fault_type=FaultType.FOULING_ACCELERATED,
            severity=FaultSeverity.MODERATE
        )
        print(f"✅ Created accelerated fouling fault: {type(fouling_fault)}")
        print(f"✅ Acceleration factor: {fouling_fault.fault_parameters.get('acceleration_factor', 'N/A')}")
        
        # Test fault injection - All fault types
        print("\n🔄 Testing all fault types...")
        all_fault_types = list(FaultType)
        all_severities = list(FaultSeverity)
        
        for i, fault_type in enumerate(all_fault_types[:5]):  # Test first 5 types
            severity = all_severities[i % len(all_severities)]
            test_fault = fault_model.inject_fault(
                f"test_system_{i:03d}",
                f"component_{i:03d}",
                fault_type,
                severity
            )
            print(f"✅ {fault_type.value} ({severity.value}): impact factors = {test_fault.capacity_impact_factor:.3f}/{test_fault.cop_impact_factor:.3f}/{test_fault.power_impact_factor:.3f}")
        
        # Test fault progression
        print("\n🔄 Testing fault progression...")
        
        # Define operating conditions
        normal_conditions = {
            'average_temperature': 298.15,  # 25°C
            'load_factor': 1.0,
            'cycles_per_hour': 2
        }
        
        harsh_conditions = {
            'average_temperature': 333.15,  # 60°C (high temperature)
            'load_factor': 1.3,             # 30% overload
            'cycles_per_hour': 8            # Excessive cycling
        }
        
        # Progress leak fault under normal conditions
        original_magnitude = leak_fault.current_magnitude
        original_capacity = leak_fault.capacity_impact_factor
        
        progressed_normal = fault_model.progress_fault(leak_fault, 100.0, normal_conditions)
        print(f"✅ Normal conditions (100 hours):")
        print(f"  - Magnitude: {original_magnitude:.3f} → {progressed_normal.current_magnitude:.3f}")
        print(f"  - Capacity impact: {original_capacity:.3f} → {progressed_normal.capacity_impact_factor:.3f}")
        print(f"  - Detection confidence: {progressed_normal.detection_confidence:.3f}")
        print(f"  - Repair urgency: {progressed_normal.repair_urgency}")
        
        # Progress fault under harsh conditions
        progressed_harsh = fault_model.progress_fault(leak_fault, 100.0, harsh_conditions)
        print(f"✅ Harsh conditions (100 hours):")
        print(f"  - Magnitude: {progressed_normal.current_magnitude:.3f} → {progressed_harsh.current_magnitude:.3f}")
        print(f"  - Capacity impact: {progressed_harsh.capacity_impact_factor:.3f}")
        print(f"  - Detection confidence: {progressed_harsh.detection_confidence:.3f}")
        print(f"  - Repair urgency: {progressed_harsh.repair_urgency}")
        
        # Test sensor fault injection
        print("\n🔄 Testing sensor fault injection...")
        
        # Test different sensor types
        sensor_types = ['temperature_sensor', 'pressure_sensor', 'flow_sensor', 'power_sensor']
        severities = [FaultSeverity.MINOR, FaultSeverity.MODERATE, FaultSeverity.SEVERE, FaultSeverity.CRITICAL]
        
        for sensor_type, severity in zip(sensor_types, severities):
            sensor_fault = fault_model.inject_sensor_fault(
                f"sensor_{sensor_type}_{severity.value}",
                sensor_type,
                severity
            )
            print(f"✅ {sensor_type} ({severity.value}):")
            print(f"  - Fault type: {sensor_fault.fault_type}")
            print(f"  - Drift rate: {sensor_fault.drift_rate:.6f}")
            print(f"  - Noise factor: {sensor_fault.noise_increase_factor:.2f}")
            print(f"  - Bias offset: {sensor_fault.bias_offset:.3f}")
            print(f"  - Scale error: {sensor_fault.scale_factor_error:.3f}")
            print(f"  - Intermittent rate: {sensor_fault.intermittent_failure_rate:.6f}")
        
        # Test sensor fault application
        print("\n🔄 Testing sensor fault application...")
        
        # Create a test sensor fault
        test_sensor_fault = fault_model.inject_sensor_fault(
            "test_temp_sensor",
            "temperature_sensor",
            FaultSeverity.MODERATE
        )
        
        # Apply fault to measurements
        true_values = [25.0, 26.5, 24.8, 27.2, 25.5]  # °C
        time_hours = 10.0
        
        print(f"✅ Applying {test_sensor_fault.fault_type} fault to temperature measurements:")
        for i, true_value in enumerate(true_values):
            faulty_value = fault_model.apply_sensor_fault(true_value, test_sensor_fault, time_hours)
            if not np.isnan(faulty_value):
                print(f"  - Measurement {i+1}: {true_value:.1f}°C → {faulty_value:.1f}°C (error: {faulty_value-true_value:+.1f}°C)")
            else:
                print(f"  - Measurement {i+1}: {true_value:.1f}°C → SENSOR FAILURE")
        
        # Test fault detection
        print("\n🔄 Testing fault detection...")
        
        # Generate test system data with some anomalies
        np.random.seed(42)  # For reproducible results
        normal_data = np.random.normal(25.0, 1.0, 50)  # Normal temperature data
        anomaly_data = np.concatenate([normal_data, [30.0, 31.0, 29.5, 32.0]])  # Add anomalies
        
        system_data = {
            'evaporator_temperature': anomaly_data.tolist(),
            'condenser_temperature': np.random.normal(45.0, 2.0, 54).tolist(),
            'compressor_power': np.random.normal(2.5, 0.2, 54).tolist()
        }
        
        # Test different detection algorithms
        detection_algorithms = ['statistical_process_control', 'model_based_detection', 'machine_learning_detection']
        
        for algorithm in detection_algorithms:
            detection_results = fault_model.detect_fault(system_data, algorithm)
            print(f"✅ {algorithm}:")
            print(f"  - Faults detected: {len(detection_results['faults_detected'])}")
            print(f"  - False alarms: {len(detection_results['false_alarms'])}")
            print(f"  - Detection delay: {detection_results['detection_delay_hours']:.1f} hours")
            
            for fault in detection_results['faults_detected']:
                print(f"    * {fault['parameter']}: {fault['fault_type']} (confidence: {fault['confidence']:.2f})")
        
        # Test sensor type determination
        print("\n🔄 Testing sensor type determination...")
        
        test_component_ids = [
            'temp_sensor_evap', 'temperature_ambient', 'press_sensor_high',
            'pressure_low', 'flow_meter_001', 'power_meter_comp', 'current_sensor'
        ]
        
        for component_id in test_component_ids:
            sensor_type = fault_model._determine_sensor_type(component_id)
            print(f"✅ {component_id} → {sensor_type}")
        
        # Test performance impact calculations
        print("\n🔄 Testing performance impact calculations...")
        
        test_cases = [
            (FaultType.REFRIGERANT_LEAK, FaultSeverity.MODERATE, 0.3),
            (FaultType.COMPRESSOR_FAILURE, FaultSeverity.SEVERE, 0.6),
            (FaultType.FAN_FAILURE, FaultSeverity.MINOR, 0.2),
            (FaultType.FOULING_ACCELERATED, FaultSeverity.CRITICAL, 0.8)
        ]
        
        for fault_type, severity, magnitude in test_cases:
            capacity, cop, power = fault_model._calculate_performance_impacts(
                fault_type, severity, magnitude, {}
            )
            print(f"✅ {fault_type.value} ({severity.value}, mag={magnitude:.1f}):")
            print(f"  - Capacity: {capacity:.3f}, COP: {cop:.3f}, Power: {power:.3f}")
        
        # Test detectability calculations
        print("\n🔄 Testing detectability calculations...")
        
        for fault_type, severity, magnitude in test_cases:
            detectable, confidence, fpr = fault_model._calculate_detectability(
                fault_type, severity, magnitude
            )
            print(f"✅ {fault_type.value} ({severity.value}, mag={magnitude:.1f}):")
            print(f"  - Detectable: {detectable}, Confidence: {confidence:.3f}, FPR: {fpr:.3f}")
        
        # Test repair urgency determination
        print("\n🔄 Testing repair urgency determination...")
        
        urgency_test_cases = [
            (FaultType.REFRIGERANT_LEAK, FaultSeverity.MINOR, 0.1),
            (FaultType.COMPRESSOR_FAILURE, FaultSeverity.MODERATE, 0.4),
            (FaultType.FAN_FAILURE, FaultSeverity.SEVERE, 0.7),
            (FaultType.FOULING_ACCELERATED, FaultSeverity.CRITICAL, 0.9)
        ]
        
        for fault_type, severity, magnitude in urgency_test_cases:
            urgency = fault_model._determine_repair_urgency(fault_type, severity, magnitude)
            print(f"✅ {fault_type.value} ({severity.value}, mag={magnitude:.1f}): {urgency}")
        
        # Test fault scenario generation
        print("\n🔄 Testing fault scenario generation...")
        
        # Generate short scenario
        short_scenario = fault_model.generate_fault_scenario(
            system_id="test_scenario_001",
            duration_hours=100.0,
            fault_probability_per_hour=0.01  # Higher probability for testing
        )
        print(f"✅ Short scenario (100 hours): {len(short_scenario)} faults")
        
        if short_scenario:
            fault_types = [f.fault_type.value for f in short_scenario]
            severities = [f.severity.value for f in short_scenario]
            print(f"✅ Fault types: {set(fault_types)}")
            print(f"✅ Severities: {set(severities)}")
        
        # Generate longer scenario
        long_scenario = fault_model.generate_fault_scenario(
            system_id="test_scenario_002",
            duration_hours=8760.0,  # 1 year
            fault_probability_per_hour=0.0001  # Realistic probability
        )
        print(f"✅ Long scenario (1 year): {len(long_scenario)} faults")
        
        # Test data saving
        print("\n🔄 Testing data saving...")
        
        test_faults = [leak_fault, sensor_drift_fault, compressor_fault, fan_fault, fouling_fault]
        fault_model.save_fault_data(test_faults, "test_fault_injection_data.json")
        print("✅ Fault injection data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test FaultState
        test_fault_state = FaultState(
            fault_id="test_fault_001",
            component_id="test_component",
            fault_type=FaultType.REFRIGERANT_LEAK,
            severity=FaultSeverity.MODERATE,
            onset_time=datetime.now().isoformat(),
            progression_rate=0.001,
            current_magnitude=0.25,
            fault_parameters={'leak_rate': 0.001},
            capacity_impact_factor=0.85,
            cop_impact_factor=0.90,
            power_impact_factor=1.15,
            detectable=True,
            detection_confidence=0.75,
            false_positive_rate=0.03,
            repair_urgency="medium",
            estimated_repair_cost=450.0,
            estimated_repair_time_hours=4.5
        )
        print(f"✅ FaultState created: {test_fault_state.fault_id}")
        print(f"✅ Fault type: {test_fault_state.fault_type.value}")
        print(f"✅ Severity: {test_fault_state.severity.value}")
        
        # Test SensorFault
        test_sensor_fault_state = SensorFault(
            sensor_id="test_sensor_001",
            sensor_type="temperature_sensor",
            fault_type="drift",
            drift_rate=0.0001,
            noise_increase_factor=1.5,
            bias_offset=1.0,
            scale_factor_error=0.02,
            intermittent_failure_rate=0.01
        )
        print(f"✅ SensorFault created: {test_sensor_fault_state.sensor_id}")
        print(f"✅ Sensor type: {test_sensor_fault_state.sensor_type}")
        print(f"✅ Fault type: {test_sensor_fault_state.fault_type}")
        
        # Test enum functionality
        print("\n🔄 Testing enum functionality...")
        
        print(f"✅ FaultType enum: {len(list(FaultType))} types")
        for fault_type in list(FaultType)[:5]:  # Show first 5
            print(f"  - {fault_type.name}: {fault_type.value}")
        
        print(f"✅ FaultSeverity enum: {len(list(FaultSeverity))} levels")
        for severity in list(FaultSeverity):
            print(f"  - {severity.name}: {severity.value}")
        
        print("\n🎉 ALL FAULT INJECTION TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in fault injection models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fault_injection_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)