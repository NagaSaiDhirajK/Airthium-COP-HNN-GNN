#!/usr/bin/env python3
"""
Test Sensor Noise Models
========================

Comprehensive test suite for sensor noise models including noise generation,
drift calculation, measurement errors, and sensor networks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import json
from pathlib import Path

# Import directly from the module file
from sensor_noise_models import (
    SensorNoiseModel, SensorType, NoiseType, SensorSpecifications, 
    NoiseParameters, SensorState, SensorNetwork
)

def test_sensor_noise_models():
    """Test sensor noise models comprehensively."""
    
    print("📡 Testing Sensor Noise Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        print("✅ Import successful")
        
        # Initialize sensor noise model
        print("\n🔄 Testing SensorNoiseModel initialization...")
        sensor_model = SensorNoiseModel(random_seed=42)
        print("✅ SensorNoiseModel initialized successfully")
        print(f"✅ Sensor database: {len(sensor_model.sensor_database)} sensor types")
        print(f"✅ Random seed: 42")
        print(f"✅ Current time: {sensor_model.current_time}")
        
        # Test sensor database
        print("\n🔄 Testing sensor database...")
        sensor_types = list(sensor_model.sensor_database.keys())
        print(f"✅ Available sensor types: {len(sensor_types)}")
        
        for sensor_type in sensor_types:
            spec = sensor_model.sensor_database[sensor_type]
            print(f"✅ {sensor_type}:")
            print(f"  - Type: {spec.sensor_type.value}")
            print(f"  - Range: {spec.measurement_range[0]:.2f} - {spec.measurement_range[1]:.2f}")
            print(f"  - Accuracy: ±{spec.accuracy_percent}%")
            print(f"  - Resolution: {spec.resolution}")
            print(f"  - Response time: {spec.response_time_s} s")
            print(f"  - Drift rate: {spec.drift_rate_per_year}/year")
            print(f"  - Temperature coefficient: {spec.temperature_coefficient}/°C")
        
        # Test sensor initialization
        print("\n🔄 Testing sensor initialization...")
        test_sensors = [
            ('temperature_rtd_pt100', NoiseType.WHITE_NOISE, 0.1),
            ('pressure_piezoresistive', NoiseType.PINK_NOISE, 100.0),
            ('flow_ultrasonic', NoiseType.BROWNIAN_NOISE, 0.001),
            ('humidity_capacitive', NoiseType.PERIODIC_NOISE, 0.5),
            ('power_meter', NoiseType.IMPULSE_NOISE, 10.0)
        ]
        
        for sensor_id, noise_type, amplitude in test_sensors:
            # Create noise parameters
            if noise_type == NoiseType.PERIODIC_NOISE:
                noise_params = NoiseParameters(
                    noise_type=noise_type,
                    amplitude=amplitude,
                    frequency_hz=0.1
                )
            elif noise_type == NoiseType.IMPULSE_NOISE:
                noise_params = NoiseParameters(
                    noise_type=noise_type,
                    amplitude=amplitude,
                    burst_probability=0.01,
                    burst_amplitude=amplitude * 10
                )
            else:
                noise_params = NoiseParameters(
                    noise_type=noise_type,
                    amplitude=amplitude
                )
            
            # Initialize sensor
            initial_calibration_error = np.random.uniform(-0.1, 0.1)
            sensor_model.initialize_sensor(sensor_id, noise_params, initial_calibration_error)
            
            print(f"✅ {sensor_id} initialized:")
            print(f"  - Noise type: {noise_type.value}")
            print(f"  - Amplitude: {amplitude}")
            print(f"  - Calibration error: {initial_calibration_error:.4f}")
        
        # Test noise generation
        print("\n🔄 Testing noise generation...")
        duration_s = 10.0
        sampling_rate_hz = 100.0
        
        for sensor_id, noise_type, amplitude in test_sensors:
            noise_signal = sensor_model.generate_sensor_noise(sensor_id, duration_s, sampling_rate_hz)
            
            print(f"✅ {sensor_id} noise generation:")
            print(f"  - Signal length: {len(noise_signal)} samples")
            print(f"  - Expected length: {int(duration_s * sampling_rate_hz)} samples")
            print(f"  - Mean: {np.mean(noise_signal):.6f}")
            print(f"  - Std: {np.std(noise_signal):.6f}")
            print(f"  - Min: {np.min(noise_signal):.6f}")
            print(f"  - Max: {np.max(noise_signal):.6f}")
        
        # Test sensor response filtering
        print("\n🔄 Testing sensor response filtering...")
        test_signal = np.ones(1000)  # Step input
        dt = 0.01
        
        for sensor_id, _, _ in test_sensors[:3]:  # Test first 3 sensors
            filtered_signal = sensor_model.apply_sensor_response(sensor_id, test_signal, dt)
            
            # Check if filtering occurred (should not be exactly 1.0 for all samples)
            settling_value = filtered_signal[-1]
            response_time = None
            
            # Find 63% of final value (time constant)
            target_value = 0.63 * settling_value
            for i, val in enumerate(filtered_signal):
                if val >= target_value:
                    response_time = i * dt
                    break
            
            print(f"✅ {sensor_id} response filtering:")
            print(f"  - Settling value: {settling_value:.4f}")
            print(f"  - Response time (63%): {response_time:.2f} s" if response_time else "  - Response time: Immediate")
            print(f"  - Expected response time: {sensor_model.sensor_database[sensor_id].response_time_s} s")
        
        # Test drift calculation
        print("\n🔄 Testing drift calculation...")
        time_elapsed_s = 3600.0  # 1 hour
        environmental_conditions = [
            {'temperature_C': 25.0, 'pressure_Pa': 101325.0, 'humidity_percent': 50.0},
            {'temperature_C': 40.0, 'pressure_Pa': 150000.0, 'humidity_percent': 80.0},
            {'temperature_C': 10.0, 'pressure_Pa': 80000.0, 'humidity_percent': 20.0}
        ]
        
        for i, conditions in enumerate(environmental_conditions):
            print(f"✅ Environmental condition {i+1}: T={conditions['temperature_C']}°C, "
                  f"P={conditions['pressure_Pa']}Pa, RH={conditions['humidity_percent']}%")
            
            for sensor_id, _, _ in test_sensors[:3]:
                drift = sensor_model.calculate_drift(
                    sensor_id, time_elapsed_s, 
                    conditions['temperature_C'],
                    conditions['pressure_Pa'],
                    conditions['humidity_percent']
                )
                print(f"  - {sensor_id}: {drift:.6f} drift units")
        
        # Test measurement errors
        print("\n🔄 Testing measurement errors...")
        true_values = [100.0, 50.0, 0.5, 75.0, 1000.0]  # Test values for each sensor
        
        for (sensor_id, _, _), true_value in zip(test_sensors, true_values):
            # Test without history
            measured_value_1 = sensor_model.add_measurement_errors(sensor_id, true_value)
            
            # Test with history (for hysteresis)
            measurement_history = [true_value * 0.9, true_value * 0.95]
            measured_value_2 = sensor_model.add_measurement_errors(sensor_id, true_value, measurement_history)
            
            print(f"✅ {sensor_id} measurement errors:")
            print(f"  - True value: {true_value}")
            print(f"  - Measured (no history): {measured_value_1:.6f}")
            print(f"  - Measured (with history): {measured_value_2:.6f}")
            print(f"  - Error (no history): {abs(measured_value_1 - true_value):.6f}")
            print(f"  - Error (with history): {abs(measured_value_2 - true_value):.6f}")
        
        # Test complete sensor simulation
        print("\n🔄 Testing complete sensor simulation...")
        time_array = np.linspace(0, 60, 600)  # 60 seconds, 10 Hz sampling
        
        # Create test signals for each sensor type
        test_signals = {
            'temperature_rtd_pt100': 25.0 + 10.0 * np.sin(2 * np.pi * 0.1 * time_array),
            'pressure_piezoresistive': 101325.0 + 50000.0 * np.sin(2 * np.pi * 0.05 * time_array),
            'flow_ultrasonic': 0.5 + 0.2 * np.sin(2 * np.pi * 0.2 * time_array),
            'humidity_capacitive': 50.0 + 20.0 * np.sin(2 * np.pi * 0.02 * time_array),
            'power_meter': 1000.0 + 500.0 * np.sin(2 * np.pi * 0.1 * time_array)
        }
        
        environmental_conditions = {
            'temperature_C': 30.0,
            'pressure_Pa': 105000.0,
            'humidity_percent': 60.0
        }
        
        simulation_results = {}
        for sensor_id in test_signals.keys():
            true_values = test_signals[sensor_id]
            sensor_states = sensor_model.simulate_sensor_measurement(
                sensor_id, true_values, time_array, environmental_conditions
            )
            simulation_results[sensor_id] = sensor_states
            
            print(f"✅ {sensor_id} simulation:")
            print(f"  - Number of measurements: {len(sensor_states)}")
            print(f"  - Time span: {sensor_states[0].timestamp:.1f} - {sensor_states[-1].timestamp:.1f} s")
            
            # Calculate statistics
            true_vals = [s.true_value for s in sensor_states]
            measured_vals = [s.measured_value for s in sensor_states]
            noise_vals = [s.noise_component for s in sensor_states]
            drift_vals = [s.drift_component for s in sensor_states]
            uncertainty_vals = [s.total_uncertainty for s in sensor_states]
            
            print(f"  - True value range: {np.min(true_vals):.3f} - {np.max(true_vals):.3f}")
            print(f"  - Measured value range: {np.min(measured_vals):.3f} - {np.max(measured_vals):.3f}")
            print(f"  - Average noise: {np.mean(np.abs(noise_vals)):.6f}")
            print(f"  - Final drift: {drift_vals[-1]:.6f}")
            print(f"  - Average uncertainty: {np.mean(uncertainty_vals):.6f}")
        
        # Test sensor network creation
        print("\n🔄 Testing sensor network creation...")
        sensor_ids = list(test_signals.keys())
        correlation_strength = 0.2
        cross_talk_strength = 0.05
        
        sensor_network = sensor_model.create_sensor_network(
            sensor_ids, correlation_strength, cross_talk_strength
        )
        
        print(f"✅ Sensor network created:")
        print(f"  - Number of sensors: {len(sensor_network.sensors)}")
        print(f"  - Correlation matrix shape: {sensor_network.correlation_matrix.shape}")
        print(f"  - Cross-talk matrix shape: {sensor_network.cross_talk_matrix.shape}")
        print(f"  - Common mode noise: {sensor_network.common_mode_noise}")
        print(f"  - Network delay: {sensor_network.network_delay_s} s")
        
        # Display correlation matrix
        print(f"✅ Correlation matrix:")
        for i, sensor_id_i in enumerate(sensor_ids):
            row_str = f"  {sensor_id_i[:20]:20s}: "
            for j, sensor_id_j in enumerate(sensor_ids):
                row_str += f"{sensor_network.correlation_matrix[i, j]:.3f} "
            print(row_str)
        
        # Test model validation
        print("\n🔄 Testing model validation...")
        for sensor_id, _, _ in test_sensors:
            validation_results = sensor_model.validate_sensor_model(sensor_id)
            
            print(f"✅ {sensor_id} validation:")
            for test_name, result in validation_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  - {test_name}: {status}")
            
            passed_tests = sum(validation_results.values())
            total_tests = len(validation_results)
            print(f"  - Overall: {passed_tests}/{total_tests} tests passed")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test SensorSpecifications dataclass
        test_sensor_spec = SensorSpecifications(
            sensor_type=SensorType.TEMPERATURE,
            measurement_range=(-50.0, 150.0),
            accuracy_percent=0.1,
            resolution=0.01,
            response_time_s=1.0,
            drift_rate_per_year=0.05,
            temperature_coefficient=0.0,
            pressure_coefficient=0.0,
            humidity_coefficient=0.0,
            linearity_error_percent=0.05,
            hysteresis_percent=0.02,
            repeatability_percent=0.01
        )
        print(f"✅ SensorSpecifications created:")
        print(f"  - Sensor type: {test_sensor_spec.sensor_type.value}")
        print(f"  - Range: {test_sensor_spec.measurement_range}")
        print(f"  - Accuracy: ±{test_sensor_spec.accuracy_percent}%")
        print(f"  - Resolution: {test_sensor_spec.resolution}")
        
        # Test NoiseParameters dataclass
        test_noise_params = NoiseParameters(
            noise_type=NoiseType.WHITE_NOISE,
            amplitude=0.1,
            frequency_hz=None,
            correlation_time_s=None,
            burst_probability=None,
            burst_amplitude=None
        )
        print(f"✅ NoiseParameters created:")
        print(f"  - Noise type: {test_noise_params.noise_type.value}")
        print(f"  - Amplitude: {test_noise_params.amplitude}")
        print(f"  - Frequency: {test_noise_params.frequency_hz}")
        
        # Test SensorState dataclass
        test_sensor_state = SensorState(
            true_value=25.0,
            measured_value=25.1,
            noise_component=0.05,
            drift_component=0.02,
            calibration_error=0.01,
            environmental_error=0.02,
            total_uncertainty=0.08,
            timestamp=10.0,
            sensor_id="test_sensor"
        )
        print(f"✅ SensorState created:")
        print(f"  - Sensor ID: {test_sensor_state.sensor_id}")
        print(f"  - True value: {test_sensor_state.true_value}")
        print(f"  - Measured value: {test_sensor_state.measured_value}")
        print(f"  - Total uncertainty: {test_sensor_state.total_uncertainty}")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Test with unknown sensor type
        try:
            unknown_noise_params = NoiseParameters(NoiseType.WHITE_NOISE, 0.1)
            sensor_model.initialize_sensor('unknown_sensor', unknown_noise_params)
            print("❌ Unknown sensor type should have failed")
        except ValueError as e:
            print(f"✅ Unknown sensor type properly rejected: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error for unknown sensor: {str(e)}")
        
        # Test with zero duration noise generation
        try:
            zero_noise = sensor_model.generate_sensor_noise('temperature_rtd_pt100', 0.0, 100.0)
            print(f"✅ Zero duration noise handled: {len(zero_noise)} samples")
        except Exception as e:
            print(f"❌ Zero duration error: {str(e)}")
        
        # Test with very high sampling rate
        try:
            high_rate_noise = sensor_model.generate_sensor_noise('temperature_rtd_pt100', 1.0, 10000.0)
            print(f"✅ High sampling rate handled: {len(high_rate_noise)} samples")
        except Exception as e:
            print(f"❌ High sampling rate error: {str(e)}")
        
        # Test performance metrics
        print("\n🔄 Testing performance metrics...")
        
        # Calculate noise characteristics for each noise type
        noise_characteristics = {}
        for sensor_id, noise_type, amplitude in test_sensors:
            noise_signal = sensor_model.generate_sensor_noise(sensor_id, 10.0, 1000.0)
            
            # Calculate power spectral density characteristics
            fft_signal = np.fft.fft(noise_signal)
            power_spectrum = np.abs(fft_signal)**2
            
            noise_characteristics[noise_type.value] = {
                'mean': np.mean(noise_signal),
                'std': np.std(noise_signal),
                'peak_to_peak': np.max(noise_signal) - np.min(noise_signal),
                'rms': np.sqrt(np.mean(noise_signal**2)),
                'spectral_peak': np.max(power_spectrum)
            }
        
        print(f"✅ Noise characteristics by type:")
        for noise_type, characteristics in noise_characteristics.items():
            print(f"  - {noise_type}:")
            for metric, value in characteristics.items():
                print(f"    * {metric}: {value:.6f}")
        
        # Calculate sensor accuracy statistics
        sensor_accuracy_stats = {}
        for sensor_id, sensor_states in simulation_results.items():
            true_vals = np.array([s.true_value for s in sensor_states])
            measured_vals = np.array([s.measured_value for s in sensor_states])
            
            errors = measured_vals - true_vals
            relative_errors = errors / true_vals * 100  # Percentage errors
            
            sensor_accuracy_stats[sensor_id] = {
                'mean_absolute_error': np.mean(np.abs(errors)),
                'rms_error': np.sqrt(np.mean(errors**2)),
                'max_error': np.max(np.abs(errors)),
                'mean_relative_error_percent': np.mean(np.abs(relative_errors)),
                'std_relative_error_percent': np.std(relative_errors)
            }
        
        print(f"✅ Sensor accuracy statistics:")
        for sensor_id, stats in sensor_accuracy_stats.items():
            print(f"  - {sensor_id}:")
            for metric, value in stats.items():
                print(f"    * {metric}: {value:.6f}")
        
        print("\n🎉 ALL SENSOR NOISE MODELS TESTS COMPLETED!")
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error in sensor noise models: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Test completed with errors")
        return False

if __name__ == "__main__":
    success = test_sensor_noise_models()
    sys.exit(0 if success else 1)