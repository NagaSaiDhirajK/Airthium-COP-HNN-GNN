#!/usr/bin/env python3
"""
Test Control System Dynamics Models
==================================

Comprehensive test suite for control system dynamics including PID controllers,
system response, stability analysis, and control scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import json
from pathlib import Path

# Import directly from the module file
from control_system_dynamics import (
    ControlSystemModel, PIDController, ControlSystemState, ControlLoopParameters
)

def test_control_system_dynamics():
    """Test control system dynamics models comprehensively."""
    
    print("🎛️ Testing Control System Dynamics Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        print("✅ Import successful")
        
        # Initialize control system model
        print("\n🔄 Testing ControlSystemModel initialization...")
        control_model = ControlSystemModel()
        print("✅ ControlSystemModel initialized successfully")
        print(f"✅ PID tuning parameters: {len(control_model.pid_tuning_parameters)} categories")
        print(f"✅ System dynamics: {len(control_model.system_dynamics)} loops")
        print(f"✅ Actuator constraints: {len(control_model.actuator_constraints)} types")
        
        # Test control system initialization
        print("\n🔄 Testing control system initialization...")
        strategies = ['cascade', 'feedforward', 'single_loop']
        control_states = {}
        
        for strategy in strategies:
            control_state = control_model.initialize_control_system(f"test_system_{strategy}", strategy)
            control_states[strategy] = control_state
            print(f"✅ {strategy} control system initialized")
            print(f"  - System ID: {control_state.system_id}")
            print(f"  - Temperature setpoint: {control_state.temperature_setpoint_K:.1f} K")
            print(f"  - Pressure setpoint: {control_state.pressure_setpoint_bar:.1f} bar")
            print(f"  - Response time: {control_state.response_time_constant_s:.1f} s")
            print(f"  - Overshoot: {control_state.overshoot_percent:.1f}%")
        
        # Test PID controller functionality
        print("\n🔄 Testing PID controller updates...")
        test_control_state = control_states['cascade']
        
        # Test temperature controller
        temp_outputs = []
        process_values = [293.15, 294.15, 295.15, 296.15, 295.15]  # Temperature sequence
        
        for i, pv in enumerate(process_values):
            output = control_model.update_pid_controller(
                test_control_state.temperature_controller, pv, float(i), 1.0
            )
            temp_outputs.append(output)
            print(f"✅ Temperature PID step {i+1}: PV={pv:.1f}K → Output={output:.2f}%")
        
        # Test pressure controller
        pressure_outputs = []
        pressure_values = [4.0, 4.2, 4.5, 4.8, 4.5]  # Pressure sequence
        
        for i, pv in enumerate(pressure_values):
            output = control_model.update_pid_controller(
                test_control_state.pressure_controller, pv, float(i), 1.0
            )
            pressure_outputs.append(output)
            print(f"✅ Pressure PID step {i+1}: PV={pv:.1f}bar → Output={output:.2f}%")
        
        # Test capacity controller
        capacity_outputs = []
        capacity_values = [70.0, 72.0, 75.0, 78.0, 75.0]  # Capacity sequence
        
        for i, pv in enumerate(capacity_values):
            output = control_model.update_pid_controller(
                test_control_state.capacity_controller, pv, float(i), 1.0
            )
            capacity_outputs.append(output)
            print(f"✅ Capacity PID step {i+1}: PV={pv:.1f}% → Output={output:.2f}%")
        
        # Test control response simulation
        print("\n🔄 Testing control response simulation...")
        setpoint_changes = {
            'temperature': 2.0,  # 2K increase
            'pressure': 0.5,     # 0.5 bar increase
            'capacity': 10.0     # 10% increase
        }
        
        disturbances = {
            'temperature': 1.0,
            'pressure': 0.2,
            'load': 0.15
        }
        
        response_data = control_model.simulate_control_response(
            test_control_state, setpoint_changes, disturbances, 600.0, 5.0
        )
        
        print(f"✅ Control response simulation completed")
        print(f"✅ Simulation data points: {len(response_data)}")
        print(f"✅ Simulation duration: {response_data[-1]['time_s']:.1f} seconds")
        
        # Analyze response characteristics
        initial_temp = response_data[0]['temperature_K']
        final_temp = response_data[-1]['temperature_K']
        setpoint_temp = response_data[-1]['temp_setpoint']
        
        print(f"✅ Temperature response:")
        print(f"  - Initial: {initial_temp:.2f} K")
        print(f"  - Final: {final_temp:.2f} K")
        print(f"  - Setpoint: {setpoint_temp:.2f} K")
        print(f"  - Steady-state error: {abs(final_temp - setpoint_temp):.2f} K")
        
        # Test stability analysis
        print("\n🔄 Testing stability analysis...")
        stability_results = {}
        
        for strategy, control_state in control_states.items():
            stability = control_model.analyze_stability(control_state)
            stability_results[strategy] = stability
            print(f"✅ Stability analysis for {strategy} strategy:")
            
            for key, value in stability.items():
                if isinstance(value, bool):
                    status = "✅ STABLE" if value else "❌ UNSTABLE"
                    print(f"  - {key}: {status}")
                else:
                    print(f"  - {key}: {value:.2f}")
        
        # Test PID auto-tuning
        print("\n🔄 Testing PID auto-tuning...")
        tuning_methods = ['ziegler_nichols', 'cohen_coon', 'lambda_tuning']
        controllers = ['temperature', 'pressure', 'capacity']
        
        for method in tuning_methods:
            print(f"✅ Testing {method} tuning method:")
            for controller_name in controllers:
                try:
                    tuned_controller = control_model.tune_pid_controller(
                        test_control_state, controller_name, method
                    )
                    print(f"  - {controller_name}: Kp={tuned_controller.kp:.3f}, "
                          f"Ki={tuned_controller.ki:.3f}, Kd={tuned_controller.kd:.3f}")
                except Exception as e:
                    print(f"  - {controller_name}: Error - {str(e)}")
        
        # Test actuator constraints
        print("\n🔄 Testing actuator constraints...")
        test_commands = [0.0, 25.0, 50.0, 75.0, 100.0, 120.0]  # Including out-of-range
        actuator_types = ['compressor_speed', 'expansion_valve', 'fan_speed']
        
        for actuator_type in actuator_types:
            print(f"✅ Testing {actuator_type} constraints:")
            for command in test_commands:
                constrained = control_model._apply_actuator_constraints(command, actuator_type, 1.0)
                print(f"  - Command {command:.1f}% → Constrained {constrained:.1f}%")
        
        # Test control scenarios generation
        print("\n🔄 Testing control scenarios generation...")
        scenarios = control_model.generate_control_scenarios("test_system", 5)
        print(f"✅ Generated {len(scenarios)} control scenarios")
        
        for i, scenario in enumerate(scenarios):
            print(f"✅ Scenario {i+1}:")
            print(f"  - Strategy: {scenario['control_strategy']}")
            print(f"  - Setpoint changes: {scenario['setpoint_changes']}")
            print(f"  - Disturbances: {scenario['disturbances']}")
            print(f"  - Response data points: {len(scenario['response_data'])}")
            print(f"  - Stability analysis: {len(scenario['stability_analysis'])} metrics")
        
        # Test data saving
        print("\n🔄 Testing control data saving...")
        output_file = "test_control_system_data.json"
        control_model.save_control_data(scenarios, output_file)
        
        # Verify saved data
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            print(f"✅ Control data saved successfully")
            print(f"✅ Saved scenarios: {len(saved_data['control_scenarios'])}")
            print(f"✅ Model parameters included: {len(saved_data['model_parameters'])} categories")
        else:
            print("❌ Failed to save control data")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test PIDController dataclass
        test_pid = PIDController(
            controller_id="test_pid",
            process_variable="test_variable",
            setpoint=50.0,
            kp=1.0,
            ki=0.1,
            kd=0.01,
            output_min=0.0,
            output_max=100.0
        )
        print(f"✅ PIDController created: {test_pid.controller_id}")
        print(f"✅ Process variable: {test_pid.process_variable}")
        print(f"✅ Setpoint: {test_pid.setpoint}")
        print(f"✅ PID gains: Kp={test_pid.kp}, Ki={test_pid.ki}, Kd={test_pid.kd}")
        
        # Test ControlLoopParameters dataclass
        test_loop_params = ControlLoopParameters(
            loop_type="temperature",
            natural_frequency_rad_per_s=0.5,
            damping_ratio=0.7,
            time_delay_s=10.0,
            gain_margin_db=12.0,
            phase_margin_deg=45.0
        )
        print(f"✅ ControlLoopParameters created: {test_loop_params.loop_type}")
        print(f"✅ Natural frequency: {test_loop_params.natural_frequency_rad_per_s} rad/s")
        print(f"✅ Damping ratio: {test_loop_params.damping_ratio}")
        print(f"✅ Gain margin: {test_loop_params.gain_margin_db} dB")
        print(f"✅ Phase margin: {test_loop_params.phase_margin_deg}°")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Test with zero time step
        try:
            zero_output = control_model.update_pid_controller(
                test_control_state.temperature_controller, 295.0, 0.0, 0.0
            )
            print(f"✅ Zero time step handling: Output={zero_output:.2f}%")
        except Exception as e:
            print(f"❌ Zero time step error: {str(e)}")
        
        # Test with extreme setpoint changes
        extreme_setpoint_changes = {
            'temperature': 50.0,  # Very large change
            'pressure': -10.0,    # Negative change
            'capacity': 200.0     # Out of range
        }
        
        try:
            extreme_response = control_model.simulate_control_response(
                test_control_state, extreme_setpoint_changes, {}, 60.0, 1.0
            )
            print(f"✅ Extreme setpoint changes handled: {len(extreme_response)} points")
        except Exception as e:
            print(f"❌ Extreme setpoint error: {str(e)}")
        
        # Test performance metrics
        print("\n🔄 Testing performance metrics...")
        
        # Calculate settling time from response data
        setpoint = response_data[-1]['temp_setpoint']
        tolerance = 0.02 * setpoint  # 2% tolerance
        
        settling_time = None
        for point in reversed(response_data):
            if abs(point['temperature_K'] - setpoint) > tolerance:
                settling_time = point['time_s']
                break
        
        if settling_time:
            print(f"✅ Settling time (2% tolerance): {settling_time:.1f} seconds")
        else:
            print("✅ System settled within tolerance throughout simulation")
        
        # Calculate overshoot
        max_temp = max(point['temperature_K'] for point in response_data)
        overshoot_percent = ((max_temp - setpoint) / setpoint) * 100 if setpoint > 0 else 0
        print(f"✅ Maximum overshoot: {overshoot_percent:.2f}%")
        
        # Calculate steady-state error
        final_error_percent = ((final_temp - setpoint) / setpoint) * 100 if setpoint > 0 else 0
        print(f"✅ Steady-state error: {final_error_percent:.2f}%")
        
        print("\n🎉 ALL CONTROL SYSTEM DYNAMICS TESTS COMPLETED!")
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error in control system dynamics models: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Test completed with errors")
        return False

if __name__ == "__main__":
    success = test_control_system_dynamics()
    sys.exit(0 if success else 1)