#!/usr/bin/env python3
"""
Control System Dynamics and Response Models for AIRTHIUM Heat Pumps
==================================================================

This module implements realistic control system dynamics including PID controllers,
setpoint tracking, system response characteristics, and control loop interactions.

Key Features:
- PID controller implementation with tuning parameters
- Temperature and pressure control loops
- Variable speed compressor control
- Electronic expansion valve control
- Fan speed modulation
- System response dynamics and stability analysis

Sources:
- ASHRAE Handbook - HVAC Systems and Equipment (2020)
- Control Systems Engineering (Nise, 2019)
- Heat Pump Control Strategies (DOE, 2021)
- Variable Refrigerant Flow Systems (AHRI, 2020)
- Carrier Control System Design Manual
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.integrate import odeint
from scipy.signal import lti, step, bode
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

@dataclass
class PIDController:
    """PID controller parameters and state."""
    controller_id: str
    process_variable: str
    setpoint: float
    
    # PID parameters
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    
    # Controller limits
    output_min: float
    output_max: float
    
    # Internal state
    integral_term: float = 0.0
    previous_error: float = 0.0
    previous_time: float = 0.0
    
    # Anti-windup
    integral_windup_limit: float = 100.0
    
    # Filtering
    derivative_filter_alpha: float = 0.1

@dataclass
class ControlSystemState:
    """Current state of the control system."""
    system_id: str
    timestamp: str
    
    # Setpoints
    temperature_setpoint_K: float
    pressure_setpoint_bar: float
    capacity_setpoint_percent: float
    
    # Process variables (current measurements)
    supply_temperature_K: float
    return_temperature_K: float
    evaporator_pressure_bar: float
    condenser_pressure_bar: float
    
    # Control outputs
    compressor_speed_percent: float
    expansion_valve_opening_percent: float
    evaporator_fan_speed_percent: float
    condenser_fan_speed_percent: float
    
    # Controller states
    temperature_controller: PIDController
    pressure_controller: PIDController
    capacity_controller: PIDController
    
    # System dynamics
    response_time_constant_s: float
    settling_time_s: float
    overshoot_percent: float
    steady_state_error_percent: float
    
    # Control loop interactions
    coupling_matrix: np.ndarray
    stability_margin_db: float

@dataclass
class ControlLoopParameters:
    """Parameters for control loop design."""
    loop_type: str
    natural_frequency_rad_per_s: float
    damping_ratio: float
    time_delay_s: float
    gain_margin_db: float
    phase_margin_deg: float

class ControlSystemModel:
    """Model control system dynamics and response."""
    
    def __init__(self):
        """Initialize control system model with realistic parameters."""
        
        # Standard PID tuning parameters for HVAC systems
        self.pid_tuning_parameters = {
            'temperature_control': {
                'supply_air': {'kp': 2.0, 'ki': 0.1, 'kd': 0.5, 'tau': 300.0},
                'zone_temperature': {'kp': 1.5, 'ki': 0.05, 'kd': 0.3, 'tau': 600.0},
                'water_temperature': {'kp': 3.0, 'ki': 0.2, 'kd': 0.8, 'tau': 180.0}
            },
            'pressure_control': {
                'suction_pressure': {'kp': 0.8, 'ki': 0.02, 'kd': 0.1, 'tau': 60.0},
                'discharge_pressure': {'kp': 1.2, 'ki': 0.03, 'kd': 0.15, 'tau': 45.0},
                'differential_pressure': {'kp': 2.5, 'ki': 0.08, 'kd': 0.4, 'tau': 30.0}
            },
            'capacity_control': {
                'compressor_speed': {'kp': 1.0, 'ki': 0.01, 'kd': 0.05, 'tau': 120.0},
                'valve_position': {'kp': 0.5, 'ki': 0.005, 'kd': 0.02, 'tau': 15.0},
                'fan_speed': {'kp': 0.3, 'ki': 0.002, 'kd': 0.01, 'tau': 10.0}
            }
        }
        
        # System transfer function parameters
        self.system_dynamics = {
            'temperature_loop': {
                'numerator': [1.0],
                'denominator': [300.0, 30.0, 1.0],  # Second-order system
                'time_delay': 15.0,  # seconds
                'natural_frequency': 0.18,  # rad/s
                'damping_ratio': 0.7
            },
            'pressure_loop': {
                'numerator': [2.0],
                'denominator': [60.0, 12.0, 1.0],
                'time_delay': 5.0,
                'natural_frequency': 0.45,
                'damping_ratio': 0.8
            },
            'capacity_loop': {
                'numerator': [0.8],
                'denominator': [120.0, 20.0, 1.0],
                'time_delay': 10.0,
                'natural_frequency': 0.41,
                'damping_ratio': 0.6
            }
        }
        
        # Control loop coupling matrix (how loops affect each other)
        self.coupling_matrix = np.array([
            [1.0, 0.3, 0.2],    # Temperature loop coupling
            [0.4, 1.0, 0.5],    # Pressure loop coupling
            [0.2, 0.6, 1.0]     # Capacity loop coupling
        ])
        
        # Actuator dynamics and constraints
        self.actuator_constraints = {
            'compressor_speed': {
                'min_percent': 20.0,
                'max_percent': 100.0,
                'max_rate_percent_per_s': 2.0,
                'response_time_s': 5.0
            },
            'expansion_valve': {
                'min_percent': 10.0,
                'max_percent': 100.0,
                'max_rate_percent_per_s': 10.0,
                'response_time_s': 2.0
            },
            'fan_speed': {
                'min_percent': 30.0,
                'max_percent': 100.0,
                'max_rate_percent_per_s': 5.0,
                'response_time_s': 3.0
            }
        }
        
        # Disturbance models
        self.disturbance_models = {
            'load_disturbance': {
                'amplitude_range': [0.1, 0.5],  # Fraction of nominal
                'frequency_range': [0.001, 0.1],  # rad/s
                'step_magnitude_range': [0.05, 0.3]
            },
            'ambient_disturbance': {
                'amplitude_range': [0.05, 0.2],
                'frequency_range': [0.0001, 0.01],
                'step_magnitude_range': [0.02, 0.15]
            },
            'supply_disturbance': {
                'amplitude_range': [0.02, 0.1],
                'frequency_range': [0.01, 1.0],
                'step_magnitude_range': [0.01, 0.08]
            }
        }
    
    def initialize_control_system(self, system_id: str, control_strategy: str = 'cascade') -> ControlSystemState:
        """Initialize control system with specified strategy."""
        
        timestamp = datetime.now().isoformat()
        
        # Initialize PID controllers based on strategy
        if control_strategy == 'cascade':
            # Cascade control with temperature master, pressure slave
            temp_params = self.pid_tuning_parameters['temperature_control']['supply_air']
            pressure_params = self.pid_tuning_parameters['pressure_control']['suction_pressure']
            capacity_params = self.pid_tuning_parameters['capacity_control']['compressor_speed']
        elif control_strategy == 'feedforward':
            # Feedforward control with load compensation
            temp_params = {'kp': 1.8, 'ki': 0.08, 'kd': 0.4, 'tau': 250.0}
            pressure_params = {'kp': 1.0, 'ki': 0.025, 'kd': 0.12, 'tau': 50.0}
            capacity_params = {'kp': 1.2, 'ki': 0.015, 'kd': 0.06, 'tau': 100.0}
        else:  # 'single_loop'
            # Single loop control
            temp_params = {'kp': 2.5, 'ki': 0.12, 'kd': 0.6, 'tau': 350.0}
            pressure_params = {'kp': 0.6, 'ki': 0.015, 'kd': 0.08, 'tau': 70.0}
            capacity_params = {'kp': 0.8, 'ki': 0.008, 'kd': 0.04, 'tau': 150.0}
        
        # Create PID controllers
        temperature_controller = PIDController(
            controller_id=f"{system_id}_temp_ctrl",
            process_variable="supply_temperature",
            setpoint=295.15,  # 22°C
            kp=temp_params['kp'],
            ki=temp_params['ki'],
            kd=temp_params['kd'],
            output_min=20.0,
            output_max=100.0
        )
        
        pressure_controller = PIDController(
            controller_id=f"{system_id}_press_ctrl",
            process_variable="suction_pressure",
            setpoint=4.5,  # bar
            kp=pressure_params['kp'],
            ki=pressure_params['ki'],
            kd=pressure_params['kd'],
            output_min=10.0,
            output_max=100.0
        )
        
        capacity_controller = PIDController(
            controller_id=f"{system_id}_cap_ctrl",
            process_variable="compressor_speed",
            setpoint=75.0,  # %
            kp=capacity_params['kp'],
            ki=capacity_params['ki'],
            kd=capacity_params['kd'],
            output_min=20.0,
            output_max=100.0
        )
        
        # Calculate system response characteristics
        temp_dynamics = self.system_dynamics['temperature_loop']
        response_time = 4 * temp_dynamics['denominator'][0] / temp_dynamics['denominator'][1]  # 4 time constants
        settling_time = response_time * 1.2
        overshoot = np.exp(-np.pi * temp_dynamics['damping_ratio'] / 
                          np.sqrt(1 - temp_dynamics['damping_ratio']**2)) * 100
        
        return ControlSystemState(
            system_id=system_id,
            timestamp=timestamp,
            temperature_setpoint_K=295.15,
            pressure_setpoint_bar=4.5,
            capacity_setpoint_percent=75.0,
            supply_temperature_K=295.15,
            return_temperature_K=290.15,
            evaporator_pressure_bar=4.5,
            condenser_pressure_bar=18.0,
            compressor_speed_percent=75.0,
            expansion_valve_opening_percent=45.0,
            evaporator_fan_speed_percent=80.0,
            condenser_fan_speed_percent=85.0,
            temperature_controller=temperature_controller,
            pressure_controller=pressure_controller,
            capacity_controller=capacity_controller,
            response_time_constant_s=response_time,
            settling_time_s=settling_time,
            overshoot_percent=overshoot,
            steady_state_error_percent=1.0,
            coupling_matrix=self.coupling_matrix,
            stability_margin_db=12.0
        )
    
    def update_pid_controller(self, controller: PIDController, process_value: float, 
                            current_time: float, dt: float) -> float:
        """Update PID controller and return control output."""
        
        # Calculate error
        error = controller.setpoint - process_value
        
        # Proportional term
        proportional = controller.kp * error
        
        # Integral term with anti-windup
        controller.integral_term += error * dt
        if abs(controller.integral_term) > controller.integral_windup_limit:
            controller.integral_term = np.sign(controller.integral_term) * controller.integral_windup_limit
        integral = controller.ki * controller.integral_term
        
        # Derivative term with filtering
        if dt > 0:
            derivative_raw = (error - controller.previous_error) / dt
            # Apply low-pass filter to derivative term
            if hasattr(controller, 'filtered_derivative'):
                controller.filtered_derivative = (controller.derivative_filter_alpha * derivative_raw + 
                                                (1 - controller.derivative_filter_alpha) * controller.filtered_derivative)
            else:
                controller.filtered_derivative = derivative_raw
            derivative = controller.kd * controller.filtered_derivative
        else:
            derivative = 0.0
        
        # Calculate total output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = np.clip(output, controller.output_min, controller.output_max)
        
        # Update controller state
        controller.previous_error = error
        controller.previous_time = current_time
        
        return output
    
    def simulate_control_response(self, control_state: ControlSystemState, 
                                setpoint_change: Dict[str, float],
                                disturbances: Dict[str, float],
                                simulation_time_s: float = 3600.0,
                                dt: float = 1.0) -> List[Dict[str, float]]:
        """Simulate control system response to setpoint changes and disturbances."""
        
        time_points = np.arange(0, simulation_time_s, dt)
        response_data = []
        
        # Current system state
        current_temp = control_state.supply_temperature_K
        current_pressure = control_state.evaporator_pressure_bar
        current_capacity = control_state.compressor_speed_percent
        
        # Apply setpoint changes
        if 'temperature' in setpoint_change:
            control_state.temperature_controller.setpoint += setpoint_change['temperature']
        if 'pressure' in setpoint_change:
            control_state.pressure_controller.setpoint += setpoint_change['pressure']
        if 'capacity' in setpoint_change:
            control_state.capacity_controller.setpoint += setpoint_change['capacity']
        
        for t in time_points:
            # Add disturbances
            temp_disturbance = disturbances.get('temperature', 0.0) * np.sin(0.01 * t)
            pressure_disturbance = disturbances.get('pressure', 0.0) * np.sin(0.02 * t)
            load_disturbance = disturbances.get('load', 0.0) * (1 + 0.1 * np.sin(0.005 * t))
            
            # Update PID controllers
            temp_output = self.update_pid_controller(
                control_state.temperature_controller, current_temp, t, dt
            )
            
            pressure_output = self.update_pid_controller(
                control_state.pressure_controller, current_pressure, t, dt
            )
            
            capacity_output = self.update_pid_controller(
                control_state.capacity_controller, current_capacity, t, dt
            )
            
            # Apply actuator constraints and dynamics
            temp_output = self._apply_actuator_constraints(temp_output, 'compressor_speed', dt)
            pressure_output = self._apply_actuator_constraints(pressure_output, 'expansion_valve', dt)
            capacity_output = self._apply_actuator_constraints(capacity_output, 'fan_speed', dt)
            
            # Simulate system response (simplified first-order dynamics)
            temp_dynamics = self.system_dynamics['temperature_loop']
            tau_temp = temp_dynamics['denominator'][0]
            current_temp += dt * (temp_output - current_temp + temp_disturbance) / tau_temp
            
            pressure_dynamics = self.system_dynamics['pressure_loop']
            tau_pressure = pressure_dynamics['denominator'][0]
            current_pressure += dt * (pressure_output - current_pressure + pressure_disturbance) / tau_pressure
            
            capacity_dynamics = self.system_dynamics['capacity_loop']
            tau_capacity = capacity_dynamics['denominator'][0]
            current_capacity += dt * (capacity_output - current_capacity + load_disturbance) / tau_capacity
            
            # Apply coupling effects
            coupling_effects = np.dot(control_state.coupling_matrix, 
                                    [temp_output, pressure_output, capacity_output])
            current_temp += coupling_effects[0] * 0.01
            current_pressure += coupling_effects[1] * 0.01
            current_capacity += coupling_effects[2] * 0.01
            
            # Record data
            response_data.append({
                'time_s': t,
                'temperature_K': current_temp,
                'pressure_bar': current_pressure,
                'capacity_percent': current_capacity,
                'temp_controller_output': temp_output,
                'pressure_controller_output': pressure_output,
                'capacity_controller_output': capacity_output,
                'temp_setpoint': control_state.temperature_controller.setpoint,
                'pressure_setpoint': control_state.pressure_controller.setpoint,
                'capacity_setpoint': control_state.capacity_controller.setpoint
            })
        
        return response_data
    
    def _apply_actuator_constraints(self, command: float, actuator_type: str, dt: float) -> float:
        """Apply actuator rate limits and constraints."""
        
        constraints = self.actuator_constraints.get(actuator_type, 
                                                   self.actuator_constraints['compressor_speed'])
        
        # Apply output limits
        command = np.clip(command, constraints['min_percent'], constraints['max_percent'])
        
        # Apply rate limits (simplified - would need previous value in practice)
        max_change = constraints['max_rate_percent_per_s'] * dt
        # In practice, would compare with previous command and limit rate of change
        
        return command
    
    def analyze_stability(self, control_state: ControlSystemState) -> Dict[str, float]:
        """Analyze control system stability margins."""
        
        stability_analysis = {}
        
        for loop_name, dynamics in self.system_dynamics.items():
            # Create transfer function
            num = dynamics['numerator']
            den = dynamics['denominator']
            
            # Calculate gain and phase margins (simplified)
            natural_freq = dynamics['natural_frequency']
            damping = dynamics['damping_ratio']
            
            # Gain margin (dB)
            gain_margin = 20 * np.log10(1 / (2 * damping))
            
            # Phase margin (degrees)
            phase_margin = np.arctan(2 * damping / np.sqrt(1 - 2 * damping**2)) * 180 / np.pi
            
            stability_analysis[f'{loop_name}_gain_margin_db'] = gain_margin
            stability_analysis[f'{loop_name}_phase_margin_deg'] = phase_margin
            stability_analysis[f'{loop_name}_stable'] = gain_margin > 6.0 and phase_margin > 30.0
        
        return stability_analysis
    
    def tune_pid_controller(self, control_state: ControlSystemState, 
                          controller_name: str, tuning_method: str = 'ziegler_nichols') -> PIDController:
        """Auto-tune PID controller using specified method."""
        
        if controller_name == 'temperature':
            controller = control_state.temperature_controller
            dynamics = self.system_dynamics['temperature_loop']
        elif controller_name == 'pressure':
            controller = control_state.pressure_controller
            dynamics = self.system_dynamics['pressure_loop']
        else:
            controller = control_state.capacity_controller
            dynamics = self.system_dynamics['capacity_loop']
        
        if tuning_method == 'ziegler_nichols':
            # Ziegler-Nichols tuning based on step response
            tau = dynamics['denominator'][0]  # Time constant
            L = dynamics['time_delay']        # Dead time
            K = dynamics['numerator'][0]      # Process gain
            
            # Ziegler-Nichols PID tuning rules
            controller.kp = 1.2 * tau / (K * L)
            controller.ki = controller.kp / (2 * L)
            controller.kd = controller.kp * L / 2
            
        elif tuning_method == 'cohen_coon':
            # Cohen-Coon tuning
            tau = dynamics['denominator'][0]
            L = dynamics['time_delay']
            K = dynamics['numerator'][0]
            
            R = L / tau
            controller.kp = (1.35 / K) * (tau / L) * (1 + 0.18 * R)
            controller.ki = controller.kp / (L * (2.5 - 2 * R) / (1 + 0.39 * R))
            controller.kd = controller.kp * L * (0.37 - 0.37 * R) / (1 + 0.81 * R)
            
        elif tuning_method == 'lambda_tuning':
            # Lambda tuning (Internal Model Control)
            tau = dynamics['denominator'][0]
            L = dynamics['time_delay']
            K = dynamics['numerator'][0]
            
            lambda_c = max(0.1 * tau, L)  # Closed-loop time constant
            controller.kp = tau / (K * (lambda_c + L))
            controller.ki = controller.kp / tau
            controller.kd = 0  # Often set to zero in lambda tuning
        
        return controller
    
    def generate_control_scenarios(self, system_id: str, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """Generate diverse control scenarios for ML training."""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Random control strategy
            strategy = np.random.choice(['cascade', 'feedforward', 'single_loop'])
            
            # Initialize control system
            control_state = self.initialize_control_system(f"{system_id}_{i:03d}", strategy)
            
            # Random setpoint changes
            setpoint_changes = {
                'temperature': np.random.uniform(-5.0, 5.0),
                'pressure': np.random.uniform(-1.0, 1.0),
                'capacity': np.random.uniform(-20.0, 20.0)
            }
            
            # Random disturbances
            disturbances = {
                'temperature': np.random.uniform(-2.0, 2.0),
                'pressure': np.random.uniform(-0.5, 0.5),
                'load': np.random.uniform(-0.3, 0.3)
            }
            
            # Simulate response
            response_data = self.simulate_control_response(
                control_state, setpoint_changes, disturbances, 1800.0, 5.0
            )
            
            # Analyze stability
            stability = self.analyze_stability(control_state)
            
            scenarios.append({
                'scenario_id': i,
                'control_strategy': strategy,
                'setpoint_changes': setpoint_changes,
                'disturbances': disturbances,
                'response_data': response_data,
                'stability_analysis': stability,
                'control_state': asdict(control_state)
            })
        
        return scenarios
    
    def save_control_data(self, control_scenarios: List[Dict[str, Any]], output_path: str) -> None:
        """Save control system data to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'control_scenarios': control_scenarios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'pid_tuning_parameters': self.pid_tuning_parameters,
                'system_dynamics': self.system_dynamics,
                'actuator_constraints': self.actuator_constraints,
                'coupling_matrix': self.coupling_matrix.tolist()
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved control system data to {output_path}")


if __name__ == "__main__":
    # Test control system models
    print("🔄 Testing Control System Dynamics Models")
    
    # Initialize model
    control_model = ControlSystemModel()
    
    # Create test control system
    control_state = control_model.initialize_control_system("test_system", "cascade")
    
    print(f"\n📊 Control System Initialized:")
    print(f"Temperature setpoint: {control_state.temperature_setpoint_K - 273.15:.1f}°C")
    print(f"Pressure setpoint: {control_state.pressure_setpoint_bar:.1f} bar")
    print(f"Response time: {control_state.response_time_constant_s:.1f} s")
    print(f"Overshoot: {control_state.overshoot_percent:.1f}%")
    
    # Test PID controller update
    temp_output = control_model.update_pid_controller(
        control_state.temperature_controller, 293.15, 0.0, 1.0
    )
    print(f"\nPID Controller Output: {temp_output:.2f}%")
    
    # Test stability analysis
    stability = control_model.analyze_stability(control_state)
    print(f"\n🔍 Stability Analysis:")
    for key, value in stability.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    
    # Test control response simulation
    print(f"\n⏱️ Simulating control response...")
    setpoint_change = {'temperature': 2.0}  # 2K step change
    disturbances = {'temperature': 0.5, 'load': 0.1}
    
    response_data = control_model.simulate_control_response(
        control_state, setpoint_change, disturbances, 600.0, 5.0
    )
    
    print(f"Simulated {len(response_data)} data points over 10 minutes")
    final_temp = response_data[-1]['temperature_K']
    print(f"Final temperature: {final_temp - 273.15:.2f}°C")
    
    # Generate control scenarios
    print(f"\n🎲 Generating control scenarios...")
    scenarios = control_model.generate_control_scenarios("test_system", 5)
    print(f"Generated {len(scenarios)} control scenarios")
    
    # Save test data
    control_model.save_control_data(scenarios, "control_system_data.json")
    
    print("\n✅ Control system dynamics models testing complete!")