"""
Real-Time Sensor Noise and Measurement Uncertainty Models
========================================================

This module implements realistic sensor imperfections for HNN/GNN training data.
It models sensor drift, noise, calibration errors, and response delays that
occur in real AIRTHIUM heat pump systems.

Key Features:
- Temperature sensor drift and noise modeling
- Pressure sensor uncertainty and calibration errors
- Flow meter accuracy limitations
- Sensor response time delays
- Correlated noise between sensors
- Time-dependent drift patterns
- Environmental effects on sensor accuracy

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
from scipy import signal
from scipy.stats import norm, uniform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Types of sensors in heat pump systems"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    HUMIDITY = "humidity"
    POWER = "power"
    VIBRATION = "vibration"
    POSITION = "position"

class NoiseType(Enum):
    """Types of sensor noise"""
    WHITE_NOISE = "white_noise"
    PINK_NOISE = "pink_noise"
    BROWNIAN_NOISE = "brownian_noise"
    PERIODIC_NOISE = "periodic_noise"
    IMPULSE_NOISE = "impulse_noise"

@dataclass
class SensorSpecifications:
    """Sensor specifications and accuracy parameters"""
    sensor_type: SensorType
    measurement_range: Tuple[float, float]  # (min, max) values
    accuracy_percent: float  # ±% of reading
    resolution: float  # Smallest detectable change
    response_time_s: float  # Time constant for step response
    drift_rate_per_year: float  # Annual drift rate
    temperature_coefficient: float  # Change per °C
    pressure_coefficient: float  # Change per Pa (for pressure-sensitive sensors)
    humidity_coefficient: float  # Change per %RH
    linearity_error_percent: float  # Non-linearity error
    hysteresis_percent: float  # Hysteresis error
    repeatability_percent: float  # Repeatability error

@dataclass
class NoiseParameters:
    """Parameters for different noise types"""
    noise_type: NoiseType
    amplitude: float  # Noise amplitude
    frequency_hz: Optional[float] = None  # For periodic noise
    correlation_time_s: Optional[float] = None  # For colored noise
    burst_probability: Optional[float] = None  # For impulse noise
    burst_amplitude: Optional[float] = None  # For impulse noise

@dataclass
class SensorState:
    """Current state of a sensor including all error sources"""
    true_value: float
    measured_value: float
    noise_component: float
    drift_component: float
    calibration_error: float
    environmental_error: float
    total_uncertainty: float
    timestamp: float
    sensor_id: str

@dataclass
class SensorNetwork:
    """Network of correlated sensors"""
    sensors: Dict[str, SensorSpecifications]
    correlation_matrix: np.ndarray
    cross_talk_matrix: np.ndarray
    common_mode_noise: float
    network_delay_s: float

class SensorNoiseModel:
    """
    Comprehensive sensor noise and uncertainty model for realistic data generation.
    
    This class implements various sources of sensor error including:
    - Random noise (white, pink, brownian)
    - Systematic drift over time
    - Calibration errors
    - Environmental effects
    - Response delays
    - Cross-talk between sensors
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize sensor noise model"""
        self.sensor_database = self._initialize_sensor_database()
        self.noise_generators = {}
        self.drift_states = {}
        self.calibration_errors = {}
        self.response_filters = {}
        
        # Random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Time tracking
        self.current_time = 0.0
        self.last_update_time = 0.0
        
        logger.info("Sensor noise model initialized")
    
    def _initialize_sensor_database(self) -> Dict[str, SensorSpecifications]:
        """Initialize database of sensor specifications for AIRTHIUM heat pumps"""
        
        database = {
            # Temperature sensors (RTD, Thermocouples, Thermistors)
            'temperature_rtd_pt100': SensorSpecifications(
                sensor_type=SensorType.TEMPERATURE,
                measurement_range=(-50.0, 200.0),  # °C
                accuracy_percent=0.1,  # ±0.1% of reading
                resolution=0.01,  # 0.01°C
                response_time_s=1.0,  # 1 second time constant
                drift_rate_per_year=0.05,  # 0.05°C/year
                temperature_coefficient=0.0,  # Self-compensating
                pressure_coefficient=0.0,
                humidity_coefficient=0.0,
                linearity_error_percent=0.05,
                hysteresis_percent=0.02,
                repeatability_percent=0.01
            ),
            
            'temperature_thermistor': SensorSpecifications(
                sensor_type=SensorType.TEMPERATURE,
                measurement_range=(-40.0, 125.0),  # °C
                accuracy_percent=0.2,  # ±0.2% of reading
                resolution=0.1,  # 0.1°C
                response_time_s=0.5,  # 0.5 second time constant
                drift_rate_per_year=0.1,  # 0.1°C/year
                temperature_coefficient=0.0,
                pressure_coefficient=0.0,
                humidity_coefficient=0.01,  # Slight humidity sensitivity
                linearity_error_percent=0.1,
                hysteresis_percent=0.05,
                repeatability_percent=0.02
            ),
            
            # Pressure sensors (Piezoresistive, Capacitive)
            'pressure_piezoresistive': SensorSpecifications(
                sensor_type=SensorType.PRESSURE,
                measurement_range=(0.0, 5e6),  # Pa (0-50 bar)
                accuracy_percent=0.25,  # ±0.25% of full scale
                resolution=100.0,  # 100 Pa
                response_time_s=0.1,  # 0.1 second
                drift_rate_per_year=0.1,  # 0.1% FS/year
                temperature_coefficient=0.02,  # 0.02% FS/°C
                pressure_coefficient=0.0,
                humidity_coefficient=0.0,
                linearity_error_percent=0.1,
                hysteresis_percent=0.05,
                repeatability_percent=0.02
            ),
            
            'pressure_capacitive': SensorSpecifications(
                sensor_type=SensorType.PRESSURE,
                measurement_range=(0.0, 1e6),  # Pa (0-10 bar)
                accuracy_percent=0.1,  # ±0.1% of full scale
                resolution=10.0,  # 10 Pa
                response_time_s=0.05,  # 0.05 second
                drift_rate_per_year=0.05,  # 0.05% FS/year
                temperature_coefficient=0.01,  # 0.01% FS/°C
                pressure_coefficient=0.0,
                humidity_coefficient=0.005,  # 0.005% FS/%RH
                linearity_error_percent=0.05,
                hysteresis_percent=0.02,
                repeatability_percent=0.01
            ),
            
            # Flow rate sensors (Turbine, Ultrasonic, Coriolis)
            'flow_turbine': SensorSpecifications(
                sensor_type=SensorType.FLOW_RATE,
                measurement_range=(0.0, 1.0),  # kg/s
                accuracy_percent=1.0,  # ±1% of reading
                resolution=0.001,  # 0.001 kg/s
                response_time_s=0.2,  # 0.2 second
                drift_rate_per_year=0.5,  # 0.5%/year
                temperature_coefficient=0.1,  # 0.1%/°C
                pressure_coefficient=1e-8,  # 1e-8 /Pa
                humidity_coefficient=0.0,
                linearity_error_percent=0.5,
                hysteresis_percent=0.1,
                repeatability_percent=0.1
            ),
            
            'flow_ultrasonic': SensorSpecifications(
                sensor_type=SensorType.FLOW_RATE,
                measurement_range=(0.0, 2.0),  # kg/s
                accuracy_percent=0.5,  # ±0.5% of reading
                resolution=0.0001,  # 0.0001 kg/s
                response_time_s=0.1,  # 0.1 second
                drift_rate_per_year=0.2,  # 0.2%/year
                temperature_coefficient=0.05,  # 0.05%/°C
                pressure_coefficient=5e-9,  # 5e-9 /Pa
                humidity_coefficient=0.0,
                linearity_error_percent=0.2,
                hysteresis_percent=0.02,
                repeatability_percent=0.05
            ),
            
            # Humidity sensors
            'humidity_capacitive': SensorSpecifications(
                sensor_type=SensorType.HUMIDITY,
                measurement_range=(0.0, 100.0),  # %RH
                accuracy_percent=2.0,  # ±2% RH
                resolution=0.1,  # 0.1% RH
                response_time_s=5.0,  # 5 second time constant
                drift_rate_per_year=1.0,  # 1% RH/year
                temperature_coefficient=0.1,  # 0.1% RH/°C
                pressure_coefficient=0.0,
                humidity_coefficient=0.0,
                linearity_error_percent=1.0,
                hysteresis_percent=0.5,
                repeatability_percent=0.2
            ),
            
            # Power sensors
            'power_meter': SensorSpecifications(
                sensor_type=SensorType.POWER,
                measurement_range=(0.0, 50000.0),  # W
                accuracy_percent=0.5,  # ±0.5% of reading
                resolution=1.0,  # 1 W
                response_time_s=0.1,  # 0.1 second
                drift_rate_per_year=0.1,  # 0.1%/year
                temperature_coefficient=0.02,  # 0.02%/°C
                pressure_coefficient=0.0,
                humidity_coefficient=0.0,
                linearity_error_percent=0.1,
                hysteresis_percent=0.05,
                repeatability_percent=0.02
            )
        }
        
        return database
    
    def create_sensor_network(self, 
                            sensor_ids: List[str],
                            correlation_strength: float = 0.1,
                            cross_talk_strength: float = 0.01) -> SensorNetwork:
        """
        Create a network of correlated sensors
        
        Args:
            sensor_ids: List of sensor identifiers
            correlation_strength: Strength of noise correlation between sensors
            cross_talk_strength: Strength of cross-talk between sensors
            
        Returns:
            SensorNetwork object
        """
        
        n_sensors = len(sensor_ids)
        
        # Create correlation matrix
        correlation_matrix = np.eye(n_sensors)
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                # Correlation decreases with sensor type difference
                sensor_i = self.sensor_database[sensor_ids[i]]
                sensor_j = self.sensor_database[sensor_ids[j]]
                
                if sensor_i.sensor_type == sensor_j.sensor_type:
                    corr = correlation_strength
                else:
                    corr = correlation_strength * 0.1
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Create cross-talk matrix
        cross_talk_matrix = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    cross_talk_matrix[i, j] = cross_talk_strength * self.rng.uniform(0.5, 1.5)
        
        # Get sensor specifications
        sensors = {sid: self.sensor_database[sid] for sid in sensor_ids}
        
        return SensorNetwork(
            sensors=sensors,
            correlation_matrix=correlation_matrix,
            cross_talk_matrix=cross_talk_matrix,
            common_mode_noise=0.01,  # 1% common mode noise
            network_delay_s=0.01  # 10ms network delay
        )
    
    def initialize_sensor(self, 
                         sensor_id: str,
                         noise_params: NoiseParameters,
                         initial_calibration_error: float = 0.0) -> None:
        """
        Initialize a sensor with noise characteristics
        
        Args:
            sensor_id: Unique sensor identifier
            noise_params: Noise parameters for this sensor
            initial_calibration_error: Initial calibration error
        """
        
        if sensor_id not in self.sensor_database:
            raise ValueError(f"Unknown sensor type: {sensor_id}")
        
        # Initialize noise generator
        self.noise_generators[sensor_id] = noise_params
        
        # Initialize drift state
        self.drift_states[sensor_id] = 0.0
        
        # Initialize calibration error
        self.calibration_errors[sensor_id] = initial_calibration_error
        
        # Initialize response filter
        sensor_spec = self.sensor_database[sensor_id]
        if sensor_spec.response_time_s > 0:
            # First-order low-pass filter
            cutoff_freq = 1.0 / (2 * np.pi * sensor_spec.response_time_s)
            self.response_filters[sensor_id] = {
                'cutoff_freq': cutoff_freq,
                'state': 0.0
            }
        
        logger.info(f"Initialized sensor {sensor_id} with {noise_params.noise_type.value} noise")
    
    def generate_sensor_noise(self, 
                            sensor_id: str,
                            duration_s: float,
                            sampling_rate_hz: float) -> np.ndarray:
        """
        Generate noise time series for a sensor
        
        Args:
            sensor_id: Sensor identifier
            duration_s: Duration of noise signal
            sampling_rate_hz: Sampling rate
            
        Returns:
            Noise time series array
        """
        
        if sensor_id not in self.noise_generators:
            raise ValueError(f"Sensor {sensor_id} not initialized")
        
        noise_params = self.noise_generators[sensor_id]
        n_samples = int(duration_s * sampling_rate_hz)
        dt = 1.0 / sampling_rate_hz
        
        if noise_params.noise_type == NoiseType.WHITE_NOISE:
            noise = self.rng.normal(0, noise_params.amplitude, n_samples)
            
        elif noise_params.noise_type == NoiseType.PINK_NOISE:
            # Generate pink noise using filtering
            white_noise = self.rng.normal(0, 1, n_samples)
            noise = self._generate_pink_noise(white_noise, noise_params.amplitude)
            
        elif noise_params.noise_type == NoiseType.BROWNIAN_NOISE:
            # Generate Brownian noise (random walk)
            white_noise = self.rng.normal(0, noise_params.amplitude, n_samples)
            noise = np.cumsum(white_noise) * np.sqrt(dt)
            
        elif noise_params.noise_type == NoiseType.PERIODIC_NOISE:
            # Generate periodic noise
            t = np.arange(n_samples) * dt
            if noise_params.frequency_hz is not None:
                noise = noise_params.amplitude * np.sin(2 * np.pi * noise_params.frequency_hz * t)
                # Add some phase noise
                phase_noise = self.rng.normal(0, 0.1, n_samples)
                noise += noise_params.amplitude * 0.1 * np.sin(2 * np.pi * noise_params.frequency_hz * t + phase_noise)
            else:
                noise = np.zeros(n_samples)
                
        elif noise_params.noise_type == NoiseType.IMPULSE_NOISE:
            # Generate impulse noise
            noise = np.zeros(n_samples)
            if noise_params.burst_probability is not None and noise_params.burst_amplitude is not None:
                for i in range(n_samples):
                    if self.rng.random() < noise_params.burst_probability * dt:
                        noise[i] = self.rng.choice([-1, 1]) * noise_params.burst_amplitude
            
            # Add background white noise
            noise += self.rng.normal(0, noise_params.amplitude * 0.1, n_samples)
        
        else:
            noise = np.zeros(n_samples)
        
        return noise
    
    def _generate_pink_noise(self, white_noise: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate pink noise from white noise using filtering"""
        
        # Simple pink noise filter (1/f characteristic)
        n_samples = len(white_noise)
        freqs = np.fft.fftfreq(n_samples)
        
        # Avoid division by zero
        freqs[0] = 1e-10
        
        # Pink noise has 1/f power spectrum
        pink_filter = 1.0 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 1.0  # DC component
        
        # Apply filter in frequency domain
        white_fft = np.fft.fft(white_noise)
        pink_fft = white_fft * pink_filter
        pink_noise = np.real(np.fft.ifft(pink_fft))
        
        # Scale to desired amplitude
        pink_noise = pink_noise / np.std(pink_noise) * amplitude
        
        return pink_noise
    
    def apply_sensor_response(self, 
                            sensor_id: str,
                            input_signal: np.ndarray,
                            dt: float) -> np.ndarray:
        """
        Apply sensor response characteristics (filtering, delays)
        
        Args:
            sensor_id: Sensor identifier
            input_signal: Input signal array
            dt: Time step
            
        Returns:
            Filtered output signal
        """
        
        if sensor_id not in self.response_filters:
            return input_signal
        
        filter_params = self.response_filters[sensor_id]
        cutoff_freq = filter_params['cutoff_freq']
        
        # First-order low-pass filter implementation
        alpha = dt / (dt + 1.0 / (2 * np.pi * cutoff_freq))
        
        output_signal = np.zeros_like(input_signal)
        state = filter_params['state']
        
        for i, x in enumerate(input_signal):
            state = alpha * x + (1 - alpha) * state
            output_signal[i] = state
        
        # Update filter state
        self.response_filters[sensor_id]['state'] = state
        
        return output_signal
    
    def calculate_drift(self, 
                       sensor_id: str,
                       time_elapsed_s: float,
                       temperature_C: float = 25.0,
                       pressure_Pa: float = 101325.0,
                       humidity_percent: float = 50.0) -> float:
        """
        Calculate sensor drift based on time and environmental conditions
        
        Args:
            sensor_id: Sensor identifier
            time_elapsed_s: Time elapsed since last calibration
            temperature_C: Ambient temperature
            pressure_Pa: Ambient pressure
            humidity_percent: Relative humidity
            
        Returns:
            Current drift value
        """
        
        if sensor_id not in self.sensor_database:
            raise ValueError(f"Unknown sensor: {sensor_id}")
        
        sensor_spec = self.sensor_database[sensor_id]
        
        # Time-based drift (linear approximation)
        time_years = time_elapsed_s / (365.25 * 24 * 3600)
        time_drift = sensor_spec.drift_rate_per_year * time_years
        
        # Temperature-induced drift
        temp_drift = sensor_spec.temperature_coefficient * (temperature_C - 25.0)
        
        # Pressure-induced drift
        pressure_drift = sensor_spec.pressure_coefficient * (pressure_Pa - 101325.0)
        
        # Humidity-induced drift
        humidity_drift = sensor_spec.humidity_coefficient * (humidity_percent - 50.0)
        
        # Total drift
        total_drift = time_drift + temp_drift + pressure_drift + humidity_drift
        
        # Update drift state
        self.drift_states[sensor_id] = total_drift
        
        return total_drift
    
    def add_measurement_errors(self, 
                             sensor_id: str,
                             true_value: float,
                             measurement_history: Optional[List[float]] = None) -> float:
        """
        Add various measurement errors to true value
        
        Args:
            sensor_id: Sensor identifier
            true_value: True measurement value
            measurement_history: Previous measurements for hysteresis
            
        Returns:
            Measured value with errors
        """
        
        if sensor_id not in self.sensor_database:
            raise ValueError(f"Unknown sensor: {sensor_id}")
        
        sensor_spec = self.sensor_database[sensor_id]
        
        # Start with true value
        measured_value = true_value
        
        # Add calibration error
        if sensor_id in self.calibration_errors:
            calibration_error = self.calibration_errors[sensor_id]
            measured_value += calibration_error
        
        # Add drift
        if sensor_id in self.drift_states:
            measured_value += self.drift_states[sensor_id]
        
        # Add linearity error
        measurement_range = sensor_spec.measurement_range[1] - sensor_spec.measurement_range[0]
        linearity_error = (sensor_spec.linearity_error_percent / 100.0) * measurement_range
        linearity_error *= np.sin(2 * np.pi * (true_value - sensor_spec.measurement_range[0]) / measurement_range)
        measured_value += linearity_error
        
        # Add hysteresis error
        if measurement_history is not None and len(measurement_history) > 0:
            direction = 1 if true_value > measurement_history[-1] else -1
            hysteresis_error = direction * (sensor_spec.hysteresis_percent / 100.0) * measurement_range * 0.5
            measured_value += hysteresis_error
        
        # Add repeatability error (random)
        repeatability_error = self.rng.normal(0, (sensor_spec.repeatability_percent / 100.0) * measurement_range)
        measured_value += repeatability_error
        
        # Apply resolution limit (quantization)
        if sensor_spec.resolution > 0:
            measured_value = np.round(measured_value / sensor_spec.resolution) * sensor_spec.resolution
        
        return measured_value
    
    def simulate_sensor_measurement(self,
                                  sensor_id: str,
                                  true_values: np.ndarray,
                                  time_array: np.ndarray,
                                  environmental_conditions: Optional[Dict] = None) -> List[SensorState]:
        """
        Simulate complete sensor measurement with all error sources
        
        Args:
            sensor_id: Sensor identifier
            true_values: Array of true measurement values
            time_array: Array of time points
            environmental_conditions: Environmental conditions dict
            
        Returns:
            List of SensorState objects
        """
        
        if environmental_conditions is None:
            environmental_conditions = {
                'temperature_C': 25.0,
                'pressure_Pa': 101325.0,
                'humidity_percent': 50.0
            }
        
        sensor_states = []
        measurement_history = []
        
        # Generate noise for entire time series
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.1
        sampling_rate = 1.0 / dt
        duration = time_array[-1] - time_array[0] if len(time_array) > 1 else 1.0
        
        noise_signal = self.generate_sensor_noise(sensor_id, duration, sampling_rate)
        
        # Apply sensor response to true values
        filtered_values = self.apply_sensor_response(sensor_id, true_values, dt)
        
        for i, (t, true_val, filtered_val) in enumerate(zip(time_array, true_values, filtered_values)):
            
            # Calculate drift
            drift = self.calculate_drift(
                sensor_id, 
                t, 
                environmental_conditions['temperature_C'],
                environmental_conditions['pressure_Pa'],
                environmental_conditions['humidity_percent']
            )
            
            # Add measurement errors
            measured_val = self.add_measurement_errors(sensor_id, filtered_val, measurement_history)
            
            # Add noise
            if i < len(noise_signal):
                noise_component = noise_signal[i]
                measured_val += noise_component
            else:
                noise_component = 0.0
            
            # Calculate total uncertainty
            sensor_spec = self.sensor_database[sensor_id]
            accuracy_error = (sensor_spec.accuracy_percent / 100.0) * abs(true_val)
            total_uncertainty = np.sqrt(accuracy_error**2 + noise_component**2)
            
            # Create sensor state
            sensor_state = SensorState(
                true_value=true_val,
                measured_value=measured_val,
                noise_component=noise_component,
                drift_component=drift,
                calibration_error=self.calibration_errors.get(sensor_id, 0.0),
                environmental_error=drift - self.drift_states.get(sensor_id, 0.0),
                total_uncertainty=total_uncertainty,
                timestamp=t,
                sensor_id=sensor_id
            )
            
            sensor_states.append(sensor_state)
            measurement_history.append(measured_val)
            
            # Keep history limited
            if len(measurement_history) > 10:
                measurement_history.pop(0)
        
        return sensor_states
    
    def validate_sensor_model(self, sensor_id: str) -> Dict[str, bool]:
        """
        Validate sensor model against specifications
        
        Args:
            sensor_id: Sensor identifier
            
        Returns:
            Dictionary with validation results
        """
        
        validation_results = {
            'noise_generation': False,
            'response_filtering': False,
            'drift_calculation': False,
            'measurement_errors': False,
            'specifications_met': False
        }
        
        try:
            # Test noise generation
            noise = self.generate_sensor_noise(sensor_id, 10.0, 100.0)
            validation_results['noise_generation'] = len(noise) == 1000
            
            # Test response filtering
            test_signal = np.ones(100)
            filtered = self.apply_sensor_response(sensor_id, test_signal, 0.01)
            validation_results['response_filtering'] = len(filtered) == 100
            
            # Test drift calculation
            drift = self.calculate_drift(sensor_id, 3600.0)  # 1 hour
            validation_results['drift_calculation'] = isinstance(drift, float)
            
            # Test measurement errors
            measured = self.add_measurement_errors(sensor_id, 100.0)
            validation_results['measurement_errors'] = isinstance(measured, float)
            
            # Check if specifications are reasonable
            sensor_spec = self.sensor_database[sensor_id]
            validation_results['specifications_met'] = (
                sensor_spec.accuracy_percent > 0 and
                sensor_spec.resolution > 0 and
                sensor_spec.response_time_s > 0
            )
            
        except Exception as e:
            logger.error(f"Validation failed for sensor {sensor_id}: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = SensorNoiseModel(random_seed=42)
    
    # Test temperature sensor
    sensor_id = 'temperature_rtd_pt100'
    
    print(f"🌡️ Testing Sensor Noise Model for {sensor_id}")
    
    # Initialize sensor with white noise
    noise_params = NoiseParameters(
        noise_type=NoiseType.WHITE_NOISE,
        amplitude=0.1  # 0.1°C noise amplitude
    )
    
    model.initialize_sensor(sensor_id, noise_params, initial_calibration_error=0.05)
    
    # Generate test data
    time_array = np.linspace(0, 100, 1000)  # 100 seconds, 10 Hz sampling
    true_values = 25.0 + 5.0 * np.sin(2 * np.pi * 0.1 * time_array)  # Sinusoidal temperature
    
    # Simulate measurements
    sensor_states = model.simulate_sensor_measurement(sensor_id, true_values, time_array)
    
    print(f"✅ Generated {len(sensor_states)} sensor measurements")
    print(f"✅ Average noise: {np.mean([s.noise_component for s in sensor_states]):.3f}°C")
    print(f"✅ Drift after 100s: {sensor_states[-1].drift_component:.3f}°C")
    print(f"✅ Total uncertainty: {np.mean([s.total_uncertainty for s in sensor_states]):.3f}°C")
    
    # Validate model
    validation = model.validate_sensor_model(sensor_id)
    print(f"✅ Model validation: {sum(validation.values())}/{len(validation)} tests passed")
    
    print("🎉 Sensor noise model working correctly!")