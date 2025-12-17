# Physics Model API Reference - Complete Method Signatures

**Generated**: 2025-12-10  
**Purpose**: Comprehensive API reference for all physics models to fix method signature errors

---

## 1. Ambient Conditions Model

**File**: `src/ambient_conditions.py`  
**Class**: `AmbientConditionModel`

### Methods

#### `calculate_ambient_conditions()`
```python
def calculate_ambient_conditions(
    dry_bulb_temp_K: float,
    relative_humidity_percent: float,
    wind_speed_m_per_s: float = 2.0,
    solar_irradiance_W_per_m2: float = 0.0,
    barometric_pressure_Pa: float = 101325.0,
    timestamp: str = None,
    location: str = "Generic"
) -> AmbientConditions
```

#### `generate_seasonal_weather_pattern()`
```python
def generate_seasonal_weather_pattern(
    location: str = "Generic",
    start_date: datetime = None,
    duration_days: int = 365,
    time_step_hours: int = 1
) -> List[AmbientConditions]
```

**Return Type**: `AmbientConditions` dataclass
- ⚠️ `relative_humidity_percent` (float) - NOT `relative_humidity`
- `dry_bulb_temp_K` (float)
- `wet_bulb_temp_K` (float)
- `dew_point_temp_K` (float)
- `absolute_humidity_kg_per_kg` (float)
- `wind_speed_m_per_s` (float)
- `solar_irradiance_W_per_m2` (float)

---

## 2. Part-Load Cycling Model

**File**: `src/part_load_cycling.py`  
**Class**: `PartLoadCyclingModel`

### Methods

#### `initialize_part_load_state()`
```python
def initialize_part_load_state(
    system_id: str,
    system_type: str,
    design_capacity_kW: float,  # ⚠️ CORRECT: design_capacity_kW
    current_load_kW: float,
    control_type: str = 'residential',
    modulation_type: str = 'single_speed'
) -> PartLoadState
```

#### `update_part_load_operation()`
```python
def update_part_load_operation(
    state: PartLoadState,
    new_load_kW: float,  # ⚠️ CORRECT: new_load_kW
    time_step_minutes: float,  # ⚠️ CORRECT: time_step_minutes
    ambient_temp_K: float = 298.15
) -> PartLoadState
```

**Return Type**: `PartLoadState` dataclass
- `part_load_cop` (float)
- `part_load_capacity_kW` (float)
- `part_load_power_kW` (float)
- `compressor_capacity_percent` (float)
- `is_cycling` (bool)
- `cycles_per_hour` (float)

---

## 3. Control System Dynamics Model

**File**: `src/control_system_dynamics.py`  
**Class**: `ControlSystemModel`

### Methods

#### `initialize_control_system()`
```python
def initialize_control_system(
    system_id: str,
    control_strategy: str = 'cascade'
) -> ControlSystemState
```

**Return Type**: `ControlSystemState` dataclass with **DIRECT ATTRIBUTE ACCESS**:
- `compressor_speed_percent` (float)
- `expansion_valve_opening_percent` (float)
- `evaporator_fan_speed_percent` (float)
- `condenser_fan_speed_percent` (float)
- `temperature_controller` (PIDController)
- `pressure_controller` (PIDController)
- `capacity_controller` (PIDController)

**⚠️ USAGE**: Access attributes directly, NO method calls:
```python
control_state = model.initialize_control_system("sys_001")
speed = control_state.compressor_speed_percent  # Direct access
valve = control_state.expansion_valve_opening_percent  # Direct access
```

---

## 4. Temporal Dynamics Model

**File**: `src/temporal_dynamics.py`  
**Class**: `TemporalDynamicsGenerator`

### Methods

#### `generate_startup_dynamics()`
```python
def generate_startup_dynamics(
    target_conditions: Dict[str, float],
    startup_duration: float = 300.0  # ⚠️ CORRECT: startup_duration (no '_s')
) -> List[HamiltonianState]
```

#### `generate_shutdown_dynamics()`
```python
def generate_shutdown_dynamics(
    initial_steady_state: HamiltonianState,
    shutdown_duration: float = 600.0
) -> List[HamiltonianState]
```

---

## 5. Fouling and Aging Models

**File**: `src/fouling_aging_models.py`  
**Class**: `HeatExchangerFoulingModel`

### Methods

#### `initialize_fouling_state()`
```python
def initialize_fouling_state(
    component_id: str,
    component_type: str,
    age_hours: float = 0
) -> FoulingState
```

#### `update_fouling_state()`
```python
def update_fouling_state(
    fouling_state: FoulingState,
    operating_hours: float,
    operating_conditions: Dict[str, float],
    component_type: str
) -> FoulingState
```

#### `initialize_aging_state()`
```python
def initialize_aging_state(
    component_id: str,
    component_type: str,
    age_hours: float = 0
) -> AgingState
```

---

## 6. Compressor Wear Models

**File**: `src/compressor_wear_models.py`  
**Class**: `CompressorWearModel`

### Methods

#### `initialize_wear_state()`
```python
def initialize_wear_state(
    component_id: str,
    compressor_type: str,
    age_hours: float = 0
) -> CompressorWearState
```

#### `update_wear_state()`
```python
def update_wear_state(
    wear_state: CompressorWearState,
    operating_hours: float,
    operating_conditions: Dict[str, float]
) -> CompressorWearState
```

**Return Type**: `CompressorWearState` dataclass
- `efficiency_degradation` (float)
- `capacity_degradation` (float)
- `power_increase` (float)
- `bearing_wear_percentage` (float)
- `valve_leakage_percentage` (float)
- `oil_quality_index` (float)

---

## 7. Refrigerant Charge Models

**File**: `src/refrigerant_charge_models.py`  
**Class**: `RefrigerantChargeModel`

### Methods

#### `initialize_charge_state()`
```python
def initialize_charge_state(
    system_id: str,
    refrigerant_type: str,
    nominal_charge_kg: float,
    current_charge_percentage: float = 100.0,
    ambient_temp_K: float = 298.15
) -> RefrigerantChargeState
```

#### `simulate_charge_loss()`
```python
def simulate_charge_loss(
    charge_state: RefrigerantChargeState,
    system_age_years: float,
    maintenance_quality: str,
    operating_conditions: Dict[str, float],
    time_period_days: float
) -> RefrigerantChargeState
```

**Return Type**: `RefrigerantChargeState` dataclass
- `cooling_capacity_factor` (float)
- `heating_capacity_factor` (float)
- `cop_cooling_factor` (float)
- `cop_heating_factor` (float)
- `subcooling_K` (float)
- `superheat_K` (float)
- `charge_level_percentage` (float)

---

## 8. Fault Injection Models

**File**: `src/fault_injection_models.py`  
**Class**: `FaultInjectionModel`

### Methods

#### `inject_fault()`
```python
def inject_fault(
    system_id: str,
    component_id: str,
    fault_type: FaultType,
    severity: FaultSeverity,
    onset_time: str = None
) -> FaultState
```

#### `progress_fault()`
```python
def progress_fault(
    fault_state: FaultState,
    time_hours: float,
    operating_conditions: Dict[str, float] = None
) -> FaultState
```

**Return Type**: `FaultState` dataclass
- `capacity_impact_factor` (float)
- `cop_impact_factor` (float)
- `power_impact_factor` (float)
- `current_magnitude` (float)
- `detectable` (bool)

---

## 9. Performance Maps

**File**: `src/performance_maps.py`  
**Class**: `ComponentPerformanceMaps`

### Methods

#### `create_component_health()`
```python
def create_component_health(
    component_id: str,
    component_type: str,
    age_hours: float = 0,
    maintenance_history: List = None
) -> ComponentHealth
```

#### `generate_compressor_performance_map()`
```python
def generate_compressor_performance_map(
    component_health: ComponentHealth
) -> Dict[str, PerformanceMap]
```

**Return Type**: `ComponentHealth` dataclass
- `health_index` (float)
- `efficiency_factor` (float)
- `capacity_factor` (float)
- `degradation_rate` (float)

---

## 10. Sensor Noise Models

**File**: `src/sensor_noise_models.py`  
**Class**: `SensorNoiseModel`

### Methods

#### `initialize_sensor()`
```python
def initialize_sensor(
    sensor_id: str,
    noise_params: NoiseParameters,
    initial_calibration_error: float = 0.0
) -> None
```

#### `simulate_sensor_measurement()`
```python
def simulate_sensor_measurement(
    sensor_id: str,
    true_values: np.ndarray,
    time_array: np.ndarray,
    environmental_conditions: Optional[Dict] = None
) -> List[SensorState]
```

---

## ⚠️ CRITICAL PARAMETER NAME FIXES

### Must Use Exact Names:

1. **Ambient Conditions**:
   - ✅ `relative_humidity_percent` 
   - ❌ NOT `relative_humidity`

2. **Part-Load Cycling**:
   - ✅ `design_capacity_kW` 
   - ❌ NOT `nominal_capacity_kW`
   - ✅ `new_load_kW` 
   - ❌ NOT `current_load_kW`
   - ✅ `time_step_minutes` 
   - ❌ NOT `operating_hours`

3. **Control System**:
   - ✅ Direct attribute access
   - ❌ NO method calls needed
   - Example: `state.compressor_speed_percent`

4. **Temporal Dynamics**:
   - ✅ `startup_duration` 
   - ❌ NOT `startup_duration_s`

5. **All Models**:
   - `operating_conditions` is always `Dict[str, float]`
   - `age_hours` is always `float` (operating hours)
   - Return types are dataclass objects - use dot notation

---

## Common Patterns

### Initialization Pattern
```python
# Most models follow this pattern
state = model.initialize_xxx_state(
    component_id="comp_001",
    component_type="scroll_compressor",
    age_hours=5000
)
```

### Update Pattern
```python
# Most models follow this pattern
updated_state = model.update_xxx_state(
    state=current_state,
    operating_hours=100.0,
    operating_conditions={
        'temperature': 298.15,
        'pressure': 101325.0,
        'load_factor': 0.8
    }
)
```

### Accessing Dataclass Attributes
```python
# All dataclass returns use dot notation
value = state.attribute_name  # Direct access
# NOT state.get_attribute_name()  # Wrong!