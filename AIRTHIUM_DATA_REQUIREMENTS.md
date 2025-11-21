# AIRTHIUM Heat Pump Data Requirements for HNN/GNN Training

## Executive Summary

This document outlines the comprehensive data requirements for training Hamiltonian Neural Networks (HNN) and Graph Neural Networks (GNN) to accurately predict Coefficient of Performance (COP) for the AIRTHIUM heat pump prototype. The goal is to create physics-informed models that capture real-world complexity and provide reliable COP predictions under varying operating conditions.

## 1. AIRTHIUM System Architecture

### 1.1 Physical Components
- **Compressor**: Variable-speed scroll compressor with inverter drive
- **Evaporator**: Finned-tube air-to-refrigerant heat exchanger
- **Condenser**: Plate heat exchanger (water-to-refrigerant)
- **Expansion Device**: Electronic expansion valve (EEV)
- **Refrigerant**: R32 or R410A working fluid
- **Control System**: Microprocessor-based with multiple sensors

### 1.2 Operating Envelope
- **Heating Capacity**: 5-15 kW
- **Operating Temperature Range**: -20°C to +35°C ambient
- **Water Temperature Range**: 35°C to 65°C supply
- **COP Range**: 2.0 to 6.0 depending on conditions
- **Modulation Range**: 20% to 100% capacity

## 2. Hamiltonian Neural Network (HNN) Data Requirements

### 2.1 Temporal Dynamics (Critical for HNN)

#### 2.1.1 Transient Behavior
- **Startup Dynamics**: 0-300 seconds after system start
  - Compressor ramp-up profiles (0-100% speed over 30-60s)
  - Refrigerant migration and distribution
  - Heat exchanger thermal mass effects
  - Control system initialization sequences
  - Oil circulation dynamics and viscosity changes

- **Shutdown Dynamics**: 0-600 seconds after system stop
  - Compressor coast-down (exponential decay)
  - Pressure equalization across system
  - Thermal decay in heat exchangers
  - Refrigerant settling patterns

- **Load Change Response**: 10-120 seconds
  - Capacity modulation dynamics (inverter response)
  - Superheat control response (EEV positioning)
  - Pressure transients and oscillations
  - Temperature overshoots/undershoots

#### 2.1.2 Cyclic Behavior
- **Defrost Cycles**: Every 30-90 minutes in heating mode
  - Reverse valve switching dynamics (2-5 second transition)
  - Frost accumulation/melting (ice crystal formation)
  - Performance degradation/recovery curves
  - Energy penalty quantification (10-15% COP reduction)

- **Compressor Cycling**: For on/off systems
  - Minimum on/off times (3-5 minutes)
  - Cycling losses and thermal inertia
  - Pressure equalization delays

#### 2.1.3 Phase-Space Trajectories
- **Thermodynamic State Evolution**: P-h, T-s diagrams over time
  - Continuous state tracking through all processes
  - Non-equilibrium states during transients
  - Hysteresis effects in heat exchangers
  - Refrigerant quality variations (0-1 dryness fraction)

- **Energy Conservation Tracking**:
  - Total system energy (internal + kinetic + potential)
  - Energy flow rates between components
  - Entropy generation rates (irreversibilities)
  - Exergy destruction analysis

### 2.2 Hamiltonian Formulation

#### 2.2.1 Generalized Coordinates (q)
- **Pressure States**: P₁, P₂, P₃, P₄ at each component
- **Volume States**: V₁, V₂, V₃, V₄ (specific volumes)
- **Temperature States**: T₁, T₂, T₃, T₄ at each state point
- **Entropy States**: S₁, S₂, S₃, S₄ (specific entropy)
- **Mass Distribution**: m₁, m₂, m₃, m₄ (refrigerant mass in each component)

#### 2.2.2 Generalized Momenta (p)
- **Pressure Rates**: dP/dt at each component
- **Volume Rates**: dV/dt (expansion/compression rates)
- **Temperature Rates**: dT/dt (heating/cooling rates)
- **Entropy Rates**: dS/dt (entropy generation rates)
- **Mass Flow Rates**: dm/dt between components

#### 2.2.3 Hamiltonian Function H(q,p)
- **Internal Energy**: U = Σ(m_i × u_i) for each component
- **Kinetic Energy**: K = ½Σ(ṁ_i × v_i²) for refrigerant flow
- **Potential Energy**: Φ = Σ(m_i × g × h_i) for elevation effects
- **Total Hamiltonian**: H = U + K + Φ

### 2.3 Time-Series Data Structure
```
Time Resolution: 0.1 seconds (10 Hz sampling)
Duration: Variable (300s startup, 3600s steady-state, 600s shutdown)

For each timestep t:
- q(t) = [P₁(t), V₁(t), T₁(t), S₁(t), m₁(t), ...]
- p(t) = [dP₁/dt(t), dV₁/dt(t), dT₁/dt(t), dS₁/dt(t), dm₁/dt(t), ...]
- H(t) = H(q(t), p(t))
- COP(t) = target variable
```

## 3. Graph Neural Network (GNN) Data Requirements

### 3.1 Component Graph Topology

#### 3.1.1 Node Definitions
- **Compressor Node**: 
  - Features: [speed_rpm, power_kW, efficiency, discharge_temp, vibration]
  - Type: "compressor"
  - Connections: [suction_line, discharge_line, oil_circuit]

- **Evaporator Node**:
  - Features: [heat_transfer_kW, effectiveness, fouling_factor, air_flow, frost_thickness]
  - Type: "heat_exchanger"
  - Connections: [refrigerant_in, refrigerant_out, air_side]

- **Condenser Node**:
  - Features: [heat_transfer_kW, effectiveness, fouling_factor, water_flow, approach_temp]
  - Type: "heat_exchanger"
  - Connections: [refrigerant_in, refrigerant_out, water_side]

- **Expansion Valve Node**:
  - Features: [opening_percent, flow_coefficient, superheat_control, pressure_drop]
  - Type: "expansion_device"
  - Connections: [high_pressure_side, low_pressure_side]

#### 3.1.2 Edge Definitions
- **Refrigerant Flow Edges**:
  - Features: [mass_flow_rate, enthalpy, pressure, temperature, quality]
  - Direction: Directional (following refrigerant flow)
  - Type: "refrigerant_flow"

- **Heat Transfer Edges**:
  - Features: [heat_rate, temperature_difference, heat_transfer_coefficient]
  - Direction: Bidirectional
  - Type: "heat_transfer"

- **Control Signal Edges**:
  - Features: [signal_value, response_time, control_error]
  - Direction: Directional (from sensor to actuator)
  - Type: "control_signal"

#### 3.1.3 Graph Adjacency Matrix
```
Components: [Compressor, Evaporator, Condenser, ExpansionValve, Controller]
Adjacency Matrix (5x5):
         Comp  Evap  Cond  ExpV  Ctrl
Comp  [  0     1     0     0     1  ]
Evap  [  1     0     1     0     1  ]
Cond  [  0     1     0     1     1  ]
ExpV  [  0     0     1     0     1  ]
Ctrl  [  1     1     1     1     0  ]
```

### 3.2 Multi-Scale Graph Representation

#### 3.2.1 System-Level Graph
- Nodes: Major components (4-6 nodes)
- Edges: Primary refrigerant flow and control connections
- Features: Aggregate performance metrics

#### 3.2.2 Component-Level Graph
- Nodes: Sub-components within each major component
- Edges: Internal heat/mass transfer and mechanical connections
- Features: Detailed component states

#### 3.2.3 Process-Level Graph
- Nodes: Individual thermodynamic processes
- Edges: Process sequences and interactions
- Features: Process-specific parameters

## 4. Real-World Complexity Factors

### 4.1 Environmental Variations
- **Ambient Temperature**: -20°C to +35°C (hourly variations)
- **Humidity**: 20% to 95% RH (affects frost formation)
- **Wind Speed**: 0-15 m/s (affects heat exchanger performance)
- **Solar Radiation**: 0-1000 W/m² (affects outdoor unit temperature)
- **Seasonal Variations**: Performance maps for each season

### 4.2 Component Degradation
- **Heat Exchanger Fouling**: 
  - Fouling factor increase: 0.0001-0.002 m²K/W over time
  - Performance degradation: 5-20% over 5 years
  - Cleaning cycle effects

- **Compressor Wear**:
  - Efficiency degradation: 0.5-2% per year
  - Vibration increase: 10-50% over lifetime
  - Oil degradation effects

- **Refrigerant Charge Variations**:
  - Undercharge: -10% to -30% (COP reduction 5-25%)
  - Overcharge: +10% to +20% (COP reduction 3-15%)
  - Leak rates: 1-5% per year

### 4.3 Control System Dynamics
- **PID Controller Parameters**: Kp, Ki, Kd for superheat control
- **Setpoint Changes**: Step responses and ramp responses
- **Sensor Noise**: ±0.5°C temperature, ±1% pressure
- **Actuator Delays**: 2-10 seconds for valve positioning

### 4.4 Fault Conditions
- **Refrigerant Leaks**: 5-50% charge loss
- **Sensor Drift**: ±2°C over 2 years
- **Fouling**: 10-80% heat transfer reduction
- **Compressor Faults**: Bearing wear, valve leakage
- **Control Faults**: Sensor failures, communication errors

## 5. Data Generation Strategy

### 5.1 Simulation Hierarchy
1. **Physics-Based Models**: Detailed component models with real thermodynamic properties
2. **Empirical Correlations**: Performance maps from manufacturer data
3. **Stochastic Variations**: Random variations within realistic bounds
4. **Fault Injection**: Systematic introduction of common faults

### 5.2 Validation Requirements
- **Energy Conservation**: <1% error in energy balance
- **Physics Constraints**: All states within physical limits
- **Performance Bounds**: COP within realistic ranges (1.5-7.0)
- **Temporal Consistency**: Smooth state transitions

### 5.3 Dataset Composition
- **Steady-State Operations**: 60% of data
- **Transient Operations**: 25% of data
- **Fault Conditions**: 10% of data
- **Extreme Conditions**: 5% of data

## 6. Training Data Structure

### 6.1 HNN Training Data Format
```python
{
    "cycle_id": "unique_identifier",
    "timestamp": "ISO_8601_timestamp",
    "duration": 3600,  # seconds
    "sampling_rate": 10,  # Hz
    "coordinates": {
        "P": [[P1_t0, P2_t0, P3_t0, P4_t0], [P1_t1, ...], ...],
        "V": [[V1_t0, V2_t0, V3_t0, V4_t0], [V1_t1, ...], ...],
        "T": [[T1_t0, T2_t0, T3_t0, T4_t0], [T1_t1, ...], ...],
        "S": [[S1_t0, S2_t0, S3_t0, S4_t0], [S1_t1, ...], ...],
        "m": [[m1_t0, m2_t0, m3_t0, m4_t0], [m1_t1, ...], ...]
    },
    "momenta": {
        "dP_dt": [[dP1_dt_t0, ...], [dP1_dt_t1, ...], ...],
        "dV_dt": [[dV1_dt_t0, ...], [dV1_dt_t1, ...], ...],
        "dT_dt": [[dT1_dt_t0, ...], [dT1_dt_t1, ...], ...],
        "dS_dt": [[dS1_dt_t0, ...], [dS1_dt_t1, ...], ...],
        "dm_dt": [[dm1_dt_t0, ...], [dm1_dt_t1, ...], ...]
    },
    "hamiltonian": [H_t0, H_t1, H_t2, ...],
    "targets": {
        "COP_heating": [COP_t0, COP_t1, COP_t2, ...],
        "COP_cooling": [COP_t0, COP_t1, COP_t2, ...],
        "efficiency": [eff_t0, eff_t1, eff_t2, ...]
    },
    "metadata": {
        "operating_mode": "heating/cooling",
        "ambient_temp": 5.0,
        "load_fraction": 0.75,
        "fault_conditions": []
    }
}
```

### 6.2 GNN Training Data Format
```python
{
    "graph_id": "unique_identifier",
    "timestamp": "ISO_8601_timestamp",
    "nodes": {
        "compressor": {
            "features": [speed, power, efficiency, temp, vibration],
            "type": "compressor",
            "position": [x, y, z]
        },
        "evaporator": {
            "features": [heat_rate, effectiveness, fouling, airflow, frost],
            "type": "heat_exchanger",
            "position": [x, y, z]
        },
        # ... other nodes
    },
    "edges": {
        "comp_to_cond": {
            "features": [mass_flow, enthalpy, pressure, temp, quality],
            "type": "refrigerant_flow",
            "source": "compressor",
            "target": "condenser"
        },
        # ... other edges
    },
    "adjacency_matrix": [[0, 1, 0, 0, 1], [1, 0, 1, 0, 1], ...],
    "targets": {
        "COP_heating": 4.2,
        "COP_cooling": 3.8,
        "system_efficiency": 0.85
    },
    "metadata": {
        "operating_conditions": {...},
        "component_health": {...},
        "control_settings": {...}
    }
}
```

## 7. Implementation Priorities

### Phase 1: Foundation (Weeks 1-2)
1. Implement basic temporal dynamics for HNN
2. Create component graph topology for GNN
3. Generate baseline training datasets

### Phase 2: Realism (Weeks 3-4)
1. Add environmental variations and degradation
2. Implement fault injection models
3. Create control system dynamics

### Phase 3: Integration (Weeks 5-6)
1. Combine HNN and GNN architectures
2. Implement physics-informed loss functions
3. Validate against known physics

### Phase 4: Optimization (Weeks 7-8)
1. Hyperparameter tuning and model selection
2. Performance benchmarking
3. Real-time inference optimization

## 8. Success Metrics

### 8.1 Data Quality Metrics
- **Physics Compliance**: >99% of generated data passes physics validation
- **Temporal Consistency**: <0.1% discontinuities in time series
- **Component Realism**: Performance within ±5% of manufacturer specifications

### 8.2 Model Performance Metrics
- **COP Prediction Accuracy**: RMSE <0.2 for COP predictions
- **Energy Conservation**: <1% error in predicted energy balance
- **Transient Prediction**: <5% error in transient response prediction

### 8.3 Operational Metrics
- **Inference Speed**: <10ms for real-time COP prediction
- **Memory Usage**: <100MB for deployed model
- **Robustness**: <2% performance degradation under sensor noise

This comprehensive data requirements document provides the foundation for creating realistic, physics-informed training data for both HNN and GNN models targeting the AIRTHIUM heat pump prototype.