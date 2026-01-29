# Complete Physics Model Mapping for HNN-GNN Architecture

## All 28 Physics Models in src/

### Analysis of Each Model

1. **hamiltonian_mechanics.py** - ✅ Temporal (100 steps)
   - Generates: q, p, H, KE, PE trajectories
   - Use: HNN branch primary input

2. **temporal_dynamics.py** - ✅ Temporal (100 steps)
   - Generates: Temperature, pressure, flow time series
   - Use: HNN branch additional temporal features

3. **control_system_dynamics.py** - ✅ Temporal (100 steps)
   - Generates: Control signals, PID response
   - Use: HNN branch control inputs

4. **startup_shutdown_transients.py** - ✅ Temporal (100 steps)
   - Generates: Transient phases, stability metrics
   - Use: HNN branch transient behavior

5. **sensor_noise_models.py** - ✅ Temporal (100 steps)
   - Generates: Noisy sensor measurements over time
   - Use: HNN branch realistic measurements

6. **part_load_cycling.py** - ✅ Temporal (100 steps)
   - Generates: Part-load operation, cycling behavior
   - Use: HNN branch operating modes

7. **phase_space_trajectories.py** - ✅ Temporal (100 steps)
   - Generates: Phase space analysis, attractors
   - Use: HNN branch trajectory analysis

8. **component_interactions.py** - ✅ Static + GNN
   - Generates: Node features for 10 components
   - Use: GNN branch node features

9. **gnn_topology.py** - ✅ Static + GNN
   - Generates: Graph structure, edge features
   - Use: GNN branch graph topology

10. **fouling_aging_models.py** - ✅ Static + GNN
    - Generates: Degradation per component
    - Use: GNN branch node degradation features

11. **compressor_wear_models.py** - ✅ Static + GNN
    - Generates: Compressor-specific wear
    - Use: GNN branch compressor node features

12. **fault_injection_models.py** - ✅ Static + GNN
    - Generates: Fault states per component
    - Use: GNN branch fault node features

13. **performance_maps.py** - ✅ Static + GNN
    - Generates: Component performance characteristics
    - Use: GNN branch component efficiency features

14. **component_models.py** - ✅ Static + GNN
    - Generates: Component-level physics
    - Use: GNN branch detailed component features

15. **ambient_conditions.py** - ✅ Static (Context)
    - Generates: Outdoor temperature, humidity, wind
    - Use: GNN graph-level features

16. **indoor_performance_models.py** - ✅ Static (Context)
    - Generates: Indoor conditions, comfort metrics
    - Use: GNN graph-level features

17. **building_load_models.py** - ✅ Static (Context)
    - Generates: Heating/cooling loads, profiles
    - Use: GNN graph-level features

18. **refrigerant_charge_models.py** - ✅ Static (Context)
    - Generates: Charge amount, subcooling
    - Use: GNN graph-level features

19. **molecular_refrigerant_properties.py** - ✅ Static (Context)
    - Generates: Molecular properties
    - Use: GNN graph-level features

20. **thermodynamic_properties.py** - ✅ Static (Context)
    - Generates: Fluid properties at states
    - Use: Both HNN and GNN context

21. **cycle_generators.py** - ✅ Infrastructure
    - Generates: Base thermodynamic cycles
    - Use: Foundation for all other models

22. **improved_cycle_generators.py** - ✅ Infrastructure
    - Generates: Enhanced cycle generation
    - Use: Foundation for all other models

23. **data_generation_pipeline.py** - ✅ Infrastructure
    - Orchestrates: Data generation
    - Use: Pipeline management

24. **enhanced_data_pipeline.py** - ✅ Infrastructure
    - Orchestrates: Enhanced generation
    - Use: Pipeline management

25. **digital_twin_validation.py** - ✅ Validation
    - Validates: Against real data
    - Use: Quality assurance

26. **physics_validation_system.py** - ✅ Validation
    - Validates: Physics constraints
    - Use: Quality assurance

27. **enhanced_validator.py** - ✅ Validation
    - Validates: Enhanced checks
    - Use: Quality assurance

28. **integrated_validation_pipeline.py** - ✅ Validation
    - Orchestrates: All validation
    - Use: Quality assurance

## Revised Data Structure

### HNN Branch - Temporal Models (7 models, 100 timesteps each)

#### 1. Hamiltonian Trajectories
```
hamiltonian_data.h5
├── trajectories/
│   ├── q [N, 100, 4]              # Positions
│   ├── p [N, 100, 4]              # Momenta
│   ├── hamiltonian [N, 100]       # Total energy
│   ├── kinetic_energy [N, 100]
│   └── potential_energy [N, 100]
```

#### 2. Temporal Dynamics
```
temporal_dynamics.h5
├── temperatures/
│   ├── T_evap [N, 100]
│   ├── T_cond [N, 100]
│   ├── T_comp_discharge [N, 100]
│   └── T_superheat [N, 100]
├── pressures/
│   ├── P_evap [N, 100]
│   ├── P_cond [N, 100]
│   └── P_ratio [N, 100]
├── flows/
│   ├── mass_flow [N, 100]
│   ├── volume_flow [N, 100]
│   └── velocity [N, 100]
```

#### 3. Control System
```
control_system.h5
├── control_signals/
│   ├── compressor_speed [N, 100]
│   ├── expansion_valve_opening [N, 100]
│   ├── fan_speed_evap [N, 100]
│   └── fan_speed_cond [N, 100]
├── pid_response/
│   ├── error [N, 100]
│   ├── integral [N, 100]
│   ├── derivative [N, 100]
│   └── control_output [N, 100]
```

#### 4. Startup/Shutdown Transients
```
transients.h5
├── phases/
│   ├── phase_indicator [N, 100]   # 0=off, 1=startup, 2=steady, 3=shutdown
│   ├── transient_magnitude [N, 100]
│   └── stability_metric [N, 100]
├── rates/
│   ├── temperature_rate [N, 100]
│   └── pressure_rate [N, 100]
```

#### 5. Sensor Measurements (with noise)
```
sensor_data.h5
├── measurements/
│   ├── temperature_sensors [N, 100, 10]  # 10 T sensors
│   ├── pressure_sensors [N, 100, 6]      # 6 P sensors
│   ├── flow_sensors [N, 100, 3]          # 3 flow sensors
│   └── power_sensors [N, 100, 2]         # 2 power sensors
├── noise_params/
│   ├── sensor_std [N, 21]                # Std dev per sensor
│   └── sensor_bias [N, 21]               # Bias per sensor
```

#### 6. Part-Load Cycling
```
part_load.h5
├── operation/
│   ├── load_ratio [N, 100]               # 0-1 load
│   ├── cycling_state [N, 100]            # On/off cycling
│   ├── capacity_modulation [N, 100]
│   └── efficiency_factor [N, 100]
```

#### 7. Phase Space Trajectories
```
phase_space.h5
├── trajectories/
│   ├── state_vector [N, 100, 8]          # Full state
│   ├── attractor_distance [N, 100]
│   └── lyapunov_exponent [N, 100]
```

### GNN Branch - Component/Graph Models (9 models)

#### 8. Component Interactions (Node Features)
```
component_interactions.h5
├── nodes/                                 # 10 components
│   ├── compressor [N, 15]
│   ├── condenser [N, 15]
│   ├── expansion_valve [N, 15]
│   ├── evaporator [N, 15]
│   ├── accumulator [N, 15]
│   ├── liquid_line [N, 15]
│   ├── suction_line [N, 15]
│   ├── discharge_line [N, 15]
│   ├── fan_evap [N, 15]
│   └── fan_cond [N, 15]
├── interactions/
│   ├── heat_transfer_matrix [N, 10, 10]
│   └── mass_transfer_matrix [N, 10, 10]
```

#### 9. GNN Topology (Edge Features)
```
gnn_topology.h5
├── edges/                                 # 11 edges
│   ├── edge_features [N, 11, 8]
│   ├── heat_transfer_coeff [N, 11]
│   ├── mass_flow_rate [N, 11]
│   ├── pressure_drop [N, 11]
│   └── temperature_diff [N, 11]
├── topology/
│   └── edge_index [2, 11]                # Fixed structure
```

#### 10. Fouling & Aging (Node Degradation)
```
fouling_aging.h5
├── fouling/
│   ├── evaporator_fouling [N]
│   ├── condenser_fouling [N]
│   └── fouling_rate [N, 10]              # Per component
├── aging/
│   ├── system_age_hours [N]
│   ├── efficiency_degradation [N, 10]
│   └── capacity_degradation [N, 10]
```

#### 11. Compressor Wear (Specific Node)
```
compressor_wear.h5
├── wear_metrics/
│   ├── bearing_wear [N]
│   ├── valve_wear [N]
│   ├── piston_wear [N]
│   └── motor_degradation [N]
├── performance_impact/
│   ├── efficiency_loss [N]
│   ├── capacity_loss [N]
│   └── power_increase [N]
```

#### 12. Fault Injection (Node Faults)
```
faults.h5
├── fault_flags/
│   ├── has_fault [N]
│   ├── fault_component [N]               # Which component
│   ├── fault_type [N]                    # Type of fault
│   └── fault_severity [N]                # 0-1 scale
├── fault_effects/
│   ├── efficiency_impact [N, 10]         # Per component
│   ├── capacity_impact [N, 10]
│   └── power_impact [N, 10]
```

#### 13. Performance Maps (Component Characteristics)
```
performance_maps.h5
├── compressor_map/
│   ├── efficiency_map [N, 20, 20]        # PR vs flow
│   ├── power_map [N, 20, 20]
│   └── capacity_map [N, 20, 20]
├── heat_exchanger_maps/
│   ├── UA_evap [N]
│   ├── UA_cond [N]
│   └── effectiveness [N, 2]              # Evap, cond
```

#### 14. Component Models (Detailed Physics)
```
component_models.h5
├── compressor/
│   ├── isentropic_efficiency [N]
│   ├── volumetric_efficiency [N]
│   ├── mechanical_efficiency [N]
│   └── displacement [N]
├── heat_exchangers/
│   ├── evap_area [N]
│   ├── cond_area [N]
│   ├── evap_circuits [N]
│   └── cond_circuits [N]
├── expansion_valve/
│   ├── valve_coefficient [N]
│   └── opening_range [N]
```

### Context Models (Graph-Level Features) (6 models)

#### 15. Ambient Conditions
```
ambient_conditions.h5
├── outdoor/
│   ├── temperature [N]
│   ├── humidity [N]
│   ├── pressure [N]
│   ├── wind_speed [N]
│   └── solar_radiation [N]
├── seasonal/
│   ├── season [N]                        # 0-3
│   └── time_of_day [N]                   # 0-23
```

#### 16. Indoor Performance
```
indoor_performance.h5
├── indoor/
│   ├── temperature [N]
│   ├── humidity [N]
│   ├── occupancy [N]
│   └── comfort_index [N]
├── setpoints/
│   ├── heating_setpoint [N]
│   └── cooling_setpoint [N]
```

#### 17. Building Load
```
building_load.h5
├── loads/
│   ├── heating_load [N]
│   ├── cooling_load [N]
│   ├── sensible_load [N]
│   └── latent_load [N]
├── profiles/
│   ├── hourly_profile [N, 24]
│   └── daily_pattern [N]
```

#### 18. Refrigerant Charge
```
refrigerant_charge.h5
├── charge/
│   ├── charge_amount [N]
│   ├── charge_deviation [N]              # From optimal
│   ├── subcooling [N]
│   └── superheat [N]
├── effects/
│   ├── efficiency_impact [N]
│   └── capacity_impact [N]
```

#### 19. Molecular Refrigerant Properties
```
molecular_refrigerant.h5
├── properties/
│   ├── molecular_weight [N]
│   ├── critical_temp [N]
│   ├── critical_pressure [N]
│   ├── acentric_factor [N]
│   └── GWP [N]                           # Global warming potential
├── refrigerant_type/
│   └── refrigerant_name [N]              # String
```

#### 20. Thermodynamic Properties (State Points)
```
thermodynamic_properties.h5
├── state_points/                          # 4-6 states per cycle
│   ├── pressure [N, 6]
│   ├── temperature [N, 6]
│   ├── enthalpy [N, 6]
│   ├── entropy [N, 6]
│   ├── density [N, 6]
│   └── quality [N, 6]
├── derived/
│   ├── specific_volume [N, 6]
│   ├── cp [N, 6]
│   └── cv [N, 6]
```

### Target Variables

#### 21. Targets
```
targets.h5
├── performance/
│   ├── COP [N]                           # PRIMARY TARGET
│   ├── capacity_heating [N]
│   ├── capacity_cooling [N]
│   ├── power_consumption [N]
│   ├── EER [N]
│   └── SEER [N]
├── validation/
│   ├── energy_balance_error [N]
│   ├── entropy_generation [N]
│   ├── carnot_efficiency [N]
│   └── physics_valid [N]
```

## Summary

### Total: 21 Data-Generating Models

**HNN Branch (Temporal - 7 models):**
1. Hamiltonian mechanics
2. Temporal dynamics
3. Control system
4. Startup/shutdown transients
5. Sensor measurements
6. Part-load cycling
7. Phase space trajectories

**GNN Branch (Static/Graph - 9 models):**
8. Component interactions (nodes)
9. GNN topology (edges)
10. Fouling & aging
11. Compressor wear
12. Fault injection
13. Performance maps
14. Component models

**Context (Graph-level - 6 models):**
15. Ambient conditions
16. Indoor performance
17. Building load
18. Refrigerant charge
19. Molecular refrigerant
20. Thermodynamic properties

**Targets:**
21. Performance metrics & validation

**Infrastructure (7 models - not data files):**
- Cycle generators (2)
- Data pipelines (2)
- Validators (3)

## Feature Dimensions

### HNN Input: [batch, 100, 49]
- Hamiltonian: 11 features (q:4, p:4, H:1, KE:1, PE:1)
- Temporal: 10 features (T:4, P:3, flow:3)
- Control: 8 features (signals:4, PID:4)
- Transients: 5 features
- Sensors: 21 features (compressed from 21 sensors)
- Part-load: 4 features
- Phase space: 10 features (state:8, attractor:1, lyapunov:1)

### GNN Input:
- Node features: [10 nodes, 35 features]
  - Component interactions: 15
  - Degradation: 10
  - Performance: 10
  
- Edge features: [11 edges, 12 features]
  - Topology: 8
  - Heat transfer: 2
  - Mass flow: 2
  
- Graph features: [40 features]
  - Ambient: 7
  - Indoor: 6
  - Building: 8
  - Refrigerant: 9
  - Thermodynamic: 10

This is the COMPLETE mapping of all physics models!