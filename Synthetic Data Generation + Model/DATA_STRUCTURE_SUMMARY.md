# Data Structure Summary - Quick Reference

## Overview
We're creating **21 separate HDF5 files**, one per data-generating physics model, to avoid NaNs and enable efficient loading.

## File Structure
```
training_dataset_500/
├── 1_hamiltonian_data.h5          # HNN: Trajectories (q, p, H)
├── 2_temporal_dynamics.h5         # HNN: T, P, flow time series
├── 3_control_system.h5            # HNN: Control signals, PID
├── 4_transients.h5                # HNN: Startup/shutdown
├── 5_sensor_data.h5               # HNN: Noisy measurements
├── 6_part_load.h5                 # HNN: Part-load operation
├── 7_phase_space.h5               # HNN: Phase space analysis
├── 8_component_interactions.h5    # GNN: Node features (10 nodes)
├── 9_gnn_topology.h5              # GNN: Edge features (11 edges)
├── 10_fouling_aging.h5            # GNN: Degradation
├── 11_compressor_wear.h5          # GNN: Compressor-specific
├── 12_faults.h5                   # GNN: Fault states
├── 13_performance_maps.h5         # GNN: Component maps
├── 14_component_models.h5         # GNN: Detailed physics
├── 15_ambient_conditions.h5       # Context: Outdoor
├── 16_indoor_performance.h5       # Context: Indoor
├── 17_building_load.h5            # Context: Loads
├── 18_refrigerant_charge.h5       # Context: Charge
├── 19_molecular_refrigerant.h5    # Context: Properties
├── 20_thermodynamic_properties.h5 # Context: State points
├── 21_targets.h5                  # Targets: COP, etc.
└── dataset_info.json              # Metadata
```

## Model Categories

### HNN Branch (7 files - Temporal, 100 timesteps)
1. **Hamiltonian** - Core dynamics (q, p, H, KE, PE)
2. **Temporal** - Temperature, pressure, flow
3. **Control** - Control signals, PID response
4. **Transients** - Startup/shutdown behavior
5. **Sensors** - Noisy measurements
6. **Part-load** - Load ratio, cycling
7. **Phase space** - Trajectory analysis

**Total HNN features: ~49 per timestep**

### GNN Branch (7 files - Static per cycle)
8. **Component interactions** - Node features (10 nodes × 15 features)
9. **GNN topology** - Edge features (11 edges × 8 features)
10. **Fouling/aging** - Degradation per component
11. **Compressor wear** - Compressor-specific wear
12. **Faults** - Fault injection per component
13. **Performance maps** - Component efficiency maps
14. **Component models** - Detailed component physics

**Total GNN features: 35 per node, 12 per edge**

### Context (6 files - Graph-level, static per cycle)
15. **Ambient** - Outdoor conditions
16. **Indoor** - Indoor conditions
17. **Building load** - Heating/cooling loads
18. **Refrigerant charge** - Charge state
19. **Molecular refrigerant** - Refrigerant properties
20. **Thermodynamic** - State point properties

**Total context features: ~40 graph-level**

### Targets (1 file)
21. **Targets** - COP (primary), capacity, power, validation

## Data Shapes

### Temporal Data (HNN)
- Shape: `[N_cycles, 100_timesteps, features]`
- Example: `hamiltonian_data.h5/trajectories/q` → `[500, 100, 4]`

### Static Data (GNN)
- Node features: `[N_cycles, 10_nodes, features]`
- Edge features: `[N_cycles, 11_edges, features]`
- Graph features: `[N_cycles, features]`

### Targets
- Shape: `[N_cycles]` or `[N_cycles, 1]`
- Example: `targets.h5/performance/COP` → `[500]`

## Advantages

✅ **No NaNs** - Each model stores only its data
✅ **No empty values** - Dense arrays only
✅ **Flexible loading** - Load only what you need
✅ **Clear separation** - Temporal vs static
✅ **Efficient storage** - Compressed HDF5
✅ **Easy debugging** - One file per model
✅ **Scalable** - Add models independently

## Loading Strategy

```python
# Load only HNN models
hnn_data = load_models([
    'hamiltonian_data',
    'temporal_dynamics',
    'control_system',
    'transients',
    'sensor_data',
    'part_load',
    'phase_space'
])

# Load only GNN models
gnn_data = load_models([
    'component_interactions',
    'gnn_topology',
    'fouling_aging',
    'compressor_wear',
    'faults',
    'performance_maps',
    'component_models'
])

# Load context
context_data = load_models([
    'ambient_conditions',
    'indoor_performance',
    'building_load',
    'refrigerant_charge',
    'molecular_refrigerant',
    'thermodynamic_properties'
])

# Load targets
targets = load_model('targets')
```

## Next Steps

1. ✅ Complete physics model mapping
2. ⏳ Implement modular dataset generator
3. ⏳ Generate 500-cycle test dataset
4. ⏳ Implement modular data loader
5. ⏳ Update hybrid model for new structure
6. ⏳ Train and validate

## Key Insight

**Separation by physics model = Clean data structure**
- Each model knows its own temporal/static nature
- No mixing of incompatible data types
- Easy to extend with new models
- Clear data lineage and debugging

This structure is optimal for the HNN-GNN hybrid architecture!