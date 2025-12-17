#!/usr/bin/env python3
"""
Validate Modular Dataset
========================

Checks that the modular dataset has:
1. Correct dimensions (500 cycles, 100 timesteps)
2. Real physics values (not zeros or NaNs)
3. Proper data ranges
"""

import h5py
import numpy as np
import json
from pathlib import Path


def validate_dataset(data_dir="modular_dataset_500"):
    """Validate the modular dataset."""
    
    print("=" * 70)
    print("VALIDATING MODULAR DATASET")
    print("=" * 70)
    
    data_path = Path(data_dir)
    
    # Load metadata
    with open(data_path / "dataset_info.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Cycles: {metadata['n_cycles']}")
    print(f"  Timesteps: {metadata['n_timesteps']}")
    if 'files' in metadata:
        print(f"  Files: {len(metadata['files'])}")
    
    # Check each HDF5 file
    print(f"\n{'='*70}")
    print("CHECKING HDF5 FILES")
    print(f"{'='*70}")
    
    results = {}
    
    # 1. Hamiltonian data (temporal)
    print(f"\n1. Hamiltonian Data (1_hamiltonian_data.h5)")
    with h5py.File(data_path / "1_hamiltonian_data.h5", 'r') as f:
        q = f['trajectories/q'][:]
        p = f['trajectories/p'][:]
        H = f['trajectories/hamiltonian'][:]
        
        print(f"   q shape: {q.shape} (expected: [500, 100, 4])")
        print(f"   p shape: {p.shape} (expected: [500, 100, 4])")
        print(f"   H shape: {H.shape} (expected: [500, 100])")
        print(f"   q range: [{q.min():.4f}, {q.max():.4f}]")
        print(f"   p range: [{p.min():.4f}, {p.max():.4f}]")
        print(f"   H range: [{H.min():.4f}, {H.max():.4f}]")
        print(f"   Has NaNs: q={np.isnan(q).any()}, p={np.isnan(p).any()}, H={np.isnan(H).any()}")
        print(f"   Has zeros only: q={np.all(q==0)}, p={np.all(p==0)}, H={np.all(H==0)}")
        
        results['hamiltonian'] = {
            'shape_correct': q.shape == (500, 100, 4) and p.shape == (500, 100, 4),
            'has_values': not (np.all(q==0) or np.all(p==0)),
            'no_nans': not (np.isnan(q).any() or np.isnan(p).any())
        }
    
    # 2. Temporal dynamics
    print(f"\n2. Temporal Dynamics (2_temporal_dynamics.h5)")
    with h5py.File(data_path / "2_temporal_dynamics.h5", 'r') as f:
        T_evap = f['temperatures/T_evap'][:]
        T_cond = f['temperatures/T_cond'][:]
        P_evap = f['pressures/P_evap'][:]
        mass_flow = f['flows/mass_flow'][:]
        
        print(f"   T_evap shape: {T_evap.shape} (expected: [500, 100])")
        print(f"   T_cond shape: {T_cond.shape} (expected: [500, 100])")
        print(f"   P_evap shape: {P_evap.shape} (expected: [500, 100])")
        print(f"   mass_flow shape: {mass_flow.shape} (expected: [500, 100])")
        print(f"   T_evap range: [{T_evap.min():.2f}, {T_evap.max():.2f}] K")
        print(f"   T_cond range: [{T_cond.min():.2f}, {T_cond.max():.2f}] K")
        print(f"   P_evap range: [{P_evap.min():.2f}, {P_evap.max():.2f}] bar")
        print(f"   mass_flow range: [{mass_flow.min():.4f}, {mass_flow.max():.4f}] kg/s")
        print(f"   Has NaNs: T_evap={np.isnan(T_evap).any()}, mass_flow={np.isnan(mass_flow).any()}")
        
        results['temporal'] = {
            'shape_correct': T_evap.shape == (500, 100),
            'has_values': not np.all(T_evap==0),
            'no_nans': not np.isnan(T_evap).any()
        }
    
    # 3. Control system
    print(f"\n3. Control System (3_control_system.h5)")
    with h5py.File(data_path / "3_control_system.h5", 'r') as f:
        comp_speed = f['control_signals/compressor_speed'][:]
        valve = f['control_signals/expansion_valve_opening'][:]
        
        print(f"   compressor_speed shape: {comp_speed.shape} (expected: [500, 100])")
        print(f"   valve_opening shape: {valve.shape} (expected: [500, 100])")
        print(f"   compressor_speed range: [{comp_speed.min():.2f}, {comp_speed.max():.2f}]")
        print(f"   valve_opening range: [{valve.min():.4f}, {valve.max():.4f}]")
        
        results['control'] = {
            'shape_correct': comp_speed.shape == (500, 100),
            'has_values': not np.all(comp_speed==0),
            'no_nans': not np.isnan(comp_speed).any()
        }
    
    # 4. Sensor data
    print(f"\n4. Sensor Data (5_sensor_data.h5)")
    with h5py.File(data_path / "5_sensor_data.h5", 'r') as f:
        T_sensors = f['measurements/temperature_sensors'][:]
        
        print(f"   temperature_sensors shape: {T_sensors.shape} (expected: [500, 100, 10])")
        print(f"   temperature range: [{T_sensors.min():.2f}, {T_sensors.max():.2f}] K")
        print(f"   Has NaNs: {np.isnan(T_sensors).any()}")
        
        results['sensors'] = {
            'shape_correct': T_sensors.shape == (500, 100, 10),
            'has_values': not np.all(T_sensors==0),
            'no_nans': not np.isnan(T_sensors).any()
        }
    
    # 5. GNN topology
    print(f"\n5. GNN Topology (9_gnn_topology.h5)")
    with h5py.File(data_path / "9_gnn_topology.h5", 'r') as f:
        edge_features = f['edges/edge_features'][:]
        
        print(f"   edge_features shape: {edge_features.shape} (expected: [500, 11, 8])")
        print(f"   edge_features range: [{edge_features.min():.4f}, {edge_features.max():.4f}]")
        
        results['gnn'] = {
            'shape_correct': edge_features.shape == (500, 11, 8),
            'has_values': not np.all(edge_features==0),
            'no_nans': not np.isnan(edge_features).any()
        }
    
    # 6. Targets
    print(f"\n6. Targets (21_targets.h5)")
    with h5py.File(data_path / "21_targets.h5", 'r') as f:
        cop = f['performance/COP'][:]
        
        print(f"   COP shape: {cop.shape} (expected: [500])")
        print(f"   COP range: [{cop.min():.2f}, {cop.max():.2f}]")
        print(f"   COP mean: {cop.mean():.2f}")
        print(f"   Has NaNs: {np.isnan(cop).any()}")
        
        results['targets'] = {
            'shape_correct': cop.shape == (500,),
            'has_values': not np.all(cop==0),
            'no_nans': not np.isnan(cop).any(),
            'in_range': np.all((cop >= 1.5) & (cop <= 8.0))
        }
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    all_passed = True
    for name, checks in results.items():
        status = "PASS" if all(checks.values()) else "FAIL"
        print(f"\n{name.upper()}: {status}")
        for check, value in checks.items():
            symbol = "[OK]" if value else "[FAIL]"
            print(f"  {symbol} {check}: {value}")
        
        if not all(checks.values()):
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("[SUCCESS] All validation checks passed!")
        print("Dataset has 100 timesteps with real physics values.")
    else:
        print("[WARNING] Some validation checks failed!")
        print("Please review the issues above.")
    print(f"{'='*70}")
    
    return all_passed


if __name__ == "__main__":
    validate_dataset()