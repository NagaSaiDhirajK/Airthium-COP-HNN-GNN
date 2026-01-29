#!/usr/bin/env python3
"""
Merge multiple HDF5 dataset batches into a single large dataset.
"""

import os
import h5py
import numpy as np
import json
import glob
from datetime import datetime

def merge_batches(batch_pattern: str, output_dir: str):
    """Merge multiple batches into single dataset."""
    
    # Find all batch directories
    batch_dirs = sorted(glob.glob(batch_pattern))
    print(f"Found {len(batch_dirs)} batch directories:")
    for d in batch_dirs:
        print(f"  - {d}")
    
    if len(batch_dirs) == 0:
        print("ERROR: No batch directories found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First pass: count total samples and verify structure
    total_samples = 0
    batch_info = []
    
    for batch_dir in batch_dirs:
        info_path = os.path.join(batch_dir, 'dataset_info.json')
        h5_path = os.path.join(batch_dir, 'physics_data.h5')
        
        if not os.path.exists(info_path) or not os.path.exists(h5_path):
            print(f"  Skipping {batch_dir} (missing files)")
            continue
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        n_cycles = info['n_cycles']
        total_samples += n_cycles
        batch_info.append({
            'dir': batch_dir,
            'n_cycles': n_cycles,
            'cop_mean': info['cop_stats']['mean'],
            'cop_std': info['cop_stats']['std']
        })
        print(f"  {batch_dir}: {n_cycles} cycles (COP mean={info['cop_stats']['mean']:.3f})")
    
    print(f"\nTotal samples to merge: {total_samples}")
    
    # Second pass: create merged HDF5 file
    output_h5 = os.path.join(output_dir, 'physics_data.h5')
    print(f"\nCreating merged dataset: {output_h5}")
    
    # Get structure from first batch
    first_h5 = os.path.join(batch_info[0]['dir'], 'physics_data.h5')
    
    with h5py.File(first_h5, 'r') as f:
        # Get dataset structure
        datasets = {}
        def get_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype
                }
        f.visititems(get_datasets)
    
    print(f"  Found {len(datasets)} datasets to merge")
    
    # Create output file with correct sizes
    with h5py.File(output_h5, 'w') as out_f:
        # Create all datasets with full size
        for name, info in datasets.items():
            shape = list(info['shape'])
            shape[0] = total_samples  # Replace first dimension with total
            
            # Create dataset
            out_f.create_dataset(
                name,
                shape=tuple(shape),
                dtype=info['dtype'],
                chunks=True,
                compression='gzip',
                compression_opts=4
            )
            print(f"  Created {name}: {shape}")
        
        # Copy data from each batch
        current_idx = 0
        for batch in batch_info:
            batch_h5 = os.path.join(batch['dir'], 'physics_data.h5')
            n = batch['n_cycles']
            
            print(f"\n  Copying {batch['dir']} ({n} cycles, idx {current_idx}-{current_idx+n-1})...")
            
            with h5py.File(batch_h5, 'r') as in_f:
                for name in datasets.keys():
                    data = in_f[name][:]
                    out_f[name][current_idx:current_idx+n] = data
            
            current_idx += n
        
        print(f"\n  Final size: {current_idx} samples")
    
    # Calculate merged statistics
    print("\nCalculating merged statistics...")
    with h5py.File(output_h5, 'r') as f:
        cop = f['targets/COP_heating'][:]
        carnot_cop = f['validation/carnot_cop'][:]
        carnot_eff = cop / carnot_cop
    
    merged_info = {
        'n_cycles': total_samples,
        'n_timesteps': 100,
        'refrigerants': ['R410A', 'R134a', 'R290'],
        'physics_models_loaded': batch_info[0].get('physics_models_loaded', {}),
        'total_models_loaded': 17,
        'cop_stats': {
            'min': float(np.min(cop)),
            'max': float(np.max(cop)),
            'mean': float(np.mean(cop)),
            'std': float(np.std(cop)),
            'median': float(np.median(cop))
        },
        'carnot_efficiency_stats': {
            'min': float(np.min(carnot_eff)),
            'max': float(np.max(carnot_eff)),
            'mean': float(np.mean(carnot_eff))
        },
        'merged_from': len(batch_info),
        'batch_sources': [b['dir'] for b in batch_info],
        'generated_at': datetime.now().isoformat(),
        'architecture': '4-layer (inputs→physics→derived→targets)',
        'features': {
            'hnn_temporal': '100 timesteps × 14 channels',
            'gnn_nodes': '10 nodes × 35 features',
            'gnn_edges': '11 edges × 12 features',
            'gnn_graph': '50 features'
        }
    }
    
    # Save metadata
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(merged_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Total cycles: {total_samples}")
    print(f"  COP stats: min={merged_info['cop_stats']['min']:.2f}, max={merged_info['cop_stats']['max']:.2f}")
    print(f"  COP stats: mean={merged_info['cop_stats']['mean']:.3f}, std={merged_info['cop_stats']['std']:.3f}")
    print(f"  Carnot efficiency: mean={merged_info['carnot_efficiency_stats']['mean']:.3f}")
    print(f"{'='*60}")
    
    return merged_info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge HDF5 dataset batches')
    parser.add_argument('--pattern', type=str, default='paper_dataset_150k_batch*',
                        help='Glob pattern for batch directories')
    parser.add_argument('--output', type=str, default='paper_dataset_150k_merged',
                        help='Output directory for merged dataset')
    
    args = parser.parse_args()
    
    merge_batches(args.pattern, args.output)
