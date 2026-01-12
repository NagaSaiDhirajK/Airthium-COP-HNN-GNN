#!/usr/bin/env python3
"""
Test script for Comprehensive Physics Generator
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

print("Testing Comprehensive Physics Generator...")
print("=" * 60)

try:
    from comprehensive_physics_generator import ComprehensivePhysicsGenerator
    print("[OK] Import successful")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

try:
    # Initialize generator
    print("\nInitializing generator...")
    generator = ComprehensivePhysicsGenerator(n_timesteps=100)
    print("[OK] Generator initialized")
except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Generate test samples
print("\nGenerating 5 test samples...")
valid_samples = []
attempts = 0
max_attempts = 20

while len(valid_samples) < 5 and attempts < max_attempts:
    attempts += 1
    try:
        sample = generator.generate_sample()
        if sample is not None:
            valid_samples.append(sample)
            print(f"  Sample {len(valid_samples)}: COP={sample.COP:.3f}, "
                  f"Features={len(sample.features)}, Valid={sample.is_valid}")
    except Exception as e:
        print(f"  Attempt {attempts} failed: {str(e)[:50]}...")

if valid_samples:
    print(f"\n[OK] Success rate: {len(valid_samples)}/{attempts} attempts ({100*len(valid_samples)/attempts:.0f}%)")
    
    sample = valid_samples[0]
    print(f"\nSample structure:")
    print(f"  - Static features: {len(sample.features)}")
    print(f"  - HNN trajectories: {list(sample.hnn_trajectories.keys())}")
    for key, val in sample.hnn_trajectories.items():
        print(f"      {key}: shape={val.shape}")
    print(f"  - GNN node features: {sample.gnn_node_features.shape}")
    print(f"  - GNN edge features: {sample.gnn_edge_features.shape}")
    print(f"  - GNN edge index: {sample.gnn_edge_index.shape}")
    
    print(f"\nCOP values generated:")
    cop_values = [s.COP for s in valid_samples]
    print(f"  COPs: {[f'{c:.3f}' for c in cop_values]}")
    print(f"  Min: {min(cop_values):.3f}, Max: {max(cop_values):.3f}")
    
    print(f"\nSample feature keys (first 20):")
    keys = list(sample.features.keys())[:20]
    for k in keys:
        print(f"  - {k}: {sample.features[k]:.4f}" if isinstance(sample.features[k], float) else f"  - {k}: {sample.features[k]}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Comprehensive Physics Generator is working!")
    print("=" * 60)
else:
    print(f"\n[ERROR] No valid samples generated after {attempts} attempts")
    sys.exit(1)
