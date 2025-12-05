#!/usr/bin/env python3
"""
Generate Mini Dataset for Kaggle Testing
========================================

Create a small 100-cycle dataset optimized for Kaggle GPU testing.
This will validate the model architecture and data pipeline before
generating the full 50,000-cycle dataset.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, 'src')

from integrated_validation_pipeline import IntegratedPhysicsValidator

def generate_mini_dataset(n_cycles=100):
    """Generate a mini dataset with specified number of cycles."""
    
    print(f"🚀 GENERATING MINI DATASET ({n_cycles} cycles)")
    print("=" * 60)
    
    # Initialize validator
    print("📊 Initializing physics validator...")
    validator = IntegratedPhysicsValidator()
    
    # Generate data with reduced cycles
    print(f"🔄 Generating {n_cycles} physics cycles...")
    start_time = time.time()
    
    try:
        validation_results = validator.validate_all_physics_models(n_test_cycles=n_cycles)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated data in {generation_time:.1f} seconds")
        print(f"📊 Models tested: {validation_results.get('models_tested', 0)}")
        
        # Create output directory
        output_dir = Path("mini_dataset")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw validation results
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Extract and save model-specific data
        model_datasets = extract_model_data(validation_results)
        
        # Save individual model datasets as compressed files
        dataset_info = {}
        for model_name, data in model_datasets.items():
            if data:  # Only save if data exists
                df = pd.DataFrame(data)
                
                # Save as compressed CSV
                filename = f"{model_name}_mini.csv.gz"
                filepath = output_dir / filename
                df.to_csv(filepath, index=False, compression='gzip')
                
                dataset_info[model_name] = {
                    'rows': len(data),
                    'columns': len(df.columns),
                    'file': filename,
                    'size_mb': filepath.stat().st_size / (1024*1024)
                }
                
                print(f"  ✅ {model_name}: {len(data)} rows × {len(df.columns)} cols ({dataset_info[model_name]['size_mb']:.1f}MB)")
        
        # Create summary
        summary = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'total_cycles': n_cycles,
            'generation_time_seconds': generation_time,
            'models_processed': len(model_datasets),
            'dataset_info': dataset_info,
            'total_size_mb': sum(info['size_mb'] for info in dataset_info.values()),
            'kaggle_ready': True
        }
        
        # Save summary
        with open(output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 MINI DATASET SUMMARY:")
        print(f"  📁 Output directory: {output_dir}")
        print(f"  📊 Total models: {len(model_datasets)}")
        print(f"  💾 Total size: {summary['total_size_mb']:.1f}MB")
        print(f"  ⏱️  Generation time: {generation_time:.1f}s")
        print(f"  🎯 Kaggle ready: ✅")
        
        return output_dir, summary
        
    except Exception as e:
        print(f"❌ Error generating dataset: {str(e)}")
        return None, None

def extract_model_data(validation_results):
    """Extract data organized by model type."""
    model_datasets = {}
    
    # Extract from the validation results structure
    model_results = validation_results.get('model_results', {})
    
    for model_name, model_result in model_results.items():
        if model_name not in model_datasets:
            model_datasets[model_name] = []
        
        # Extract generated data from each model
        generated_data = model_result.get('generated_data', [])
        
        if isinstance(generated_data, list):
            # Process each cycle's data
            for i, cycle_data in enumerate(generated_data):
                # Extract base info
                row_data = {
                    'cycle_id': i,
                    'model_name': model_name,
                    'model_type': cycle_data.get('model_type', model_name),
                    'validation_score': model_result.get('model_score', 0.0),
                    'model_status': model_result.get('model_status', 'UNKNOWN'),
                    'timestamp': validation_results.get('validation_timestamp', '')
                }
                
                # Flatten and extract cycle data (simplified)
                if cycle_data:
                    flattened = flatten_data_simple(cycle_data)
                    row_data.update(flattened)
                
                model_datasets[model_name].append(row_data)
    
    return model_datasets

def flatten_data_simple(data, prefix='', max_depth=2, current_depth=0):
    """Simplified flattening for faster processing."""
    if current_depth >= max_depth:
        return {prefix.rstrip('_'): str(data)[:100]}  # Truncate long strings
    
    flattened = {}
    
    if isinstance(data, dict):
        for key, value in list(data.items())[:20]:  # Limit to first 20 keys
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, (dict, list)) and current_depth < max_depth - 1:
                flattened.update(flatten_data_simple(value, f"{new_key}_", max_depth, current_depth + 1))
            else:
                # Convert to appropriate type
                if isinstance(value, (int, float, bool)):
                    flattened[new_key] = value
                elif isinstance(value, str):
                    flattened[new_key] = value[:50]  # Truncate strings
                else:
                    flattened[new_key] = str(value)[:50]
    
    elif isinstance(data, list):
        for i, item in enumerate(data[:10]):  # Limit to first 10 items
            new_key = f"{prefix}{i}" if prefix else f"item_{i}"
            if isinstance(item, (dict, list)) and current_depth < max_depth - 1:
                flattened.update(flatten_data_simple(item, f"{new_key}_", max_depth, current_depth + 1))
            else:
                if isinstance(item, (int, float, bool)):
                    flattened[new_key] = item
                else:
                    flattened[new_key] = str(item)[:50]
    
    return flattened

def create_kaggle_package():
    """Create a Kaggle-ready package."""
    
    print("\n📦 CREATING KAGGLE PACKAGE")
    print("=" * 40)
    
    # Generate mini dataset
    output_dir, summary = generate_mini_dataset(100)
    
    if output_dir is None:
        print("❌ Failed to generate dataset")
        return
    
    # Create Kaggle package structure
    kaggle_dir = Path("kaggle_package")
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy data files
    data_dir = kaggle_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    for file in output_dir.glob("*.csv.gz"):
        import shutil
        shutil.copy2(file, data_dir / file.name)
    
    # Copy summary
    shutil.copy2(output_dir / "dataset_summary.json", data_dir / "dataset_summary.json")
    
    # Create README for Kaggle
    readme_content = f"""# AIRTHIUM Heat Pump Mini Dataset

## Overview
This is a mini dataset (100 cycles) for testing HNN-GNN COP prediction models.

## Files
- `*_mini.csv.gz`: Compressed model-specific datasets
- `dataset_summary.json`: Dataset metadata and statistics

## Usage
```python
import pandas as pd
import gzip

# Load a specific model dataset
df = pd.read_csv('hamiltonian_mini.csv.gz', compression='gzip')
print(f"Dataset shape: {{df.shape}}")
```

## Dataset Info
- Total cycles: {summary['total_cycles'] if summary else 100}
- Total size: {summary['total_size_mb']:.1f}MB if summary else 'N/A'
- Models: {summary['models_processed'] if summary else 'N/A'}
- Generated: {summary['generation_timestamp'] if summary else 'N/A'}

## Next Steps
1. Upload to Kaggle as a dataset
2. Create notebook for model training
3. Test HNN-GNN architecture on T4 x2 GPUs
"""
    
    with open(kaggle_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Calculate total package size
    total_size = sum(f.stat().st_size for f in kaggle_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024*1024)
    
    print(f"✅ Kaggle package created: {kaggle_dir}")
    print(f"📊 Package size: {total_size_mb:.1f}MB")
    print(f"📁 Files: {len(list(kaggle_dir.rglob('*')))}")
    print(f"🎯 Ready for Kaggle upload!")
    
    return kaggle_dir

if __name__ == "__main__":
    # Create mini dataset and Kaggle package
    kaggle_package = create_kaggle_package()
    
    print("\n🎉 MINI DATASET GENERATION COMPLETE!")
    print("\nNext steps:")
    print("1. Upload kaggle_package/ to Kaggle as a dataset")
    print("2. Create Kaggle notebook for model training")
    print("3. Test on T4 x2 GPUs")