# AIRTHIUM Heat Pump Mini Dataset

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
print(f"Dataset shape: {df.shape}")
```

## Dataset Info
- Total cycles: 100
- Total size: 0.1MB if summary else 'N/A'
- Models: 26
- Generated: 2025-12-05T04:59:17.038942

## Next Steps
1. Upload to Kaggle as a dataset
2. Create notebook for model training
3. Test HNN-GNN architecture on T4 x2 GPUs
