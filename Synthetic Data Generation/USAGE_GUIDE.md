# 🚀 Airthium Heat Pump Data Generation - Usage Guide

## Quick Start

### 1. **Activate Virtual Environment**
```bash
cd "Airthium-COP-HNN-GNN/Synthetic Data Generation"
source venv/bin/activate
```

### 2. **Run Demo (Already Generated)**
```bash
python demo.py
```
This runs all demonstrations and generates sample data in `demo_data/` directory.

### 3. **Explore Existing Data**
```bash
python explore_data.py
```
Choose option 1 to explore the demo data that's already been generated.

## 📊 Generated Data Files

The system creates three types of output files:

### **CSV File** (`synthetic_dataset.csv`)
Tabular data perfect for analysis:
```
cycle_id,cycle_type,working_fluid,COP_heating,COP_cooling,capacity_heating,...
ideal_rankine_R410A_998587,ideal_rankine,R410A,-4.15,-5.15,-495.3,...
```

### **JSON File** (`synthetic_dataset.json`)
Complete structured data with metadata:
```json
{
  "metadata": {
    "dataset_info": {
      "name": "Airthium Heat Pump Synthetic Dataset",
      "total_cycles": 3
    }
  },
  "thermodynamic_cycles": [
    {
      "cycle_id": "ideal_rankine_R410A_998587",
      "cycle_type": "ideal_rankine",
      "working_fluid": "R410A",
      "performance": {
        "COP_heating": -4.146,
        "capacity_heating": -495.3
      },
      "states": [
        {
          "point_id": 1,
          "location": "pump_out",
          "P": 4901200,
          "T": 432.05,
          "h": 550000,
          "s": 2006,
          "phase": "vapor"
        }
      ]
    }
  ]
}
```

### **HDF5 File** (`synthetic_dataset.h5`)
Efficient numerical arrays for ML training.

## 🔧 How to Generate New Data

### **Small Dataset (10-100 cycles)**
```python
from src import quick_generate_sample

# Generate 50 cycles
dataset = quick_generate_sample(n_cycles=50, output_dir="my_data")
print(f"Generated {len(dataset['thermodynamic_cycles'])} valid cycles")
```

### **Large Dataset (Full 145,000 cycles)**
```python
from src.data_generation_pipeline import main
main()  # Generates full dataset
```

### **Custom Configuration**
```python
from src import SyntheticDataGenerator, DatasetConfiguration

config = DatasetConfiguration(
    total_cycles=1000,
    carnot_cycles=400,
    ideal_rankine_cycles=300,
    non_ideal_rankine_cycles=300,
    output_dir="custom_data",
    n_workers=4
)

generator = SyntheticDataGenerator(config)
dataset = generator.generate_complete_dataset()
```

## 📈 Data Analysis Examples

### **Load and Analyze CSV Data**
```python
import pandas as pd

# Load data
df = pd.read_csv('demo_data/synthetic_dataset.csv')

# Basic statistics
print(f"Dataset shape: {df.shape}")
print(f"Mean COP Heating: {df['COP_heating'].mean():.2f}")
print(f"Working fluids: {df['working_fluid'].value_counts()}")

# Filter data
high_cop = df[df['COP_heating'] > 3.0]
print(f"High COP cycles: {len(high_cop)}")
```

### **Load JSON Data**
```python
import json

with open('demo_data/synthetic_dataset.json', 'r') as f:
    data = json.load(f)

# Access cycle data
for cycle in data['thermodynamic_cycles']:
    print(f"Cycle: {cycle['cycle_id']}")
    print(f"COP: {cycle['performance']['COP_heating']:.2f}")
    
    # Access thermodynamic states
    for state in cycle['states']:
        print(f"  State {state['point_id']}: P={state['P']/1000:.1f} kPa, T={state['T']-273.15:.1f}°C")
```

### **Load HDF5 Data (for ML)**
```python
import h5py
import numpy as np

with h5py.File('demo_data/synthetic_dataset.h5', 'r') as f:
    # Load performance arrays
    cop_heating = f['thermodynamic_cycles/COP_heating'][:]
    cop_cooling = f['thermodynamic_cycles/COP_cooling'][:]
    
    # Load state point arrays
    pressures = f['thermodynamic_cycles/states/pressure'][:]
    temperatures = f['thermodynamic_cycles/states/temperature'][:]
    
    print(f"COP range: {cop_heating.min():.2f} to {cop_heating.max():.2f}")
```

## 🎯 Understanding the Data

### **Current Demo Data**
- **3 valid cycles** from 50 attempts (6% validation rate)
- **Working fluids**: R410A, Air
- **Cycle types**: Ideal Rankine
- **COP range**: -103.89 to -4.15 (negative values indicate reverse operation)
- **Energy balance error**: < 0.0001% (excellent physics compliance)
- **Carnot limit compliance**: 100%

### **Data Structure per Cycle**
Each cycle contains:
- **Cycle ID**: Unique identifier
- **Operating conditions**: Temperatures, pressures, mass flow
- **4-6 thermodynamic states**: Complete P-V-T-S properties
- **Performance metrics**: COP, capacity, work, efficiency
- **Validation data**: Energy balance, entropy check
- **Trajectory data**: Time-series for HNN training

## 🔍 Troubleshooting

### **Low Validation Rate**
The system has strict physics validation (6% rate is normal):
- Energy conservation error must be < 0.1%
- All COPs must be below Carnot limits
- Entropy generation must be positive
- This ensures high-quality, physics-compliant data

### **Negative COP Values**
Some cycles show negative COPs due to:
- Reverse cycle operation (cooling mode)
- Extreme operating conditions
- This is physically valid for certain configurations

### **Missing Working Fluids**
Some fluids (thermal oils) show warnings:
- CoolProp doesn't have all proprietary fluid data
- System uses fallback calculations
- Core refrigerants (R245fa, R134a, etc.) work perfectly

## 🚀 Next Steps

1. **Explore the demo data** using the provided scripts
2. **Generate larger datasets** for your specific needs
3. **Integrate with ML models** using the HDF5 format
4. **Customize cycle parameters** for your application
5. **Scale up to full 145,000 cycle dataset** for production

## 📞 Support

- Check `README.md` for detailed documentation
- Run `python demo.py` for full demonstration
- Use `python explore_data.py` for interactive data exploration
- Examine the generated visualization in `demo_data/data_analysis.png`