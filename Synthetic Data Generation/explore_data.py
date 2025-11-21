#!/usr/bin/env python3
"""
Data Exploration Script for Airthium Heat Pump Synthetic Dataset
Simple script to examine and analyze the generated thermodynamic data
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset(data_dir="demo_data"):
    """Explore the generated synthetic dataset"""
    
    data_path = Path(data_dir)
    
    print("🔍 AIRTHIUM HEAT PUMP DATASET EXPLORER")
    print("=" * 50)
    
    # Check available files
    if data_path.exists():
        files = list(data_path.glob("*"))
        print(f"📁 Available files in {data_dir}:")
        for file in files:
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name} ({size_kb:.1f} KB)")
        print()
    else:
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    # Load and explore CSV data
    csv_file = data_path / "synthetic_dataset.csv"
    if csv_file.exists():
        print("📊 CSV DATA ANALYSIS")
        print("-" * 30)
        
        df = pd.read_csv(csv_file)
        print(f"Dataset shape: {df.shape[0]} cycles × {df.shape[1]} features")
        print(f"Cycle types: {df['cycle_type'].value_counts().to_dict()}")
        print(f"Working fluids: {df['working_fluid'].value_counts().to_dict()}")
        
        print("\n📈 PERFORMANCE STATISTICS:")
        print(f"COP Heating - Mean: {df['COP_heating'].mean():.2f}, Range: [{df['COP_heating'].min():.2f}, {df['COP_heating'].max():.2f}]")
        print(f"COP Cooling - Mean: {df['COP_cooling'].mean():.2f}, Range: [{df['COP_cooling'].min():.2f}, {df['COP_cooling'].max():.2f}]")
        print(f"Heating Capacity - Mean: {df['capacity_heating'].mean():.1f} kW, Range: [{df['capacity_heating'].min():.1f}, {df['capacity_heating'].max():.1f}] kW")
        print(f"Energy Balance Error - Mean: {df['energy_balance_error'].mean():.6f}%, Max: {df['energy_balance_error'].max():.6f}%")
        print(f"Carnot Limit Compliance: {df['carnot_limit_check'].sum()}/{len(df)} ({df['carnot_limit_check'].mean()*100:.1f}%)")
        print()
    
    # Load and explore JSON data
    json_file = data_path / "synthetic_dataset.json"
    if json_file.exists():
        print("🔬 DETAILED CYCLE ANALYSIS")
        print("-" * 30)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"Dataset: {data['metadata']['dataset_info']['name']}")
        print(f"Generated: {data['metadata']['dataset_info']['generation_date']}")
        print(f"Total cycles: {len(data['thermodynamic_cycles'])}")
        
        if data['thermodynamic_cycles']:
            print("\n🌡️  SAMPLE CYCLE DETAILS:")
            cycle = data['thermodynamic_cycles'][0]
            print(f"Cycle ID: {cycle['cycle_id']}")
            print(f"Type: {cycle['cycle_type']}")
            print(f"Working Fluid: {cycle['working_fluid']}")
            print(f"COP Heating: {cycle['performance']['COP_heating']:.3f}")
            print(f"Heating Capacity: {cycle['performance']['capacity_heating']:.1f} kW")
            
            print("\n⚙️  THERMODYNAMIC STATES:")
            for state in cycle['states']:
                print(f"  State {state['point_id']} ({state['location']}):")
                print(f"    P = {state['P']/1000:.1f} kPa, T = {state['T']-273.15:.1f}°C")
                print(f"    h = {state['h']/1000:.1f} kJ/kg, s = {state['s']/1000:.3f} kJ/(kg·K)")
                print(f"    Phase: {state['phase']}")
            
            # Show operating conditions
            if 'operating_conditions' in cycle:
                print("\n🎛️  OPERATING CONDITIONS:")
                ops = cycle['operating_conditions']
                print(f"    Evaporator Temp: {ops['T_evap']-273.15:.1f}°C")
                print(f"    Condenser Temp: {ops['T_cond']-273.15:.1f}°C")
                print(f"    Mass Flow: {ops['mass_flow']:.2f} kg/s")
                print(f"    Part Load Ratio: {ops['part_load_ratio']:.2f}")
                print(f"    Compressor Efficiency: {ops['eta_comp_isentropic']:.3f}")
        
        # Validation report
        if 'validation_report' in data:
            print("\n✅ VALIDATION REPORT:")
            val = data['validation_report']
            print(f"    Total generated: {val['total_cycles_generated']}")
            print(f"    Valid cycles: {val['valid_cycles']}")
            print(f"    Validation rate: {val['validation_rate']:.1f}%")
            print(f"    Energy conservation: {val['physics_compliance']['energy_conservation']:.1f}%")
            print(f"    Carnot compliance: {val['physics_compliance']['carnot_limit_compliance']:.1f}%")

def run_new_generation(n_cycles=10):
    """Generate a fresh small dataset"""
    print(f"\n🚀 GENERATING NEW DATASET ({n_cycles} cycles)")
    print("=" * 50)
    
    try:
        import sys
        sys.path.append('src')
        from src import quick_generate_sample
        
        print("Generating cycles...")
        dataset = quick_generate_sample(n_cycles=n_cycles, output_dir="fresh_data")
        
        print(f"✅ Generated {len(dataset['thermodynamic_cycles'])} valid cycles")
        print("📁 Data saved to 'fresh_data' directory")
        
        # Quick analysis
        if dataset['thermodynamic_cycles']:
            cycles = dataset['thermodynamic_cycles']
            cops = [c['performance']['COP_heating'] for c in cycles]
            print(f"📊 COP Heating range: {min(cops):.2f} to {max(cops):.2f}")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        print("Make sure you're in the virtual environment:")
        print("  source venv/bin/activate")

def main():
    """Main exploration function"""
    print("Choose an option:")
    print("1. Explore existing demo data")
    print("2. Generate fresh small dataset (10 cycles)")
    print("3. Generate larger dataset (100 cycles)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            explore_dataset("demo_data")
        elif choice == "2":
            run_new_generation(10)
            explore_dataset("fresh_data")
        elif choice == "3":
            run_new_generation(100)
            explore_dataset("fresh_data")
        else:
            print("Invalid choice. Exploring demo data...")
            explore_dataset("demo_data")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to demo data exploration...")
        explore_dataset("demo_data")

if __name__ == "__main__":
    main()