"""
Demonstration Script for Airthium Heat Pump Synthetic Data Generation
Shows how to use the system and generates a small test dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

# Import our modules
from src import (
    quick_generate_sample, list_available_fluids, list_available_cycles,
    get_fluid_properties, CycleGenerator, CycleType, DatasetConfiguration,
    SyntheticDataGenerator
)

def demo_fluid_properties():
    """Demonstrate fluid property calculations"""
    print("\n" + "="*60)
    print("DEMONSTRATION: FLUID PROPERTIES")
    print("="*60)
    
    # Show available fluids
    list_available_fluids()
    
    # Test R245fa properties
    print("Testing R245fa properties at 5 bar, 300 K:")
    try:
        r245fa = get_fluid_properties('R245fa')
        state = r245fa.get_properties(500000, 300)  # 5 bar, 300 K
        
        print(f"  Pressure: {state.P/1000:.1f} kPa")
        print(f"  Temperature: {state.T-273.15:.1f} °C")
        print(f"  Enthalpy: {state.h/1000:.1f} kJ/kg")
        print(f"  Entropy: {state.s/1000:.3f} kJ/(kg·K)")
        print(f"  Density: {state.rho:.2f} kg/m³")
        print(f"  Phase: {state.phase}")
        if state.x is not None:
            print(f"  Quality: {state.x:.3f}")
        
        # Test saturation properties
        T_sat = r245fa.get_saturation_temperature(500000)
        P_sat = r245fa.get_saturation_pressure(300)
        print(f"  Saturation temp at 5 bar: {T_sat-273.15:.1f} °C")
        print(f"  Saturation pressure at 300 K: {P_sat/1000:.1f} kPa")
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Note: CoolProp may not be installed. Using fallback calculations.")

def demo_single_cycle():
    """Demonstrate single cycle generation"""
    print("\n" + "="*60)
    print("DEMONSTRATION: SINGLE CYCLE GENERATION")
    print("="*60)
    
    # Show available cycles
    list_available_cycles()
    
    try:
        generator = CycleGenerator()
        
        # Generate a Carnot cycle with R245fa
        print("\nGenerating Carnot cycle with R245fa...")
        params = generator.generate_random_parameters(CycleType.CARNOT, 'R245fa')
        results = generator.generate_cycle(CycleType.CARNOT, 'R245fa', params)
        
        print(f"Cycle Type: {results.cycle_type}")
        print(f"Working Fluid: {results.working_fluid}")
        print(f"Evaporator Temperature: {params.T_evap-273.15:.1f} °C")
        print(f"Condenser Temperature: {params.T_cond-273.15:.1f} °C")
        print(f"Mass Flow Rate: {params.mass_flow:.2f} kg/s")
        print(f"COP Heating: {results.COP_heating:.2f}")
        print(f"COP Cooling: {results.COP_cooling:.2f}")
        print(f"Heating Capacity: {results.capacity_heating:.1f} kW")
        print(f"Cooling Capacity: {results.capacity_cooling:.1f} kW")
        print(f"Net Work: {results.W_net:.1f} kW")
        print(f"Energy Balance Error: {results.energy_balance_error:.3f}%")
        print(f"Carnot Limit Check: {'PASS' if results.carnot_limit_check else 'FAIL'}")
        
        # Show state points
        print(f"\nThermodynamic States ({len(results.states)} points):")
        for i, (state, label) in enumerate(zip(results.states, results.state_labels)):
            print(f"  State {i+1} ({label}):")
            print(f"    P = {state.P/1000:.1f} kPa, T = {state.T-273.15:.1f} °C")
            print(f"    h = {state.h/1000:.1f} kJ/kg, s = {state.s/1000:.3f} kJ/(kg·K)")
            print(f"    Phase: {state.phase}")
        
    except Exception as e:
        print(f"Error generating cycle: {e}")

def demo_small_dataset():
    """Demonstrate small dataset generation"""
    print("\n" + "="*60)
    print("DEMONSTRATION: SMALL DATASET GENERATION")
    print("="*60)
    
    print("Generating 50 cycles for demonstration...")
    start_time = time.time()
    
    try:
        # Generate small dataset
        dataset = quick_generate_sample(n_cycles=50, output_dir="demo_data")
        
        generation_time = time.time() - start_time
        print(f"Dataset generated in {generation_time:.2f} seconds")
        
        # Show validation report
        validation = dataset['validation_report']
        print(f"\nValidation Report:")
        print(f"  Total cycles generated: {validation['total_cycles_generated']}")
        print(f"  Valid cycles: {validation['valid_cycles']}")
        print(f"  Validation rate: {validation['validation_rate']:.1f}%")
        print(f"  Energy conservation: {validation['physics_compliance']['energy_conservation']:.1f}%")
        print(f"  Carnot limit compliance: {validation['physics_compliance']['carnot_limit_compliance']:.1f}%")
        
        # Analyze generated data
        cycles = dataset['thermodynamic_cycles']
        if cycles:
            print(f"\nDataset Analysis:")
            print(f"  Number of valid cycles: {len(cycles)}")
            
            # Extract performance data
            cop_heating = [c['performance']['COP_heating'] for c in cycles]
            cop_cooling = [c['performance']['COP_cooling'] for c in cycles]
            capacities = [c['performance']['capacity_heating'] for c in cycles]
            
            print(f"  COP Heating range: {min(cop_heating):.2f} - {max(cop_heating):.2f}")
            print(f"  COP Cooling range: {min(cop_cooling):.2f} - {max(cop_cooling):.2f}")
            print(f"  Heating capacity range: {min(capacities):.1f} - {max(capacities):.1f} kW")
            
            # Count cycle types
            cycle_types = {}
            working_fluids = {}
            for cycle in cycles:
                cycle_type = cycle['cycle_type']
                fluid = cycle['working_fluid']
                cycle_types[cycle_type] = cycle_types.get(cycle_type, 0) + 1
                working_fluids[fluid] = working_fluids.get(fluid, 0) + 1
            
            print(f"  Cycle types: {dict(cycle_types)}")
            print(f"  Working fluids: {dict(working_fluids)}")
            
            # Check if files were saved
            demo_path = Path("demo_data")
            if demo_path.exists():
                files = list(demo_path.glob("*"))
                print(f"  Output files: {[f.name for f in files]}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()

def demo_visualization():
    """Create simple visualizations of generated data"""
    print("\n" + "="*60)
    print("DEMONSTRATION: DATA VISUALIZATION")
    print("="*60)
    
    try:
        # Check if demo data exists
        csv_file = Path("demo_data/synthetic_dataset.csv")
        if not csv_file.exists():
            print("No demo data found. Generating small dataset first...")
            demo_small_dataset()
        
        if csv_file.exists():
            # Load and visualize data
            df = pd.read_csv(csv_file)
            print(f"Loaded dataset with {len(df)} cycles")
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Airthium Heat Pump Synthetic Data Analysis', fontsize=16)
            
            # COP distribution
            axes[0, 0].hist(df['COP_heating'], bins=20, alpha=0.7, color='red', label='Heating')
            axes[0, 0].hist(df['COP_cooling'], bins=20, alpha=0.7, color='blue', label='Cooling')
            axes[0, 0].set_xlabel('COP')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('COP Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Capacity vs COP
            axes[0, 1].scatter(df['capacity_heating'], df['COP_heating'], alpha=0.6, color='red')
            axes[0, 1].set_xlabel('Heating Capacity (kW)')
            axes[0, 1].set_ylabel('COP Heating')
            axes[0, 1].set_title('Capacity vs COP')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Cycle type distribution
            cycle_counts = df['cycle_type'].value_counts()
            axes[1, 0].pie(cycle_counts.values, labels=cycle_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Cycle Type Distribution')
            
            # Working fluid distribution
            fluid_counts = df['working_fluid'].value_counts()
            axes[1, 1].bar(range(len(fluid_counts)), fluid_counts.values)
            axes[1, 1].set_xticks(range(len(fluid_counts)))
            axes[1, 1].set_xticklabels(fluid_counts.index, rotation=45)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Working Fluid Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path("demo_data/data_analysis.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {plot_path}")
            
            # Show basic statistics
            print(f"\nDataset Statistics:")
            print(f"  Mean COP Heating: {df['COP_heating'].mean():.2f}")
            print(f"  Mean COP Cooling: {df['COP_cooling'].mean():.2f}")
            print(f"  Mean Heating Capacity: {df['capacity_heating'].mean():.1f} kW")
            print(f"  Energy Balance Error (mean): {df['energy_balance_error'].mean():.4f}%")
            print(f"  Carnot Limit Compliance: {df['carnot_limit_check'].sum()}/{len(df)} ({df['carnot_limit_check'].mean()*100:.1f}%)")
            
        else:
            print("Could not find or generate demo data for visualization")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Note: matplotlib and pandas may not be installed")

def main():
    """Run all demonstrations"""
    print("AIRTHIUM HEAT PUMP SYNTHETIC DATA GENERATION")
    print("Physics-Informed Machine Learning Dataset")
    print("="*60)
    
    # Run demonstrations
    demo_fluid_properties()
    demo_single_cycle()
    demo_small_dataset()
    demo_visualization()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nTo generate the full 145,000 cycle dataset, run:")
    print("python -c \"from src.data_generation_pipeline import main; main()\"")
    print("\nOr use the SyntheticDataGenerator class directly with custom configuration.")
    print("\nGenerated files are saved in the 'demo_data' directory.")

if __name__ == "__main__":
    main()