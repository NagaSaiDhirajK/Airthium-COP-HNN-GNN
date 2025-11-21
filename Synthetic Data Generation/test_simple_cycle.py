#!/usr/bin/env python3
"""
Simple test to debug thermodynamic cycle calculations
"""

import sys
sys.path.append('src')

from cycle_generators import CycleGenerator, CycleType, CycleParameters
from thermodynamic_properties import get_fluid_properties
import numpy as np

def test_simple_carnot_cycle():
    """Test a simple Carnot cycle with known parameters"""
    
    print("🔬 Testing Simple Carnot Cycle")
    print("=" * 50)
    
    # Use simple, realistic parameters
    params = CycleParameters(
        T_evap=273.15,      # 0°C
        T_cond=313.15,      # 40°C  
        mass_flow=1.0,      # 1 kg/s
        working_fluid='R134a',
        superheat=5.0,
        subcool=5.0
    )
    
    print(f"Parameters:")
    print(f"  T_evap: {params.T_evap-273.15:.1f}°C")
    print(f"  T_cond: {params.T_cond-273.15:.1f}°C")
    print(f"  Mass flow: {params.mass_flow} kg/s")
    print(f"  Working fluid: {params.working_fluid}")
    
    # Calculate theoretical Carnot COP
    carnot_cop_heating = params.T_cond / (params.T_cond - params.T_evap)
    carnot_cop_cooling = params.T_evap / (params.T_cond - params.T_evap)
    
    print(f"\nTheoretical Carnot Limits:")
    print(f"  COP_heating: {carnot_cop_heating:.2f}")
    print(f"  COP_cooling: {carnot_cop_cooling:.2f}")
    
    # Test fluid properties first
    print(f"\n🧪 Testing Fluid Properties:")
    try:
        fluid = get_fluid_properties('R134a')
        
        # Test saturation pressures
        P_evap = fluid.get_saturation_pressure(params.T_evap)
        P_cond = fluid.get_saturation_pressure(params.T_cond)
        
        print(f"  P_evap: {P_evap/1000:.1f} kPa")
        print(f"  P_cond: {P_cond/1000:.1f} kPa")
        print(f"  Pressure ratio: {P_cond/P_evap:.2f}")
        
        # Test saturated states
        state_evap_liq = fluid.get_saturated_liquid(params.T_evap)
        state_evap_vap = fluid.get_saturated_vapor(params.T_evap)
        
        if state_evap_liq and state_evap_vap:
            h_fg = state_evap_vap.h - state_evap_liq.h
            print(f"  Latent heat at evaporator: {h_fg/1000:.1f} kJ/kg")
        else:
            print("  ❌ Could not get saturated states at evaporator")
            return
            
    except Exception as e:
        print(f"  ❌ Fluid property error: {e}")
        return
    
    # Generate cycle
    print(f"\n⚙️  Generating Carnot Cycle:")
    generator = CycleGenerator()
    
    try:
        results = generator.generate_cycle(CycleType.CARNOT, 'R134a', params)
        
        print(f"  Generated cycle: {results.cycle_type}")
        print(f"  States calculated: {len(results.states)}")
        
        # Check states
        for i, (state, label) in enumerate(zip(results.states, results.state_labels)):
            if state:
                print(f"  State {i+1} ({label}): P={state.P/1000:.1f} kPa, T={state.T-273.15:.1f}°C, h={state.h/1000:.1f} kJ/kg")
            else:
                print(f"  State {i+1} ({label}): ❌ NULL")
        
        # Check performance
        print(f"\n📊 Performance Results:")
        print(f"  Q_evap: {results.Q_evap:.2f} kW")
        print(f"  Q_cond: {results.Q_cond:.2f} kW") 
        print(f"  W_comp: {results.W_comp:.2f} kW")
        print(f"  W_exp: {results.W_exp:.2f} kW")
        print(f"  W_net: {results.W_net:.2f} kW")
        
        print(f"\n🎯 COPs:")
        print(f"  COP_heating: {results.COP_heating:.2f} (Carnot: {carnot_cop_heating:.2f})")
        print(f"  COP_cooling: {results.COP_cooling:.2f} (Carnot: {carnot_cop_cooling:.2f})")
        
        # Energy balance check
        energy_in = results.Q_evap + results.W_net
        energy_out = results.Q_cond
        energy_error = abs(energy_in - energy_out) / energy_out * 100 if energy_out > 0 else 100
        
        print(f"\n⚖️  Energy Balance:")
        print(f"  Energy in (Q_evap + W_net): {energy_in:.2f} kW")
        print(f"  Energy out (Q_cond): {energy_out:.2f} kW")
        print(f"  Balance error: {energy_error:.2f}%")
        print(f"  Validation error: {results.energy_balance_error:.2f}%")
        
        # Check physics violations
        print(f"\n🔍 Physics Checks:")
        print(f"  Carnot limit check: {'✅ PASS' if results.carnot_limit_check else '❌ FAIL'}")
        print(f"  Entropy generation: {results.entropy_generation:.6f} kW/K")
        
        if energy_error < 5.0 and results.carnot_limit_check:
            print(f"\n✅ CYCLE GENERATION SUCCESS!")
        else:
            print(f"\n❌ CYCLE GENERATION FAILED!")
            
    except Exception as e:
        print(f"  ❌ Cycle generation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_carnot_cycle()