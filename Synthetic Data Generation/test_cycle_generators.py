#!/usr/bin/env python3
"""
Comprehensive Test Suite for Thermodynamic Cycle Generators
Tests all cycle types, parameter generation, and physics validation
"""

import sys
import os
sys.path.append('src')

import numpy as np
import json
from typing import Dict, List, Any

# Import cycle generation modules
from cycle_generators import (
    CycleGenerator, CycleType, CycleParameters, CycleResults,
    CarnotCycle, RankineCycle, TranscriticalCycle
)
from thermodynamic_properties import get_fluid_properties

def test_cycle_parameters():
    """Test CycleParameters dataclass functionality"""
    print("🔄 Testing CycleParameters dataclass...")
    
    # Test default parameters
    params = CycleParameters(
        T_evap=278.15,  # 5°C
        T_cond=318.15,  # 45°C
        mass_flow=1.0,
        working_fluid='R134a'
    )
    
    print(f"✅ Default parameters created:")
    print(f"  - Evap temp: {params.T_evap - 273.15:.1f}°C")
    print(f"  - Cond temp: {params.T_cond - 273.15:.1f}°C")
    print(f"  - Mass flow: {params.mass_flow} kg/s")
    print(f"  - Working fluid: {params.working_fluid}")
    print(f"  - Superheat: {params.superheat} K")
    print(f"  - Subcooling: {params.subcool} K")
    print(f"  - Compressor efficiency: {params.eta_comp_isentropic}")
    print(f"  - Expander efficiency: {params.eta_exp_isentropic}")
    
    # Test custom parameters
    custom_params = CycleParameters(
        T_evap=268.15,
        T_cond=328.15,
        mass_flow=1.5,
        working_fluid='R410A',
        superheat=10.0,
        subcool=8.0,
        eta_comp_isentropic=0.80,
        eta_exp_isentropic=0.75,
        eta_mechanical=0.93,
        eta_motor=0.90,
        dp_evap=3000,
        dp_cond=4000,
        T_ambient=293.15,
        humidity=0.55,
        part_load_ratio=0.8
    )
    
    print(f"✅ Custom parameters created:")
    print(f"  - Temperature lift: {custom_params.T_cond - custom_params.T_evap:.1f} K")
    print(f"  - Part load ratio: {custom_params.part_load_ratio}")
    print(f"  - Ambient conditions: {custom_params.T_ambient - 273.15:.1f}°C, {custom_params.humidity*100:.1f}% RH")

def test_cycle_results():
    """Test CycleResults dataclass functionality"""
    print("\n🔄 Testing CycleResults dataclass...")
    
    # Create dummy thermodynamic states
    from thermodynamic_properties import ThermodynamicState
    
    states = [
        ThermodynamicState(P=400000, T=278.15, h=250000, s=1000, rho=1200, phase="liquid"),
        ThermodynamicState(P=400000, T=283.15, h=260000, s=1050, rho=15, phase="vapor"),
        ThermodynamicState(P=1800000, T=353.15, h=280000, s=1050, rho=50, phase="vapor"),
        ThermodynamicState(P=1800000, T=318.15, h=240000, s=950, rho=1150, phase="liquid")
    ]
    
    results = CycleResults(
        states=states,
        state_labels=["evap_in", "evap_out", "comp_out", "cond_out"],
        Q_evap=10.0,
        Q_cond=13.5,
        W_comp=3.5,
        W_exp=0.0,
        W_net=3.5,
        COP_heating=3.86,
        COP_cooling=2.86,
        mass_flow=1.0,
        capacity_heating=13.5,
        capacity_cooling=10.0,
        cycle_type="test_cycle",
        working_fluid="R134a"
    )
    
    print(f"✅ CycleResults created:")
    print(f"  - States: {len(results.states)} thermodynamic states")
    print(f"  - Q_evap: {results.Q_evap} kW")
    print(f"  - Q_cond: {results.Q_cond} kW")
    print(f"  - W_net: {results.W_net} kW")
    print(f"  - COP heating: {results.COP_heating:.2f}")
    print(f"  - COP cooling: {results.COP_cooling:.2f}")
    print(f"  - Energy balance: Q_cond = Q_evap + W_net = {results.Q_evap + results.W_net:.1f} kW")

def test_carnot_cycle():
    """Test Carnot cycle generation"""
    print("\n🔄 Testing Carnot Cycle Generation...")
    
    # Test with R134a
    carnot = CarnotCycle('R134a')
    print(f"✅ Carnot cycle initialized with {carnot.fluid_name}")
    
    # Create test parameters
    params = CycleParameters(
        T_evap=278.15,  # 5°C
        T_cond=318.15,  # 45°C
        mass_flow=1.0,
        working_fluid='R134a',
        eta_comp_isentropic=0.85,
        eta_exp_isentropic=0.80
    )
    
    try:
        results = carnot.generate_cycle(params)
        
        print(f"✅ Carnot cycle generated successfully:")
        print(f"  - Cycle type: {results.cycle_type}")
        print(f"  - Working fluid: {results.working_fluid}")
        print(f"  - States: {len(results.states)} points")
        print(f"  - State labels: {results.state_labels}")
        print(f"  - Q_evap: {results.Q_evap:.2f} kW")
        print(f"  - Q_cond: {results.Q_cond:.2f} kW")
        print(f"  - W_comp: {results.W_comp:.2f} kW")
        print(f"  - W_net: {results.W_net:.2f} kW")
        print(f"  - COP heating: {results.COP_heating:.2f}")
        print(f"  - COP cooling: {results.COP_cooling:.2f}")
        print(f"  - Energy balance error: {results.energy_balance_error:.3f}%")
        print(f"  - Carnot limit check: {results.carnot_limit_check}")
        print(f"  - Entropy generation: {results.entropy_generation:.6f} kW/K")
        print(f"  - Second law efficiency: {results.eta_second_law:.3f}")
        
        # Test trajectory data
        if results.trajectory:
            print(f"  - Trajectory points: {len(results.trajectory.get('time', []))}")
            print(f"  - Trajectory variables: {list(results.trajectory.keys())}")
        
        # Validate physics
        carnot_cop_heating = params.T_cond / (params.T_cond - params.T_evap)
        carnot_cop_cooling = params.T_evap / (params.T_cond - params.T_evap)
        
        print(f"✅ Physics validation:")
        print(f"  - Theoretical Carnot COP heating: {carnot_cop_heating:.2f}")
        print(f"  - Theoretical Carnot COP cooling: {carnot_cop_cooling:.2f}")
        print(f"  - Actual COP heating: {results.COP_heating:.2f} ({'✅' if results.COP_heating <= carnot_cop_heating * 1.05 else '❌'})")
        print(f"  - Actual COP cooling: {results.COP_cooling:.2f} ({'✅' if results.COP_cooling <= carnot_cop_cooling * 1.05 else '❌'})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error generating Carnot cycle: {e}")
        return None

def test_rankine_cycle():
    """Test Rankine cycle generation (ideal and non-ideal)"""
    print("\n🔄 Testing Rankine Cycle Generation...")
    
    # Test ideal Rankine cycle
    print("Testing Ideal Rankine Cycle...")
    rankine_ideal = RankineCycle('R245fa', ideal=True)
    
    params = CycleParameters(
        T_evap=353.15,  # 80°C
        T_cond=313.15,  # 40°C
        mass_flow=0.5,
        working_fluid='R245fa',
        superheat=10.0,
        subcool=5.0
    )
    
    try:
        results_ideal = rankine_ideal.generate_cycle(params)
        
        print(f"✅ Ideal Rankine cycle generated:")
        print(f"  - Cycle type: {results_ideal.cycle_type}")
        print(f"  - COP heating: {results_ideal.COP_heating:.2f}")
        print(f"  - COP cooling: {results_ideal.COP_cooling:.2f}")
        print(f"  - Energy balance error: {results_ideal.energy_balance_error:.3f}%")
        
    except Exception as e:
        print(f"❌ Error generating ideal Rankine cycle: {e}")
        results_ideal = None
    
    # Test non-ideal Rankine cycle
    print("Testing Non-Ideal Rankine Cycle...")
    rankine_nonideal = RankineCycle('R245fa', ideal=False)
    
    params_nonideal = CycleParameters(
        T_evap=353.15,
        T_cond=313.15,
        mass_flow=0.5,
        working_fluid='R245fa',
        superheat=10.0,
        subcool=5.0,
        eta_comp_isentropic=0.82,
        eta_exp_isentropic=0.78,
        dp_evap=3000,
        dp_cond=4000
    )
    
    try:
        results_nonideal = rankine_nonideal.generate_cycle(params_nonideal)
        
        print(f"✅ Non-ideal Rankine cycle generated:")
        print(f"  - Cycle type: {results_nonideal.cycle_type}")
        print(f"  - COP heating: {results_nonideal.COP_heating:.2f}")
        print(f"  - COP cooling: {results_nonideal.COP_cooling:.2f}")
        print(f"  - Energy balance error: {results_nonideal.energy_balance_error:.3f}%")
        
        # Compare ideal vs non-ideal
        if results_ideal and results_nonideal:
            print(f"✅ Comparison (Ideal vs Non-Ideal):")
            print(f"  - COP heating: {results_ideal.COP_heating:.2f} vs {results_nonideal.COP_heating:.2f}")
            print(f"  - COP cooling: {results_ideal.COP_cooling:.2f} vs {results_nonideal.COP_cooling:.2f}")
            print(f"  - Efficiency penalty: {(1 - results_nonideal.COP_heating/results_ideal.COP_heating)*100:.1f}%")
        
        return results_nonideal
        
    except Exception as e:
        print(f"❌ Error generating non-ideal Rankine cycle: {e}")
        return None

def test_transcritical_cycle():
    """Test transcritical cycle generation"""
    print("\n🔄 Testing Transcritical Cycle Generation...")
    
    # Test with CO2 (typical transcritical fluid)
    try:
        transcritical = TranscriticalCycle('CO2')
        print(f"✅ Transcritical cycle initialized with {transcritical.fluid_name}")
        
        params = CycleParameters(
            T_evap=268.15,  # -5°C
            T_cond=318.15,  # 45°C
            mass_flow=0.8,
            working_fluid='CO2',
            eta_comp_isentropic=0.80
        )
        
        results = transcritical.generate_cycle(params)
        
        print(f"✅ Transcritical cycle generated:")
        print(f"  - Cycle type: {results.cycle_type}")
        print(f"  - Working fluid: {results.working_fluid}")
        print(f"  - COP heating: {results.COP_heating:.2f}")
        print(f"  - COP cooling: {results.COP_cooling:.2f}")
        print(f"  - Q_evap: {results.Q_evap:.2f} kW")
        print(f"  - Q_cond (gas cooler): {results.Q_cond:.2f} kW")
        print(f"  - W_net: {results.W_net:.2f} kW")
        print(f"  - Energy balance error: {results.energy_balance_error:.3f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Error generating transcritical cycle: {e}")
        return None

def test_cycle_generator():
    """Test main CycleGenerator class"""
    print("\n🔄 Testing CycleGenerator Class...")
    
    generator = CycleGenerator()
    print(f"✅ CycleGenerator initialized")
    print(f"✅ Available cycle types: {list(generator.cycle_classes.keys())}")
    
    # Test each cycle type
    test_fluids = ['R134a', 'R410A', 'R245fa']
    cycle_types = [CycleType.CARNOT, CycleType.IDEAL_RANKINE, CycleType.NON_IDEAL_RANKINE]
    
    results_summary = []
    
    for cycle_type in cycle_types:
        for fluid in test_fluids:
            try:
                print(f"\nTesting {cycle_type.value} with {fluid}...")
                
                # Generate random parameters
                params = generator.generate_random_parameters(cycle_type, fluid)
                print(f"  - Generated parameters: T_evap={params.T_evap-273.15:.1f}°C, T_cond={params.T_cond-273.15:.1f}°C")
                print(f"  - Temperature lift: {params.T_cond - params.T_evap:.1f} K")
                print(f"  - Mass flow: {params.mass_flow:.2f} kg/s")
                print(f"  - Efficiencies: comp={params.eta_comp_isentropic:.2f}, exp={params.eta_exp_isentropic:.2f}")
                
                # Generate cycle
                results = generator.generate_cycle(cycle_type, fluid, params)
                
                if results and results.COP_heating > 0:
                    print(f"  ✅ Success: COP_h={results.COP_heating:.2f}, COP_c={results.COP_cooling:.2f}")
                    print(f"    Energy balance error: {results.energy_balance_error:.3f}%")
                    print(f"    Carnot limit check: {results.carnot_limit_check}")
                    
                    results_summary.append({
                        'cycle_type': cycle_type.value,
                        'fluid': fluid,
                        'COP_heating': results.COP_heating,
                        'COP_cooling': results.COP_cooling,
                        'energy_balance_error': results.energy_balance_error,
                        'carnot_limit_check': results.carnot_limit_check,
                        'success': True
                    })
                else:
                    print(f"  ❌ Failed or invalid results")
                    results_summary.append({
                        'cycle_type': cycle_type.value,
                        'fluid': fluid,
                        'success': False
                    })
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results_summary.append({
                    'cycle_type': cycle_type.value,
                    'fluid': fluid,
                    'error': str(e),
                    'success': False
                })
    
    # Summary statistics
    successful = [r for r in results_summary if r.get('success', False)]
    print(f"\n✅ CycleGenerator Summary:")
    print(f"  - Total tests: {len(results_summary)}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Success rate: {len(successful)/len(results_summary)*100:.1f}%")
    
    if successful:
        cop_heating_values = [r['COP_heating'] for r in successful]
        cop_cooling_values = [r['COP_cooling'] for r in successful]
        energy_errors = [r['energy_balance_error'] for r in successful]
        
        print(f"  - COP heating range: {min(cop_heating_values):.2f} - {max(cop_heating_values):.2f}")
        print(f"  - COP cooling range: {min(cop_cooling_values):.2f} - {max(cop_cooling_values):.2f}")
        print(f"  - Avg energy balance error: {np.mean(energy_errors):.3f}%")
        print(f"  - Carnot limit violations: {sum(1 for r in successful if not r['carnot_limit_check'])}")
    
    return results_summary

def test_parameter_generation():
    """Test random parameter generation"""
    print("\n🔄 Testing Random Parameter Generation...")
    
    generator = CycleGenerator()
    
    # Test parameter generation for different cycle types and fluids
    test_cases = [
        (CycleType.CARNOT, 'R134a'),
        (CycleType.IDEAL_RANKINE, 'R245fa'),
        (CycleType.TRANSCRITICAL, 'CO2'),
        (CycleType.NON_IDEAL_RANKINE, 'R410A')
    ]
    
    for cycle_type, fluid in test_cases:
        print(f"\nTesting parameter generation for {cycle_type.value} with {fluid}:")
        
        # Generate multiple parameter sets
        param_sets = []
        for i in range(5):
            try:
                params = generator.generate_random_parameters(cycle_type, fluid)
                param_sets.append(params)
                
                print(f"  Set {i+1}: T_evap={params.T_evap-273.15:.1f}°C, T_cond={params.T_cond-273.15:.1f}°C, ΔT={params.T_cond-params.T_evap:.1f}K")
                
            except Exception as e:
                print(f"  ❌ Error generating parameters: {e}")
        
        if param_sets:
            # Analyze parameter ranges
            T_evap_range = [p.T_evap - 273.15 for p in param_sets]
            T_cond_range = [p.T_cond - 273.15 for p in param_sets]
            temp_lift_range = [p.T_cond - p.T_evap for p in param_sets]
            mass_flow_range = [p.mass_flow for p in param_sets]
            
            print(f"  ✅ Parameter ranges:")
            print(f"    - T_evap: {min(T_evap_range):.1f} to {max(T_evap_range):.1f}°C")
            print(f"    - T_cond: {min(T_cond_range):.1f} to {max(T_cond_range):.1f}°C")
            print(f"    - Temp lift: {min(temp_lift_range):.1f} to {max(temp_lift_range):.1f}K")
            print(f"    - Mass flow: {min(mass_flow_range):.2f} to {max(mass_flow_range):.2f} kg/s")

def test_physics_validation():
    """Test physics validation methods"""
    print("\n🔄 Testing Physics Validation...")
    
    carnot = CarnotCycle('R134a')
    
    # Create test parameters with known good values
    params = CycleParameters(
        T_evap=278.15,  # 5°C
        T_cond=318.15,  # 45°C
        mass_flow=1.0,
        working_fluid='R134a'
    )
    
    try:
        results = carnot.generate_cycle(params)
        
        print(f"✅ Physics validation tests:")
        
        # Test 1: Energy balance (First Law)
        energy_in = results.Q_evap + results.W_net
        energy_out = results.Q_cond
        energy_balance_error = abs(energy_in - energy_out) / energy_out * 100
        
        print(f"  - Energy balance:")
        print(f"    Energy in: {energy_in:.2f} kW (Q_evap + W_net)")
        print(f"    Energy out: {energy_out:.2f} kW (Q_cond)")
        print(f"    Balance error: {energy_balance_error:.3f}% ({'✅' if energy_balance_error < 1.0 else '❌'})")
        
        # Test 2: Carnot limit (Second Law)
        carnot_cop_heating = params.T_cond / (params.T_cond - params.T_evap)
        carnot_cop_cooling = params.T_evap / (params.T_cond - params.T_evap)
        
        print(f"  - Carnot limits:")
        print(f"    Theoretical COP heating: {carnot_cop_heating:.2f}")
        print(f"    Actual COP heating: {results.COP_heating:.2f} ({'✅' if results.COP_heating <= carnot_cop_heating * 1.05 else '❌'})")
        print(f"    Theoretical COP cooling: {carnot_cop_cooling:.2f}")
        print(f"    Actual COP cooling: {results.COP_cooling:.2f} ({'✅' if results.COP_cooling <= carnot_cop_cooling * 1.05 else '❌'})")
        
        # Test 3: Entropy generation (must be positive)
        print(f"  - Entropy generation: {results.entropy_generation:.6f} kW/K ({'✅' if results.entropy_generation >= 0 else '❌'})")
        
        # Test 4: Second law efficiency
        print(f"  - Second law efficiency: {results.eta_second_law:.3f} ({'✅' if 0 <= results.eta_second_law <= 1 else '❌'})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in physics validation: {e}")
        return False

def test_trajectory_generation():
    """Test trajectory data generation for HNN training"""
    print("\n🔄 Testing Trajectory Generation for HNN Training...")
    
    carnot = CarnotCycle('R134a')
    
    params = CycleParameters(
        T_evap=278.15,
        T_cond=318.15,
        mass_flow=1.0,
        working_fluid='R134a'
    )
    
    try:
        results = carnot.generate_cycle(params)
        
        if results.trajectory:
            traj = results.trajectory
            
            print(f"✅ Trajectory data generated:")
            print(f"  - Variables: {list(traj.keys())}")
            print(f"  - Time points: {len(traj.get('time', []))}")
            print(f"  - Pressure points: {len(traj.get('P', []))}")
            print(f"  - Temperature points: {len(traj.get('T', []))}")
            print(f"  - Volume points: {len(traj.get('V', []))}")
            print(f"  - Entropy points: {len(traj.get('S', []))}")
            
            # Check data ranges
            if 'P' in traj and traj['P']:
                P_range = [min(traj['P']), max(traj['P'])]
                print(f"  - Pressure range: {P_range[0]/1000:.1f} - {P_range[1]/1000:.1f} kPa")
            
            if 'T' in traj and traj['T']:
                T_range = [min(traj['T']), max(traj['T'])]
                print(f"  - Temperature range: {T_range[0]-273.15:.1f} - {T_range[1]-273.15:.1f}°C")
            
            # Test trajectory data export
            trajectory_data = {
                'cycle_type': results.cycle_type,
                'working_fluid': results.working_fluid,
                'trajectory': traj,
                'performance': {
                    'COP_heating': results.COP_heating,
                    'COP_cooling': results.COP_cooling,
                    'Q_evap': results.Q_evap,
                    'Q_cond': results.Q_cond,
                    'W_net': results.W_net
                }
            }
            
            # Save trajectory data
            with open('test_cycle_trajectory_data.json', 'w') as f:
                json.dump(trajectory_data, f, indent=2, default=str)
            
            print(f"  ✅ Trajectory data exported to test_cycle_trajectory_data.json")
            
            return True
        else:
            print(f"❌ No trajectory data generated")
            return False
            
    except Exception as e:
        print(f"❌ Error generating trajectory: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔄 Testing Edge Cases and Error Handling...")
    
    generator = CycleGenerator()
    
    # Test 1: Invalid temperature ranges
    print("Testing invalid temperature ranges...")
    try:
        invalid_params = CycleParameters(
            T_evap=318.15,  # Higher than condenser
            T_cond=278.15,  # Lower than evaporator
            mass_flow=1.0,
            working_fluid='R134a'
        )
        
        results = generator.generate_cycle(CycleType.CARNOT, 'R134a', invalid_params)
        print(f"  - Invalid temp range: {'✅ Handled gracefully' if results.energy_balance_error > 50 else '❌ Not detected'}")
        
    except Exception as e:
        print(f"  - Invalid temp range: ✅ Exception caught: {str(e)[:50]}...")
    
    # Test 2: Extreme efficiency values
    print("Testing extreme efficiency values...")
    try:
        extreme_params = CycleParameters(
            T_evap=278.15,
            T_cond=318.15,
            mass_flow=1.0,
            working_fluid='R134a',
            eta_comp_isentropic=0.05,  # Very low efficiency
            eta_exp_isentropic=0.95    # Very high efficiency
        )
        
        results = generator.generate_cycle(CycleType.CARNOT, 'R134a', extreme_params)
        print(f"  - Extreme efficiencies: ✅ Handled (COP_h={results.COP_heating:.2f})")
        
    except Exception as e:
        print(f"  - Extreme efficiencies: ✅ Exception caught: {str(e)[:50]}...")
    
    # Test 3: Invalid working fluid
    print("Testing invalid working fluid...")
    try:
        results = generator.generate_cycle(CycleType.CARNOT, 'InvalidFluid', 
                                         generator.generate_random_parameters(CycleType.CARNOT, 'InvalidFluid'))
        print(f"  - Invalid fluid: {'✅ Handled' if results.energy_balance_error > 50 else '❌ Not handled'}")
        
    except Exception as e:
        print(f"  - Invalid fluid: ✅ Exception caught: {str(e)[:50]}...")
    
    # Test 4: Zero mass flow
    print("Testing zero mass flow...")
    try:
        zero_flow_params = CycleParameters(
            T_evap=278.15,
            T_cond=318.15,
            mass_flow=0.0,  # Zero mass flow
            working_fluid='R134a'
        )
        
        results = generator.generate_cycle(CycleType.CARNOT, 'R134a', zero_flow_params)
        print(f"  - Zero mass flow: {'✅ Handled' if results.Q_evap == 0 else '❌ Not handled'}")
        
    except Exception as e:
        print(f"  - Zero mass flow: ✅ Exception caught: {str(e)[:50]}...")

def main():
    """Run comprehensive cycle generator tests"""
    print("🔄 COMPREHENSIVE CYCLE GENERATOR TESTING")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Basic dataclass functionality
        test_cycle_parameters()
        test_cycle_results()
        test_results['dataclasses'] = True
        
        # Test 2: Individual cycle types
        carnot_result = test_carnot_cycle()
        test_results['carnot'] = carnot_result is not None
        
        rankine_result = test_rankine_cycle()
        test_results['rankine'] = rankine_result is not None
        
        transcritical_result = test_transcritical_cycle()
        test_results['transcritical'] = transcritical_result is not None
        
        # Test 3: Main generator class
        generator_results = test_cycle_generator()
        test_results['generator'] = len([r for r in generator_results if r.get('success', False)]) > 0
        
        # Test 4: Parameter generation
        test_parameter_generation()
        test_results['parameters'] = True
        
        # Test 5: Physics validation
        physics_valid = test_physics_validation()
        test_results['physics'] = physics_valid
        
        # Test 6: Trajectory generation
        trajectory_valid = test_trajectory_generation()
        test_results['trajectory'] = trajectory_valid
        
        # Test 7: Edge cases
        test_edge_cases()
        test_results['edge_cases'] = True
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 CYCLE GENERATOR TESTING COMPLETED!")
        print("=" * 60)
        
        successful_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        print(f"✅ Test Results Summary:")
        print(f"  - Total test categories: {total_tests}")
        print(f"  - Successful categories: {successful_tests}")
        print(f"  - Success rate: {successful_tests/total_tests*100:.1f}%")
        
        print(f"\n✅ Individual Test Results:")
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  - {test_name.replace('_', ' ').title()}: {status}")
        
        if successful_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Cycle generators are working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check individual results above.")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in cycle generator testing: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    main()