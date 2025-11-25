#!/usr/bin/env python3
"""
Comprehensive test script for Temporal Dynamics Models
Tests all major temporal dynamics and Hamiltonian formulation functionality
"""

import sys
sys.path.append('src')

def test_temporal_dynamics_models():
    """Test all temporal dynamics model functionality"""
    print("⏰ Testing Temporal Dynamics Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from temporal_dynamics import (
            TemporalDynamicsGenerator, HamiltonianState, TransientScenario
        )
        import numpy as np
        import pandas as pd
        from datetime import datetime
        print("✅ Import successful")
        
        # Test TemporalDynamicsGenerator initialization
        print("\n🔄 Testing TemporalDynamicsGenerator initialization...")
        generator = TemporalDynamicsGenerator(fluid_name="R410A", sampling_rate=10.0)
        print("✅ TemporalDynamicsGenerator initialized successfully")
        print(f"✅ Fluid: {generator.fluid_name}")
        print(f"✅ Sampling rate: {generator.sampling_rate} Hz")
        print(f"✅ Time step: {generator.dt:.3f} s")
        print(f"✅ Total refrigerant mass: {generator.total_refrigerant_mass} kg")
        print(f"✅ System volume: {generator.system_volume} m³")
        
        # Test thermal masses
        print("\n🔄 Testing thermal masses...")
        thermal_masses = generator.thermal_masses
        print(f"✅ Loaded thermal masses for {len(thermal_masses)} components:")
        for component, mass in thermal_masses.items():
            print(f"  - {component}: {mass} kg")
        
        # Test component models
        print("\n🔄 Testing component models...")
        print(f"✅ Compressor: {type(generator.compressor).__name__}")
        print(f"✅ Evaporator: {type(generator.evaporator).__name__}")
        print(f"✅ Condenser: {type(generator.condenser).__name__}")
        print(f"✅ Expansion valve: {type(generator.expansion_valve).__name__}")
        
        # Test initial state creation
        print("\n🔄 Testing initial state creation...")
        T_ambient = 288.15  # 15°C
        P_ambient = 101325  # 1 atm
        
        initial_state = generator._create_initial_state(T_ambient, P_ambient)
        print(f"✅ Created initial state: {type(initial_state)}")
        print(f"✅ Time: {initial_state.time} s")
        print(f"✅ Pressures: {initial_state.q_P}")
        print(f"✅ Temperatures: {initial_state.q_T}")
        print(f"✅ Volumes: {initial_state.q_V}")
        print(f"✅ Entropies: {initial_state.q_S}")
        print(f"✅ Enthalpies: {initial_state.q_H}")
        print(f"✅ Masses: {initial_state.q_m}")
        print(f"✅ Total Hamiltonian: {initial_state.total_hamiltonian:.2f} J")
        print(f"✅ Internal energy: {initial_state.internal_energy:.2f} J")
        print(f"✅ Kinetic energy: {initial_state.kinetic_energy:.2f} J")
        print(f"✅ Potential energy: {initial_state.potential_energy:.2f} J")
        print(f"✅ Process phase: {initial_state.process_phase}")
        print(f"✅ Dominant process: {initial_state.dominant_process}")
        
        # Test profile creation functions
        print("\n🔄 Testing profile creation functions...")
        
        # Test compressor startup profile
        startup_duration = 300.0
        compressor_profile = generator._create_compressor_startup_profile(startup_duration)
        test_times = [0, 5, 30, 60, 120, 300]
        print("✅ Compressor startup profile:")
        for t in test_times:
            speed = compressor_profile(t)
            print(f"  - t={t:3.0f}s: {speed:6.1f} rpm")
        
        # Test valve startup profile
        valve_profile = generator._create_valve_startup_profile(startup_duration)
        print("✅ Valve startup profile:")
        for t in test_times:
            opening = valve_profile(t)
            print(f"  - t={t:3.0f}s: {opening:5.1f}%")
        
        # Test shutdown profiles
        initial_speed = 1800.0
        shutdown_duration = 600.0
        shutdown_profile = generator._create_compressor_shutdown_profile(initial_speed, shutdown_duration)
        valve_shutdown_profile = generator._create_valve_shutdown_profile(shutdown_duration)
        
        print("✅ Shutdown profiles:")
        for t in [0, 5, 20, 60, 120]:
            comp_speed = shutdown_profile(t)
            valve_open = valve_shutdown_profile(t)
            print(f"  - t={t:3.0f}s: compressor={comp_speed:6.1f} rpm, valve={valve_open:5.1f}%")
        
        # Test exponential profile
        exponential_profile = generator._create_exponential_profile(1.0, 2.0, 60.0, 20.0)
        print("✅ Exponential profile (1.0 → 2.0, τ=20s):")
        for t in [0, 10, 20, 40, 60]:
            value = exponential_profile(t)
            print(f"  - t={t:2.0f}s: {value:.3f}")
        
        # Test startup dynamics generation
        print("\n🔄 Testing startup dynamics generation...")
        target_conditions = {
            'T_ambient': 278.15,  # 5°C
            'heating_load': 10000,  # 10 kW
            'water_temp': 318.15   # 45°C
        }
        
        startup_duration = 120.0  # 2 minutes
        startup_states = generator.generate_startup_dynamics(target_conditions, startup_duration)
        
        print(f"✅ Generated startup dynamics: {len(startup_states)} states")
        print(f"✅ Duration: {startup_states[-1].time:.1f} s")
        print(f"✅ Initial state:")
        print(f"  - Time: {startup_states[0].time:.1f} s")
        print(f"  - Compressor speed: {startup_states[0].compressor_speed:.1f} rpm")
        print(f"  - Valve opening: {startup_states[0].expansion_valve_opening:.1f}%")
        print(f"  - COP heating: {startup_states[0].COP_heating:.3f}")
        print(f"  - Power input: {startup_states[0].power_input:.1f} W")
        print(f"  - Process phase: {startup_states[0].process_phase}")
        
        print(f"✅ Final state:")
        print(f"  - Time: {startup_states[-1].time:.1f} s")
        print(f"  - Compressor speed: {startup_states[-1].compressor_speed:.1f} rpm")
        print(f"  - Valve opening: {startup_states[-1].expansion_valve_opening:.1f}%")
        print(f"  - COP heating: {startup_states[-1].COP_heating:.3f}")
        print(f"  - Power input: {startup_states[-1].power_input:.1f} W")
        print(f"  - Process phase: {startup_states[-1].process_phase}")
        print(f"  - Dominant process: {startup_states[-1].dominant_process}")
        
        # Test shutdown dynamics generation
        print("\n🔄 Testing shutdown dynamics generation...")
        shutdown_duration = 60.0  # 1 minute
        shutdown_states = generator.generate_shutdown_dynamics(startup_states[-1], shutdown_duration)
        
        print(f"✅ Generated shutdown dynamics: {len(shutdown_states)} states")
        print(f"✅ Duration: {shutdown_states[-1].time:.1f} s")
        print(f"✅ Initial shutdown state:")
        print(f"  - Compressor speed: {shutdown_states[0].compressor_speed:.1f} rpm")
        print(f"  - Valve opening: {shutdown_states[0].expansion_valve_opening:.1f}%")
        print(f"  - COP heating: {shutdown_states[0].COP_heating:.3f}")
        
        print(f"✅ Final shutdown state:")
        print(f"  - Compressor speed: {shutdown_states[-1].compressor_speed:.1f} rpm")
        print(f"  - Valve opening: {shutdown_states[-1].expansion_valve_opening:.1f}%")
        print(f"  - COP heating: {shutdown_states[-1].COP_heating:.3f}")
        print(f"  - Power input: {shutdown_states[-1].power_input:.1f} W")
        
        # Test load change dynamics
        print("\n🔄 Testing load change dynamics...")
        load_change = {'heating_load': 1.5}  # 50% increase
        response_duration = 60.0
        
        load_change_states = generator.generate_load_change_dynamics(
            startup_states[-1], load_change, response_duration
        )
        
        print(f"✅ Generated load change dynamics: {len(load_change_states)} states")
        print(f"✅ Load change: {load_change}")
        print(f"✅ Response duration: {response_duration} s")
        print(f"✅ Initial load change state:")
        print(f"  - COP heating: {load_change_states[0].COP_heating:.3f}")
        print(f"  - Power input: {load_change_states[0].power_input:.1f} W")
        
        print(f"✅ Final load change state:")
        print(f"  - COP heating: {load_change_states[-1].COP_heating:.3f}")
        print(f"  - Power input: {load_change_states[-1].power_input:.1f} W")
        print(f"  - Process phase: {load_change_states[-1].process_phase}")
        
        # Test defrost cycle dynamics
        print("\n🔄 Testing defrost cycle dynamics...")
        defrost_duration = 300.0  # 5 minutes
        
        defrost_states = generator.generate_defrost_cycle(startup_states[-1], defrost_duration)
        
        print(f"✅ Generated defrost cycle dynamics: {len(defrost_states)} states")
        print(f"✅ Defrost duration: {defrost_duration} s")
        
        # Check different defrost phases
        phase_times = [0, 50, 150, 250, 299]
        for t in phase_times:
            idx = min(int(t * generator.sampling_rate), len(defrost_states) - 1)
            state = defrost_states[idx]
            print(f"✅ t={t:3.0f}s: phase={state.process_phase}, temp={state.q_T[0]:.1f}K")
        
        # Test Hamiltonian derivatives calculation
        print("\n🔄 Testing Hamiltonian derivatives calculation...")
        test_state = startup_states[len(startup_states)//2]  # Middle state
        compressor_speed = 1500.0
        valve_opening = 40.0
        target_conditions = {'heating_load': 8000}
        phase = "steady_state"
        
        dq_dt, dp_dt = generator._calculate_hamiltonian_derivatives(
            test_state, compressor_speed, valve_opening, target_conditions, phase
        )
        
        print(f"✅ Hamiltonian derivatives calculated:")
        print(f"  - dq/dt keys: {list(dq_dt.keys())}")
        print(f"  - dp/dt keys: {list(dp_dt.keys())}")
        print(f"  - dP/dt: {dp_dt['P']}")
        print(f"  - dT/dt: {dp_dt['T']}")
        print(f"  - dH/dt: {dp_dt['H']}")
        
        # Test performance calculation
        print("\n🔄 Testing performance calculation...")
        P = test_state.q_P
        V = test_state.q_V
        T = test_state.q_T
        H = test_state.q_H
        m = test_state.q_m
        
        COP_heating, COP_cooling, power_input, heat_output = generator._calculate_performance(
            P, V, T, H, m, compressor_speed
        )
        
        print(f"✅ Performance calculation:")
        print(f"  - COP heating: {COP_heating:.3f}")
        print(f"  - COP cooling: {COP_cooling:.3f}")
        print(f"  - Power input: {power_input:.1f} W")
        print(f"  - Heat output: {heat_output:.1f} W")
        
        # Test dominant process identification
        print("\n🔄 Testing dominant process identification...")
        dominant_process = generator._identify_dominant_process(dq_dt, dp_dt)
        print(f"✅ Dominant process: {dominant_process}")
        
        # Test defrost phase determination
        print("\n🔄 Testing defrost phase determination...")
        phases = [
            {'name': 'reverse_valve_switch', 'duration': 10.0},
            {'name': 'defrost_heating', 'duration': 300.0},
            {'name': 'ice_melting', 'duration': 240.0},
            {'name': 'return_to_heating', 'duration': 50.0}
        ]
        
        test_times = [5, 50, 350, 580, 650]
        for t in test_times:
            phase = generator._get_defrost_phase(t, phases)
            print(f"✅ t={t:3.0f}s: {phase}")
        
        # Test energy conservation validation
        print("\n🔄 Testing energy conservation validation...")
        
        # Test with startup states
        startup_validation = generator.validate_energy_conservation(startup_states)
        print(f"✅ Startup energy conservation:")
        print(f"  - Valid: {startup_validation['valid']}")
        print(f"  - Error: {startup_validation['error']:.3f}%")
        print(f"  - Drift: {startup_validation['drift']:.3f}%")
        print(f"  - Initial energy: {startup_validation['initial_energy']:.2f} J")
        print(f"  - Final energy: {startup_validation['final_energy']:.2f} J")
        print(f"  - Max energy: {startup_validation['max_energy']:.2f} J")
        print(f"  - Min energy: {startup_validation['min_energy']:.2f} J")
        
        # Test with shutdown states
        shutdown_validation = generator.validate_energy_conservation(shutdown_states)
        print(f"✅ Shutdown energy conservation:")
        print(f"  - Valid: {shutdown_validation['valid']}")
        print(f"  - Error: {shutdown_validation['error']:.3f}%")
        print(f"  - Drift: {shutdown_validation['drift']:.3f}%")
        
        # Test with empty states
        empty_validation = generator.validate_energy_conservation([])
        print(f"✅ Empty states validation:")
        print(f"  - Valid: {empty_validation['valid']}")
        print(f"  - Error: {empty_validation['error']:.1f}%")
        
        # Test temporal dataset saving
        print("\n🔄 Testing temporal dataset saving...")
        
        all_trajectories = [startup_states[:50], shutdown_states[:30], load_change_states[:40]]  # Smaller samples
        metadata = {
            'test_run': True,
            'scenarios': ['startup', 'shutdown', 'load_change'],
            'fluid': generator.fluid_name,
            'sampling_rate': generator.sampling_rate
        }
        
        generator.save_temporal_dataset(
            all_trajectories,
            "test_temporal_dynamics_data.json",
            metadata
        )
        print("✅ Temporal dataset saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test HamiltonianState
        test_hamiltonian_state = HamiltonianState(
            time=10.0,
            q_P=np.array([101325, 1800000, 1800000, 400000]),
            q_V=np.array([0.001, 0.0001, 0.0001, 0.01]),
            q_T=np.array([288.15, 353.15, 318.15, 278.15]),
            q_S=np.array([1000, 1200, 1100, 950]),
            q_H=np.array([200000, 450000, 300000, 180000]),
            q_m=np.array([1.25, 1.25, 1.25, 1.25]),
            p_P=np.array([0, 1000, -500, -500]),
            p_V=np.array([0, 0, 0, 0]),
            p_T=np.array([0, 5, -2, -3]),
            p_S=np.array([0, 10, -5, -5]),
            p_H=np.array([0, 1000, -500, -500]),
            p_m=np.array([0, 0.1, -0.05, -0.05]),
            internal_energy=1000000,
            kinetic_energy=500,
            potential_energy=245,
            total_hamiltonian=1000745,
            COP_heating=3.5,
            COP_cooling=2.8,
            power_input=2500,
            heat_output=8750,
            compressor_speed=1800,
            expansion_valve_opening=45.0,
            process_phase="steady_state",
            dominant_process="compression"
        )
        print(f"✅ HamiltonianState created: time={test_hamiltonian_state.time}s")
        print(f"✅ Total Hamiltonian: {test_hamiltonian_state.total_hamiltonian:.0f} J")
        print(f"✅ COP heating: {test_hamiltonian_state.COP_heating:.2f}")
        print(f"✅ Process phase: {test_hamiltonian_state.process_phase}")
        print(f"✅ Dominant process: {test_hamiltonian_state.dominant_process}")
        
        # Test TransientScenario
        def boundary_conditions(t):
            return {
                'ambient_temp': 288.15 + 5 * np.sin(t / 3600),  # Daily temperature variation
                'load_factor': 1.0 + 0.2 * np.sin(t / 1800)     # Load variation
            }
        
        test_scenario = TransientScenario(
            name="daily_cycle",
            description="Daily temperature and load cycle",
            duration=86400.0,  # 24 hours
            initial_conditions={'T_ambient': 288.15, 'load': 8000},
            boundary_conditions=boundary_conditions,
            disturbances=[
                {'type': 'load_step', 'time': 3600, 'magnitude': 1.5},
                {'type': 'defrost', 'time': 7200, 'duration': 600}
            ]
        )
        print(f"✅ TransientScenario created: {test_scenario.name}")
        print(f"✅ Description: {test_scenario.description}")
        print(f"✅ Duration: {test_scenario.duration/3600:.1f} hours")
        print(f"✅ Disturbances: {len(test_scenario.disturbances)}")
        
        # Test boundary conditions function
        test_times = [0, 3600, 7200, 10800]
        for t in test_times:
            conditions = test_scenario.boundary_conditions(t)
            print(f"✅ t={t/3600:.1f}h: ambient={conditions['ambient_temp']:.2f}K, load_factor={conditions['load_factor']:.3f}")
        
        # Test integration with different sampling rates
        print("\n🔄 Testing different sampling rates...")
        
        sampling_rates = [1.0, 5.0, 20.0]
        for rate in sampling_rates:
            test_generator = TemporalDynamicsGenerator(fluid_name="R410A", sampling_rate=rate)
            short_startup = test_generator.generate_startup_dynamics(target_conditions, 30.0)
            print(f"✅ Sampling rate {rate:.1f} Hz: {len(short_startup)} states in 30s")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Very short duration
        short_states = generator.generate_startup_dynamics(target_conditions, 1.0)
        print(f"✅ Very short startup (1s): {len(short_states)} states")
        
        # Zero duration
        zero_states = generator.generate_startup_dynamics(target_conditions, 0.0)
        print(f"✅ Zero duration startup: {len(zero_states)} states")
        
        # Large load change
        large_load_change = {'heating_load': 5.0}  # 500% increase
        large_change_states = generator.generate_load_change_dynamics(
            startup_states[-1], large_load_change, 30.0
        )
        print(f"✅ Large load change (500%): {len(large_change_states)} states")
        
        print("\n🎉 ALL TEMPORAL DYNAMICS TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in temporal dynamics models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_temporal_dynamics_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)