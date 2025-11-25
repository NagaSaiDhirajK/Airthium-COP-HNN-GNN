#!/usr/bin/env python3
"""
Comprehensive test script for Part-Load Cycling Models
Tests all major part-load operation and cycling functionality
"""

import sys
sys.path.append('src')

def test_part_load_cycling_models():
    """Test all part-load cycling model functionality"""
    print("⚡ Testing Part-Load Cycling Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from part_load_cycling import (
            PartLoadCyclingModel, PartLoadState, CyclingParameters
        )
        import numpy as np
        import pandas as pd
        from datetime import datetime
        print("✅ Import successful")
        
        # Test PartLoadCyclingModel initialization
        print("\n🔄 Testing PartLoadCyclingModel initialization...")
        cycling_model = PartLoadCyclingModel()
        print("✅ PartLoadCyclingModel initialized successfully")
        
        # Test part-load curves
        print("\n🔄 Testing part-load curves...")
        part_load_curves = cycling_model.part_load_curves
        print(f"✅ Loaded part-load curves for {len(part_load_curves)} system types:")
        for system_type, curves in part_load_curves.items():
            print(f"  - {system_type}: {len(curves)} curve types")
            # Test curve functions
            test_plr = 0.5
            cop_value = curves['cop_curve'](test_plr)
            capacity_value = curves['capacity_curve'](test_plr)
            eir_value = curves['eir_curve'](test_plr)
            print(f"    * At PLR=0.5: COP={cop_value:.3f}, Capacity={capacity_value:.3f}, EIR={eir_value:.3f}")
        
        # Test cycling degradation parameters
        print("\n🔄 Testing cycling degradation parameters...")
        cycling_degradation = cycling_model.cycling_degradation
        print(f"✅ Loaded cycling degradation for {len(cycling_degradation)} system types:")
        for system_type, params in cycling_degradation.items():
            print(f"  - {system_type}: degradation coeff = {params['degradation_coefficient']:.2f}")
            print(f"    * minimum PLR = {params['minimum_plr']:.2f}")
            print(f"    * cycling loss = {params['cycling_loss_factor']:.3f}")
        
        # Test control parameters
        print("\n🔄 Testing control parameters...")
        control_parameters = cycling_model.control_parameters
        print(f"✅ Loaded control parameters for {len(control_parameters)} building types:")
        for building_type, params in control_parameters.items():
            print(f"  - {building_type}: deadband = {params['deadband_K']:.1f} K")
            print(f"    * min on time = {params['minimum_on_time_min']:.1f} min")
        
        # Test thermal mass effects
        print("\n🔄 Testing thermal mass effects...")
        thermal_mass_effects = cycling_model.thermal_mass_effects
        print(f"✅ Loaded thermal mass effects for {len(thermal_mass_effects)} construction types:")
        for mass_type, effects in thermal_mass_effects.items():
            print(f"  - {mass_type}: time constant = {effects['time_constant_hours']:.1f} h")
            print(f"    * smoothing factor = {effects['load_smoothing_factor']:.1f}")
        
        # Test part-load state initialization - Single speed system
        print("\n🔄 Testing single-speed system initialization...")
        single_speed_state = cycling_model.initialize_part_load_state(
            system_id="test_single_001",
            system_type="residential",
            design_capacity_kW=10.0,
            current_load_kW=3.0,  # 30% load
            control_type="residential",
            modulation_type="single_speed"
        )
        print(f"✅ Created single-speed state: {type(single_speed_state)}")
        print(f"✅ System ID: {single_speed_state.system_id}")
        print(f"✅ Load factor: {single_speed_state.load_factor:.3f}")
        print(f"✅ Is cycling: {single_speed_state.is_cycling}")
        print(f"✅ Compressor capacity: {single_speed_state.compressor_capacity_percent:.1f}%")
        print(f"✅ Part-load COP: {single_speed_state.part_load_cop:.3f}")
        print(f"✅ Part-load capacity: {single_speed_state.part_load_capacity_kW:.2f} kW")
        print(f"✅ Part-load power: {single_speed_state.part_load_power_kW:.2f} kW")
        print(f"✅ Cycles per hour: {single_speed_state.cycles_per_hour:.1f}")
        print(f"✅ Cycling degradation: {single_speed_state.cycling_degradation_factor:.3f}")
        
        # Test part-load state initialization - Variable speed system
        print("\n🔄 Testing variable-speed system initialization...")
        variable_speed_state = cycling_model.initialize_part_load_state(
            system_id="test_variable_001",
            system_type="commercial",
            design_capacity_kW=15.0,
            current_load_kW=4.5,  # 30% load
            control_type="commercial",
            modulation_type="variable_speed"
        )
        print(f"✅ Created variable-speed state: {type(variable_speed_state)}")
        print(f"✅ Load factor: {variable_speed_state.load_factor:.3f}")
        print(f"✅ Is cycling: {variable_speed_state.is_cycling}")
        print(f"✅ Compressor capacity: {variable_speed_state.compressor_capacity_percent:.1f}%")
        print(f"✅ Modulation level: {variable_speed_state.modulation_level:.3f}")
        print(f"✅ Part-load COP: {variable_speed_state.part_load_cop:.3f}")
        print(f"✅ Cycling degradation: {variable_speed_state.cycling_degradation_factor:.3f}")
        
        # Test part-load state initialization - Two speed system
        print("\n🔄 Testing two-speed system initialization...")
        two_speed_state = cycling_model.initialize_part_load_state(
            system_id="test_two_speed_001",
            system_type="commercial",
            design_capacity_kW=12.0,
            current_load_kW=6.0,  # 50% load
            control_type="commercial",
            modulation_type="two_speed"
        )
        print(f"✅ Created two-speed state: {type(two_speed_state)}")
        print(f"✅ Load factor: {two_speed_state.load_factor:.3f}")
        print(f"✅ Is cycling: {two_speed_state.is_cycling}")
        print(f"✅ Part-load COP: {two_speed_state.part_load_cop:.3f}")
        
        # Test part-load state initialization - Hot gas bypass system
        print("\n🔄 Testing hot gas bypass system initialization...")
        hgb_state = cycling_model.initialize_part_load_state(
            system_id="test_hgb_001",
            system_type="industrial",
            design_capacity_kW=20.0,
            current_load_kW=8.0,  # 40% load
            control_type="industrial",
            modulation_type="hot_gas_bypass"
        )
        print(f"✅ Created hot gas bypass state: {type(hgb_state)}")
        print(f"✅ Load factor: {hgb_state.load_factor:.3f}")
        print(f"✅ Is cycling: {hgb_state.is_cycling}")
        print(f"✅ Part-load COP: {hgb_state.part_load_cop:.3f}")
        
        # Test part-load operation updates
        print("\n🔄 Testing part-load operation updates...")
        
        # Update single-speed system with changing load
        print("✅ Testing single-speed load changes:")
        original_load = single_speed_state.building_load_kW
        original_cop = single_speed_state.part_load_cop
        
        # Increase load
        updated_state = cycling_model.update_part_load_operation(
            single_speed_state, 6.0, 15.0  # 60% load, 15-minute time step
        )
        print(f"  - Load change: {original_load:.1f} → {updated_state.building_load_kW:.1f} kW")
        print(f"  - Load factor: {updated_state.load_factor:.3f}")
        print(f"  - COP change: {original_cop:.3f} → {updated_state.part_load_cop:.3f}")
        print(f"  - Is cycling: {updated_state.is_cycling}")
        print(f"  - Compressor running: {updated_state.compressor_running}")
        
        # Decrease load to very low
        low_load_state = cycling_model.update_part_load_operation(
            updated_state, 0.5, 15.0  # 5% load
        )
        print(f"  - Very low load: {low_load_state.building_load_kW:.1f} kW")
        print(f"  - Load factor: {low_load_state.load_factor:.3f}")
        print(f"  - Is cycling: {low_load_state.is_cycling}")
        print(f"  - Cycles per hour: {low_load_state.cycles_per_hour:.1f}")
        
        # Test variable speed updates
        print("✅ Testing variable-speed load changes:")
        vs_updated = cycling_model.update_part_load_operation(
            variable_speed_state, 9.0, 15.0  # 60% load
        )
        print(f"  - Load factor: {vs_updated.load_factor:.3f}")
        print(f"  - Compressor capacity: {vs_updated.compressor_capacity_percent:.1f}%")
        print(f"  - Is cycling: {vs_updated.is_cycling}")
        
        # Test cycling degradation calculation
        print("\n🔄 Testing cycling degradation calculation...")
        
        # Test different PLR values
        plr_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        cycling_params = cycling_model.cycling_degradation['single_speed']
        
        for plr in plr_values:
            degradation = cycling_model._calculate_cycling_degradation(plr, cycling_params)
            print(f"✅ PLR {plr:.2f}: degradation factor = {degradation:.3f}")
        
        # Test cycling frequency calculation
        print("\n🔄 Testing cycling frequency calculation...")
        
        control_params = cycling_model.control_parameters['residential']
        
        for plr in [0.05, 0.1, 0.2, 0.3]:
            frequency = cycling_model._calculate_cycling_frequency(
                plr, control_params, cycling_params
            )
            print(f"✅ PLR {plr:.2f}: cycling frequency = {frequency:.1f} cycles/hour")
        
        # Test daily load profile generation
        print("\n🔄 Testing daily load profile generation...")
        
        # Test different building types
        building_types = ['residential', 'commercial', 'industrial']
        thermal_masses = ['light', 'medium', 'heavy']
        
        for building_type in building_types:
            for thermal_mass in thermal_masses:
                profile = cycling_model._generate_daily_load_profile(building_type, thermal_mass)
                print(f"✅ {building_type} + {thermal_mass}: profile range = {profile.min():.2f} - {profile.max():.2f}")
        
        # Test daily cycling pattern simulation
        print("\n🔄 Testing daily cycling pattern simulation...")
        
        # Simulate single-speed system
        daily_states_single = cycling_model.simulate_daily_cycling_pattern(
            system_id="daily_single_001",
            design_capacity_kW=8.0,
            modulation_type="single_speed",
            building_type="residential",
            thermal_mass="medium"
        )
        print(f"✅ Single-speed daily simulation: {len(daily_states_single)} time steps")
        print(f"✅ First state load factor: {daily_states_single[0].load_factor:.3f}")
        print(f"✅ Peak state load factor: {max(s.load_factor for s in daily_states_single):.3f}")
        print(f"✅ Min state load factor: {min(s.load_factor for s in daily_states_single):.3f}")
        
        # Simulate variable-speed system
        daily_states_variable = cycling_model.simulate_daily_cycling_pattern(
            system_id="daily_variable_001",
            design_capacity_kW=8.0,
            modulation_type="variable_speed",
            building_type="residential",
            thermal_mass="medium"
        )
        print(f"✅ Variable-speed daily simulation: {len(daily_states_variable)} time steps")
        
        # Test seasonal efficiency calculation
        print("\n🔄 Testing seasonal efficiency calculation...")
        
        # Calculate efficiency for single-speed system
        single_efficiency = cycling_model.calculate_seasonal_efficiency(daily_states_single)
        print(f"✅ Single-speed seasonal metrics:")
        print(f"  - Seasonal COP: {single_efficiency['seasonal_cop']:.3f}")
        print(f"  - Runtime fraction: {single_efficiency['runtime_fraction']:.3f}")
        print(f"  - Cycles per hour: {single_efficiency['cycles_per_hour']:.1f}")
        print(f"  - Part-load efficiency: {single_efficiency['part_load_efficiency']:.3f}")
        print(f"  - Total energy input: {single_efficiency['total_energy_input_kWh']:.1f} kWh")
        print(f"  - Total energy output: {single_efficiency['total_energy_output_kWh']:.1f} kWh")
        print(f"  - Total runtime: {single_efficiency['total_runtime_hours']:.1f} hours")
        print(f"  - Total cycles: {single_efficiency['total_cycles']:.0f}")
        
        # Calculate efficiency for variable-speed system
        variable_efficiency = cycling_model.calculate_seasonal_efficiency(daily_states_variable)
        print(f"✅ Variable-speed seasonal metrics:")
        print(f"  - Seasonal COP: {variable_efficiency['seasonal_cop']:.3f}")
        print(f"  - Runtime fraction: {variable_efficiency['runtime_fraction']:.3f}")
        print(f"  - Cycles per hour: {variable_efficiency['cycles_per_hour']:.1f}")
        print(f"  - Part-load efficiency: {variable_efficiency['part_load_efficiency']:.3f}")
        
        # Test control strategy optimization
        print("\n🔄 Testing control strategy optimization...")
        
        # Create test load profile
        test_load_profile = np.array([
            0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6,
            0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05
        ])
        
        available_strategies = ['single_speed', 'two_speed', 'variable_speed', 'hot_gas_bypass']
        
        optimization_results = cycling_model.optimize_control_strategy(
            design_capacity_kW=10.0,
            building_load_profile=test_load_profile,
            available_strategies=available_strategies
        )
        
        print(f"✅ Control strategy optimization:")
        print(f"  - Best strategy: {optimization_results['best_strategy']}")
        print(f"  - Optimization criteria: {optimization_results['optimization_criteria']}")
        print(f"✅ Strategy comparison:")
        for strategy, metrics in optimization_results['strategy_comparison'].items():
            print(f"  - {strategy}: COP = {metrics['seasonal_cop']:.3f}, Runtime = {metrics['runtime_fraction']:.3f}")
        
        # Test data saving
        print("\n🔄 Testing data saving...")
        
        # Save sample of daily states
        sample_states = daily_states_single[:24]  # First 24 time steps (6 hours)
        cycling_model.save_part_load_data(sample_states, "test_part_load_cycling_data.json")
        print("✅ Part-load cycling data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test PartLoadState
        test_part_load_state = PartLoadState(
            system_id="test_dataclass",
            timestamp=datetime.now().isoformat(),
            building_load_kW=5.5,
            design_load_kW=10.0,
            load_factor=0.55,
            compressor_capacity_percent=55.0,
            modulation_type="variable_speed",
            modulation_level=0.55,
            is_cycling=False,
            cycle_on_time_minutes=0.0,
            cycle_off_time_minutes=0.0,
            cycles_per_hour=0.0,
            part_load_cop=3.2,
            part_load_capacity_kW=5.5,
            part_load_power_kW=1.72,
            part_load_ratio=0.55,
            energy_input_ratio=0.31,
            cycling_degradation_factor=1.0,
            thermostat_deadband_K=1.0,
            minimum_on_time_minutes=3.0,
            minimum_off_time_minutes=3.0,
            compressor_running=True,
            defrost_active=False,
            startup_transient=False
        )
        print(f"✅ PartLoadState created: {test_part_load_state.system_id}")
        print(f"✅ Load factor: {test_part_load_state.load_factor:.3f}")
        print(f"✅ Part-load COP: {test_part_load_state.part_load_cop:.2f}")
        
        # Test CyclingParameters
        test_cycling_params = CyclingParameters(
            system_type="air_source_heat_pump",
            capacity_control_type="variable_speed",
            cycling_characteristics={'degradation_coefficient': 0.05},
            part_load_curves={'cop_curve': lambda x: 3.0 + 0.5*x}
        )
        print(f"✅ CyclingParameters created: {test_cycling_params.system_type}")
        print(f"✅ Control type: {test_cycling_params.capacity_control_type}")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Zero load
        zero_load_state = cycling_model.initialize_part_load_state(
            "zero_load", "residential", 10.0, 0.0, "residential", "single_speed"
        )
        print(f"✅ Zero load: load factor = {zero_load_state.load_factor:.3f}")
        print(f"✅ Zero load: is cycling = {zero_load_state.is_cycling}")
        
        # Overload condition
        overload_state = cycling_model.initialize_part_load_state(
            "overload", "residential", 10.0, 12.0, "residential", "variable_speed"
        )
        print(f"✅ Overload: load factor = {overload_state.load_factor:.3f}")
        print(f"✅ Overload: compressor capacity = {overload_state.compressor_capacity_percent:.1f}%")
        
        # Very small capacity
        small_capacity_state = cycling_model.initialize_part_load_state(
            "small", "residential", 0.5, 0.1, "residential", "single_speed"
        )
        print(f"✅ Small capacity: load factor = {small_capacity_state.load_factor:.3f}")
        
        print("\n🎉 ALL PART-LOAD CYCLING TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in part-load cycling models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_part_load_cycling_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)