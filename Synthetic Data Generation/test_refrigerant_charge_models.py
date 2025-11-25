#!/usr/bin/env python3
"""
Comprehensive test script for Refrigerant Charge Models
Tests all major refrigerant charge variation functionality
"""

import sys
sys.path.append('src')

def test_refrigerant_charge_models():
    """Test all refrigerant charge model functionality"""
    print("🧪 Testing Refrigerant Charge Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from refrigerant_charge_models import (
            RefrigerantChargeModel, RefrigerantChargeState, ChargeVariationParameters
        )
        import numpy as np
        import pandas as pd
        print("✅ Import successful")
        
        # Test RefrigerantChargeModel initialization
        print("\n🔄 Testing RefrigerantChargeModel initialization...")
        charge_model = RefrigerantChargeModel()
        print("✅ RefrigerantChargeModel initialized successfully")
        
        # Test charge sensitivity parameters
        print("\n🔄 Testing charge sensitivity parameters...")
        charge_sensitivity = charge_model.charge_sensitivity
        print(f"✅ Loaded charge sensitivity for {len(charge_sensitivity)} refrigerants:")
        for refrigerant, params in charge_sensitivity.items():
            print(f"  - {refrigerant}: subcooling = {params['subcooling_per_percent']:.1f} K/%")
            print(f"    * superheat = {params['superheat_per_percent']:.1f} K/%")
            print(f"    * capacity = {params['capacity_per_percent']:.1f} %/%")
            print(f"    * COP = {params['cop_per_percent']:.1f} %/%")
            print(f"    * optimal range = {params['optimal_charge_range'][0]}-{params['optimal_charge_range'][1]}%")
        
        # Test leak models
        print("\n🔄 Testing leak models...")
        leak_models = charge_model.leak_models
        print(f"✅ Loaded leak models for {len(leak_models)} system conditions:")
        for condition, params in leak_models.items():
            print(f"  - {condition}: base rate = {params['base_leak_rate']:.1f}%/year")
        
        # Test seasonal effects
        print("\n🔄 Testing seasonal effects...")
        seasonal_effects = charge_model.seasonal_effects
        print(f"✅ Loaded seasonal effects for {len(seasonal_effects)} seasons:")
        for season, effects in seasonal_effects.items():
            print(f"  - {season}: migration = {effects['charge_migration_factor']:.3f}")
        
        # Test performance curves
        print("\n🔄 Testing performance curves...")
        performance_curves = charge_model.performance_curves
        print(f"✅ Loaded performance curves for {len(performance_curves)} refrigerants:")
        for refrigerant, curves in performance_curves.items():
            print(f"  - {refrigerant}: {len(curves)} curve types")
        
        # Test charge state initialization - Normal charge
        print("\n🔄 Testing normal charge state initialization...")
        normal_charge = charge_model.initialize_charge_state(
            system_id="test_system_001",
            refrigerant_type="R410A",
            nominal_charge_kg=3.5,
            current_charge_percentage=100.0,
            ambient_temp_K=298.15
        )
        print(f"✅ Created normal charge state: {type(normal_charge)}")
        print(f"✅ System ID: {normal_charge.system_id}")
        print(f"✅ Refrigerant: {normal_charge.refrigerant_type}")
        print(f"✅ Nominal charge: {normal_charge.nominal_charge_kg:.1f} kg")
        print(f"✅ Current charge: {normal_charge.current_charge_kg:.1f} kg")
        print(f"✅ Charge level: {normal_charge.charge_level_percentage:.1f}%")
        print(f"✅ Subcooling: {normal_charge.subcooling_K:.1f} K")
        print(f"✅ Superheat: {normal_charge.superheat_K:.1f} K")
        print(f"✅ Pressure ratio: {normal_charge.pressure_ratio:.2f}")
        print(f"✅ Cooling capacity factor: {normal_charge.cooling_capacity_factor:.3f}")
        print(f"✅ COP cooling factor: {normal_charge.cop_cooling_factor:.3f}")
        
        # Test charge state initialization - Low charge
        print("\n🔄 Testing low charge state initialization...")
        low_charge = charge_model.initialize_charge_state(
            system_id="test_system_002",
            refrigerant_type="R32",
            nominal_charge_kg=2.8,
            current_charge_percentage=75.0,
            ambient_temp_K=308.15
        )
        print(f"✅ Created low charge state: {type(low_charge)}")
        print(f"✅ Refrigerant: {low_charge.refrigerant_type}")
        print(f"✅ Charge level: {low_charge.charge_level_percentage:.1f}%")
        print(f"✅ Subcooling: {low_charge.subcooling_K:.1f} K")
        print(f"✅ Superheat: {low_charge.superheat_K:.1f} K")
        print(f"✅ Evaporator flooding risk: {low_charge.evaporator_flooding_risk:.3f}")
        print(f"✅ Condenser starving risk: {low_charge.condenser_starving_risk:.3f}")
        
        # Test charge state initialization - High charge
        print("\n🔄 Testing high charge state initialization...")
        high_charge = charge_model.initialize_charge_state(
            system_id="test_system_003",
            refrigerant_type="R134a",
            nominal_charge_kg=4.2,
            current_charge_percentage=125.0,
            ambient_temp_K=288.15
        )
        print(f"✅ Created high charge state: {type(high_charge)}")
        print(f"✅ Charge level: {high_charge.charge_level_percentage:.1f}%")
        print(f"✅ Subcooling: {high_charge.subcooling_K:.1f} K")
        print(f"✅ Superheat: {high_charge.superheat_K:.1f} K")
        print(f"✅ Power consumption factor: {high_charge.power_consumption_factor:.3f}")
        
        # Test charge loss simulation
        print("\n🔄 Testing charge loss simulation...")
        
        # Define operating conditions
        operating_conditions = {
            'connection_count': 15,
            'vibration_level': 4.5,  # mm/s RMS
            'average_temperature': 308.15,  # 35°C
            'ambient_temperature': 308.15
        }
        
        # Simulate charge loss for new system
        new_system_loss = charge_model.simulate_charge_loss(
            normal_charge, 1.5, 'good', operating_conditions, 365  # 1 year
        )
        print(f"✅ New system after 1 year:")
        print(f"  - Original charge: {normal_charge.charge_level_percentage:.1f}%")
        print(f"  - New charge: {new_system_loss.charge_level_percentage:.1f}%")
        print(f"  - Leak rate: {new_system_loss.leak_rate_kg_per_year:.3f} kg/year")
        
        # Simulate charge loss for aged system
        aged_system_loss = charge_model.simulate_charge_loss(
            normal_charge, 8.0, 'good', operating_conditions, 365  # 1 year
        )
        print(f"✅ Aged system after 1 year:")
        print(f"  - New charge: {aged_system_loss.charge_level_percentage:.1f}%")
        print(f"  - Leak rate: {aged_system_loss.leak_rate_kg_per_year:.3f} kg/year")
        
        # Simulate charge loss for poor maintenance
        poor_maintenance_loss = charge_model.simulate_charge_loss(
            normal_charge, 5.0, 'poor', operating_conditions, 365  # 1 year
        )
        print(f"✅ Poor maintenance system after 1 year:")
        print(f"  - New charge: {poor_maintenance_loss.charge_level_percentage:.1f}%")
        print(f"  - Leak rate: {poor_maintenance_loss.leak_rate_kg_per_year:.3f} kg/year")
        
        # Test seasonal effects simulation
        print("\n🔄 Testing seasonal effects simulation...")
        
        # Summer effects
        summer_state = charge_model.simulate_seasonal_effects(
            normal_charge, 'summer', 313.15  # 40°C
        )
        print(f"✅ Summer effects:")
        print(f"  - Original charge: {normal_charge.charge_level_percentage:.1f}%")
        print(f"  - Summer apparent charge: {summer_state.charge_level_percentage:.1f}%")
        print(f"  - Migration: {summer_state.charge_migration_seasonal_kg:.3f} kg")
        
        # Winter effects
        winter_state = charge_model.simulate_seasonal_effects(
            normal_charge, 'winter', 273.15  # 0°C
        )
        print(f"✅ Winter effects:")
        print(f"  - Winter apparent charge: {winter_state.charge_level_percentage:.1f}%")
        print(f"  - Migration: {winter_state.charge_migration_seasonal_kg:.3f} kg")
        
        # Spring/Fall effects
        spring_state = charge_model.simulate_seasonal_effects(
            normal_charge, 'spring_fall', 298.15  # 25°C
        )
        print(f"✅ Spring/Fall effects:")
        print(f"  - Apparent charge: {spring_state.charge_level_percentage:.1f}%")
        print(f"  - Migration: {spring_state.charge_migration_seasonal_kg:.3f} kg")
        
        # Test charge adjustment simulation
        print("\n🔄 Testing charge adjustment simulation...")
        
        # Adjust low charge system to optimal
        adjusted_charge = charge_model.simulate_charge_adjustment(
            low_charge, 100.0  # Target 100%
        )
        print(f"✅ Charge adjustment:")
        print(f"  - Before: {low_charge.charge_level_percentage:.1f}%")
        print(f"  - After: {adjusted_charge.charge_level_percentage:.1f}%")
        print(f"  - Adjustment: {adjusted_charge.last_charge_adjustment_kg:.3f} kg")
        
        # Test saturation pressure calculation
        print("\n🔄 Testing saturation pressure calculation...")
        
        # Test different refrigerants and temperatures
        refrigerants = ['R410A', 'R32', 'R134a']
        temperatures = [273.15, 298.15, 323.15]  # 0°C, 25°C, 50°C
        
        for refrigerant in refrigerants:
            for temp in temperatures:
                pressure = charge_model._calculate_saturation_pressure(refrigerant, temp)
                print(f"✅ {refrigerant} at {temp-273.15:.0f}°C: {pressure:.2f} bar")
        
        # Test COP impact calculation
        print("\n🔄 Testing COP impact calculation...")
        
        cop_cooling_normal = charge_model.calculate_charge_impact_on_cop(normal_charge, 'cooling')
        cop_heating_normal = charge_model.calculate_charge_impact_on_cop(normal_charge, 'heating')
        cop_cooling_low = charge_model.calculate_charge_impact_on_cop(low_charge, 'cooling')
        cop_heating_low = charge_model.calculate_charge_impact_on_cop(low_charge, 'heating')
        
        print(f"✅ COP impact - Normal charge (100%):")
        print(f"  - Cooling: {cop_cooling_normal:.3f}")
        print(f"  - Heating: {cop_heating_normal:.3f}")
        print(f"✅ COP impact - Low charge (75%):")
        print(f"  - Cooling: {cop_cooling_low:.3f}")
        print(f"  - Heating: {cop_heating_low:.3f}")
        
        # Test capacity impact calculation
        print("\n🔄 Testing capacity impact calculation...")
        
        capacity_cooling_normal = charge_model.calculate_charge_impact_on_capacity(normal_charge, 'cooling')
        capacity_heating_normal = charge_model.calculate_charge_impact_on_capacity(normal_charge, 'heating')
        capacity_cooling_low = charge_model.calculate_charge_impact_on_capacity(low_charge, 'cooling')
        capacity_heating_low = charge_model.calculate_charge_impact_on_capacity(low_charge, 'heating')
        
        print(f"✅ Capacity impact - Normal charge (100%):")
        print(f"  - Cooling: {capacity_cooling_normal:.3f}")
        print(f"  - Heating: {capacity_heating_normal:.3f}")
        print(f"✅ Capacity impact - Low charge (75%):")
        print(f"  - Cooling: {capacity_cooling_low:.3f}")
        print(f"  - Heating: {capacity_heating_low:.3f}")
        
        # Test charge issue detection
        print("\n🔄 Testing charge issue detection...")
        
        # Test normal charge issues
        normal_issues = charge_model.detect_charge_issues(normal_charge)
        print(f"✅ Normal charge issues:")
        print(f"  - Low charge warning: {normal_issues['low_charge_warning']}")
        print(f"  - High charge warning: {normal_issues['high_charge_warning']}")
        print(f"  - Severity score: {normal_issues['severity_score']}")
        print(f"  - Recommended action: {normal_issues['recommended_action']}")
        
        # Test low charge issues
        low_issues = charge_model.detect_charge_issues(low_charge)
        print(f"✅ Low charge issues:")
        print(f"  - Low charge warning: {low_issues['low_charge_warning']}")
        print(f"  - Critical low charge: {low_issues['critical_low_charge']}")
        print(f"  - Evaporator flooding risk: {low_issues['evaporator_flooding_risk']}")
        print(f"  - Severity score: {low_issues['severity_score']}")
        print(f"  - Recommended action: {low_issues['recommended_action']}")
        
        # Test high charge issues
        high_issues = charge_model.detect_charge_issues(high_charge)
        print(f"✅ High charge issues:")
        print(f"  - High charge warning: {high_issues['high_charge_warning']}")
        print(f"  - Critical high charge: {high_issues['critical_high_charge']}")
        print(f"  - Condenser starving risk: {high_issues['condenser_starving_risk']}")
        print(f"  - Excessive subcooling: {high_issues['excessive_subcooling']}")
        print(f"  - Severity score: {high_issues['severity_score']}")
        print(f"  - Recommended action: {high_issues['recommended_action']}")
        
        # Test critical charge conditions
        critical_low_charge = charge_model.initialize_charge_state(
            "critical_low", "R410A", 3.5, 65.0, 298.15
        )
        critical_low_issues = charge_model.detect_charge_issues(critical_low_charge)
        print(f"✅ Critical low charge (65%) issues:")
        print(f"  - Critical low charge: {critical_low_issues['critical_low_charge']}")
        print(f"  - Severity score: {critical_low_issues['severity_score']}")
        print(f"  - Recommended action: {critical_low_issues['recommended_action']}")
        
        # Test dataset generation
        print("\n🔄 Testing dataset generation...")
        
        system_specs = {
            'nominal_charge': 3.5
        }
        
        dataset = charge_model.generate_charge_variation_dataset(system_specs, 100)
        print(f"✅ Generated dataset: {len(dataset)} samples")
        print(f"✅ Dataset columns: {list(dataset.columns)}")
        print(f"✅ Charge level range: {dataset['charge_level_pct'].min():.1f}% - {dataset['charge_level_pct'].max():.1f}%")
        print(f"✅ COP cooling factor range: {dataset['cop_cooling_factor'].min():.3f} - {dataset['cop_cooling_factor'].max():.3f}")
        print(f"✅ Severity score range: {dataset['severity_score'].min():.0f} - {dataset['severity_score'].max():.0f}")
        print(f"✅ Refrigerant types: {dataset['refrigerant_type'].unique()}")
        
        # Test data saving
        print("\n🔄 Testing data saving...")
        
        charge_states = [normal_charge, low_charge, high_charge, new_system_loss, aged_system_loss]
        charge_model.save_charge_data(charge_states, "test_refrigerant_charge_data.json")
        print("✅ Refrigerant charge data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test RefrigerantChargeState
        test_charge_state = RefrigerantChargeState(
            system_id="test_dataclass",
            refrigerant_type="R410A",
            nominal_charge_kg=3.5,
            current_charge_kg=3.2,
            charge_level_percentage=91.4,
            subcooling_K=7.2,
            superheat_K=6.8,
            suction_pressure_bar=6.5,
            discharge_pressure_bar=24.8,
            pressure_ratio=3.8,
            cooling_capacity_factor=0.98,
            heating_capacity_factor=0.97,
            cop_cooling_factor=0.96,
            cop_heating_factor=0.95,
            power_consumption_factor=1.04,
            liquid_line_temperature_K=305.2,
            suction_line_temperature_K=281.8,
            compressor_discharge_temp_K=348.5,
            evaporator_flooding_risk=0.15,
            condenser_starving_risk=0.05,
            leak_rate_kg_per_year=0.08,
            charge_migration_seasonal_kg=0.02,
            last_charge_adjustment_kg=0.0
        )
        print(f"✅ RefrigerantChargeState created: {test_charge_state.system_id}")
        print(f"✅ Charge level: {test_charge_state.charge_level_percentage:.1f}%")
        print(f"✅ Pressure ratio: {test_charge_state.pressure_ratio:.2f}")
        
        # Test ChargeVariationParameters
        test_charge_params = ChargeVariationParameters(
            refrigerant="R410A",
            system_type="air_source_heat_pump",
            nominal_charge=3.5,
            charge_sensitivity={'subcooling_per_percent': 0.8},
            performance_curves={'cooling_capacity': lambda x: 1.0 + 0.01*x}
        )
        print(f"✅ ChargeVariationParameters created: {test_charge_params.refrigerant}")
        print(f"✅ System type: {test_charge_params.system_type}")
        print(f"✅ Nominal charge: {test_charge_params.nominal_charge} kg")
        
        print("\n🎉 ALL REFRIGERANT CHARGE TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in refrigerant charge models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_refrigerant_charge_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)