#!/usr/bin/env python3
"""
Comprehensive test script for Fouling and Aging Models
Tests all major fouling and aging functionality with proper error handling
"""

import sys
sys.path.append('src')

def test_fouling_aging_models():
    """Test all fouling and aging model functionality"""
    print("🧽 Testing Fouling and Aging Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from fouling_aging_models import (
            HeatExchangerFoulingModel, FoulingState, AgingState
        )
        import numpy as np
        import pandas as pd
        print("✅ Import successful")
        
        # Test HeatExchangerFoulingModel initialization
        print("\n🔄 Testing HeatExchangerFoulingModel initialization...")
        fouling_model = HeatExchangerFoulingModel()
        print("✅ HeatExchangerFoulingModel initialized successfully")
        
        # Test fouling parameters
        print("\n🔄 Testing fouling parameters...")
        fouling_params = fouling_model.fouling_parameters
        print(f"✅ Loaded fouling parameters for {len(fouling_params)} component types:")
        for comp_type, params in fouling_params.items():
            print(f"  - {comp_type}: asymptotic fouling = {params['asymptotic_fouling']:.6f} m²K/W")
        
        # Test aging parameters
        print("\n🔄 Testing aging parameters...")
        aging_params = fouling_model.aging_parameters
        print(f"✅ Loaded aging parameters for {len(aging_params)} component types:")
        for comp_type, params in aging_params.items():
            print(f"  - {comp_type}: design life = {params['design_life']} years")
        
        # Test fouling state initialization
        print("\n🔄 Testing fouling state initialization...")
        
        # Test evaporator fouling state
        evap_fouling = fouling_model.initialize_fouling_state(
            component_id="test_evap_001",
            component_type="finned_tube_evaporator",
            age_hours=0
        )
        print(f"✅ Created evaporator fouling state: {type(evap_fouling)}")
        print(f"✅ Initial fouling resistance: {evap_fouling.fouling_resistance:.6f} m²K/W")
        print(f"✅ Initial fouling thickness: {evap_fouling.fouling_thickness:.3f} mm")
        print(f"✅ Fouling type: {evap_fouling.fouling_type}")
        
        # Test condenser fouling state
        cond_fouling = fouling_model.initialize_fouling_state(
            component_id="test_cond_001",
            component_type="plate_condenser",
            age_hours=0
        )
        print(f"✅ Created condenser fouling state: {type(cond_fouling)}")
        print(f"✅ Initial fouling resistance: {cond_fouling.fouling_resistance:.6f} m²K/W")
        
        # Test aging state initialization
        print("\n🔄 Testing aging state initialization...")
        
        # Test evaporator aging state
        evap_aging = fouling_model.initialize_aging_state(
            component_id="test_evap_001",
            component_type="finned_tube_evaporator",
            age_hours=8760  # 1 year
        )
        print(f"✅ Created evaporator aging state: {type(evap_aging)}")
        print(f"✅ Age: {evap_aging.age_hours} hours")
        print(f"✅ Corrosion depth: {evap_aging.corrosion_depth:.3f} mm")
        print(f"✅ Fin degradation: {evap_aging.fin_degradation:.1f}%")
        print(f"✅ Material properties: {len(evap_aging.material_properties)} items")
        
        # Test condenser aging state
        cond_aging = fouling_model.initialize_aging_state(
            component_id="test_cond_001",
            component_type="plate_condenser",
            age_hours=17520  # 2 years
        )
        print(f"✅ Created condenser aging state: {type(cond_aging)}")
        print(f"✅ Gasket condition: {cond_aging.gasket_condition:.3f}")
        
        # Test fouling state updates
        print("\n🔄 Testing fouling state updates...")
        
        # Define operating conditions
        operating_conditions = {
            'temperature': 278.15,  # 5°C
            'humidity': 0.7,
            'season': 'summer',
            'defrost_cycles': 50,
            'thermal_cycles': 1000,
            'contamination': 0.1
        }
        
        # Update evaporator fouling after 1000 hours
        evap_fouling_updated = fouling_model.update_fouling_state(
            evap_fouling, 1000, operating_conditions, "finned_tube_evaporator"
        )
        print(f"✅ Updated evaporator fouling after 1000 hours")
        print(f"✅ New fouling resistance: {evap_fouling_updated.fouling_resistance:.6f} m²K/W")
        print(f"✅ Effectiveness reduction: {evap_fouling_updated.effectiveness_reduction:.2f}%")
        print(f"✅ Pressure drop increase: {evap_fouling_updated.pressure_drop_increase:.2f}%")
        print(f"✅ Fouling age: {evap_fouling_updated.fouling_age:.0f} hours")
        
        # Update condenser fouling
        cond_fouling_updated = fouling_model.update_fouling_state(
            cond_fouling, 2000, operating_conditions, "plate_condenser"
        )
        print(f"✅ Updated condenser fouling after 2000 hours")
        print(f"✅ New fouling resistance: {cond_fouling_updated.fouling_resistance:.6f} m²K/W")
        
        # Test aging state updates
        print("\n🔄 Testing aging state updates...")
        
        # Update evaporator aging
        evap_aging_updated = fouling_model.update_aging_state(
            evap_aging, 8760, operating_conditions, "finned_tube_evaporator"
        )
        print(f"✅ Updated evaporator aging after additional 8760 hours")
        print(f"✅ Total age: {evap_aging_updated.age_hours} hours")
        print(f"✅ Corrosion depth: {evap_aging_updated.corrosion_depth:.3f} mm")
        print(f"✅ Fin degradation: {evap_aging_updated.fin_degradation:.1f}%")
        print(f"✅ Tube degradation: {evap_aging_updated.tube_degradation:.1f}%")
        
        # Update condenser aging
        cond_aging_updated = fouling_model.update_aging_state(
            cond_aging, 8760, operating_conditions, "plate_condenser"
        )
        print(f"✅ Updated condenser aging")
        print(f"✅ Gasket condition: {cond_aging_updated.gasket_condition:.3f}")
        
        # Test cleaning event simulation
        print("\n🔄 Testing cleaning event simulation...")
        
        # Test different cleaning types
        cleaning_types = ['water_rinse', 'chemical', 'mechanical', 'ultrasonic']
        
        for cleaning_type in cleaning_types:
            # Create a copy for testing
            test_fouling = FoulingState(
                component_id="test_cleaning",
                fouling_resistance=0.001,  # Moderate fouling
                fouling_thickness=0.1,
                fouling_type='mixed',
                fouling_age=2000,
                cleaning_history=[],
                effectiveness_reduction=10.0,
                pressure_drop_increase=15.0
            )
            
            # Simulate cleaning
            cleaned_fouling = fouling_model.simulate_cleaning_event(test_fouling, cleaning_type)
            print(f"✅ {cleaning_type} cleaning: fouling reduced from 0.001000 to {cleaned_fouling.fouling_resistance:.6f} m²K/W")
            print(f"  - Cleaning history entries: {len(cleaned_fouling.cleaning_history)}")
            print(f"  - Fouling age reset to: {cleaned_fouling.fouling_age} hours")
        
        # Test fouling schedule generation
        print("\n🔄 Testing fouling schedule generation...")
        
        # Generate schedule for evaporator
        evap_schedule = fouling_model.generate_fouling_schedule(
            component_id="test_evap_001",
            component_type="finned_tube_evaporator",
            simulation_hours=17520  # 2 years
        )
        print(f"✅ Generated evaporator cleaning schedule: {len(evap_schedule)} events")
        if evap_schedule:
            print(f"✅ First cleaning at: {evap_schedule[0]['time_hours']} hours")
            print(f"✅ Cleaning type: {evap_schedule[0]['cleaning_type']}")
        
        # Generate schedule for condenser
        cond_schedule = fouling_model.generate_fouling_schedule(
            component_id="test_cond_001",
            component_type="plate_condenser",
            simulation_hours=17520  # 2 years
        )
        print(f"✅ Generated condenser cleaning schedule: {len(cond_schedule)} events")
        
        # Test private calculation methods
        print("\n🔄 Testing calculation methods...")
        
        # Test temperature fouling factor
        temp_factor_evap = fouling_model._calculate_temperature_fouling_factor(
            temperature=273.15, component_type="finned_tube_evaporator"
        )
        print(f"✅ Temperature fouling factor (evaporator, 0°C): {temp_factor_evap:.3f}")
        
        temp_factor_cond = fouling_model._calculate_temperature_fouling_factor(
            temperature=323.15, component_type="plate_condenser"
        )
        print(f"✅ Temperature fouling factor (condenser, 50°C): {temp_factor_cond:.3f}")
        
        # Test humidity fouling factor
        humidity_factor = fouling_model._calculate_humidity_fouling_factor(
            humidity=0.8, component_type="finned_tube_evaporator"
        )
        print(f"✅ Humidity fouling factor (80% RH): {humidity_factor:.3f}")
        
        # Test seasonal fouling factor
        seasonal_factor = fouling_model._calculate_seasonal_fouling_factor(
            season='summer', params={'seasonal_factor': 1.5}
        )
        print(f"✅ Seasonal fouling factor (summer): {seasonal_factor:.3f}")
        
        # Test temperature aging factor
        aging_factor = fouling_model._calculate_temperature_aging_factor(temperature=333.15)
        print(f"✅ Temperature aging factor (60°C): {aging_factor:.3f}")
        
        # Test corrosion factor
        corrosion_factor = fouling_model._calculate_corrosion_factor(
            humidity=0.9, contamination=0.2
        )
        print(f"✅ Corrosion factor (90% RH, 20% contamination): {corrosion_factor:.3f}")
        
        # Test effectiveness reduction calculation
        effectiveness_reduction = fouling_model._calculate_effectiveness_reduction(
            fouling_resistance=0.001, max_fouling=0.002
        )
        print(f"✅ Effectiveness reduction (50% fouling): {effectiveness_reduction:.1f}%")
        
        # Test pressure drop increase calculation
        pressure_drop_increase = fouling_model._calculate_pressure_drop_increase(
            fouling_thickness=0.1, component_type="finned_tube_evaporator"
        )
        print(f"✅ Pressure drop increase (0.1mm fouling): {pressure_drop_increase:.1f}%")
        
        # Test data saving
        print("\n🔄 Testing data saving...")
        
        fouling_states = [evap_fouling_updated, cond_fouling_updated]
        aging_states = [evap_aging_updated, cond_aging_updated]
        
        fouling_model.save_fouling_data(
            fouling_states, aging_states, "test_fouling_aging_data.json"
        )
        print("✅ Fouling and aging data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test FoulingState
        test_fouling_state = FoulingState(
            component_id="test_fouling",
            fouling_resistance=0.0005,
            fouling_thickness=0.05,
            fouling_type='biological',
            fouling_age=1500,
            cleaning_history=[{'type': 'chemical', 'effectiveness': 0.85}],
            effectiveness_reduction=5.0,
            pressure_drop_increase=8.0
        )
        print(f"✅ FoulingState created: {test_fouling_state.component_id}")
        print(f"✅ Fouling resistance: {test_fouling_state.fouling_resistance:.6f} m²K/W")
        
        # Test AgingState
        test_aging_state = AgingState(
            component_id="test_aging",
            age_hours=26280,  # 3 years
            corrosion_depth=0.03,
            scaling_thickness=0.01,
            fin_degradation=6.0,
            tube_degradation=1.5,
            gasket_condition=0.85,
            material_properties={'thermal_conductivity_degradation': 0.03}
        )
        print(f"✅ AgingState created: {test_aging_state.component_id}")
        print(f"✅ Age: {test_aging_state.age_hours} hours")
        print(f"✅ Corrosion depth: {test_aging_state.corrosion_depth:.3f} mm")
        
        print("\n🎉 ALL FOULING AND AGING TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in fouling and aging models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fouling_aging_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)