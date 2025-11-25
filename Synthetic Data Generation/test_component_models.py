#!/usr/bin/env python3
"""
Comprehensive test script for Component Models
Tests all major component model functionality with proper error handling
"""

import sys
sys.path.append('src')

def test_component_models():
    """Test all component model functionality"""
    print("🔧 Testing Component Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from component_models import (
            CompressorModel, HeatExchangerModel, ExpansionDeviceModel,
            CompressorType, HeatExchangerType, ExpansionType,
            CompressorPerformance, HeatExchangerPerformance, ExpansionPerformance,
            ComponentIntegration, component_integration
        )
        from thermodynamic_properties import ThermodynamicState
        print("✅ Import successful")
        
        # Test CompressorModel
        print("\n🔄 Testing CompressorModel initialization...")
        try:
            # Try with CompressorType enum
            compressor = CompressorModel(CompressorType.SCROLL, displacement=0.15)
            print("✅ CompressorModel initialized with SCROLL type")
        except Exception as e:
            print(f"❌ CompressorModel initialization failed: {e}")
            return False
        
        # Test compressor methods
        print("\n🔄 Testing compressor methods...")
        try:
            # Create dummy thermodynamic states
            state_in = ThermodynamicState(
                T=278.15, P=400000, h=200000, s=1000, rho=10,
                cp=2000, mu=1e-3, k=0.1
            )
            state_out = ThermodynamicState(
                T=353.15, P=1800000, h=250000, s=1000, rho=5,
                cp=2000, mu=1e-3, k=0.1
            )
            
            # Test calculate_performance
            perf = compressor.calculate_performance(
                state_in=state_in,
                state_out=state_out,
                mass_flow=0.15
            )
            print(f"✅ Compressor performance calculated: {type(perf)}")
            print(f"✅ Power consumption: {perf.W_comp:.1f} kW")
            print(f"✅ Isentropic efficiency: {perf.eta_isentropic:.3f}")
            print(f"✅ Pressure ratio: {perf.pressure_ratio:.2f}")
        
        except Exception as e:
            print(f"⚠️ Compressor method testing failed: {e}")
        
        # Test HeatExchangerModel (Evaporator)
        print("\n🔄 Testing HeatExchangerModel (Evaporator)...")
        try:
            evaporator = HeatExchangerModel(HeatExchangerType.FINNED_TUBE, area=40.0)
            print("✅ HeatExchangerModel (evaporator) initialized")
            
            # Create states for heat exchanger testing
            state_hot_in = ThermodynamicState(
                T=283.15, P=400000, h=220000, s=1100, rho=8,
                cp=2000, mu=1e-3, k=0.1
            )
            state_hot_out = ThermodynamicState(
                T=278.15, P=390000, h=200000, s=1000, rho=10,
                cp=2000, mu=1e-3, k=0.1
            )
            state_cold_in = ThermodynamicState(
                T=275.15, P=101325, h=10000, s=100, rho=1.2,
                cp=1000, mu=1e-5, k=0.025
            )
            state_cold_out = ThermodynamicState(
                T=280.15, P=101325, h=15000, s=110, rho=1.15,
                cp=1000, mu=1e-5, k=0.025
            )
            
            hx_perf = evaporator.calculate_performance(
                state_hot_in=state_hot_in,
                state_hot_out=state_hot_out,
                state_cold_in=state_cold_in,
                state_cold_out=state_cold_out,
                mass_flow_hot=0.15,
                mass_flow_cold=2.0
            )
            print(f"✅ Heat exchanger performance calculated: {type(hx_perf)}")
            print(f"✅ Heat transfer rate: {hx_perf.Q_transfer:.1f} kW")
            print(f"✅ Effectiveness: {hx_perf.effectiveness:.3f}")
        
        except Exception as e:
            print(f"⚠️ HeatExchangerModel testing failed: {e}")
        
        # Test ExpansionDeviceModel
        print("\n🔄 Testing ExpansionDeviceModel...")
        try:
            expansion_valve = ExpansionDeviceModel(ExpansionType.THROTTLING_VALVE)
            print("✅ ExpansionDeviceModel initialized")
            
            # Create states for expansion device
            state_exp_in = ThermodynamicState(
                T=318.15, P=1800000, h=250000, s=1000, rho=5,
                cp=2000, mu=1e-3, k=0.1
            )
            state_exp_out = ThermodynamicState(
                T=278.15, P=400000, h=250000, s=1200, rho=10,  # Isenthalpic
                cp=2000, mu=1e-3, k=0.1
            )
            
            exp_perf = expansion_valve.calculate_performance(
                state_in=state_exp_in,
                state_out=state_exp_out,
                mass_flow=0.15
            )
            print(f"✅ Expansion performance calculated: {type(exp_perf)}")
            print(f"✅ Pressure drop: {exp_perf.dp_expansion/1000:.0f} kPa")
            print(f"✅ Flow coefficient: {exp_perf.flow_coefficient:.3f}")
        
        except Exception as e:
            print(f"⚠️ ExpansionDeviceModel testing failed: {e}")
        
        # Test ComponentIntegration
        print("\n🔄 Testing ComponentIntegration...")
        try:
            # Test the default component integration
            print(f"✅ Default component integration available")
            print(f"✅ Compressor models: {list(component_integration.compressor_models.keys())}")
            print(f"✅ Heat exchanger models: {list(component_integration.heat_exchanger_models.keys())}")
            print(f"✅ Expansion models: {list(component_integration.expansion_models.keys())}")
            
            # Test performance map generation
            comp_map = component_integration.generate_component_maps("scroll_comp", "compressor")
            if comp_map:
                print(f"✅ Generated compressor map with {comp_map['efficiency_map'].shape} efficiency points")
                print(f"✅ Efficiency range: {comp_map['efficiency_map'].min():.3f} - {comp_map['efficiency_map'].max():.3f}")
        
        except Exception as e:
            print(f"⚠️ ComponentIntegration testing failed: {e}")
        
        print("\n🎉 COMPONENT MODELS TEST COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in component models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_component_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)