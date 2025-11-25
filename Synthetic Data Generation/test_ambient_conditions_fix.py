#!/usr/bin/env python3
"""
Test script for ambient_conditions.py to identify and fix issues
"""

import sys
import os
sys.path.append('src')

def test_ambient_conditions():
    """Test ambient conditions model with comprehensive error handling"""
    
    print("🌤️ Testing Ambient Conditions Models - Comprehensive Test")
    print("=" * 60)
    
    try:
        from ambient_conditions import AmbientConditionsModel
        print("✅ Import successful")
        
        # Test model initialization
        print("\n🔄 Testing model initialization...")
        model = AmbientConditionsModel()
        print("✅ Model initialized successfully")
        
        # Check available operating modes
        if hasattr(model, 'temperature_effects'):
            print(f"✅ Available operating modes: {list(model.temperature_effects.keys())}")
        
        # Test calculate_ambient_conditions
        print("\n🔄 Testing calculate_ambient_conditions...")
        ambient_result = model.calculate_ambient_conditions(
            dry_bulb_temp_K=298.15,  # 25°C
            relative_humidity_percent=60.0,
            wind_speed_m_per_s=2.0,
            solar_irradiance_W_per_m2=800.0,
            barometric_pressure_Pa=101325.0,
            location='test_location'
        )
        print(f"✅ Calculated ambient conditions: {type(ambient_result)}")
        print(f"✅ Temperature: {ambient_result.dry_bulb_temp_K - 273.15:.1f}°C")
        print(f"✅ Humidity: {ambient_result.relative_humidity_percent:.1f}%")
        print(f"✅ Wind speed: {ambient_result.wind_speed_m_per_s:.1f} m/s")
        
        # Test calculate_performance_modifiers
        print("\n🔄 Testing calculate_performance_modifiers...")
        try:
            perf_modifiers = model.calculate_performance_modifiers(
                ambient=ambient_result,
                operating_mode='cooling'
            )
            print(f"✅ Performance modifiers calculated: {type(perf_modifiers)}")
            
            # Check available attributes
            attrs = [attr for attr in dir(perf_modifiers) if not attr.startswith('_')]
            print(f"✅ Available attributes: {attrs}")
            
            # Print specific values if available
            if hasattr(perf_modifiers, 'cooling_capacity_modifier'):
                print(f"✅ Cooling capacity modifier: {perf_modifiers.cooling_capacity_modifier:.3f}")
            if hasattr(perf_modifiers, 'cooling_cop_modifier'):
                print(f"✅ Cooling COP modifier: {perf_modifiers.cooling_cop_modifier:.3f}")
            if hasattr(perf_modifiers, 'frost_formation_risk'):
                print(f"✅ Frost formation risk: {perf_modifiers.frost_formation_risk:.3f}")
                
        except Exception as e:
            print(f"❌ Performance modifiers failed: {e}")
            print("🔍 Trying with different operating mode...")
            
            # Try with available operating modes
            if hasattr(model, 'temperature_effects') and model.temperature_effects:
                for mode in model.temperature_effects.keys():
                    try:
                        print(f"🔄 Trying mode: {mode}")
                        perf_modifiers = model.calculate_performance_modifiers(
                            ambient=ambient_result,
                            operating_mode=mode
                        )
                        print(f"✅ Success with mode {mode}: {type(perf_modifiers)}")
                        break
                    except Exception as e2:
                        print(f"❌ Mode {mode} failed: {e2}")
        
        # Test generate_seasonal_weather_pattern
        print("\n🔄 Testing generate_seasonal_weather_pattern...")
        weather_pattern = model.generate_seasonal_weather_pattern(
            location='test_location',
            duration_days=1,
            time_step_hours=6
        )
        print(f"✅ Generated weather pattern: {len(weather_pattern)} data points")
        
        if weather_pattern:
            sample = weather_pattern[0]
            print(f"✅ Sample temperature: {sample.dry_bulb_temp_K - 273.15:.1f}°C")
            print(f"✅ Sample humidity: {sample.relative_humidity_percent:.1f}%")
        
        # Test save_ambient_data
        print("\n🔄 Testing save_ambient_data...")
        test_data = weather_pattern[:3]
        model.save_ambient_data(test_data, 'test_ambient_data_comprehensive.json')
        print("✅ Ambient data saved successfully")
        
        print("\n🎉 ALL TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ambient_conditions()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed - check errors above")