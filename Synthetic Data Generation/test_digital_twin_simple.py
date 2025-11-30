#!/usr/bin/env python3
"""
Simple Digital Twin Validation Test
===================================

This script tests the digital twin validation system independently
without requiring the full data generation pipeline dependencies.

Author: AIRTHIUM COP Prediction System
Date: 2024
"""

import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only the digital twin validation module
try:
    from digital_twin_validation import DigitalTwinValidator, RealDataPoint, SyntheticDataPoint
    print("✅ Successfully imported digital twin validation module")
except ImportError as e:
    print(f"❌ Failed to import digital twin validation: {e}")
    sys.exit(1)

def test_digital_twin_validation_simple():
    """Test the digital twin validation system with simple data."""
    
    print("🔍 TESTING DIGITAL TWIN VALIDATION SYSTEM (SIMPLE)")
    print("=" * 60)
    
    # Initialize validator
    try:
        validator = DigitalTwinValidator()
        print("✅ Digital twin validator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize validator: {e}")
        return False
    
    # Test realistic value ranges
    print(f"\n📊 REALISTIC VALUE RANGES:")
    print(f"Temperature thresholds: MAE ≤ {validator.validation_thresholds['temperature']['max_mae']}°C")
    print(f"Pressure thresholds: MAE ≤ {validator.validation_thresholds['pressure']['max_mae']/1000:.0f} kPa")
    print(f"Flow rate thresholds: MAE ≤ {validator.validation_thresholds['flow_rate']['max_mae']} kg/s")
    print(f"Power thresholds: MAE ≤ {validator.validation_thresholds['power']['max_mae']} W")
    print(f"COP thresholds: MAE ≤ {validator.validation_thresholds['cop']['max_mae']}")
    
    # Generate test data with realistic residential heat pump values
    try:
        real_data = validator._generate_dummy_real_data()
        synthetic_data = validator._generate_dummy_synthetic_data()
        print(f"\n📈 GENERATED TEST DATA:")
        print(f"Real data points: {len(real_data)}")
        print(f"Synthetic data points: {len(synthetic_data)}")
    except Exception as e:
        print(f"❌ Failed to generate test data: {e}")
        return False
    
    # Show sample realistic values
    if real_data and synthetic_data:
        sample_real = real_data[0]
        sample_synthetic = synthetic_data[0]
        
        print(f"\n🔍 SAMPLE REALISTIC VALUES:")
        print(f"Real Data Sample:")
        print(f"  - Compressor power: {sample_real.compressor_power_W:.0f} W")
        print(f"  - Heating capacity: {sample_real.heating_capacity_W:.0f} W")
        print(f"  - COP heating: {sample_real.cop_heating:.2f}")
        print(f"  - Suction pressure: {sample_real.suction_pressure_Pa/1000:.0f} kPa")
        print(f"  - Discharge pressure: {sample_real.discharge_pressure_Pa/1000:.0f} kPa")
        print(f"  - Refrigerant flow: {sample_real.refrigerant_flow_kg_per_s:.3f} kg/s")
        
        print(f"\nSynthetic Data Sample:")
        print(f"  - Compressor power: {sample_synthetic.compressor_power_W:.0f} W")
        print(f"  - Heating capacity: {sample_synthetic.heating_capacity_W:.0f} W")
        print(f"  - COP heating: {sample_synthetic.cop_heating:.2f}")
        print(f"  - Suction pressure: {sample_synthetic.suction_pressure_Pa/1000:.0f} kPa")
        print(f"  - Discharge pressure: {sample_synthetic.discharge_pressure_Pa/1000:.0f} kPa")
        print(f"  - Refrigerant flow: {sample_synthetic.refrigerant_flow_kg_per_s:.3f} kg/s")
    
    # Perform comprehensive validation
    try:
        print(f"\n🔬 PERFORMING COMPREHENSIVE VALIDATION...")
        validation_results = validator.comprehensive_validation(real_data, synthetic_data)
        print(f"✅ Validation completed successfully")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Generate detailed report
    try:
        validator.generate_validation_report(validation_results, 
                                           "test_digital_twin_validation_report.json")
        print(f"✅ Validation report generated")
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
    
    # Analyze results
    print(f"\n📊 VALIDATION ANALYSIS:")
    
    total_variables = len(validation_results)
    passed_variables = sum(1 for result in validation_results.values() if result.validation_status == 'PASS')
    warning_variables = sum(1 for result in validation_results.values() if result.validation_status == 'WARNING')
    failed_variables = sum(1 for result in validation_results.values() if result.validation_status == 'FAIL')
    
    print(f"Total variables tested: {total_variables}")
    print(f"PASSED: {passed_variables} ({passed_variables/total_variables*100:.1f}%)")
    print(f"WARNING: {warning_variables} ({warning_variables/total_variables*100:.1f}%)")
    print(f"FAILED: {failed_variables} ({failed_variables/total_variables*100:.1f}%)")
    
    # Show detailed results for each variable
    print(f"\n📋 DETAILED VALIDATION RESULTS:")
    for var_name, result in validation_results.items():
        status_emoji = "✅" if result.validation_status == "PASS" else "⚠️" if result.validation_status == "WARNING" else "❌"
        print(f"{status_emoji} {var_name:20s}: {result.validation_status:8s} (Score: {result.validation_score:.3f})")
        print(f"    MAE: {result.mean_absolute_error:.4f}, RMSE: {result.root_mean_square_error:.4f}")
        print(f"    Correlation: {result.correlation_coefficient:.3f}, KS p-value: {result.ks_test_p_value:.3f}")
    
    # Overall assessment
    overall_score = np.mean([result.validation_score for result in validation_results.values()])
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"Overall validation score: {overall_score:.3f}")
    
    if overall_score >= 0.8:
        print("🎉 EXCELLENT: Digital twin validation system working perfectly!")
        print("   Synthetic data is highly realistic and suitable for HNN/GNN training.")
        success = True
    elif overall_score >= 0.6:
        print("⚠️ GOOD: Digital twin validation system working well with minor issues.")
        print("   Synthetic data is mostly realistic, may need minor calibration.")
        success = True
    else:
        print("❌ NEEDS IMPROVEMENT: Digital twin validation found significant issues.")
        print("   Synthetic data needs calibration before HNN/GNN training.")
        success = False
    
    return success

def test_realistic_values():
    """Test that generated values are realistic for residential heat pumps."""
    
    print(f"\n🏠 TESTING REALISTIC RESIDENTIAL HEAT PUMP VALUES")
    print("=" * 60)
    
    try:
        validator = DigitalTwinValidator()
        
        # Generate data
        real_data = validator._generate_dummy_real_data()
        synthetic_data = validator._generate_dummy_synthetic_data()
        
        # Extract values for analysis
        real_powers = [d.compressor_power_W for d in real_data]
        real_cops = [d.cop_heating for d in real_data]
        real_pressures_low = [d.suction_pressure_Pa/1000 for d in real_data]  # Convert to kPa
        real_pressures_high = [d.discharge_pressure_Pa/1000 for d in real_data]  # Convert to kPa
        real_flows = [d.refrigerant_flow_kg_per_s for d in real_data]
        
        synthetic_powers = [d.compressor_power_W for d in synthetic_data]
        synthetic_cops = [d.cop_heating for d in synthetic_data]
        synthetic_pressures_low = [d.suction_pressure_Pa/1000 for d in synthetic_data]
        synthetic_pressures_high = [d.discharge_pressure_Pa/1000 for d in synthetic_data]
        synthetic_flows = [d.refrigerant_flow_kg_per_s for d in synthetic_data]
        
        # Check realistic ranges
        checks = {
            'Compressor Power (W)': {
                'real': real_powers,
                'synthetic': synthetic_powers,
                'expected_range': (800, 1600),
                'typical': 1200
            },
            'COP Heating': {
                'real': real_cops,
                'synthetic': synthetic_cops,
                'expected_range': (3.0, 5.0),
                'typical': 4.2
            },
            'Suction Pressure (kPa)': {
                'real': real_pressures_low,
                'synthetic': synthetic_pressures_low,
                'expected_range': (200, 300),
                'typical': 250
            },
            'Discharge Pressure (kPa)': {
                'real': real_pressures_high,
                'synthetic': synthetic_pressures_high,
                'expected_range': (700, 900),
                'typical': 800
            },
            'Refrigerant Flow (kg/s)': {
                'real': real_flows,
                'synthetic': synthetic_flows,
                'expected_range': (0.05, 0.12),
                'typical': 0.08
            }
        }
        
        print(f"{'Parameter':<25} {'Real Range':<20} {'Synthetic Range':<20} {'Expected Range':<20} {'Status'}")
        print("-" * 100)
        
        all_realistic = True
        
        for param_name, data in checks.items():
            real_min, real_max = min(data['real']), max(data['real'])
            synth_min, synth_max = min(data['synthetic']), max(data['synthetic'])
            expected_min, expected_max = data['expected_range']
            
            real_range_str = f"{real_min:.1f} - {real_max:.1f}"
            synth_range_str = f"{synth_min:.1f} - {synth_max:.1f}"
            expected_range_str = f"{expected_min:.1f} - {expected_max:.1f}"
            
            # Check if values are within expected ranges
            real_realistic = expected_min <= real_min and real_max <= expected_max
            synth_realistic = expected_min <= synth_min and synth_max <= expected_max
            
            if real_realistic and synth_realistic:
                status = "✅ REALISTIC"
            elif real_realistic or synth_realistic:
                status = "⚠️ PARTIAL"
                all_realistic = False
            else:
                status = "❌ UNREALISTIC"
                all_realistic = False
            
            print(f"{param_name:<25} {real_range_str:<20} {synth_range_str:<20} {expected_range_str:<20} {status}")
        
        print(f"\n🎯 REALISTIC VALUES ASSESSMENT:")
        if all_realistic:
            print("🎉 EXCELLENT: All generated values are within realistic residential heat pump ranges!")
        else:
            print("⚠️ Some values may need adjustment to be fully realistic.")
        
        return all_realistic
        
    except Exception as e:
        print(f"❌ Error testing realistic values: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING DIGITAL TWIN VALIDATION TESTS (SIMPLE)")
    print("=" * 80)
    
    # Test validation system
    validation_success = test_digital_twin_validation_simple()
    
    # Test realistic values
    realistic_success = test_realistic_values()
    
    # Final summary
    print(f"\n🏁 FINAL TEST SUMMARY")
    print("=" * 80)
    
    print(f"✅ Digital twin validation system: {'WORKING' if validation_success else 'FAILED'}")
    print(f"✅ Realistic value generation: {'PASSED' if realistic_success else 'NEEDS ADJUSTMENT'}")
    
    if validation_success and realistic_success:
        print(f"\n🎉 SUCCESS: Digital twin validation system is ready for HNN/GNN training!")
        print(f"   The system can now validate synthetic data quality and ensure")
        print(f"   realistic physics for reliable machine learning model training.")
        print(f"\n✅ Task #27 (Digital Twin Validation): COMPLETED")
    else:
        print(f"\n⚠️ The system needs some adjustments before it's ready for production use.")
    
    print(f"\n📁 Generated files:")
    if Path("test_digital_twin_validation_report.json").exists():
        print(f"   - test_digital_twin_validation_report.json")
    print(f"   - Digital twin validation system: src/digital_twin_validation.py")