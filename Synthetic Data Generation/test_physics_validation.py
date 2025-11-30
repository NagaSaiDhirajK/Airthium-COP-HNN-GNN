#!/usr/bin/env python3
"""
Test script for physics validation system
"""

import sys
sys.path.append('src')

from physics_validation_system import PhysicsModelValidator, validate_against_nist_data
import numpy as np

def test_validation_system():
    """Test the physics validation system with simple data"""
    print("🔬 Testing Physics Validation System")
    print("=" * 50)
    
    # Create validator
    validator = PhysicsModelValidator()
    
    # Generate simple test data
    np.random.seed(42)
    n_samples = 100
    
    # Temperature data (K) - some within range, some outside
    temperature_data = np.array([
        *np.random.uniform(250, 320, 80),  # Most within range
        *np.random.uniform(180, 200, 10),  # Some below range
        *np.random.uniform(350, 380, 10)   # Some above range
    ])
    
    # Pressure data (kPa) - correlated with temperature
    pressure_data = np.exp(10.5 - 2500/temperature_data) * 100 + np.random.normal(0, 50, n_samples)
    pressure_data = np.maximum(pressure_data, 10)  # Ensure positive pressures
    
    # Properties with some values outside expected ranges
    properties = {
        'density_liquid': np.random.uniform(600, 1400, n_samples),  # Some below range
        'density_vapor': np.random.uniform(5, 150, n_samples),      # Some above range
        'specific_heat': np.random.uniform(0.5, 4.0, n_samples),   # Some outside range
        'thermal_conductivity': np.random.uniform(0.005, 0.3, n_samples),  # Some outside range
        'viscosity': np.random.uniform(5e-7, 2e-3, n_samples)      # Within range
    }
    
    # COP data with some violations
    t_hot = temperature_data + 20  # Condenser temperature
    t_cold = temperature_data - 30  # Evaporator temperature
    carnot_cop = t_hot / (t_hot - t_cold)
    
    # Create some COP values that violate Carnot bound
    cop_data = np.array([
        *carnot_cop[:80] * np.random.uniform(0.3, 0.8, 80),  # Realistic efficiency
        *carnot_cop[80:90] * np.random.uniform(1.1, 1.3, 10),  # Carnot violations
        *np.random.uniform(0.5, 0.9, 10)  # Some low COP values
    ])
    
    print(f"📊 Test Data Generated:")
    print(f"   Temperature range: {np.min(temperature_data):.1f} - {np.max(temperature_data):.1f} K")
    print(f"   Pressure range: {np.min(pressure_data):.1f} - {np.max(pressure_data):.1f} kPa")
    print(f"   COP range: {np.min(cop_data):.2f} - {np.max(cop_data):.2f}")
    print(f"   Max Carnot COP: {np.max(carnot_cop):.2f}")
    print()
    
    # Run validation for R32
    print("🔍 Running R32 Validation...")
    validation_results = validator.comprehensive_validation(
        refrigerant="R32",
        temperature_data=temperature_data,
        pressure_data=pressure_data,
        properties=properties,
        cop_data=cop_data,
        t_hot=t_hot,
        t_cold=t_cold
    )
    
    print(f"✅ Validation completed. Found {len(validator.validation_results)} issues.")
    print()
    
    # Generate and print report
    report = validator.generate_validation_report()
    print(report)
    
    return len(validator.validation_results) > 0

if __name__ == "__main__":
    success = test_validation_system()
    if success:
        print("\n🎉 Physics validation system working correctly!")
    else:
        print("\n❌ No validation issues found - check test data generation")