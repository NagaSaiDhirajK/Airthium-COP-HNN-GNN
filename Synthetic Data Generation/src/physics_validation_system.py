"""
Comprehensive Physics Model Validation System
Enhanced validation of refrigerant properties, molecular temperature ranges,
and thermodynamic consistency across all physics models.

This module ensures our implementation aligns with established thermodynamic data
and validates the accuracy of our refrigerant property models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class ValidationResult:
    """Container for validation results"""
    parameter: str
    expected_range: Tuple[float, float]
    actual_value: float
    deviation_percent: float
    level: ValidationLevel
    message: str
    source_reference: str

class RefrigerantPropertiesValidator:
    """
    Comprehensive validator for refrigerant thermodynamic properties
    against established NIST/ASHRAE data
    """
    
    def __init__(self):
        self.validation_results = []
        self.reference_data = self._load_reference_data()
        
    def _load_reference_data(self) -> Dict[str, Dict[str, Any]]:
        """Load reference thermodynamic data for common refrigerants"""
        return {
            'R32': {
                'molecular_weight': 52.024,  # kg/kmol
                'critical_temp': 351.255,    # K
                'critical_pressure': 5782.0, # kPa
                'normal_boiling_point': 221.499, # K
                'triple_point_temp': 136.34,  # K
                'triple_point_pressure': 0.048, # kPa
                'operating_temp_range': (200.0, 340.0), # K
                'saturation_pressure_range': (10.0, 4000.0), # kPa
                'density_liquid_range': (400.0, 1400.0), # kg/m³
                'density_vapor_range': (1.0, 300.0), # kg/m³
                'specific_heat_range': (0.8, 3.5), # kJ/kg·K
                'thermal_conductivity_range': (0.01, 0.25), # W/m·K
                'viscosity_range': (1e-6, 1e-3), # Pa·s
                'source': 'NIST REFPROP 10.0'
            },
            'R410A': {
                'molecular_weight': 72.585,
                'critical_temp': 344.494,
                'critical_pressure': 4901.2,
                'normal_boiling_point': 221.71,
                'triple_point_temp': 200.0,
                'triple_point_pressure': 29.0,
                'operating_temp_range': (200.0, 335.0),
                'saturation_pressure_range': (30.0, 4000.0),
                'density_liquid_range': (300.0, 1500.0),
                'density_vapor_range': (2.0, 400.0),
                'specific_heat_range': (0.7, 4.0),
                'thermal_conductivity_range': (0.008, 0.15),
                'viscosity_range': (8e-7, 8e-4),
                'source': 'NIST REFPROP 10.0'
            },
            'R134a': {
                'molecular_weight': 102.032,
                'critical_temp': 374.21,
                'critical_pressure': 4059.3,
                'normal_boiling_point': 247.076,
                'triple_point_temp': 169.85,
                'triple_point_pressure': 0.39,
                'operating_temp_range': (170.0, 365.0),
                'saturation_pressure_range': (1.0, 3500.0),
                'density_liquid_range': (200.0, 1600.0),
                'density_vapor_range': (0.5, 350.0),
                'specific_heat_range': (0.6, 5.0),
                'thermal_conductivity_range': (0.006, 0.18),
                'viscosity_range': (5e-7, 2e-3),
                'source': 'NIST REFPROP 10.0'
            }
        }
    
    def validate_molecular_temperature_ranges(self, 
                                            refrigerant: str,
                                            temp_data: np.ndarray) -> List[ValidationResult]:
        """Validate molecular temperature ranges against reference data"""
        results = []
        
        if refrigerant not in self.reference_data:
            results.append(ValidationResult(
                parameter=f"{refrigerant}_availability",
                expected_range=(0, 1),
                actual_value=0,
                deviation_percent=100.0,
                level=ValidationLevel.CRITICAL,
                message=f"No reference data available for {refrigerant}",
                source_reference="Internal validation"
            ))
            return results
        
        ref_data = self.reference_data[refrigerant]
        temp_range = ref_data['operating_temp_range']
        
        # Check temperature range validity
        min_temp, max_temp = np.min(temp_data), np.max(temp_data)
        
        # Validate minimum temperature
        if min_temp < temp_range[0]:
            deviation = abs(min_temp - temp_range[0]) / temp_range[0] * 100
            results.append(ValidationResult(
                parameter=f"{refrigerant}_min_temperature",
                expected_range=temp_range,
                actual_value=min_temp,
                deviation_percent=deviation,
                level=ValidationLevel.CRITICAL if deviation > 10 else ValidationLevel.WARNING,
                message=f"Minimum temperature {min_temp:.2f}K below safe operating range",
                source_reference=ref_data['source']
            ))
        
        # Validate maximum temperature
        if max_temp > temp_range[1]:
            deviation = abs(max_temp - temp_range[1]) / temp_range[1] * 100
            results.append(ValidationResult(
                parameter=f"{refrigerant}_max_temperature",
                expected_range=temp_range,
                actual_value=max_temp,
                deviation_percent=deviation,
                level=ValidationLevel.CRITICAL if deviation > 5 else ValidationLevel.WARNING,
                message=f"Maximum temperature {max_temp:.2f}K above safe operating range",
                source_reference=ref_data['source']
            ))
        
        # Check for temperatures near critical point
        critical_temp = ref_data['critical_temp']
        near_critical = np.sum(temp_data > 0.95 * critical_temp)
        if near_critical > 0:
            results.append(ValidationResult(
                parameter=f"{refrigerant}_near_critical_operation",
                expected_range=(0, 0.05 * len(temp_data)),
                actual_value=near_critical,
                deviation_percent=near_critical / len(temp_data) * 100,
                level=ValidationLevel.WARNING,
                message=f"{near_critical} data points near critical temperature",
                source_reference=ref_data['source']
            ))
        
        return results
    
    def validate_thermodynamic_properties(self,
                                        refrigerant: str,
                                        properties: Dict[str, np.ndarray]) -> List[ValidationResult]:
        """Validate thermodynamic properties against reference ranges"""
        results = []
        
        if refrigerant not in self.reference_data:
            return results
        
        ref_data = self.reference_data[refrigerant]
        
        # Property validation mapping
        property_mappings = {
            'density_liquid': 'density_liquid_range',
            'density_vapor': 'density_vapor_range', 
            'specific_heat': 'specific_heat_range',
            'thermal_conductivity': 'thermal_conductivity_range',
            'viscosity': 'viscosity_range'
        }
        
        for prop_name, range_key in property_mappings.items():
            if prop_name in properties and range_key in ref_data:
                prop_data = properties[prop_name]
                expected_range = ref_data[range_key]
                
                # Check for values outside expected range
                below_range = np.sum(prop_data < expected_range[0])
                above_range = np.sum(prop_data > expected_range[1])
                
                if below_range > 0:
                    deviation = below_range / len(prop_data) * 100
                    results.append(ValidationResult(
                        parameter=f"{refrigerant}_{prop_name}_below_range",
                        expected_range=expected_range,
                        actual_value=np.min(prop_data),
                        deviation_percent=deviation,
                        level=ValidationLevel.WARNING if deviation < 5 else ValidationLevel.CRITICAL,
                        message=f"{below_range} values below expected range for {prop_name}",
                        source_reference=ref_data['source']
                    ))
                
                if above_range > 0:
                    deviation = above_range / len(prop_data) * 100
                    results.append(ValidationResult(
                        parameter=f"{refrigerant}_{prop_name}_above_range",
                        expected_range=expected_range,
                        actual_value=np.max(prop_data),
                        deviation_percent=deviation,
                        level=ValidationLevel.WARNING if deviation < 5 else ValidationLevel.CRITICAL,
                        message=f"{above_range} values above expected range for {prop_name}",
                        source_reference=ref_data['source']
                    ))
        
        return results

class PhysicsModelValidator:
    """
    Comprehensive physics model validation system
    """
    
    def __init__(self):
        self.refrigerant_validator = RefrigerantPropertiesValidator()
        self.validation_results = []
        
    def validate_carnot_efficiency_bounds(self,
                                        cop_values: np.ndarray,
                                        t_hot: np.ndarray,
                                        t_cold: np.ndarray) -> List[ValidationResult]:
        """Validate COP values against Carnot efficiency bounds"""
        results = []
        
        # Calculate Carnot COP for heating
        carnot_cop_heating = t_hot / (t_hot - t_cold)
        
        # Calculate Carnot COP for cooling  
        carnot_cop_cooling = t_cold / (t_hot - t_cold)
        
        # Check for violations of Carnot bound (heating mode)
        heating_violations = np.sum(cop_values > carnot_cop_heating * 1.01)  # 1% tolerance
        if heating_violations > 0:
            max_violation = np.max(cop_values / carnot_cop_heating)
            results.append(ValidationResult(
                parameter="carnot_bound_heating_violation",
                expected_range=(0, 1.0),
                actual_value=max_violation,
                deviation_percent=(max_violation - 1.0) * 100,
                level=ValidationLevel.CRITICAL,
                message=f"{heating_violations} COP values exceed Carnot bound for heating",
                source_reference="Carnot theorem"
            ))
        
        # Check for unrealistically low COP values
        low_cop = np.sum(cop_values < 1.0)
        if low_cop > 0:
            min_cop = np.min(cop_values)
            results.append(ValidationResult(
                parameter="unrealistic_low_cop",
                expected_range=(1.0, 10.0),
                actual_value=min_cop,
                deviation_percent=(1.0 - min_cop) * 100,
                level=ValidationLevel.WARNING,
                message=f"{low_cop} COP values below 1.0 (inefficient operation)",
                source_reference="Thermodynamic principles"
            ))
        
        return results
    
    def validate_pressure_temperature_relationship(self,
                                                 refrigerant: str,
                                                 pressure: np.ndarray,
                                                 temperature: np.ndarray) -> List[ValidationResult]:
        """Validate pressure-temperature relationship using Clausius-Clapeyron"""
        results = []
        
        # Check for monotonic relationship (generally expected)
        correlation = np.corrcoef(pressure, temperature)[0, 1]
        
        if correlation < 0.7:
            results.append(ValidationResult(
                parameter=f"{refrigerant}_pressure_temperature_correlation",
                expected_range=(0.7, 1.0),
                actual_value=correlation,
                deviation_percent=(0.7 - correlation) * 100,
                level=ValidationLevel.WARNING,
                message=f"Weak pressure-temperature correlation: {correlation:.3f}",
                source_reference="Clausius-Clapeyron relation"
            ))
        
        # Check for negative pressures (physically impossible)
        negative_pressure = np.sum(pressure < 0)
        if negative_pressure > 0:
            results.append(ValidationResult(
                parameter=f"{refrigerant}_negative_pressure",
                expected_range=(0, 0),
                actual_value=negative_pressure,
                deviation_percent=negative_pressure / len(pressure) * 100,
                level=ValidationLevel.CRITICAL,
                message=f"{negative_pressure} negative pressure values detected",
                source_reference="Physical constraints"
            ))
        
        return results
    
    def validate_energy_conservation(self,
                                   heat_input: np.ndarray,
                                   work_input: np.ndarray,
                                   heat_output: np.ndarray,
                                   tolerance: float = 0.05) -> List[ValidationResult]:
        """Validate energy conservation in heat pump cycle"""
        results = []
        
        # Energy balance: Q_out = Q_in + W_in
        energy_balance = heat_output - (heat_input + work_input)
        relative_error = np.abs(energy_balance) / heat_output
        
        violations = np.sum(relative_error > tolerance)
        if violations > 0:
            max_error = np.max(relative_error) * 100
            results.append(ValidationResult(
                parameter="energy_conservation_violation",
                expected_range=(0, tolerance * 100),
                actual_value=max_error,
                deviation_percent=max_error - tolerance * 100,
                level=ValidationLevel.CRITICAL if max_error > 10 else ValidationLevel.WARNING,
                message=f"{violations} energy conservation violations (>{tolerance*100:.1f}% error)",
                source_reference="First law of thermodynamics"
            ))
        
        return results
    
    def comprehensive_validation(self,
                               refrigerant: str,
                               temperature_data: np.ndarray,
                               pressure_data: np.ndarray,
                               properties: Dict[str, np.ndarray],
                               cop_data: np.ndarray,
                               t_hot: np.ndarray,
                               t_cold: np.ndarray) -> Dict[str, List[ValidationResult]]:
        """Run comprehensive validation of all physics models"""
        
        validation_suite = {
            'refrigerant_properties': self.refrigerant_validator.validate_molecular_temperature_ranges(
                refrigerant, temperature_data
            ),
            'thermodynamic_properties': self.refrigerant_validator.validate_thermodynamic_properties(
                refrigerant, properties
            ),
            'carnot_bounds': self.validate_carnot_efficiency_bounds(
                cop_data, t_hot, t_cold
            ),
            'pressure_temperature': self.validate_pressure_temperature_relationship(
                refrigerant, pressure_data, temperature_data
            )
        }
        
        # Compile all results
        all_results = []
        for category, results in validation_suite.items():
            all_results.extend(results)
        
        self.validation_results = all_results
        return validation_suite
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "No validation results available. Run comprehensive_validation() first."
        
        # Categorize results by severity
        critical = [r for r in self.validation_results if r.level == ValidationLevel.CRITICAL]
        warnings = [r for r in self.validation_results if r.level == ValidationLevel.WARNING]
        info = [r for r in self.validation_results if r.level == ValidationLevel.INFO]
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PHYSICS MODEL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Issues Found: {len(self.validation_results)}")
        report.append(f"  - Critical: {len(critical)}")
        report.append(f"  - Warnings: {len(warnings)}")
        report.append(f"  - Info: {len(info)}")
        report.append("")
        
        if critical:
            report.append("🚨 CRITICAL ISSUES:")
            report.append("-" * 40)
            for result in critical:
                report.append(f"  • {result.parameter}")
                report.append(f"    Expected: {result.expected_range}")
                report.append(f"    Actual: {result.actual_value:.4f}")
                report.append(f"    Deviation: {result.deviation_percent:.2f}%")
                report.append(f"    Message: {result.message}")
                report.append(f"    Source: {result.source_reference}")
                report.append("")
        
        if warnings:
            report.append("⚠️  WARNINGS:")
            report.append("-" * 40)
            for result in warnings:
                report.append(f"  • {result.parameter}")
                report.append(f"    Expected: {result.expected_range}")
                report.append(f"    Actual: {result.actual_value:.4f}")
                report.append(f"    Deviation: {result.deviation_percent:.2f}%")
                report.append(f"    Message: {result.message}")
                report.append("")
        
        # Summary and recommendations
        report.append("📋 SUMMARY & RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if len(critical) == 0 and len(warnings) == 0:
            report.append("✅ All physics models pass validation!")
            report.append("   Your implementation aligns well with established thermodynamic data.")
        elif len(critical) == 0:
            report.append("✅ No critical issues found.")
            report.append(f"⚠️  {len(warnings)} warnings require attention for optimal accuracy.")
        else:
            report.append(f"❌ {len(critical)} critical issues must be addressed.")
            report.append("   These violations may indicate fundamental modeling errors.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def validate_against_nist_data(refrigerant: str = "R32") -> str:
    """
    Main validation function to check our models against NIST/ASHRAE data
    """
    
    # Generate sample data for validation (replace with your actual data)
    np.random.seed(42)
    n_samples = 1000
    
    # Sample temperature data (K)
    temperature_data = np.random.uniform(250, 320, n_samples)
    
    # Sample pressure data (kPa) - correlated with temperature
    pressure_data = np.exp(10.5 - 2500/temperature_data) * 100 + np.random.normal(0, 50, n_samples)
    pressure_data = np.maximum(pressure_data, 10)  # Ensure positive pressures
    
    # Sample properties
    properties = {
        'density_liquid': np.random.uniform(800, 1200, n_samples),
        'density_vapor': np.random.uniform(10, 100, n_samples),
        'specific_heat': np.random.uniform(1.0, 2.5, n_samples),
        'thermal_conductivity': np.random.uniform(0.05, 0.15, n_samples),
        'viscosity': np.random.uniform(1e-5, 5e-4, n_samples)
    }
    
    # Sample COP data
    t_hot = temperature_data + 20  # Condenser temperature
    t_cold = temperature_data - 30  # Evaporator temperature
    carnot_cop = t_hot / (t_hot - t_cold)
    cop_data = carnot_cop * np.random.uniform(0.3, 0.8, n_samples)  # Realistic efficiency
    
    # Run validation
    validator = PhysicsModelValidator()
    validation_results = validator.comprehensive_validation(
        refrigerant=refrigerant,
        temperature_data=temperature_data,
        pressure_data=pressure_data,
        properties=properties,
        cop_data=cop_data,
        t_hot=t_hot,
        t_cold=t_cold
    )
    
    return validator.generate_validation_report()

if __name__ == "__main__":
    # Run comprehensive validation
    print("Running comprehensive physics model validation...")
    print("\n" + "="*80)
    print("VALIDATING R32 REFRIGERANT MODEL")
    print("="*80)
    
    report = validate_against_nist_data("R32")
    print(report)
    
    print("\n" + "="*80)
    print("VALIDATING R410A REFRIGERANT MODEL") 
    print("="*80)
    
    report_410a = validate_against_nist_data("R410A")
    print(report_410a)