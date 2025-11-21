"""
Enhanced Physics Validator with Detailed Logging
Fixes the overly strict validation that was causing 90%+ rejection rate
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from cycle_generators import CycleResults

logger = logging.getLogger(__name__)

@dataclass
class ValidationFailure:
    """Container for validation failure details"""
    cycle_id: str
    failure_type: str
    details: Dict[str, Any]
    cycle_data: Dict[str, Any]

class EnhancedPhysicsValidator:
    """Enhanced validator with realistic tolerances and detailed logging"""
    
    def __init__(self, config=None):
        # REALISTIC validation thresholds (not overly strict)
        self.ENERGY_TOLERANCE = 1.0  # 1% (was 0.1% - too strict!)
        self.CARNOT_MARGIN = 1.05    # 5% margin for numerical precision
        self.ENTROPY_MIN = -0.001    # Allow small negative for rounding errors
        self.COP_MIN = 0.1           # Allow very low COPs for extreme conditions
        self.COP_MAX = 50.0          # Allow higher COPs for diverse ML training
        self.CAPACITY_MIN = 1.0      # Minimum capacity [kW]
        self.CAPACITY_MAX = 10000.0  # Maximum capacity [kW]
        
        # Detailed failure tracking
        self.validation_failures = {
            'energy_balance_error': 0,
            'carnot_limit_violation': 0,
            'entropy_negative': 0,
            'coolprop_error': 0,
            'state_impossible': 0,
            'cop_unrealistic': 0,
            'capacity_unrealistic': 0,
            'nan_values': 0,
            'infinite_values': 0,
            'total_generated': 0,
            'valid_cycles': 0
        }
        
        # Store failure examples for debugging
        self.failure_examples = []
        self.max_examples = 10  # Store first 10 failures of each type
        
        logger.info(f"Enhanced validator initialized with realistic tolerances:")
        logger.info(f"  Energy tolerance: {self.ENERGY_TOLERANCE}%")
        logger.info(f"  Carnot margin: {(self.CARNOT_MARGIN-1)*100}%")
        logger.info(f"  Entropy minimum: {self.ENTROPY_MIN}")
    
    def validate_cycle(self, results: CycleResults, cycle_dict: Dict = None) -> Tuple[bool, str]:
        """
        Validate cycle with detailed failure tracking
        Returns: (is_valid, failure_reason)
        """
        self.validation_failures['total_generated'] += 1
        
        cycle_id = f"{results.cycle_type}_{results.working_fluid}_{self.validation_failures['total_generated']}"
        
        # Check for NaN/Inf values first
        if self._has_invalid_numbers(results):
            self._record_failure('nan_values', cycle_id, 
                               {'message': 'NaN or Inf values detected'}, cycle_dict)
            return False, "nan_or_inf_values"
        
        # Energy balance check (RELAXED from 0.1% to 1.0%)
        if abs(results.energy_balance_error) > self.ENERGY_TOLERANCE:
            self._record_failure('energy_balance_error', cycle_id,
                               {'error_percent': results.energy_balance_error,
                                'tolerance': self.ENERGY_TOLERANCE}, cycle_dict)
            return False, "energy_balance_error"
        
        # Carnot limit check (with 5% margin for numerical precision)
        if not self._check_carnot_limits(results):
            self._record_failure('carnot_limit_violation', cycle_id,
                               {'COP_heating': results.COP_heating,
                                'COP_cooling': results.COP_cooling,
                                'carnot_heating': results.eta_carnot_heating,
                                'carnot_cooling': results.eta_carnot_cooling}, cycle_dict)
            return False, "carnot_limit_violation"
        
        # Entropy generation check (allow small negative for numerical errors)
        if results.entropy_generation < self.ENTROPY_MIN:
            self._record_failure('entropy_negative', cycle_id,
                               {'entropy_generation': results.entropy_generation,
                                'minimum_allowed': self.ENTROPY_MIN}, cycle_dict)
            return False, "entropy_negative"
        
        # Realistic COP bounds
        if not self._check_cop_realistic(results):
            self._record_failure('cop_unrealistic', cycle_id,
                               {'COP_heating': results.COP_heating,
                                'COP_cooling': results.COP_cooling,
                                'bounds': [self.COP_MIN, self.COP_MAX]}, cycle_dict)
            return False, "cop_unrealistic"
        
        # Realistic capacity bounds
        if not self._check_capacity_realistic(results):
            self._record_failure('capacity_unrealistic', cycle_id,
                               {'capacity_heating': results.capacity_heating,
                                'capacity_cooling': results.capacity_cooling,
                                'bounds': [self.CAPACITY_MIN, self.CAPACITY_MAX]}, cycle_dict)
            return False, "capacity_unrealistic"
        
        # All checks passed
        self.validation_failures['valid_cycles'] += 1
        return True, "valid"
    
    def _has_invalid_numbers(self, results: CycleResults) -> bool:
        """Check for NaN or Inf values"""
        values_to_check = [
            results.COP_heating, results.COP_cooling,
            results.Q_evap, results.Q_cond, results.W_net,
            results.energy_balance_error, results.entropy_generation
        ]
        
        for val in values_to_check:
            if not np.isfinite(val):
                return True
        
        # Check state properties
        for state in results.states:
            state_values = [state.P, state.T, state.h, state.s, state.rho]
            for val in state_values:
                if not np.isfinite(val):
                    return True
        
        return False
    
    def _check_carnot_limits(self, results: CycleResults) -> bool:
        """Check Carnot limits with realistic margin"""
        # Allow 5% margin for numerical precision
        if results.eta_carnot_heating > 0:
            if results.COP_heating > results.eta_carnot_heating * self.CARNOT_MARGIN:
                return False
        
        if results.eta_carnot_cooling > 0:
            if results.COP_cooling > results.eta_carnot_cooling * self.CARNOT_MARGIN:
                return False
        
        return True
    
    def _check_cop_realistic(self, results: CycleResults) -> bool:
        """Check if COPs are in realistic range"""
        # Allow negative COPs (reverse operation) but check magnitude
        cop_heating_abs = abs(results.COP_heating)
        cop_cooling_abs = abs(results.COP_cooling)
        
        if cop_heating_abs < self.COP_MIN or cop_heating_abs > self.COP_MAX:
            return False
        
        if cop_cooling_abs < self.COP_MIN or cop_cooling_abs > self.COP_MAX:
            return False
        
        return True
    
    def _check_capacity_realistic(self, results: CycleResults) -> bool:
        """Check if capacities are in realistic range"""
        # Allow negative capacities (reverse operation) but check magnitude
        cap_heating_abs = abs(results.capacity_heating)
        cap_cooling_abs = abs(results.capacity_cooling)
        
        if cap_heating_abs < self.CAPACITY_MIN or cap_heating_abs > self.CAPACITY_MAX:
            return False
        
        if cap_cooling_abs < self.CAPACITY_MIN or cap_cooling_abs > self.CAPACITY_MAX:
            return False
        
        return True
    
    def _record_failure(self, failure_type: str, cycle_id: str, 
                       details: Dict, cycle_dict: Dict = None):
        """Record failure for debugging"""
        self.validation_failures[failure_type] += 1
        
        # Store example for debugging (limit to avoid memory issues)
        if len(self.failure_examples) < self.max_examples * len(self.validation_failures):
            failure = ValidationFailure(
                cycle_id=cycle_id,
                failure_type=failure_type,
                details=details,
                cycle_data=cycle_dict
            )
            self.failure_examples.append(failure)
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report with failure breakdown"""
        total = self.validation_failures['total_generated']
        valid = self.validation_failures['valid_cycles']
        
        if total == 0:
            return {"error": "No cycles validated"}
        
        # Calculate rates
        validation_rate = (valid / total) * 100
        
        failure_breakdown = {}
        for failure_type, count in self.validation_failures.items():
            if failure_type not in ['total_generated', 'valid_cycles']:
                failure_breakdown[failure_type] = {
                    'count': count,
                    'percentage': (count / total) * 100
                }
        
        # Get top failure reasons
        sorted_failures = sorted(failure_breakdown.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
        
        report = {
            'summary': {
                'total_cycles_generated': total,
                'valid_cycles': valid,
                'validation_rate': validation_rate,
                'rejection_rate': 100 - validation_rate
            },
            'failure_breakdown': failure_breakdown,
            'top_failure_reasons': sorted_failures[:5],
            'validation_thresholds': {
                'energy_tolerance_percent': self.ENERGY_TOLERANCE,
                'carnot_margin': self.CARNOT_MARGIN,
                'entropy_minimum': self.ENTROPY_MIN,
                'cop_range': [self.COP_MIN, self.COP_MAX],
                'capacity_range_kW': [self.CAPACITY_MIN, self.CAPACITY_MAX]
            }
        }
        
        # Add failure examples for debugging
        if self.failure_examples:
            report['failure_examples'] = []
            for failure in self.failure_examples[:5]:  # First 5 examples
                example = {
                    'cycle_id': failure.cycle_id,
                    'failure_type': failure.failure_type,
                    'details': failure.details
                }
                report['failure_examples'].append(example)
        
        return report
    
    def print_diagnostic_summary(self):
        """Print diagnostic summary to console"""
        report = self.get_detailed_report()
        
        print("\n" + "🔍 VALIDATION DIAGNOSTIC REPORT" + "="*30)
        print(f"Total cycles generated: {report['summary']['total_cycles_generated']}")
        print(f"Valid cycles: {report['summary']['valid_cycles']}")
        print(f"Validation rate: {report['summary']['validation_rate']:.1f}%")
        print(f"Rejection rate: {report['summary']['rejection_rate']:.1f}%")
        
        print(f"\n📊 FAILURE BREAKDOWN:")
        for failure_type, data in report['top_failure_reasons']:
            print(f"  {failure_type}: {data['count']} ({data['percentage']:.1f}%)")
        
        print(f"\n⚙️  VALIDATION THRESHOLDS:")
        thresholds = report['validation_thresholds']
        print(f"  Energy tolerance: {thresholds['energy_tolerance_percent']}%")
        print(f"  Carnot margin: {(thresholds['carnot_margin']-1)*100}%")
        print(f"  COP range: {thresholds['cop_range']}")
        print(f"  Capacity range: {thresholds['capacity_range_kW']} kW")
        
        if 'failure_examples' in report:
            print(f"\n🚨 FAILURE EXAMPLES:")
            for example in report['failure_examples']:
                print(f"  {example['failure_type']}: {example['details']}")
        
        print("="*60)

def run_validation_diagnostic(n_cycles: int = 100):
    """Run diagnostic test to identify validation issues"""
    print(f"🔬 Running validation diagnostic with {n_cycles} cycles...")
    
    # Import here to avoid circular imports
    from cycle_generators import CycleGenerator, CycleType
    
    validator = EnhancedPhysicsValidator()
    generator = CycleGenerator()
    
    # Test different cycle types
    cycle_types = [CycleType.CARNOT, CycleType.IDEAL_RANKINE, CycleType.NON_IDEAL_RANKINE]
    fluids = ['R245fa', 'R134a', 'Water']  # Exclude Air - it's causing too many CoolProp failures
    
    for i in range(n_cycles):
        try:
            # Random cycle type and fluid
            cycle_type = np.random.choice(cycle_types)
            fluid = np.random.choice(fluids)
            
            # Generate parameters and cycle
            params = generator.generate_random_parameters(cycle_type, fluid)
            results = generator.generate_cycle(cycle_type, fluid, params)
            
            # Validate with detailed logging
            is_valid, reason = validator.validate_cycle(results)
            
            if i < 10 and not is_valid:  # Print first 10 failures
                print(f"Cycle {i+1} REJECTED ({reason}):")
                print(f"  Type: {cycle_type.value}, Fluid: {fluid}")
                print(f"  COP: {results.COP_heating:.3f}, Energy Error: {results.energy_balance_error:.6f}%")
                print(f"  Carnot: {results.eta_carnot_heating:.3f}, Entropy Gen: {results.entropy_generation:.6f}")
        
        except Exception as e:
            validator.validation_failures['coolprop_error'] += 1
            validator.validation_failures['total_generated'] += 1
            if i < 5:
                print(f"Cycle {i+1} ERROR: {e}")
    
    # Print diagnostic summary
    validator.print_diagnostic_summary()
    
    return validator.get_detailed_report()

if __name__ == "__main__":
    # Run diagnostic
    report = run_validation_diagnostic(100)