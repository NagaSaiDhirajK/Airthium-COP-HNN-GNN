"""
Improved Thermodynamic Cycle Generators with Enhanced Parameter Generation
Targets 80%+ validation rate through physics-constrained parameter generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from enum import Enum

from thermodynamic_properties import (
    FluidProperties, ThermodynamicState, get_fluid_properties
)
from cycle_generators import (
    CycleType, CycleParameters, CycleResults, ThermodynamicCycle,
    CarnotCycle, RankineCycle, TranscriticalCycle
)

class ImprovedCycleGenerator:
    """Enhanced cycle generator with physics-constrained parameter generation"""
    
    def __init__(self):
        # Reliable working fluids (exclude problematic ones like Air, thermal oils)
        self.reliable_refrigerants = [
            'R245fa', 'R134a', 'R410A', 'R600a', 'R290', 'R1234ze(E)', 'R1233zd(E)'
        ]
        
        self.reliable_secondary_fluids = [
            'Water', 'Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol'
        ]
        
        # Fluid-specific operating envelopes (conservative ranges)
        self.fluid_envelopes = {
            'R245fa': {'T_evap_range': (268.15, 288.15), 'T_cond_range': (318.15, 348.15)},  # -5 to 15°C evap, 45-75°C cond
            'R134a': {'T_evap_range': (263.15, 283.15), 'T_cond_range': (313.15, 343.15)},   # -10 to 10°C evap, 40-70°C cond
            'R410A': {'T_evap_range': (258.15, 278.15), 'T_cond_range': (318.15, 348.15)},   # -15 to 5°C evap, 45-75°C cond
            'R600a': {'T_evap_range': (268.15, 288.15), 'T_cond_range': (313.15, 343.15)},   # -5 to 15°C evap, 40-70°C cond
            'R290': {'T_evap_range': (263.15, 283.15), 'T_cond_range': (313.15, 343.15)},    # -10 to 10°C evap, 40-70°C cond
            'R1234ze(E)': {'T_evap_range': (268.15, 288.15), 'T_cond_range': (318.15, 348.15)},
            'R1233zd(E)': {'T_evap_range': (273.15, 293.15), 'T_cond_range': (323.15, 353.15)},
            'Water': {'T_evap_range': (353.15, 373.15), 'T_cond_range': (373.15, 423.15)},   # Steam cycles
            'Toluene': {'T_evap_range': (293.15, 323.15), 'T_cond_range': (343.15, 373.15)}, # ORC
            'Cyclohexane': {'T_evap_range': (293.15, 323.15), 'T_cond_range': (343.15, 373.15)},
            'n-Pentane': {'T_evap_range': (283.15, 313.15), 'T_cond_range': (333.15, 363.15)},
            'Ethanol': {'T_evap_range': (293.15, 323.15), 'T_cond_range': (343.15, 373.15)}
        }
        
        # Initialize cycle classes
        self.cycle_classes = {
            CycleType.CARNOT: CarnotCycle,
            CycleType.IDEAL_RANKINE: lambda fluid: RankineCycle(fluid, ideal=True),
            CycleType.NON_IDEAL_RANKINE: lambda fluid: RankineCycle(fluid, ideal=False),
            CycleType.TRANSCRITICAL: TranscriticalCycle,
            CycleType.ORC: lambda fluid: RankineCycle(fluid, ideal=False),
        }
    
    def generate_physics_constrained_parameters(self, cycle_type: CycleType, 
                                              fluid_name: str) -> CycleParameters:
        """
        Generate thermodynamically feasible parameters with high success rate
        
        Key improvements:
        1. Fluid-specific operating envelopes
        2. Physics-constrained temperature lifts
        3. Realistic component efficiencies
        4. Conservative pressure ranges
        """
        
        # Get fluid envelope or use defaults
        if fluid_name in self.fluid_envelopes:
            envelope = self.fluid_envelopes[fluid_name]
            T_evap_min, T_evap_max = envelope['T_evap_range']
            T_cond_min, T_cond_max = envelope['T_cond_range']
        else:
            # Conservative defaults for unknown fluids
            T_evap_min, T_evap_max = 268.15, 288.15  # -5 to 15°C
            T_cond_min, T_cond_max = 318.15, 348.15  # 45 to 75°C
        
        # Generate evaporator temperature
        T_evap = np.random.uniform(T_evap_min, T_evap_max)
        
        # Generate condenser temperature with physics constraints
        min_temp_lift = 25.0  # Minimum 25K temperature difference
        max_temp_lift = 50.0  # Maximum 50K for good efficiency
        optimal_temp_lift = 35.0  # Target around 35K
        
        # Bias towards optimal temperature lift
        temp_lift = np.random.normal(optimal_temp_lift, 8.0)  # Normal distribution around 35K
        temp_lift = np.clip(temp_lift, min_temp_lift, max_temp_lift)
        
        T_cond = T_evap + temp_lift
        T_cond = np.clip(T_cond, T_cond_min, T_cond_max)
        
        # Validate against fluid critical temperature if available
        try:
            fluid = get_fluid_properties(fluid_name)
            if hasattr(fluid, 'T_crit') and fluid.T_crit:
                T_max_safe = fluid.T_crit * 0.85  # 85% of critical temperature
                if T_cond > T_max_safe:
                    T_cond = T_max_safe - np.random.uniform(10, 20)
                if T_evap > T_max_safe * 0.70:
                    T_evap = T_max_safe * 0.70 - np.random.uniform(5, 15)
        except:
            pass  # Use envelope limits if fluid properties unavailable
        
        # Conservative mass flow rate
        mass_flow = np.random.uniform(0.8, 1.2)  # kg/s (narrow range for stability)
        
        # Realistic superheat and subcooling
        superheat = np.random.uniform(5.0, 10.0)   # 5-10K superheat
        subcool = np.random.uniform(3.0, 8.0)      # 3-8K subcooling
        
        # High-quality component efficiencies (realistic but not extreme)
        eta_comp = np.random.uniform(0.82, 0.88)   # 82-88% compressor efficiency
        eta_exp = np.random.uniform(0.78, 0.85)    # 78-85% expander efficiency
        eta_mech = np.random.uniform(0.94, 0.97)   # 94-97% mechanical efficiency
        eta_motor = np.random.uniform(0.91, 0.95)  # 91-95% motor efficiency
        
        # Conservative pressure drops
        dp_evap = np.random.uniform(2000, 4000)    # 2-4 kPa
        dp_cond = np.random.uniform(2000, 4000)    # 2-4 kPa
        
        # Realistic environmental conditions
        T_ambient = np.random.uniform(283.15, 303.15)  # 10 to 30°C
        humidity = np.random.uniform(0.40, 0.70)       # 40-70% RH
        part_load = np.random.uniform(0.75, 1.0)       # 75-100% load (avoid very low loads)
        
        return CycleParameters(
            T_evap=T_evap,
            T_cond=T_cond,
            mass_flow=mass_flow,
            working_fluid=fluid_name,
            superheat=superheat,
            subcool=subcool,
            eta_comp_isentropic=eta_comp,
            eta_exp_isentropic=eta_exp,
            eta_mechanical=eta_mech,
            eta_motor=eta_motor,
            dp_evap=dp_evap,
            dp_cond=dp_cond,
            T_ambient=T_ambient,
            humidity=humidity,
            part_load_ratio=part_load
        )
    
    def generate_cycle_with_retry(self, cycle_type: CycleType, fluid_name: str, 
                                max_retries: int = 3) -> Optional[CycleResults]:
        """
        Generate cycle with retry logic for improved success rate
        """
        
        for attempt in range(max_retries):
            try:
                # Generate physics-constrained parameters
                params = self.generate_physics_constrained_parameters(cycle_type, fluid_name)
                
                # Create cycle instance
                if cycle_type in [CycleType.IDEAL_RANKINE, CycleType.NON_IDEAL_RANKINE, CycleType.ORC]:
                    cycle = self.cycle_classes[cycle_type](fluid_name)
                else:
                    cycle = self.cycle_classes[cycle_type](fluid_name)
                
                # Generate cycle
                results = cycle.generate_cycle(params)
                
                # Quick validation check
                if (results.energy_balance_error < 1.0 and  # Energy balance within 1%
                    results.carnot_limit_check and          # Below Carnot limits
                    results.COP_heating > 0.5 and          # Reasonable COP
                    results.COP_heating < 15.0 and         # Not unrealistically high
                    results.W_net > 0):                     # Positive work input
                    
                    return results
                
            except Exception as e:
                if attempt == max_retries - 1:
                    warnings.warn(f"Failed to generate {cycle_type.value} cycle for {fluid_name} after {max_retries} attempts: {e}")
                continue
        
        return None
    
    def select_reliable_fluid(self, cycle_type: CycleType) -> str:
        """Select a reliable fluid for the given cycle type"""
        
        if cycle_type == CycleType.TRANSCRITICAL:
            return 'R744'  # CO2 for transcritical cycles
        elif cycle_type == CycleType.ORC:
            return np.random.choice(['Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol'])
        else:
            # For Carnot and Rankine cycles, use reliable refrigerants
            return np.random.choice(self.reliable_refrigerants)
    
    def generate_batch_improved(self, cycle_type: CycleType, n_cycles: int, 
                              target_fluids: List[str] = None) -> List[CycleResults]:
        """
        Generate a batch of cycles with improved success rate
        """
        
        results = []
        fluids_to_use = target_fluids if target_fluids else [self.select_reliable_fluid(cycle_type) for _ in range(n_cycles)]
        
        for i in range(n_cycles):
            fluid = fluids_to_use[i % len(fluids_to_use)]
            
            # Skip problematic fluids
            if fluid in ['Air', 'Therminol_VP1', 'Dowtherm_A', 'Mineral_Oil']:
                continue
            
            cycle_result = self.generate_cycle_with_retry(cycle_type, fluid)
            if cycle_result:
                results.append(cycle_result)
        
        return results

def test_improved_generator():
    """Test the improved generator to verify higher success rate"""
    
    print("🧪 Testing Improved Cycle Generator")
    print("=" * 50)
    
    generator = ImprovedCycleGenerator()
    
    # Test different cycle types
    test_cases = [
        (CycleType.CARNOT, 20),
        (CycleType.IDEAL_RANKINE, 20),
        (CycleType.NON_IDEAL_RANKINE, 20)
    ]
    
    total_attempts = 0
    total_successes = 0
    
    for cycle_type, n_cycles in test_cases:
        print(f"\n🔄 Testing {cycle_type.value} cycles ({n_cycles} attempts)...")
        
        results = generator.generate_batch_improved(cycle_type, n_cycles)
        success_rate = len(results) / n_cycles * 100
        
        print(f"  ✅ Success rate: {success_rate:.1f}% ({len(results)}/{n_cycles})")
        
        if results:
            cops = [r.COP_heating for r in results]
            print(f"  📊 COP range: {min(cops):.2f} - {max(cops):.2f}")
            print(f"  ⚖️  Energy balance errors: {[f'{r.energy_balance_error:.3f}%' for r in results[:3]]}")
        
        total_attempts += n_cycles
        total_successes += len(results)
    
    overall_success_rate = total_successes / total_attempts * 100
    print(f"\n🎯 Overall Success Rate: {overall_success_rate:.1f}% ({total_successes}/{total_attempts})")
    print(f"🎯 Target: 80%+ - {'✅ ACHIEVED' if overall_success_rate >= 80 else '❌ NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    test_improved_generator()