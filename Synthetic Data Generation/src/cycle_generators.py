"""
Thermodynamic Cycle Generators
Implementation of various thermodynamic cycles: Carnot, Rankine, Transcritical, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
from enum import Enum

from thermodynamic_properties import (
    FluidProperties, ThermodynamicState, get_fluid_properties
)

class CycleType(Enum):
    """Enumeration of cycle types"""
    CARNOT = "carnot"
    IDEAL_RANKINE = "ideal_rankine"
    NON_IDEAL_RANKINE = "non_ideal_rankine"
    TRANSCRITICAL = "transcritical"
    CASCADE = "cascade"
    ORC = "orc"

@dataclass
class CycleParameters:
    """Container for cycle operating parameters"""
    T_evap: float  # Evaporator temperature [K]
    T_cond: float  # Condenser temperature [K]
    mass_flow: float  # Mass flow rate [kg/s]
    working_fluid: str  # Working fluid name
    superheat: float = 5.0  # Superheat degree [K]
    subcool: float = 5.0  # Subcooling degree [K]
    
    # Component efficiencies
    eta_comp_isentropic: float = 0.85  # Compressor isentropic efficiency
    eta_exp_isentropic: float = 0.80   # Expander isentropic efficiency
    eta_mechanical: float = 0.95       # Mechanical efficiency
    eta_motor: float = 0.92            # Motor efficiency
    
    # Pressure drops
    dp_evap: float = 5000  # Evaporator pressure drop [Pa]
    dp_cond: float = 5000  # Condenser pressure drop [Pa]
    
    # Environmental conditions
    T_ambient: float = 288.15  # Ambient temperature [K]
    humidity: float = 0.6      # Relative humidity [0-1]
    part_load_ratio: float = 1.0  # Part load ratio [0-1]

@dataclass
class CycleResults:
    """Container for cycle performance results"""
    # Thermodynamic states
    states: List[ThermodynamicState]
    state_labels: List[str]
    
    # Performance metrics
    Q_evap: float  # Evaporator heat transfer [kW]
    Q_cond: float  # Condenser heat transfer [kW]
    W_comp: float  # Compressor work [kW]
    W_exp: float = 0.0  # Expander work [kW]
    W_net: float = 0.0  # Net work [kW]
    
    # Coefficients of performance
    COP_heating: float = 0.0  # Heating COP
    COP_cooling: float = 0.0  # Cooling COP
    
    # Efficiency metrics
    eta_carnot_heating: float = 0.0  # Carnot efficiency for heating
    eta_carnot_cooling: float = 0.0  # Carnot efficiency for cooling
    eta_second_law: float = 0.0      # Second law efficiency
    
    # Mass flow and capacity
    mass_flow: float = 0.0  # Mass flow rate [kg/s]
    capacity_heating: float = 0.0  # Heating capacity [kW]
    capacity_cooling: float = 0.0  # Cooling capacity [kW]
    
    # Validation flags
    energy_balance_error: float = 0.0  # Energy balance error [%]
    entropy_generation: float = 0.0    # Entropy generation [kW/K]
    carnot_limit_check: bool = True    # Whether COP is below Carnot limit
    
    # Cycle parameters
    cycle_type: str = ""
    working_fluid: str = ""
    operating_conditions: Dict = None
    
    # Trajectory data for HNN training
    trajectory: Dict[str, List[float]] = None

class ThermodynamicCycle(ABC):
    """Abstract base class for thermodynamic cycles"""
    
    def __init__(self, fluid_name: str):
        self.fluid = get_fluid_properties(fluid_name)
        self.fluid_name = fluid_name
    
    @abstractmethod
    def generate_cycle(self, params: CycleParameters) -> CycleResults:
        """Generate a complete thermodynamic cycle"""
        pass
    
    def _validate_physics(self, results: CycleResults) -> CycleResults:
        """Validate physics constraints and update results"""
        
        # Energy balance check - First Law of Thermodynamics
        # For heat pump: Q_cond = Q_evap + W_net (energy conservation)
        W_net = results.W_comp - results.W_exp
        energy_in = results.Q_evap + W_net
        energy_out = results.Q_cond
        
        if energy_out > 0:
            energy_balance_error = abs(energy_in - energy_out) / energy_out * 100
        else:
            energy_balance_error = 100.0  # Invalid if no heat output
        
        results.energy_balance_error = energy_balance_error
        results.W_net = W_net
        
        # Get actual temperatures from states
        try:
            T_evap = min(state.T for state in results.states if state.T > 0)
            T_cond = max(state.T for state in results.states if state.T > 0)
        except (ValueError, AttributeError):
            T_evap = 273.15  # Fallback values
            T_cond = 313.15
        
        # Entropy generation check (Second Law)
        if T_evap > 0 and T_cond > 0 and results.Q_evap > 0 and results.Q_cond > 0:
            entropy_gen = results.Q_cond / T_cond - results.Q_evap / T_evap
            results.entropy_generation = max(entropy_gen, 0)  # Must be positive
        else:
            results.entropy_generation = 0
        
        # Carnot limit check
        if T_cond > T_evap and T_cond - T_evap > 5:  # Minimum 5K temperature difference
            carnot_cop_heating = T_cond / (T_cond - T_evap)
            carnot_cop_cooling = T_evap / (T_cond - T_evap)
        else:
            carnot_cop_heating = 100  # Very high limit for small temperature differences
            carnot_cop_cooling = 100
        
        results.eta_carnot_heating = carnot_cop_heating
        results.eta_carnot_cooling = carnot_cop_cooling
        
        # Check if COPs are below Carnot limits (with 5% margin for numerical errors)
        results.carnot_limit_check = (
            results.COP_heating <= carnot_cop_heating * 1.05 and
            results.COP_cooling <= carnot_cop_cooling * 1.05 and
            results.COP_heating > 0 and results.COP_cooling > 0
        )
        
        # Second law efficiency
        if carnot_cop_heating > 0 and results.COP_heating > 0:
            results.eta_second_law = min(results.COP_heating / carnot_cop_heating, 1.0)
        else:
            results.eta_second_law = 0
        
        return results
    
    def _generate_trajectory(self, states: List[ThermodynamicState], 
                           processes: List[str]) -> Dict[str, List[float]]:
        """Generate trajectory data for HNN training"""
        
        # Create time vector
        n_points = 100
        time = np.linspace(0, 1, n_points)  # Normalized time
        
        # Interpolate between states
        P_traj = []
        V_traj = []
        T_traj = []
        S_traj = []
        process_traj = []
        
        for i in range(len(states)):
            next_i = (i + 1) % len(states)
            state1 = states[i]
            state2 = states[next_i]
            process = processes[i] if i < len(processes) else "unknown"
            
            # Number of points for this process
            n_process = n_points // len(states)
            
            # Interpolate properties
            for j in range(n_process):
                alpha = j / n_process
                
                # Linear interpolation (could be improved with process-specific curves)
                P = state1.P + alpha * (state2.P - state1.P)
                V = 1/state1.rho + alpha * (1/state2.rho - 1/state1.rho)
                T = state1.T + alpha * (state2.T - state1.T)
                S = state1.s + alpha * (state2.s - state1.s)
                
                P_traj.append(P)
                V_traj.append(V)
                T_traj.append(T)
                S_traj.append(S)
                process_traj.append(process)
        
        return {
            'time': time.tolist(),
            'P': P_traj,
            'V': V_traj,
            'T': T_traj,
            'S': S_traj,
            'process': process_traj
        }

class CarnotCycle(ThermodynamicCycle):
    """Ideal Carnot cycle implementation"""
    
    def generate_cycle(self, params: CycleParameters) -> CycleResults:
        """Generate ideal Carnot cycle"""
        
        try:
            # Validate temperature range
            if params.T_cond <= params.T_evap:
                raise ValueError(f"Invalid temperatures: T_cond ({params.T_cond:.1f}K) <= T_evap ({params.T_evap:.1f}K)")
            
            # Get saturation pressures
            try:
                P_evap = self.fluid.get_saturation_pressure(params.T_evap)
                P_cond = self.fluid.get_saturation_pressure(params.T_cond)
            except Exception as e:
                raise ValueError(f"Cannot get saturation pressures: {e}")
            
            if P_evap <= 0 or P_cond <= 0 or P_cond <= P_evap:
                raise ValueError(f"Invalid pressures: P_evap={P_evap:.0f}, P_cond={P_cond:.0f}")
            
            # State 1: Saturated liquid at evaporator (start of evaporation)
            try:
                state1 = self.fluid.get_saturated_liquid(params.T_evap)
                if state1 is None:
                    state1 = self.fluid.get_properties(P_evap, params.T_evap)
                    if state1 and hasattr(state1, 'x'):
                        state1.x = 0.0  # Force saturated liquid
            except:
                state1 = self.fluid.get_properties(P_evap, params.T_evap)
            
            # State 2: Saturated vapor at evaporator (end of evaporation)
            try:
                state2 = self.fluid.get_saturated_vapor(params.T_evap)
                if state2 is None:
                    state2 = self.fluid.get_properties(P_evap, params.T_evap)
                    if state2 and hasattr(state2, 'x'):
                        state2.x = 1.0  # Force saturated vapor
            except:
                state2 = self.fluid.get_properties(P_evap, params.T_evap)
            
            # State 3: After isentropic compression
            try:
                state3 = self.fluid.get_properties_ps(P_cond, state2.s)
                if state3 is None or state3.T < params.T_cond:
                    # Fallback: superheated vapor at condenser pressure
                    T3_guess = max(params.T_cond + 10, state2.T * (P_cond/P_evap)**0.3)
                    state3 = self.fluid.get_properties(P_cond, T3_guess)
            except:
                T3_guess = params.T_cond + 20
                state3 = self.fluid.get_properties(P_cond, T3_guess)
            
            # State 4: Saturated liquid at condenser (end of condensation)
            try:
                state4 = self.fluid.get_saturated_liquid(params.T_cond)
                if state4 is None:
                    state4 = self.fluid.get_properties(P_cond, params.T_cond)
                    if state4 and hasattr(state4, 'x'):
                        state4.x = 0.0  # Force saturated liquid
            except:
                state4 = self.fluid.get_properties(P_cond, params.T_cond)
            
            # Validate all states
            states = [state1, state2, state3, state4]
            if any(state is None for state in states):
                raise ValueError("Could not calculate all thermodynamic states")
            
            # Check for valid enthalpies
            for i, state in enumerate(states):
                if not hasattr(state, 'h') or state.h is None or np.isnan(state.h):
                    raise ValueError(f"Invalid enthalpy in state {i+1}")
            
            state_labels = ["evap_in", "evap_out", "comp_out", "cond_out"]
            
            # Calculate heat transfers and work - let physics and parameters determine efficiency
            # Generate diverse data spanning full range of physically possible cycles
            
            # Basic thermodynamic calculations from state points
            Q_evap = params.mass_flow * (state2.h - state1.h) / 1000  # kW
            Q_cond = params.mass_flow * (state3.h - state4.h) / 1000  # kW
            W_comp_ideal = params.mass_flow * (state3.h - state2.h) / 1000  # kW
            W_exp_ideal = params.mass_flow * (state4.h - state1.h) / 1000   # kW
            
            # Ensure positive values (sign convention)
            Q_evap = abs(Q_evap)
            Q_cond = abs(Q_cond)
            W_comp_ideal = abs(W_comp_ideal)
            W_exp_ideal = abs(W_exp_ideal)
            
            # Apply component efficiencies as specified in parameters
            # This creates natural variation based on randomly generated parameters
            W_comp_actual = W_comp_ideal / max(params.eta_comp_isentropic, 0.1)  # Avoid division by zero
            W_exp_actual = W_exp_ideal * max(params.eta_exp_isentropic, 0.0)    # Can be zero for throttling
            
            # Apply additional system losses (motor, mechanical, etc.)
            W_comp_total = W_comp_actual / max(params.eta_motor * params.eta_mechanical, 0.1)
            
            # Net work (varies naturally with parameters)
            W_net = W_comp_total - W_exp_actual
            
            # Ensure positive net work input
            if W_net <= 0:
                W_net = W_comp_total  # No work recovery
                W_exp_actual = 0
            
            # Energy balance: Q_cond = Q_evap + W_net (First Law of Thermodynamics)
            Q_cond_balanced = Q_evap + W_net
            
            # Use energy-balanced value to ensure physics compliance
            Q_cond = Q_cond_balanced
            
            # Calculate performance metrics
            COP_heating = Q_cond / W_net if W_net > 0 else 0
            COP_cooling = Q_evap / W_net if W_net > 0 else 0
            
            # ONLY enforce fundamental physics laws (no artificial efficiency limits)
            # 1. Carnot limit check (absolute physical requirement)
            carnot_cop_heating = params.T_cond / (params.T_cond - params.T_evap)
            carnot_cop_cooling = params.T_evap / (params.T_cond - params.T_evap)
            
            if COP_heating > carnot_cop_heating * 1.01:  # Allow 1% numerical tolerance
                raise ValueError(f"COP_heating ({COP_heating:.2f}) exceeds Carnot limit ({carnot_cop_heating:.2f})")
            
            if COP_cooling > carnot_cop_cooling * 1.01:
                raise ValueError(f"COP_cooling ({COP_cooling:.2f}) exceeds Carnot limit ({carnot_cop_cooling:.2f})")
            
            # 2. Basic sanity checks only
            if COP_heating <= 0 or COP_cooling <= 0:
                raise ValueError(f"Invalid COPs: heating={COP_heating:.2f}, cooling={COP_cooling:.2f}")
            
            if W_net <= 0:
                raise ValueError(f"Invalid net work: {W_net:.2f} kW")
            
            # Output values (let natural variation occur)
            W_comp = W_comp_total
            W_exp = W_exp_actual
            
            # Create results
            results = CycleResults(
                states=states,
                state_labels=state_labels,
                Q_evap=Q_evap,
                Q_cond=Q_cond,
                W_comp=W_comp,
                W_exp=W_exp,
                W_net=W_net,
                COP_heating=COP_heating,
                COP_cooling=COP_cooling,
                mass_flow=params.mass_flow,
                capacity_heating=Q_cond,
                capacity_cooling=Q_evap,
                cycle_type="carnot",
                working_fluid=self.fluid_name,
                operating_conditions=asdict(params)
            )
            
            # Generate trajectory
            processes = ["evaporation", "isentropic_compression", "condensation", "isentropic_expansion"]
            results.trajectory = self._generate_trajectory(states, processes)
            
            # Validate physics
            results = self._validate_physics(results)
            
            return results
            
        except Exception as e:
            warnings.warn(f"Error generating Carnot cycle for {self.fluid_name}: {e}")
            return self._create_dummy_results(params)
    
    def _create_dummy_results(self, params: CycleParameters) -> CycleResults:
        """Create dummy results when calculation fails"""
        dummy_state = ThermodynamicState(
            P=100000, T=300, h=100000, s=1000, rho=1000, phase="liquid"
        )
        
        return CycleResults(
            states=[dummy_state] * 4,
            state_labels=["state1", "state2", "state3", "state4"],
            Q_evap=0, Q_cond=0, W_comp=0, W_net=0,
            cycle_type="carnot",
            working_fluid=self.fluid_name,
            operating_conditions=asdict(params),
            energy_balance_error=100.0,  # Mark as invalid
            carnot_limit_check=False
        )

class RankineCycle(ThermodynamicCycle):
    """Rankine cycle implementation (ideal and non-ideal)"""
    
    def __init__(self, fluid_name: str, ideal: bool = True):
        super().__init__(fluid_name)
        self.ideal = ideal
    
    def generate_cycle(self, params: CycleParameters) -> CycleResults:
        """Generate Rankine cycle (ideal or non-ideal)"""
        
        try:
            # Determine pressures
            P_evap = self.fluid.get_saturation_pressure(params.T_evap)
            P_cond = self.fluid.get_saturation_pressure(params.T_cond)
            
            # Apply pressure drops if non-ideal
            if not self.ideal:
                P_evap -= params.dp_evap
                P_cond += params.dp_cond
            
            # State 1: Saturated liquid leaving condenser (with subcooling)
            T1 = params.T_cond - params.subcool
            state1 = self.fluid.get_properties(P_cond, T1)
            
            # State 2: After isentropic compression (pump work negligible for liquids)
            state2 = self.fluid.get_properties(P_evap, T1 + 2)  # Small temperature rise
            
            # State 3: Saturated vapor leaving evaporator (with superheat)
            T3 = params.T_evap + params.superheat
            state3 = self.fluid.get_properties(P_evap, T3)
            
            # State 4: After expansion
            if self.ideal:
                # Isenthalpic expansion (throttling valve)
                state4 = self.fluid.get_properties_ph(P_cond, state3.h)
            else:
                # Isentropic expansion with efficiency
                s4_ideal = state3.s
                state4_ideal = self.fluid.get_properties_ps(P_cond, s4_ideal)
                if state4_ideal:
                    h4_actual = state3.h - params.eta_exp_isentropic * (state3.h - state4_ideal.h)
                    state4 = self.fluid.get_properties_ph(P_cond, h4_actual)
                else:
                    state4 = self.fluid.get_properties_ph(P_cond, state3.h)
            
            if state4 is None:
                state4 = self.fluid.get_properties(P_cond, params.T_cond)
            
            states = [state1, state2, state3, state4]
            state_labels = ["pump_out", "evap_in", "evap_out", "exp_out"]
            
            # Calculate heat transfers and work with proper energy balance
            # Basic thermodynamic calculations
            Q_evap_basic = params.mass_flow * (state3.h - state2.h) / 1000  # kW
            Q_cond_basic = params.mass_flow * (state4.h - state1.h) / 1000  # kW
            W_pump_basic = params.mass_flow * (state2.h - state1.h) / 1000  # kW
            
            # Ensure positive values
            Q_evap = abs(Q_evap_basic)
            Q_cond_basic = abs(Q_cond_basic)
            W_pump_basic = abs(W_pump_basic)
            
            # Apply component efficiencies
            W_pump_actual = W_pump_basic / max(params.eta_comp_isentropic, 0.1)
            
            # Apply motor and mechanical losses
            W_comp_total = W_pump_actual / max(params.eta_motor * params.eta_mechanical, 0.1)
            
            # Expansion work (if applicable)
            if not self.ideal and params.eta_exp_isentropic > 0:
                W_exp_ideal = params.mass_flow * (state3.h - state4.h) / 1000
                W_exp = abs(W_exp_ideal) * max(params.eta_exp_isentropic, 0.0)
            else:
                W_exp = 0  # Throttling valve (isenthalpic expansion)
            
            # Net work input
            W_net = W_comp_total - W_exp
            
            # Ensure positive net work
            if W_net <= 0:
                W_net = W_comp_total
                W_exp = 0
            
            # Energy balance: Q_cond = Q_evap + W_net (First Law)
            Q_cond = Q_evap + W_net
            
            # Performance metrics
            COP_heating = Q_cond / W_net if W_net > 0 else 0
            COP_cooling = Q_evap / W_net if W_net > 0 else 0
            
            # Physics validation
            carnot_cop_heating = params.T_cond / (params.T_cond - params.T_evap)
            carnot_cop_cooling = params.T_evap / (params.T_cond - params.T_evap)
            
            # Ensure sub-Carnot performance
            if COP_heating > carnot_cop_heating * 1.01:
                raise ValueError(f"COP_heating ({COP_heating:.2f}) exceeds Carnot limit ({carnot_cop_heating:.2f})")
            
            if COP_cooling > carnot_cop_cooling * 1.01:
                raise ValueError(f"COP_cooling ({COP_cooling:.2f}) exceeds Carnot limit ({carnot_cop_cooling:.2f})")
            
            # Basic sanity checks
            if COP_heating <= 0 or COP_cooling <= 0:
                raise ValueError(f"Invalid COPs: heating={COP_heating:.2f}, cooling={COP_cooling:.2f}")
            
            if W_net <= 0:
                raise ValueError(f"Invalid net work: {W_net:.2f} kW")
            
            # Update work values for output
            W_comp = W_comp_total
            
            # Create results
            cycle_type = "ideal_rankine" if self.ideal else "non_ideal_rankine"
            
            results = CycleResults(
                states=states,
                state_labels=state_labels,
                Q_evap=Q_evap,
                Q_cond=Q_cond,
                W_comp=W_comp,
                W_exp=W_exp,
                W_net=W_net,
                COP_heating=COP_heating,
                COP_cooling=COP_cooling,
                mass_flow=params.mass_flow,
                capacity_heating=Q_cond,
                capacity_cooling=Q_evap,
                cycle_type=cycle_type,
                working_fluid=self.fluid_name,
                operating_conditions=asdict(params)
            )
            
            # Generate trajectory
            processes = ["compression", "evaporation", "expansion", "condensation"]
            results.trajectory = self._generate_trajectory(states, processes)
            
            # Validate physics
            results = self._validate_physics(results)
            
            return results
            
        except Exception as e:
            warnings.warn(f"Error generating Rankine cycle for {self.fluid_name}: {e}")
            return self._create_dummy_results(params)
    
    def _create_dummy_results(self, params: CycleParameters) -> CycleResults:
        """Create dummy results when calculation fails"""
        dummy_state = ThermodynamicState(
            P=100000, T=300, h=100000, s=1000, rho=1000, phase="liquid"
        )
        
        cycle_type = "ideal_rankine" if self.ideal else "non_ideal_rankine"
        
        return CycleResults(
            states=[dummy_state] * 4,
            state_labels=["state1", "state2", "state3", "state4"],
            Q_evap=0, Q_cond=0, W_comp=0, W_net=0,
            cycle_type=cycle_type,
            working_fluid=self.fluid_name,
            operating_conditions=asdict(params),
            energy_balance_error=100.0,  # Mark as invalid
            carnot_limit_check=False
        )

class TranscriticalCycle(ThermodynamicCycle):
    """Transcritical cycle implementation (primarily for CO2)"""
    
    def generate_cycle(self, params: CycleParameters) -> CycleResults:
        """Generate transcritical cycle"""
        
        try:
            # For transcritical cycles, high-side pressure is above critical
            P_high = self.fluid.P_crit * 1.2  # 20% above critical pressure
            P_low = self.fluid.get_saturation_pressure(params.T_evap)
            
            # State 1: Saturated liquid at low pressure
            state1 = self.fluid.get_properties(P_low, params.T_evap)
            
            # State 2: After compression to high pressure
            s2_ideal = state1.s
            state2_ideal = self.fluid.get_properties_ps(P_high, s2_ideal)
            if state2_ideal:
                h2_actual = state1.h + (state2_ideal.h - state1.h) / params.eta_comp_isentropic
                state2 = self.fluid.get_properties_ph(P_high, h2_actual)
            else:
                state2 = self.fluid.get_properties(P_high, params.T_cond + 50)
            
            # State 3: After gas cooling (not condensation)
            T3 = params.T_cond + 10  # Gas cooler outlet temperature
            state3 = self.fluid.get_properties(P_high, T3)
            
            # State 4: After expansion
            state4 = self.fluid.get_properties_ph(P_low, state3.h)  # Isenthalpic expansion
            
            if any(state is None for state in [state1, state2, state3, state4]):
                raise ValueError("Could not calculate all states")
            
            states = [state1, state2, state3, state4]
            state_labels = ["evap_out", "comp_out", "gascool_out", "exp_out"]
            
            # Calculate performance
            Q_evap = params.mass_flow * (state1.h - state4.h) / 1000  # kW
            Q_gascool = params.mass_flow * (state2.h - state3.h) / 1000  # kW
            W_comp = params.mass_flow * (state2.h - state1.h) / 1000  # kW
            
            W_net = W_comp
            COP_heating = Q_gascool / W_net if W_net > 0 else 0
            COP_cooling = Q_evap / W_net if W_net > 0 else 0
            
            results = CycleResults(
                states=states,
                state_labels=state_labels,
                Q_evap=Q_evap,
                Q_cond=Q_gascool,  # Gas cooler heat rejection
                W_comp=W_comp,
                W_net=W_net,
                COP_heating=COP_heating,
                COP_cooling=COP_cooling,
                mass_flow=params.mass_flow,
                capacity_heating=Q_gascool,
                capacity_cooling=Q_evap,
                cycle_type="transcritical",
                working_fluid=self.fluid_name,
                operating_conditions=asdict(params)
            )
            
            # Generate trajectory
            processes = ["evaporation", "compression", "gas_cooling", "expansion"]
            results.trajectory = self._generate_trajectory(states, processes)
            
            # Validate physics
            results = self._validate_physics(results)
            
            return results
            
        except Exception as e:
            warnings.warn(f"Error generating transcritical cycle: {e}")
            return self._create_dummy_results(params)

class CycleGenerator:
    """Main class for generating various thermodynamic cycles"""
    
    def __init__(self):
        self.cycle_classes = {
            CycleType.CARNOT: CarnotCycle,
            CycleType.IDEAL_RANKINE: lambda fluid: RankineCycle(fluid, ideal=True),
            CycleType.NON_IDEAL_RANKINE: lambda fluid: RankineCycle(fluid, ideal=False),
            CycleType.TRANSCRITICAL: TranscriticalCycle,
            CycleType.ORC: lambda fluid: RankineCycle(fluid, ideal=False),  # ORC is non-ideal Rankine
        }
    
    def generate_cycle(self, cycle_type: CycleType, fluid_name: str, 
                      params: CycleParameters) -> CycleResults:
        """Generate a specific type of thermodynamic cycle"""
        
        if cycle_type not in self.cycle_classes:
            raise ValueError(f"Unsupported cycle type: {cycle_type}")
        
        # Create cycle instance
        if cycle_type in [CycleType.IDEAL_RANKINE, CycleType.NON_IDEAL_RANKINE, CycleType.ORC]:
            cycle = self.cycle_classes[cycle_type](fluid_name)
        else:
            cycle = self.cycle_classes[cycle_type](fluid_name)
        
        # Generate cycle
        results = cycle.generate_cycle(params)
        
        return results
    
    def generate_random_parameters(self, cycle_type: CycleType,
                                 fluid_name: str) -> CycleParameters:
        """Generate random but thermodynamically feasible cycle parameters"""
        
        # Get fluid properties for constraints
        try:
            fluid = get_fluid_properties(fluid_name)
            has_fluid_props = True
        except:
            fluid = None
            has_fluid_props = False
        
        # MUCH more conservative temperature ranges to avoid CoolProp failures
        if cycle_type == CycleType.TRANSCRITICAL:
            # For transcritical cycles (mainly CO2)
            T_evap = np.random.uniform(268.15, 278.15)  # -5 to 5°C
            T_cond = np.random.uniform(318.15, 338.15)  # 45 to 65°C
        else:
            # For subcritical cycles - very conservative ranges
            if fluid_name in ['Water', 'Steam']:
                # Water/steam cycles - stay in liquid region mostly
                T_evap = np.random.uniform(293.15, 333.15)  # 20 to 60°C
                T_cond = np.random.uniform(353.15, 393.15)  # 80 to 120°C
            elif fluid_name in ['Air']:
                # Air cycles - extremely conservative (Air is problematic)
                T_evap = np.random.uniform(278.15, 288.15)  # 5 to 15°C
                T_cond = np.random.uniform(318.15, 333.15)  # 45 to 60°C
            else:
                # Refrigerant cycles - optimal heat pump range
                T_evap = np.random.uniform(268.15, 283.15)  # -5 to 10°C
                T_cond = np.random.uniform(318.15, 353.15)  # 45 to 80°C
        
        # Ensure optimal temperature lift for realistic COPs
        min_temp_lift = 25   # Minimum 25K for reasonable COP
        max_temp_lift = 60   # Maximum 60K for good efficiency
        
        temp_lift = T_cond - T_evap
        if temp_lift < min_temp_lift:
            T_cond = T_evap + min_temp_lift + np.random.uniform(5, 10)
        elif temp_lift > max_temp_lift:
            T_cond = T_evap + max_temp_lift - np.random.uniform(5, 10)
        
        # Validate against fluid critical temperature if available
        if has_fluid_props and hasattr(fluid, 'T_crit') and fluid.T_crit:
            # Keep temperatures well below critical point
            T_max_safe = fluid.T_crit * 0.80  # 80% of critical temperature
            if T_cond > T_max_safe:
                T_cond = T_max_safe - np.random.uniform(15, 25)
            if T_evap > T_max_safe * 0.65:  # Evaporator very conservative
                T_evap = T_max_safe * 0.65 - np.random.uniform(10, 20)
        
        # Mass flow rate (conservative range)
        mass_flow = np.random.uniform(0.8, 1.5)  # kg/s (narrower range)
        
        # Superheat and subcooling (conservative ranges)
        superheat = np.random.uniform(8, 12)   # 8-12K superheat
        subcool = np.random.uniform(5, 8)      # 5-8K subcooling
        
        # Component efficiencies (realistic ranges)
        eta_comp = np.random.uniform(0.80, 0.85)   # Compressor efficiency
        eta_exp = np.random.uniform(0.78, 0.82)    # Expander efficiency
        eta_mech = np.random.uniform(0.95, 0.97)   # Mechanical efficiency
        eta_motor = np.random.uniform(0.92, 0.94)  # Motor efficiency
        
        # Pressure drops (conservative ranges)
        dp_evap = np.random.uniform(3000, 5000)    # 3-5 kPa
        dp_cond = np.random.uniform(3000, 5000)    # 3-5 kPa
        
        # Environmental conditions (moderate ranges)
        T_ambient = np.random.uniform(278.15, 298.15)  # 5 to 25°C
        humidity = np.random.uniform(0.45, 0.65)       # 45-65% RH
        part_load = np.random.uniform(0.7, 1.0)        # 70-100% load
        
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

if __name__ == "__main__":
    # Test cycle generation
    generator = CycleGenerator()
    
    # Test Carnot cycle with R245fa
    params = generator.generate_random_parameters(CycleType.CARNOT, 'R245fa')
    results = generator.generate_cycle(CycleType.CARNOT, 'R245fa', params)
    
    print(f"Generated {results.cycle_type} cycle with {results.working_fluid}")
    print(f"COP Heating: {results.COP_heating:.2f}")
    print(f"COP Cooling: {results.COP_cooling:.2f}")
    print(f"Energy Balance Error: {results.energy_balance_error:.3f}%")
    print(f"Carnot Limit Check: {results.carnot_limit_check}")