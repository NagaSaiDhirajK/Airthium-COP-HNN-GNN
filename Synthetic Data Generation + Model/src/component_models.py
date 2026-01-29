"""
Component Models
Detailed models for heat pump components: compressors, heat exchangers, expansion devices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from thermodynamic_properties import ThermodynamicState, FluidProperties

class CompressorType(Enum):
    """Types of compressors"""
    RECIPROCATING = "reciprocating"
    SCROLL = "scroll"
    SCREW = "screw"
    CENTRIFUGAL = "centrifugal"

class HeatExchangerType(Enum):
    """Types of heat exchangers"""
    SHELL_TUBE = "shell_tube"
    PLATE = "plate"
    FINNED_TUBE = "finned_tube"
    FLOODED_EVAPORATOR = "flooded_evaporator"
    DX_EVAPORATOR = "dx_evaporator"

class ExpansionType(Enum):
    """Types of expansion devices"""
    THROTTLING_VALVE = "throttling_valve"
    EXPANDER = "expander"
    TURBINE = "turbine"

@dataclass
class CompressorPerformance:
    """Compressor performance data"""
    W_comp: float  # Compressor power [kW]
    eta_isentropic: float  # Isentropic efficiency
    eta_volumetric: float  # Volumetric efficiency
    eta_mechanical: float  # Mechanical efficiency
    eta_motor: float  # Motor efficiency
    mass_flow_actual: float  # Actual mass flow [kg/s]
    discharge_temp: float  # Discharge temperature [K]
    pressure_ratio: float  # Pressure ratio
    displacement: float  # Displacement volume [m³/s]
    rpm: float  # Rotational speed [rpm]
    oil_temp: float  # Oil temperature [K]
    vibration_level: float  # Vibration level [mm/s]

@dataclass
class HeatExchangerPerformance:
    """Heat exchanger performance data"""
    Q_transfer: float  # Heat transfer rate [kW]
    U_overall: float  # Overall heat transfer coefficient [W/(m²·K)]
    A_heat_transfer: float  # Heat transfer area [m²]
    LMTD: float  # Log mean temperature difference [K]
    effectiveness: float  # Heat exchanger effectiveness
    NTU: float  # Number of transfer units
    dp_hot: float  # Hot side pressure drop [Pa]
    dp_cold: float  # Cold side pressure drop [Pa]
    fouling_factor: float  # Fouling resistance [m²·K/W]
    approach_temp: float  # Approach temperature [K]

@dataclass
class ExpansionPerformance:
    """Expansion device performance data"""
    W_exp: float  # Expansion work [kW] (0 for throttling valve)
    eta_isentropic: float  # Isentropic efficiency (for expanders)
    dp_expansion: float  # Pressure drop [Pa]
    flow_coefficient: float  # Flow coefficient
    cavitation_factor: float  # Cavitation factor (0-1)

class CompressorModel:
    """Detailed compressor model with performance maps"""
    
    def __init__(self, compressor_type: CompressorType, displacement: float = 0.1):
        self.type = compressor_type
        self.displacement = displacement  # m³/s at design conditions
        self.design_rpm = 1800  # rpm
        self.clearance_ratio = 0.05  # Clearance volume ratio
        
        # Performance map coefficients (simplified)
        self.eta_coeffs = self._get_efficiency_coefficients()
        
    def _get_efficiency_coefficients(self) -> Dict[str, float]:
        """Get efficiency correlation coefficients based on compressor type"""
        coefficients = {
            CompressorType.RECIPROCATING: {
                'eta_base': 0.75, 'pr_coeff': -0.05, 'flow_coeff': 0.1
            },
            CompressorType.SCROLL: {
                'eta_base': 0.85, 'pr_coeff': -0.03, 'flow_coeff': 0.05
            },
            CompressorType.SCREW: {
                'eta_base': 0.80, 'pr_coeff': -0.04, 'flow_coeff': 0.08
            },
            CompressorType.CENTRIFUGAL: {
                'eta_base': 0.78, 'pr_coeff': -0.06, 'flow_coeff': 0.12
            }
        }
        return coefficients[self.type]
    
    def calculate_performance(self, state_in: ThermodynamicState, 
                            state_out: ThermodynamicState,
                            mass_flow: float, rpm: float = None) -> CompressorPerformance:
        """Calculate detailed compressor performance"""
        
        if rpm is None:
            rpm = self.design_rpm
            
        # Pressure ratio
        pressure_ratio = state_out.P / state_in.P
        
        # Flow ratio (actual vs design)
        flow_ratio = mass_flow / (self.displacement * state_in.rho)
        
        # Isentropic efficiency correlation
        eta_base = self.eta_coeffs['eta_base']
        pr_effect = self.eta_coeffs['pr_coeff'] * (pressure_ratio - 3.0)
        flow_effect = self.eta_coeffs['flow_coeff'] * (flow_ratio - 1.0)
        eta_isentropic = eta_base + pr_effect + flow_effect
        eta_isentropic = np.clip(eta_isentropic, 0.5, 0.95)
        
        # Volumetric efficiency
        eta_volumetric = self._calculate_volumetric_efficiency(pressure_ratio, rpm)
        
        # Mechanical efficiency
        eta_mechanical = 0.95 - 0.02 * (pressure_ratio - 2.0)
        eta_mechanical = np.clip(eta_mechanical, 0.85, 0.98)
        
        # Motor efficiency (load dependent)
        load_factor = mass_flow / (self.displacement * state_in.rho)
        eta_motor = self._calculate_motor_efficiency(load_factor)
        
        # Actual mass flow considering volumetric efficiency
        mass_flow_actual = mass_flow * eta_volumetric
        
        # Compressor work
        h_out_isentropic = state_in.h + (state_out.h - state_in.h) / eta_isentropic
        W_comp = mass_flow_actual * (h_out_isentropic - state_in.h) / 1000  # kW
        
        # Discharge temperature (with heat addition from inefficiencies)
        discharge_temp = state_out.T + (1 - eta_isentropic) * 20  # Approximate
        
        # Oil temperature (for lubricated compressors)
        oil_temp = discharge_temp - 30 + np.random.uniform(-5, 5)
        
        # Vibration level (depends on type and condition)
        base_vibration = {'reciprocating': 8, 'scroll': 3, 'screw': 5, 'centrifugal': 2}
        vibration = base_vibration.get(self.type.value, 5) * (1 + 0.1 * (pressure_ratio - 3))
        vibration += np.random.uniform(-1, 1)  # Random variation
        
        return CompressorPerformance(
            W_comp=W_comp,
            eta_isentropic=eta_isentropic,
            eta_volumetric=eta_volumetric,
            eta_mechanical=eta_mechanical,
            eta_motor=eta_motor,
            mass_flow_actual=mass_flow_actual,
            discharge_temp=discharge_temp,
            pressure_ratio=pressure_ratio,
            displacement=self.displacement,
            rpm=rpm,
            oil_temp=oil_temp,
            vibration_level=vibration
        )
    
    def _calculate_volumetric_efficiency(self, pressure_ratio: float, rpm: float) -> float:
        """Calculate volumetric efficiency"""
        # Clearance volume effect
        clearance_effect = 1 - self.clearance_ratio * (pressure_ratio**(1/1.3) - 1)
        
        # Speed effect
        speed_ratio = rpm / self.design_rpm
        speed_effect = 1 - 0.05 * abs(speed_ratio - 1)
        
        eta_vol = clearance_effect * speed_effect
        return np.clip(eta_vol, 0.6, 0.95)
    
    def _calculate_motor_efficiency(self, load_factor: float) -> float:
        """Calculate motor efficiency based on load"""
        # Typical motor efficiency curve
        if load_factor < 0.25:
            eta = 0.75 + 0.13 * load_factor
        elif load_factor < 0.75:
            eta = 0.88 + 0.08 * (load_factor - 0.25)
        else:
            eta = 0.92 + 0.03 * (load_factor - 0.75) - 0.02 * (load_factor - 1)**2
        
        return np.clip(eta, 0.75, 0.95)

class HeatExchangerModel:
    """Detailed heat exchanger model"""
    
    def __init__(self, hx_type: HeatExchangerType, area: float = 50.0):
        self.type = hx_type
        self.area = area  # m²
        self.fouling_factor = 0.0002  # m²·K/W (typical)
        
        # Heat transfer correlations
        self.correlations = self._get_heat_transfer_correlations()
    
    def _get_heat_transfer_correlations(self) -> Dict[str, Dict]:
        """Get heat transfer correlation parameters"""
        return {
            HeatExchangerType.SHELL_TUBE: {
                'Nu_shell': {'C': 0.36, 'n': 0.55, 'm': 0.33},
                'Nu_tube': {'C': 0.023, 'n': 0.8, 'm': 0.4},
                'f_shell': 0.25, 'f_tube': 0.02
            },
            HeatExchangerType.PLATE: {
                'Nu': {'C': 0.2, 'n': 0.65, 'm': 0.4},
                'f': 0.35
            },
            HeatExchangerType.FINNED_TUBE: {
                'Nu_air': {'C': 0.27, 'n': 0.63, 'm': 0.36},
                'Nu_tube': {'C': 0.023, 'n': 0.8, 'm': 0.4},
                'fin_efficiency': 0.85
            }
        }
    
    def calculate_performance(self, state_hot_in: ThermodynamicState,
                            state_hot_out: ThermodynamicState,
                            state_cold_in: ThermodynamicState,
                            state_cold_out: ThermodynamicState,
                            mass_flow_hot: float,
                            mass_flow_cold: float) -> HeatExchangerPerformance:
        """Calculate heat exchanger performance"""
        
        # Heat transfer rates
        Q_hot = mass_flow_hot * (state_hot_in.h - state_hot_out.h) / 1000  # kW
        Q_cold = mass_flow_cold * (state_cold_out.h - state_cold_in.h) / 1000  # kW
        Q_transfer = (Q_hot + Q_cold) / 2  # Average (should be equal ideally)
        
        # Temperature differences
        dT1 = state_hot_in.T - state_cold_out.T
        dT2 = state_hot_out.T - state_cold_in.T
        
        # Log mean temperature difference
        if abs(dT1 - dT2) < 0.1:
            LMTD = (dT1 + dT2) / 2
        else:
            LMTD = (dT1 - dT2) / np.log(dT1 / dT2) if dT1 > 0 and dT2 > 0 else 1.0
        
        # Overall heat transfer coefficient
        U_overall = Q_transfer * 1000 / (self.area * LMTD) if LMTD > 0 else 1000
        
        # Heat capacity rates
        Cp_hot = state_hot_in.cp if state_hot_in.cp else 2000  # J/(kg·K)
        Cp_cold = state_cold_in.cp if state_cold_in.cp else 4180  # J/(kg·K)
        
        C_hot = mass_flow_hot * Cp_hot / 1000  # kW/K
        C_cold = mass_flow_cold * Cp_cold / 1000  # kW/K
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max if C_max > 0 else 1
        
        # Number of transfer units
        NTU = U_overall * self.area / (C_min * 1000) if C_min > 0 else 1
        
        # Effectiveness
        effectiveness = self._calculate_effectiveness(NTU, C_ratio)
        
        # Pressure drops
        dp_hot = self._calculate_pressure_drop(mass_flow_hot, state_hot_in, "hot")
        dp_cold = self._calculate_pressure_drop(mass_flow_cold, state_cold_in, "cold")
        
        # Approach temperature
        approach_temp = min(dT1, dT2)
        
        return HeatExchangerPerformance(
            Q_transfer=Q_transfer,
            U_overall=U_overall,
            A_heat_transfer=self.area,
            LMTD=LMTD,
            effectiveness=effectiveness,
            NTU=NTU,
            dp_hot=dp_hot,
            dp_cold=dp_cold,
            fouling_factor=self.fouling_factor,
            approach_temp=approach_temp
        )
    
    def _calculate_effectiveness(self, NTU: float, C_ratio: float) -> float:
        """Calculate heat exchanger effectiveness"""
        if self.type == HeatExchangerType.SHELL_TUBE:
            # Counterflow approximation
            if C_ratio < 0.99:
                eff = (1 - np.exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * np.exp(-NTU * (1 - C_ratio)))
            else:
                eff = NTU / (1 + NTU)
        else:
            # Parallel flow approximation
            eff = (1 - np.exp(-NTU * (1 + C_ratio))) / (1 + C_ratio)
        
        return np.clip(eff, 0, 1)
    
    def _calculate_pressure_drop(self, mass_flow: float, state: ThermodynamicState, 
                               side: str) -> float:
        """Calculate pressure drop"""
        # Simplified pressure drop calculation
        velocity = mass_flow / (state.rho * 0.01)  # Assume 0.01 m² flow area
        
        # Friction factor (simplified)
        Re = state.rho * velocity * 0.02 / (state.mu if state.mu else 1e-3)
        if Re < 2300:
            f = 64 / Re
        else:
            f = 0.316 / (Re**0.25)
        
        # Pressure drop
        dp = f * (velocity**2 * state.rho / 2) * 10  # Assume L/D = 10
        
        # Add form losses
        dp += 0.5 * velocity**2 * state.rho  # K = 0.5 for bends, etc.
        
        return dp

class ExpansionDeviceModel:
    """Model for expansion devices"""
    
    def __init__(self, expansion_type: ExpansionType):
        self.type = expansion_type
        
        # Device-specific parameters
        if expansion_type == ExpansionType.THROTTLING_VALVE:
            self.flow_coefficient = 0.6
        elif expansion_type == ExpansionType.EXPANDER:
            self.eta_isentropic = 0.80
        elif expansion_type == ExpansionType.TURBINE:
            self.eta_isentropic = 0.85
    
    def calculate_performance(self, state_in: ThermodynamicState,
                            state_out: ThermodynamicState,
                            mass_flow: float) -> ExpansionPerformance:
        """Calculate expansion device performance"""
        
        dp_expansion = state_in.P - state_out.P
        
        if self.type == ExpansionType.THROTTLING_VALVE:
            # Isenthalpic expansion
            W_exp = 0.0
            eta_isentropic = 0.0
            
            # Flow coefficient calculation
            flow_coeff = mass_flow / np.sqrt(state_in.rho * dp_expansion)
            
            # Cavitation check
            P_vapor = 0.1 * state_in.P  # Simplified vapor pressure
            cavitation_factor = max(0, (state_out.P - P_vapor) / (state_in.P - P_vapor))
            
        else:
            # Expansion with work recovery
            eta_isentropic = self.eta_isentropic
            
            # Ideal expansion work
            W_ideal = mass_flow * (state_in.h - state_out.h) / 1000  # kW
            W_exp = W_ideal * eta_isentropic
            
            flow_coeff = 1.0  # Not applicable for expanders
            cavitation_factor = 1.0  # No cavitation in expanders
        
        return ExpansionPerformance(
            W_exp=W_exp,
            eta_isentropic=eta_isentropic,
            dp_expansion=dp_expansion,
            flow_coefficient=flow_coeff,
            cavitation_factor=cavitation_factor
        )

class ComponentIntegration:
    """Integration of all component models"""
    
    def __init__(self):
        self.compressor_models = {}
        self.heat_exchanger_models = {}
        self.expansion_models = {}
    
    def add_compressor(self, name: str, comp_type: CompressorType, 
                      displacement: float = 0.1):
        """Add a compressor model"""
        self.compressor_models[name] = CompressorModel(comp_type, displacement)
    
    def add_heat_exchanger(self, name: str, hx_type: HeatExchangerType, 
                          area: float = 50.0):
        """Add a heat exchanger model"""
        self.heat_exchanger_models[name] = HeatExchangerModel(hx_type, area)
    
    def add_expansion_device(self, name: str, exp_type: ExpansionType):
        """Add an expansion device model"""
        self.expansion_models[name] = ExpansionDeviceModel(exp_type)
    
    def get_component_performance(self, component_name: str, 
                                component_type: str, **kwargs) -> Any:
        """Get performance data for any component"""
        
        if component_type == "compressor":
            model = self.compressor_models.get(component_name)
            if model:
                return model.calculate_performance(**kwargs)
        
        elif component_type == "heat_exchanger":
            model = self.heat_exchanger_models.get(component_name)
            if model:
                return model.calculate_performance(**kwargs)
        
        elif component_type == "expansion":
            model = self.expansion_models.get(component_name)
            if model:
                return model.calculate_performance(**kwargs)
        
        return None
    
    def generate_component_maps(self, component_name: str, 
                              component_type: str) -> Dict[str, np.ndarray]:
        """Generate performance maps for components"""
        
        if component_type == "compressor" and component_name in self.compressor_models:
            return self._generate_compressor_map(component_name)
        
        return {}
    
    def _generate_compressor_map(self, compressor_name: str) -> Dict[str, np.ndarray]:
        """Generate compressor performance map"""
        
        model = self.compressor_models[compressor_name]
        
        # Create parameter grids
        pressure_ratios = np.linspace(2, 8, 20)
        flow_ratios = np.linspace(0.5, 1.5, 15)
        
        # Initialize result arrays
        efficiency_map = np.zeros((len(pressure_ratios), len(flow_ratios)))
        power_map = np.zeros((len(pressure_ratios), len(flow_ratios)))
        
        # Generate map data
        for i, pr in enumerate(pressure_ratios):
            for j, fr in enumerate(flow_ratios):
                # Create dummy states for map generation
                P_in = 300000  # 3 bar
                P_out = P_in * pr
                T_in = 280  # 7°C
                T_out = T_in * (pr**0.286)  # Isentropic relation
                
                # Dummy states (simplified)
                state_in = type('State', (), {
                    'P': P_in, 'T': T_in, 'h': 200000, 's': 1000, 'rho': 10
                })()
                state_out = type('State', (), {
                    'P': P_out, 'T': T_out, 'h': 250000, 's': 1000, 'rho': 5
                })()
                
                mass_flow = model.displacement * state_in.rho * fr
                
                try:
                    perf = model.calculate_performance(state_in, state_out, mass_flow)
                    efficiency_map[i, j] = perf.eta_isentropic
                    power_map[i, j] = perf.W_comp
                except:
                    efficiency_map[i, j] = 0.5
                    power_map[i, j] = 10
        
        return {
            'pressure_ratios': pressure_ratios,
            'flow_ratios': flow_ratios,
            'efficiency_map': efficiency_map,
            'power_map': power_map
        }

# Default component integration instance
component_integration = ComponentIntegration()

# Add default components
component_integration.add_compressor("scroll_comp", CompressorType.SCROLL, 0.15)
component_integration.add_compressor("recip_comp", CompressorType.RECIPROCATING, 0.12)
component_integration.add_heat_exchanger("evaporator", HeatExchangerType.FINNED_TUBE, 40)
component_integration.add_heat_exchanger("condenser", HeatExchangerType.SHELL_TUBE, 60)
component_integration.add_expansion_device("txv", ExpansionType.THROTTLING_VALVE)
component_integration.add_expansion_device("expander", ExpansionType.EXPANDER)

if __name__ == "__main__":
    # Test component models
    print("Testing component models...")
    
    # Test compressor map generation
    comp_map = component_integration.generate_component_maps("scroll_comp", "compressor")
    print(f"Generated compressor map with efficiency range: "
          f"{comp_map['efficiency_map'].min():.3f} - {comp_map['efficiency_map'].max():.3f}")