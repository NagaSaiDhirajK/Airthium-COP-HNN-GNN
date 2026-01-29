"""
Thermodynamic Properties Database
Comprehensive fluid property calculations using CoolProp for all working fluids
"""

import numpy as np
import CoolProp.CoolProp as CP
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum

class FluidType(Enum):
    """Enumeration of fluid types"""
    REFRIGERANT = "refrigerant"
    WATER_STEAM = "water_steam"
    THERMAL_OIL = "thermal_oil"
    AIR = "air"
    ORC_FLUID = "orc_fluid"

@dataclass
class ThermodynamicState:
    """Container for thermodynamic state properties"""
    P: float  # Pressure [Pa]
    T: float  # Temperature [K]
    h: float  # Specific enthalpy [J/kg]
    s: float  # Specific entropy [J/(kg·K)]
    rho: float  # Density [kg/m³]
    x: Optional[float] = None  # Vapor quality [0-1] or None if single-phase
    phase: str = "unknown"  # "liquid", "vapor", "two-phase", "supercritical"
    cp: Optional[float] = None  # Specific heat at constant pressure [J/(kg·K)]
    cv: Optional[float] = None  # Specific heat at constant volume [J/(kg·K)]
    mu: Optional[float] = None  # Dynamic viscosity [Pa·s]
    k: Optional[float] = None  # Thermal conductivity [W/(m·K)]

class FluidProperties:
    """Base class for fluid property calculations"""
    
    def __init__(self, fluid_name: str, fluid_type: FluidType):
        self.fluid_name = fluid_name
        self.fluid_type = fluid_type
        self.coolprop_name = self._get_coolprop_name()
        
        # Get critical properties
        try:
            self.T_crit = CP.PropsSI('Tcrit', self.coolprop_name)  # [K]
            self.P_crit = CP.PropsSI('Pcrit', self.coolprop_name)  # [Pa]
            self.rho_crit = CP.PropsSI('rhocrit', self.coolprop_name)  # [kg/m³]
            self.M = CP.PropsSI('M', self.coolprop_name)  # [kg/mol]
            self.acentric_factor = CP.PropsSI('acentric', self.coolprop_name)
        except Exception as e:
            warnings.warn(f"Could not get critical properties for {fluid_name}: {e}")
            self.T_crit = None
            self.P_crit = None
            self.rho_crit = None
            self.M = None
            self.acentric_factor = None
    
    def _get_coolprop_name(self) -> str:
        """Map fluid names to CoolProp identifiers"""
        coolprop_mapping = {
            # Refrigerants
            'R245fa': 'R245fa',
            'R1234ze(E)': 'R1234zeE',
            'R1233zd(E)': 'R1233zdE',
            'R134a': 'R134a',
            'R410A': 'R410A',
            'R600a': 'R600a',
            'R744': 'CO2',
            'CO2': 'CO2',
            'R717': 'Ammonia',
            'NH3': 'Ammonia',
            'Ammonia': 'Ammonia',
            'R290': 'Propane',
            'Propane': 'Propane',
            
            # Water/Steam
            'Water': 'Water',
            'H2O': 'Water',
            
            # Air
            'Air': 'Air',
            
            # ORC Fluids
            'Toluene': 'Toluene',
            'Cyclohexane': 'Cyclohexane',
            'n-Pentane': 'n-Pentane',
            'Ethanol': 'Ethanol',
            
            # Thermal oils (approximated with similar fluids)
            'Therminol_VP1': 'DiphenylOxide',  # Approximation
            'Dowtherm_A': 'DiphenylOxide',     # Approximation
            'Mineral_Oil': 'n-Decane',         # Approximation
        }
        
        return coolprop_mapping.get(self.fluid_name, self.fluid_name)
    
    def get_saturation_pressure(self, T: float) -> float:
        """Get saturation pressure at given temperature [Pa]"""
        try:
            if T >= self.T_crit:
                return self.P_crit
            return CP.PropsSI('P', 'T', T, 'Q', 0, self.coolprop_name)
        except Exception:
            # Fallback to Antoine equation approximation
            return self._antoine_equation(T)
    
    def get_saturation_temperature(self, P: float) -> float:
        """Get saturation temperature at given pressure [K]"""
        try:
            if P >= self.P_crit:
                return self.T_crit
            return CP.PropsSI('T', 'P', P, 'Q', 0, self.coolprop_name)
        except Exception:
            # Fallback to inverse Antoine equation
            return self._inverse_antoine_equation(P)
    
    def _antoine_equation(self, T: float) -> float:
        """Antoine equation for vapor pressure (fallback)"""
        # Generic coefficients - should be fluid-specific in production
        A, B, C = 8.07131, 1730.63, 233.426  # Water approximation
        log_P_mmHg = A - B / (T - 273.15 + C)
        P_Pa = 10**(log_P_mmHg) * 133.322  # Convert mmHg to Pa
        return P_Pa
    
    def _inverse_antoine_equation(self, P: float) -> float:
        """Inverse Antoine equation for saturation temperature"""
        A, B, C = 8.07131, 1730.63, 233.426
        P_mmHg = P / 133.322
        T_celsius = B / (A - np.log10(P_mmHg)) - C
        return T_celsius + 273.15
    
    def get_saturated_liquid(self, T: float) -> ThermodynamicState:
        """Get saturated liquid properties at given temperature"""
        try:
            P_sat = self.get_saturation_pressure(T)
            h = CP.PropsSI('H', 'T', T, 'Q', 0, self.coolprop_name)
            s = CP.PropsSI('S', 'T', T, 'Q', 0, self.coolprop_name)
            rho = CP.PropsSI('D', 'T', T, 'Q', 0, self.coolprop_name)
            
            return ThermodynamicState(
                P=P_sat, T=T, h=h, s=s, rho=rho, x=0.0, phase="liquid"
            )
        except Exception as e:
            warnings.warn(f"Error getting saturated liquid for {self.fluid_name}: {e}")
            return None
    
    def get_saturated_vapor(self, T: float) -> ThermodynamicState:
        """Get saturated vapor properties at given temperature"""
        try:
            P_sat = self.get_saturation_pressure(T)
            h = CP.PropsSI('H', 'T', T, 'Q', 1, self.coolprop_name)
            s = CP.PropsSI('S', 'T', T, 'Q', 1, self.coolprop_name)
            rho = CP.PropsSI('D', 'T', T, 'Q', 1, self.coolprop_name)
            
            return ThermodynamicState(
                P=P_sat, T=T, h=h, s=s, rho=rho, x=1.0, phase="vapor"
            )
        except Exception as e:
            warnings.warn(f"Error getting saturated vapor for {self.fluid_name}: {e}")
            return None
    
    def get_properties(self, P: float, T: float, **kwargs) -> ThermodynamicState:
        """Get complete thermodynamic properties at given P, T"""
        try:
            # Determine phase
            phase = self._determine_phase(P, T)
            
            # Get basic properties
            h = CP.PropsSI('H', 'P', P, 'T', T, self.coolprop_name)
            s = CP.PropsSI('S', 'P', P, 'T', T, self.coolprop_name)
            rho = CP.PropsSI('D', 'P', P, 'T', T, self.coolprop_name)
            
            # Get transport properties
            try:
                cp = CP.PropsSI('C', 'P', P, 'T', T, self.coolprop_name)
                cv = CP.PropsSI('O', 'P', P, 'T', T, self.coolprop_name)
                mu = CP.PropsSI('V', 'P', P, 'T', T, self.coolprop_name)
                k = CP.PropsSI('L', 'P', P, 'T', T, self.coolprop_name)
            except:
                cp = cv = mu = k = None
            
            # Get vapor quality if two-phase
            x = None
            if phase == "two-phase":
                try:
                    x = CP.PropsSI('Q', 'P', P, 'T', T, self.coolprop_name)
                except:
                    x = None
            
            return ThermodynamicState(
                P=P, T=T, h=h, s=s, rho=rho, x=x, phase=phase,
                cp=cp, cv=cv, mu=mu, k=k
            )
            
        except Exception as e:
            warnings.warn(f"Error calculating properties for {self.fluid_name}: {e}")
            return self._fallback_properties(P, T)
    
    def get_properties_ph(self, P: float, h: float) -> ThermodynamicState:
        """Get properties from pressure and enthalpy"""
        try:
            T = CP.PropsSI('T', 'P', P, 'H', h, self.coolprop_name)
            return self.get_properties(P, T)
        except Exception as e:
            warnings.warn(f"Error in P-h calculation: {e}")
            return None
    
    def get_properties_ps(self, P: float, s: float) -> ThermodynamicState:
        """Get properties from pressure and entropy"""
        try:
            T = CP.PropsSI('T', 'P', P, 'S', s, self.coolprop_name)
            return self.get_properties(P, T)
        except Exception as e:
            warnings.warn(f"Error in P-s calculation: {e}")
            return None
    
    def _determine_phase(self, P: float, T: float) -> str:
        """Determine phase of fluid at given P, T"""
        try:
            if P > self.P_crit and T > self.T_crit:
                return "supercritical"
            
            T_sat = self.get_saturation_temperature(P)
            
            if abs(T - T_sat) < 0.1:  # Within 0.1 K of saturation
                return "two-phase"
            elif T > T_sat:
                return "vapor"
            else:
                return "liquid"
                
        except:
            return "unknown"
    
    def _fallback_properties(self, P: float, T: float) -> ThermodynamicState:
        """Fallback property calculation using ideal gas or simple correlations"""
        # Very basic fallback - should be improved for production
        R_specific = 8314.0 / (self.M if self.M else 29.0)  # J/(kg·K)
        
        # Ideal gas approximations
        rho = P / (R_specific * T)
        h = 1000 * T  # Very rough approximation
        s = 1000 * np.log(T) - R_specific * np.log(P)
        
        return ThermodynamicState(
            P=P, T=T, h=h, s=s, rho=rho, phase="unknown"
        )

class RefrigerantProperties(FluidProperties):
    """Specialized class for refrigerant properties"""
    
    def __init__(self, refrigerant_name: str):
        super().__init__(refrigerant_name, FluidType.REFRIGERANT)
        
        # Refrigerant-specific properties
        self.GWP = self._get_gwp()  # Global Warming Potential
        self.ODP = self._get_odp()  # Ozone Depletion Potential
        self.safety_class = self._get_safety_class()
    
    def _get_gwp(self) -> float:
        """Get Global Warming Potential"""
        gwp_data = {
            'R245fa': 1030,
            'R1234ze(E)': 4,
            'R1233zd(E)': 1,
            'R134a': 1430,
            'R410A': 2088,
            'R600a': 3,
            'R744': 1,
            'R717': 0,
            'R290': 3,
        }
        return gwp_data.get(self.fluid_name, 0)
    
    def _get_odp(self) -> float:
        """Get Ozone Depletion Potential"""
        # All modern refrigerants have ODP = 0
        return 0.0
    
    def _get_safety_class(self) -> str:
        """Get ASHRAE safety classification"""
        safety_data = {
            'R245fa': 'B1',
            'R1234ze(E)': 'A2L',
            'R1233zd(E)': 'A1',
            'R134a': 'A1',
            'R410A': 'A1',
            'R600a': 'A3',
            'R744': 'A1',
            'R717': 'B2L',
            'R290': 'A3',
        }
        return safety_data.get(self.fluid_name, 'Unknown')

class WaterSteamProperties(FluidProperties):
    """Specialized class for water/steam properties using IAPWS-IF97"""
    
    def __init__(self):
        super().__init__('Water', FluidType.WATER_STEAM)
    
    def get_steam_quality(self, P: float, h: float) -> float:
        """Calculate steam quality from pressure and enthalpy"""
        try:
            h_f = CP.PropsSI('H', 'P', P, 'Q', 0, 'Water')  # Saturated liquid
            h_g = CP.PropsSI('H', 'P', P, 'Q', 1, 'Water')  # Saturated vapor
            
            if h <= h_f:
                return 0.0  # Subcooled liquid
            elif h >= h_g:
                return 1.0  # Superheated vapor
            else:
                return (h - h_f) / (h_g - h_f)  # Two-phase
                
        except Exception:
            return None

class ThermalOilProperties(FluidProperties):
    """Specialized class for thermal oil properties"""
    
    def __init__(self, oil_name: str):
        super().__init__(oil_name, FluidType.THERMAL_OIL)
        self.max_temp = self._get_max_temperature()
    
    def _get_max_temperature(self) -> float:
        """Get maximum operating temperature [K]"""
        max_temps = {
            'Therminol_VP1': 673.15,  # 400°C
            'Dowtherm_A': 673.15,     # 400°C
            'Mineral_Oil': 573.15,    # 300°C
        }
        return max_temps.get(self.fluid_name, 573.15)
    
    def get_viscosity_temperature_correlation(self, T: float) -> float:
        """Temperature-dependent viscosity correlation"""
        # Arrhenius-type correlation: μ = A * exp(B/T)
        correlations = {
            'Therminol_VP1': {'A': 1e-6, 'B': 2000},
            'Dowtherm_A': {'A': 1.2e-6, 'B': 1900},
            'Mineral_Oil': {'A': 2e-6, 'B': 2200},
        }
        
        params = correlations.get(self.fluid_name, {'A': 1e-6, 'B': 2000})
        return params['A'] * np.exp(params['B'] / T)

class FluidDatabase:
    """Central database for all fluid properties"""
    
    def __init__(self):
        self.fluids = {}
        self._initialize_fluids()
    
    def _initialize_fluids(self):
        """Initialize all fluid property objects"""
        
        # Refrigerants
        refrigerants = [
            'R245fa', 'R1234ze(E)', 'R1233zd(E)', 'R134a', 'R410A',
            'R600a', 'R744', 'R717', 'R290'
        ]
        
        for ref in refrigerants:
            self.fluids[ref] = RefrigerantProperties(ref)
        
        # Water/Steam
        self.fluids['Water'] = WaterSteamProperties()
        
        # Thermal Oils
        thermal_oils = ['Therminol_VP1', 'Dowtherm_A', 'Mineral_Oil']
        for oil in thermal_oils:
            self.fluids[oil] = ThermalOilProperties(oil)
        
        # Air
        self.fluids['Air'] = FluidProperties('Air', FluidType.AIR)
        
        # ORC Fluids
        orc_fluids = ['Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol']
        for fluid in orc_fluids:
            self.fluids[fluid] = FluidProperties(fluid, FluidType.ORC_FLUID)
    
    def get_fluid(self, fluid_name: str) -> FluidProperties:
        """Get fluid property object by name"""
        if fluid_name not in self.fluids:
            raise ValueError(f"Fluid '{fluid_name}' not found in database")
        return self.fluids[fluid_name]
    
    def list_available_fluids(self) -> Dict[str, List[str]]:
        """List all available fluids by category"""
        categories = {
            'Refrigerants': [],
            'Water/Steam': [],
            'Thermal Oils': [],
            'Air': [],
            'ORC Fluids': []
        }
        
        for name, fluid in self.fluids.items():
            if fluid.fluid_type == FluidType.REFRIGERANT:
                categories['Refrigerants'].append(name)
            elif fluid.fluid_type == FluidType.WATER_STEAM:
                categories['Water/Steam'].append(name)
            elif fluid.fluid_type == FluidType.THERMAL_OIL:
                categories['Thermal Oils'].append(name)
            elif fluid.fluid_type == FluidType.AIR:
                categories['Air'].append(name)
            elif fluid.fluid_type == FluidType.ORC_FLUID:
                categories['ORC Fluids'].append(name)
        
        return categories

# Global fluid database instance
fluid_db = FluidDatabase()

def get_fluid_properties(fluid_name: str) -> FluidProperties:
    """Convenience function to get fluid properties"""
    return fluid_db.get_fluid(fluid_name)

if __name__ == "__main__":
    # Test the fluid database
    print("Available fluids:")
    for category, fluids in fluid_db.list_available_fluids().items():
        print(f"{category}: {fluids}")
    
    # Test R245fa properties
    r245fa = get_fluid_properties('R245fa')
    state = r245fa.get_properties(500000, 300)  # 5 bar, 300 K
    print(f"\nR245fa at 5 bar, 300 K:")
    print(f"Enthalpy: {state.h:.1f} J/kg")
    print(f"Entropy: {state.s:.1f} J/(kg·K)")
    print(f"Density: {state.rho:.2f} kg/m³")
    print(f"Phase: {state.phase}")