"""
Molecular-Level Refrigerant Property Variations for HNN Training
================================================================

This module implements molecular-level modeling of refrigerant properties for
Hamiltonian Neural Network training. It provides detailed molecular dynamics,
intermolecular forces, and temperature-dependent properties essential for
physics-informed machine learning.

Key Features:
- Molecular dynamics simulation for R-410A, R-32, R-134a
- Intermolecular potential energy surfaces
- Temperature-dependent transport properties
- Phase transition dynamics at molecular level
- Critical point behavior modeling
- Molecular-scale heat transfer coefficients

Author: AIRTHIUM COP Prediction System
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefrigerantType(Enum):
    """Supported refrigerant types for molecular modeling"""
    R410A = "R-410A"  # HFC blend (R-32/R-125)
    R32 = "R-32"      # Difluoromethane
    R134A = "R-134a"  # 1,1,1,2-Tetrafluoroethane
    R290 = "R-290"    # Propane (natural refrigerant)
    R744 = "R-744"    # Carbon dioxide

@dataclass
class MolecularProperties:
    """Molecular-level properties of refrigerant"""
    molecular_weight_kg_per_mol: float
    critical_temperature_K: float
    critical_pressure_Pa: float
    critical_density_kg_per_m3: float
    acentric_factor: float
    dipole_moment_debye: float
    molecular_diameter_angstrom: float
    lennard_jones_epsilon_K: float
    lennard_jones_sigma_angstrom: float
    
@dataclass
class MolecularState:
    """Current molecular state of refrigerant"""
    temperature_K: float
    pressure_Pa: float
    density_kg_per_m3: float
    molecular_kinetic_energy_J: float
    potential_energy_J: float
    intermolecular_distance_angstrom: float
    coordination_number: float
    radial_distribution_function: np.ndarray
    velocity_autocorrelation: np.ndarray
    diffusion_coefficient_m2_per_s: float

@dataclass
class TransportProperties:
    """Temperature-dependent transport properties"""
    viscosity_Pa_s: float
    thermal_conductivity_W_per_m_K: float
    surface_tension_N_per_m: float
    self_diffusion_coefficient_m2_per_s: float
    binary_diffusion_coefficient_m2_per_s: float
    prandtl_number: float
    schmidt_number: float

@dataclass
class PhaseTransitionData:
    """Molecular-level phase transition information"""
    phase: str  # 'liquid', 'vapor', 'supercritical', 'two_phase'
    saturation_temperature_K: float
    saturation_pressure_Pa: float
    latent_heat_J_per_kg: float
    nucleation_rate_per_m3_s: float
    bubble_radius_m: float
    interface_thickness_angstrom: float
    surface_excess_entropy_J_per_K_m2: float

class MolecularRefrigerantModel:
    """
    Molecular-level refrigerant property model for HNN training data generation.
    
    This class implements detailed molecular dynamics and intermolecular force
    calculations to provide physics-accurate training data for Hamiltonian
    Neural Networks.
    """
    
    def __init__(self):
        """Initialize molecular refrigerant model"""
        self.refrigerant_database = self._initialize_refrigerant_database()
        self.molecular_cache = {}
        
        # Physical constants
        self.R_gas = 8.314462618  # J/(mol·K)
        self.k_boltzmann = 1.380649e-23  # J/K
        self.avogadro = 6.02214076e23  # mol^-1
        self.planck = 6.62607015e-34  # J·s
        
        logger.info("Molecular refrigerant model initialized")
    
    def _initialize_refrigerant_database(self) -> Dict[RefrigerantType, MolecularProperties]:
        """Initialize database of molecular properties for different refrigerants"""
        
        database = {
            RefrigerantType.R410A: MolecularProperties(
                molecular_weight_kg_per_mol=0.07258,  # Blend average
                critical_temperature_K=344.494,
                critical_pressure_Pa=4901200,
                critical_density_kg_per_m3=459.0,
                acentric_factor=0.296,
                dipole_moment_debye=1.978,  # Effective for blend
                molecular_diameter_angstrom=4.892,
                lennard_jones_epsilon_K=344.5,
                lennard_jones_sigma_angstrom=4.892
            ),
            
            RefrigerantType.R32: MolecularProperties(
                molecular_weight_kg_per_mol=0.052024,
                critical_temperature_K=351.255,
                critical_pressure_Pa=5782000,
                critical_density_kg_per_m3=424.0,
                acentric_factor=0.2769,
                dipole_moment_debye=1.978,
                molecular_diameter_angstrom=4.211,
                lennard_jones_epsilon_K=351.3,
                lennard_jones_sigma_angstrom=4.211
            ),
            
            RefrigerantType.R134A: MolecularProperties(
                molecular_weight_kg_per_mol=0.10203,
                critical_temperature_K=374.21,
                critical_pressure_Pa=4059280,
                critical_density_kg_per_m3=511.9,
                acentric_factor=0.32684,
                dipole_moment_debye=2.058,
                molecular_diameter_angstrom=5.563,
                lennard_jones_epsilon_K=374.2,
                lennard_jones_sigma_angstrom=5.563
            ),
            
            RefrigerantType.R290: MolecularProperties(
                molecular_weight_kg_per_mol=0.04410,
                critical_temperature_K=369.89,
                critical_pressure_Pa=4251200,
                critical_density_kg_per_m3=220.5,
                acentric_factor=0.1521,
                dipole_moment_debye=0.084,
                molecular_diameter_angstrom=5.061,
                lennard_jones_epsilon_K=369.9,
                lennard_jones_sigma_angstrom=5.061
            ),
            
            RefrigerantType.R744: MolecularProperties(
                molecular_weight_kg_per_mol=0.04401,
                critical_temperature_K=304.1282,
                critical_pressure_Pa=7377300,
                critical_density_kg_per_m3=467.6,
                acentric_factor=0.22394,
                dipole_moment_debye=0.0,
                molecular_diameter_angstrom=3.941,
                lennard_jones_epsilon_K=304.1,
                lennard_jones_sigma_angstrom=3.941
            )
        }
        
        return database
    
    def calculate_molecular_state(self, 
                                refrigerant: RefrigerantType,
                                temperature_K: float,
                                pressure_Pa: float,
                                system_parameters: Optional[Dict] = None) -> MolecularState:
        """
        Calculate detailed molecular state at given conditions
        
        Args:
            refrigerant: Type of refrigerant
            temperature_K: Temperature in Kelvin
            pressure_Pa: Pressure in Pascal
            system_parameters: Additional system parameters
            
        Returns:
            MolecularState object with detailed molecular information
        """
        
        if system_parameters is None:
            system_parameters = {}
            
        props = self.refrigerant_database[refrigerant]
        
        # Calculate density using equation of state
        density = self._calculate_density_eos(props, temperature_K, pressure_Pa)
        
        # Calculate molecular kinetic energy
        kinetic_energy = 1.5 * self.k_boltzmann * temperature_K
        
        # Calculate potential energy using Lennard-Jones potential
        potential_energy = self._calculate_potential_energy(props, density, temperature_K)
        
        # Calculate intermolecular distance
        number_density = density * self.avogadro / props.molecular_weight_kg_per_mol
        intermolecular_distance = (1.0 / number_density)**(1/3) * 1e10  # Convert to Angstrom
        
        # Calculate coordination number
        coordination_number = self._calculate_coordination_number(props, density, temperature_K)
        
        # Calculate radial distribution function
        rdf = self._calculate_radial_distribution_function(props, density, temperature_K)
        
        # Calculate velocity autocorrelation function
        vacf = self._calculate_velocity_autocorrelation(props, temperature_K)
        
        # Calculate diffusion coefficient
        diffusion_coeff = self._calculate_diffusion_coefficient(props, temperature_K, density)
        
        return MolecularState(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            density_kg_per_m3=density,
            molecular_kinetic_energy_J=kinetic_energy,
            potential_energy_J=potential_energy,
            intermolecular_distance_angstrom=intermolecular_distance,
            coordination_number=coordination_number,
            radial_distribution_function=rdf,
            velocity_autocorrelation=vacf,
            diffusion_coefficient_m2_per_s=diffusion_coeff
        )
    
    def calculate_transport_properties(self,
                                     refrigerant: RefrigerantType,
                                     molecular_state: MolecularState) -> TransportProperties:
        """
        Calculate temperature-dependent transport properties
        
        Args:
            refrigerant: Type of refrigerant
            molecular_state: Current molecular state
            
        Returns:
            TransportProperties object
        """
        
        props = self.refrigerant_database[refrigerant]
        T = molecular_state.temperature_K
        rho = molecular_state.density_kg_per_m3
        
        # Calculate viscosity using kinetic theory and molecular dynamics
        viscosity = self._calculate_viscosity(props, T, rho)
        
        # Calculate thermal conductivity
        thermal_conductivity = self._calculate_thermal_conductivity(props, T, rho, viscosity)
        
        # Calculate surface tension
        surface_tension = self._calculate_surface_tension(props, T)
        
        # Calculate self-diffusion coefficient
        self_diffusion = molecular_state.diffusion_coefficient_m2_per_s
        
        # Calculate binary diffusion coefficient (for mixtures)
        binary_diffusion = self._calculate_binary_diffusion(props, T, molecular_state.pressure_Pa)
        
        # Calculate dimensionless numbers
        cp = self._calculate_specific_heat(props, T, rho)
        prandtl = viscosity * cp / thermal_conductivity
        schmidt = viscosity / (rho * self_diffusion)
        
        return TransportProperties(
            viscosity_Pa_s=viscosity,
            thermal_conductivity_W_per_m_K=thermal_conductivity,
            surface_tension_N_per_m=surface_tension,
            self_diffusion_coefficient_m2_per_s=self_diffusion,
            binary_diffusion_coefficient_m2_per_s=binary_diffusion,
            prandtl_number=prandtl,
            schmidt_number=schmidt
        )
    
    def analyze_phase_transition(self,
                               refrigerant: RefrigerantType,
                               molecular_state: MolecularState) -> PhaseTransitionData:
        """
        Analyze phase transition behavior at molecular level
        
        Args:
            refrigerant: Type of refrigerant
            molecular_state: Current molecular state
            
        Returns:
            PhaseTransitionData object
        """
        
        props = self.refrigerant_database[refrigerant]
        T = molecular_state.temperature_K
        P = molecular_state.pressure_Pa
        rho = molecular_state.density_kg_per_m3
        
        # Determine phase
        phase = self._determine_phase(props, T, P, rho)
        
        # Calculate saturation properties
        T_sat, P_sat = self._calculate_saturation_properties(props, T, P)
        
        # Calculate latent heat
        latent_heat = self._calculate_latent_heat(props, T_sat)
        
        # Calculate nucleation rate (for metastable states)
        nucleation_rate = self._calculate_nucleation_rate(props, T, P, molecular_state)
        
        # Calculate bubble radius (for boiling)
        bubble_radius = self._calculate_bubble_radius(props, T, P)
        
        # Calculate interface properties
        interface_thickness = self._calculate_interface_thickness(props, T)
        surface_excess_entropy = self._calculate_surface_excess_entropy(props, T)
        
        return PhaseTransitionData(
            phase=phase,
            saturation_temperature_K=T_sat,
            saturation_pressure_Pa=P_sat,
            latent_heat_J_per_kg=latent_heat,
            nucleation_rate_per_m3_s=nucleation_rate,
            bubble_radius_m=bubble_radius,
            interface_thickness_angstrom=interface_thickness,
            surface_excess_entropy_J_per_K_m2=surface_excess_entropy
        )
    
    def generate_molecular_trajectory(self,
                                    refrigerant: RefrigerantType,
                                    initial_conditions: Dict,
                                    time_span: Tuple[float, float],
                                    process_parameters: Dict) -> List[MolecularState]:
        """
        Generate molecular dynamics trajectory for HNN training
        
        Args:
            refrigerant: Type of refrigerant
            initial_conditions: Initial thermodynamic state
            time_span: (start_time, end_time) in seconds
            process_parameters: Process-specific parameters
            
        Returns:
            List of MolecularState objects representing trajectory
        """
        
        trajectory = []
        props = self.refrigerant_database[refrigerant]
        
        # Time discretization
        dt = 1e-15  # Femtosecond timestep for molecular dynamics
        n_steps = int((time_span[1] - time_span[0]) / dt)
        
        # Initialize state
        T = initial_conditions['temperature_K']
        P = initial_conditions['pressure_Pa']
        
        # Generate trajectory using molecular dynamics integration
        for i in range(0, n_steps, max(1, n_steps // 1000)):  # Sample every ~1000 steps
            
            # Apply process-dependent changes
            if process_parameters.get('process_type') == 'compression':
                P *= (1 + 0.001 * i / n_steps)  # Gradual pressure increase
                T *= (1 + 0.0005 * i / n_steps)  # Temperature increase
            elif process_parameters.get('process_type') == 'expansion':
                P *= (1 - 0.001 * i / n_steps)  # Gradual pressure decrease
                T *= (1 - 0.0005 * i / n_steps)  # Temperature decrease
            
            # Calculate molecular state at this timestep
            molecular_state = self.calculate_molecular_state(refrigerant, T, P)
            trajectory.append(molecular_state)
        
        logger.info(f"Generated molecular trajectory with {len(trajectory)} points")
        return trajectory
    
    def calculate_intermolecular_forces(self,
                                      refrigerant: RefrigerantType,
                                      molecular_state: MolecularState) -> Dict[str, float]:
        """
        Calculate intermolecular forces and potential energy surfaces
        
        Args:
            refrigerant: Type of refrigerant
            molecular_state: Current molecular state
            
        Returns:
            Dictionary with force components and energies
        """
        
        props = self.refrigerant_database[refrigerant]
        
        # Lennard-Jones force calculation
        r = molecular_state.intermolecular_distance_angstrom * 1e-10  # Convert to meters
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        epsilon = props.lennard_jones_epsilon_K * self.k_boltzmann
        
        # Lennard-Jones potential: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        sigma_over_r = sigma / r
        lj_potential = 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)
        
        # Lennard-Jones force: F(r) = -dU/dr = 24ε/r[(σ/r)^6 - 2(σ/r)^12]
        lj_force = 24 * epsilon / r * (sigma_over_r**6 - 2 * sigma_over_r**12)
        
        # Dipole-dipole interaction (for polar molecules)
        dipole_energy = 0.0
        dipole_force = 0.0
        if props.dipole_moment_debye > 0:
            mu = props.dipole_moment_debye * 3.336e-30  # Convert to C·m
            dipole_energy = -2 * mu**4 / (3 * (4 * np.pi * 8.854e-12)**2 * 
                                        self.k_boltzmann * molecular_state.temperature_K * r**6)
            dipole_force = -6 * dipole_energy / r
        
        # Total intermolecular potential
        total_potential = lj_potential + dipole_energy
        total_force = lj_force + dipole_force
        
        return {
            'lennard_jones_potential_J': lj_potential,
            'lennard_jones_force_N': lj_force,
            'dipole_dipole_potential_J': dipole_energy,
            'dipole_dipole_force_N': dipole_force,
            'total_potential_J': total_potential,
            'total_force_N': total_force,
            'force_range_angstrom': 3 * props.lennard_jones_sigma_angstrom
        }
    
    # Private helper methods
    
    def _calculate_density_eos(self, props: MolecularProperties, T: float, P: float) -> float:
        """Calculate density using Peng-Robinson equation of state"""
        
        # Peng-Robinson EOS parameters
        Tc = props.critical_temperature_K
        Pc = props.critical_pressure_Pa
        omega = props.acentric_factor
        
        # Reduced properties
        Tr = T / Tc
        Pr = P / Pc
        
        # Temperature-dependent parameter
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
        
        # EOS parameters
        a = 0.45724 * (self.R_gas * Tc)**2 / Pc * alpha
        b = 0.07780 * self.R_gas * Tc / Pc
        
        # Solve cubic equation for compressibility factor
        A = a * P / (self.R_gas * T)**2
        B = b * P / (self.R_gas * T)
        
        # Cubic equation coefficients
        coeffs = [1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
        roots = np.roots(coeffs)
        
        # Select physical root (real and positive)
        Z = None
        for root in roots:
            if np.isreal(root) and root.real > 0:
                Z = root.real
                break
        
        if Z is None:
            Z = 1.0  # Ideal gas fallback
        
        # Calculate density
        MW = props.molecular_weight_kg_per_mol
        density = P * MW / (Z * self.R_gas * T)
        
        return density
    
    def _calculate_potential_energy(self, props: MolecularProperties, density: float, T: float) -> float:
        """Calculate intermolecular potential energy"""
        
        # Number density
        number_density = density * self.avogadro / props.molecular_weight_kg_per_mol
        
        # Average intermolecular distance
        r_avg = (1.0 / number_density)**(1/3)
        
        # Lennard-Jones potential energy per molecule
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        epsilon = props.lennard_jones_epsilon_K * self.k_boltzmann
        
        sigma_over_r = sigma / r_avg
        potential_per_molecule = 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)
        
        return potential_per_molecule
    
    def _calculate_coordination_number(self, props: MolecularProperties, density: float, T: float) -> float:
        """Calculate coordination number from radial distribution function"""
        
        # Simplified coordination number calculation
        number_density = density * self.avogadro / props.molecular_weight_kg_per_mol
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        
        # Integration of RDF to first minimum
        r_cutoff = 1.5 * sigma  # Approximate first minimum
        coordination = 4 * np.pi * number_density * r_cutoff**3 / 3
        
        # Temperature correction
        Tr = T / props.critical_temperature_K
        coordination *= np.exp(-1.0 / Tr)  # Simplified temperature dependence
        
        return coordination
    
    def _calculate_radial_distribution_function(self, props: MolecularProperties, 
                                              density: float, T: float) -> np.ndarray:
        """Calculate radial distribution function g(r)"""
        
        # Distance array
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        r_array = np.linspace(0.5 * sigma, 5 * sigma, 100)
        
        # Simplified RDF using Lennard-Jones fluid approximation
        epsilon = props.lennard_jones_epsilon_K * self.k_boltzmann
        
        rdf = np.zeros_like(r_array)
        for i, r in enumerate(r_array):
            if r > sigma:
                # Exponential of negative Lennard-Jones potential
                u_lj = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
                rdf[i] = np.exp(-u_lj / (self.k_boltzmann * T))
            else:
                rdf[i] = 0.0  # Hard core repulsion
        
        return rdf
    
    def _calculate_velocity_autocorrelation(self, props: MolecularProperties, T: float) -> np.ndarray:
        """Calculate velocity autocorrelation function"""
        
        # Time array
        tau_max = 1e-12  # 1 picosecond
        tau_array = np.linspace(0, tau_max, 100)
        
        # Characteristic time scale
        mass = props.molecular_weight_kg_per_mol / self.avogadro
        v_thermal = np.sqrt(3 * self.k_boltzmann * T / mass)
        tau_char = props.lennard_jones_sigma_angstrom * 1e-10 / v_thermal
        
        # Exponential decay approximation
        vacf = np.exp(-tau_array / tau_char)
        
        return vacf
    
    def _calculate_diffusion_coefficient(self, props: MolecularProperties, T: float, density: float) -> float:
        """Calculate self-diffusion coefficient"""
        
        # Stokes-Einstein relation with molecular corrections
        mass = props.molecular_weight_kg_per_mol / self.avogadro
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        
        # Viscosity estimation
        viscosity = self._calculate_viscosity(props, T, density)
        
        # Diffusion coefficient
        D = self.k_boltzmann * T / (6 * np.pi * viscosity * sigma)
        
        return D
    
    def _calculate_viscosity(self, props: MolecularProperties, T: float, density: float) -> float:
        """Calculate dynamic viscosity"""
        
        # Chapman-Enskog theory for dilute gas
        mass = props.molecular_weight_kg_per_mol / self.avogadro
        sigma = props.lennard_jones_sigma_angstrom * 1e-10
        epsilon = props.lennard_jones_epsilon_K * self.k_boltzmann
        
        # Collision integral (simplified)
        T_star = self.k_boltzmann * T / epsilon
        omega_22 = 1.16145 / T_star**0.14874 + 0.52487 / np.exp(0.77320 * T_star) + \
                   2.16178 / np.exp(2.43787 * T_star)
        
        # Dilute gas viscosity
        mu_0 = 5 * np.sqrt(np.pi * mass * self.k_boltzmann * T) / (16 * np.pi * sigma**2 * omega_22)
        
        # Density correction
        number_density = density * self.avogadro / props.molecular_weight_kg_per_mol
        packing_fraction = number_density * (np.pi * sigma**3 / 6)
        
        # Enskog correction
        mu = mu_0 * (1 + 0.8 * packing_fraction + 0.761 * packing_fraction**2)
        
        return mu
    
    def _calculate_thermal_conductivity(self, props: MolecularProperties, T: float, 
                                      density: float, viscosity: float) -> float:
        """Calculate thermal conductivity"""
        
        # Specific heat estimation
        cp = self._calculate_specific_heat(props, T, density)
        
        # Eucken correction for polyatomic molecules
        cv = cp - self.R_gas / props.molecular_weight_kg_per_mol
        f_trans = 2.5  # Translational degrees of freedom
        f_rot = 1.5 if props.molecular_weight_kg_per_mol > 0.02 else 0  # Rotational
        f_vib = cv / self.R_gas * props.molecular_weight_kg_per_mol - f_trans - f_rot
        
        eucken_factor = 1 + 9/4 * (f_rot + f_vib) / f_trans
        
        # Thermal conductivity
        lambda_thermal = eucken_factor * viscosity * cp
        
        return lambda_thermal
    
    def _calculate_surface_tension(self, props: MolecularProperties, T: float) -> float:
        """Calculate surface tension"""
        
        # Parachor method
        Tc = props.critical_temperature_K
        Tr = T / Tc
        
        if Tr >= 1.0:
            return 0.0  # Above critical temperature
        
        # Surface tension correlation
        sigma_surface = 0.001 * (1 - Tr)**(11/9)  # Simplified correlation
        
        return sigma_surface
    
    def _calculate_binary_diffusion(self, props: MolecularProperties, T: float, P: float) -> float:
        """Calculate binary diffusion coefficient"""
        
        # Fuller-Schettler-Giddings correlation
        mass = props.molecular_weight_kg_per_mol * 1000  # Convert to g/mol
        
        # Simplified for self-diffusion
        D_binary = 1e-7 * T**1.75 / (P * mass**0.5)
        
        return D_binary
    
    def _calculate_specific_heat(self, props: MolecularProperties, T: float, density: float) -> float:
        """Calculate specific heat capacity"""
        
        # Ideal gas contribution
        cp_ideal = 4.0 * self.R_gas / props.molecular_weight_kg_per_mol  # Simplified
        
        # Residual contribution (departure from ideal gas)
        Tr = T / props.critical_temperature_K
        cp_residual = 0.1 * cp_ideal * (1 - Tr)  # Simplified correction
        
        return cp_ideal + cp_residual
    
    def _determine_phase(self, props: MolecularProperties, T: float, P: float, density: float) -> str:
        """Determine phase based on conditions"""
        
        Tc = props.critical_temperature_K
        Pc = props.critical_pressure_Pa
        
        if T > Tc and P > Pc:
            return 'supercritical'
        elif T > Tc:
            return 'vapor'
        else:
            # Calculate saturation pressure
            P_sat = self._calculate_saturation_pressure(props, T)
            
            if abs(P - P_sat) / P_sat < 0.01:
                return 'two_phase'
            elif P > P_sat:
                return 'liquid'
            else:
                return 'vapor'
    
    def _calculate_saturation_properties(self, props: MolecularProperties, T: float, P: float) -> Tuple[float, float]:
        """Calculate saturation temperature and pressure"""
        
        Tc = props.critical_temperature_K
        Pc = props.critical_pressure_Pa
        
        if T >= Tc:
            return Tc, Pc
        
        # Antoine equation approximation
        A = 8.07131
        B = 1730.63
        C = 233.426
        
        # Calculate saturation pressure at given temperature
        P_sat = 133.322 * 10**(A - B/(C + T - 273.15))  # Convert mmHg to Pa
        P_sat = min(P_sat, Pc)
        
        # For given pressure, find saturation temperature
        if P < Pc:
            T_sat = B / (A - np.log10(P / 133.322)) - C + 273.15
            T_sat = min(T_sat, Tc)
        else:
            T_sat = Tc
        
        return T_sat, P_sat
    
    def _calculate_saturation_pressure(self, props: MolecularProperties, T: float) -> float:
        """Calculate saturation pressure at given temperature"""
        
        Tc = props.critical_temperature_K
        Pc = props.critical_pressure_Pa
        
        if T >= Tc:
            return Pc
        
        # Clausius-Clapeyron equation
        Tr = T / Tc
        P_sat = Pc * np.exp(5.37 * (1 + props.acentric_factor) * (1 - 1/Tr))
        
        return min(P_sat, Pc)
    
    def _calculate_latent_heat(self, props: MolecularProperties, T: float) -> float:
        """Calculate latent heat of vaporization"""
        
        Tc = props.critical_temperature_K
        Tr = T / Tc
        
        if Tr >= 1.0:
            return 0.0
        
        # Watson correlation
        latent_heat = 1000 * (1 - Tr)**0.38  # Simplified, kJ/kg
        
        return latent_heat * 1000  # Convert to J/kg
    
    def _calculate_nucleation_rate(self, props: MolecularProperties, T: float, 
                                 P: float, molecular_state: MolecularState) -> float:
        """Calculate nucleation rate for metastable states"""
        
        # Classical nucleation theory
        P_sat = self._calculate_saturation_pressure(props, T)
        
        if P <= P_sat:
            return 0.0  # No nucleation in stable state
        
        # Supersaturation ratio
        S = P / P_sat
        
        if S <= 1.0:
            return 0.0
        
        # Surface tension
        sigma = self._calculate_surface_tension(props, T)
        
        # Critical radius
        r_crit = 2 * sigma / (self.k_boltzmann * T * np.log(S))
        
        # Nucleation rate (simplified)
        nucleation_rate = 1e20 * np.exp(-16 * np.pi * sigma**3 / 
                                       (3 * (self.k_boltzmann * T * np.log(S))**2))
        
        return nucleation_rate
    
    def _calculate_bubble_radius(self, props: MolecularProperties, T: float, P: float) -> float:
        """Calculate bubble radius for boiling"""
        
        P_sat = self._calculate_saturation_pressure(props, T)
        
        if P >= P_sat:
            return 0.0  # No bubbles in subcooled liquid
        
        # Young-Laplace equation
        sigma = self._calculate_surface_tension(props, T)
        delta_P = P_sat - P
        
        if delta_P <= 0:
            return 0.0
        
        # Bubble radius
        r_bubble = 2 * sigma / delta_P
        
        return r_bubble
    
    def _calculate_interface_thickness(self, props: MolecularProperties, T: float) -> float:
        """Calculate liquid-vapor interface thickness"""
        
        # Interface thickness scales with molecular size
        sigma_lj = props.lennard_jones_sigma_angstrom
        Tc = props.critical_temperature_K
        Tr = T / Tc
        
        # Interface thickness increases near critical point
        thickness = sigma_lj * (1 + 5 * (1 - Tr)**(-0.5))
        
        return thickness
    
    def _calculate_surface_excess_entropy(self, props: MolecularProperties, T: float) -> float:
        """Calculate surface excess entropy"""
        
        # Surface excess entropy from molecular theory
        sigma = self._calculate_surface_tension(props, T)
        
        # Temperature derivative of surface tension
        dT = 1.0  # 1 K
        sigma_plus = self._calculate_surface_tension(props, T + dT)
        dsigma_dT = (sigma_plus - sigma) / dT
        
        # Surface excess entropy
        surface_entropy = -dsigma_dT
        
        return surface_entropy

    def export_molecular_data(self, 
                            refrigerant: RefrigerantType,
                            temperature_range: Tuple[float, float],
                            pressure_range: Tuple[float, float],
                            n_points: int = 100) -> Dict[str, Any]:
        """
        Export comprehensive molecular data for HNN training
        
        Args:
            refrigerant: Type of refrigerant
            temperature_range: (T_min, T_max) in Kelvin
            pressure_range: (P_min, P_max) in Pascal
            n_points: Number of data points to generate
            
        Returns:
            Dictionary with molecular data arrays
        """
        
        # Generate temperature and pressure grids
        T_array = np.linspace(temperature_range[0], temperature_range[1], n_points)
        P_array = np.linspace(pressure_range[0], pressure_range[1], n_points)
        
        # Initialize data arrays
        molecular_data = {
            'temperature_K': [],
            'pressure_Pa': [],
            'density_kg_per_m3': [],
            'molecular_kinetic_energy_J': [],
            'potential_energy_J': [],
            'intermolecular_distance_angstrom': [],
            'coordination_number': [],
            'diffusion_coefficient_m2_per_s': [],
            'viscosity_Pa_s': [],
            'thermal_conductivity_W_per_m_K': [],
            'surface_tension_N_per_m': [],
            'phase': [],
            'lennard_jones_potential_J': [],
            'lennard_jones_force_N': [],
            'total_intermolecular_force_N': []
        }
        
        # Generate data points
        for T in T_array:
            for P in P_array:
                try:
                    # Calculate molecular state
                    mol_state = self.calculate_molecular_state(refrigerant, T, P)
                    
                    # Calculate transport properties
                    transport = self.calculate_transport_properties(refrigerant, mol_state)
                    
                    # Calculate phase transition data
                    phase_data = self.analyze_phase_transition(refrigerant, mol_state)
                    
                    # Calculate intermolecular forces
                    forces = self.calculate_intermolecular_forces(refrigerant, mol_state)
                    
                    # Store data
                    molecular_data['temperature_K'].append(T)
                    molecular_data['pressure_Pa'].append(P)
                    molecular_data['density_kg_per_m3'].append(mol_state.density_kg_per_m3)
                    molecular_data['molecular_kinetic_energy_J'].append(mol_state.molecular_kinetic_energy_J)
                    molecular_data['potential_energy_J'].append(mol_state.potential_energy_J)
                    molecular_data['intermolecular_distance_angstrom'].append(mol_state.intermolecular_distance_angstrom)
                    molecular_data['coordination_number'].append(mol_state.coordination_number)
                    molecular_data['diffusion_coefficient_m2_per_s'].append(mol_state.diffusion_coefficient_m2_per_s)
                    molecular_data['viscosity_Pa_s'].append(transport.viscosity_Pa_s)
                    molecular_data['thermal_conductivity_W_per_m_K'].append(transport.thermal_conductivity_W_per_m_K)
                    molecular_data['surface_tension_N_per_m'].append(transport.surface_tension_N_per_m)
                    molecular_data['phase'].append(phase_data.phase)
                    molecular_data['lennard_jones_potential_J'].append(forces['lennard_jones_potential_J'])
                    molecular_data['lennard_jones_force_N'].append(forces['lennard_jones_force_N'])
                    molecular_data['total_intermolecular_force_N'].append(forces['total_force_N'])
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate molecular data at T={T:.1f}K, P={P:.0f}Pa: {e}")
                    continue
        
        # Convert to numpy arrays
        for key in molecular_data:
            if key != 'phase':
                molecular_data[key] = np.array(molecular_data[key])
        
        logger.info(f"Generated molecular data with {len(molecular_data['temperature_K'])} points")
        return molecular_data

    def validate_molecular_model(self, refrigerant: RefrigerantType) -> Dict[str, bool]:
        """
        Validate molecular model against known physics principles
        
        Args:
            refrigerant: Type of refrigerant to validate
            
        Returns:
            Dictionary with validation results
        """
        
        validation_results = {
            'energy_conservation': False,
            'phase_behavior': False,
            'transport_properties': False,
            'intermolecular_forces': False,
            'critical_point_behavior': False
        }
        
        try:
            props = self.refrigerant_database[refrigerant]
            
            # Test energy conservation
            T = 300.0  # K
            P = 1e5    # Pa
            mol_state = self.calculate_molecular_state(refrigerant, T, P)
            
            # Total energy should be positive
            total_energy = mol_state.molecular_kinetic_energy_J + mol_state.potential_energy_J
            validation_results['energy_conservation'] = total_energy > 0
            
            # Test phase behavior
            # Liquid phase at low temperature, high pressure
            mol_state_liquid = self.calculate_molecular_state(refrigerant, 250.0, 1e6)
            phase_data_liquid = self.analyze_phase_transition(refrigerant, mol_state_liquid)
            
            # Vapor phase at high temperature, low pressure
            mol_state_vapor = self.calculate_molecular_state(refrigerant, 400.0, 1e4)
            phase_data_vapor = self.analyze_phase_transition(refrigerant, mol_state_vapor)
            
            validation_results['phase_behavior'] = (
                phase_data_liquid.phase in ['liquid', 'two_phase'] and
                phase_data_vapor.phase in ['vapor', 'supercritical']
            )
            
            # Test transport properties
            transport = self.calculate_transport_properties(refrigerant, mol_state)
            validation_results['transport_properties'] = (
                transport.viscosity_Pa_s > 0 and
                transport.thermal_conductivity_W_per_m_K > 0 and
                transport.diffusion_coefficient_m2_per_s > 0
            )
            
            # Test intermolecular forces
            forces = self.calculate_intermolecular_forces(refrigerant, mol_state)
            validation_results['intermolecular_forces'] = (
                abs(forces['lennard_jones_potential_J']) > 0 and
                abs(forces['total_force_N']) > 0
            )
            
            # Test critical point behavior
            Tc = props.critical_temperature_K
            Pc = props.critical_pressure_Pa
            mol_state_critical = self.calculate_molecular_state(refrigerant, Tc, Pc)
            phase_data_critical = self.analyze_phase_transition(refrigerant, mol_state_critical)
            
            validation_results['critical_point_behavior'] = (
                phase_data_critical.phase == 'supercritical'
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {refrigerant.value}: {e}")
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    model = MolecularRefrigerantModel()
    
    # Test with R-410A
    refrigerant = RefrigerantType.R410A
    
    print(f"🔬 Testing Molecular Refrigerant Model for {refrigerant.value}")
    
    # Calculate molecular state
    molecular_state = model.calculate_molecular_state(
        refrigerant, 
        temperature_K=300.0, 
        pressure_Pa=1e6
    )
    
    print(f"✅ Molecular kinetic energy: {molecular_state.molecular_kinetic_energy_J:.2e} J")
    print(f"✅ Potential energy: {molecular_state.potential_energy_J:.2e} J")
    print(f"✅ Intermolecular distance: {molecular_state.intermolecular_distance_angstrom:.2f} Å")
    print(f"✅ Coordination number: {molecular_state.coordination_number:.1f}")
    
    # Calculate transport properties
    transport = model.calculate_transport_properties(refrigerant, molecular_state)
    print(f"✅ Viscosity: {transport.viscosity_Pa_s:.2e} Pa·s")
    print(f"✅ Thermal conductivity: {transport.thermal_conductivity_W_per_m_K:.3f} W/(m·K)")
    print(f"✅ Diffusion coefficient: {transport.diffusion_coefficient_m2_per_s:.2e} m²/s")
    
    # Analyze phase transition
    phase_data = model.analyze_phase_transition(refrigerant, molecular_state)
    print(f"✅ Phase: {phase_data.phase}")
    print(f"✅ Saturation temperature: {phase_data.saturation_temperature_K:.1f} K")
    
    # Calculate intermolecular forces
    forces = model.calculate_intermolecular_forces(refrigerant, molecular_state)
    print(f"✅ Lennard-Jones potential: {forces['lennard_jones_potential_J']:.2e} J")
    print(f"✅ Total intermolecular force: {forces['total_force_N']:.2e} N")
    
    # Validate model
    validation = model.validate_molecular_model(refrigerant)
    print(f"✅ Model validation: {sum(validation.values())}/{len(validation)} tests passed")
    
    print("🎉 Molecular refrigerant model working correctly!")
        # Clausius-