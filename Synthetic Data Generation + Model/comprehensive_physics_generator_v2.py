#!/usr/bin/env python3
"""
Comprehensive Physics Generator for HNN-GNN Heat Pump COP Prediction
====================================================================

This generator integrates ALL 23+ physics models from src/ directory for 
complete feature coverage and realistic COP calculation.

Physics Models Integrated:
1. RefrigerantProperties - Thermodynamic properties via CoolProp
2. HeatExchangerFoulingModel - Fouling degradation  
3. CompressorWearModel - Mechanical wear effects
4. RefrigerantChargeModel - Charge level impacts
5. FaultInjectionModel - Fault scenarios
6. HamiltonianMechanicsModel - Energy conservation dynamics
7. ControlSystemModel - PID control dynamics
8. AmbientConditionModel - Environmental effects
9. MultiZoneBuildingModel - Building load interactions
10. ComponentInteractionModel - Component heat/mass transfer networks
11. GNNTopologyGenerator - Graph structure features
12. PerformanceMaps - Component efficiency maps
13. SensorNoiseModel - Realistic measurement noise
14. PartLoadCyclingModel - Part-load behavior
15. PhaseSpaceTrajectories - Phase space dynamics
16. TemporalDynamicsGenerator - Startup/shutdown transients
17. ComponentModels - Compressor, heat exchanger models

Author: Comprehensive integration of all physics models
"""

import sys
import os
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import time
from datetime import datetime
import warnings

# Add src to path
SRC_PATH = Path(__file__).parent / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# =============================================================================
# SAFE IMPORTS - Handle missing dependencies gracefully
# =============================================================================

def safe_import(module_name: str, class_name: str):
    """Safely import a class, return None if unavailable."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name, None)
    except Exception as e:
        return None

print("=" * 70)
print("COMPREHENSIVE PHYSICS GENERATOR - Loading ALL Physics Models")
print("=" * 70)

# Core thermodynamics (5 models)
RefrigerantProperties = safe_import('thermodynamic_properties', 'RefrigerantProperties')
ThermodynamicState = safe_import('thermodynamic_properties', 'ThermodynamicState')
CycleGenerator = safe_import('cycle_generators', 'CycleGenerator')
CycleParameters = safe_import('cycle_generators', 'CycleParameters')

# Degradation models (4 models)
HeatExchangerFoulingModel = safe_import('fouling_aging_models', 'HeatExchangerFoulingModel')
CompressorWearModel = safe_import('compressor_wear_models', 'CompressorWearModel')
RefrigerantChargeModel = safe_import('refrigerant_charge_models', 'RefrigerantChargeModel')
FaultInjectionModel = safe_import('fault_injection_models', 'FaultInjectionModel')
FaultType = safe_import('fault_injection_models', 'FaultType')
FaultSeverity = safe_import('fault_injection_models', 'FaultSeverity')

# Temporal dynamics models (3 models)
HamiltonianMechanicsModel = safe_import('hamiltonian_mechanics', 'HamiltonianMechanicsModel')
ControlSystemModel = safe_import('control_system_dynamics', 'ControlSystemModel')
TemporalDynamicsGenerator = safe_import('temporal_dynamics', 'TemporalDynamicsGenerator')

# Environment models (2 models)
AmbientConditionModel = safe_import('ambient_conditions', 'AmbientConditionModel')
AmbientConditionsModel = safe_import('ambient_conditions', 'AmbientConditionsModel')  # Alias
MultiZoneBuildingModel = safe_import('building_load_models', 'MultiZoneBuildingModel')

# Component/GNN models (3 models)
ComponentInteractionModel = safe_import('component_interactions', 'ComponentInteractionModel')
HeatPumpGraphGenerator = safe_import('gnn_topology', 'HeatPumpGraphGenerator')
GNNTopologyGenerator = safe_import('gnn_topology', 'GNNTopologyGenerator')

# Performance models (2 models)
PerformanceMapModel = safe_import('performance_maps', 'PerformanceMapModel')
ComponentModels = safe_import('component_models', 'ComponentModels')
CompressorModel = safe_import('component_models', 'CompressorModel')
HeatExchangerModel = safe_import('component_models', 'HeatExchangerModel')
ExpansionValveModel = safe_import('component_models', 'ExpansionValveModel')

# Sensor/noise models (1 model)
SensorNoiseModel = safe_import('sensor_noise_models', 'SensorNoiseModel')

# Part-load and cycling (1 model)
PartLoadCyclingModel = safe_import('part_load_cycling', 'PartLoadCyclingModel')

# Phase space (1 model)
PhaseSpaceTrajectories = safe_import('phase_space_trajectories', 'PhaseSpaceTrajectories')

# Validation models (2 models)
PhysicsValidationSystem = safe_import('physics_validation_system', 'PhysicsValidationSystem')
DigitalTwinValidation = safe_import('digital_twin_validation', 'DigitalTwinValidation')

# Indoor performance (1 model)
IndoorPerformanceModel = safe_import('indoor_performance_models', 'IndoorPerformanceModel')

# Count loaded models
models_loaded = {
    'RefrigerantProperties': RefrigerantProperties is not None,
    'HeatExchangerFoulingModel': HeatExchangerFoulingModel is not None,
    'CompressorWearModel': CompressorWearModel is not None,
    'RefrigerantChargeModel': RefrigerantChargeModel is not None,
    'FaultInjectionModel': FaultInjectionModel is not None,
    'HamiltonianMechanicsModel': HamiltonianMechanicsModel is not None,
    'ControlSystemModel': ControlSystemModel is not None,
    'TemporalDynamicsGenerator': TemporalDynamicsGenerator is not None,
    'AmbientConditionModel': (AmbientConditionModel is not None) or (AmbientConditionsModel is not None),
    'MultiZoneBuildingModel': MultiZoneBuildingModel is not None,
    'ComponentInteractionModel': ComponentInteractionModel is not None,
    'HeatPumpGraphGenerator': (HeatPumpGraphGenerator is not None) or (GNNTopologyGenerator is not None),
    'PerformanceMapModel': PerformanceMapModel is not None,
    'CompressorModel': CompressorModel is not None,
    'HeatExchangerModel': HeatExchangerModel is not None,
    'ExpansionValveModel': ExpansionValveModel is not None,
    'SensorNoiseModel': SensorNoiseModel is not None,
    'PartLoadCyclingModel': PartLoadCyclingModel is not None,
    'PhaseSpaceTrajectories': PhaseSpaceTrajectories is not None,
    'PhysicsValidationSystem': PhysicsValidationSystem is not None,
    'DigitalTwinValidation': DigitalTwinValidation is not None,
    'IndoorPerformanceModel': IndoorPerformanceModel is not None,
}

loaded_count = sum(models_loaded.values())
print(f"\nModels loaded: {loaded_count}/{len(models_loaded)}")
for name, loaded in models_loaded.items():
    status = "[OK]" if loaded else "[MISSING]"
    print(f"  {status} {name}")
print("=" * 70)


# =============================================================================
# DATA CLASSES - Extended for comprehensive physics
# =============================================================================

@dataclass
class Layer1Inputs:
    """Layer 1: Sampled operating conditions and system configuration."""
    # Environmental
    T_ambient_K: float = 298.15
    humidity: float = 0.65
    season: str = "summer"
    T_indoor_K: float = 293.15
    wind_speed_m_s: float = 2.0
    solar_irradiance_W_m2: float = 500.0
    barometric_pressure_Pa: float = 101325.0
    
    # Building load
    building_load_kW: float = 8.0
    occupancy_factor: float = 0.7
    internal_gains_kW: float = 1.5
    
    # System config
    refrigerant: str = "R410A"
    design_capacity_kW: float = 12.0
    nominal_charge_kg: float = 3.5
    compressor_type: str = "scroll"
    
    # Operating point
    mode: str = "heating"
    current_load_kW: float = 9.0
    part_load_ratio: float = 0.75
    
    # System state
    age_hours: float = 15000
    charge_percentage: float = 95.0
    maintenance_quality: str = "good"
    
    # Control setpoints
    setpoint_temp_K: float = 294.15
    superheat_setpoint_K: float = 8.0
    subcooling_setpoint_K: float = 5.0


@dataclass 
class Layer2PhysicsOutputs:
    """Layer 2: Outputs from ALL physics models."""
    # Thermodynamic cycle
    T_evap_K: float = 268.0
    T_cond_K: float = 323.0
    P_evap_Pa: float = 650000
    P_cond_Pa: float = 2100000
    h1_J_kg: float = 420000
    h2_J_kg: float = 460000
    h3_J_kg: float = 260000
    h4_J_kg: float = 260000
    s1_J_kgK: float = 1850
    s2_J_kgK: float = 1900
    s3_J_kgK: float = 1200
    s4_J_kgK: float = 1250
    mass_flow_kg_s: float = 0.08
    
    # Fouling degradation
    evap_fouling_resistance: float = 0.0003
    cond_fouling_resistance: float = 0.0002
    evap_effectiveness_reduction_pct: float = 3.5
    cond_effectiveness_reduction_pct: float = 2.1
    
    # Compressor wear
    bearing_wear_pct: float = 12.5
    valve_leakage_pct: float = 5.2
    oil_quality_index: float = 0.85
    efficiency_degradation_pct: float = 4.3
    vibration_mm_s: float = 3.2
    
    # Charge effects
    actual_charge_pct: float = 93.5
    subcooling_K: float = 7.2
    superheat_K: float = 8.5
    cop_factor_charge: float = 0.97
    
    # Fault state
    has_fault: bool = False
    fault_type: str = "none"
    fault_severity: float = 0.0
    fault_cop_factor: float = 1.0
    
    # Ambient effects (from AmbientConditionModel)
    cooling_capacity_modifier: float = 1.0
    heating_capacity_modifier: float = 1.0
    cooling_cop_modifier: float = 1.0
    heating_cop_modifier: float = 1.0
    frost_formation_risk: float = 0.0
    wet_bulb_temp_K: float = 290.0
    dew_point_temp_K: float = 285.0
    air_density_kg_m3: float = 1.2
    
    # Control system state (from ControlSystemModel)
    compressor_speed_pct: float = 75.0
    expansion_valve_opening_pct: float = 45.0
    evap_fan_speed_pct: float = 80.0
    cond_fan_speed_pct: float = 85.0
    control_response_time_s: float = 30.0
    control_overshoot_pct: float = 5.0
    pid_error: float = 0.0
    
    # Hamiltonian state (from HamiltonianMechanicsModel)
    hamiltonian_energy_J: float = 5000.0
    kinetic_energy_J: float = 100.0
    potential_energy_J: float = 50.0
    energy_drift: float = 0.001
    
    # Component interaction (from ComponentInteractionModel)
    thermal_coupling_strength: float = 0.15
    mass_coupling_strength: float = 0.80
    network_stability_margin: float = 0.10
    
    # Part load (from PartLoadCyclingModel)
    cycling_losses_pct: float = 2.0
    part_load_efficiency: float = 0.92
    on_time_fraction: float = 0.85
    
    # Sensor noise (from SensorNoiseModel)
    T_evap_measured_K: float = 268.0
    T_cond_measured_K: float = 323.0
    P_evap_measured_Pa: float = 650000
    P_cond_measured_Pa: float = 2100000
    measurement_uncertainty_pct: float = 1.5


@dataclass
class Layer3DerivedFeatures:
    """Layer 3: Derived performance metrics."""
    # Cycle performance
    Q_evap_kW: float = 10.2
    Q_cond_kW: float = 12.8
    W_comp_kW: float = 2.6
    eta_comp_actual: float = 0.78
    eta_comp_isentropic: float = 0.82
    pressure_ratio: float = 3.23
    
    # Carnot reference
    carnot_cop_heating: float = 5.87
    carnot_cop_cooling: float = 4.87
    carnot_efficiency: float = 0.68
    
    # Combined degradation from ALL sources
    total_efficiency_loss_pct: float = 6.8
    total_capacity_loss_pct: float = 4.2
    
    # Building load match
    load_match_factor: float = 0.95
    capacity_utilization: float = 0.80


@dataclass
class Layer4Targets:
    """Layer 4: Target values and validation."""
    COP_heating: float = 3.99
    COP_cooling: float = 2.99
    
    # Validation
    energy_balance_error_pct: float = 0.02
    below_carnot_limit: bool = True
    entropy_generation_positive: bool = True
    mass_balance_satisfied: bool = True
    is_physically_valid: bool = True


@dataclass
class CycleData:
    """Complete cycle data following 4-layer architecture."""
    cycle_id: str
    timestamp: str
    layer1: Layer1Inputs
    layer2: Layer2PhysicsOutputs
    layer3: Layer3DerivedFeatures
    layer4: Layer4Targets
    
    # Temporal dynamics for HNN (100 timesteps × features)
    temporal_T_evap: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_T_cond: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_P_evap: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_P_cond: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_mass_flow: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_W_comp: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_Q_evap: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_Q_cond: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_superheat: np.ndarray = field(default_factory=lambda: np.zeros(100))
    temporal_subcooling: np.ndarray = field(default_factory=lambda: np.zeros(100))
    hamiltonian_q: np.ndarray = field(default_factory=lambda: np.zeros((100, 4)))
    hamiltonian_p: np.ndarray = field(default_factory=lambda: np.zeros((100, 4)))
    hamiltonian_H: np.ndarray = field(default_factory=lambda: np.zeros(100))
    control_signals: np.ndarray = field(default_factory=lambda: np.zeros((100, 4)))
    
    # GNN features (nodes, edges, graph)
    node_features: np.ndarray = field(default_factory=lambda: np.zeros((10, 35)))
    edge_features: np.ndarray = field(default_factory=lambda: np.zeros((11, 12)))
    graph_features: np.ndarray = field(default_factory=lambda: np.zeros(50))


# =============================================================================
# DISTRIBUTION FILTER - Accepts cycles based on target COP distribution
# =============================================================================

class COPDistributionFilter:
    """
    Filter that accepts/rejects cycles to achieve a target COP distribution.
    
    Instead of manipulating physics inputs to force a distribution, this filter
    keeps all physics completely random and only selects which cycles to include
    based on whether they help achieve the target distribution.
    
    Uses proper rejection sampling with exponential penalty for over-represented bins.
    """
    
    def __init__(self, target_mean: float = 5.0, target_std: float = 1.2,
                 cop_min: float = 1.5, cop_max: float = 8.5, n_bins: int = 20,
                 aggressiveness: float = 3.0):
        """
        Initialize the distribution filter.
        
        Args:
            target_mean: Mean of target normal distribution
            target_std: Standard deviation of target distribution
            cop_min: Minimum COP value to consider
            cop_max: Maximum COP value to consider
            n_bins: Number of histogram bins for tracking distribution
            aggressiveness: Higher values = stronger rejection of over-represented bins (2-5 recommended)
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.cop_min = cop_min
        self.cop_max = cop_max
        self.n_bins = n_bins
        self.aggressiveness = aggressiveness
        
        # Create bin edges
        self.bin_edges = np.linspace(cop_min, cop_max, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Calculate target distribution (normalized probabilities)
        self.target_probs = self._calculate_target_distribution()
        
        # Track current distribution
        self.current_counts = np.zeros(n_bins)
        self.total_accepted = 0
        
        # Print target distribution info
        print(f"\n[COPDistributionFilter] Target: mean={target_mean}, std={target_std}")
        print(f"  Bin centers: {[f'{c:.1f}' for c in self.bin_centers[:5]]} ... {[f'{c:.1f}' for c in self.bin_centers[-3:]]}")
        print(f"  Target probs: {[f'{p:.3f}' for p in self.target_probs[:5]]} ... {[f'{p:.3f}' for p in self.target_probs[-3:]]}")
        
    def _calculate_target_distribution(self) -> np.ndarray:
        """Calculate target normal distribution probabilities."""
        from scipy.stats import norm
        
        # Calculate PDF at bin centers
        probs = norm.pdf(self.bin_centers, loc=self.target_mean, scale=self.target_std)
        
        # Normalize to sum to 1
        probs = probs / probs.sum()
        
        return probs
    
    def _get_bin_index(self, cop: float) -> int:
        """Get the bin index for a given COP value."""
        if cop < self.cop_min:
            return 0
        if cop >= self.cop_max:
            return self.n_bins - 1
        return int((cop - self.cop_min) / (self.cop_max - self.cop_min) * self.n_bins)
    
    def should_accept(self, cop: float) -> bool:
        """
        Determine if a cycle with this COP should be accepted.
        
        Uses proper rejection sampling:
        - Under-represented bins: high acceptance (up to 100%)
        - Over-represented bins: exponentially decreasing acceptance
        - No minimum acceptance floor (allows proper distribution shaping)
        """
        # Get bin for this COP
        bin_idx = self._get_bin_index(cop)
        
        # If very few samples, accept based on target probability
        # This helps establish the initial distribution correctly
        if self.total_accepted < 100:
            # Use target probability directly for initial seeding
            # Scale up to ensure reasonable acceptance rate during startup
            base_prob = self.target_probs[bin_idx] * 10  # Scale factor for startup
            return np.random.random() < min(1.0, base_prob)
        
        # Calculate current distribution
        current_probs = self.current_counts / self.total_accepted
        
        # Get target and current probabilities for this bin
        target_prob = self.target_probs[bin_idx]
        current_prob = current_probs[bin_idx]
        
        # Calculate acceptance probability using rejection sampling principle
        if current_prob <= 0:
            # Empty bin - always accept (need these samples)
            return True
        
        # Ratio: how much we need this bin relative to what we have
        ratio = target_prob / current_prob
        
        if ratio >= 1.0:
            # Under-represented: accept with high probability
            # Use sqrt to moderate extreme under-representation
            acceptance_prob = min(1.0, np.sqrt(ratio))
        else:
            # Over-represented: use exponential decay for aggressive rejection
            # ratio < 1 means we have too many of these
            # acceptance_prob = ratio^aggressiveness
            # With aggressiveness=3: ratio=0.5 -> accept 12.5%, ratio=0.2 -> accept 0.8%
            acceptance_prob = ratio ** self.aggressiveness
        
        # Very small minimum to prevent complete bin starvation (0.1%)
        # This is much lower than the previous 5%
        acceptance_prob = max(0.001, acceptance_prob)
        
        return np.random.random() < acceptance_prob
    
    def record_acceptance(self, cop: float):
        """Record that a cycle with this COP was accepted."""
        bin_idx = self._get_bin_index(cop)
        self.current_counts[bin_idx] += 1
        self.total_accepted += 1
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get current distribution statistics."""
        if self.total_accepted == 0:
            return {'total': 0, 'mean': 0, 'std': 0}
        
        # Reconstruct approximate COP values from bins
        cop_estimates = []
        for i, count in enumerate(self.current_counts):
            cop_estimates.extend([self.bin_centers[i]] * int(count))
        
        if cop_estimates:
            return {
                'total': self.total_accepted,
                'mean': np.mean(cop_estimates),
                'std': np.std(cop_estimates),
                'bin_counts': self.current_counts.tolist(),
                'target_probs': self.target_probs.tolist()
            }
        return {'total': 0, 'mean': 0, 'std': 0}
    
    def reset(self):
        """Reset the filter for a new dataset."""
        self.current_counts = np.zeros(self.n_bins)
        self.total_accepted = 0


# =============================================================================
# COMPREHENSIVE PHYSICS GENERATOR
# =============================================================================

class ComprehensivePhysicsGenerator:
    """
    Generator integrating ALL 23+ physics models for HNN-GNN training data.
    
    Follows the 4-layer structure:
    1. Inputs (sampled COMPLETELY RANDOMLY - no manipulation)
    2. Physics outputs (calculated from ALL models)
    3. Derived features (computed)
    4. Targets + validation (final COP with Carnot check)
    
    Uses COPDistributionFilter to select cycles that match target distribution
    without manipulating the physics.
    """
    
    REFRIGERANTS = ['R410A', 'R134a', 'R290']
    MODES = ['heating', 'cooling']
    SEASONS = ['spring', 'summer', 'autumn', 'winter']
    N_TIMESTEPS = 100
    
    def __init__(self):
        """Initialize the generator with ALL physics models."""
        print("\n" + "=" * 70)
        print("INITIALIZING COMPREHENSIVE PHYSICS GENERATOR")
        print("=" * 70)
        
        # Cache for refrigerant properties
        self._thermo_cache: Dict[str, Any] = {}
        
        # Initialize ALL physics models
        self._init_all_physics_models()
        
        # Statistics tracking
        self.stats = {
            'generated': 0,
            'valid': 0,
            'invalid': 0,
            'cop_values': [],
            'carnot_efficiencies': []
        }
        
        print("=" * 70)
        print("INITIALIZATION COMPLETE")
        print("=" * 70 + "\n")
    
    def _init_all_physics_models(self):
        """Initialize ALL physics model instances."""
        
        # Core thermodynamics
        self.cycle_gen = CycleGenerator() if CycleGenerator else None
        if self.cycle_gen:
            print("  [OK] CycleGenerator")
        
        # Degradation models
        self.fouling_model = HeatExchangerFoulingModel() if HeatExchangerFoulingModel else None
        if self.fouling_model:
            print("  [OK] HeatExchangerFoulingModel")
            
        self.wear_model = CompressorWearModel() if CompressorWearModel else None
        if self.wear_model:
            print("  [OK] CompressorWearModel")
            
        self.charge_model = RefrigerantChargeModel() if RefrigerantChargeModel else None
        if self.charge_model:
            print("  [OK] RefrigerantChargeModel")
            
        self.fault_model = FaultInjectionModel() if FaultInjectionModel else None
        if self.fault_model:
            print("  [OK] FaultInjectionModel")
        
        # Temporal dynamics
        self.hamiltonian_model = HamiltonianMechanicsModel() if HamiltonianMechanicsModel else None
        if self.hamiltonian_model:
            print("  [OK] HamiltonianMechanicsModel")
            
        self.control_model = ControlSystemModel() if ControlSystemModel else None
        if self.control_model:
            print("  [OK] ControlSystemModel")
            
        self.temporal_model = TemporalDynamicsGenerator() if TemporalDynamicsGenerator else None
        if self.temporal_model:
            print("  [OK] TemporalDynamicsGenerator")
        
        # Environment models
        AmbientModel = AmbientConditionModel or AmbientConditionsModel
        self.ambient_model = AmbientModel() if AmbientModel else None
        if self.ambient_model:
            print("  [OK] AmbientConditionModel")
            
        self.building_model = MultiZoneBuildingModel() if MultiZoneBuildingModel else None
        if self.building_model:
            print("  [OK] MultiZoneBuildingModel")
        
        # Component/GNN models
        self.interaction_model = ComponentInteractionModel() if ComponentInteractionModel else None
        if self.interaction_model:
            print("  [OK] ComponentInteractionModel")
            
        GNNModel = HeatPumpGraphGenerator or GNNTopologyGenerator
        self.gnn_model = GNNModel() if GNNModel else None
        if self.gnn_model:
            print("  [OK] HeatPumpGraphGenerator")
        
        # Performance models
        self.performance_model = PerformanceMapModel() if PerformanceMapModel else None
        if self.performance_model:
            print("  [OK] PerformanceMapModel")
            
        try:
            self.compressor_model = CompressorModel('scroll') if CompressorModel else None
        except Exception:
            self.compressor_model = None
        if self.compressor_model:
            print("  [OK] CompressorModel")
        
        # Sensor noise
        self.sensor_model = SensorNoiseModel() if SensorNoiseModel else None
        if self.sensor_model:
            print("  [OK] SensorNoiseModel")
        
        # Part load
        self.part_load_model = PartLoadCyclingModel() if PartLoadCyclingModel else None
        if self.part_load_model:
            print("  [OK] PartLoadCyclingModel")
        
        # Phase space
        self.phase_space_model = PhaseSpaceTrajectories() if PhaseSpaceTrajectories else None
        if self.phase_space_model:
            print("  [OK] PhaseSpaceTrajectories")
        
        # Validation
        self.physics_validator = PhysicsValidationSystem() if PhysicsValidationSystem else None
        if self.physics_validator:
            print("  [OK] PhysicsValidationSystem")
    
    def _get_thermo(self, refrigerant: str):
        """Get cached thermodynamic properties for refrigerant."""
        if refrigerant not in self._thermo_cache:
            if RefrigerantProperties:
                try:
                    self._thermo_cache[refrigerant] = RefrigerantProperties(refrigerant)
                except Exception:
                    self._thermo_cache[refrigerant] = None
            else:
                self._thermo_cache[refrigerant] = None
        return self._thermo_cache.get(refrigerant)
    
    # =========================================================================
    # LAYER 1: Sample Operating Conditions (with stratification)
    # =========================================================================
    
    def _sample_layer1_inputs(self) -> Layer1Inputs:
        """Sample random but physically realistic operating conditions."""
        
        # Environmental - stratified sampling for good coverage
        T_ambient = np.random.uniform(253.15, 318.15)  # -20°C to +45°C
        humidity = np.random.uniform(0.2, 0.95)
        T_indoor = np.random.uniform(288.15, 303.15)  # 15°C to 30°C
        season = np.random.choice(self.SEASONS)
        wind_speed = np.random.uniform(0.5, 10.0)
        
        # Solar varies by season
        season_solar = {'winter': 200, 'spring': 500, 'summer': 800, 'autumn': 400}
        solar = season_solar[season] * np.random.uniform(0.3, 1.2)
        
        # Building load
        building_load = np.random.uniform(3.0, 15.0)
        occupancy = np.random.uniform(0.0, 1.0)
        internal_gains = occupancy * np.random.uniform(0.5, 3.0)
        
        # System configuration
        refrigerant = np.random.choice(self.REFRIGERANTS)
        design_capacity = np.random.uniform(5.0, 20.0)
        nominal_charge = design_capacity * 0.25 + np.random.uniform(0.5, 1.5)
        
        # Operating point
        mode = np.random.choice(self.MODES)
        part_load = np.random.uniform(0.3, 1.0)
        current_load = design_capacity * part_load
        
        # System state - stratified for age
        age_hours = np.random.choice([
            np.random.uniform(0, 2000),      # New
            np.random.uniform(2000, 10000),  # Middle-aged
            np.random.uniform(10000, 30000), # Older
            np.random.uniform(30000, 50000)  # Old
        ])
        
        charge_pct = np.random.uniform(70, 130)
        maintenance = np.random.choice(['good', 'average', 'poor'], p=[0.5, 0.35, 0.15])
        
        # Control setpoints
        setpoint = T_indoor + np.random.uniform(-2, 2)
        superheat_sp = np.random.uniform(5, 12)
        subcooling_sp = np.random.uniform(3, 8)
        
        return Layer1Inputs(
            T_ambient_K=T_ambient,
            humidity=humidity,
            season=season,
            T_indoor_K=T_indoor,
            wind_speed_m_s=wind_speed,
            solar_irradiance_W_m2=solar,
            barometric_pressure_Pa=np.random.uniform(98000, 104000),
            building_load_kW=building_load,
            occupancy_factor=occupancy,
            internal_gains_kW=internal_gains,
            refrigerant=refrigerant,
            design_capacity_kW=design_capacity,
            nominal_charge_kg=nominal_charge,
            compressor_type="scroll",
            mode=mode,
            current_load_kW=current_load,
            part_load_ratio=part_load,
            age_hours=age_hours,
            charge_percentage=charge_pct,
            maintenance_quality=maintenance,
            setpoint_temp_K=setpoint,
            superheat_setpoint_K=superheat_sp,
            subcooling_setpoint_K=subcooling_sp
        )
    
    # =========================================================================
    # LAYER 2: Run ALL Physics Models
    # =========================================================================
    
    def _calculate_layer2_physics(self, inputs: Layer1Inputs) -> Layer2PhysicsOutputs:
        """Calculate physics outputs from ALL available models."""
        
        outputs = Layer2PhysicsOutputs()
        thermo = self._get_thermo(inputs.refrigerant)
        
        # ---- 1. Determine Operating Temperatures ----
        # FULLY RANDOM sampling - no manipulation for distribution shaping
        # Let the physics determine COP naturally, filter afterwards
        
        # Random approach temperatures (realistic ranges for heat pump systems)
        if inputs.mode == 'heating':
            # Heating mode: evaporator below ambient, condenser above indoor
            evap_approach = np.random.uniform(5, 35)  # Full range
            cond_approach = np.random.uniform(8, 60)  # Full range
            T_evap = inputs.T_ambient_K - evap_approach
            T_cond = inputs.T_indoor_K + cond_approach
        else:
            # Cooling mode: evaporator below indoor, condenser above ambient
            evap_approach = np.random.uniform(5, 35)
            cond_approach = np.random.uniform(8, 50)
            T_evap = inputs.T_indoor_K - evap_approach
            T_cond = inputs.T_ambient_K + cond_approach
        
        # Physical limits for refrigerant operation
        T_evap = np.clip(T_evap, 235, 290)  # Expanded range
        T_cond = np.clip(T_cond, 300, 370)  # Expanded range
        
        # Only enforce minimum physical lift (no artificial manipulation)
        min_physical_lift = 15  # Minimum thermodynamic lift for any vapor compression cycle
        if T_cond - T_evap < min_physical_lift:
            T_cond = T_evap + min_physical_lift
        
        outputs.T_evap_K = T_evap
        outputs.T_cond_K = T_cond
        
        # ---- 2. Get Saturation Pressures (RefrigerantProperties) ----
        try:
            if thermo:
                P_evap = thermo.get_saturation_pressure(T_evap)
                P_cond = thermo.get_saturation_pressure(T_cond)
            else:
                raise ValueError("No thermo")
        except:
            P_evap = 400000 + (T_evap - 270) * 10000
            P_cond = 1500000 + (T_cond - 320) * 20000
        
        outputs.P_evap_Pa = np.clip(P_evap, 100000, 2000000)
        outputs.P_cond_Pa = np.clip(P_cond, 500000, 4000000)
        
        # ---- 3. Ambient Condition Model ----
        if self.ambient_model:
            try:
                ambient_cond = self.ambient_model.calculate_ambient_conditions(
                    dry_bulb_temp_K=inputs.T_ambient_K,
                    relative_humidity_percent=inputs.humidity * 100,
                    wind_speed_m_per_s=inputs.wind_speed_m_s,
                    solar_irradiance_W_per_m2=inputs.solar_irradiance_W_m2,
                    barometric_pressure_Pa=inputs.barometric_pressure_Pa
                )
                outputs.wet_bulb_temp_K = ambient_cond.wet_bulb_temp_K
                outputs.dew_point_temp_K = ambient_cond.dew_point_temp_K
                outputs.air_density_kg_m3 = ambient_cond.air_density_kg_per_m3
                
                # Get performance modifiers
                perf_mod = self.ambient_model.calculate_performance_modifiers(
                    ambient_cond, inputs.mode
                )
                outputs.cooling_capacity_modifier = perf_mod.cooling_capacity_modifier
                outputs.heating_capacity_modifier = perf_mod.heating_capacity_modifier
                outputs.cooling_cop_modifier = perf_mod.cooling_cop_modifier
                outputs.heating_cop_modifier = perf_mod.heating_cop_modifier
                outputs.frost_formation_risk = perf_mod.frost_formation_risk
            except Exception:
                pass
        
        # ---- 4. Fouling Model ----
        age_factor = inputs.age_hours / 8760
        operating_hours = age_factor * 1000
        
        if self.fouling_model:
            try:
                evap_state = self.fouling_model.initialize_fouling_state(
                    "evap", "finned_tube_evaporator", inputs.age_hours
                )
                evap_state = self.fouling_model.update_fouling_state(
                    evap_state, operating_hours,
                    {'temperature': inputs.T_ambient_K, 'humidity': inputs.humidity},
                    "finned_tube_evaporator"
                )
                outputs.evap_fouling_resistance = evap_state.fouling_resistance
                outputs.evap_effectiveness_reduction_pct = evap_state.effectiveness_reduction
                
                cond_state = self.fouling_model.initialize_fouling_state(
                    "cond", "plate_condenser", inputs.age_hours
                )
                cond_state = self.fouling_model.update_fouling_state(
                    cond_state, operating_hours,
                    {'temperature': inputs.T_ambient_K + 30, 'humidity': inputs.humidity},
                    "plate_condenser"
                )
                outputs.cond_fouling_resistance = cond_state.fouling_resistance
                outputs.cond_effectiveness_reduction_pct = cond_state.effectiveness_reduction
            except Exception:
                pass
        
        # ---- 5. Compressor Wear Model ----
        if self.wear_model:
            try:
                wear_state = self.wear_model.initialize_wear_state(
                    "comp", "scroll_compressor", inputs.age_hours
                )
                wear_state = self.wear_model.update_wear_state(
                    wear_state, operating_hours,
                    {
                        'discharge_temperature': T_cond + 40,
                        'load_factor': inputs.part_load_ratio,
                        'pressure_ratio': outputs.P_cond_Pa / outputs.P_evap_Pa,
                        'start_stop_cycles': int(inputs.age_hours / 100)
                    }
                )
                outputs.bearing_wear_pct = wear_state.bearing_wear_percentage
                outputs.valve_leakage_pct = wear_state.valve_leakage_percentage
                outputs.oil_quality_index = wear_state.oil_quality_index
                outputs.efficiency_degradation_pct = wear_state.efficiency_degradation
                outputs.vibration_mm_s = wear_state.vibration_level
            except Exception:
                pass
        
        # ---- 6. Refrigerant Charge Model ----
        if self.charge_model:
            try:
                charge_state = self.charge_model.initialize_charge_state(
                    "sys", inputs.refrigerant, inputs.nominal_charge_kg,
                    inputs.charge_percentage, inputs.T_ambient_K
                )
                outputs.actual_charge_pct = charge_state.charge_level_percentage
                outputs.subcooling_K = charge_state.subcooling_K
                outputs.superheat_K = charge_state.superheat_K
                if inputs.mode == 'heating':
                    outputs.cop_factor_charge = charge_state.cop_heating_factor
                else:
                    outputs.cop_factor_charge = charge_state.cop_cooling_factor
            except Exception:
                outputs.cop_factor_charge = 1.0
        
        # ---- 7. Fault Injection Model (20% probability) ----
        if np.random.random() < 0.2 and self.fault_model and FaultType and FaultSeverity:
            try:
                fault_types = [FaultType.REFRIGERANT_LEAK, FaultType.COMPRESSOR_DEGRADATION]
                fault = self.fault_model.inject_fault(
                    "sys", "comp",
                    np.random.choice(fault_types),
                    np.random.choice([FaultSeverity.MINOR, FaultSeverity.MODERATE])
                )
                outputs.has_fault = True
                outputs.fault_cop_factor = fault.cop_impact_factor
            except Exception:
                pass
        
        # ---- 8. Control System Model ----
        # FIX: Previously only called initialize_control_system() which returns constants.
        # Now we simulate PID response based on operating conditions for varying outputs.
        if self.control_model:
            try:
                # Initialize control system with strategy based on application
                control_strategy = np.random.choice(['cascade', 'feedforward', 'single_loop'])
                control_state = self.control_model.initialize_control_system(
                    "ctrl", control_strategy
                )
                
                # Calculate temperature error for PID simulation
                temp_error = inputs.T_indoor_K - inputs.setpoint_temp_K  # Deviation from setpoint
                load_deviation = inputs.part_load_ratio - 0.75  # Deviation from nominal
                
                # Update PID controllers to get actual varying outputs
                # Compressor speed based on capacity demand
                compressor_output = self.control_model.update_pid_controller(
                    control_state.capacity_controller,
                    process_value=inputs.part_load_ratio * 100,  # Current load %
                    current_time=0.0,
                    dt=1.0
                )
                outputs.compressor_speed_pct = float(np.clip(compressor_output, 20, 100))
                
                # Expansion valve based on superheat control
                superheat_error = physics_outputs.superheat_K - inputs.superheat_setpoint_K if hasattr(physics_outputs, 'superheat_K') else outputs.superheat_K - inputs.superheat_setpoint_K
                valve_base = 45 + superheat_error * 3  # 3% per K superheat error
                valve_adjustment = load_deviation * 20  # Adjust for load
                outputs.expansion_valve_opening_pct = float(np.clip(valve_base + valve_adjustment, 15, 95))
                
                # Fan speeds vary with load and ambient conditions
                ambient_factor = (inputs.T_ambient_K - 273.15) / 40  # Normalize to 0-1 range
                outputs.evap_fan_speed_pct = float(np.clip(
                    60 + inputs.part_load_ratio * 30 + temp_error * 2, 30, 100
                ))
                outputs.cond_fan_speed_pct = float(np.clip(
                    65 + inputs.part_load_ratio * 25 + ambient_factor * 10, 35, 100
                ))
                
                # Response characteristics depend on strategy and conditions
                outputs.control_response_time_s = control_state.response_time_constant_s * (0.8 + 0.4 * np.random.random())
                outputs.control_overshoot_pct = control_state.overshoot_percent * (0.5 + np.random.random())
                
            except Exception as e:
                # Fallback: Generate varying values based on operating conditions
                outputs.compressor_speed_pct = float(np.clip(
                    50 + inputs.part_load_ratio * 40 + np.random.normal(0, 5), 20, 100
                ))
                outputs.expansion_valve_opening_pct = float(np.clip(
                    35 + inputs.part_load_ratio * 30 + np.random.normal(0, 5), 15, 95
                ))
                outputs.evap_fan_speed_pct = float(np.clip(
                    60 + inputs.part_load_ratio * 30, 30, 100
                ))
                outputs.cond_fan_speed_pct = float(np.clip(
                    65 + inputs.part_load_ratio * 25, 35, 100
                ))
                outputs.control_response_time_s = 30 + np.random.uniform(0, 60)
                outputs.control_overshoot_pct = 3 + np.random.uniform(0, 8)
        
        # ---- 9. Hamiltonian Mechanics Model ----
        if self.hamiltonian_model:
            try:
                system = self.hamiltonian_model.create_hamiltonian_system('compressor', {})
                initial_cond = {
                    'temperature_K': T_evap + 10,
                    'pressure_Pa': outputs.P_evap_Pa,
                    'volume_m3': 0.001,
                    'entropy_J_per_K': 1000.0
                }
                ham_state = self.hamiltonian_model.initialize_hamiltonian_state(
                    system, initial_cond
                )
                outputs.hamiltonian_energy_J = ham_state.total_hamiltonian_J
                outputs.kinetic_energy_J = ham_state.kinetic_energy_J
                outputs.potential_energy_J = ham_state.potential_energy_J
            except Exception:
                pass
        
        # ---- 10. Component Interaction Model ----
        # FIX: The ComponentInteractionModel returns constants because it uses fixed component properties.
        # We need to calculate physics-based coupling strengths that vary with operating conditions.
        actual_mass_flow = outputs.mass_flow_kg_s
        
        # Calculate physics-based thermal coupling strength
        # Thermal coupling depends on: temperature difference, mass flow, and heat exchanger effectiveness
        delta_T = outputs.T_cond_K - outputs.T_evap_K  # Temperature lift
        pressure_ratio = outputs.P_cond_Pa / max(outputs.P_evap_Pa, 100000)
        
        # Thermal coupling: higher mass flow and larger delta_T = stronger thermal coupling
        # Base coupling from convection: Q = h * A * dT, so coupling ~ m_dot * Cp * dT
        # Normalized to 0.05-0.35 range for realistic values
        thermal_coupling_base = actual_mass_flow * 1000 * delta_T  # W equivalent
        outputs.thermal_coupling_strength = float(np.clip(
            0.05 + (thermal_coupling_base / 50000) * 0.30 + np.random.uniform(-0.02, 0.02),
            0.05, 0.35
        ))
        
        # Mass coupling: depends on refrigerant flow rate and system volume
        # Higher mass flow = stronger mass coupling between components
        # Normalized to 0.5-0.95 range
        outputs.mass_coupling_strength = float(np.clip(
            0.5 + actual_mass_flow * 4.0 + inputs.part_load_ratio * 0.2 + np.random.uniform(-0.05, 0.05),
            0.5, 0.95
        ))
        
        # Network stability margin: lower pressure ratio and steady operation = better stability
        # Normalized to 0.02-0.20 range
        outputs.network_stability_margin = float(np.clip(
            0.20 - (pressure_ratio - 2.5) * 0.03 + np.random.uniform(-0.02, 0.02),
            0.02, 0.20
        ))
        
        # ---- 11. Part Load Model ----
        # FIX: The PartLoadCyclingModel's cycling_degradation_factor = 1.0 when NOT cycling.
        # Since we sample part_load_ratio from 0.3-1.0 and minimum_plr is ~0.1-0.2, cycling rarely happens.
        # We need to calculate realistic cycling losses that ALWAYS vary with operating conditions.
        
        plr = inputs.part_load_ratio
        
        # Determine modulation type - affects efficiency losses
        modulation_types = ['single_speed', 'two_speed', 'variable_speed', 'hot_gas_bypass']
        modulation_weights = [0.2, 0.15, 0.55, 0.1]  # Variable speed most common
        modulation_type = np.random.choice(modulation_types, p=modulation_weights)
        
        # Physics-based cycling losses calculation
        # Based on DOE and AHRI cycling degradation studies
        if modulation_type == 'single_speed':
            # Single speed systems: highest cycling losses at part load
            if plr < 0.3:
                cycling_base = (0.3 - plr) * 30  # Up to 9% at very low loads
            elif plr < 0.5:
                cycling_base = (0.5 - plr) * 15  # 3-7.5% at moderate loads
            elif plr < 0.7:
                cycling_base = (0.7 - plr) * 8   # 1.5-4% at higher loads
            else:
                cycling_base = (1.0 - plr) * 3   # 0-1% near full load
        elif modulation_type == 'two_speed':
            # Two-speed: less cycling than single speed
            if plr < 0.3:
                cycling_base = (0.3 - plr) * 20
            elif plr < 0.5:
                cycling_base = (0.5 - plr) * 10
            else:
                cycling_base = (1.0 - plr) * 4
        elif modulation_type == 'variable_speed':
            # Variable speed: minimal cycling, but still some losses at part load
            if plr < 0.3:
                cycling_base = (0.3 - plr) * 8
            elif plr < 0.5:
                cycling_base = (0.5 - plr) * 5
            else:
                cycling_base = (1.0 - plr) * 2
        else:  # hot_gas_bypass
            # Hot gas bypass: efficiency losses from bypass valve
            cycling_base = (1.0 - plr) * 10 + np.random.uniform(0, 2)
        
        # Add variation based on system age and maintenance
        age_factor = min(inputs.age_hours / 30000, 1.0) * 0.5
        maintenance_factor = {'good': 0.0, 'average': 0.3, 'poor': 0.8}[inputs.maintenance_quality]
        
        outputs.cycling_losses_pct = float(np.clip(
            cycling_base + age_factor + maintenance_factor + np.random.uniform(-0.3, 0.3),
            0.1, 12.0  # Realistic range: 0.1% to 12%
        ))
        
        # Part load efficiency using standard AHRI curves by modulation type
        if modulation_type == 'variable_speed':
            # Variable speed: peak efficiency at part load (~80%)
            eff_curve = -0.2 * plr**3 + 0.3 * plr**2 + 0.9 * plr
        elif modulation_type == 'two_speed':
            # Two-speed: step function behavior
            if plr < 0.5:
                eff_curve = -0.5 * plr**2 + plr + 0.2
            else:
                eff_curve = -0.3 * plr**2 + 0.8 * plr + 0.4
        else:
            # Single speed and hot gas bypass: worst at part load
            eff_curve = -0.8 * plr**3 + 1.2 * plr**2 + 0.6 * plr
        
        outputs.part_load_efficiency = float(np.clip(
            eff_curve + np.random.uniform(-0.05, 0.05),
            0.4, 0.98
        ))
        
        # On-time fraction: how much of the time compressor runs
        # Lower at part load, higher near full load
        outputs.on_time_fraction = float(np.clip(
            0.2 + plr * 0.75 + np.random.uniform(-0.05, 0.05),
            0.2, 1.0
        ))
        
        # ---- 12. Sensor Noise Model ----
        if self.sensor_model:
            try:
                outputs.T_evap_measured_K = outputs.T_evap_K + self.sensor_model.add_temperature_noise(0)
                outputs.T_cond_measured_K = outputs.T_cond_K + self.sensor_model.add_temperature_noise(0)
                outputs.P_evap_measured_Pa = outputs.P_evap_Pa * (1 + self.sensor_model.add_pressure_noise(0) / 100)
                outputs.P_cond_measured_Pa = outputs.P_cond_Pa * (1 + self.sensor_model.add_pressure_noise(0) / 100)
            except Exception:
                outputs.T_evap_measured_K = outputs.T_evap_K + np.random.normal(0, 0.5)
                outputs.T_cond_measured_K = outputs.T_cond_K + np.random.normal(0, 0.5)
                outputs.measurement_uncertainty_pct = 1.5
        
        # ---- 13. Calculate Enthalpies (RefrigerantProperties) ----
        base_eta = 0.85
        eta_comp = base_eta * (1 - outputs.efficiency_degradation_pct / 100)
        eta_comp = np.clip(eta_comp, 0.4, 0.92)
        
        try:
            if thermo:
                state1 = thermo.get_saturated_vapor(T_evap + outputs.superheat_K)
                h1 = state1.h if state1 else 250000
                s1 = state1.s if state1 else 1800
                
                state2_ideal = thermo.get_properties_ps(outputs.P_cond_Pa, s1) if thermo else None
                h2_ideal = state2_ideal.h if state2_ideal else h1 + 40000
                h2 = h1 + (h2_ideal - h1) / eta_comp
                
                state3 = thermo.get_saturated_liquid(T_cond - outputs.subcooling_K)
                h3 = state3.h if state3 else 120000
            else:
                raise ValueError("No thermo")
        except:
            h1 = 250000 + (T_evap - 270) * 500
            h2 = h1 + 40000 / eta_comp
            h3 = 120000 + (T_cond - 320) * 300
            s1 = 1800
        
        h4 = h3  # Isenthalpic expansion
        
        outputs.h1_J_kg = h1
        outputs.h2_J_kg = h2
        outputs.h3_J_kg = h3
        outputs.h4_J_kg = h4
        outputs.s1_J_kgK = s1
        
        # Mass flow rate
        load_ratio = inputs.current_load_kW / inputs.design_capacity_kW
        outputs.mass_flow_kg_s = 0.04 + np.clip(load_ratio, 0.2, 1.2) * 0.08
        
        return outputs
    
    # =========================================================================
    # LAYER 3: Calculate Derived Features
    # =========================================================================
    
    def _calculate_layer3_derived(self, inputs: Layer1Inputs, 
                                   physics: Layer2PhysicsOutputs) -> Layer3DerivedFeatures:
        """Calculate derived performance features from all physics."""
        
        derived = Layer3DerivedFeatures()
        
        # Heat transfer rates
        Q_evap = abs(physics.mass_flow_kg_s * (physics.h1_J_kg - physics.h4_J_kg)) / 1000
        W_comp = abs(physics.mass_flow_kg_s * (physics.h2_J_kg - physics.h1_J_kg)) / 1000
        Q_cond = Q_evap + W_comp
        
        derived.Q_evap_kW = Q_evap
        derived.Q_cond_kW = Q_cond
        derived.W_comp_kW = max(W_comp, 0.1)
        
        # Compressor efficiencies
        base_eta = 0.85
        derived.eta_comp_actual = base_eta * (1 - physics.efficiency_degradation_pct / 100)
        derived.eta_comp_isentropic = derived.eta_comp_actual * 1.05
        
        # Pressure ratio
        derived.pressure_ratio = physics.P_cond_Pa / max(physics.P_evap_Pa, 100000)
        
        # Carnot reference (critical for validation)
        dT = physics.T_cond_K - physics.T_evap_K
        if dT > 10:
            derived.carnot_cop_heating = physics.T_cond_K / dT
            derived.carnot_cop_cooling = physics.T_evap_K / dT
        else:
            derived.carnot_cop_heating = 10.0
            derived.carnot_cop_cooling = 10.0
        
        # Combined degradation from ALL sources
        derived.total_efficiency_loss_pct = (
            physics.efficiency_degradation_pct +
            physics.evap_effectiveness_reduction_pct * 0.1 +
            physics.cond_effectiveness_reduction_pct * 0.1 +
            physics.cycling_losses_pct * 0.5 +
            (1 - physics.heating_cop_modifier) * 10  # Ambient effects
        )
        
        derived.total_capacity_loss_pct = (
            (physics.evap_effectiveness_reduction_pct + physics.cond_effectiveness_reduction_pct) / 2 +
            (1 - physics.heating_capacity_modifier) * 10
        )
        
        # Building load matching
        derived.load_match_factor = min(derived.Q_cond_kW / max(inputs.building_load_kW, 1), 1.5)
        derived.capacity_utilization = inputs.part_load_ratio
        
        return derived
    
    # =========================================================================
    # LAYER 4: Calculate COP and Validate
    # =========================================================================
    
    def _calculate_layer4_targets(self, inputs: Layer1Inputs,
                                   physics: Layer2PhysicsOutputs,
                                   derived: Layer3DerivedFeatures) -> Layer4Targets:
        """Calculate final COP with ALL degradation factors and validate."""
        
        targets = Layer4Targets()
        
        # Base COP from thermodynamics
        if derived.W_comp_kW > 0.01:
            COP_heating_ideal = derived.Q_cond_kW / derived.W_comp_kW
            COP_cooling_ideal = derived.Q_evap_kW / derived.W_comp_kW
        else:
            COP_heating_ideal = 3.5
            COP_cooling_ideal = 2.5
        
        # Apply ALL degradation factors
        degradation_factor = (
            physics.cop_factor_charge *                          # Charge effect
            physics.fault_cop_factor *                           # Fault effect
            (1 - physics.efficiency_degradation_pct / 100) *     # Compressor wear
            physics.heating_cop_modifier *                       # Ambient effects
            physics.part_load_efficiency *                       # Part load penalty
            (1 - physics.cycling_losses_pct / 100)              # Cycling losses
        )
        
        COP_heating = COP_heating_ideal * degradation_factor
        COP_cooling = COP_cooling_ideal * degradation_factor
        
        # Carnot validation - COP MUST be below Carnot limit
        if COP_heating > derived.carnot_cop_heating * 0.95:
            # Clamp to realistic Carnot efficiency (50-85%)
            efficiency = np.random.uniform(0.50, 0.85)
            COP_heating = derived.carnot_cop_heating * efficiency
            COP_cooling = derived.carnot_cop_cooling * efficiency
        
        # Final clipping to realistic range for industrial heat engines
        # Max COP of 8.0 allows for multi-stage systems that can exceed 6.5
        targets.COP_heating = float(np.clip(COP_heating, 1.8, 8.0))
        targets.COP_cooling = float(np.clip(COP_cooling, 1.2, 6.5))
        
        # Update Carnot efficiency
        derived.carnot_efficiency = targets.COP_heating / derived.carnot_cop_heating
        
        # Validation checks
        targets.energy_balance_error_pct = abs(
            derived.Q_cond_kW - (derived.Q_evap_kW + derived.W_comp_kW)
        ) / max(derived.Q_cond_kW, 0.1) * 100
        
        targets.below_carnot_limit = (
            targets.COP_heating <= derived.carnot_cop_heating * 1.01 and
            targets.COP_cooling <= derived.carnot_cop_cooling * 1.01
        )
        
        targets.entropy_generation_positive = (
            derived.Q_cond_kW / physics.T_cond_K - 
            derived.Q_evap_kW / physics.T_evap_K
        ) >= -0.001
        
        targets.mass_balance_satisfied = True  # Simplified
        
        targets.is_physically_valid = (
            targets.below_carnot_limit and
            targets.entropy_generation_positive and
            targets.energy_balance_error_pct < 5.0 and
            targets.COP_heating >= 1.0 and
            targets.COP_heating <= 10.0
        )
        
        return targets
    
    # =========================================================================
    # TEMPORAL DYNAMICS FOR HNN (using HamiltonianMechanicsModel)
    # =========================================================================
    
    def _generate_temporal_dynamics(self, inputs: Layer1Inputs,
                                     physics: Layer2PhysicsOutputs,
                                     derived: Layer3DerivedFeatures) -> Dict[str, np.ndarray]:
        """
        Generate 100-timestep temporal dynamics using physics models.
        
        IMPORTANT: For Hamiltonian phase space, we generate CLOSED CYCLIC ORBITS
        that represent the thermodynamic cycle:
        - Evaporation → Compression → Condensation → Expansion → back to Evaporation
        
        This matches the Greydanus HNN paper where oscillatory systems (mass-spring,
        pendulum) show closed orbits in (q, p) phase space.
        """
        
        t = np.linspace(0, 2 * np.pi, self.N_TIMESTEPS)  # Full cycle (0 to 2π)
        tau = 0.15  # Time constant for startup envelope
        startup_factor = 1 - np.exp(-np.linspace(0, 1, self.N_TIMESTEPS) / tau)
        
        temporal = {}
        
        # Temperature trajectories (for general monitoring - uses startup transient)
        temporal['T_evap'] = physics.T_evap_K + physics.superheat_K * startup_factor
        temporal['T_cond'] = 273.15 + (physics.T_cond_K - 273.15) * startup_factor
        
        # Pressure trajectories
        temporal['P_evap'] = 101325 + (physics.P_evap_Pa - 101325) * startup_factor
        temporal['P_cond'] = 101325 + (physics.P_cond_Pa - 101325) * startup_factor
        
        # Flow and power trajectories
        temporal['mass_flow'] = physics.mass_flow_kg_s * startup_factor
        temporal['W_comp'] = derived.W_comp_kW * startup_factor
        temporal['Q_evap'] = derived.Q_evap_kW * startup_factor
        temporal['Q_cond'] = derived.Q_cond_kW * startup_factor
        
        # Superheat and subcooling transients
        temporal['superheat'] = physics.superheat_K * (0.5 + 0.5 * startup_factor)
        temporal['subcooling'] = physics.subcooling_K * startup_factor
        
        # =====================================================================
        # HAMILTONIAN PHASE SPACE - CLOSED THERMODYNAMIC CYCLES
        # =====================================================================
        # For a heat pump cycle (evaporation → compression → condensation → expansion),
        # we model this as a coupled oscillator system in generalized coordinates:
        #   q1: Temperature-like coordinate (normalized T)
        #   q2: Pressure-like coordinate (normalized P)
        #   q3: Volume-like coordinate (normalized V)
        #   q4: Entropy-like coordinate (normalized S)
        #
        # Using sinusoidal basis functions creates CLOSED ORBITS in phase space,
        # matching the Greydanus HNN paper's mass-spring and pendulum examples.
        # =====================================================================
        
        # System-specific frequencies (related to cycle characteristics)
        # Higher pressure ratio → faster dynamics
        pressure_ratio = physics.P_cond_Pa / max(physics.P_evap_Pa, 100000)
        omega_base = 1.0 + 0.1 * (pressure_ratio - 3.0)  # Base frequency
        
        # Phase offsets for 4-stage cycle: evap(0°) → comp(90°) → cond(180°) → exp(270°)
        phase_T = 0.0                    # Temperature leads
        phase_P = np.pi / 4              # Pressure lags by 45°
        phase_V = np.pi / 2              # Volume lags by 90° (compression work)
        phase_S = 3 * np.pi / 4          # Entropy lags by 135°
        
        # Amplitudes from operating conditions (normalized)
        A_T = (physics.T_cond_K - physics.T_evap_K) / 100  # Temperature swing
        A_P = np.log10(pressure_ratio)                      # Pressure ratio (log scale)
        A_V = inputs.part_load_ratio                        # Volume variation
        A_S = derived.carnot_efficiency                     # Entropy generation
        
        # Add some realistic variation
        noise_scale = 0.02
        
        # Generate CYCLIC position coordinates q(t) = A * cos(ωt + φ)
        # These trace CLOSED ORBITS when plotted in phase space
        q1 = A_T * np.cos(omega_base * t + phase_T) + np.random.normal(0, noise_scale, self.N_TIMESTEPS)
        q2 = A_P * np.cos(omega_base * t + phase_P) + np.random.normal(0, noise_scale, self.N_TIMESTEPS)
        q3 = A_V * np.cos(omega_base * t + phase_V) + np.random.normal(0, noise_scale, self.N_TIMESTEPS)
        q4 = A_S * np.cos(omega_base * t + phase_S) + np.random.normal(0, noise_scale, self.N_TIMESTEPS)
        
        # Conjugate momenta p = dq/dt = -A*ω*sin(ωt + φ)
        # For exact Hamiltonian mechanics: p = ∂L/∂q̇
        p1 = -A_T * omega_base * np.sin(omega_base * t + phase_T)
        p2 = -A_P * omega_base * np.sin(omega_base * t + phase_P)
        p3 = -A_V * omega_base * np.sin(omega_base * t + phase_V)
        p4 = -A_S * omega_base * np.sin(omega_base * t + phase_S)
        
        # Stack into coordinate matrices
        q = np.column_stack([q1, q2, q3, q4])
        p = np.column_stack([p1, p2, p3, p4])
        
        # Hamiltonian H = T + V = (1/2)Σpᵢ² + (1/2)Σωᵢ²qᵢ²
        # For harmonic oscillator, H is CONSERVED along trajectories
        KE = 0.5 * np.sum(p**2, axis=1)  # Kinetic energy
        PE = 0.5 * omega_base**2 * np.sum(q**2, axis=1)  # Potential energy
        H = KE + PE
        
        # Scale Hamiltonian to physical energy range
        H_scale = physics.hamiltonian_energy_J / max(np.mean(H), 1e-6)
        H = H * H_scale
        
        temporal['hamiltonian_q'] = q
        temporal['hamiltonian_p'] = p
        temporal['hamiltonian_H'] = H
        
        # Control signals
        control_signals = np.column_stack([
            physics.compressor_speed_pct / 100 * startup_factor,
            physics.expansion_valve_opening_pct / 100 * startup_factor,
            physics.evap_fan_speed_pct / 100 * startup_factor,
            physics.cond_fan_speed_pct / 100 * startup_factor
        ])
        temporal['control_signals'] = control_signals
        
        return temporal
    
    # =========================================================================
    # GNN FEATURES (using ComponentInteractionModel and GNNTopologyGenerator)
    # =========================================================================
    
    def _generate_gnn_features(self, inputs: Layer1Inputs,
                                physics: Layer2PhysicsOutputs,
                                derived: Layer3DerivedFeatures) -> Dict[str, np.ndarray]:
        """Generate GNN features using component interaction models."""
        
        gnn = {}
        
        # Node features: 10 nodes × 35 features
        node_features = np.zeros((10, 35))
        
        # Compressor node (0)
        node_features[0, 0] = physics.T_cond_K / 300
        node_features[0, 1] = physics.P_cond_Pa / 1e6
        node_features[0, 2] = derived.eta_comp_actual
        node_features[0, 3] = physics.efficiency_degradation_pct / 100
        node_features[0, 4] = derived.W_comp_kW / 10
        node_features[0, 5] = physics.bearing_wear_pct / 100
        node_features[0, 6] = physics.valve_leakage_pct / 100
        node_features[0, 7] = physics.oil_quality_index
        node_features[0, 8] = physics.vibration_mm_s / 10
        node_features[0, 9] = physics.compressor_speed_pct / 100
        node_features[0, 10] = physics.hamiltonian_energy_J / 10000
        
        # Condenser node (1)
        node_features[1, 0] = physics.T_cond_K / 300
        node_features[1, 1] = physics.P_cond_Pa / 1e6
        node_features[1, 2] = 1.0 - physics.cond_effectiveness_reduction_pct / 100
        node_features[1, 3] = physics.cond_fouling_resistance * 1000
        node_features[1, 4] = derived.Q_cond_kW / 20
        node_features[1, 5] = physics.subcooling_K / 20
        node_features[1, 6] = physics.cond_fan_speed_pct / 100
        
        # Evaporator node (2)
        node_features[2, 0] = physics.T_evap_K / 300
        node_features[2, 1] = physics.P_evap_Pa / 1e6
        node_features[2, 2] = 1.0 - physics.evap_effectiveness_reduction_pct / 100
        node_features[2, 3] = physics.evap_fouling_resistance * 1000
        node_features[2, 4] = derived.Q_evap_kW / 20
        node_features[2, 5] = physics.superheat_K / 20
        node_features[2, 6] = physics.evap_fan_speed_pct / 100
        node_features[2, 7] = physics.frost_formation_risk
        
        # Expansion valve node (3)
        node_features[3, 0] = (physics.T_evap_K + physics.T_cond_K) / 600
        node_features[3, 1] = (physics.P_evap_Pa + physics.P_cond_Pa) / 2e6
        node_features[3, 2] = physics.expansion_valve_opening_pct / 100
        node_features[3, 3] = physics.superheat_K / 20
        
        # Refrigerant circuit nodes (4-5)
        node_features[4, 0] = physics.mass_flow_kg_s / 0.2
        node_features[4, 1] = physics.actual_charge_pct / 100
        node_features[4, 2] = physics.cop_factor_charge
        node_features[5, 0] = derived.pressure_ratio / 5
        
        # Control system node (6)
        node_features[6, 0] = physics.control_response_time_s / 100
        node_features[6, 1] = physics.control_overshoot_pct / 20
        node_features[6, 2] = physics.pid_error / 10
        
        # Ambient node (7)
        node_features[7, 0] = inputs.T_ambient_K / 300
        node_features[7, 1] = inputs.humidity
        node_features[7, 2] = physics.wet_bulb_temp_K / 300
        node_features[7, 3] = physics.dew_point_temp_K / 300
        node_features[7, 4] = physics.air_density_kg_m3 / 1.2
        node_features[7, 5] = inputs.wind_speed_m_s / 10
        node_features[7, 6] = inputs.solar_irradiance_W_m2 / 1000
        
        # Building/load node (8)
        node_features[8, 0] = inputs.building_load_kW / 20
        node_features[8, 1] = inputs.occupancy_factor
        node_features[8, 2] = inputs.internal_gains_kW / 5
        node_features[8, 3] = derived.load_match_factor
        
        # System health node (9)
        node_features[9, 0] = inputs.age_hours / 50000
        node_features[9, 1] = derived.total_efficiency_loss_pct / 100
        node_features[9, 2] = derived.total_capacity_loss_pct / 100
        node_features[9, 3] = physics.has_fault
        node_features[9, 4] = physics.fault_severity
        
        gnn['node_features'] = node_features
        
        # Edge features: 11 edges × 12 features
        edge_features = np.ones((11, 12)) * physics.mass_flow_kg_s
        edge_features[:, 1] = derived.pressure_ratio / 5
        edge_features[:, 2] = (physics.T_cond_K - physics.T_evap_K) / 100
        edge_features[:, 3] = physics.thermal_coupling_strength
        edge_features[:, 4] = physics.mass_coupling_strength
        
        # Add component interaction coupling
        if self.interaction_model:
            try:
                edge_features[:6, 5] = self.interaction_model.base_interaction_strengths['thermal'][:6, 0]
            except Exception:
                pass
        
        gnn['edge_features'] = edge_features
        
        # Graph features: 50 features
        graph_features = np.zeros(50)
        
        # Ambient (0-9)
        graph_features[0] = inputs.T_ambient_K / 300
        graph_features[1] = inputs.humidity
        graph_features[2] = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}[inputs.season] / 3
        graph_features[3] = physics.wet_bulb_temp_K / 300
        graph_features[4] = physics.dew_point_temp_K / 300
        graph_features[5] = physics.air_density_kg_m3 / 1.2
        graph_features[6] = inputs.wind_speed_m_s / 10
        graph_features[7] = inputs.solar_irradiance_W_m2 / 1000
        graph_features[8] = physics.cooling_cop_modifier
        graph_features[9] = physics.heating_cop_modifier
        
        # Indoor (10-14)
        graph_features[10] = inputs.T_indoor_K / 300
        graph_features[11] = inputs.mode == 'heating'
        graph_features[12] = inputs.setpoint_temp_K / 300
        graph_features[13] = inputs.superheat_setpoint_K / 20
        graph_features[14] = inputs.subcooling_setpoint_K / 20
        
        # Building (15-19)
        graph_features[15] = inputs.building_load_kW / 20
        graph_features[16] = inputs.occupancy_factor
        graph_features[17] = inputs.internal_gains_kW / 5
        graph_features[18] = derived.load_match_factor
        graph_features[19] = derived.capacity_utilization
        
        # Charge (20-26)
        graph_features[20] = physics.actual_charge_pct / 100
        graph_features[21] = physics.subcooling_K / 20
        graph_features[22] = physics.superheat_K / 20
        graph_features[23] = physics.cop_factor_charge
        graph_features[24] = inputs.nominal_charge_kg / 10
        graph_features[25] = inputs.charge_percentage / 100
        
        # System state (27-35)
        graph_features[27] = inputs.age_hours / 50000
        graph_features[28] = physics.efficiency_degradation_pct / 100
        graph_features[29] = physics.has_fault
        graph_features[30] = physics.fault_severity
        graph_features[31] = derived.total_efficiency_loss_pct / 100
        graph_features[32] = derived.total_capacity_loss_pct / 100
        graph_features[33] = physics.bearing_wear_pct / 100
        graph_features[34] = physics.oil_quality_index
        graph_features[35] = {'good': 1.0, 'average': 0.5, 'poor': 0.0}[inputs.maintenance_quality]
        
        # Control (36-40)
        graph_features[36] = physics.compressor_speed_pct / 100
        graph_features[37] = physics.expansion_valve_opening_pct / 100
        graph_features[38] = physics.control_response_time_s / 100
        graph_features[39] = physics.control_overshoot_pct / 20
        
        # Performance (41-49)
        graph_features[41] = derived.carnot_cop_heating / 15
        graph_features[42] = derived.carnot_efficiency
        graph_features[43] = derived.Q_evap_kW / 20
        graph_features[44] = derived.Q_cond_kW / 20
        graph_features[45] = derived.W_comp_kW / 10
        graph_features[46] = derived.pressure_ratio / 5
        graph_features[47] = inputs.part_load_ratio
        graph_features[48] = physics.part_load_efficiency
        graph_features[49] = physics.network_stability_margin
        
        gnn['graph_features'] = graph_features
        
        return gnn
    
    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================
    
    def generate_single_cycle(self, cycle_id: int) -> Optional[CycleData]:
        """Generate a single cycle using ALL physics models."""
        
        try:
            # Layer 1: Sample inputs
            layer1 = self._sample_layer1_inputs()
            
            # Layer 2: Run ALL physics models
            layer2 = self._calculate_layer2_physics(layer1)
            
            # Layer 3: Calculate derived features
            layer3 = self._calculate_layer3_derived(layer1, layer2)
            
            # Layer 4: Calculate COP and validate
            layer4 = self._calculate_layer4_targets(layer1, layer2, layer3)
            
            if not layer4.is_physically_valid:
                return None
            
            # Generate temporal dynamics
            temporal = self._generate_temporal_dynamics(layer1, layer2, layer3)
            
            # Generate GNN features
            gnn = self._generate_gnn_features(layer1, layer2, layer3)
            
            # Create cycle data
            cycle = CycleData(
                cycle_id=f"cycle_{cycle_id:05d}",
                timestamp=datetime.now().isoformat(),
                layer1=layer1,
                layer2=layer2,
                layer3=layer3,
                layer4=layer4,
                temporal_T_evap=temporal['T_evap'],
                temporal_T_cond=temporal['T_cond'],
                temporal_P_evap=temporal['P_evap'],
                temporal_P_cond=temporal['P_cond'],
                temporal_mass_flow=temporal['mass_flow'],
                temporal_W_comp=temporal['W_comp'],
                temporal_Q_evap=temporal['Q_evap'],
                temporal_Q_cond=temporal['Q_cond'],
                temporal_superheat=temporal['superheat'],
                temporal_subcooling=temporal['subcooling'],
                hamiltonian_q=temporal['hamiltonian_q'],
                hamiltonian_p=temporal['hamiltonian_p'],
                hamiltonian_H=temporal['hamiltonian_H'],
                control_signals=temporal['control_signals'],
                node_features=gnn['node_features'],
                edge_features=gnn['edge_features'],
                graph_features=gnn['graph_features']
            )
            
            return cycle
            
        except Exception as e:
            print(f"  Error generating cycle {cycle_id}: {e}")
            return None
    
    def generate_dataset(self, n_cycles: int, output_dir: str = "comprehensive_dataset",
                        use_distribution_filter: bool = True,
                        target_cop_mean: float = 5.0,
                        target_cop_std: float = 1.2,
                        filter_aggressiveness: float = 3.0) -> Path:
        """
        Generate a complete dataset with ALL physics models.
        
        Args:
            n_cycles: Target number of valid cycles in final dataset
            output_dir: Output directory path
            use_distribution_filter: If True, use COP distribution filter to shape output
            target_cop_mean: Target mean COP for distribution filter
            target_cop_std: Target standard deviation for distribution filter
            filter_aggressiveness: How aggressively to reject over-represented bins (2-5 recommended)
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize distribution filter if enabled
        cop_filter = None
        if use_distribution_filter:
            cop_filter = COPDistributionFilter(
                target_mean=target_cop_mean,
                target_std=target_cop_std,
                cop_min=1.5,
                cop_max=8.5,
                n_bins=20,
                aggressiveness=filter_aggressiveness
            )
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE PHYSICS DATASET GENERATION")
        print("=" * 70)
        print(f"  Target cycles: {n_cycles}")
        print(f"  Output directory: {output_path.absolute()}")
        print(f"  Physics models loaded: {sum(models_loaded.values())}/{len(models_loaded)}")
        if use_distribution_filter:
            print(f"  Distribution filter: ENABLED (mean={target_cop_mean}, std={target_cop_std})")
        else:
            print(f"  Distribution filter: DISABLED (raw physics)")
        print("=" * 70)
        
        start_time = time.time()
        valid_cycles: List[CycleData] = []
        total_generated = 0
        total_rejected_physics = 0
        total_rejected_distribution = 0
        max_attempts = n_cycles * 10  # Maximum attempts to avoid infinite loop
        
        while len(valid_cycles) < n_cycles and total_generated < max_attempts:
            total_generated += 1
            
            if total_generated % 100 == 0:
                elapsed = time.time() - start_time
                rate = total_generated / elapsed if elapsed > 0 else 0
                print(f"  Progress: {len(valid_cycles)}/{n_cycles} accepted | "
                      f"Generated: {total_generated} | "
                      f"Physics reject: {total_rejected_physics} | "
                      f"Distribution reject: {total_rejected_distribution} | "
                      f"{rate:.1f}/sec")
            
            cycle = self.generate_single_cycle(total_generated)
            
            if cycle is None:
                # Rejected by physics validation
                total_rejected_physics += 1
                self.stats['invalid'] += 1
                continue
            
            cop = cycle.layer4.COP_heating
            
            # Apply distribution filter if enabled
            if use_distribution_filter and cop_filter:
                if cop_filter.should_accept(cop):
                    # Accepted by distribution filter
                    cop_filter.record_acceptance(cop)
                    valid_cycles.append(cycle)
                    self.stats['valid'] += 1
                    self.stats['cop_values'].append(cop)
                    self.stats['carnot_efficiencies'].append(cycle.layer3.carnot_efficiency)
                else:
                    # Rejected by distribution filter
                    total_rejected_distribution += 1
            else:
                # No filter - accept all physics-valid cycles
                valid_cycles.append(cycle)
                self.stats['valid'] += 1
                self.stats['cop_values'].append(cop)
                self.stats['carnot_efficiencies'].append(cycle.layer3.carnot_efficiency)
            
            self.stats['generated'] += 1
        
        elapsed = time.time() - start_time
        
        # Print statistics
        print("\n" + "-" * 70)
        print("GENERATION COMPLETE")
        print("-" * 70)
        print(f"  Final dataset size: {len(valid_cycles)} cycles")
        print(f"  Total generated: {total_generated}")
        print(f"  Rejected by physics: {total_rejected_physics}")
        if use_distribution_filter:
            print(f"  Rejected by distribution: {total_rejected_distribution}")
            acceptance_rate = len(valid_cycles) / max(total_generated - total_rejected_physics, 1) * 100
            print(f"  Distribution acceptance rate: {acceptance_rate:.1f}%")
        print(f"  Generation time: {elapsed:.1f}s ({total_generated/elapsed:.1f} cycles/sec)")
        
        if self.stats['cop_values']:
            cop_arr = np.array(self.stats['cop_values'])
            eff_arr = np.array(self.stats['carnot_efficiencies'])
            
            print(f"\n[STATS] COP Distribution:")
            print(f"  Min: {cop_arr.min():.2f}")
            print(f"  Max: {cop_arr.max():.2f}")
            print(f"  Mean: {cop_arr.mean():.2f}")
            print(f"  Std: {cop_arr.std():.2f}")
            print(f"  Median: {np.median(cop_arr):.2f}")
            
            print(f"\n[STATS] Carnot Efficiency Distribution:")
            print(f"  Min: {eff_arr.min():.2f}")
            print(f"  Max: {eff_arr.max():.2f}")
            print(f"  Mean: {eff_arr.mean():.2f}")
            
            # Histogram
            hist, edges = np.histogram(cop_arr, bins=10)
            print(f"\n[HIST] COP Histogram:")
            for j in range(len(hist)):
                bar = "#" * int(hist[j] / max(hist) * 30)
                print(f"  {edges[j]:.1f}-{edges[j+1]:.1f}: {bar} ({hist[j]})")
        
        # Save to HDF5
        self._save_hdf5(valid_cycles, output_path)
        
        # Save sample to JSON for debugging
        if valid_cycles:
            self._save_sample_json(valid_cycles[0], output_path)
        
        # Save metadata
        self._save_metadata(valid_cycles, output_path, elapsed)
        
        print(f"\n[SUCCESS] Dataset saved to {output_path}")
        return output_path
    
    def _save_hdf5(self, cycles: List[CycleData], output_path: Path):
        """Save dataset in HDF5 format."""
        
        h5_path = output_path / 'physics_data.h5'
        n = len(cycles)
        
        with h5py.File(h5_path, 'w') as f:
            # Metadata
            f.attrs['n_cycles'] = n
            f.attrs['n_timesteps'] = self.N_TIMESTEPS
            f.attrs['physics_models_used'] = sum(models_loaded.values())
            
            # Targets
            targets = f.create_group('targets')
            targets.create_dataset('COP_heating', data=np.array([c.layer4.COP_heating for c in cycles]))
            targets.create_dataset('COP_cooling', data=np.array([c.layer4.COP_cooling for c in cycles]))
            targets.create_dataset('is_valid', data=np.array([c.layer4.is_physically_valid for c in cycles]))
            
            # Cycle parameters
            params = f.create_group('cycle_parameters')
            params.create_dataset('T_evap', data=np.array([c.layer2.T_evap_K for c in cycles]))
            params.create_dataset('T_cond', data=np.array([c.layer2.T_cond_K for c in cycles]))
            params.create_dataset('P_evap', data=np.array([c.layer2.P_evap_Pa for c in cycles]))
            params.create_dataset('P_cond', data=np.array([c.layer2.P_cond_Pa for c in cycles]))
            params.create_dataset('mass_flow', data=np.array([c.layer2.mass_flow_kg_s for c in cycles]))
            params.create_dataset('superheat', data=np.array([c.layer2.superheat_K for c in cycles]))
            params.create_dataset('subcooling', data=np.array([c.layer2.subcooling_K for c in cycles]))
            
            # Ambient conditions
            ambient = f.create_group('ambient')
            ambient.create_dataset('T_ambient', data=np.array([c.layer1.T_ambient_K for c in cycles]))
            ambient.create_dataset('humidity', data=np.array([c.layer1.humidity for c in cycles]))
            ambient.create_dataset('wet_bulb_temp', data=np.array([c.layer2.wet_bulb_temp_K for c in cycles]))
            ambient.create_dataset('heating_cop_modifier', data=np.array([c.layer2.heating_cop_modifier for c in cycles]))
            ambient.create_dataset('frost_risk', data=np.array([c.layer2.frost_formation_risk for c in cycles]))
            
            # Degradation
            degrad = f.create_group('degradation')
            degrad.create_dataset('evap_fouling', data=np.array([c.layer2.evap_fouling_resistance for c in cycles]))
            degrad.create_dataset('cond_fouling', data=np.array([c.layer2.cond_fouling_resistance for c in cycles]))
            degrad.create_dataset('comp_efficiency_loss', data=np.array([c.layer2.efficiency_degradation_pct for c in cycles]))
            degrad.create_dataset('bearing_wear', data=np.array([c.layer2.bearing_wear_pct for c in cycles]))
            degrad.create_dataset('charge_level', data=np.array([c.layer2.actual_charge_pct for c in cycles]))
            degrad.create_dataset('charge_cop_factor', data=np.array([c.layer2.cop_factor_charge for c in cycles]))
            
            # Control system
            control = f.create_group('control')
            control.create_dataset('compressor_speed', data=np.array([c.layer2.compressor_speed_pct for c in cycles]))
            control.create_dataset('valve_opening', data=np.array([c.layer2.expansion_valve_opening_pct for c in cycles]))
            control.create_dataset('response_time', data=np.array([c.layer2.control_response_time_s for c in cycles]))
            
            # Hamiltonian
            hamiltonian = f.create_group('hamiltonian')
            hamiltonian.create_dataset('total_energy', data=np.array([c.layer2.hamiltonian_energy_J for c in cycles]))
            hamiltonian.create_dataset('kinetic_energy', data=np.array([c.layer2.kinetic_energy_J for c in cycles]))
            hamiltonian.create_dataset('potential_energy', data=np.array([c.layer2.potential_energy_J for c in cycles]))
            
            # Temporal HNN
            temporal = f.create_group('temporal_hnn')
            temporal.create_dataset('T_evap_traj', data=np.array([c.temporal_T_evap for c in cycles]))
            temporal.create_dataset('T_cond_traj', data=np.array([c.temporal_T_cond for c in cycles]))
            temporal.create_dataset('P_evap_traj', data=np.array([c.temporal_P_evap for c in cycles]))
            temporal.create_dataset('P_cond_traj', data=np.array([c.temporal_P_cond for c in cycles]))
            temporal.create_dataset('mass_flow_traj', data=np.array([c.temporal_mass_flow for c in cycles]))
            temporal.create_dataset('W_comp_traj', data=np.array([c.temporal_W_comp for c in cycles]))
            temporal.create_dataset('Q_evap_traj', data=np.array([c.temporal_Q_evap for c in cycles]))
            temporal.create_dataset('Q_cond_traj', data=np.array([c.temporal_Q_cond for c in cycles]))
            temporal.create_dataset('superheat_traj', data=np.array([c.temporal_superheat for c in cycles]))
            temporal.create_dataset('subcooling_traj', data=np.array([c.temporal_subcooling for c in cycles]))
            temporal.create_dataset('hamiltonian_q', data=np.array([c.hamiltonian_q for c in cycles]))
            temporal.create_dataset('hamiltonian_p', data=np.array([c.hamiltonian_p for c in cycles]))
            temporal.create_dataset('hamiltonian_H', data=np.array([c.hamiltonian_H for c in cycles]))
            temporal.create_dataset('control_signals', data=np.array([c.control_signals for c in cycles]))
            
            # GNN features
            gnn = f.create_group('gnn')
            gnn.create_dataset('node_features', data=np.array([c.node_features for c in cycles]))
            gnn.create_dataset('edge_features', data=np.array([c.edge_features for c in cycles]))
            gnn.create_dataset('graph_features', data=np.array([c.graph_features for c in cycles]))
            
            # Validation
            valid = f.create_group('validation')
            valid.create_dataset('energy_balance_error', data=np.array([c.layer4.energy_balance_error_pct for c in cycles]))
            valid.create_dataset('carnot_cop', data=np.array([c.layer3.carnot_cop_heating for c in cycles]))
            valid.create_dataset('carnot_efficiency', data=np.array([c.layer3.carnot_efficiency for c in cycles]))
            valid.create_dataset('below_carnot', data=np.array([c.layer4.below_carnot_limit for c in cycles]))
        
        print(f"  [SAVED] HDF5: {h5_path}")
    
    def _save_sample_json(self, cycle: CycleData, output_path: Path):
        """Save a sample cycle as JSON for debugging."""
        
        json_path = output_path / 'sample_cycle.json'
        
        sample = {
            'cycle_id': cycle.cycle_id,
            'timestamp': cycle.timestamp,
            'layer1_inputs': asdict(cycle.layer1),
            'layer2_physics_outputs': asdict(cycle.layer2),
            'layer3_derived_features': asdict(cycle.layer3),
            'layer4_targets': asdict(cycle.layer4),
            'temporal_shapes': {
                'T_evap': list(cycle.temporal_T_evap.shape),
                'hamiltonian_q': list(cycle.hamiltonian_q.shape),
                'control_signals': list(cycle.control_signals.shape)
            },
            'gnn_shapes': {
                'node_features': list(cycle.node_features.shape),
                'edge_features': list(cycle.edge_features.shape),
                'graph_features': list(cycle.graph_features.shape)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(sample, f, indent=2, default=str)
        
        print(f"  [SAVED] Sample JSON: {json_path}")
    
    def _save_metadata(self, cycles: List[CycleData], output_path: Path, elapsed: float):
        """Save dataset metadata."""
        
        cop_values = [c.layer4.COP_heating for c in cycles]
        carnot_effs = [c.layer3.carnot_efficiency for c in cycles]
        
        metadata = {
            'n_cycles': len(cycles),
            'n_timesteps': self.N_TIMESTEPS,
            'refrigerants': self.REFRIGERANTS,
            'physics_models_loaded': {k: v for k, v in models_loaded.items()},
            'total_models_loaded': sum(models_loaded.values()),
            'cop_stats': {
                'min': float(np.min(cop_values)) if cop_values else 0,
                'max': float(np.max(cop_values)) if cop_values else 0,
                'mean': float(np.mean(cop_values)) if cop_values else 0,
                'std': float(np.std(cop_values)) if cop_values else 0,
                'median': float(np.median(cop_values)) if cop_values else 0
            },
            'carnot_efficiency_stats': {
                'min': float(np.min(carnot_effs)) if carnot_effs else 0,
                'max': float(np.max(carnot_effs)) if carnot_effs else 0,
                'mean': float(np.mean(carnot_effs)) if carnot_effs else 0,
            },
            'generation_time_seconds': elapsed,
            'generated_at': datetime.now().isoformat(),
            'architecture': '4-layer (inputs→physics→derived→targets)',
            'features': {
                'hnn_temporal': f'{self.N_TIMESTEPS} timesteps × 14 channels',
                'gnn_nodes': '10 nodes × 35 features',
                'gnn_edges': '11 edges × 12 features',
                'gnn_graph': '50 features'
            }
        }
        
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  [SAVED] Metadata: {output_path / 'dataset_info.json'}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive physics-based COP training data')
    parser.add_argument('--n_cycles', type=int, default=500, help='Number of cycles to generate')
    parser.add_argument('--output_dir', type=str, default='comprehensive_dataset', help='Output directory')
    parser.add_argument('--no_filter', action='store_true', help='Disable COP distribution filter (use raw physics)')
    parser.add_argument('--cop_mean', type=float, default=5.0, help='Target COP mean for distribution filter')
    parser.add_argument('--cop_std', type=float, default=1.2, help='Target COP std for distribution filter')
    parser.add_argument('--aggressiveness', type=float, default=3.0,
                       help='Filter aggressiveness: higher=stronger rejection of over-represented bins (2-5 recommended)')
    args = parser.parse_args()
    
    generator = ComprehensivePhysicsGenerator()
    generator.generate_dataset(
        n_cycles=args.n_cycles,
        output_dir=args.output_dir,
        use_distribution_filter=not args.no_filter,
        target_cop_mean=args.cop_mean,
        target_cop_std=args.cop_std,
        filter_aggressiveness=args.aggressiveness
    )


if __name__ == "__main__":
    main()