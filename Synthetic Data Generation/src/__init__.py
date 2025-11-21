"""
Airthium Heat Pump Synthetic Data Generation Package
Comprehensive physics-informed synthetic data generation for heat pump systems
"""

from .thermodynamic_properties import (
    FluidProperties, ThermodynamicState, FluidDatabase,
    RefrigerantProperties, WaterSteamProperties, ThermalOilProperties,
    fluid_db, get_fluid_properties
)

from .cycle_generators import (
    CycleGenerator, CycleType, CycleParameters, CycleResults,
    CarnotCycle, RankineCycle, TranscriticalCycle
)

from .component_models import (
    CompressorModel, HeatExchangerModel, ExpansionDeviceModel,
    CompressorType, HeatExchangerType, ExpansionType,
    CompressorPerformance, HeatExchangerPerformance, ExpansionPerformance,
    component_integration
)

from .data_generation_pipeline import (
    SyntheticDataGenerator, DatasetConfiguration, PhysicsValidator
)

__version__ = "1.0.0"
__author__ = "Airthium Heat Pump Team"
__description__ = "Physics-informed synthetic data generation for heat pump systems"

# Package-level configuration
DEFAULT_CONFIG = DatasetConfiguration(
    total_cycles=10000,  # Smaller default for testing
    carnot_cycles=4000,
    ideal_rankine_cycles=2000,
    non_ideal_rankine_cycles=3000,
    transcritical_cycles=500,
    orc_cycles=500,
    output_dir="test_data",
    n_workers=2,
    batch_size=100
)

def quick_generate_sample(n_cycles: int = 100, output_dir: str = "sample_data") -> dict:
    """
    Quick function to generate a small sample dataset for testing
    
    Args:
        n_cycles: Number of cycles to generate
        output_dir: Output directory for data
    
    Returns:
        Dictionary containing generated dataset
    """
    config = DatasetConfiguration(
        total_cycles=n_cycles,
        carnot_cycles=int(n_cycles * 0.4),
        ideal_rankine_cycles=int(n_cycles * 0.3),
        non_ideal_rankine_cycles=int(n_cycles * 0.3),
        transcritical_cycles=0,
        orc_cycles=0,
        output_dir=output_dir,
        n_workers=1,
        batch_size=50
    )
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_complete_dataset()

# Available working fluids
AVAILABLE_FLUIDS = {
    'refrigerants': ['R245fa', 'R1234ze(E)', 'R1233zd(E)', 'R134a', 'R410A', 'R600a', 'R744', 'R717', 'R290'],
    'water_steam': ['Water'],
    'thermal_oils': ['Therminol_VP1', 'Dowtherm_A', 'Mineral_Oil'],
    'air': ['Air'],
    'orc_fluids': ['Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol']
}

# Cycle types available
AVAILABLE_CYCLES = [
    CycleType.CARNOT,
    CycleType.IDEAL_RANKINE,
    CycleType.NON_IDEAL_RANKINE,
    CycleType.TRANSCRITICAL,
    CycleType.ORC
]

def list_available_fluids():
    """List all available working fluids by category"""
    print("Available Working Fluids:")
    print("=" * 40)
    for category, fluids in AVAILABLE_FLUIDS.items():
        print(f"{category.replace('_', ' ').title()}: {', '.join(fluids)}")
    print()

def list_available_cycles():
    """List all available cycle types"""
    print("Available Cycle Types:")
    print("=" * 40)
    for cycle in AVAILABLE_CYCLES:
        print(f"- {cycle.value.replace('_', ' ').title()}")
    print()

# Export main classes and functions
__all__ = [
    # Core classes
    'FluidProperties', 'ThermodynamicState', 'FluidDatabase',
    'CycleGenerator', 'CycleType', 'CycleParameters', 'CycleResults',
    'SyntheticDataGenerator', 'DatasetConfiguration', 'PhysicsValidator',
    
    # Component models
    'CompressorModel', 'HeatExchangerModel', 'ExpansionDeviceModel',
    'CompressorType', 'HeatExchangerType', 'ExpansionType',
    
    # Utility functions
    'get_fluid_properties', 'quick_generate_sample',
    'list_available_fluids', 'list_available_cycles',
    
    # Constants
    'DEFAULT_CONFIG', 'AVAILABLE_FLUIDS', 'AVAILABLE_CYCLES',
    
    # Global instances
    'fluid_db', 'component_integration'
]