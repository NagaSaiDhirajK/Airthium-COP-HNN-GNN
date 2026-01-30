"""
Synthetic Data Generation Pipeline
Main orchestrator for generating comprehensive thermodynamic cycle datasets
"""

import numpy as np
import pandas as pd
import json
import h5py
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from thermodynamic_properties import fluid_db, get_fluid_properties
from cycle_generators import (
    CycleGenerator, CycleType, CycleParameters, CycleResults
)
from component_models import (
    component_integration, CompressorType, HeatExchangerType, ExpansionType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfiguration:
    """Configuration for dataset generation"""
    # Dataset size specifications
    total_cycles: int = 145000
    carnot_cycles: int = 50000
    ideal_rankine_cycles: int = 30000
    non_ideal_rankine_cycles: int = 40000
    transcritical_cycles: int = 10000
    cascade_cycles: int = 5000
    orc_cycles: int = 10000
    
    # Component performance points
    compressor_points: int = 20000
    heat_exchanger_points: int = 30000
    expansion_points: int = 10000
    
    # Working fluids
    primary_refrigerants: List[str] = None
    secondary_fluids: List[str] = None
    
    # Output configuration
    output_dir: str = "generated_data"
    save_json: bool = True
    save_hdf5: bool = True
    save_csv: bool = True
    
    # Validation thresholds
    max_energy_balance_error: float = 0.1  # %
    min_carnot_compliance: float = 0.99  # 99% of cycles must comply
    
    # Parallel processing
    n_workers: int = None
    batch_size: int = 1000
    
    def __post_init__(self):
        if self.primary_refrigerants is None:
            self.primary_refrigerants = [
                'R245fa', 'R1234ze(E)', 'R1233zd(E)', 'R134a', 'R410A',
                'R600a', 'R744', 'R717', 'R290'
            ]
        
        if self.secondary_fluids is None:
            self.secondary_fluids = [
                'Water', 'Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol',
                'Therminol_VP1', 'Air'
            ]
        
        if self.n_workers is None:
            self.n_workers = min(mp.cpu_count(), 8)

class PhysicsValidator:
    """Validates physics constraints in generated data"""
    
    def __init__(self, config: DatasetConfiguration):
        self.config = config
        self.validation_stats = {
            'total_cycles': 0,
            'energy_balance_violations': 0,
            'carnot_limit_violations': 0,
            'entropy_violations': 0,
            'valid_cycles': 0
        }
    
    def validate_cycle(self, results: CycleResults) -> bool:
        """Validate a single cycle against physics constraints"""
        self.validation_stats['total_cycles'] += 1
        
        is_valid = True
        
        # Energy balance check
        if results.energy_balance_error > self.config.max_energy_balance_error:
            self.validation_stats['energy_balance_violations'] += 1
            is_valid = False
        
        # Carnot limit check
        if not results.carnot_limit_check:
            self.validation_stats['carnot_limit_violations'] += 1
            is_valid = False
        
        # Entropy generation check (must be non-negative)
        if results.entropy_generation < -1e-6:  # Small tolerance for numerical errors
            self.validation_stats['entropy_violations'] += 1
            is_valid = False
        
        if is_valid:
            self.validation_stats['valid_cycles'] += 1
        
        return is_valid
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        total = self.validation_stats['total_cycles']
        if total == 0:
            return {"error": "No cycles validated"}
        
        return {
            'total_cycles_generated': total,
            'valid_cycles': self.validation_stats['valid_cycles'],
            'validation_rate': self.validation_stats['valid_cycles'] / total * 100,
            'energy_balance_violation_rate': self.validation_stats['energy_balance_violations'] / total * 100,
            'carnot_limit_violation_rate': self.validation_stats['carnot_limit_violations'] / total * 100,
            'entropy_violation_rate': self.validation_stats['entropy_violations'] / total * 100,
            'physics_compliance': {
                'energy_conservation': (total - self.validation_stats['energy_balance_violations']) / total * 100,
                'carnot_limit_compliance': (total - self.validation_stats['carnot_limit_violations']) / total * 100,
                'entropy_compliance': (total - self.validation_stats['entropy_violations']) / total * 100
            }
        }

def generate_cycle_batch(args: Tuple[CycleType, List[str], int, int]) -> List[Dict]:
    """Generate a batch of cycles (for parallel processing)"""
    cycle_type, fluids, batch_size, seed = args
    
    np.random.seed(seed)
    generator = CycleGenerator()
    results = []
    
    for i in range(batch_size):
        try:
            # Select random fluid
            fluid = np.random.choice(fluids)
            
            # Generate random parameters
            params = generator.generate_random_parameters(cycle_type, fluid)
            
            # Add some realistic variations
            params = add_realistic_variations(params, cycle_type)
            
            # Generate cycle
            cycle_results = generator.generate_cycle(cycle_type, fluid, params)
            
            # Convert to dictionary for serialization
            result_dict = cycle_results_to_dict(cycle_results)
            results.append(result_dict)
            
        except Exception as e:
            logger.warning(f"Error generating cycle {i} of type {cycle_type}: {e}")
            continue
    
    return results

def add_realistic_variations(params: CycleParameters, cycle_type: CycleType) -> CycleParameters:
    """Add realistic variations and degradation effects"""
    
    # Seasonal temperature variations
    season_factor = np.random.choice([0.8, 1.0, 1.2, 1.0])  # Winter, spring, summer, fall
    params.T_ambient *= season_factor
    
    # Part-load operation (more common than full load)
    if np.random.random() < 0.7:  # 70% chance of part-load
        params.part_load_ratio = np.random.uniform(0.3, 0.9)
    
    # Component degradation (age effects)
    age_factor = np.random.uniform(0.95, 1.0)  # 0-5% degradation
    params.eta_comp_isentropic *= age_factor
    params.eta_exp_isentropic *= age_factor
    params.eta_mechanical *= age_factor
    
    # Fouling effects on heat exchangers
    fouling_factor = np.random.uniform(0.9, 1.0)
    params.dp_evap *= (2 - fouling_factor)  # Increased pressure drop
    params.dp_cond *= (2 - fouling_factor)
    
    # Installation quality variations
    installation_quality = np.random.uniform(0.95, 1.0)
    params.eta_motor *= installation_quality
    
    # Refrigerant charge variations
    charge_factor = np.random.uniform(0.9, 1.1)  # ±10% charge variation
    if charge_factor < 0.95 or charge_factor > 1.05:
        # Undercharge or overcharge effects
        params.eta_comp_isentropic *= (1 - 0.1 * abs(charge_factor - 1))
    
    return params

def cycle_results_to_dict(results: CycleResults) -> Dict:
    """Convert CycleResults to dictionary for serialization"""
    
    # Convert thermodynamic states
    states_dict = []
    for i, state in enumerate(results.states):
        state_dict = {
            'point_id': i + 1,
            'location': results.state_labels[i] if i < len(results.state_labels) else f"state_{i+1}",
            'P': float(state.P),
            'T': float(state.T),
            'h': float(state.h),
            's': float(state.s),
            'rho': float(state.rho),
            'x': float(state.x) if state.x is not None else None,
            'phase': state.phase,
            'cp': float(state.cp) if state.cp is not None else None,
            'cv': float(state.cv) if state.cv is not None else None,
            'mu': float(state.mu) if state.mu is not None else None,
            'k': float(state.k) if state.k is not None else None
        }
        states_dict.append(state_dict)
    
    # Main results dictionary
    result_dict = {
        'cycle_id': f"{results.cycle_type}_{results.working_fluid}_{int(time.time()*1000000) % 1000000}",
        'cycle_type': results.cycle_type,
        'working_fluid': results.working_fluid,
        
        # Operating conditions
        'operating_conditions': results.operating_conditions,
        
        # Thermodynamic states
        'states': states_dict,
        
        # Performance metrics
        'performance': {
            'Q_evap': float(results.Q_evap),
            'Q_cond': float(results.Q_cond),
            'W_comp': float(results.W_comp),
            'W_exp': float(results.W_exp),
            'W_net': float(results.W_net),
            'COP_heating': float(results.COP_heating),
            'COP_cooling': float(results.COP_cooling),
            'mass_flow': float(results.mass_flow),
            'capacity_heating': float(results.capacity_heating),
            'capacity_cooling': float(results.capacity_cooling)
        },
        
        # Efficiency metrics
        'efficiency': {
            'eta_carnot_heating': float(results.eta_carnot_heating),
            'eta_carnot_cooling': float(results.eta_carnot_cooling),
            'eta_second_law': float(results.eta_second_law)
        },
        
        # Validation metrics
        'validation': {
            'energy_balance_error': float(results.energy_balance_error),
            'entropy_generation': float(results.entropy_generation),
            'carnot_limit_check': bool(results.carnot_limit_check)
        },
        
        # Trajectory data for HNN training
        'trajectory': results.trajectory if results.trajectory else {}
    }
    
    return result_dict

class SyntheticDataGenerator:
    """Main class for generating synthetic thermodynamic data"""
    
    def __init__(self, config: DatasetConfiguration):
        self.config = config
        self.validator = PhysicsValidator(config)
        self.generated_data = []
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized data generator with {config.total_cycles} total cycles")
        logger.info(f"Output directory: {self.output_path}")
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate the complete synthetic dataset"""
        
        logger.info("Starting comprehensive dataset generation...")
        start_time = time.time()
        
        # Generate thermodynamic cycles
        cycle_data = self._generate_thermodynamic_cycles()
        
        # Generate component performance maps
        component_data = self._generate_component_performance_maps()
        
        # Combine all data
        complete_dataset = {
            'metadata': self._generate_metadata(),
            'thermodynamic_cycles': cycle_data,
            'component_performance': component_data,
            'validation_report': self.validator.get_validation_report()
        }
        
        # Save dataset
        self._save_dataset(complete_dataset)
        
        generation_time = time.time() - start_time
        logger.info(f"Dataset generation completed in {generation_time:.2f} seconds")
        
        return complete_dataset
    
    def _generate_thermodynamic_cycles(self) -> List[Dict]:
        """Generate all thermodynamic cycles"""
        
        logger.info("Generating thermodynamic cycles...")
        
        # Define cycle generation tasks
        tasks = [
            (CycleType.CARNOT, self.config.primary_refrigerants, self.config.carnot_cycles),
            (CycleType.IDEAL_RANKINE, self.config.primary_refrigerants + self.config.secondary_fluids, 
             self.config.ideal_rankine_cycles),
            (CycleType.NON_IDEAL_RANKINE, self.config.primary_refrigerants + self.config.secondary_fluids,
             self.config.non_ideal_rankine_cycles),
            (CycleType.TRANSCRITICAL, ['R744'], self.config.transcritical_cycles),
            (CycleType.ORC, ['Toluene', 'Cyclohexane', 'n-Pentane', 'Ethanol'], self.config.orc_cycles)
        ]
        
        all_cycles = []
        
        for cycle_type, fluids, n_cycles in tasks:
            logger.info(f"Generating {n_cycles} {cycle_type.value} cycles...")
            
            # Prepare batches for parallel processing
            n_batches = max(1, n_cycles // self.config.batch_size)
            batch_sizes = [n_cycles // n_batches] * n_batches
            batch_sizes[-1] += n_cycles % n_batches  # Add remainder to last batch
            
            batch_args = [
                (cycle_type, fluids, batch_size, np.random.randint(0, 1000000))
                for batch_size in batch_sizes
            ]
            
            # Generate cycles in parallel
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                future_to_batch = {
                    executor.submit(generate_cycle_batch, args): i 
                    for i, args in enumerate(batch_args)
                }
                
                with tqdm(total=len(batch_args), desc=f"{cycle_type.value} batches") as pbar:
                    for future in as_completed(future_to_batch):
                        try:
                            batch_results = future.result()
                            
                            # Validate and filter results
                            valid_results = []
                            for result in batch_results:
                                # Convert back to CycleResults for validation
                                cycle_results = dict_to_cycle_results(result)
                                if self.validator.validate_cycle(cycle_results):
                                    valid_results.append(result)
                            
                            all_cycles.extend(valid_results)
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            pbar.update(1)
        
        logger.info(f"Generated {len(all_cycles)} valid thermodynamic cycles")
        return all_cycles
    
    def _generate_component_performance_maps(self) -> Dict[str, Any]:
        """Generate component performance maps"""
        
        logger.info("Generating component performance maps...")
        
        component_data = {
            'compressor_maps': {},
            'heat_exchanger_performance': [],
            'expansion_device_performance': []
        }
        
        # Generate compressor maps
        compressor_names = ['scroll_comp', 'recip_comp']
        for comp_name in compressor_names:
            try:
                comp_map = component_integration.generate_component_maps(comp_name, "compressor")
                component_data['compressor_maps'][comp_name] = {
                    'pressure_ratios': comp_map['pressure_ratios'].tolist(),
                    'flow_ratios': comp_map['flow_ratios'].tolist(),
                    'efficiency_map': comp_map['efficiency_map'].tolist(),
                    'power_map': comp_map['power_map'].tolist()
                }
            except Exception as e:
                logger.warning(f"Error generating compressor map for {comp_name}: {e}")
        
        logger.info("Component performance maps generated")
        return component_data
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate dataset metadata"""
        
        return {
            'dataset_info': {
                'name': 'Airthium Heat Pump Synthetic Dataset',
                'version': '1.0',
                'description': 'Comprehensive synthetic thermodynamic cycle data for physics-informed ML',
                'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_cycles': self.config.total_cycles,
                'working_fluids': self.config.primary_refrigerants + self.config.secondary_fluids
            },
            'cycle_distribution': {
                'carnot': self.config.carnot_cycles,
                'ideal_rankine': self.config.ideal_rankine_cycles,
                'non_ideal_rankine': self.config.non_ideal_rankine_cycles,
                'transcritical': self.config.transcritical_cycles,
                'orc': self.config.orc_cycles
            },
            'physics_constraints': {
                'energy_conservation_tolerance': self.config.max_energy_balance_error,
                'carnot_limit_enforcement': True,
                'entropy_generation_check': True
            },
            'data_structure': {
                'state_points_per_cycle': '4-6',
                'trajectory_points_per_cycle': 100,
                'properties_per_state': ['P', 'T', 'h', 's', 'rho', 'x', 'phase'],
                'performance_metrics': ['COP', 'capacity', 'efficiency', 'work', 'heat_transfer']
            }
        }
    
    def _save_dataset(self, dataset: Dict[str, Any]):
        """Save dataset in multiple formats"""
        
        logger.info("Saving dataset...")
        
        # Save JSON (structured data)
        if self.config.save_json:
            json_path = self.output_path / "synthetic_dataset.json"
            with open(json_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Saved JSON dataset: {json_path}")
        
        # Save HDF5 (numerical data)
        if self.config.save_hdf5:
            self._save_hdf5_dataset(dataset)
        
        # Save CSV (tabular data)
        if self.config.save_csv:
            self._save_csv_dataset(dataset)
    
    def _save_hdf5_dataset(self, dataset: Dict[str, Any]):
        """Save dataset in HDF5 format for efficient numerical access"""
        
        hdf5_path = self.output_path / "synthetic_dataset.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                if isinstance(value, dict):
                    subgroup = meta_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str)):
                            subgroup.attrs[subkey] = subvalue
                        elif isinstance(subvalue, list):
                            subgroup.create_dataset(subkey, data=subvalue)
                else:
                    meta_group.attrs[key] = value
            
            # Thermodynamic cycles
            cycles_group = f.create_group('thermodynamic_cycles')
            
            if dataset['thermodynamic_cycles']:
                # Extract numerical arrays
                n_cycles = len(dataset['thermodynamic_cycles'])
                
                # Performance metrics arrays
                cop_heating = np.array([c['performance']['COP_heating'] for c in dataset['thermodynamic_cycles']])
                cop_cooling = np.array([c['performance']['COP_cooling'] for c in dataset['thermodynamic_cycles']])
                capacity_heating = np.array([c['performance']['capacity_heating'] for c in dataset['thermodynamic_cycles']])
                
                cycles_group.create_dataset('COP_heating', data=cop_heating)
                cycles_group.create_dataset('COP_cooling', data=cop_cooling)
                cycles_group.create_dataset('capacity_heating', data=capacity_heating)
                
                # State point arrays (assuming 4 states per cycle)
                max_states = 6
                P_states = np.zeros((n_cycles, max_states))
                T_states = np.zeros((n_cycles, max_states))
                h_states = np.zeros((n_cycles, max_states))
                s_states = np.zeros((n_cycles, max_states))
                
                for i, cycle in enumerate(dataset['thermodynamic_cycles']):
                    states = cycle['states']
                    for j, state in enumerate(states[:max_states]):
                        P_states[i, j] = state['P']
                        T_states[i, j] = state['T']
                        h_states[i, j] = state['h']
                        s_states[i, j] = state['s']
                
                states_group = cycles_group.create_group('states')
                states_group.create_dataset('pressure', data=P_states)
                states_group.create_dataset('temperature', data=T_states)
                states_group.create_dataset('enthalpy', data=h_states)
                states_group.create_dataset('entropy', data=s_states)
        
        logger.info(f"Saved HDF5 dataset: {hdf5_path}")
    
    def _save_csv_dataset(self, dataset: Dict[str, Any]):
        """Save dataset in CSV format for analysis"""
        
        if not dataset['thermodynamic_cycles']:
            return
        
        # Create DataFrame from cycle data
        rows = []
        for cycle in dataset['thermodynamic_cycles']:
            row = {
                'cycle_id': cycle['cycle_id'],
                'cycle_type': cycle['cycle_type'],
                'working_fluid': cycle['working_fluid'],
                'COP_heating': cycle['performance']['COP_heating'],
                'COP_cooling': cycle['performance']['COP_cooling'],
                'capacity_heating': cycle['performance']['capacity_heating'],
                'capacity_cooling': cycle['performance']['capacity_cooling'],
                'W_net': cycle['performance']['W_net'],
                'energy_balance_error': cycle['validation']['energy_balance_error'],
                'carnot_limit_check': cycle['validation']['carnot_limit_check']
            }
            
            # Add operating conditions
            if 'operating_conditions' in cycle:
                for key, value in cycle['operating_conditions'].items():
                    if isinstance(value, (int, float)):
                        row[f'op_{key}'] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_path / "synthetic_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved CSV dataset: {csv_path}")
        logger.info(f"Dataset shape: {df.shape}")

def dict_to_cycle_results(result_dict: Dict) -> CycleResults:
    """Convert dictionary back to CycleResults for validation"""
    # This is a simplified conversion for validation purposes
    from thermodynamic_properties import ThermodynamicState
    
    states = []
    for state_dict in result_dict['states']:
        state = ThermodynamicState(
            P=state_dict['P'],
            T=state_dict['T'],
            h=state_dict['h'],
            s=state_dict['s'],
            rho=state_dict['rho'],
            x=state_dict['x'],
            phase=state_dict['phase']
        )
        states.append(state)
    
    return CycleResults(
        states=states,
        state_labels=[s['location'] for s in result_dict['states']],
        Q_evap=result_dict['performance']['Q_evap'],
        Q_cond=result_dict['performance']['Q_cond'],
        W_comp=result_dict['performance']['W_comp'],
        W_exp=result_dict['performance']['W_exp'],
        W_net=result_dict['performance']['W_net'],
        COP_heating=result_dict['performance']['COP_heating'],
        COP_cooling=result_dict['performance']['COP_cooling'],
        energy_balance_error=result_dict['validation']['energy_balance_error'],
        entropy_generation=result_dict['validation']['entropy_generation'],
        carnot_limit_check=result_dict['validation']['carnot_limit_check'],
        cycle_type=result_dict['cycle_type'],
        working_fluid=result_dict['working_fluid']
    )

def main():
    """Main function to generate the complete dataset"""
    
    # Configuration
    config = DatasetConfiguration(
        total_cycles=145000,
        output_dir="generated_data",
        n_workers=4,  # Adjust based on your system
        batch_size=500
    )
    
    # Generate dataset
    generator = SyntheticDataGenerator(config)
    dataset = generator.generate_complete_dataset()
    
    # Print summary
    validation_report = dataset['validation_report']
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    print(f"Total cycles generated: {validation_report['total_cycles_generated']}")
    print(f"Valid cycles: {validation_report['valid_cycles']}")
    print(f"Validation rate: {validation_report['validation_rate']:.2f}%")
    print(f"Energy conservation compliance: {validation_report['physics_compliance']['energy_conservation']:.2f}%")
    print(f"Carnot limit compliance: {validation_report['physics_compliance']['carnot_limit_compliance']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()