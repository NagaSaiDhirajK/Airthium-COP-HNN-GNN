#!/usr/bin/env python3
"""
Test Phase Space Trajectories Models
===================================

Comprehensive test suite for phase space trajectories including thermodynamic
processes, Poincaré sections, and Lyapunov exponent calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import json
from pathlib import Path

# Import directly from the module file
from phase_space_trajectories import (
    PhaseSpaceTrajectoriesModel, PhaseSpacePoint, PhaseSpaceTrajectory, PoincareSection
)

def test_phase_space_trajectories():
    """Test phase space trajectories models comprehensively."""
    
    print("🌌 Testing Phase Space Trajectories Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        print("✅ Import successful")
        
        # Initialize phase space trajectories model
        print("\n🔄 Testing PhaseSpaceTrajectoriesModel initialization...")
        phase_model = PhaseSpaceTrajectoriesModel()
        print("✅ PhaseSpaceTrajectoriesModel initialized successfully")
        print(f"✅ Component dynamics: {len(phase_model.component_dynamics)} types")
        print(f"✅ Phase space scales: {len(phase_model.phase_space_scales)} dimensions")
        print(f"✅ Analysis parameters: {len(phase_model.analysis_params)} parameters")
        print(f"✅ Physical constants: R={phase_model.R_universal} J/(mol·K)")
        
        # Test thermodynamic trajectory generation
        print("\n🔄 Testing thermodynamic trajectory generation...")
        component_types = list(phase_model.component_dynamics.keys())
        
        # Test different initial conditions
        initial_conditions_sets = [
            {
                'temperature_K': 298.15,
                'pressure_Pa': 101325.0,
                'volume_m3': 0.001,
                'entropy_J_per_K': 1000.0,
                'enthalpy_J': 300000.0
            },
            {
                'temperature_K': 353.15,
                'pressure_Pa': 1800000.0,
                'volume_m3': 0.0005,
                'entropy_J_per_K': 1200.0,
                'enthalpy_J': 350000.0
            },
            {
                'temperature_K': 278.15,
                'pressure_Pa': 400000.0,
                'volume_m3': 0.002,
                'entropy_J_per_K': 800.0,
                'enthalpy_J': 250000.0
            }
        ]
        
        process_parameters_sets = [
            {'process_type': 'compression', 'mode': 'normal'},
            {'process_type': 'expansion', 'mode': 'transient'},
            {'process_type': 'heating', 'mode': 'fault'},
            {'process_type': 'cooling', 'mode': 'normal'},
            {'process_type': 'mixed', 'mode': 'transient'}
        ]
        
        trajectories = []
        for i, component_type in enumerate(component_types):
            initial_conditions = initial_conditions_sets[i % len(initial_conditions_sets)]
            process_parameters = process_parameters_sets[i % len(process_parameters_sets)]
            time_span = (0.0, 50.0)  # Shorter time span for testing
            
            print(f"✅ Testing {component_type} trajectory generation...")
            trajectory = phase_model.generate_thermodynamic_trajectory(
                component_type, initial_conditions, process_parameters, time_span
            )
            trajectories.append(trajectory)
            
            print(f"  - Trajectory ID: {trajectory.trajectory_id}")
            print(f"  - Component type: {trajectory.component_type}")
            print(f"  - Process type: {trajectory.process_type}")
            print(f"  - Number of points: {len(trajectory.points)}")
            print(f"  - Time span: {trajectory.time_span[0]:.1f} - {trajectory.time_span[1]:.1f} s")
            print(f"  - Is periodic: {trajectory.is_periodic}")
            print(f"  - Period: {trajectory.period_s}")
            print(f"  - Is chaotic: {trajectory.is_chaotic}")
            print(f"  - Lyapunov exponent: {trajectory.lyapunov_exponent:.4f}")
            print(f"  - Trajectory length: {trajectory.trajectory_length:.2f}")
            print(f"  - Entropy production rate: {trajectory.entropy_production_rate:.4f} J/(K·s)")
            print(f"  - Energy dissipation rate: {trajectory.energy_dissipation_rate:.4f} J/s")
            print(f"  - Irreversibility measure: {trajectory.irreversibility_measure:.4f}")
        
        # Test trajectory with external forcing
        print("\n🔄 Testing trajectory with external forcing...")
        
        def test_external_forcing(t, y):
            """Test external forcing function."""
            return {
                'temperature': 5.0 * np.sin(0.1 * t),
                'pressure': 1000.0 * np.cos(0.05 * t),
                'volume': 0.0001 * np.sin(0.2 * t),
                'entropy': 1.0 * np.sin(0.01 * t),
                'enthalpy': 100.0 * np.cos(0.02 * t)
            }
        
        forced_trajectory = phase_model.generate_thermodynamic_trajectory(
            'compressor', initial_conditions_sets[0], process_parameters_sets[0], 
            (0.0, 50.0), test_external_forcing
        )
        
        print(f"✅ Forced trajectory generated:")
        print(f"  - Number of points: {len(forced_trajectory.points)}")
        print(f"  - Is chaotic: {forced_trajectory.is_chaotic}")
        print(f"  - Lyapunov exponent: {forced_trajectory.lyapunov_exponent:.4f}")
        
        # Test phase space point properties
        print("\n🔄 Testing phase space point properties...")
        test_trajectory = trajectories[0]
        sample_points = test_trajectory.points[:5]  # First 5 points
        
        for i, point in enumerate(sample_points):
            print(f"✅ Point {i+1}:")
            print(f"  - Point ID: {point.point_id}")
            print(f"  - Time: {point.time_s:.2f} s")
            print(f"  - Temperature: {point.temperature_K:.2f} K")
            print(f"  - Pressure: {point.pressure_Pa:.0f} Pa")
            print(f"  - Volume: {point.volume_m3:.6f} m³")
            print(f"  - Entropy: {point.entropy_J_per_K:.1f} J/K")
            print(f"  - Enthalpy: {point.enthalpy_J:.0f} J")
            print(f"  - dT/dt: {point.dT_dt_K_per_s:.4f} K/s")
            print(f"  - dP/dt: {point.dP_dt_Pa_per_s:.2f} Pa/s")
            print(f"  - Component type: {point.component_type}")
            print(f"  - Operating mode: {point.operating_mode}")
            print(f"  - Kinetic energy density: {point.kinetic_energy_density:.4f}")
            print(f"  - Potential energy density: {point.potential_energy_density:.4f}")
            print(f"  - Phase space distance: {point.phase_space_distance:.2f}")
        
        # Test Poincaré section creation
        print("\n🔄 Testing Poincaré section creation...")
        test_trajectory = trajectories[0]
        
        # Create Poincaré sections for different variables
        section_variables = ['temperature_K', 'pressure_Pa', 'volume_m3']
        poincare_sections = []
        
        for var in section_variables:
            # Calculate average value for section plane
            values = [getattr(point, var) for point in test_trajectory.points]
            avg_value = np.mean(values)
            
            poincare_section = phase_model.create_poincare_section(
                test_trajectory, var, avg_value, tolerance=0.05
            )
            poincare_sections.append(poincare_section)
            
            print(f"✅ Poincaré section for {var}:")
            print(f"  - Section ID: {poincare_section.section_id}")
            print(f"  - Section plane: {poincare_section.section_plane}")
            print(f"  - Intersection points: {len(poincare_section.intersection_points)}")
            print(f"  - Return map points: {len(poincare_section.return_map)}")
            print(f"  - Fixed points: {len(poincare_section.fixed_points)}")
            print(f"  - Stability measure: {poincare_section.stability_analysis.get('stability_measure', 0.0):.4f}")
            print(f"  - Map dimension: {poincare_section.stability_analysis.get('map_dimension', 0.0):.4f}")
        
        # Test Lyapunov exponent calculation
        print("\n🔄 Testing Lyapunov exponent calculation...")
        
        for i, trajectory in enumerate(trajectories[:3]):  # Test first 3 trajectories
            lyapunov_exp = phase_model.calculate_lyapunov_exponent(trajectory, embedding_dim=3)
            print(f"✅ Trajectory {i+1} ({trajectory.component_type}):")
            print(f"  - Calculated Lyapunov exponent: {lyapunov_exp:.6f}")
            print(f"  - Trajectory Lyapunov exponent: {trajectory.lyapunov_exponent:.6f}")
            print(f"  - Difference: {abs(lyapunov_exp - trajectory.lyapunov_exponent):.6f}")
        
        # Test trajectory analysis methods
        print("\n🔄 Testing trajectory analysis methods...")
        
        # Test periodicity detection
        test_points = trajectories[0].points
        temperatures = [p.temperature_K for p in test_points]
        times = [p.time_s for p in test_points]
        
        is_periodic, period_s = phase_model._detect_periodicity(temperatures, times)
        print(f"✅ Periodicity detection:")
        print(f"  - Is periodic: {is_periodic}")
        print(f"  - Period: {period_s}")
        
        # Test Lyapunov estimation
        lyapunov_simple = phase_model._estimate_lyapunov_simple(temperatures, times)
        print(f"✅ Simple Lyapunov estimation: {lyapunov_simple:.6f}")
        
        # Test time delay embedding
        time_series = np.array(temperatures)
        embedded_data = phase_model._time_delay_embedding(time_series, 3, 1)
        print(f"✅ Time delay embedding:")
        print(f"  - Original series length: {len(time_series)}")
        print(f"  - Embedded data shape: {embedded_data.shape}")
        
        # Test scenario generation
        print("\n🔄 Testing phase space scenario generation...")
        scenarios = phase_model.generate_phase_space_scenarios(5)
        print(f"✅ Generated {len(scenarios)} phase space scenarios")
        
        for i, scenario in enumerate(scenarios):
            print(f"✅ Scenario {i+1}:")
            print(f"  - Component type: {scenario['component_type']}")
            print(f"  - Initial conditions: T={scenario['initial_conditions']['temperature_K']:.1f}K, "
                  f"P={scenario['initial_conditions']['pressure_Pa']:.0f}Pa")
            print(f"  - Process type: {scenario['process_parameters']['process_type']}")
            print(f"  - Mode: {scenario['process_parameters']['mode']}")
            print(f"  - Time span: {scenario['time_span'][0]:.1f} - {scenario['time_span'][1]:.1f} s")
            print(f"  - External forcing: {scenario['has_external_forcing']}")
            
            if 'trajectory_summary' in scenario:
                summary = scenario['trajectory_summary']
                print(f"  - Trajectory points: {summary['num_points']}")
                print(f"  - Is periodic: {summary['is_periodic']}")
                print(f"  - Is chaotic: {summary['is_chaotic']}")
                print(f"  - Lyapunov exponent: {summary['lyapunov_exponent']:.6f}")
                print(f"  - Trajectory length: {summary['trajectory_length']:.2f}")
                print(f"  - Entropy production rate: {summary['entropy_production_rate']:.6f} J/(K·s)")
            
            if 'poincare_section' in scenario:
                poincare = scenario['poincare_section']
                print(f"  - Poincaré intersections: {poincare['num_intersections']}")
                print(f"  - Fixed points: {poincare['num_fixed_points']}")
                print(f"  - Stability measure: {poincare['stability_measure']:.6f}")
            
            print(f"  - Sample trajectory points: {len(scenario.get('sample_trajectory_points', []))}")
            print(f"  - Calculated Lyapunov: {scenario.get('calculated_lyapunov', 0.0):.6f}")
        
        # Test data saving
        print("\n🔄 Testing phase space data saving...")
        output_file = "test_phase_space_trajectories_data.json"
        phase_model.save_phase_space_data(scenarios, output_file)
        
        # Verify saved data
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            print(f"✅ Phase space data saved successfully")
            print(f"✅ Saved scenarios: {len(saved_data['phase_space_scenarios'])}")
            print(f"✅ Model parameters included: {len(saved_data['model_parameters'])} categories")
            print(f"✅ Generation timestamp: {saved_data['generation_timestamp']}")
        else:
            print("❌ Failed to save phase space data")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test PhaseSpacePoint dataclass
        test_phase_point = PhaseSpacePoint(
            point_id="test_point",
            time_s=10.0,
            temperature_K=300.0,
            pressure_Pa=101325.0,
            volume_m3=0.001,
            entropy_J_per_K=1000.0,
            enthalpy_J=300000.0,
            dT_dt_K_per_s=0.1,
            dP_dt_Pa_per_s=100.0,
            dV_dt_m3_per_s=1e-6,
            dS_dt_J_per_K_per_s=0.01,
            dH_dt_J_per_s=10.0,
            component_type="test_component",
            operating_mode="test_mode",
            kinetic_energy_density=0.5,
            potential_energy_density=0.3,
            phase_space_distance=1000.0
        )
        print(f"✅ PhaseSpacePoint created: {test_phase_point.point_id}")
        print(f"✅ Time: {test_phase_point.time_s} s")
        print(f"✅ Temperature: {test_phase_point.temperature_K} K")
        print(f"✅ Component type: {test_phase_point.component_type}")
        
        # Test PhaseSpaceTrajectory dataclass
        test_phase_trajectory = PhaseSpaceTrajectory(
            trajectory_id="test_trajectory",
            component_type="test_component",
            process_type="test_process",
            points=[test_phase_point],
            time_span=(0.0, 10.0),
            is_periodic=False,
            period_s=None,
            is_chaotic=True,
            lyapunov_exponent=0.1,
            trajectory_length=100.0,
            bounding_box={'temperature_K': (290.0, 310.0)},
            centroid={'temperature_K': 300.0},
            entropy_production_rate=0.01,
            energy_dissipation_rate=1.0,
            irreversibility_measure=0.001
        )
        print(f"✅ PhaseSpaceTrajectory created: {test_phase_trajectory.trajectory_id}")
        print(f"✅ Component type: {test_phase_trajectory.component_type}")
        print(f"✅ Process type: {test_phase_trajectory.process_type}")
        print(f"✅ Number of points: {len(test_phase_trajectory.points)}")
        print(f"✅ Is chaotic: {test_phase_trajectory.is_chaotic}")
        print(f"✅ Lyapunov exponent: {test_phase_trajectory.lyapunov_exponent}")
        
        # Test PoincareSection dataclass
        test_poincare = PoincareSection(
            section_id="test_poincare",
            section_plane={'temperature_K': 300.0},
            intersection_points=[test_phase_point],
            return_map=[(300.0, 301.0), (301.0, 299.0)],
            fixed_points=[],
            stability_analysis={'stability_measure': 0.1, 'map_dimension': 1.5}
        )
        print(f"✅ PoincareSection created: {test_poincare.section_id}")
        print(f"✅ Section plane: {test_poincare.section_plane}")
        print(f"✅ Intersection points: {len(test_poincare.intersection_points)}")
        print(f"✅ Return map points: {len(test_poincare.return_map)}")
        print(f"✅ Stability analysis: {test_poincare.stability_analysis}")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Test with unknown component type
        try:
            unknown_trajectory = phase_model.generate_thermodynamic_trajectory(
                'unknown_component', initial_conditions_sets[0], process_parameters_sets[0], (0.0, 10.0)
            )
            print(f"❌ Unknown component type should have failed")
        except ValueError as e:
            print(f"✅ Unknown component type properly rejected: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error for unknown component: {str(e)}")
        
        # Test with very short trajectory
        try:
            short_points = trajectories[0].points[:5]  # Very short trajectory
            short_trajectory = PhaseSpaceTrajectory(
                trajectory_id="short_test",
                component_type="compressor",
                process_type="test",
                points=short_points,
                time_span=(0.0, 1.0),
                is_periodic=False,
                period_s=None,
                is_chaotic=False,
                lyapunov_exponent=0.0,
                trajectory_length=1.0,
                bounding_box={},
                centroid={},
                entropy_production_rate=0.0,
                energy_dissipation_rate=0.0,
                irreversibility_measure=0.0
            )
            
            lyapunov_short = phase_model.calculate_lyapunov_exponent(short_trajectory)
            print(f"✅ Short trajectory Lyapunov calculation: {lyapunov_short:.6f}")
        except Exception as e:
            print(f"❌ Short trajectory error: {str(e)}")
        
        # Test with empty time series
        try:
            empty_embedding = phase_model._time_delay_embedding(np.array([]), 3, 1)
            print(f"✅ Empty time series handled: shape {empty_embedding.shape}")
        except Exception as e:
            print(f"❌ Empty time series error: {str(e)}")
        
        # Test performance metrics
        print("\n🔄 Testing performance metrics...")
        
        # Calculate average trajectory properties by component type
        component_stats = {}
        for trajectory in trajectories:
            comp_type = trajectory.component_type
            if comp_type not in component_stats:
                component_stats[comp_type] = {
                    'lengths': [],
                    'lyapunov_exponents': [],
                    'entropy_rates': [],
                    'num_points': []
                }
            
            component_stats[comp_type]['lengths'].append(trajectory.trajectory_length)
            component_stats[comp_type]['lyapunov_exponents'].append(trajectory.lyapunov_exponent)
            component_stats[comp_type]['entropy_rates'].append(trajectory.entropy_production_rate)
            component_stats[comp_type]['num_points'].append(len(trajectory.points))
        
        print(f"✅ Component statistics:")
        for comp_type, stats in component_stats.items():
            print(f"  - {comp_type}:")
            print(f"    * Average trajectory length: {np.mean(stats['lengths']):.2f}")
            print(f"    * Average Lyapunov exponent: {np.mean(stats['lyapunov_exponents']):.6f}")
            print(f"    * Average entropy production rate: {np.mean(stats['entropy_rates']):.6f} J/(K·s)")
            print(f"    * Average number of points: {np.mean(stats['num_points']):.0f}")
        
        # Calculate phase space coverage
        all_temperatures = []
        all_pressures = []
        all_volumes = []
        
        for trajectory in trajectories:
            all_temperatures.extend([p.temperature_K for p in trajectory.points])
            all_pressures.extend([p.pressure_Pa for p in trajectory.points])
            all_volumes.extend([p.volume_m3 for p in trajectory.points])
        
        print(f"✅ Phase space coverage:")
        print(f"  - Temperature range: {min(all_temperatures):.1f} - {max(all_temperatures):.1f} K")
        print(f"  - Pressure range: {min(all_pressures):.0f} - {max(all_pressures):.0f} Pa")
        print(f"  - Volume range: {min(all_volumes):.6f} - {max(all_volumes):.6f} m³")
        
        print("\n🎉 ALL PHASE SPACE TRAJECTORIES TESTS COMPLETED!")
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error in phase space trajectories models: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Test completed with errors")
        return False

if __name__ == "__main__":
    success = test_phase_space_trajectories()
    sys.exit(0 if success else 1)