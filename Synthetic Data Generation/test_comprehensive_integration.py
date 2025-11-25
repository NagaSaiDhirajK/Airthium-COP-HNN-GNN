#!/usr/bin/env python3
"""
Comprehensive Integration Test for Enhanced Data Generation Pipeline
===================================================================

This script tests the complete integration of all 23+ physics models with
comprehensive CoolProp fluid support for HNN/GNN training data generation.

Author: AIRTHIUM COP Prediction System
Date: 2024
"""

import sys
import os
sys.path.append('src')

import numpy as np
import json
import time
from pathlib import Path
import traceback

# Import comprehensive physics models
try:
    from enhanced_data_pipeline import EnhancedDatasetConfiguration, EnhancedSyntheticDataGenerator
    from molecular_refrigerant_properties import MolecularRefrigerantModel, RefrigerantType
    from hamiltonian_mechanics import HamiltonianMechanicsModel
    from phase_space_trajectories import PhaseSpaceTrajectoriesModel
    from component_interactions import ComponentInteractionModel
    from building_load_models import MultiZoneBuildingModel
    COMPREHENSIVE_MODELS_AVAILABLE = True
    print("✅ All comprehensive physics models imported successfully")
except ImportError as e:
    COMPREHENSIVE_MODELS_AVAILABLE = False
    print(f"⚠️ Some models not available: {e}")

def test_molecular_refrigerant_comprehensive():
    """Test comprehensive molecular refrigerant properties with CoolProp"""
    
    print("\n" + "="*80)
    print("🔬 TESTING COMPREHENSIVE MOLECULAR REFRIGERANT PROPERTIES")
    print("="*80)
    
    try:
        # Initialize molecular model
        model = MolecularRefrigerantModel()
        
        # Test comprehensive fluid database
        available_fluids = model.get_available_fluids()
        print(f"✅ Total available fluids: {len(available_fluids)}")
        
        # Test fluid categories
        categories = model.get_fluid_categories()
        print(f"✅ Fluid categories: {len(categories)}")
        
        total_fluids = 0
        for category, fluids in categories.items():
            print(f"  • {category}: {len(fluids)} fluids")
            total_fluids += len(fluids)
        
        print(f"✅ Total categorized fluids: {total_fluids}")
        
        # Test different fluid types
        test_fluids = [
            RefrigerantType.R410A,    # HFC blend
            RefrigerantType.R32,      # HFC
            RefrigerantType.R1234YF,  # HFO
            RefrigerantType.R290,     # Natural (propane)
            RefrigerantType.R717,     # Natural (ammonia)
            RefrigerantType.R744,     # Natural (CO2)
            RefrigerantType.TOLUENE,  # Organic
            RefrigerantType.ETHANOL,  # Organic
        ]
        
        successful_tests = 0
        for refrigerant in test_fluids:
            try:
                print(f"\n🔄 Testing {refrigerant.value}...")
                
                # Test molecular state calculation
                molecular_state = model.calculate_molecular_state(
                    refrigerant, 
                    temperature_K=300.0, 
                    pressure_Pa=1e6
                )
                
                # Test transport properties
                transport = model.calculate_transport_properties(refrigerant, molecular_state)
                
                # Test phase transition analysis
                phase_data = model.analyze_phase_transition(refrigerant, molecular_state)
                
                # Test intermolecular forces
                forces = model.calculate_intermolecular_forces(refrigerant, molecular_state)
                
                print(f"  ✅ Molecular state: T={molecular_state.temperature_K:.1f}K, "
                      f"ρ={molecular_state.density_kg_per_m3:.1f}kg/m³")
                print(f"  ✅ Transport: μ={transport.viscosity_Pa_s:.2e}Pa·s, "
                      f"λ={transport.thermal_conductivity_W_per_m_K:.3f}W/(m·K)")
                print(f"  ✅ Phase: {phase_data.phase}, T_sat={phase_data.saturation_temperature_K:.1f}K")
                print(f"  ✅ Forces: LJ={forces['lennard_jones_potential_J']:.2e}J")
                
                successful_tests += 1
                
            except Exception as e:
                print(f"  ❌ Error with {refrigerant.value}: {e}")
        
        print(f"\n✅ Molecular refrigerant tests: {successful_tests}/{len(test_fluids)} successful")
        return successful_tests == len(test_fluids)
        
    except Exception as e:
        print(f"❌ Molecular refrigerant test failed: {e}")
        print(traceback.format_exc())
        return False

def test_enhanced_data_pipeline_comprehensive():
    """Test comprehensive enhanced data pipeline integration"""
    
    print("\n" + "="*80)
    print("🚀 TESTING COMPREHENSIVE ENHANCED DATA PIPELINE")
    print("="*80)
    
    try:
        # Create comprehensive configuration
        config = EnhancedDatasetConfiguration(
            total_cycles=5,  # Small for testing
            hnn_training_cycles=3,
            gnn_training_cycles=2,
            output_dir='test_comprehensive_integration',
            n_workers=1,
            batch_size=2,
            
            # Enable all physics models
            enable_hamiltonian_mechanics=True,
            enable_phase_space_trajectories=True,
            enable_molecular_dynamics=True,
            enable_component_interactions=True,
            enable_temporal_dynamics=True,
            enable_sensor_noise=True,
            enable_degradation_models=True,
            enable_fault_injection=True,
            enable_building_loads=True,
            enable_seasonal_variations=True,
            enable_defrost_cycles=True,
            
            # Comprehensive fluid selection
            primary_refrigerants=None,  # Will use CoolProp comprehensive list
            secondary_fluids=None,      # Will use CoolProp comprehensive list
        )
        
        print(f"✅ Configuration created with {config.total_cycles} cycles")
        print(f"✅ Primary refrigerants: {len(config.primary_refrigerants)}")
        print(f"✅ Secondary fluids: {len(config.secondary_fluids)}")
        
        # Initialize comprehensive data generator
        generator = EnhancedSyntheticDataGenerator(config)
        print(f"✅ Enhanced data generator initialized")
        print(f"✅ Physics models available: {len(generator.physics_models)}")
        
        # Test individual physics model integration
        test_results = {}
        
        # Test base cycle generation with comprehensive physics
        print(f"\n🔄 Testing comprehensive base cycle generation...")
        test_fluids = ['R410A', 'R32', 'R290', 'Toluene']
        
        for fluid in test_fluids:
            try:
                cycle_data = generator._generate_base_cycle_with_physics(fluid)
                
                # Check comprehensive data structure
                required_fields = ['cycle_id', 'working_fluid', 'states', 'performance']
                physics_fields = ['molecular_data', 'sensor_data', 'degradation_data', 
                                'building_data', 'seasonal_data', 'transient_data']
                
                basic_check = all(field in cycle_data for field in required_fields)
                physics_check = any(field in cycle_data for field in physics_fields)
                
                print(f"  ✅ {fluid}: Basic fields={basic_check}, Physics data={physics_check}")
                test_results[f'base_cycle_{fluid}'] = basic_check and physics_check
                
            except Exception as e:
                print(f"  ❌ {fluid}: {e}")
                test_results[f'base_cycle_{fluid}'] = False
        
        # Test Hamiltonian mechanics integration
        print(f"\n🔄 Testing Hamiltonian mechanics integration...")
        try:
            cycle_data = generator._generate_base_cycle_with_physics('R410A')
            hamiltonian_data = generator._generate_hamiltonian_data(cycle_data)
            
            hamiltonian_success = (
                'error' not in hamiltonian_data and
                'hamiltonian_trajectory' in hamiltonian_data and
                len(hamiltonian_data['hamiltonian_trajectory']) > 0
            )
            
            print(f"  ✅ Hamiltonian integration: {hamiltonian_success}")
            test_results['hamiltonian'] = hamiltonian_success
            
        except Exception as e:
            print(f"  ❌ Hamiltonian integration: {e}")
            test_results['hamiltonian'] = False
        
        # Test phase-space trajectory integration
        print(f"\n🔄 Testing phase-space trajectory integration...")
        try:
            trajectory_data = generator._generate_trajectory_data(cycle_data)
            
            trajectory_success = (
                'error' not in trajectory_data and
                'trajectory_points' in trajectory_data and
                len(trajectory_data['trajectory_points']) > 0
            )
            
            print(f"  ✅ Trajectory integration: {trajectory_success}")
            test_results['trajectory'] = trajectory_success
            
        except Exception as e:
            print(f"  ❌ Trajectory integration: {e}")
            test_results['trajectory'] = False
        
        # Test graph data integration
        print(f"\n🔄 Testing graph data integration...")
        try:
            graph_data = generator._generate_graph_data(cycle_data)
            
            graph_success = (
                'error' not in graph_data and
                'nodes' in graph_data and
                'edges' in graph_data and
                len(graph_data['nodes']) > 0 and
                len(graph_data['edges']) > 0
            )
            
            print(f"  ✅ Graph integration: {graph_success}")
            test_results['graph'] = graph_success
            
        except Exception as e:
            print(f"  ❌ Graph integration: {e}")
            test_results['graph'] = False
        
        # Test comprehensive validation
        print(f"\n🔄 Testing comprehensive validation...")
        try:
            validation_success = generator.validator.validate_enhanced_cycle(cycle_data)
            print(f"  ✅ Validation: {validation_success}")
            test_results['validation'] = validation_success
            
        except Exception as e:
            print(f"  ❌ Validation: {e}")
            test_results['validation'] = False
        
        # Summary
        successful_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n✅ Enhanced data pipeline tests: {successful_tests}/{total_tests} successful")
        print(f"✅ Test results: {test_results}")
        
        return successful_tests >= total_tests * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Enhanced data pipeline test failed: {e}")
        print(traceback.format_exc())
        return False

def test_comprehensive_dataset_generation():
    """Test comprehensive dataset generation with all physics models"""
    
    print("\n" + "="*80)
    print("📊 TESTING COMPREHENSIVE DATASET GENERATION")
    print("="*80)
    
    try:
        # Create minimal configuration for full test
        config = EnhancedDatasetConfiguration(
            total_cycles=3,  # Very small for testing
            hnn_training_cycles=2,
            gnn_training_cycles=1,
            output_dir='test_comprehensive_dataset',
            n_workers=1,
            batch_size=1,
            
            # Enable key physics models
            enable_hamiltonian_mechanics=True,
            enable_phase_space_trajectories=True,
            enable_molecular_dynamics=True,
            enable_component_interactions=True,
            enable_temporal_dynamics=True,
        )
        
        # Generate comprehensive dataset
        generator = EnhancedSyntheticDataGenerator(config)
        print(f"✅ Generator initialized with {len(generator.physics_models)} physics models")
        
        # Test dataset generation
        print(f"\n🔄 Generating comprehensive dataset...")
        start_time = time.time()
        
        dataset = generator.generate_enhanced_dataset()
        
        generation_time = time.time() - start_time
        print(f"✅ Dataset generated in {generation_time:.2f} seconds")
        
        # Validate dataset structure
        required_sections = ['metadata', 'hnn_training_data', 'gnn_training_data', 
                           'test_data', 'validation_report', 'physics_model_info']
        
        structure_valid = all(section in dataset for section in required_sections)
        print(f"✅ Dataset structure valid: {structure_valid}")
        
        # Check data content
        hnn_data_count = len(dataset.get('hnn_training_data', []))
        gnn_data_count = len(dataset.get('gnn_training_data', []))
        test_data_count = len(dataset.get('test_data', []))
        
        print(f"✅ HNN training data: {hnn_data_count} cycles")
        print(f"✅ GNN training data: {gnn_data_count} cycles")
        print(f"✅ Test data: {test_data_count} cycles")
        
        # Check validation report
        validation_report = dataset.get('validation_report', {})
        if validation_report and 'validation_rate' in validation_report:
            validation_rate = validation_report['validation_rate']
            print(f"✅ Validation rate: {validation_rate:.1f}%")
        
        # Check physics model integration
        physics_info = dataset.get('physics_model_info', {})
        integration_status = physics_info.get('integration_status', {})
        
        if integration_status:
            print(f"✅ Physics integration status:")
            for key, status in integration_status.items():
                print(f"  • {key}: {status}")
        
        # Overall success criteria
        success = (
            structure_valid and
            hnn_data_count > 0 and
            gnn_data_count > 0 and
            validation_report.get('validation_rate', 0) > 50
        )
        
        print(f"\n✅ Comprehensive dataset generation: {'SUCCESS' if success else 'PARTIAL'}")
        return success
        
    except Exception as e:
        print(f"❌ Comprehensive dataset generation failed: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Run comprehensive integration tests"""
    
    print("🚀 COMPREHENSIVE INTEGRATION TEST FOR ENHANCED DATA GENERATION PIPELINE")
    print("=" * 100)
    print("Testing complete integration of 23+ physics models with CoolProp support")
    print("for Hamiltonian Neural Network and Graph Neural Network training data")
    print("=" * 100)
    
    if not COMPREHENSIVE_MODELS_AVAILABLE:
        print("❌ Comprehensive models not available - cannot run full integration test")
        return False
    
    # Run comprehensive tests
    test_results = {}
    
    # Test 1: Molecular refrigerant properties with CoolProp
    print(f"\n{'='*20} TEST 1: MOLECULAR REFRIGERANT PROPERTIES {'='*20}")
    test_results['molecular_properties'] = test_molecular_refrigerant_comprehensive()
    
    # Test 2: Enhanced data pipeline integration
    print(f"\n{'='*20} TEST 2: ENHANCED DATA PIPELINE INTEGRATION {'='*20}")
    test_results['data_pipeline'] = test_enhanced_data_pipeline_comprehensive()
    
    # Test 3: Comprehensive dataset generation
    print(f"\n{'='*20} TEST 3: COMPREHENSIVE DATASET GENERATION {'='*20}")
    test_results['dataset_generation'] = test_comprehensive_dataset_generation()
    
    # Final summary
    print("\n" + "="*100)
    print("🎯 COMPREHENSIVE INTEGRATION TEST SUMMARY")
    print("="*100)
    
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    overall_success = successful_tests >= total_tests * 0.8  # 80% success rate
    
    print(f"\n🏆 OVERALL RESULT: {successful_tests}/{total_tests} tests passed")
    print(f"🎉 COMPREHENSIVE INTEGRATION: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print("\n✅ The enhanced data generation pipeline with comprehensive CoolProp")
        print("   integration is ready for large-scale HNN/GNN training data generation!")
    else:
        print("\n⚠️ Some integration issues detected. Review failed tests above.")
    
    print("="*100)
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)