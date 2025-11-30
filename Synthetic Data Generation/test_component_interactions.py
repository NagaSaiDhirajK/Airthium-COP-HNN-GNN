#!/usr/bin/env python3
"""
Comprehensive test script for Component Interactions Models
Tests all major component interaction and heat/mass transfer network functionality
"""

import sys
sys.path.append('src')

def test_component_interactions_models():
    """Test all component interaction model functionality"""
    print("🔗 Testing Component Interactions Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from component_interactions import (
            ComponentInteractionModel, ComponentNode, InteractionEdge, SystemInteractionMatrix
        )
        import numpy as np
        import pandas as pd
        from datetime import datetime
        print("✅ Import successful")
        
        # Test ComponentInteractionModel initialization
        print("\n🔄 Testing ComponentInteractionModel initialization...")
        interaction_model = ComponentInteractionModel()
        print("✅ ComponentInteractionModel initialized successfully")
        print(f"✅ Component properties: {len(interaction_model.component_properties)}")
        print(f"✅ Component labels: {len(interaction_model.component_labels)}")
        print(f"✅ Interaction strengths: {len(interaction_model.base_interaction_strengths)}")
        
        # Test component network creation
        print("\n🔄 Testing component network creation...")
        system_id = "test_system_001"
        
        # Create test component states
        component_states = {
            'compressor': {
                'temperature_K': 350.0,  # K
                'pressure_Pa': 1800000,   # Pa
                'mass_flow_rate_kg_per_s': 0.5,      # kg/s
            },
            'evaporator': {
                'temperature_K': 278.15,  # K
                'pressure_Pa': 400000,     # Pa
                'mass_flow_rate_kg_per_s': 0.5,       # kg/s
            },
            'condenser': {
                'temperature_K': 318.15,  # K
                'pressure_Pa': 1800000,    # Pa
                'mass_flow_rate_kg_per_s': 0.5,       # kg/s
            },
            'expansion_valve': {
                'temperature_K': 318.15,  # K
                'pressure_Pa': 900000, # Pa
                'mass_flow_rate_kg_per_s': 0.5,       # kg/s
            },
            'refrigerant_lines': {
                'temperature_K': 288.15,  # K
                'pressure_Pa': 600000,    # Pa
                'mass_flow_rate_kg_per_s': 0.5,       # kg/s
            },
            'fan': {
                'temperature_K': 298.15,  # K
                'pressure_Pa': 101325,    # Pa
                'mass_flow_rate_kg_per_s': 0.0,       # kg/s
            }
        }
        
        nodes, edges = interaction_model.create_component_network(system_id, component_states)
        
        print(f"✅ Created component network: {len(nodes)} nodes, {len(edges)} edges")
        print(f"✅ Node types: {[node.component_type for node in nodes]}")
        print(f"✅ Edge types: {[edge.interaction_type for edge in edges]}")
        
        # Test individual nodes
        for i, node in enumerate(nodes):
            print(f"✅ Node {i}: {node.node_id} ({node.component_type})")
            print(f"  - Temperature: {node.temperature_K:.1f} K")
            print(f"  - Pressure: {node.pressure_Pa:.0f} Pa")
            print(f"  - Mass flow: {node.mass_flow_rate_kg_per_s:.2f} kg/s")
        
        # Test individual edges
        for i, edge in enumerate(edges):
            print(f"✅ Edge {i}: {edge.source_node} → {edge.target_node}")
            print(f"  - Type: {edge.interaction_type}")
            print(f"  - Conductance: {edge.conductance_W_per_K:.2f} W/K")
            print(f"  - Resistance: {edge.flow_resistance_Pa_s_per_kg:.2f} Pa·s/kg")
        
        # Test system interaction matrix
        print("\n🔄 Testing system interaction matrix...")
        interaction_matrix = interaction_model.build_system_interaction_matrix(system_id, nodes, edges)
        
        print("✅ System interaction matrix created successfully")
        print(f"✅ System ID: {interaction_matrix.matrix_id}")
        print(f"✅ Matrix dimensions: {interaction_matrix.thermal_interaction_matrix.shape}")
        print(f"✅ Mass matrix dimensions: {interaction_matrix.mass_interaction_matrix.shape}")
        print(f"✅ Component count: {len(interaction_matrix.component_list)}")
        print(f"✅ Stability margin: {interaction_matrix.stability_margin:.3f}")
        print(f"✅ Coupling strength: {interaction_matrix.coupling_strength:.3f}")
        
        # Display matrix values
        print(f"✅ Thermal matrix:\n{interaction_matrix.thermal_interaction_matrix}")
        print(f"✅ Mass matrix:\n{interaction_matrix.mass_interaction_matrix}")
        
        # Test heat transfer network simulation
        print("\n🔄 Testing heat transfer network simulation...")
        heat_sources = {'compressor': 1000.0, 'evaporator': -2000.0}  # W
        
        heat_results = interaction_model.simulate_heat_transfer_network(
            interaction_matrix, nodes, heat_sources, 300.0, 5.0
        )
        
        print("✅ Heat transfer network simulation completed")
        print(f"✅ Simulation results: {len(heat_results)} time steps")
        print(f"✅ Initial temperatures:")
        for key, value in heat_results[0].items():
            if 'temperature' in key:
                print(f"  - {key}: {value:.2f} K")
        
        # Test network properties analysis
        print("\n🔄 Testing network properties analysis...")
        network_props = interaction_model.analyze_network_properties(interaction_matrix)
        
        print("✅ Network properties analyzed")
        print(f"✅ Network properties: {list(network_props.keys())}")
        for prop, value in network_props.items():
            if isinstance(value, float):
                print(f"  - {prop}: {value:.3f}")
            else:
                print(f"  - {prop}: {value}")
        
        # Test interaction scenarios generation
        print("\n🔄 Testing interaction scenarios generation...")
        num_scenarios = 5
        scenarios = interaction_model.generate_interaction_scenarios(system_id, num_scenarios)
        
        print(f"✅ Generated {len(scenarios)} interaction scenarios")
        for i, scenario in enumerate(scenarios):
            print(f"✅ Scenario {i+1}:")
            print(f"  - Scenario ID: {scenario['scenario_id']}")
            print(f"  - Component states: {len(scenario['component_states'])} components")
            print(f"  - Network properties: {len(scenario['network_properties'])} metrics")
            print(f"  - Stability: {scenario['network_properties']['stability_margin']:.3f}")
            print(f"  - Coupling: {scenario['network_properties']['coupling_strength']:.3f}")
        
        # Test data saving
        print("\n🔄 Testing interaction data saving...")
        test_scenarios = scenarios[:3]  # Save first 3 scenarios
        interaction_model.save_interaction_data(test_scenarios, "test_component_interactions_data.json")
        print("✅ Component interaction data saved successfully")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test ComponentNode
        test_node = ComponentNode(
            node_id="test_compressor",
            component_type="compressor",
            position=(1.0, 2.0, 3.0),
            thermal_mass_J_per_K=15000.0,
            thermal_resistance_K_per_W=0.02,
            surface_area_m2=2.5,
            temperature_K=350.0,
            pressure_Pa=1800000,
            mass_flow_rate_kg_per_s=0.5,
            convection_coefficient_W_per_m2_K=25.0,
            radiation_coefficient_W_per_m2_K=5.0,
            conduction_coefficient_W_per_m_K=50.0
        )
        print(f"✅ ComponentNode created: {test_node.node_id}")
        print(f"✅ Component type: {test_node.component_type}")
        print(f"✅ Temperature: {test_node.temperature_K} K")
        print(f"✅ Thermal mass: {test_node.thermal_mass_J_per_K} J/K")
        
        # Test InteractionEdge
        test_edge = InteractionEdge(
            edge_id="test_edge",
            source_node="compressor",
            target_node="condenser",
            interaction_type="heat_transfer",
            conductance_W_per_K=150.0,
            flow_resistance_Pa_s_per_kg=1000.0,
            coupling_strength=0.8,
            time_constant_s=30.0,
            delay_s=1.0
        )
        print(f"✅ InteractionEdge created: {test_edge.source_node} → {test_edge.target_node}")
        print(f"✅ Interaction type: {test_edge.interaction_type}")
        print(f"✅ Thermal conductance: {test_edge.conductance_W_per_K} W/K")
        print(f"✅ Flow resistance: {test_edge.flow_resistance_Pa_s_per_kg} Pa·s/kg")
        
        # Test SystemInteractionMatrix
        test_matrix = SystemInteractionMatrix(
            matrix_id="test_matrix",
            component_list=["comp1", "comp2", "comp3"],
            thermal_interaction_matrix=np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),
            mass_interaction_matrix=np.array([[1.0, 0.1, 0.05], [0.1, 1.0, 0.15], [0.05, 0.15, 1.0]]),
            control_interaction_matrix=np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.25], [0.1, 0.25, 1.0]]),
            connectivity_matrix=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            adjacency_matrix=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            eigenvalues=np.array([1.7, 0.8, 0.5]),
            stability_margin=0.85,
            coupling_strength=0.65
        )
        print(f"✅ SystemInteractionMatrix created: {test_matrix.matrix_id}")
        print(f"✅ Components: {len(test_matrix.component_list)}")
        print(f"✅ Stability margin: {test_matrix.stability_margin}")
        print(f"✅ Coupling strength: {test_matrix.coupling_strength}")
        
        # Test edge cases and error handling
        print("\n🔄 Testing edge cases...")
        
        # Test with empty component states
        try:
            empty_nodes, empty_edges = interaction_model.create_component_network("empty_system", {})
            print(f"✅ Empty system handling: {len(empty_nodes)} nodes, {len(empty_edges)} edges")
        except Exception as e:
            print(f"⚠️ Empty system error (expected): {e}")
        
        # Test with single component
        single_component = {
            'compressor': component_states['compressor']
        }
        single_nodes, single_edges = interaction_model.create_component_network("single_system", single_component)
        print(f"✅ Single component system: {len(single_nodes)} nodes, {len(single_edges)} edges")
        
        # Test thermal conductance calculation
        print("\n🔄 Testing thermal conductance calculations...")
        test_conductances = []
        component_pairs = [
            ('compressor', 'condenser'),
            ('condenser', 'expansion_valve'),
            ('expansion_valve', 'evaporator'),
            ('evaporator', 'compressor')
        ]
        
        for source, target in component_pairs:
            if source in component_states and target in component_states:
                conductance = interaction_model._calculate_thermal_conductance(
                    source, target, component_states
                )
                test_conductances.append(conductance)
                print(f"✅ {source} → {target}: {conductance:.2f} W/K")
        
        # Test flow resistance calculation
        print("\n🔄 Testing flow resistance calculations...")
        test_resistances = []
        
        for source, target in component_pairs:
            if source in component_states and target in component_states:
                resistance = interaction_model._calculate_flow_resistance(
                    source, target, component_states
                )
                test_resistances.append(resistance)
                print(f"✅ {source} → {target}: {resistance:.2f} Pa·s/kg")
        
        print("\n🎉 ALL COMPONENT INTERACTIONS TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in component interactions models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_component_interactions_models()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)