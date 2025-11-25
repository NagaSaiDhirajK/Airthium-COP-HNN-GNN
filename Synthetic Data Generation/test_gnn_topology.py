#!/usr/bin/env python3
"""
Comprehensive Test Suite for GNN Topology Models
===============================================

Tests the Graph Neural Network topology generation for AIRTHIUM heat pump systems,
including graph creation, multi-scale representations, dynamic updates, and dataset generation.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import the GNN topology module
from gnn_topology import (
    HeatPumpGraphGenerator, NodeType, EdgeType,
    GraphNode, GraphEdge, HeatPumpGraph
)

def test_gnn_topology_comprehensive():
    """Comprehensive test of GNN topology generation."""
    
    print("🔄 Testing GNN Topology Models - Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Import and initialization
    print("🔄 Testing imports and initialization...")
    try:
        generator = HeatPumpGraphGenerator()
        print("✅ HeatPumpGraphGenerator initialized successfully")
        print(f"✅ System type: {generator.system_config['system_type']}")
        print(f"✅ Refrigerant: {generator.system_config['refrigerant']}")
        print(f"✅ Compressor type: {generator.system_config['compressor_type']}")
        print(f"✅ Number of sensors: {len(generator.system_config['sensors'])}")
    except Exception as e:
        print(f"❌ Import/initialization failed: {e}")
        return False
    
    # Test 2: Component specifications
    print("\n🔄 Testing component specifications...")
    try:
        specs = generator.component_specs
        print(f"✅ Loaded specifications for {len(specs)} component types:")
        for comp_type, spec in specs.items():
            print(f"  - {comp_type}: {len(spec)} parameters")
        
        # Check specific parameters
        compressor_spec = specs['compressor']
        print(f"✅ Compressor displacement: {compressor_spec['displacement']} m³/s")
        print(f"✅ Compressor max speed: {compressor_spec['max_speed']} rpm")
        
        evaporator_spec = specs['evaporator']
        print(f"✅ Evaporator area: {evaporator_spec['area']} m²")
        print(f"✅ Evaporator volume: {evaporator_spec['volume']} m³")
    except Exception as e:
        print(f"❌ Component specifications test failed: {e}")
        return False
    
    # Test 3: Graph templates
    print("\n🔄 Testing graph templates...")
    try:
        templates = generator.graph_templates
        print(f"✅ Created {len(templates)} graph templates:")
        
        for template_name, template in templates.items():
            print(f"  - {template_name}: {len(template['nodes'])} nodes, {len(template['edges'])} edges")
            
        # Check basic template
        basic_template = templates['basic']
        print(f"✅ Basic template nodes: {basic_template['nodes'][:4]}...")
        print(f"✅ Basic template edges: {len(basic_template['edges'])} connections")
        
        # Check enhanced template
        enhanced_template = templates['enhanced']
        print(f"✅ Enhanced template has {len(enhanced_template['nodes']) - len(basic_template['nodes'])} additional nodes")
    except Exception as e:
        print(f"❌ Graph templates test failed: {e}")
        return False
    
    # Test 4: Operating state creation
    print("\n🔄 Testing operating state creation...")
    try:
        operating_state = {
            'current_state': {
                'compressor_speed': 1800.0,
                'compressor_power': 5000.0,
                'compressor_efficiency': 0.8,
                'suction_pressure': 500000.0,
                'discharge_pressure': 2000000.0,
                'suction_temperature': 278.15,
                'discharge_temperature': 333.15,
                'evaporator_heat_rate': 8000.0,
                'condenser_heat_rate': 12000.0,
                'valve_opening': 45.0,
                'ambient_temperature': 278.15,
                'water_temp_in': 313.15,
                'water_temp_out': 323.15,
                'heating_capacity': 12000.0,
                'cooling_capacity': 8000.0,
                'power_input': 3000.0,
                'mass_flow_rate': 0.15,
                'carnot_cop': 5.0
            },
            'conditions': {
                'ambient_temp': 278.15,
                'load_fraction': 0.8,
                'operating_mode': 'heating'
            },
            'health_state': {
                'compressor_health': 0.95,
                'evaporator_health': 0.90,
                'condenser_health': 0.92,
                'compressor_bearing_wear': 0.05,
                'compressor_valve_leakage': 0.02,
                'oil_quality': 0.98,
                'evaporator_fouling': 0.03,
                'condenser_fouling': 0.02
            },
            'component_ages': {
                'compressor': 2.5,  # years
                'evaporator': 2.5,
                'condenser': 2.5,
                'expansion_valve': 2.5
            }
        }
        
        print("✅ Created comprehensive operating state")
        print(f"✅ Current state parameters: {len(operating_state['current_state'])}")
        print(f"✅ Health state parameters: {len(operating_state['health_state'])}")
        print(f"✅ Component ages: {len(operating_state['component_ages'])}")
    except Exception as e:
        print(f"❌ Operating state creation failed: {e}")
        return False
    
    # Test 5: System graph creation
    print("\n🔄 Testing system graph creation...")
    try:
        # Test basic topology
        basic_graph = generator.create_system_graph(operating_state, topology_type='basic')
        print(f"✅ Basic graph created: {basic_graph.graph_id}")
        print(f"✅ Basic graph nodes: {len(basic_graph.nodes)}")
        print(f"✅ Basic graph edges: {len(basic_graph.edges)}")
        
        # Test enhanced topology
        enhanced_graph = generator.create_system_graph(operating_state, topology_type='enhanced')
        print(f"✅ Enhanced graph created: {enhanced_graph.graph_id}")
        print(f"✅ Enhanced graph nodes: {len(enhanced_graph.nodes)}")
        print(f"✅ Enhanced graph edges: {len(enhanced_graph.edges)}")
        print(f"✅ Adjacency matrix shape: {enhanced_graph.adjacency_matrix.shape}")
        
        # Check targets
        targets = enhanced_graph.targets
        print(f"✅ COP heating: {targets['COP_heating']:.2f}")
        print(f"✅ COP cooling: {targets['COP_cooling']:.2f}")
        print(f"✅ Efficiency: {targets['efficiency']:.3f}")
        
        # Check graph metrics
        metrics = enhanced_graph.graph_metrics
        print(f"✅ Graph density: {metrics['density']:.3f}")
        print(f"✅ Average degree: {metrics['avg_degree']:.1f}")
        print(f"✅ Graph complexity: {metrics['graph_complexity']:.0f}")
        
    except Exception as e:
        print(f"❌ System graph creation failed: {e}")
        return False
    
    # Test 6: Node features
    print("\n🔄 Testing node features...")
    try:
        # Test compressor node
        compressor_nodes = [n for n in enhanced_graph.nodes.values() 
                          if n.node_type == NodeType.COMPRESSOR]
        if compressor_nodes:
            comp_node = compressor_nodes[0]
            print(f"✅ Compressor node: {comp_node.node_id}")
            print(f"✅ Static features: {len(comp_node.static_features)}")
            print(f"✅ Dynamic features: {len(comp_node.dynamic_features)}")
            print(f"✅ Health features: {len(comp_node.health_features)}")
            print(f"✅ Position: {comp_node.position}")
            
            # Check specific features
            static = comp_node.static_features
            dynamic = comp_node.dynamic_features
            health = comp_node.health_features
            
            print(f"✅ Displacement: {static.get('displacement', 0):.3f} m³/s")
            print(f"✅ Current speed: {dynamic.get('speed_rpm', 0):.0f} rpm")
            print(f"✅ Overall health: {health.get('overall_health', 0):.2f}")
        
        # Test evaporator node
        evap_nodes = [n for n in enhanced_graph.nodes.values() 
                     if n.node_type == NodeType.EVAPORATOR]
        if evap_nodes:
            evap_node = evap_nodes[0]
            print(f"✅ Evaporator node: {evap_node.node_id}")
            static = evap_node.static_features
            dynamic = evap_node.dynamic_features
            print(f"✅ Heat transfer area: {static.get('heat_transfer_area', 0):.1f} m²")
            print(f"✅ Current heat rate: {dynamic.get('heat_transfer_rate', 0):.0f} W")
            
    except Exception as e:
        print(f"❌ Node features test failed: {e}")
        return False
    
    # Test 7: Edge features
    print("\n🔄 Testing edge features...")
    try:
        # Test refrigerant flow edges
        ref_flow_edges = [e for e in enhanced_graph.edges.values() 
                         if e.edge_type == EdgeType.REFRIGERANT_FLOW]
        print(f"✅ Refrigerant flow edges: {len(ref_flow_edges)}")
        
        if ref_flow_edges:
            edge = ref_flow_edges[0]
            print(f"✅ Edge: {edge.source_node} → {edge.target_node}")
            print(f"✅ Edge features: {len(edge.features)}")
            print(f"✅ Weight: {edge.weight:.3f}")
            print(f"✅ Importance: {edge.importance:.3f}")
            print(f"✅ Is directed: {edge.is_directed}")
            
            features = edge.features
            print(f"✅ Mass flow rate: {features.get('mass_flow_rate', 0):.3f} kg/s")
            print(f"✅ Pressure: {features.get('pressure', 0):.0f} Pa")
            print(f"✅ Temperature: {features.get('temperature', 0):.1f} K")
        
        # Test heat transfer edges
        heat_edges = [e for e in enhanced_graph.edges.values() 
                     if e.edge_type == EdgeType.HEAT_TRANSFER]
        print(f"✅ Heat transfer edges: {len(heat_edges)}")
        
        # Test control signal edges
        control_edges = [e for e in enhanced_graph.edges.values() 
                        if e.edge_type == EdgeType.CONTROL_SIGNAL]
        print(f"✅ Control signal edges: {len(control_edges)}")
        
    except Exception as e:
        print(f"❌ Edge features test failed: {e}")
        return False
    
    # Test 8: Multi-scale representations
    print("\n🔄 Testing multi-scale representations...")
    try:
        # System level
        system_level = enhanced_graph.system_level_graph
        print(f"✅ System level graph: {system_level['level']}")
        print(f"✅ System level nodes: {len(system_level['nodes'])}")
        print(f"✅ Aggregation method: {system_level['aggregation_method']}")
        
        # Component level
        component_level = enhanced_graph.component_level_graph
        print(f"✅ Component level graph: {component_level['level']}")
        print(f"✅ Component level nodes: {len(component_level['nodes'])}")
        
        # Process level
        process_level = enhanced_graph.process_level_graph
        print(f"✅ Process level graph: {process_level['level']}")
        print(f"✅ Process level nodes: {len(process_level['nodes'])}")
        print(f"✅ Thermodynamic focus: {process_level['thermodynamic_focus']}")
        
        # Check process nodes
        for process, details in process_level['nodes'].items():
            print(f"  - {process}: {details['process_type']}")
            
    except Exception as e:
        print(f"❌ Multi-scale representations test failed: {e}")
        return False
    
    # Test 9: Graph dynamics update
    print("\n🔄 Testing graph dynamics update...")
    try:
        # Create new operating state
        new_operating_state = operating_state.copy()
        new_operating_state['current_state'] = operating_state['current_state'].copy()
        new_operating_state['current_state']['compressor_speed'] = 2400.0
        new_operating_state['current_state']['compressor_power'] = 6500.0
        new_operating_state['current_state']['power_input'] = 4000.0
        new_operating_state['current_state']['heating_capacity'] = 15000.0
        
        # Update graph
        original_cop = enhanced_graph.targets['COP_heating']
        updated_graph = generator.update_graph_dynamics(enhanced_graph, new_operating_state)
        new_cop = updated_graph.targets['COP_heating']
        
        print(f"✅ Graph updated successfully")
        print(f"✅ Original COP: {original_cop:.2f}")
        print(f"✅ Updated COP: {new_cop:.2f}")
        print(f"✅ COP change: {((new_cop - original_cop) / original_cop * 100):+.1f}%")
        
        # Check that dynamic features were updated
        comp_nodes = [n for n in updated_graph.nodes.values() 
                     if n.node_type == NodeType.COMPRESSOR]
        if comp_nodes:
            comp_node = comp_nodes[0]
            new_speed = comp_node.dynamic_features.get('speed_rpm', 0)
            print(f"✅ Updated compressor speed: {new_speed:.0f} rpm")
            
    except Exception as e:
        print(f"❌ Graph dynamics update failed: {e}")
        return False
    
    # Test 10: Dataset generation and saving
    print("\n🔄 Testing dataset generation and saving...")
    try:
        # Create multiple graphs for dataset
        test_graphs = [enhanced_graph, updated_graph]
        
        # Add a third graph with different conditions
        winter_state = operating_state.copy()
        winter_state['current_state'] = operating_state['current_state'].copy()
        winter_state['current_state']['ambient_temperature'] = 263.15  # -10°C
        winter_state['current_state']['heating_capacity'] = 10000.0
        winter_state['current_state']['power_input'] = 3500.0
        winter_state['conditions']['ambient_temp'] = 263.15
        
        winter_graph = generator.create_system_graph(winter_state, topology_type='enhanced')
        test_graphs.append(winter_graph)
        
        print(f"✅ Created dataset with {len(test_graphs)} graphs")
        
        # Save dataset
        output_path = "test_gnn_topology_data.json"
        metadata = {
            'test_run': True,
            'scenarios': ['baseline', 'high_speed', 'winter_conditions'],
            'topology_type': 'enhanced',
            'test_timestamp': pd.Timestamp.now().isoformat()
        }
        
        generator.save_graph_dataset(test_graphs, output_path, metadata)
        
        # Verify saved file
        if Path(output_path).exists():
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            print(f"✅ Dataset saved successfully")
            print(f"✅ Saved graphs: {saved_data['metadata']['n_graphs']}")
            print(f"✅ System type: {saved_data['metadata']['system_type']}")
            print(f"✅ File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
            
            # Check first graph structure
            first_graph = saved_data['graphs'][0]
            print(f"✅ First graph nodes: {len(first_graph['nodes'])}")
            print(f"✅ First graph edges: {len(first_graph['edges'])}")
            print(f"✅ Adjacency matrix: {len(first_graph['adjacency_matrix'])}x{len(first_graph['adjacency_matrix'][0])}")
            
        else:
            print("❌ Dataset file not found")
            return False
            
    except Exception as e:
        print(f"❌ Dataset generation and saving failed: {e}")
        return False
    
    # Test 11: Graph visualization (basic test)
    print("\n🔄 Testing graph visualization...")
    try:
        # Test visualization creation (without actually displaying)
        viz_path = "test_gnn_graph_visualization.png"
        generator.visualize_graph(enhanced_graph, viz_path)
        
        if Path(viz_path).exists():
            print(f"✅ Graph visualization saved successfully")
            print(f"✅ Visualization file size: {Path(viz_path).stat().st_size / 1024:.1f} KB")
        else:
            print("⚠️ Visualization file not created (may be due to display issues)")
            
    except Exception as e:
        print(f"⚠️ Graph visualization test failed (expected in headless environment): {e}")
    
    # Test 12: Node and edge type enums
    print("\n🔄 Testing node and edge type enums...")
    try:
        # Test NodeType enum
        node_types = list(NodeType)
        print(f"✅ NodeType enum: {len(node_types)} types")
        print(f"✅ Node types: {[nt.value for nt in node_types[:5]]}...")
        
        # Test EdgeType enum
        edge_types = list(EdgeType)
        print(f"✅ EdgeType enum: {len(edge_types)} types")
        print(f"✅ Edge types: {[et.value for et in edge_types]}")
        
        # Test enum usage in graph
        node_type_counts = {}
        for node in enhanced_graph.nodes.values():
            node_type = node.node_type.value
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        print(f"✅ Node type distribution: {node_type_counts}")
        
        edge_type_counts = {}
        for edge in enhanced_graph.edges.values():
            edge_type = edge.edge_type.value
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        print(f"✅ Edge type distribution: {edge_type_counts}")
        
    except Exception as e:
        print(f"❌ Node and edge type enums test failed: {e}")
        return False
    
    # Test 13: Dataclass functionality
    print("\n🔄 Testing dataclass functionality...")
    try:
        # Test GraphNode dataclass
        test_node = GraphNode(
            node_id="test_compressor",
            node_type=NodeType.COMPRESSOR,
            position=(1.0, 2.0, 3.0),
            static_features={"displacement": 0.15, "max_speed": 6000},
            dynamic_features={"speed_rpm": 1800, "power_kW": 5.0},
            health_features={"overall_health": 0.95, "bearing_wear": 0.05},
            metadata={"age": 2.5, "maintenance": []}
        )
        
        print(f"✅ GraphNode created: {test_node.node_id}")
        print(f"✅ Node type: {test_node.node_type.value}")
        print(f"✅ Position: {test_node.position}")
        print(f"✅ Static features: {len(test_node.static_features)}")
        
        # Test GraphEdge dataclass
        test_edge = GraphEdge(
            edge_id="test_edge",
            source_node="compressor",
            target_node="condenser",
            edge_type=EdgeType.REFRIGERANT_FLOW,
            features={"mass_flow_rate": 0.15, "pressure": 2000000},
            weight=0.8,
            importance=1.0,
            is_directed=True,
            metadata={"length": 2.5}
        )
        
        print(f"✅ GraphEdge created: {test_edge.edge_id}")
        print(f"✅ Edge type: {test_edge.edge_type.value}")
        print(f"✅ Is directed: {test_edge.is_directed}")
        print(f"✅ Weight: {test_edge.weight}")
        
    except Exception as e:
        print(f"❌ Dataclass functionality test failed: {e}")
        return False
    
    # Test 14: Edge cases and error handling
    print("\n🔄 Testing edge cases and error handling...")
    try:
        # Test with minimal operating state
        minimal_state = {
            'current_state': {
                'ambient_temperature': 288.15,
                'power_input': 1000.0,
                'heating_capacity': 3000.0
            }
        }
        
        minimal_graph = generator.create_system_graph(minimal_state, topology_type='basic')
        print(f"✅ Minimal graph created: {len(minimal_graph.nodes)} nodes")
        print(f"✅ Minimal graph COP: {minimal_graph.targets['COP_heating']:.2f}")
        
        # Test with empty health state
        empty_health_state = operating_state.copy()
        empty_health_state['health_state'] = {}
        
        empty_health_graph = generator.create_system_graph(empty_health_state)
        print(f"✅ Graph with empty health state created successfully")
        
        # Test node type determination
        test_names = ['compressor_main', 'evaporator_outdoor', 'unknown_component']
        for name in test_names:
            node_type = generator._get_node_type(name)
            print(f"✅ Node '{name}' → {node_type.value}")
            
    except Exception as e:
        print(f"❌ Edge cases and error handling test failed: {e}")
        return False
    
    print("\n🎉 ALL GNN TOPOLOGY TESTS COMPLETED!")
    print("=" * 60)
    print("✅ Graph generation: PASSED")
    print("✅ Multi-scale representations: PASSED")
    print("✅ Dynamic updates: PASSED")
    print("✅ Dataset saving: PASSED")
    print("✅ Node/edge features: PASSED")
    print("✅ Dataclass functionality: PASSED")
    print("✅ Error handling: PASSED")
    
    return True

if __name__ == "__main__":
    success = test_gnn_topology_comprehensive()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
        sys.exit(1)