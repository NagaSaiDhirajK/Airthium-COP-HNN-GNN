#!/usr/bin/env python3
"""
Graph Neural Network Topology for AIRTHIUM Heat Pump Systems
============================================================

This module creates graph representations of heat pump systems for Graph Neural Network
training, capturing component relationships, heat/mass transfer networks, and system
topology for accurate COP prediction.

Key Features:
- Multi-scale graph representations (system, component, process levels)
- Dynamic edge features for refrigerant flow and heat transfer
- Component interaction matrices
- Hierarchical graph structures
- Real-time graph updates for operational changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import warnings

from thermodynamic_properties import ThermodynamicState
from component_models import (
    CompressorType, HeatExchangerType, ExpansionType,
    CompressorPerformance, HeatExchangerPerformance, ExpansionPerformance
)

class NodeType(Enum):
    """Types of nodes in the heat pump graph."""
    COMPRESSOR = "compressor"
    EVAPORATOR = "evaporator"
    CONDENSER = "condenser"
    EXPANSION_VALVE = "expansion_valve"
    ACCUMULATOR = "accumulator"
    FILTER_DRIER = "filter_drier"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    AMBIENT = "ambient"
    LOAD = "load"

class EdgeType(Enum):
    """Types of edges in the heat pump graph."""
    REFRIGERANT_FLOW = "refrigerant_flow"
    HEAT_TRANSFER = "heat_transfer"
    CONTROL_SIGNAL = "control_signal"
    MECHANICAL_CONNECTION = "mechanical_connection"
    ELECTRICAL_CONNECTION = "electrical_connection"
    INFORMATION_FLOW = "information_flow"

@dataclass
class GraphNode:
    """
    Node representation in the heat pump graph.
    """
    node_id: str
    node_type: NodeType
    position: Tuple[float, float, float]  # 3D coordinates
    
    # Static features (component specifications)
    static_features: Dict[str, float]
    
    # Dynamic features (operational state)
    dynamic_features: Dict[str, float]
    
    # Component health and degradation
    health_features: Dict[str, float]
    
    # Metadata
    metadata: Dict[str, Any]

@dataclass
class GraphEdge:
    """
    Edge representation in the heat pump graph.
    """
    edge_id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    
    # Edge features (flow rates, heat transfer, etc.)
    features: Dict[str, float]
    
    # Edge weights and importance
    weight: float
    importance: float
    
    # Directional information
    is_directed: bool
    
    # Metadata
    metadata: Dict[str, Any]

@dataclass
class HeatPumpGraph:
    """
    Complete graph representation of a heat pump system.
    """
    graph_id: str
    timestamp: float
    
    # Graph structure
    nodes: Dict[str, GraphNode]
    edges: Dict[str, GraphEdge]
    adjacency_matrix: np.ndarray
    
    # Multi-scale representations
    system_level_graph: Dict[str, Any]
    component_level_graph: Dict[str, Any]
    process_level_graph: Dict[str, Any]
    
    # Performance targets
    targets: Dict[str, float]
    
    # Operating conditions
    operating_conditions: Dict[str, float]
    
    # Validation metrics
    graph_metrics: Dict[str, float]

class HeatPumpGraphGenerator:
    """
    Generate graph representations for AIRTHIUM heat pump systems.
    """
    
    def __init__(self, system_config: Dict[str, Any] = None):
        """
        Initialize graph generator.
        
        Args:
            system_config: Configuration of the heat pump system
        """
        self.system_config = system_config or self._get_default_config()
        
        # Component specifications
        self.component_specs = self._load_component_specifications()
        
        # Graph templates
        self.graph_templates = self._create_graph_templates()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default AIRTHIUM heat pump configuration."""
        return {
            'system_type': 'air_to_water_heat_pump',
            'capacity_range': [5000, 15000],  # W
            'refrigerant': 'R410A',
            'compressor_type': 'scroll_variable_speed',
            'evaporator_type': 'finned_tube_air',
            'condenser_type': 'plate_water',
            'expansion_type': 'electronic_valve',
            'control_type': 'microprocessor',
            'sensors': [
                'suction_pressure', 'discharge_pressure',
                'suction_temperature', 'discharge_temperature',
                'evaporator_air_temp', 'condenser_water_temp',
                'ambient_temperature', 'compressor_current'
            ]
        }
    
    def _load_component_specifications(self) -> Dict[str, Dict]:
        """Load component specifications for graph features."""
        return {
            'compressor': {
                'displacement': 0.15,  # m³/s
                'max_speed': 6000,     # rpm
                'efficiency_range': [0.7, 0.9],
                'power_range': [1000, 8000],  # W
                'vibration_limit': 10.0,      # mm/s
                'oil_temp_limit': 393.15      # K
            },
            'evaporator': {
                'area': 40.0,          # m²
                'volume': 0.02,        # m³
                'air_flow_range': [2000, 8000],  # m³/h
                'effectiveness_range': [0.6, 0.85],
                'fouling_limit': 0.002,       # m²K/W
                'frost_thickness_limit': 0.005 # m
            },
            'condenser': {
                'area': 60.0,          # m²
                'volume': 0.015,       # m³
                'water_flow_range': [0.5, 2.0],  # m³/h
                'effectiveness_range': [0.7, 0.9],
                'fouling_limit': 0.001,       # m²K/W
                'approach_temp_min': 3.0      # K
            },
            'expansion_valve': {
                'cv_range': [0.1, 2.0],       # Flow coefficient
                'opening_range': [0, 100],    # %
                'response_time': 5.0,         # s
                'superheat_control_range': [3, 15]  # K
            }
        }
    
    def _create_graph_templates(self) -> Dict[str, Dict]:
        """Create graph templates for different system configurations."""
        
        # Basic heat pump topology
        basic_topology = {
            'nodes': [
                'compressor', 'evaporator', 'condenser', 'expansion_valve',
                'suction_line', 'discharge_line', 'liquid_line',
                'controller', 'ambient', 'load'
            ],
            'edges': [
                ('evaporator', 'compressor', 'refrigerant_flow'),
                ('compressor', 'condenser', 'refrigerant_flow'),
                ('condenser', 'expansion_valve', 'refrigerant_flow'),
                ('expansion_valve', 'evaporator', 'refrigerant_flow'),
                ('ambient', 'evaporator', 'heat_transfer'),
                ('condenser', 'load', 'heat_transfer'),
                ('controller', 'compressor', 'control_signal'),
                ('controller', 'expansion_valve', 'control_signal')
            ]
        }
        
        # Enhanced topology with additional components
        enhanced_topology = {
            'nodes': basic_topology['nodes'] + [
                'accumulator', 'filter_drier', 'oil_separator',
                'pressure_sensors', 'temperature_sensors',
                'flow_sensors', 'defrost_valve'
            ],
            'edges': basic_topology['edges'] + [
                ('compressor', 'oil_separator', 'refrigerant_flow'),
                ('oil_separator', 'condenser', 'refrigerant_flow'),
                ('liquid_line', 'filter_drier', 'refrigerant_flow'),
                ('filter_drier', 'expansion_valve', 'refrigerant_flow'),
                ('evaporator', 'accumulator', 'refrigerant_flow'),
                ('accumulator', 'compressor', 'refrigerant_flow'),
                ('pressure_sensors', 'controller', 'information_flow'),
                ('temperature_sensors', 'controller', 'information_flow')
            ]
        }
        
        return {
            'basic': basic_topology,
            'enhanced': enhanced_topology
        }
    
    def create_system_graph(self, operating_state: Dict[str, Any],
                          topology_type: str = 'enhanced') -> HeatPumpGraph:
        """
        Create complete system-level graph representation.
        
        Args:
            operating_state: Current operating state of the system
            topology_type: Type of topology ('basic' or 'enhanced')
            
        Returns:
            Complete heat pump graph
        """
        
        # Get topology template
        template = self.graph_templates[topology_type]
        
        # Create nodes
        nodes = {}
        for node_name in template['nodes']:
            node = self._create_node(node_name, operating_state)
            nodes[node.node_id] = node
        
        # Create edges
        edges = {}
        for source, target, edge_type in template['edges']:
            if source in [n.node_id for n in nodes.values()] and \
               target in [n.node_id for n in nodes.values()]:
                edge = self._create_edge(source, target, edge_type, operating_state)
                edges[edge.edge_id] = edge
        
        # Create adjacency matrix
        node_list = list(nodes.keys())
        n_nodes = len(node_list)
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for edge in edges.values():
            source_idx = node_list.index(edge.source_node)
            target_idx = node_list.index(edge.target_node)
            adjacency_matrix[source_idx, target_idx] = edge.weight
            if not edge.is_directed:
                adjacency_matrix[target_idx, source_idx] = edge.weight
        
        # Create multi-scale representations
        system_level = self._create_system_level_graph(nodes, edges)
        component_level = self._create_component_level_graph(nodes, edges)
        process_level = self._create_process_level_graph(nodes, edges)
        
        # Calculate targets
        targets = self._calculate_targets(operating_state)
        
        # Calculate graph metrics
        graph_metrics = self._calculate_graph_metrics(nodes, edges, adjacency_matrix)
        
        return HeatPumpGraph(
            graph_id=f"hp_graph_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=pd.Timestamp.now().timestamp(),
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency_matrix,
            system_level_graph=system_level,
            component_level_graph=component_level,
            process_level_graph=process_level,
            targets=targets,
            operating_conditions=operating_state.get('conditions', {}),
            graph_metrics=graph_metrics
        )
    
    def _create_node(self, node_name: str, operating_state: Dict[str, Any]) -> GraphNode:
        """Create a graph node with features."""
        
        # Determine node type
        node_type = self._get_node_type(node_name)
        
        # Get position (simplified 3D layout)
        position = self._get_node_position(node_name)
        
        # Create static features
        static_features = self._get_static_features(node_name, node_type)
        
        # Create dynamic features
        dynamic_features = self._get_dynamic_features(node_name, node_type, operating_state)
        
        # Create health features
        health_features = self._get_health_features(node_name, node_type, operating_state)
        
        # Metadata
        metadata = {
            'component_age': operating_state.get('component_ages', {}).get(node_name, 0),
            'maintenance_history': operating_state.get('maintenance', {}).get(node_name, []),
            'fault_history': operating_state.get('faults', {}).get(node_name, [])
        }
        
        return GraphNode(
            node_id=node_name,
            node_type=node_type,
            position=position,
            static_features=static_features,
            dynamic_features=dynamic_features,
            health_features=health_features,
            metadata=metadata
        )
    
    def _create_edge(self, source: str, target: str, edge_type_str: str,
                    operating_state: Dict[str, Any]) -> GraphEdge:
        """Create a graph edge with features."""
        
        edge_type = EdgeType(edge_type_str)
        edge_id = f"{source}_to_{target}_{edge_type_str}"
        
        # Create edge features based on type
        features = self._get_edge_features(source, target, edge_type, operating_state)
        
        # Calculate edge weight and importance
        weight = self._calculate_edge_weight(source, target, edge_type, features)
        importance = self._calculate_edge_importance(source, target, edge_type, features)
        
        # Determine if edge is directed
        is_directed = edge_type in [EdgeType.REFRIGERANT_FLOW, EdgeType.CONTROL_SIGNAL, 
                                   EdgeType.INFORMATION_FLOW]
        
        metadata = {
            'connection_type': edge_type_str,
            'physical_length': self._estimate_connection_length(source, target),
            'installation_date': operating_state.get('installation_dates', {}).get(edge_id, None)
        }
        
        return GraphEdge(
            edge_id=edge_id,
            source_node=source,
            target_node=target,
            edge_type=edge_type,
            features=features,
            weight=weight,
            importance=importance,
            is_directed=is_directed,
            metadata=metadata
        )
    
    def _get_node_type(self, node_name: str) -> NodeType:
        """Determine node type from node name."""
        type_mapping = {
            'compressor': NodeType.COMPRESSOR,
            'evaporator': NodeType.EVAPORATOR,
            'condenser': NodeType.CONDENSER,
            'expansion_valve': NodeType.EXPANSION_VALVE,
            'accumulator': NodeType.ACCUMULATOR,
            'filter_drier': NodeType.FILTER_DRIER,
            'controller': NodeType.CONTROLLER,
            'ambient': NodeType.AMBIENT,
            'load': NodeType.LOAD
        }
        
        for key, node_type in type_mapping.items():
            if key in node_name.lower():
                return node_type
        
        return NodeType.SENSOR  # Default
    
    def _get_node_position(self, node_name: str) -> Tuple[float, float, float]:
        """Get 3D position for node (simplified layout)."""
        positions = {
            'compressor': (0.0, 0.0, 0.5),
            'evaporator': (-2.0, 0.0, 0.0),
            'condenser': (2.0, 0.0, 0.0),
            'expansion_valve': (1.0, 0.0, -0.5),
            'accumulator': (-1.0, 0.0, -0.5),
            'controller': (0.0, 2.0, 1.0),
            'ambient': (-3.0, 0.0, 0.0),
            'load': (3.0, 0.0, 0.0)
        }
        
        for key, pos in positions.items():
            if key in node_name.lower():
                return pos
        
        return (0.0, 0.0, 0.0)  # Default
    
    def _get_static_features(self, node_name: str, node_type: NodeType) -> Dict[str, float]:
        """Get static features for a node."""
        
        if node_type == NodeType.COMPRESSOR:
            specs = self.component_specs['compressor']
            return {
                'displacement': specs['displacement'],
                'max_speed': specs['max_speed'],
                'efficiency_nominal': np.mean(specs['efficiency_range']),
                'power_nominal': np.mean(specs['power_range']),
                'type_scroll': 1.0,  # One-hot encoding
                'type_reciprocating': 0.0,
                'variable_speed': 1.0
            }
        
        elif node_type == NodeType.EVAPORATOR:
            specs = self.component_specs['evaporator']
            return {
                'heat_transfer_area': specs['area'],
                'volume': specs['volume'],
                'air_flow_nominal': np.mean(specs['air_flow_range']),
                'effectiveness_nominal': np.mean(specs['effectiveness_range']),
                'type_finned_tube': 1.0,
                'type_plate': 0.0
            }
        
        elif node_type == NodeType.CONDENSER:
            specs = self.component_specs['condenser']
            return {
                'heat_transfer_area': specs['area'],
                'volume': specs['volume'],
                'water_flow_nominal': np.mean(specs['water_flow_range']),
                'effectiveness_nominal': np.mean(specs['effectiveness_range']),
                'type_plate': 1.0,
                'type_shell_tube': 0.0
            }
        
        elif node_type == NodeType.EXPANSION_VALVE:
            specs = self.component_specs['expansion_valve']
            return {
                'cv_nominal': np.mean(specs['cv_range']),
                'response_time': specs['response_time'],
                'superheat_setpoint': np.mean(specs['superheat_control_range']),
                'type_electronic': 1.0,
                'type_thermostatic': 0.0
            }
        
        else:
            return {'nominal_capacity': 1.0, 'efficiency': 1.0}
    
    def _get_dynamic_features(self, node_name: str, node_type: NodeType,
                            operating_state: Dict[str, Any]) -> Dict[str, float]:
        """Get dynamic features for a node."""
        
        # Get current operating data
        current_data = operating_state.get('current_state', {})
        
        if node_type == NodeType.COMPRESSOR:
            return {
                'speed_rpm': current_data.get('compressor_speed', 1800.0),
                'power_kW': current_data.get('compressor_power', 5.0),
                'efficiency': current_data.get('compressor_efficiency', 0.8),
                'suction_pressure': current_data.get('suction_pressure', 500000.0),
                'discharge_pressure': current_data.get('discharge_pressure', 2000000.0),
                'suction_temperature': current_data.get('suction_temperature', 278.15),
                'discharge_temperature': current_data.get('discharge_temperature', 333.15),
                'oil_temperature': current_data.get('oil_temperature', 323.15),
                'vibration_level': current_data.get('vibration', 5.0),
                'current_draw': current_data.get('compressor_current', 20.0)
            }
        
        elif node_type == NodeType.EVAPORATOR:
            return {
                'heat_transfer_rate': current_data.get('evaporator_heat_rate', 8000.0),
                'effectiveness': current_data.get('evaporator_effectiveness', 0.75),
                'air_flow_rate': current_data.get('air_flow_rate', 5000.0),
                'air_temp_in': current_data.get('ambient_temperature', 278.15),
                'air_temp_out': current_data.get('evap_air_out_temp', 273.15),
                'refrigerant_temp_in': current_data.get('evap_ref_in_temp', 268.15),
                'refrigerant_temp_out': current_data.get('evap_ref_out_temp', 278.15),
                'frost_thickness': current_data.get('frost_thickness', 0.001),
                'pressure_drop': current_data.get('evap_pressure_drop', 5000.0)
            }
        
        elif node_type == NodeType.CONDENSER:
            return {
                'heat_transfer_rate': current_data.get('condenser_heat_rate', 12000.0),
                'effectiveness': current_data.get('condenser_effectiveness', 0.8),
                'water_flow_rate': current_data.get('water_flow_rate', 1.0),
                'water_temp_in': current_data.get('water_temp_in', 313.15),
                'water_temp_out': current_data.get('water_temp_out', 323.15),
                'refrigerant_temp_in': current_data.get('cond_ref_in_temp', 333.15),
                'refrigerant_temp_out': current_data.get('cond_ref_out_temp', 318.15),
                'approach_temperature': current_data.get('approach_temp', 5.0),
                'pressure_drop': current_data.get('cond_pressure_drop', 3000.0)
            }
        
        elif node_type == NodeType.EXPANSION_VALVE:
            return {
                'opening_percent': current_data.get('valve_opening', 50.0),
                'superheat_actual': current_data.get('superheat', 8.0),
                'superheat_setpoint': current_data.get('superheat_setpoint', 10.0),
                'mass_flow_rate': current_data.get('mass_flow_rate', 0.1),
                'pressure_drop': current_data.get('valve_pressure_drop', 1500000.0),
                'control_signal': current_data.get('valve_control_signal', 50.0)
            }
        
        else:
            return {
                'temperature': current_data.get('ambient_temperature', 288.15),
                'pressure': current_data.get('ambient_pressure', 101325.0),
                'load_fraction': current_data.get('load_fraction', 0.8)
            }
    
    def _get_health_features(self, node_name: str, node_type: NodeType,
                           operating_state: Dict[str, Any]) -> Dict[str, float]:
        """Get component health and degradation features."""
        
        health_data = operating_state.get('health_state', {})
        
        base_health = {
            'overall_health': health_data.get(f'{node_name}_health', 1.0),
            'degradation_rate': health_data.get(f'{node_name}_degradation', 0.0),
            'remaining_life': health_data.get(f'{node_name}_life', 1.0),
            'maintenance_due': health_data.get(f'{node_name}_maintenance', 0.0)
        }
        
        if node_type == NodeType.COMPRESSOR:
            base_health.update({
                'bearing_wear': health_data.get('compressor_bearing_wear', 0.0),
                'valve_leakage': health_data.get('compressor_valve_leakage', 0.0),
                'oil_quality': health_data.get('oil_quality', 1.0),
                'efficiency_degradation': health_data.get('compressor_eff_deg', 0.0)
            })
        
        elif node_type in [NodeType.EVAPORATOR, NodeType.CONDENSER]:
            base_health.update({
                'fouling_factor': health_data.get(f'{node_name}_fouling', 0.0),
                'corrosion_level': health_data.get(f'{node_name}_corrosion', 0.0),
                'effectiveness_degradation': health_data.get(f'{node_name}_eff_deg', 0.0)
            })
        
        return base_health
    
    def _get_edge_features(self, source: str, target: str, edge_type: EdgeType,
                          operating_state: Dict[str, Any]) -> Dict[str, float]:
        """Get features for an edge."""
        
        current_data = operating_state.get('current_state', {})
        
        if edge_type == EdgeType.REFRIGERANT_FLOW:
            return {
                'mass_flow_rate': current_data.get('mass_flow_rate', 0.1),
                'pressure': current_data.get(f'{source}_pressure', 1000000.0),
                'temperature': current_data.get(f'{source}_temperature', 298.15),
                'enthalpy': current_data.get(f'{source}_enthalpy', 250000.0),
                'entropy': current_data.get(f'{source}_entropy', 1000.0),
                'quality': current_data.get(f'{source}_quality', 1.0),
                'velocity': current_data.get(f'{source}_velocity', 5.0),
                'pressure_drop': current_data.get(f'{source}_{target}_dp', 1000.0)
            }
        
        elif edge_type == EdgeType.HEAT_TRANSFER:
            return {
                'heat_rate': current_data.get(f'{source}_{target}_heat_rate', 5000.0),
                'temperature_difference': current_data.get(f'{source}_{target}_dt', 10.0),
                'heat_transfer_coefficient': current_data.get(f'{source}_{target}_htc', 1000.0),
                'effectiveness': current_data.get(f'{source}_{target}_eff', 0.8)
            }
        
        elif edge_type == EdgeType.CONTROL_SIGNAL:
            return {
                'signal_value': current_data.get(f'{source}_{target}_signal', 50.0),
                'setpoint': current_data.get(f'{target}_setpoint', 50.0),
                'error': current_data.get(f'{target}_error', 0.0),
                'control_action': current_data.get(f'{target}_action', 0.0),
                'response_time': current_data.get(f'{target}_response_time', 5.0)
            }
        
        else:
            return {
                'connection_strength': 1.0,
                'signal_quality': 1.0,
                'bandwidth': 1000.0
            }
    
    def _calculate_edge_weight(self, source: str, target: str, edge_type: EdgeType,
                             features: Dict[str, float]) -> float:
        """Calculate edge weight based on features."""
        
        if edge_type == EdgeType.REFRIGERANT_FLOW:
            # Weight based on mass flow rate and pressure
            return features.get('mass_flow_rate', 0.1) * features.get('pressure', 1000000) / 1e8
        
        elif edge_type == EdgeType.HEAT_TRANSFER:
            # Weight based on heat transfer rate
            return features.get('heat_rate', 5000) / 10000
        
        elif edge_type == EdgeType.CONTROL_SIGNAL:
            # Weight based on signal strength and importance
            return min(features.get('signal_value', 50) / 100, 1.0)
        
        else:
            return 1.0
    
    def _calculate_edge_importance(self, source: str, target: str, edge_type: EdgeType,
                                 features: Dict[str, float]) -> float:
        """Calculate edge importance for system performance."""
        
        # Critical connections have higher importance
        critical_connections = [
            ('compressor', 'condenser'),
            ('evaporator', 'compressor'),
            ('controller', 'compressor'),
            ('controller', 'expansion_valve')
        ]
        
        base_importance = 0.5
        if any(c1 in source and c2 in target for c1, c2 in critical_connections):
            base_importance = 1.0
        
        # Adjust based on edge type
        type_multipliers = {
            EdgeType.REFRIGERANT_FLOW: 1.0,
            EdgeType.HEAT_TRANSFER: 0.8,
            EdgeType.CONTROL_SIGNAL: 0.9,
            EdgeType.MECHANICAL_CONNECTION: 0.6,
            EdgeType.ELECTRICAL_CONNECTION: 0.7,
            EdgeType.INFORMATION_FLOW: 0.5
        }
        
        return base_importance * type_multipliers.get(edge_type, 0.5)
    
    def _estimate_connection_length(self, source: str, target: str) -> float:
        """Estimate physical connection length between components."""
        
        # Get positions
        pos1 = self._get_node_position(source)
        pos2 = self._get_node_position(target)
        
        # Calculate Euclidean distance
        distance = np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
        return distance
    
    def _create_system_level_graph(self, nodes: Dict[str, GraphNode],
                                 edges: Dict[str, GraphEdge]) -> Dict[str, Any]:
        """Create system-level graph representation."""
        
        # Aggregate nodes into major components
        major_components = ['compressor', 'evaporator', 'condenser', 'expansion_valve']
        
        system_nodes = {}
        for comp in major_components:
            comp_nodes = [n for n in nodes.values() if comp in n.node_id.lower()]
            if comp_nodes:
                # Aggregate features
                agg_static = {}
                agg_dynamic = {}
                agg_health = {}
                
                for node in comp_nodes:
                    for key, value in node.static_features.items():
                        agg_static[key] = agg_static.get(key, 0) + value
                    for key, value in node.dynamic_features.items():
                        agg_dynamic[key] = agg_dynamic.get(key, 0) + value / len(comp_nodes)
                    for key, value in node.health_features.items():
                        agg_health[key] = min(agg_health.get(key, 1), value)
                
                system_nodes[comp] = {
                    'static_features': agg_static,
                    'dynamic_features': agg_dynamic,
                    'health_features': agg_health
                }
        
        return {
            'nodes': system_nodes,
            'level': 'system',
            'aggregation_method': 'component_grouping'
        }
    
    def _create_component_level_graph(self, nodes: Dict[str, GraphNode],
                                    edges: Dict[str, GraphEdge]) -> Dict[str, Any]:
        """Create component-level graph representation."""
        
        # Keep all individual components but group sub-components
        component_nodes = {}
        for node_id, node in nodes.items():
            if node.node_type in [NodeType.COMPRESSOR, NodeType.EVAPORATOR,
                                 NodeType.CONDENSER, NodeType.EXPANSION_VALVE]:
                component_nodes[node_id] = {
                    'static_features': node.static_features,
                    'dynamic_features': node.dynamic_features,
                    'health_features': node.health_features,
                    'position': node.position
                }
        
        return {
            'nodes': component_nodes,
            'level': 'component',
            'detail_level': 'individual_components'
        }
    
    def _create_process_level_graph(self, nodes: Dict[str, GraphNode],
                                  edges: Dict[str, GraphEdge]) -> Dict[str, Any]:
        """Create process-level graph representation."""
        
        # Create nodes for thermodynamic processes
        process_nodes = {
            'compression': {
                'components': ['compressor'],
                'process_type': 'work_input',
                'energy_change': 'enthalpy_increase'
            },
            'condensation': {
                'components': ['condenser'],
                'process_type': 'heat_rejection',
                'energy_change': 'enthalpy_decrease'
            },
            'expansion': {
                'components': ['expansion_valve'],
                'process_type': 'pressure_reduction',
                'energy_change': 'enthalpy_constant'
            },
            'evaporation': {
                'components': ['evaporator'],
                'process_type': 'heat_absorption',
                'energy_change': 'enthalpy_increase'
            }
        }
        
        return {
            'nodes': process_nodes,
            'level': 'process',
            'thermodynamic_focus': True
        }
    
    def _calculate_targets(self, operating_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate target values for the graph."""
        
        current_data = operating_state.get('current_state', {})
        
        # Calculate COPs
        heating_capacity = current_data.get('heating_capacity', 10000.0)  # W
        cooling_capacity = current_data.get('cooling_capacity', 8000.0)   # W
        power_input = current_data.get('power_input', 3000.0)             # W
        
        COP_heating = heating_capacity / power_input if power_input > 0 else 1.0
        COP_cooling = cooling_capacity / power_input if power_input > 0 else 1.0
        
        # Calculate efficiency
        carnot_cop = current_data.get('carnot_cop', 5.0)
        efficiency = COP_heating / carnot_cop if carnot_cop > 0 else 0.8
        
        return {
            'COP_heating': COP_heating,
            'COP_cooling': COP_cooling,
            'efficiency': efficiency,
            'heating_capacity': heating_capacity,
            'cooling_capacity': cooling_capacity,
            'power_input': power_input
        }
    
    def _calculate_graph_metrics(self, nodes: Dict[str, GraphNode],
                               edges: Dict[str, GraphEdge],
                               adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate graph topology metrics."""
        
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        # Basic metrics
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        
        # Centrality measures (simplified)
        degree_centrality = np.sum(adjacency_matrix > 0, axis=1)
        max_degree = np.max(degree_centrality) if len(degree_centrality) > 0 else 0
        avg_degree = np.mean(degree_centrality) if len(degree_centrality) > 0 else 0
        
        # Connectivity
        connected_components = 1  # Simplified - assume connected
        
        return {
            'n_nodes': float(n_nodes),
            'n_edges': float(n_edges),
            'density': density,
            'max_degree': float(max_degree),
            'avg_degree': float(avg_degree),
            'connected_components': float(connected_components),
            'graph_complexity': float(n_nodes * n_edges)
        }
    
    def update_graph_dynamics(self, graph: HeatPumpGraph,
                            new_operating_state: Dict[str, Any]) -> HeatPumpGraph:
        """Update graph with new operating state."""
        
        # Update dynamic features for all nodes
        for node_id, node in graph.nodes.items():
            new_dynamic_features = self._get_dynamic_features(
                node_id, node.node_type, new_operating_state
            )
            node.dynamic_features = new_dynamic_features
            
            # Update health features
            new_health_features = self._get_health_features(
                node_id, node.node_type, new_operating_state
            )
            node.health_features = new_health_features
        
        # Update edge features
        for edge_id, edge in graph.edges.items():
            new_edge_features = self._get_edge_features(
                edge.source_node, edge.target_node, edge.edge_type, new_operating_state
            )
            edge.features = new_edge_features
            
            # Recalculate weight and importance
            edge.weight = self._calculate_edge_weight(
                edge.source_node, edge.target_node, edge.edge_type, new_edge_features
            )
            edge.importance = self._calculate_edge_importance(
                edge.source_node, edge.target_node, edge.edge_type, new_edge_features
            )
        
        # Update adjacency matrix
        node_list = list(graph.nodes.keys())
        n_nodes = len(node_list)
        new_adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for edge in graph.edges.values():
            source_idx = node_list.index(edge.source_node)
            target_idx = node_list.index(edge.target_node)
            new_adjacency_matrix[source_idx, target_idx] = edge.weight
            if not edge.is_directed:
                new_adjacency_matrix[target_idx, source_idx] = edge.weight
        
        graph.adjacency_matrix = new_adjacency_matrix
        
        # Update targets
        graph.targets = self._calculate_targets(new_operating_state)
        
        # Update operating conditions
        graph.operating_conditions = new_operating_state.get('conditions', {})
        
        # Update timestamp
        graph.timestamp = pd.Timestamp.now().timestamp()
        
        return graph
    
    def save_graph_dataset(self, graphs: List[HeatPumpGraph],
                          output_path: str, metadata: Dict[str, Any] = None) -> None:
        """Save graph dataset for GNN training."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'n_graphs': len(graphs),
                'system_type': self.system_config['system_type'],
                'refrigerant': self.system_config['refrigerant'],
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                **(metadata or {})
            },
            'graphs': []
        }
        
        for graph in graphs:
            # Convert graph to serializable format
            graph_data = {
                'graph_id': graph.graph_id,
                'timestamp': graph.timestamp,
                
                # Nodes
                'nodes': {
                    node_id: {
                        'node_type': node.node_type.value,
                        'position': list(node.position),
                        'static_features': node.static_features,
                        'dynamic_features': node.dynamic_features,
                        'health_features': node.health_features,
                        'metadata': node.metadata
                    }
                    for node_id, node in graph.nodes.items()
                },
                
                # Edges
                'edges': {
                    edge_id: {
                        'source_node': edge.source_node,
                        'target_node': edge.target_node,
                        'edge_type': edge.edge_type.value,
                        'features': edge.features,
                        'weight': edge.weight,
                        'importance': edge.importance,
                        'is_directed': edge.is_directed,
                        'metadata': edge.metadata
                    }
                    for edge_id, edge in graph.edges.items()
                },
                
                # Adjacency matrix
                'adjacency_matrix': graph.adjacency_matrix.tolist(),
                
                # Multi-scale representations
                'system_level_graph': graph.system_level_graph,
                'component_level_graph': graph.component_level_graph,
                'process_level_graph': graph.process_level_graph,
                
                # Targets
                'targets': graph.targets,
                
                # Operating conditions
                'operating_conditions': graph.operating_conditions,
                
                # Graph metrics
                'graph_metrics': graph.graph_metrics
            }
            
            dataset['graphs'].append(graph_data)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"💾 Saved GNN dataset with {len(graphs)} graphs to {output_path}")
    
    def visualize_graph(self, graph: HeatPumpGraph, output_path: str = None) -> None:
        """Visualize the heat pump graph."""
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            G.add_node(node_id,
                      node_type=node.node_type.value,
                      pos=node.position[:2])  # Use 2D position
        
        # Add edges
        for edge in graph.edges.values():
            G.add_edge(edge.source_node, edge.target_node,
                      edge_type=edge.edge_type.value,
                      weight=edge.weight)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G)
        
        # Draw nodes with different colors for different types
        node_colors = {
            'compressor': 'red',
            'evaporator': 'blue',
            'condenser': 'orange',
            'expansion_valve': 'green',
            'controller': 'purple',
            'sensor': 'gray',
            'ambient': 'lightblue',
            'load': 'yellow'
        }
        
        for node_type, color in node_colors.items():
            nodes_of_type = [n for n, d in G.nodes(data=True)
                           if node_type in d.get('node_type', '')]
            if nodes_of_type:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                     node_color=color, node_size=1000, alpha=0.8)
        
        # Draw edges with different styles for different types
        edge_styles = {
            'refrigerant_flow': '-',
            'heat_transfer': '--',
            'control_signal': ':',
            'information_flow': '-.'
        }
        
        for edge_type, style in edge_styles.items():
            edges_of_type = [(u, v) for u, v, d in G.edges(data=True)
                           if edge_type in d.get('edge_type', '')]
            if edges_of_type:
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     style=style, alpha=0.6, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Heat Pump System Graph - {graph.graph_id}")
        plt.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=10, label=node_type)
                         for node_type, color in node_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved graph visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # Test GNN topology generation
    print("🔄 Testing GNN Topology Generator")
    
    # Initialize generator
    generator = HeatPumpGraphGenerator()
    
    # Create test operating state
    operating_state = {
        'current_state': {
            'compressor_speed': 1800.0,
            'compressor_power': 5000.0,
            'compressor_efficiency': 0.8,
            'suction_pressure': 500000.0,
            'discharge_pressure': 2000000.0,
            'evaporator_heat_rate': 8000.0,
            'condenser_heat_rate': 12000.0,
            'valve_opening': 45.0,
            'ambient_temperature': 278.15,
            'water_temp_in': 313.15,
            'heating_capacity': 12000.0,
            'power_input': 3000.0
        },
        'conditions': {
            'ambient_temp': 278.15,
            'load_fraction': 0.8,
            'operating_mode': 'heating'
        }
    }
    
    # Generate system graph
    print("\n🏗️ Generating system graph...")
    system_graph = generator.create_system_graph(operating_state, topology_type='enhanced')
    
    print(f"✅ Generated graph with {len(system_graph.nodes)} nodes and {len(system_graph.edges)} edges")
    print(f"🎯 Target COP Heating: {system_graph.targets['COP_heating']:.2f}")
    print(f"📊 Graph density: {system_graph.graph_metrics['density']:.3f}")
    
    # Test graph update
    print("\n🔄 Testing graph update...")
    new_operating_state = operating_state.copy()
    new_operating_state['current_state']['compressor_speed'] = 2400.0
    new_operating_state['current_state']['power_input'] = 4000.0
    
    updated_graph = generator.update_graph_dynamics(system_graph, new_operating_state)
    print(f"🎯 Updated COP Heating: {updated_graph.targets['COP_heating']:.2f}")
    
    # Save test dataset
    test_graphs = [system_graph, updated_graph]
    generator.save_graph_dataset(
        test_graphs,
        "test_gnn_data.json",
        metadata={'test_run': True, 'scenarios': ['baseline', 'high_speed']}
    )
    
    # Create visualization
    print("\n📊 Creating graph visualization...")
    generator.visualize_graph(system_graph, "test_graph_visualization.png")
    
    print("\n✅ GNN topology implementation complete!")