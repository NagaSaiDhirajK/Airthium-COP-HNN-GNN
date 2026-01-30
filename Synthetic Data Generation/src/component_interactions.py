#!/usr/bin/env python3
"""
Component Interaction Matrices and Heat/Mass Transfer Networks for AIRTHIUM Heat Pumps
=====================================================================================

This module implements component interaction matrices and heat/mass transfer networks
to model how different components affect each other in the heat pump system.

Key Features:
- Component interaction matrices
- Heat transfer network modeling
- Mass transfer network modeling
- Coupling effects between components
- Energy flow analysis
- System-level interactions

Sources:
- ASHRAE Handbook - Fundamentals (2021)
- Heat Transfer and Mass Transfer (Incropera & DeWitt, 2017)
- System Dynamics in Heat Pump Applications (DOE, 2020)
- Component Interaction Analysis (NIST, 2019)
- Thermal Network Modeling (IEEE, 2021)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
from scipy.linalg import solve, eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

@dataclass
class ComponentNode:
    """Represents a component node in the heat/mass transfer network."""
    node_id: str
    component_type: str
    position: Tuple[float, float, float]  # x, y, z coordinates
    
    # Thermal properties
    thermal_mass_J_per_K: float
    thermal_resistance_K_per_W: float
    surface_area_m2: float
    
    # Current state
    temperature_K: float
    pressure_Pa: float
    mass_flow_rate_kg_per_s: float
    
    # Heat transfer coefficients
    convection_coefficient_W_per_m2_K: float
    radiation_coefficient_W_per_m2_K: float
    conduction_coefficient_W_per_m_K: float

@dataclass
class InteractionEdge:
    """Represents an interaction edge between components."""
    edge_id: str
    source_node: str
    target_node: str
    interaction_type: str  # 'heat_transfer', 'mass_transfer', 'control', 'mechanical'
    
    # Transfer properties
    conductance_W_per_K: float
    flow_resistance_Pa_s_per_kg: float
    coupling_strength: float  # 0-1 scale
    
    # Dynamic properties
    time_constant_s: float
    delay_s: float

@dataclass
class SystemInteractionMatrix:
    """System-level interaction matrix."""
    matrix_id: str
    component_list: List[str]
    
    # Interaction matrices
    thermal_interaction_matrix: np.ndarray
    mass_interaction_matrix: np.ndarray
    control_interaction_matrix: np.ndarray
    
    # Network properties
    connectivity_matrix: np.ndarray
    adjacency_matrix: np.ndarray
    
    # System characteristics
    eigenvalues: np.ndarray
    stability_margin: float
    coupling_strength: float

class ComponentInteractionModel:
    """Model component interactions and heat/mass transfer networks."""
    
    def __init__(self):
        """Initialize component interaction model."""
        
        # Standard component thermal properties
        self.component_properties = {
            'compressor': {
                'thermal_mass_J_per_K': 15000.0,
                'thermal_resistance_K_per_W': 0.02,
                'surface_area_m2': 2.5,
                'convection_coefficient_W_per_m2_K': 25.0,
                'radiation_coefficient_W_per_m2_K': 5.0,
                'conduction_coefficient_W_per_m_K': 50.0
            },
            'evaporator': {
                'thermal_mass_J_per_K': 8000.0,
                'thermal_resistance_K_per_W': 0.001,
                'surface_area_m2': 15.0,
                'convection_coefficient_W_per_m2_K': 150.0,
                'radiation_coefficient_W_per_m2_K': 8.0,
                'conduction_coefficient_W_per_m_K': 200.0
            },
            'condenser': {
                'thermal_mass_J_per_K': 12000.0,
                'thermal_resistance_K_per_W': 0.0015,
                'surface_area_m2': 18.0,
                'convection_coefficient_W_per_m2_K': 120.0,
                'radiation_coefficient_W_per_m2_K': 10.0,
                'conduction_coefficient_W_per_m_K': 180.0
            },
            'expansion_valve': {
                'thermal_mass_J_per_K': 500.0,
                'thermal_resistance_K_per_W': 0.1,
                'surface_area_m2': 0.1,
                'convection_coefficient_W_per_m2_K': 20.0,
                'radiation_coefficient_W_per_m2_K': 3.0,
                'conduction_coefficient_W_per_m_K': 15.0
            },
            'refrigerant_lines': {
                'thermal_mass_J_per_K': 2000.0,
                'thermal_resistance_K_per_W': 0.05,
                'surface_area_m2': 5.0,
                'convection_coefficient_W_per_m2_K': 30.0,
                'radiation_coefficient_W_per_m2_K': 4.0,
                'conduction_coefficient_W_per_m_K': 400.0
            },
            'fan': {
                'thermal_mass_J_per_K': 3000.0,
                'thermal_resistance_K_per_W': 0.03,
                'surface_area_m2': 1.0,
                'convection_coefficient_W_per_m2_K': 40.0,
                'radiation_coefficient_W_per_m2_K': 6.0,
                'conduction_coefficient_W_per_m_K': 25.0
            }
        }
        
        # Interaction strength matrices (how much each component affects others)
        self.base_interaction_strengths = {
            'thermal': np.array([
                #  comp  evap  cond  valve lines  fan
                [1.00, 0.15, 0.25, 0.05, 0.20, 0.10],  # compressor
                [0.15, 1.00, 0.05, 0.30, 0.40, 0.60],  # evaporator
                [0.25, 0.05, 1.00, 0.20, 0.35, 0.70],  # condenser
                [0.05, 0.30, 0.20, 1.00, 0.50, 0.02],  # expansion_valve
                [0.20, 0.40, 0.35, 0.50, 1.00, 0.15],  # refrigerant_lines
                [0.10, 0.60, 0.70, 0.02, 0.15, 1.00]   # fan
            ]),
            'mass': np.array([
                [1.00, 0.80, 0.05, 0.10, 0.90, 0.00],  # compressor
                [0.80, 1.00, 0.02, 0.85, 0.95, 0.00],  # evaporator
                [0.05, 0.02, 1.00, 0.75, 0.90, 0.00],  # condenser
                [0.10, 0.85, 0.75, 1.00, 0.95, 0.00],  # expansion_valve
                [0.90, 0.95, 0.90, 0.95, 1.00, 0.00],  # refrigerant_lines
                [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]   # fan
            ]),
            'control': np.array([
                [1.00, 0.30, 0.20, 0.40, 0.10, 0.50],  # compressor
                [0.30, 1.00, 0.15, 0.25, 0.05, 0.80],  # evaporator
                [0.20, 0.15, 1.00, 0.35, 0.05, 0.90],  # condenser
                [0.40, 0.25, 0.35, 1.00, 0.10, 0.05],  # expansion_valve
                [0.10, 0.05, 0.05, 0.10, 1.00, 0.02],  # refrigerant_lines
                [0.50, 0.80, 0.90, 0.05, 0.02, 1.00]   # fan
            ])
        }
        
        # Component labels for matrix indexing
        self.component_labels = [
            'compressor', 'evaporator', 'condenser', 
            'expansion_valve', 'refrigerant_lines', 'fan'
        ]
        
        # Time constants for different interaction types
        self.interaction_time_constants = {
            'thermal': {
                'fast': 5.0,    # seconds (direct contact)
                'medium': 30.0,  # seconds (convection)
                'slow': 300.0   # seconds (radiation, conduction)
            },
            'mass': {
                'fast': 1.0,    # seconds (direct flow)
                'medium': 5.0,   # seconds (mixing)
                'slow': 30.0    # seconds (accumulation)
            },
            'control': {
                'fast': 0.1,    # seconds (electronic)
                'medium': 1.0,   # seconds (mechanical)
                'slow': 10.0    # seconds (thermal response)
            }
        }
    
    def create_component_network(self, system_id: str, 
                               component_states: Dict[str, Dict[str, float]]) -> Tuple[List[ComponentNode], List[InteractionEdge]]:
        """Create component network with nodes and edges."""
        
        nodes = []
        edges = []
        
        # Create component nodes
        for i, comp_type in enumerate(self.component_labels):
            if comp_type in component_states:
                state = component_states[comp_type]
                props = self.component_properties[comp_type]
                
                node = ComponentNode(
                    node_id=f"{system_id}_{comp_type}",
                    component_type=comp_type,
                    position=(i * 2.0, 0.0, 0.0),  # Simple linear layout
                    thermal_mass_J_per_K=props['thermal_mass_J_per_K'],
                    thermal_resistance_K_per_W=props['thermal_resistance_K_per_W'],
                    surface_area_m2=props['surface_area_m2'],
                    temperature_K=state.get('temperature_K', 298.15),
                    pressure_Pa=state.get('pressure_Pa', 101325.0),
                    mass_flow_rate_kg_per_s=state.get('mass_flow_rate_kg_per_s', 0.1),
                    convection_coefficient_W_per_m2_K=props['convection_coefficient_W_per_m2_K'],
                    radiation_coefficient_W_per_m2_K=props['radiation_coefficient_W_per_m2_K'],
                    conduction_coefficient_W_per_m_K=props['conduction_coefficient_W_per_m_K']
                )
                nodes.append(node)
        
        # Create interaction edges
        edge_id = 0
        for i, source_comp in enumerate(self.component_labels):
            for j, target_comp in enumerate(self.component_labels):
                if i != j and source_comp in component_states and target_comp in component_states:
                    
                    # Thermal interaction edge
                    if self.base_interaction_strengths['thermal'][i, j] > 0.1:
                        thermal_edge = InteractionEdge(
                            edge_id=f"{system_id}_thermal_{edge_id}",
                            source_node=f"{system_id}_{source_comp}",
                            target_node=f"{system_id}_{target_comp}",
                            interaction_type='heat_transfer',
                            conductance_W_per_K=self._calculate_thermal_conductance(
                                source_comp, target_comp, component_states
                            ),
                            flow_resistance_Pa_s_per_kg=0.0,
                            coupling_strength=self.base_interaction_strengths['thermal'][i, j],
                            time_constant_s=self.interaction_time_constants['thermal']['medium'],
                            delay_s=1.0
                        )
                        edges.append(thermal_edge)
                        edge_id += 1
                    
                    # Mass transfer edge (for refrigerant circuit)
                    if self.base_interaction_strengths['mass'][i, j] > 0.1:
                        mass_edge = InteractionEdge(
                            edge_id=f"{system_id}_mass_{edge_id}",
                            source_node=f"{system_id}_{source_comp}",
                            target_node=f"{system_id}_{target_comp}",
                            interaction_type='mass_transfer',
                            conductance_W_per_K=0.0,
                            flow_resistance_Pa_s_per_kg=self._calculate_flow_resistance(
                                source_comp, target_comp, component_states
                            ),
                            coupling_strength=self.base_interaction_strengths['mass'][i, j],
                            time_constant_s=self.interaction_time_constants['mass']['fast'],
                            delay_s=0.5
                        )
                        edges.append(mass_edge)
                        edge_id += 1
        
        return nodes, edges
    
    def build_system_interaction_matrix(self, system_id: str, nodes: List[ComponentNode], 
                                      edges: List[InteractionEdge]) -> SystemInteractionMatrix:
        """Build system-level interaction matrices."""
        
        n_components = len(nodes)
        component_ids = [node.node_id for node in nodes]
        
        # Initialize matrices
        thermal_matrix = np.zeros((n_components, n_components))
        mass_matrix = np.zeros((n_components, n_components))
        control_matrix = np.zeros((n_components, n_components))
        connectivity_matrix = np.zeros((n_components, n_components))
        
        # Build matrices from edges
        for edge in edges:
            try:
                source_idx = component_ids.index(edge.source_node)
                target_idx = component_ids.index(edge.target_node)
                
                connectivity_matrix[source_idx, target_idx] = 1.0
                
                if edge.interaction_type == 'heat_transfer':
                    thermal_matrix[source_idx, target_idx] = edge.conductance_W_per_K
                elif edge.interaction_type == 'mass_transfer':
                    mass_matrix[source_idx, target_idx] = 1.0 / max(edge.flow_resistance_Pa_s_per_kg, 1e-6)
                elif edge.interaction_type == 'control':
                    control_matrix[source_idx, target_idx] = edge.coupling_strength
                    
            except ValueError:
                # Node not found in component list
                continue
        
        # Make thermal matrix symmetric (heat transfer is bidirectional)
        thermal_matrix = (thermal_matrix + thermal_matrix.T) / 2
        
        # Add diagonal terms (self-interaction)
        for i in range(n_components):
            thermal_matrix[i, i] = nodes[i].thermal_mass_J_per_K / nodes[i].thermal_resistance_K_per_W
            mass_matrix[i, i] = 1.0  # Self mass transfer
            control_matrix[i, i] = 1.0  # Self control
        
        # Calculate system properties
        eigenvalues_thermal = eig(thermal_matrix)[0]
        eigenvalues_mass = eig(mass_matrix)[0]
        
        # Stability margin (minimum real part of eigenvalues)
        stability_margin = min(np.real(eigenvalues_thermal).min(), np.real(eigenvalues_mass).min())
        
        # Overall coupling strength
        coupling_strength = np.mean([
            np.mean(thermal_matrix[thermal_matrix > 0]),
            np.mean(mass_matrix[mass_matrix > 0]),
            np.mean(control_matrix[control_matrix > 0])
        ])
        
        # Adjacency matrix (binary connectivity)
        adjacency_matrix = (connectivity_matrix > 0).astype(float)
        
        return SystemInteractionMatrix(
            matrix_id=f"{system_id}_interaction_matrix",
            component_list=component_ids,
            thermal_interaction_matrix=thermal_matrix,
            mass_interaction_matrix=mass_matrix,
            control_interaction_matrix=control_matrix,
            connectivity_matrix=connectivity_matrix,
            adjacency_matrix=adjacency_matrix,
            eigenvalues=eigenvalues_thermal,
            stability_margin=stability_margin,
            coupling_strength=coupling_strength
        )
    
    def simulate_heat_transfer_network(self, interaction_matrix: SystemInteractionMatrix,
                                     nodes: List[ComponentNode], 
                                     heat_sources: Dict[str, float],
                                     simulation_time_s: float = 3600.0,
                                     dt: float = 1.0) -> List[Dict[str, float]]:
        """Simulate heat transfer through the component network."""
        
        n_components = len(nodes)
        temperatures = np.array([node.temperature_K for node in nodes])
        thermal_masses = np.array([node.thermal_mass_J_per_K for node in nodes])
        
        time_points = np.arange(0, simulation_time_s, dt)
        results = []
        
        for t in time_points:
            # Heat generation/removal
            heat_input = np.zeros(n_components)
            for i, node in enumerate(nodes):
                comp_name = node.component_type
                if comp_name in heat_sources:
                    heat_input[i] = heat_sources[comp_name]
            
            # Heat transfer between components
            thermal_matrix = interaction_matrix.thermal_interaction_matrix
            heat_transfer = np.dot(thermal_matrix, temperatures) - thermal_matrix.diagonal() * temperatures
            
            # Temperature update (explicit Euler method)
            dT_dt = (heat_input + heat_transfer) / thermal_masses
            temperatures += dT_dt * dt
            
            # Record results
            result = {'time_s': t}
            for i, node in enumerate(nodes):
                result[f"{node.component_type}_temperature_K"] = temperatures[i]
            results.append(result)
        
        return results
    
    def analyze_network_properties(self, interaction_matrix: SystemInteractionMatrix) -> Dict[str, float]:
        """Analyze network properties and characteristics."""
        
        adjacency = interaction_matrix.adjacency_matrix
        n_components = len(interaction_matrix.component_list)
        
        # Create NetworkX graph for analysis
        G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        
        # Network metrics
        try:
            # Connectivity metrics
            density = nx.density(G)
            
            # Centrality measures
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Average centralities
            avg_in_centrality = np.mean(list(in_degree_centrality.values()))
            avg_out_centrality = np.mean(list(out_degree_centrality.values()))
            avg_betweenness = np.mean(list(betweenness_centrality.values()))
            
            # Clustering coefficient
            clustering = nx.average_clustering(G.to_undirected())
            
            # Path lengths
            if nx.is_strongly_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                avg_path_length = float('inf')
            
        except:
            # Fallback values if NetworkX analysis fails
            density = np.sum(adjacency) / (n_components * (n_components - 1))
            avg_in_centrality = avg_out_centrality = avg_betweenness = density
            clustering = 0.0
            avg_path_length = float('inf')
        
        # Matrix condition numbers
        thermal_condition = np.linalg.cond(interaction_matrix.thermal_interaction_matrix)
        mass_condition = np.linalg.cond(interaction_matrix.mass_interaction_matrix)
        
        return {
            'network_density': density,
            'avg_in_degree_centrality': avg_in_centrality,
            'avg_out_degree_centrality': avg_out_centrality,
            'avg_betweenness_centrality': avg_betweenness,
            'clustering_coefficient': clustering,
            'average_path_length': avg_path_length,
            'thermal_matrix_condition': thermal_condition,
            'mass_matrix_condition': mass_condition,
            'stability_margin': interaction_matrix.stability_margin,
            'coupling_strength': interaction_matrix.coupling_strength,
            'eigenvalue_spread': np.max(np.real(interaction_matrix.eigenvalues)) - np.min(np.real(interaction_matrix.eigenvalues))
        }
    
    def _calculate_thermal_conductance(self, source_comp: str, target_comp: str,
                                     component_states: Dict[str, Dict[str, float]]) -> float:
        """Calculate thermal conductance between components."""
        
        source_props = self.component_properties[source_comp]
        target_props = self.component_properties[target_comp]
        
        # Series thermal resistance
        R_source = source_props['thermal_resistance_K_per_W']
        R_target = target_props['thermal_resistance_K_per_W']
        R_contact = 0.01  # Contact resistance
        
        total_resistance = R_source + R_target + R_contact
        conductance = 1.0 / total_resistance
        
        # Scale by interaction strength
        source_idx = self.component_labels.index(source_comp)
        target_idx = self.component_labels.index(target_comp)
        interaction_strength = self.base_interaction_strengths['thermal'][source_idx, target_idx]
        
        return conductance * interaction_strength
    
    def _calculate_flow_resistance(self, source_comp: str, target_comp: str,
                                 component_states: Dict[str, Dict[str, float]]) -> float:
        """Calculate flow resistance between components."""
        
        # Base flow resistance (simplified)
        base_resistance = 1000.0  # Pa·s/kg
        
        # Component-specific factors
        if source_comp == 'compressor' and target_comp == 'condenser':
            resistance = base_resistance * 0.1  # Low resistance (high pressure line)
        elif source_comp == 'expansion_valve' and target_comp == 'evaporator':
            resistance = base_resistance * 0.2  # Low resistance (low pressure line)
        elif 'valve' in source_comp or 'valve' in target_comp:
            resistance = base_resistance * 2.0   # High resistance (valve restriction)
        else:
            resistance = base_resistance
        
        return resistance
    
    def generate_interaction_scenarios(self, system_id: str, num_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Generate diverse component interaction scenarios."""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Random component states
            component_states = {}
            for comp_type in self.component_labels:
                component_states[comp_type] = {
                    'temperature_K': np.random.uniform(273.15, 373.15),
                    'pressure_Pa': np.random.uniform(50000, 2000000),
                    'mass_flow_rate_kg_per_s': np.random.uniform(0.01, 0.5)
                }
            
            # Create network
            nodes, edges = self.create_component_network(f"{system_id}_{i:03d}", component_states)
            
            # Build interaction matrix
            interaction_matrix = self.build_system_interaction_matrix(f"{system_id}_{i:03d}", nodes, edges)
            
            # Analyze network properties
            network_properties = self.analyze_network_properties(interaction_matrix)
            
            # Random heat sources
            heat_sources = {}
            for comp_type in ['compressor', 'evaporator', 'condenser']:
                heat_sources[comp_type] = np.random.uniform(-5000, 5000)  # W
            
            # Simulate heat transfer
            heat_transfer_results = self.simulate_heat_transfer_network(
                interaction_matrix, nodes, heat_sources, 600.0, 5.0
            )
            
            scenarios.append({
                'scenario_id': i,
                'component_states': component_states,
                'nodes': [asdict(node) for node in nodes],
                'edges': [asdict(edge) for edge in edges],
                'interaction_matrix': asdict(interaction_matrix),
                'network_properties': network_properties,
                'heat_sources': heat_sources,
                'heat_transfer_simulation': heat_transfer_results[:12]  # First minute only
            })
        
        return scenarios
    
    def save_interaction_data(self, interaction_scenarios: List[Dict[str, Any]], output_path: str) -> None:
        """Save component interaction data to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            return obj
        
        # Convert scenarios
        serializable_scenarios = []
        for scenario in interaction_scenarios:
            serializable_scenario = {}
            for key, value in scenario.items():
                if key == 'interaction_matrix':
                    # Convert interaction matrix arrays
                    matrix_dict = {}
                    for matrix_key, matrix_value in value.items():
                        matrix_dict[matrix_key] = convert_numpy(matrix_value)
                    serializable_scenario[key] = matrix_dict
                else:
                    serializable_scenario[key] = convert_numpy(value)
            serializable_scenarios.append(serializable_scenario)
        
        data = {
            'interaction_scenarios': serializable_scenarios,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'model_parameters': {
                'component_properties': self.component_properties,
                'base_interaction_strengths': convert_numpy(self.base_interaction_strengths),
                'component_labels': self.component_labels,
                'interaction_time_constants': self.interaction_time_constants
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved component interaction data to {output_path}")


if __name__ == "__main__":
    # Test component interaction models
    print("🔄 Testing Component Interaction Models")
    
    # Initialize model
    interaction_model = ComponentInteractionModel()
    
    # Create test component states
    component_states = {
        'compressor': {'temperature_K': 353.15, 'pressure_Pa': 1800000, 'mass_flow_rate_kg_per_s': 0.15},
        'evaporator': {'temperature_K': 278.15, 'pressure_Pa': 400000, 'mass_flow_rate_kg_per_s': 0.15},
        'condenser': {'temperature_K': 318.15, 'pressure_Pa': 1800000, 'mass_flow_rate_kg_per_s': 0.15},
        'expansion_valve': {'temperature_K': 298.15, 'pressure_Pa': 900000, 'mass_flow_rate_kg_per_s': 0.15},
        'refrigerant_lines': {'temperature_K': 288.15, 'pressure_Pa': 600000, 'mass_flow_rate_kg_per_s': 0.15},
        'fan': {'temperature_K': 298.15, 'pressure_Pa': 101325, 'mass_flow_rate_kg_per_s': 0.0}
    }
    
    # Create component network
    nodes, edges = interaction_model.create_component_network("test_system", component_states)
    
    print(f"\n📊 Component Network Created:")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    
    # Build interaction matrix
    interaction_matrix = interaction_model.build_system_interaction_matrix("test_system", nodes, edges)
    
    print(f"\n🔗 Interaction Matrix Properties:")
    print(f"Matrix size: {interaction_matrix.thermal_interaction_matrix.shape}")
    print(f"Stability margin: {interaction_matrix.stability_margin:.2f}")
    print(f"Coupling strength: {interaction_matrix.coupling_strength:.3f}")
    
    # Analyze network properties
    network_props = interaction_model.analyze_network_properties(interaction_matrix)
    
    print(f"\n📈 Network Analysis:")
    print(f"Network density: {network_props['network_density']:.3f}")
    print(f"Clustering coefficient: {network_props['clustering_coefficient']:.3f}")
    print(f"Average path length: {network_props['average_path_length']:.2f}")
    
    # Simulate heat transfer
    heat_sources = {'compressor': 2000, 'evaporator': -3000, 'condenser': 1000}
    heat_results = interaction_model.simulate_heat_transfer_network(
        interaction_matrix, nodes, heat_sources, 300.0, 5.0
    )
    
    print(f"\n🌡️ Heat Transfer Simulation:")
    print(f"Simulated {len(heat_results)} time steps")
    initial_temp = heat_results[0]['compressor_temperature_K']
    final_temp = heat_results[-1]['compressor_temperature_K']
    print(f"Compressor temp change: {initial_temp:.1f}K → {final_temp:.1f}K")
    
    # Generate interaction scenarios
    print(f"\n🎲 Generating interaction scenarios...")
    scenarios = interaction_model.generate_interaction_scenarios("test_system", 3)
    print(f"Generated {len(scenarios)} interaction scenarios")
    
    # Save test data
    interaction_model.save_interaction_data(scenarios, "component_interaction_data.json")
    
    print("\n✅ Component interaction models testing complete!")