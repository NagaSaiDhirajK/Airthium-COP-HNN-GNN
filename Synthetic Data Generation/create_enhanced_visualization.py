#!/usr/bin/env python3
"""
Enhanced Graph Visualization for AIRTHIUM Heat Pump System
=========================================================

Creates a clear, professional visualization of the heat pump system graph
showing component relationships, flow directions, and system topology.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import json

def create_enhanced_heat_pump_visualization():
    """Create an enhanced, professional heat pump system visualization."""
    
    # Create figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Define colors for different component types
    colors = {
        'compressor': '#FF6B6B',      # Red
        'evaporator': '#4ECDC4',      # Teal
        'condenser': '#45B7D1',       # Blue
        'expansion_valve': '#96CEB4', # Green
        'controller': '#FFEAA7',      # Yellow
        'sensor': '#DDA0DD',          # Plum
        'ambient': '#87CEEB',         # Sky Blue
        'load': '#FFB347',            # Orange
        'refrigerant': '#FF69B4',     # Hot Pink
        'control': '#9370DB'          # Medium Purple
    }
    
    # Component positions (optimized for clarity)
    components = {
        'compressor': {'pos': (2, 4), 'size': (1.2, 0.8), 'label': 'Scroll\nCompressor\n1800 RPM'},
        'evaporator': {'pos': (0.5, 2), 'size': (1.4, 1.0), 'label': 'Evaporator\n(Air Side)\n8 kW'},
        'condenser': {'pos': (7.5, 6), 'size': (1.4, 1.0), 'label': 'Condenser\n(Water Side)\n12 kW'},
        'expansion_valve': {'pos': (5, 2), 'size': (1.0, 0.6), 'label': 'Electronic\nExpansion Valve\n45% Open'},
        'controller': {'pos': (4, 6.5), 'size': (1.2, 0.6), 'label': 'Microprocessor\nController'},
        'ambient': {'pos': (0.5, 0.5), 'size': (1.0, 0.6), 'label': 'Ambient\n5°C'},
        'load': {'pos': (8.5, 4), 'size': (1.0, 0.8), 'label': 'Building\nLoad'}
    }
    
    # Draw components as fancy boxes
    component_patches = {}
    for comp_name, comp_data in components.items():
        x, y = comp_data['pos']
        w, h = comp_data['size']
        
        # Create fancy box
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=colors.get(comp_name, '#CCCCCC'),
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        component_patches[comp_name] = box
        
        # Add component label
        ax.text(x, y, comp_data['label'], 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Define refrigerant flow path with arrows
    refrigerant_path = [
        ('evaporator', 'compressor', 'Suction Line\nLow P, Low T'),
        ('compressor', 'condenser', 'Discharge Line\nHigh P, High T'),
        ('condenser', 'expansion_valve', 'Liquid Line\nHigh P, Med T'),
        ('expansion_valve', 'evaporator', 'Expansion\nLow P, Low T')
    ]
    
    # Draw refrigerant flow arrows
    for source, target, label in refrigerant_path:
        source_pos = components[source]['pos']
        target_pos = components[target]['pos']
        
        # Calculate arrow positions (offset from component centers)
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize and offset
        offset = 0.7  # Distance from component center
        start_x = source_pos[0] + (dx/length) * offset
        start_y = source_pos[1] + (dy/length) * offset
        end_x = target_pos[0] - (dx/length) * offset
        end_y = target_pos[1] - (dy/length) * offset
        
        # Draw thick arrow for refrigerant flow
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->', mutation_scale=25,
            color=colors['refrigerant'], linewidth=4,
            alpha=0.8
        )
        ax.add_patch(arrow)
        
        # Add flow label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.3, label, 
               ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    # Draw heat transfer connections
    heat_transfers = [
        ('ambient', 'evaporator', 'Heat Absorption\n8 kW'),
        ('condenser', 'load', 'Heat Rejection\n12 kW')
    ]
    
    for source, target, label in heat_transfers:
        source_pos = components[source]['pos']
        target_pos = components[target]['pos']
        
        # Draw dashed line for heat transfer
        ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], 
               '--', color='red', linewidth=3, alpha=0.7)
        
        # Add heat transfer label
        mid_x = (source_pos[0] + target_pos[0]) / 2
        mid_y = (source_pos[1] + target_pos[1]) / 2
        ax.text(mid_x, mid_y, label, 
               ha='center', va='center', fontsize=9, color='red', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    # Draw control signals
    control_signals = [
        ('controller', 'compressor', 'Speed Control'),
        ('controller', 'expansion_valve', 'Superheat Control')
    ]
    
    for source, target, label in control_signals:
        source_pos = components[source]['pos']
        target_pos = components[target]['pos']
        
        # Draw dotted line for control signals
        ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], 
               ':', color=colors['control'], linewidth=2, alpha=0.8)
        
        # Add small arrow
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        ax.annotate('', xy=target_pos, xytext=source_pos,
                   arrowprops=dict(arrowstyle='->', color=colors['control'], lw=2))
    
    # Add system performance metrics box
    perf_text = """SYSTEM PERFORMANCE
    
COP Heating: 4.00
COP Cooling: 2.67
Power Input: 3.0 kW
Heating Capacity: 12.0 kW
Efficiency: 80%

Operating Conditions:
• Ambient: 5°C
• Water In: 40°C
• Load: 80%"""
    
    ax.text(1, 7, perf_text, fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9),
           verticalalignment='top')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['refrigerant'], linewidth=4, label='Refrigerant Flow'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='Heat Transfer'),
        plt.Line2D([0], [0], color=colors['control'], linestyle=':', linewidth=2, label='Control Signal')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title and labels
    ax.set_title('AIRTHIUM Heat Pump System - Graph Neural Network Topology\n' +
                'Component Relationships and Energy Flows', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Save high-quality image
    plt.tight_layout()
    plt.savefig('AIRTHIUM_Heat_Pump_Graph_Enhanced.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('AIRTHIUM_Heat_Pump_Graph_Enhanced.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("📊 Enhanced visualization saved as:")
    print("   • AIRTHIUM_Heat_Pump_Graph_Enhanced.png (High-res PNG)")
    print("   • AIRTHIUM_Heat_Pump_Graph_Enhanced.pdf (Vector PDF)")
    
    plt.show()

if __name__ == "__main__":
    create_enhanced_heat_pump_visualization()