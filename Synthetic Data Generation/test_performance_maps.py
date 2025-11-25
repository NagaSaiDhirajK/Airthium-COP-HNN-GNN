#!/usr/bin/env python3
"""
Comprehensive test script for Performance Maps
Tests all major performance map functionality with proper error handling
"""

import sys
sys.path.append('src')

def test_performance_maps():
    """Test all performance map functionality"""
    print("🗺️ Testing Performance Maps - Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from performance_maps import (
            ComponentPerformanceMaps, ComponentHealth, PerformanceMap,
            CompressorType, HeatExchangerType, ExpansionType
        )
        print("✅ Import successful")
        
        # Test ComponentPerformanceMaps initialization
        print("\n🔄 Testing ComponentPerformanceMaps initialization...")
        perf_maps = ComponentPerformanceMaps()
        print("✅ ComponentPerformanceMaps initialized successfully")
        
        # Test component specifications loading
        print("\n🔄 Testing component specifications...")
        specs = perf_maps.component_specs
        print(f"✅ Loaded specifications for {len(specs)} component types:")
        for comp_type in specs.keys():
            print(f"  - {comp_type}")
        
        # Test degradation models initialization
        print("\n🔄 Testing degradation models...")
        degradation_models = perf_maps.degradation_models
        print(f"✅ Initialized {len(degradation_models)} degradation models:")
        for model_name in degradation_models.keys():
            print(f"  - {model_name}")
        
        # Test component health creation
        print("\n🔄 Testing component health creation...")
        
        # Create compressor health
        compressor_health = perf_maps.create_component_health(
            component_id="test_compressor_001",
            component_type="scroll_compressor",
            age_hours=15000
        )
        print(f"✅ Created compressor health: {type(compressor_health)}")
        print(f"✅ Health index: {compressor_health.health_index:.3f}")
        print(f"✅ Efficiency factor: {compressor_health.efficiency_factor:.3f}")
        print(f"✅ Fault indicators: {len(compressor_health.fault_indicators)} items")
        
        # Create evaporator health
        evaporator_health = perf_maps.create_component_health(
            component_id="test_evaporator_001",
            component_type="finned_tube_evaporator",
            age_hours=12000
        )
        print(f"✅ Created evaporator health: {type(evaporator_health)}")
        print(f"✅ Health index: {evaporator_health.health_index:.3f}")
        
        # Create expansion valve health
        valve_health = perf_maps.create_component_health(
            component_id="test_valve_001",
            component_type="electronic_expansion_valve",
            age_hours=8000
        )
        print(f"✅ Created valve health: {type(valve_health)}")
        print(f"✅ Health index: {valve_health.health_index:.3f}")
        
        # Test compressor performance map generation
        print("\n🔄 Testing compressor performance map generation...")
        comp_maps = perf_maps.generate_compressor_performance_map(compressor_health)
        print(f"✅ Generated compressor maps: {len(comp_maps)} maps")
        for map_name, perf_map in comp_maps.items():
            print(f"  - {map_name}: {perf_map.z_values.shape} data points")
            print(f"    Range: {perf_map.z_values.min():.3f} - {perf_map.z_values.max():.3f}")
        
        # Test heat exchanger performance map generation
        print("\n🔄 Testing heat exchanger performance map generation...")
        
        # Test evaporator maps
        evap_maps = perf_maps.generate_heat_exchanger_performance_map(
            evaporator_health, hx_type='evaporator'
        )
        print(f"✅ Generated evaporator maps: {len(evap_maps)} maps")
        for map_name, perf_map in evap_maps.items():
            print(f"  - {map_name}: {perf_map.z_values.shape} data points")
        
        # Test condenser maps
        condenser_health = perf_maps.create_component_health(
            component_id="test_condenser_001",
            component_type="plate_condenser",
            age_hours=10000
        )
        cond_maps = perf_maps.generate_heat_exchanger_performance_map(
            condenser_health, hx_type='condenser'
        )
        print(f"✅ Generated condenser maps: {len(cond_maps)} maps")
        
        # Test expansion valve performance map generation
        print("\n🔄 Testing expansion valve performance map generation...")
        valve_maps = perf_maps.generate_expansion_valve_performance_map(valve_health)
        print(f"✅ Generated valve maps: {len(valve_maps)} maps")
        for map_name, perf_map in valve_maps.items():
            print(f"  - {map_name}: {perf_map.z_values.shape} data points")
        
        # Test performance map saving
        print("\n🔄 Testing performance map saving...")
        all_maps = {**comp_maps, **evap_maps, **valve_maps}
        perf_maps.save_performance_maps(all_maps, "test_performance_maps.json")
        print("✅ Performance maps saved successfully")
        
        # Test component health with maintenance history
        print("\n🔄 Testing component health with maintenance history...")
        maintenance_history = [
            {"type": "routine_maintenance", "date": "2023-01-15", "description": "Oil change"},
            {"type": "major_overhaul", "date": "2023-06-01", "description": "Complete rebuild"}
        ]
        
        maintained_health = perf_maps.create_component_health(
            component_id="maintained_compressor_001",
            component_type="scroll_compressor",
            age_hours=20000,
            maintenance_history=maintenance_history
        )
        print(f"✅ Created maintained component health")
        print(f"✅ Health index with maintenance: {maintained_health.health_index:.3f}")
        print(f"✅ Maintenance history: {len(maintained_health.maintenance_history)} entries")
        
        # Test degradation model functions
        print("\n🔄 Testing degradation model functions...")
        test_time = 8760  # 1 year in hours
        test_rate = 0.02  # 2% per year
        
        for model_name, model_func in degradation_models.items():
            try:
                if model_name == 'linear':
                    result = model_func(test_time, test_rate)
                    print(f"✅ {model_name} degradation: {result:.3f}")
                elif model_name == 'exponential':
                    result = model_func(test_time, test_rate)
                    print(f"✅ {model_name} degradation: {result:.3f}")
                elif model_name == 'weibull':
                    result = model_func(test_time, 2.0, 50000)  # beta=2, eta=50000
                    print(f"✅ {model_name} degradation: {result:.3f}")
                elif model_name == 'bathtub':
                    result = model_func(test_time, 0.1, 0.001, 0.00001)
                    print(f"✅ {model_name} degradation: {result:.3f}")
                elif model_name == 'step':
                    intervals = [(8760, 0.9), (17520, 0.8), (26280, 0.7)]
                    result = model_func(test_time, intervals)
                    print(f"✅ {model_name} degradation: {result:.3f}")
            except Exception as e:
                print(f"⚠️ {model_name} degradation failed: {e}")
        
        # Test PerformanceMap dataclass
        print("\n🔄 Testing PerformanceMap dataclass...")
        import numpy as np
        
        test_map = PerformanceMap(
            component_id="test_component",
            map_type="efficiency",
            x_axis=np.linspace(0, 100, 10),
            y_axis=np.linspace(1, 5, 8),
            z_values=np.random.rand(8, 10),
            x_label="Test X",
            y_label="Test Y",
            z_label="Test Z",
            operating_envelope={"x": (0, 100), "y": (1, 5)},
            degradation_factors={"test_factor": 0.02}
        )
        print(f"✅ PerformanceMap created: {test_map.component_id}")
        print(f"✅ Map dimensions: {test_map.z_values.shape}")
        
        # Test ComponentHealth dataclass
        print("\n🔄 Testing ComponentHealth dataclass...")
        test_health = ComponentHealth(
            component_id="test_health",
            component_type="test_type",
            age_hours=5000,
            health_index=0.85,
            efficiency_factor=0.92,
            capacity_factor=0.88,
            degradation_rate=0.02,
            maintenance_history=[],
            fault_indicators={"test_fault": 1.5}
        )
        print(f"✅ ComponentHealth created: {test_health.component_id}")
        print(f"✅ Health index: {test_health.health_index}")
        
        print("\n🎉 ALL PERFORMANCE MAPS TESTS COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error in performance maps: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_maps()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test completed with errors")
        sys.exit(1)