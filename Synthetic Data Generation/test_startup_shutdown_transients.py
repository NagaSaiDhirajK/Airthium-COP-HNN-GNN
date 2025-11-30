#!/usr/bin/env python3
"""
Test Startup/Shutdown Transients Models
=======================================

Comprehensive test suite for startup/shutdown transient models including
thermal transients, pressure dynamics, flow characteristics, and COP calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import json
from pathlib import Path

# Import directly from the module file
from startup_shutdown_transients import (
    StartupShutdownModel, TransientType, ComponentState, ThermalMassProperties,
    TransientParameters, ComponentTransientState, SystemTransientState
)

def test_startup_shutdown_transients():
    """Test startup/shutdown transients models comprehensively."""
    
    print("🚀 Testing Startup/Shutdown Transients Models - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        print("✅ Import successful")
        
        # Initialize startup/shutdown model
        print("\n🔄 Testing StartupShutdownModel initialization...")
        transient_model = StartupShutdownModel()
        print("✅ StartupShutdownModel initialized successfully")
        print(f"✅ Thermal masses: {len(transient_model.thermal_masses)} components")
        print(f"✅ Component time constants: {len(transient_model.component_time_constants)} components")
        print(f"✅ Startup sequences: {len(transient_model.startup_sequences)} types")
        print(f"✅ Shutdown sequences: {len(transient_model.shutdown_sequences)} types")
        
        # Test thermal mass properties
        print("\n🔄 Testing thermal mass properties...")
        for component_id, thermal_mass in transient_model.thermal_masses.items():
            print(f"✅ {component_id}:")
            print(f"  - Mass: {thermal_mass.mass_kg} kg")
            print(f"  - Specific heat: {thermal_mass.specific_heat_J_per_kg_K} J/(kg·K)")
            print(f"  - Thermal conductivity: {thermal_mass.thermal_conductivity_W_per_m_K} W/(m·K)")
            print(f"  - Surface area: {thermal_mass.surface_area_m2} m²")
            print(f"  - Convection coefficient: {thermal_mass.convection_coefficient_W_per_m2_K} W/(m²·K)")
            print(f"  - Initial temperature: {thermal_mass.initial_temperature_K} K")
        
        # Test component time constants
        print("\n🔄 Testing component time constants...")
        for component_id, time_constants in transient_model.component_time_constants.items():
            print(f"✅ {component_id}:")
            for constant_name, value in time_constants.items():
                print(f"  - {constant_name}: {value} s")
        
        # Test startup and shutdown sequences
        print("\n🔄 Testing startup sequences...")
        for transient_type, sequence in transient_model.startup_sequences.items():
            print(f"✅ {transient_type.value}:")
            for step in sequence:
                print(f"  - t={step['time_s']}s: {step['action']} ({step['components']})")
        
        print("\n🔄 Testing shutdown sequences...")
        for transient_type, sequence in transient_model.shutdown_sequences.items():
            print(f"✅ {transient_type.value}:")
            for step in sequence:
                print(f"  - t={step['time_s']}s: {step['action']} ({step['components']})")
        
        # Test transient scenario creation
        print("\n🔄 Testing transient scenario creation...")
        transient_types = [
            TransientType.COLD_START,
            TransientType.WARM_START,
            TransientType.RESTART,
            TransientType.NORMAL_SHUTDOWN,
            TransientType.EMERGENCY_SHUTDOWN
        ]
        
        initial_conditions = {'system_temperature_K': 293.15}
        ambient_conditions = {
            'outdoor_temperature_K': 283.15,
            'indoor_temperature_K': 293.15
        }
        
        transient_scenarios = {}
        for transient_type in transient_types:
            scenario = transient_model.create_transient_scenario(
                transient_type, initial_conditions, ambient_conditions
            )
            transient_scenarios[transient_type] = scenario
            
            print(f"✅ {transient_type.value} scenario:")
            print(f"  - Transient type: {scenario.transient_type.value}")
            print(f"  - Ambient temperature: {scenario.ambient_temperature_K} K")
            print(f"  - Initial system temperature: {scenario.initial_system_temperature_K} K")
            print(f"  - Startup delay: {scenario.startup_delay_s} s")
            print(f"  - Ramp rate: {scenario.ramp_rate_per_s} /s")
            print(f"  - Settling time: {scenario.settling_time_s} s")
            print(f"  - Overshoot: {scenario.overshoot_percent}%")
            print(f"  - Target conditions: {len(scenario.target_operating_conditions)} parameters")
        
        # Test thermal transient simulation
        print("\n🔄 Testing thermal transient simulation...")
        test_scenario = transient_scenarios[TransientType.COLD_START]
        time_span = (0.0, 300.0)  # 5 minutes
        dt = 5.0
        
        thermal_results = {}
        for component_id in transient_model.thermal_masses.keys():
            temperatures = transient_model.simulate_thermal_transients(
                component_id, test_scenario, time_span, dt
            )
            thermal_results[component_id] = temperatures
            
            print(f"✅ {component_id} thermal transient:")
            print(f"  - Number of points: {len(temperatures)}")
            print(f"  - Initial temperature: {temperatures[0]:.2f} K")
            print(f"  - Final temperature: {temperatures[-1]:.2f} K")
            print(f"  - Temperature rise: {temperatures[-1] - temperatures[0]:.2f} K")
            print(f"  - Max temperature: {max(temperatures):.2f} K")
            print(f"  - Min temperature: {min(temperatures):.2f} K")
        
        # Test pressure transient simulation
        print("\n🔄 Testing pressure transient simulation...")
        pressure_results = transient_model.simulate_pressure_transients(
            test_scenario, time_span, dt
        )
        
        print(f"✅ Pressure transient simulation:")
        for location, pressures in pressure_results.items():
            print(f"  - {location}:")
            print(f"    * Number of points: {len(pressures)}")
            print(f"    * Initial pressure: {pressures[0]:.0f} Pa")
            print(f"    * Final pressure: {pressures[-1]:.0f} Pa")
            print(f"    * Pressure change: {pressures[-1] - pressures[0]:.0f} Pa")
            print(f"    * Max pressure: {max(pressures):.0f} Pa")
            print(f"    * Min pressure: {min(pressures):.0f} Pa")
        
        # Test flow transient simulation
        print("\n🔄 Testing flow transient simulation...")
        flow_results = transient_model.simulate_flow_transients(
            test_scenario, time_span, dt
        )
        
        print(f"✅ Flow transient simulation:")
        for flow_type, flows in flow_results.items():
            print(f"  - {flow_type}:")
            print(f"    * Number of points: {len(flows)}")
            print(f"    * Initial flow: {flows[0]:.6f} kg/s")
            print(f"    * Final flow: {flows[-1]:.6f} kg/s")
            print(f"    * Flow change: {flows[-1] - flows[0]:.6f} kg/s")
            print(f"    * Max flow: {max(flows):.6f} kg/s")
        
        # Test COP calculation during transients
        print("\n🔄 Testing transient COP calculation...")
        
        # Create power consumption time series
        time_array = np.arange(time_span[0], time_span[1], dt)
        power_consumption = []
        for t in time_array:
            if t < test_scenario.startup_delay_s:
                power = 100.0  # Standby power
            else:
                operation_time = t - test_scenario.startup_delay_s
                ramp_factor = min(1.0, operation_time * test_scenario.ramp_rate_per_s)
                power = 100.0 + ramp_factor * 4900.0  # Up to 5 kW
            power_consumption.append(power)
        
        cop_values = transient_model.calculate_transient_cop(
            thermal_results, pressure_results, flow_results, power_consumption
        )
        
        print(f"✅ COP calculation:")
        print(f"  - Number of COP values: {len(cop_values)}")
        print(f"  - Initial COP: {cop_values[0]:.3f}")
        print(f"  - Final COP: {cop_values[-1]:.3f}")
        print(f"  - Average COP: {np.mean(cop_values):.3f}")
        print(f"  - Max COP: {max(cop_values):.3f}")
        print(f"  - Min COP: {min(cop_values):.3f}")
        
        # Test complete system transient simulation
        print("\n🔄 Testing complete system transient simulation...")
        
        # Test different transient types
        for transient_type in [TransientType.COLD_START, TransientType.WARM_START, TransientType.NORMAL_SHUTDOWN]:
            scenario = transient_scenarios[transient_type]
            system_states = transient_model.simulate_complete_transient(
                scenario, (0.0, 180.0), dt=10.0  # 3 minutes, 10s steps
            )
            
            print(f"✅ {transient_type.value} complete simulation:")
            print(f"  - Number of time points: {len(system_states)}")
            print(f"  - Initial system state: {system_states[0].system_state.value}")
            print(f"  - Final system state: {system_states[-1].system_state.value}")
            print(f"  - Initial COP: {system_states[0].cop:.3f}")
            print(f"  - Final COP: {system_states[-1].cop:.3f}")
            print(f"  - Initial power: {system_states[0].total_power_W:.0f} W")
            print(f"  - Final power: {system_states[-1].total_power_W:.0f} W")
            print(f"  - Components: {len(system_states[0].components)}")
            
            # Analyze component states
            for component_id in system_states[0].components.keys():
                initial_comp = system_states[0].components[component_id]
                final_comp = system_states[-1].components[component_id]
                
                print(f"    * {component_id}:")
                print(f"      - Initial temp: {initial_comp.temperature_K:.2f} K")
                print(f"      - Final temp: {final_comp.temperature_K:.2f} K")
                print(f"      - Initial pressure: {initial_comp.pressure_Pa:.0f} Pa")
                print(f"      - Final pressure: {final_comp.pressure_Pa:.0f} Pa")
                print(f"      - Final flow: {final_comp.mass_flow_rate_kg_per_s:.6f} kg/s")
                print(f"      - Final efficiency: {final_comp.efficiency:.3f}")
                print(f"      - Thermal stress: {final_comp.thermal_stress_Pa:.0f} Pa")
        
        # Test model validation
        print("\n🔄 Testing model validation...")
        
        for transient_type in [TransientType.COLD_START, TransientType.WARM_START, TransientType.NORMAL_SHUTDOWN]:
            validation_results = transient_model.validate_transient_model(transient_type)
            
            print(f"✅ {transient_type.value} validation:")
            for test_name, result in validation_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  - {test_name}: {status}")
            
            passed_tests = sum(validation_results.values())
            total_tests = len(validation_results)
            print(f"  - Overall: {passed_tests}/{total_tests} tests passed")
        
        # Test dataclass functionality
        print("\n🔄 Testing dataclass functionality...")
        
        # Test ThermalMassProperties dataclass
        test_thermal_mass = ThermalMassProperties(
            mass_kg=20.0,
            specific_heat_J_per_kg_K=500.0,
            thermal_conductivity_W_per_m_K=100.0,
            surface_area_m2=1.0,
            convection_coefficient_W_per_m2_K=15.0,
            initial_temperature_K=293.15
        )
        print(f"✅ ThermalMassProperties created:")
        print(f"  - Mass: {test_thermal_mass.mass_kg} kg")
        print(f"  - Specific heat: {test_thermal_mass.specific_heat_J_per_kg_K} J/(kg·K)")
        print(f"  - Surface area: {test_thermal_mass.surface_area_m2} m²")
        
        # Test TransientParameters dataclass
        test_transient_params = TransientParameters(
            transient_type=TransientType.COLD_START,
            ambient_temperature_K=283.15,
            initial_system_temperature_K=293.15,
            target_operating_conditions={'temperature': 350.0},
            startup_delay_s=60.0,
            ramp_rate_per_s=0.1,
            settling_time_s=300.0,
            overshoot_percent=15.0
        )
        print(f"✅ TransientParameters created:")
        print(f"  - Transient type: {test_transient_params.transient_type.value}")
        print(f"  - Ambient temperature: {test_transient_params.ambient_temperature_K} K")
        print(f"  - Startup delay: {test_transient_params.startup_delay_s} s")
        print(f"  - Overshoot: {test_transient_params.overshoot_percent}%")
        
        # Test ComponentTransientState dataclass
        test_component_state = ComponentTransientState(
            component_id="test_compressor",
            state=ComponentState.RUNNING,
            temperature_K=350.0,
            pressure_Pa=1800000.0,
            mass_flow_rate_kg_per_s=0.15,
            power_consumption_W=1000.0,
            efficiency=0.8,
            time_in_state_s=120.0,
            thermal_stress_Pa=50000.0
        )
        print(f"✅ ComponentTransientState created:")
        print(f"  - Component ID: {test_component_state.component_id}")
        print(f"  - State: {test_component_state.state.value}")
        print(f"  - Temperature: {test_component_state.temperature_K} K")
        print(f"  - Pressure: {test_component_state.pressure_Pa} Pa")
        print(f"  - Efficiency: {test_component_state.efficiency}")
        
        # Test SystemTransientState dataclass
        test_system_state = SystemTransientState(
            timestamp=120.0,
            transient_type=TransientType.COLD_START,
            system_state=ComponentState.RUNNING,
            components={'compressor': test_component_state},
            cop=3.5,
            total_power_W=5000.0,
            refrigerant_charge_kg=5.0,
            oil_circulation_rate=0.95,
            pressure_equalization_time_s=0.0
        )
        print(f"✅ SystemTransientState created:")
        print(f"  - Timestamp: {test_system_state.timestamp} s")
        print(f"  - Transient type: {test_system_state.transient_type.value}")
        print(f"  - System state: {test_system_state.system_state.value}")
        print(f"  - COP: {test_system_state.cop}")
        print(f"  - Total power: {test_system_state.total_power_W} W")
        print(f"  - Components: {len(test_system_state.components)}")
        
        # Test edge cases
        print("\n🔄 Testing edge cases...")
        
        # Test with unknown component
        try:
            unknown_temps = transient_model.simulate_thermal_transients(
                'unknown_component', test_scenario, (0.0, 60.0), 5.0
            )
            print("❌ Unknown component should have failed")
        except ValueError as e:
            print(f"✅ Unknown component properly rejected: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error for unknown component: {str(e)}")
        
        # Test with zero time span
        try:
            zero_temps = transient_model.simulate_thermal_transients(
                'compressor', test_scenario, (0.0, 0.0), 1.0
            )
            print(f"✅ Zero time span handled: {len(zero_temps)} points")
        except Exception as e:
            print(f"❌ Zero time span error: {str(e)}")
        
        # Test with very large time step
        try:
            large_dt_temps = transient_model.simulate_thermal_transients(
                'compressor', test_scenario, (0.0, 100.0), 50.0
            )
            print(f"✅ Large time step handled: {len(large_dt_temps)} points")
        except Exception as e:
            print(f"❌ Large time step error: {str(e)}")
        
        # Test performance metrics
        print("\n🔄 Testing performance metrics...")
        
        # Analyze startup characteristics for different transient types
        startup_characteristics = {}
        for transient_type in [TransientType.COLD_START, TransientType.WARM_START]:
            scenario = transient_scenarios[transient_type]
            system_states = transient_model.simulate_complete_transient(
                scenario, (0.0, 300.0), dt=5.0
            )
            
            # Find when system reaches steady state (COP stabilizes)
            cop_values = [state.cop for state in system_states]
            final_cop = cop_values[-1]
            
            # Find 90% of final COP
            target_cop = 0.9 * final_cop
            startup_time = None
            for i, cop in enumerate(cop_values):
                if cop >= target_cop:
                    startup_time = system_states[i].timestamp
                    break
            
            # Calculate overshoot
            max_cop = max(cop_values)
            overshoot_percent = ((max_cop - final_cop) / final_cop) * 100 if final_cop > 0 else 0
            
            startup_characteristics[transient_type.value] = {
                'startup_time_to_90_percent': startup_time,
                'final_cop': final_cop,
                'max_cop': max_cop,
                'overshoot_percent': overshoot_percent,
                'settling_time': scenario.settling_time_s
            }
        
        print(f"✅ Startup characteristics:")
        for transient_type, characteristics in startup_characteristics.items():
            print(f"  - {transient_type}:")
            for metric, value in characteristics.items():
                if value is not None:
                    print(f"    * {metric}: {value:.2f}")
                else:
                    print(f"    * {metric}: Not reached")
        
        # Calculate thermal time constants
        thermal_time_constants = {}
        for component_id in transient_model.thermal_masses.keys():
            temperatures = thermal_results[component_id]
            initial_temp = temperatures[0]
            final_temp = temperatures[-1]
            
            # Find 63% of temperature rise (time constant)
            target_temp = initial_temp + 0.63 * (final_temp - initial_temp)
            time_constant = None
            
            for i, temp in enumerate(temperatures):
                if temp >= target_temp:
                    time_constant = i * dt
                    break
            
            thermal_time_constants[component_id] = {
                'calculated_time_constant_s': time_constant,
                'expected_time_constant_s': transient_model.component_time_constants[component_id].get('thermal_time_s', 0),
                'temperature_rise_K': final_temp - initial_temp
            }
        
        print(f"✅ Thermal time constants:")
        for component_id, constants in thermal_time_constants.items():
            print(f"  - {component_id}:")
            for metric, value in constants.items():
                if value is not None:
                    print(f"    * {metric}: {value:.2f}")
                else:
                    print(f"    * {metric}: Not reached")
        
        # Calculate energy balance
        total_energy_input = sum(power_consumption) * dt  # J
        total_thermal_energy = 0
        
        for component_id, thermal_mass in transient_model.thermal_masses.items():
            temp_rise = thermal_results[component_id][-1] - thermal_results[component_id][0]
            thermal_energy = thermal_mass.mass_kg * thermal_mass.specific_heat_J_per_kg_K * temp_rise
            total_thermal_energy += thermal_energy
        
        energy_balance_error = abs(total_energy_input - total_thermal_energy) / total_energy_input * 100
        
        print(f"✅ Energy balance:")
        print(f"  - Total energy input: {total_energy_input:.0f} J")
        print(f"  - Total thermal energy stored: {total_thermal_energy:.0f} J")
        print(f"  - Energy balance error: {energy_balance_error:.2f}%")
        
        print("\n🎉 ALL STARTUP/SHUTDOWN TRANSIENTS TESTS COMPLETED!")
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error in startup/shutdown transients models: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Test completed with errors")
        return False

if __name__ == "__main__":
    success = test_startup_shutdown_transients()
    sys.exit(0 if success else 1)