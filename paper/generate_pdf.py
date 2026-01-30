#!/usr/bin/env python3
"""
Generate PDF from Section 2 Methodology content using FPDF2.
Enhanced version with more scientific context and explanations.
"""

from fpdf import FPDF
from pathlib import Path

class MethodologyPDF(FPDF):
    def header(self):
        pass
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    
    def section_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 14)
            self.ln(8)
            self.multi_cell(0, 8, title)
            self.ln(2)
        elif level == 2:
            self.set_font('Helvetica', 'B', 12)
            self.ln(6)
            self.multi_cell(0, 7, title)
            self.ln(2)
        elif level == 3:
            self.set_font('Helvetica', 'B', 11)
            self.ln(4)
            self.multi_cell(0, 6, title)
            self.ln(1)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def equation(self, text, num=None):
        self.set_font('Courier', '', 9)
        eq_text = f"  {text}"
        if num:
            eq_text += f"  ({num})"
        self.multi_cell(0, 5, eq_text)
        self.ln(2)
    
    def bold_start(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.write(5, text)
        self.set_font('Helvetica', '', 10)
    
    def add_table(self, headers, rows, caption=None):
        self.set_font('Helvetica', 'B', 9)
        col_widths = [35, 55, 90]
        
        # Header
        self.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True)
        self.ln()
        
        # Rows
        self.set_font('Helvetica', '', 8)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, cell, border=1)
            self.ln()
        
        if caption:
            self.set_font('Helvetica', 'I', 8)
            self.ln(2)
            self.multi_cell(0, 4, caption)
        self.ln(4)

# Create PDF
pdf = MethodologyPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 16)
pdf.cell(0, 10, 'Section 2: Methodology', align='C')
pdf.ln(8)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 6, 'Physics-Informed Hamiltonian Neural Network with Graph Neural Network', align='C')
pdf.ln(5)
pdf.cell(0, 6, 'for Heat Pump COP Prediction', align='C')
pdf.ln(15)

# Introduction paragraph
pdf.set_font('Helvetica', '', 10)
pdf.multi_cell(0, 5, 
    "This section presents our comprehensive methodology for predicting heat pump Coefficient "
    "of Performance (COP) using a physics-informed hybrid neural network architecture. We "
    "develop a digital twin that integrates 17 validated physics models to generate "
    "thermodynamically consistent training data, then employ a combined Hamiltonian Neural "
    "Network (HNN) and Graph Neural Network (GNN) to learn the complex nonlinear relationships "
    "between operating conditions and energy efficiency. The key innovation is embedding "
    "physical laws directly into both data generation and model architecture, ensuring "
    "predictions respect fundamental thermodynamic constraints."
)
pdf.ln(4)

# 2.1 Architecture
pdf.section_title('2.1 Physics-Based Digital Twin Architecture', 1)
pdf.body_text(
    "Heat pump systems exhibit complex, nonlinear behavior arising from coupled thermodynamic, "
    "mechanical, and control processes. Traditional black-box machine learning approaches "
    "struggle to capture these physics-driven relationships and may produce predictions that "
    "violate fundamental laws. Our physics-based digital twin addresses this by implementing "
    "a hierarchical four-layer data architecture that systematically transforms raw operating "
    "conditions into validated COP predictions through comprehensive physics simulation."
)

pdf.bold_start("Layer 1 (Inputs) ")
pdf.body_text(
    "consists of 24 sampled operating parameters that define the complete state of the heat "
    "pump system and its environment. Environmental conditions include ambient temperature "
    "(T_ambient: 253-318 K), relative humidity (phi: 0.2-0.95), wind speed, and solar "
    "irradiance. Building characteristics capture thermal load requirements (Q_load: 3-15 kW) "
    "and occupancy patterns. System configuration specifies refrigerant type (R410A, R134a, "
    "R290), design capacity, and equipment age. Control setpoints define target temperatures "
    "and superheat/subcooling margins. These inputs are sampled using stratified Latin "
    "hypercube sampling to ensure comprehensive coverage of the operating envelope while "
    "avoiding clustering in any particular region of the parameter space."
)

pdf.bold_start("Layer 2 (Physics Outputs) ")
pdf.body_text(
    "contains 47 intermediate variables computed by executing all 17 physics models in "
    "sequence. This layer captures the physical state of every component: evaporator and "
    "condenser temperatures and pressures, refrigerant mass flow rates, compressor efficiency "
    "degradation, heat exchanger fouling resistance, control system response characteristics, "
    "and Hamiltonian energy states. Each variable is computed from first principles rather "
    "than empirical correlations, ensuring physical consistency."
)

pdf.bold_start("Layer 3 (Derived Features) ")
pdf.body_text(
    "computes 12 higher-level performance metrics that aggregate physics outputs into "
    "quantities directly relevant to COP prediction. Heat transfer rates (Q_evap, Q_cond) "
    "quantify thermal energy exchange. Compressor work (W_comp) represents electrical input. "
    "Efficiency metrics compare actual performance to theoretical limits (Carnot efficiency). "
    "Combined degradation factors synthesize individual component losses into system-level "
    "impacts."
)

pdf.bold_start("Layer 4 (Targets) ")
pdf.body_text(
    "produces the final validated COP values along with five thermodynamic consistency "
    "checks. Every generated cycle must satisfy: (1) energy balance within 5%, (2) COP "
    "below the Carnot limit, (3) positive entropy generation, (4) mass conservation, and "
    "(5) physically realistic operating ranges. Cycles failing any check are rejected, "
    "ensuring the training dataset contains only thermodynamically valid examples."
)

# 2.2 Physics Models
pdf.section_title('2.2 Integrated Physics Models', 1)
pdf.body_text(
    "The accuracy of physics-informed machine learning depends critically on the fidelity "
    "of the underlying physics models. We integrate 17 specialized models covering "
    "thermodynamic cycles, component degradation, environmental effects, control dynamics, "
    "and Hamiltonian mechanics. Each model has been validated against manufacturer data "
    "and published experimental results. Table 1 summarizes the models and their key outputs."
)

# Table 1
headers = ['Category', 'Model', 'Key Outputs']
rows = [
    ['Thermodynamic', 'RefrigerantProperties', 'P_sat, h, s, rho'],
    ['', 'CycleGenerator', 'State points 1-4, m_dot'],
    ['Degradation', 'HeatExchangerFoulingModel', 'R_f, delta_epsilon'],
    ['', 'CompressorWearModel', 'W_b, L_v, eta_degrad'],
    ['', 'RefrigerantChargeModel', 'f_COP,charge, dT_sc'],
    ['', 'FaultInjectionModel', 'f_COP,fault, fault type'],
    ['Environmental', 'AmbientConditionModel', 'T_wb, rho_air, r_frost'],
    ['', 'MultiZoneBuildingModel', 'Q_load, zone temps'],
    ['Control', 'ControlSystemModel', 'N_comp, EXV%, PID states'],
    ['Hamiltonian', 'HamiltonianMechanicsModel', 'H, q, p trajectories'],
    ['', 'ComponentInteractionModel', 'alpha_thermal, alpha_mass'],
    ['Part-Load', 'PartLoadCyclingModel', 'eta_PLR, l_cycling'],
    ['', 'TemporalDynamicsGenerator', '100-step transients'],
    ['Measurement', 'SensorNoiseModel', 'Noisy T, P measurements'],
    ['Component', 'CompressorModel', 'eta_isen(r_p, N)'],
    ['', 'HeatExchangerModel', 'epsilon, NTU, UA'],
    ['', 'GNNTopologyGenerator', 'Node/edge features'],
]
pdf.add_table(headers, rows, "Table 1: Summary of 17 integrated physics models in the digital twin.")

# 2.2.1 Thermodynamic
pdf.section_title('2.2.1 Thermodynamic Cycle Models', 2)

pdf.body_text(
    "The vapor compression refrigeration cycle forms the thermodynamic foundation of heat "
    "pump operation. Accurate COP prediction requires precise calculation of refrigerant "
    "properties at each state point in the cycle."
)

pdf.bold_start("RefrigerantProperties Model. ")
pdf.body_text(
    "Thermodynamic state calculations are performed using the CoolProp library, which "
    "implements high-accuracy equations of state based on Helmholtz free energy "
    "formulations. For refrigerants R410A (zeotropic blend), R134a (HFC), and R290 "
    "(propane), saturation pressures follow the Antoine equation:"
)
pdf.equation("ln(P_sat) = A - B/(T + C)", "1")
pdf.body_text(
    "where A, B, C are refrigerant-specific constants derived from NIST reference data. "
    "The model provides complete thermodynamic states including enthalpy (h), entropy (s), "
    "and density (rho) at all four cycle points."
)

pdf.bold_start("CycleGenerator Model. ")
pdf.body_text(
    "The ideal vapor compression cycle consists of four processes: isentropic compression, "
    "isobaric condensation, isenthalpic expansion, and isobaric evaporation. Real cycles "
    "deviate from this ideal due to compressor irreversibilities, pressure drops, and heat "
    "transfer limitations. Our model captures these effects:"
)
pdf.equation("State 1->2: h_2 = h_1 + (h_2s - h_1)/eta_isen", "2")
pdf.equation("State 2->3: Q_cond = m_dot(h_2 - h_3)", "3")
pdf.equation("State 3->4: h_4 = h_3 (isenthalpic)", "4")
pdf.equation("State 4->1: Q_evap = m_dot(h_1 - h_4)", "5")
pdf.body_text(
    "The isentropic efficiency (eta_isen) accounts for compression irreversibilities and "
    "typically ranges from 0.65-0.85 for scroll compressors."
)

# 2.2.2 Degradation
pdf.section_title('2.2.2 Degradation Models', 2)
pdf.body_text(
    "Heat pump performance degrades over time due to fouling, mechanical wear, and "
    "refrigerant charge loss. These degradation mechanisms can reduce COP by 10-30% "
    "over a system's lifetime. Our models capture these effects to enable accurate "
    "prediction across the full range of system conditions."
)

pdf.bold_start("HeatExchangerFoulingModel. ")
pdf.body_text(
    "Fouling deposits on heat exchanger surfaces create thermal resistance that reduces "
    "heat transfer effectiveness. The Kern-Seaton asymptotic model captures the "
    "time-dependent buildup:"
)
pdf.equation("R_f(t) = R_f* (1 - exp(-t/tau_f))", "6")
pdf.body_text(
    "where R_f* is the asymptotic fouling resistance (typically 0.0001-0.001 m2K/W) and "
    "tau_f is the characteristic time (1000-5000 hours). Effectiveness degradation:"
)
pdf.equation("delta_epsilon/epsilon_0 = R_f/(R_f + R_clean)", "7")

pdf.bold_start("CompressorWearModel. ")
pdf.body_text(
    "Mechanical degradation occurs through bearing wear, valve seat erosion, and lubricant "
    "degradation. These effects compound to reduce compressor efficiency:"
)
pdf.equation("eta_comp(t) = eta_0 * (1 - alpha_b*W_b - alpha_v*L_v) * f_oil", "8")
pdf.body_text("Wear accumulation depends on operating severity:")
pdf.equation("dW_b/dt = k_b * (P_cond/P_evap)^n_b * f_load", "9")
pdf.body_text(
    "Higher pressure ratios and frequent cycling accelerate wear. After 50,000 hours, "
    "efficiency degradation can reach 15-20%."
)

pdf.bold_start("RefrigerantChargeModel. ")
pdf.body_text(
    "Refrigerant charge deviation from nominal significantly impacts subcooling, superheat, "
    "and overall COP. Undercharge reduces capacity; overcharge increases pressure ratio:"
)
pdf.equation("dT_sc = dT_sc,nom * (m_charge/m_nom)^1.3", "10")
pdf.equation("dT_sh = dT_sh,nom * (m_nom/m_charge)^0.8", "11")
pdf.equation("f_COP,charge = 1 - alpha_c*|(m_charge - m_nom)/m_nom|^1.2", "12")
pdf.body_text("A 20% charge deviation typically reduces COP by 8-15%.")

pdf.bold_start("FaultInjectionModel. ")
pdf.body_text(
    "System faults are stochastically injected with 20% probability to train the model "
    "on degraded conditions. Severity-dependent COP impact factors:"
)
pdf.equation("f_COP,fault = 1.0 (no fault)", "13a")
pdf.equation("f_COP,fault = 0.95 - 0.15s (refrigerant leak)", "13b")
pdf.equation("f_COP,fault = 0.90 - 0.20s (compressor degradation)", "13c")

# 2.2.3 Environmental
pdf.section_title('2.2.3 Environmental Models', 2)
pdf.body_text(
    "Ambient conditions strongly influence heat pump performance. Capacity and efficiency "
    "vary with temperature, humidity, and solar load. These effects must be captured "
    "for accurate prediction across diverse climates."
)

pdf.bold_start("AmbientConditionModel. ")
pdf.body_text(
    "Psychrometric calculations determine air properties affecting heat exchanger performance. "
    "The Stull formula approximates wet-bulb temperature:"
)
pdf.equation("T_wb = T_db*arctan[0.152(RH + 8.314)^0.5] + arctan(T_db + RH) - arctan(RH - 1.676)", "14")
pdf.body_text("Moist air density:")
pdf.equation("rho_air = P_atm/(R_air*T_db) * 1/(1 + 0.608*omega)", "15")
pdf.body_text("Temperature-dependent performance modifiers:")
pdf.equation("f_cap,heat = 1 - 0.015(T_ambient - 280)", "16")
pdf.equation("f_COP,heat = 1 - 0.012(280 - T_ambient)", "17")
pdf.body_text(
    "Frost formation on evaporator coils occurs when surface temperature drops below dew point:"
)
pdf.equation("r_frost = max(0, (T_dp - T_evap,surf)/5)", "18")

pdf.bold_start("MultiZoneBuildingModel. ")
pdf.body_text(
    "Building thermal response is modeled using an equivalent RC (resistance-capacitance) "
    "network that captures envelope heat loss, internal gains, and thermal mass:"
)
pdf.equation("C_bldg dT_zone/dt = (T_ambient - T_zone)/R_envelope + Q_internal + Q_solar - Q_hvac", "19")
pdf.body_text(
    "Internal gains include occupancy (70 W/person), lighting, and equipment. This model "
    "enables realistic load profiles for capacity matching analysis."
)

# 2.2.4 Control
pdf.section_title('2.2.4 Control System Models', 2)
pdf.body_text(
    "Modern heat pumps use sophisticated control strategies to optimize efficiency and "
    "comfort. The control system model captures the dynamic response of PID controllers "
    "that regulate compressor speed and expansion valve opening."
)
pdf.bold_start("ControlSystemModel. ")
pdf.body_text("Standard PID control law:")
pdf.equation("u(t) = K_p*e(t) + K_i*integral(e(tau)dtau) + K_d*de/dt", "20")
pdf.body_text(
    "The compressor speed controller maintains capacity to match building load, while the "
    "electronic expansion valve (EXV) controller maintains target superheat. Control "
    "response time (30-90 seconds) and overshoot (3-10%) are captured as features."
)

# 2.2.5 Hamiltonian
pdf.section_title('2.2.5 Hamiltonian Mechanics Models', 2)
pdf.body_text(
    "A key innovation in our approach is formulating the thermodynamic cycle in Hamiltonian "
    "phase space. This enables the Hamiltonian Neural Network to learn energy-conserving "
    "dynamics, improving generalization and physical consistency."
)
pdf.bold_start("HamiltonianMechanicsModel. ")
pdf.body_text("Generalized coordinates represent thermodynamic state variables:")
pdf.equation("q = [q_T, q_P, q_V, q_S]^T,  p = [q_dot_T, q_dot_P, q_dot_V, q_dot_S]^T", "21")
pdf.body_text("The system Hamiltonian represents total energy (kinetic + potential):")
pdf.equation("H(q, p) = (1/2)*p^T*M^(-1)*p + U(q)", "22")
pdf.body_text("Hamilton's canonical equations govern temporal evolution:")
pdf.equation("dq/dt = dH/dp,  dp/dt = -dH/dq", "23")
pdf.body_text(
    "For the cyclic vapor compression process, phase space trajectories form closed orbits, "
    "analogous to harmonic oscillators. The HNN learns to preserve this structure, "
    "guaranteeing energy conservation: dH/dt = 0."
)

# 2.2.6 Component Interaction
pdf.section_title('2.2.6 Component Interaction Models', 2)
pdf.body_text(
    "Heat pump components are thermally and hydraulically coupled. The ComponentInteractionModel "
    "quantifies these couplings for the GNN's message-passing mechanism."
)
pdf.bold_start("ComponentInteractionModel. ")
pdf.body_text("Thermal and mass coupling strengths:")
pdf.equation("alpha_thermal = m_dot*c_p*dT/Q_ref", "24")
pdf.equation("alpha_mass = (m_dot/m_dot_nom)*f_PLR", "25")
pdf.body_text(
    "Network stability margin (sigma_stability = min_i Re(lambda_i)) quantifies proximity "
    "to instability in the coupled system."
)

pdf.bold_start("GNNTopologyGenerator. ")
pdf.body_text(
    "The heat pump is represented as a directed graph G = (V, E) with 10 nodes (compressor, "
    "condenser, evaporator, expansion valve, refrigerant circuit, control system, ambient, "
    "building, system health) and 11 edges representing refrigerant flow and thermal "
    "coupling. Node features (35 per node) encode component states; edge features (12 per "
    "edge) encode mass flow rates, temperature differences, and coupling strengths."
)

# 2.2.7 Part-Load
pdf.section_title('2.2.7 Part-Load and Cycling Models', 2)
pdf.body_text(
    "Heat pumps rarely operate at full capacity. Part-load performance significantly "
    "impacts seasonal efficiency. Variable-speed systems maintain better efficiency at "
    "part load than single-speed systems that must cycle on and off."
)
pdf.bold_start("PartLoadCyclingModel. ")
pdf.body_text("Part-load efficiency degradation follows AHRI Standard 210/240 curves:")
pdf.equation("eta_PLR = 1 - 0.2(1-PLR)^3 + 0.1(1-PLR)  [variable speed]", "26")
pdf.equation("eta_PLR = 1 - 0.4(1-PLR)^2 - 0.2(1-PLR)  [single speed]", "27")
pdf.body_text("Cycling losses for on-off operation:")
pdf.equation("eta_cycling = 1 - C_D(1 - f_on),  C_D ~ 0.25", "28")
pdf.body_text(
    "Variable-speed systems can achieve 95%+ efficiency at 50% load, while single-speed "
    "systems may drop to 70%."
)

# 2.2.8 Temporal
pdf.section_title('2.2.8 Temporal Dynamics Models', 2)
pdf.body_text(
    "Transient behavior during startup, load changes, and defrost cycles affects efficiency "
    "and must be captured for accurate dynamic prediction."
)
pdf.bold_start("TemporalDynamicsGenerator. ")
pdf.body_text("Startup transients follow first-order dynamics:")
pdf.equation("X(t) = X_inf(1 - exp(-t/tau))", "29")
pdf.body_text(
    "The model generates 100-timestep trajectories (typical cycle duration: 5-15 minutes) "
    "for temperature, pressure, mass flow, and heat transfer rates. These temporal "
    "sequences provide the sequential input for the HNN's recurrent processing."
)

# 2.2.9 Sensor
pdf.section_title('2.2.9 Sensor and Measurement Models', 2)
pdf.body_text(
    "Real sensor measurements contain noise and uncertainty. The SensorNoiseModel adds "
    "realistic measurement errors to improve model robustness."
)
pdf.bold_start("SensorNoiseModel. ")
pdf.body_text("Gaussian measurement noise:")
pdf.equation("T_measured = T_actual + N(0, sigma_T^2),  sigma_T = 0.5 K", "30")
pdf.equation("P_measured = P_actual*(1 + N(0, sigma_P^2)),  sigma_P = 0.015", "31")
pdf.body_text(
    "Temperature uncertainty of 0.5 K and pressure uncertainty of 1.5% reflect typical "
    "industrial sensor specifications."
)

# 2.2.10 Component Performance
pdf.section_title('2.2.10 Component Performance Models', 2)
pdf.body_text(
    "Detailed component models capture how efficiency varies with operating point."
)
pdf.bold_start("CompressorModel. ")
pdf.body_text("Performance map relating efficiency to pressure ratio and speed:")
pdf.equation("eta_isen = eta_0 - alpha_1(r_p - r_p,0)^2 - alpha_2(N - N_0)^2", "32")
pdf.body_text(
    "Peak efficiency occurs at design conditions; off-design operation reduces efficiency."
)

pdf.bold_start("HeatExchangerModel. ")
pdf.body_text("Effectiveness-NTU method for heat exchanger sizing:")
pdf.equation("epsilon = 1 - exp[-(1/C_r)(1 - exp(-C_r*NTU))]", "33")
pdf.body_text(
    "where C_r = C_min/C_max is the capacity ratio and NTU = UA/C_min is the number of "
    "transfer units."
)

# 2.3 COP Calculation
pdf.section_title('2.3 COP Calculation Methodology', 1)
pdf.body_text(
    "The Coefficient of Performance (COP) is the fundamental efficiency metric for heat "
    "pumps, defined as the ratio of useful thermal output to electrical input. Our "
    "calculation methodology aggregates effects from all 17 physics models."
)
pdf.body_text("Base thermodynamic COP from the ideal cycle:")
pdf.equation("COP_heating,base = Q_cond/W_comp = m_dot(h_2 - h_3)/m_dot(h_2 - h_1)", "34")
pdf.equation("COP_cooling,base = Q_evap/W_comp = m_dot(h_1 - h_4)/m_dot(h_2 - h_1)", "35")

pdf.body_text(
    "The actual COP is reduced from the base value by multiplicative degradation factors "
    "from each physics model:"
)
pdf.equation("COP_actual = COP_base * f_charge * f_fault * (1-eta_degrad) * f_ambient * eta_PLR * (1-l_cycling)", "36")
pdf.body_text(
    "This formulation ensures that all physical degradation mechanisms are captured and "
    "that the predicted COP reflects real-world performance."
)

pdf.bold_start("Thermodynamic Validation. ")
pdf.body_text("All generated cycles must satisfy four fundamental constraints:")
pdf.body_text(
    "1. Energy balance: |Q_cond - (Q_evap + W_comp)| < 0.05*Q_cond\n"
    "2. Carnot limit: COP <= COP_Carnot = T_H/(T_H - T_C)\n"
    "3. Entropy generation: S_gen = Q_cond/T_H - Q_evap/T_C >= 0\n"
    "4. Mass conservation: m_dot_in = m_dot_out at all components"
)
pdf.body_text(
    "Cycles failing any validation check are rejected. This ensures the neural network "
    "trains exclusively on physically consistent examples, preventing it from learning "
    "spurious patterns that violate thermodynamic laws."
)

# 2.4 Dataset Generation
pdf.section_title('2.4 Dataset Generation and Distribution Filtering', 1)
pdf.body_text(
    "Training data quality directly impacts model performance. Our generation pipeline "
    "produces 100,000 thermodynamically validated cycles with realistic COP distributions. "
    "A key challenge is that random sampling of physics parameters tends to produce "
    "skewed COP distributions. We address this through probabilistic acceptance filtering."
)
pdf.body_text(
    "The COPDistributionFilter implements rejection sampling to shape the output distribution "
    "toward a target normal distribution N(mu_target, sigma_target^2) while preserving "
    "physical relationships between inputs and outputs:"
)
pdf.equation("P_accept(c) = min(1, sqrt(p_target,i/p_current,i))  when p_current,i < p_target,i", "37a")
pdf.equation("P_accept(c) = (p_target,i/p_current,i)^gamma  when p_current,i >= p_target,i", "37b")
pdf.body_text(
    "where gamma = 3 controls filter aggressiveness. This approach accepts cycles from "
    "under-represented COP bins with higher probability while rejecting over-represented "
    "bins, achieving a balanced training distribution without manipulating the physics."
)

# Feature dimensions table
pdf.ln(4)
pdf.set_font('Helvetica', 'B', 10)
pdf.multi_cell(0, 6, "Table 2: Feature dimensions for HNN-GNN model inputs")
pdf.set_font('Helvetica', '', 9)
pdf.cell(60, 6, "Feature Type", border=1, fill=True)
pdf.cell(60, 6, "Dimensions", border=1, fill=True)
pdf.cell(40, 6, "Total", border=1, fill=True)
pdf.ln()
pdf.cell(60, 5, "HNN Temporal", border=1)
pdf.cell(60, 5, "100 timesteps x 23 channels", border=1)
pdf.cell(40, 5, "2,300", border=1)
pdf.ln()
pdf.cell(60, 5, "GNN Node Features", border=1)
pdf.cell(60, 5, "10 nodes x 35 features", border=1)
pdf.cell(40, 5, "350", border=1)
pdf.ln()
pdf.cell(60, 5, "GNN Edge Features", border=1)
pdf.cell(60, 5, "11 edges x 12 features", border=1)
pdf.cell(40, 5, "132", border=1)
pdf.ln()
pdf.cell(60, 5, "GNN Graph Features", border=1)
pdf.cell(60, 5, "1 x 50 features", border=1)
pdf.cell(40, 5, "50", border=1)
pdf.ln()
pdf.set_font('Helvetica', 'B', 9)
pdf.cell(60, 5, "Total per cycle", border=1)
pdf.cell(60, 5, "", border=1)
pdf.cell(40, 5, "2,832", border=1)
pdf.ln(6)

pdf.body_text(
    "Each cycle provides 2,832 features: 2,300 from temporal dynamics (for the HNN) and "
    "532 from graph structure (for the GNN). This rich representation captures both the "
    "sequential evolution of thermodynamic states and the spatial relationships between "
    "system components."
)

# Dataset stats table
pdf.ln(2)
pdf.set_font('Helvetica', 'B', 10)
pdf.multi_cell(0, 6, "Table 3: Generated dataset statistics")
pdf.set_font('Helvetica', '', 9)
stats = [
    ("Total cycles", "100,000"),
    ("COP mean (mu)", "4.53"),
    ("COP std (sigma)", "1.57"),
    ("COP range", "[1.8, 8.0]"),
    ("Carnot efficiency range", "[0.50, 0.85]"),
    ("Refrigerants", "R410A, R134a, R290"),
    ("Operating modes", "Heating, Cooling"),
    ("Ambient temperature range", "253-318 K (-20C to +45C)"),
    ("System age range", "0-50,000 hours"),
]
for metric, value in stats:
    pdf.cell(80, 5, metric, border=1)
    pdf.cell(80, 5, value, border=1)
    pdf.ln()
pdf.ln(4)

pdf.body_text(
    "The resulting dataset spans realistic operating conditions for residential and "
    "commercial heat pumps across diverse climates. COP values range from 1.8 (extreme "
    "conditions with significant degradation) to 8.0 (optimal mild-climate operation), "
    "with Carnot efficiencies between 50-85% reflecting real-world system performance."
)

# Sample demonstration
pdf.section_title('Sample Dataset Demonstration', 1)
pdf.body_text(
    "To illustrate the data generation pipeline, we present a representative cycle from "
    "the dataset. This example shows how input conditions propagate through physics "
    "models to produce validated COP predictions."
)

pdf.set_fill_color(245, 249, 255)
pdf.set_draw_color(200, 220, 255)

pdf.bold_start("Layer 1 Inputs (sample):")
pdf.ln()
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4, 
    "- Ambient: T_ambient = 283.2 K (10C), phi = 0.72 (72% RH), v_wind = 3.4 m/s\n"
    "- Building: Q_load = 8.5 kW, f_occupancy = 0.65 (65% occupied)\n"
    "- System: Refrigerant = R410A, Design capacity = 12 kW, Age = 15,000 hours\n"
    "- Operating Mode: Heating, Part-load ratio = 0.75 (75% capacity)"
)
pdf.ln(2)

pdf.bold_start("Layer 2 Physics Outputs (sample):")
pdf.ln()
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4,
    "- Thermodynamic cycle: T_evap = 268.3 K, T_cond = 323.1 K, P_evap = 652 kPa, P_cond = 2,105 kPa\n"
    "- Degradation states: Evap. fouling = 0.00032 m2K/W, Bearing wear = 12.5%, Charge = 93.5%\n"
    "- Control outputs: Compressor speed = 72.3%, EXV opening = 43.8%\n"
    "- Hamiltonian state: H = 4,850 J, Energy conservation drift = 0.12%"
)
pdf.ln(2)

pdf.bold_start("Layer 3 Derived Features (sample):")
pdf.ln()
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4,
    "- Heat transfer: Q_evap = 10.2 kW (absorbed), Q_cond = 12.8 kW (delivered)\n"
    "- Power input: W_comp = 2.6 kW\n"
    "- Reference efficiency: COP_Carnot = 5.89, Carnot efficiency = 68%"
)
pdf.ln(2)

pdf.bold_start("Layer 4 Targets (sample):")
pdf.ln()
pdf.set_font('Helvetica', '', 9)
pdf.multi_cell(0, 4,
    "- Predicted COP_heating = 3.99 (68% of Carnot limit)\n"
    "- Validation: Energy balance error = 0.02% (PASS)\n"
    "- Validation: Below Carnot limit (PASS)\n"
    "- Validation: Positive entropy generation (PASS)\n"
    "- Status: Physically valid cycle accepted into dataset"
)

# Save
output_path = Path(__file__).parent / "section2_methodology.pdf"
pdf.output(str(output_path))
print(f"PDF generated: {output_path}")
