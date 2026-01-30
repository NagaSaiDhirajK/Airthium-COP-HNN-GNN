# Section 2: Methodology

## 2.1 Physics-Based Digital Twin Architecture

Our physics-based digital twin employs a hierarchical four-layer data architecture that transforms raw operating conditions into validated COP predictions through comprehensive physics simulation. This architecture ensures complete traceability from inputs to outputs while maintaining thermodynamic consistency at every stage.

**Layer 1 (Inputs)** consists of 24 sampled operating parameters spanning environmental conditions ($T_{\text{ambient}}$, $\phi$, $v_{\text{wind}}$, $G_{\text{solar}}$), building characteristics ($Q_{\text{load}}$, $f_{\text{occupancy}}$), system configuration (refrigerant type, design capacity, nominal charge), and control setpoints ($T_{\text{setpoint}}$, $\Delta T_{\text{superheat}}$, $\Delta T_{\text{subcooling}}$). These inputs are sampled using stratified Latin hypercube sampling to ensure comprehensive coverage of the operating envelope.

**Layer 2 (Physics Outputs)** contains 47 intermediate variables computed by 17 integrated physics models. Each model contributes specific physical phenomena—from refrigerant thermodynamics to component degradation—ensuring that all relevant physics affecting COP are captured.

**Layer 3 (Derived Features)** computes 12 performance metrics including heat transfer rates ($\dot{Q}_{\text{evap}}$, $\dot{Q}_{\text{cond}}$), compressor work ($\dot{W}_{\text{comp}}$), efficiency metrics ($\eta_{\text{Carnot}}$, $\eta_{\text{actual}}$), and combined degradation factors from all physics models.

**Layer 4 (Targets)** produces the final validated COP values along with five thermodynamic consistency checks: energy balance verification, Carnot limit compliance, entropy generation positivity, mass balance satisfaction, and overall physical validity.

## 2.2 Integrated Physics Models

We integrate 17 physics models that comprehensively capture heat pump behavior across thermodynamic, degradation, control, and environmental domains.

### 2.2.1 Thermodynamic Cycle Models

**RefrigerantProperties Model.** Thermodynamic state calculations are performed using CoolProp [12] with real-gas equations of state. For refrigerants R410A, R134a, and R290, saturation pressures are computed via the Antoine equation:

$$\ln P_{\text{sat}} = A - \frac{B}{T + C}$$

where $A$, $B$, $C$ are refrigerant-specific constants. Specific enthalpies at cycle state points are obtained from:

$$h = h(T, P) = \int_{T_0}^{T} c_p \, dT + \left[ h - \int_0^P \left( v - T \frac{\partial v}{\partial T} \right) dP \right]_{T_0}$$

The model provides saturated vapor properties at the evaporator outlet (state 1), superheated discharge conditions (state 2), saturated liquid at condenser outlet (state 3), and two-phase mixture after expansion (state 4).

**CycleGenerator Model.** The vapor compression cycle is modeled following the standard four-process sequence with real-gas deviations:

$$\text{State 1→2: } h_2 = h_1 + \frac{h_{2s} - h_1}{\eta_{\text{isen}}}$$

$$\text{State 2→3: } \dot{Q}_{\text{cond}} = \dot{m}(h_2 - h_3)$$

$$\text{State 3→4: } h_4 = h_3 \text{ (isenthalpic)}$$

$$\text{State 4→1: } \dot{Q}_{\text{evap}} = \dot{m}(h_1 - h_4)$$

where $\eta_{\text{isen}}$ is the isentropic efficiency and $\dot{m}$ is the refrigerant mass flow rate.

### 2.2.2 Degradation Models

**HeatExchangerFoulingModel.** Fouling resistance evolution follows the Kern-Seaton asymptotic model [13]:

$$R_f(t) = R_f^* \left(1 - e^{-t/\tau_f}\right)$$

where $R_f^*$ is the asymptotic fouling resistance and $\tau_f$ is the fouling time constant dependent on water quality, temperature, and velocity. Heat exchanger effectiveness degradation is computed as:

$$\frac{\Delta \varepsilon}{\varepsilon_0} = \frac{R_f}{R_f + R_{\text{clean}}}$$

For finned-tube evaporators, we include particulate deposition effects scaled by air-side fouling factors.

**CompressorWearModel.** Mechanical degradation is modeled through bearing wear, valve leakage, and oil degradation:

$$\eta_{\text{comp}}(t) = \eta_0 \cdot (1 - \alpha_b W_b - \alpha_v L_v) \cdot f_{\text{oil}}$$

where $W_b$ is bearing wear percentage (function of operating hours and pressure ratio), $L_v$ is valve leakage percentage (function of start-stop cycles), and $f_{\text{oil}}$ is oil quality factor. The wear accumulation follows:

$$\frac{dW_b}{dt} = k_b \cdot \left(\frac{P_{\text{cond}}}{P_{\text{evap}}}\right)^{n_b} \cdot f_{\text{load}}$$

**RefrigerantChargeModel.** Charge deviation effects on subcooling and superheat follow empirical correlations [14]:

$$\Delta T_{\text{sc}} = \Delta T_{\text{sc,nom}} \cdot \left(\frac{m_{\text{charge}}}{m_{\text{nom}}}\right)^{1.3}$$

$$\Delta T_{\text{sh}} = \Delta T_{\text{sh,nom}} \cdot \left(\frac{m_{\text{nom}}}{m_{\text{charge}}}\right)^{0.8}$$

The COP impact factor accounts for both undercharge (reduced capacity) and overcharge (increased pressure ratio):

$$f_{\text{COP,charge}} = 1 - \alpha_c \left| \frac{m_{\text{charge}} - m_{\text{nom}}}{m_{\text{nom}}} \right|^{1.2}$$

**FaultInjectionModel.** System faults are stochastically injected with 20% probability, modeling refrigerant leaks, compressor degradation, and sensor drift. Each fault type has severity-dependent COP impact factors:

$$f_{\text{COP,fault}} = \begin{cases} 1.0 & \text{no fault} \\ 0.95 - 0.15s & \text{refrigerant leak} \\ 0.90 - 0.20s & \text{compressor degradation} \end{cases}$$

where $s \in [0, 1]$ is fault severity.

### 2.2.3 Environmental Models

**AmbientConditionModel.** Psychrometric calculations determine wet-bulb temperature, dew point, and air density:

$$T_{\text{wb}} = T_{\text{db}} \arctan\left[0.152(RH + 8.314)^{0.5}\right] + \arctan(T_{\text{db}} + RH) - \arctan(RH - 1.676)$$

$$\rho_{\text{air}} = \frac{P_{\text{atm}}}{R_{\text{air}} T_{\text{db}}} \cdot \frac{1}{1 + 0.608 \omega}$$

where $\omega$ is humidity ratio. Performance modifiers for heating and cooling modes account for temperature-dependent capacity and efficiency:

$$f_{\text{cap,heat}} = 1 - 0.015(T_{\text{ambient}} - 280)$$

$$f_{\text{COP,heat}} = 1 - 0.012(280 - T_{\text{ambient}})$$

Frost formation risk is computed when evaporator surface temperature approaches the dew point:

$$r_{\text{frost}} = \max\left(0, \frac{T_{\text{dp}} - T_{\text{evap,surf}}}{5}\right)$$

**MultiZoneBuildingModel.** Building thermal loads are computed using a resistance-capacitance network:

$$C_{\text{bldg}} \frac{dT_{\text{zone}}}{dt} = \frac{T_{\text{ambient}} - T_{\text{zone}}}{R_{\text{envelope}}} + \dot{Q}_{\text{internal}} + \dot{Q}_{\text{solar}} - \dot{Q}_{\text{hvac}}$$

Internal gains include occupancy ($70$ W/person), lighting, and equipment loads scaled by occupancy factor.

### 2.2.4 Control System Models

**ControlSystemModel.** PID controllers regulate compressor speed and expansion valve opening:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de}{dt}$$

The compressor speed controller maintains capacity to match building load:

$$\dot{N}_{\text{comp}} = f_{\text{PID}}(Q_{\text{load}} - Q_{\text{delivered}})$$

The expansion valve controller maintains superheat setpoint:

$$\text{EXV}_{\text{opening}} = f_{\text{PID}}(\Delta T_{\text{sh,setpoint}} - \Delta T_{\text{sh,actual}})$$

Control response time and overshoot are captured as features for the neural network.

### 2.2.5 Hamiltonian Mechanics Models

**HamiltonianMechanicsModel.** The thermodynamic cycle is formulated in Hamiltonian phase space with generalized coordinates representing temperature, pressure, volume, and entropy [1, 2]:

$$\mathbf{q} = [q_T, q_P, q_V, q_S]^T, \quad \mathbf{p} = [\dot{q}_T, \dot{q}_P, \dot{q}_V, \dot{q}_S]^T$$

The system Hamiltonian representing total energy is:

$$\mathcal{H}(\mathbf{q}, \mathbf{p}) = \underbrace{\frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}}_{T(\mathbf{p})} + \underbrace{U(\mathbf{q})}_{V(\mathbf{q})}$$

where $\mathbf{M}$ is the generalized mass matrix and $U(\mathbf{q})$ is the potential energy from thermodynamic state. Hamilton's equations govern the temporal evolution:

$$\frac{d\mathbf{q}}{dt} = \frac{\partial \mathcal{H}}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial \mathcal{H}}{\partial \mathbf{q}}$$

For the cyclic vapor compression process, the phase space trajectory forms closed orbits, analogous to the harmonic oscillator systems studied in [1]. Energy conservation is enforced through:

$$\frac{d\mathcal{H}}{dt} = 0$$

### 2.2.6 Component Interaction Models

**ComponentInteractionModel.** Heat and mass transfer coupling between components is modeled through interaction strengths:

$$\alpha_{\text{thermal}} = \frac{\dot{m} c_p \Delta T}{Q_{\text{ref}}}$$

$$\alpha_{\text{mass}} = \frac{\dot{m}}{\dot{m}_{\text{nom}}} \cdot f_{\text{PLR}}$$

Network stability margin quantifies the distance from instability in the coupled component system:

$$\sigma_{\text{stability}} = \min_i \text{Re}(\lambda_i)$$

where $\lambda_i$ are eigenvalues of the linearized system matrix.

**GNNTopologyGenerator.** The heat pump system is represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with 10 nodes (compressor, condenser, evaporator, expansion valve, refrigerant circuits, control system, ambient, building, system health) and 11 edges representing refrigerant flow paths and thermal couplings. Node features (35 per node) encode component states; edge features (12 per edge) encode flow rates and temperature differences.

### 2.2.7 Part-Load and Cycling Models

**PartLoadCyclingModel.** Part-load efficiency degradation follows AHRI Standard 210/240 [15]:

$$\eta_{\text{PLR}} = \begin{cases} 1 - 0.2(1 - \text{PLR})^3 + 0.1(1 - \text{PLR}) & \text{variable speed} \\ 1 - 0.4(1 - \text{PLR})^2 - 0.2(1 - \text{PLR}) & \text{single speed} \end{cases}$$

Cycling losses for on-off operation:

$$\eta_{\text{cycling}} = 1 - C_D(1 - f_{\text{on}})$$

where $C_D \approx 0.25$ is the cycling degradation coefficient and $f_{\text{on}}$ is runtime fraction.

### 2.2.8 Temporal Dynamics Models

**TemporalDynamicsGenerator.** Startup transients follow first-order dynamics:

$$X(t) = X_{\infty}(1 - e^{-t/\tau})$$

where $\tau$ is the thermal time constant. The model generates 100-timestep trajectories for temperature, pressure, mass flow, and heat transfer rates, providing the temporal sequences for HNN training.

### 2.2.9 Sensor and Measurement Models

**SensorNoiseModel.** Realistic measurement uncertainty is added to all sensor readings:

$$T_{\text{measured}} = T_{\text{actual}} + \mathcal{N}(0, \sigma_T^2)$$

$$P_{\text{measured}} = P_{\text{actual}} \cdot (1 + \mathcal{N}(0, \sigma_P^2))$$

where $\sigma_T = 0.5$ K for temperature and $\sigma_P = 0.015$ for pressure (1.5% uncertainty).

### 2.2.10 Component Performance Models

**CompressorModel.** Compressor performance maps relate isentropic efficiency to pressure ratio and speed:

$$\eta_{\text{isen}} = \eta_0 - \alpha_1(r_p - r_{p,0})^2 - \alpha_2(N - N_0)^2$$

where $r_p = P_{\text{cond}}/P_{\text{evap}}$ is pressure ratio and $N$ is compressor speed.

**HeatExchangerModel.** Heat exchanger effectiveness is computed using the $\varepsilon$-NTU method [16]:

$$\varepsilon = 1 - \exp\left[-\frac{1}{C_r}\left(1 - \exp(-C_r \cdot \text{NTU})\right)\right]$$

where $C_r = C_{\min}/C_{\max}$ is the capacity ratio and $\text{NTU} = UA/C_{\min}$.

## 2.3 COP Calculation Methodology

The coefficient of performance is computed by aggregating outputs from all 17 physics models. The base thermodynamic COP is:

$$\text{COP}_{\text{heating,base}} = \frac{\dot{Q}_{\text{cond}}}{\dot{W}_{\text{comp}}} = \frac{\dot{m}(h_2 - h_3)}{\dot{m}(h_2 - h_1)}$$

$$\text{COP}_{\text{cooling,base}} = \frac{\dot{Q}_{\text{evap}}}{\dot{W}_{\text{comp}}} = \frac{\dot{m}(h_1 - h_4)}{\dot{m}(h_2 - h_1)}$$

The final COP incorporates all degradation factors:

$$\text{COP}_{\text{actual}} = \text{COP}_{\text{base}} \cdot f_{\text{charge}} \cdot f_{\text{fault}} \cdot (1 - \eta_{\text{degrad}}) \cdot f_{\text{ambient}} \cdot \eta_{\text{PLR}} \cdot (1 - \ell_{\text{cycling}})$$

where each factor corresponds to outputs from the respective physics model.

**Thermodynamic Validation.** All generated cycles must satisfy:

1. **Energy balance:** $|\dot{Q}_{\text{cond}} - (\dot{Q}_{\text{evap}} + \dot{W}_{\text{comp}})| < 0.05 \dot{Q}_{\text{cond}}$

2. **Carnot limit:** $\text{COP} \leq \text{COP}_{\text{Carnot}} = T_H / (T_H - T_C)$

3. **Entropy generation:** $\dot{S}_{\text{gen}} = \dot{Q}_{\text{cond}}/T_H - \dot{Q}_{\text{evap}}/T_C \geq 0$

4. **Mass conservation:** $\dot{m}_{\text{in}} = \dot{m}_{\text{out}}$ at all components

Cycles failing any validation check are rejected, ensuring thermodynamic consistency of the training dataset.

## 2.4 Dataset Generation and Distribution Filtering

To achieve a representative COP distribution for model training, we employ probabilistic acceptance filtering rather than input manipulation. The COPDistributionFilter implements rejection sampling to shape the output distribution toward a target normal distribution $\mathcal{N}(\mu_{\text{target}}, \sigma_{\text{target}}^2)$ while preserving the physical relationships between inputs and outputs.

The acceptance probability for a cycle with COP value $c$ falling in histogram bin $i$ is:

$$P_{\text{accept}}(c) = \begin{cases} \min(1, \sqrt{p_{\text{target},i} / p_{\text{current},i}}) & p_{\text{current},i} < p_{\text{target},i} \\ (p_{\text{target},i} / p_{\text{current},i})^\gamma & p_{\text{current},i} \geq p_{\text{target},i} \end{cases}$$

where $\gamma = 3$ controls filter aggressiveness.

**Final Dataset Characteristics.** The generated dataset contains 100,000 validated cycles with:
- HNN temporal features: 100 timesteps × 23 channels = 2,300 features
- GNN structural features: 10 nodes × 35 features + 11 edges × 12 features + 50 graph features = 532 features
- Total features per cycle: 2,832
- COP distribution: $\mu = 4.53$, $\sigma = 1.57$, range [1.8, 8.0]
- Carnot efficiency range: 50–85%

---

## LaTeX TABLES AND FIGURES

### Table 1: Physics Models Summary
**Placement: After Section 2.2 introduction, before 2.2.1**

```latex
\begin{table}[t]
\caption{Summary of 17 integrated physics models in the digital twin.}
\label{tab:physics_models}
\centering
\small
\begin{tabular}{@{}llp{4.5cm}@{}}
\toprule
\textbf{Category} & \textbf{Model} & \textbf{Key Outputs} \\
\midrule
\multirow{2}{*}{Thermodynamic} & RefrigerantProperties & $P_{\text{sat}}$, $h$, $s$, $\rho$ \\
& CycleGenerator & State points 1-4, $\dot{m}$ \\
\midrule
\multirow{4}{*}{Degradation} & HeatExchangerFoulingModel & $R_f$, $\Delta\varepsilon$ \\
& CompressorWearModel & $W_b$, $L_v$, $\eta_{\text{degrad}}$ \\
& RefrigerantChargeModel & $f_{\text{COP,charge}}$, $\Delta T_{\text{sc}}$ \\
& FaultInjectionModel & $f_{\text{COP,fault}}$, fault type \\
\midrule
\multirow{2}{*}{Environmental} & AmbientConditionModel & $T_{\text{wb}}$, $\rho_{\text{air}}$, $r_{\text{frost}}$ \\
& MultiZoneBuildingModel & $Q_{\text{load}}$, zone temperatures \\
\midrule
Control & ControlSystemModel & $N_{\text{comp}}$, EXV$_{\%}$, PID states \\
\midrule
\multirow{2}{*}{Hamiltonian} & HamiltonianMechanicsModel & $\mathcal{H}$, $\mathbf{q}$, $\mathbf{p}$ trajectories \\
& ComponentInteractionModel & $\alpha_{\text{thermal}}$, $\alpha_{\text{mass}}$ \\
\midrule
\multirow{2}{*}{Part-Load} & PartLoadCyclingModel & $\eta_{\text{PLR}}$, $\ell_{\text{cycling}}$ \\
& TemporalDynamicsGenerator & 100-step transients \\
\midrule
Measurement & SensorNoiseModel & Noisy $T$, $P$ measurements \\
\midrule
\multirow{3}{*}{Component} & CompressorModel & $\eta_{\text{isen}}(r_p, N)$ \\
& HeatExchangerModel & $\varepsilon$, NTU, UA \\
& GNNTopologyGenerator & Node/edge features \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2: Feature Dimensions
**Placement: At the end of Section 2.4**

```latex
\begin{table}[t]
\caption{Feature dimensions for HNN-GNN model inputs.}
\label{tab:features}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Feature Type} & \textbf{Dimensions} & \textbf{Total} \\
\midrule
HNN Temporal & 100 timesteps $\times$ 23 channels & 2,300 \\
GNN Node Features & 10 nodes $\times$ 35 features & 350 \\
GNN Edge Features & 11 edges $\times$ 12 features & 132 \\
GNN Graph Features & 1 $\times$ 50 features & 50 \\
\midrule
\textbf{Total per cycle} & & \textbf{2,832} \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 3: Dataset Statistics
**Placement: At the end of Section 2.4**

```latex
\begin{table}[t]
\caption{Generated dataset statistics.}
\label{tab:dataset_stats}
\centering
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total cycles & 100,000 \\
Physics-valid cycles & 100,000 (filtered) \\
COP mean ($\mu$) & 4.53 \\
COP std ($\sigma$) & 1.57 \\
COP range & [1.8, 8.0] \\
Carnot efficiency range & [0.50, 0.85] \\
Refrigerants & R410A, R134a, R290 \\
Operating modes & Heating, Cooling \\
Ambient temperature range & 253–318 K \\
System age range & 0–50,000 hours \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure 1: Four-Layer Architecture
**Placement: After first paragraph of Section 2.1**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/architecture_4layer.pdf}
\caption{Four-layer physics-based digital twin architecture. Layer 1 samples operating conditions, Layer 2 executes 17 physics models, Layer 3 computes derived performance metrics, and Layer 4 produces validated COP targets.}
\label{fig:architecture}
\end{figure}
```

### Figure 2: Hamiltonian Phase Space
**Placement: In Section 2.2.5 after Hamiltonian equations**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\columnwidth]{figures/hamiltonian_phase_space.pdf}
\caption{Phase space trajectory $(q, p)$ for a thermodynamic cycle showing closed orbits characteristic of energy-conserving Hamiltonian systems. Colors indicate cycle progression: evaporation (blue) $\rightarrow$ compression (red) $\rightarrow$ condensation (orange) $\rightarrow$ expansion (green).}
\label{fig:phase_space}
\end{figure}
```

### Figure 3: COP Distribution
**Placement: At the end of Section 2.4**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/cop_distribution.pdf}
\caption{COP distribution of the generated dataset. (a) Histogram showing approximately normal distribution ($\mu=4.53$, $\sigma=1.57$). (b) COP vs. Carnot efficiency demonstrating all cycles satisfy $\text{COP} < \text{COP}_{\text{Carnot}}$.}
\label{fig:cop_dist}
\end{figure}
```

---

## Sample Dataset Demonstration

A representative cycle from the dataset demonstrates the physics model outputs:

**Layer 1 Inputs (sample):**
- Ambient: $T_{\text{ambient}} = 283.2$ K, $\phi = 0.72$, $v_{\text{wind}} = 3.4$ m/s
- Building: $Q_{\text{load}} = 8.5$ kW, $f_{\text{occupancy}} = 0.65$
- System: Refrigerant = R410A, Design capacity = 12 kW, Age = 15,000 hours
- Mode: Heating, PLR = 0.75

**Layer 2 Physics Outputs (sample):**
- Cycle: $T_{\text{evap}} = 268.3$ K, $T_{\text{cond}} = 323.1$ K, $P_{\text{evap}} = 652$ kPa, $P_{\text{cond}} = 2,105$ kPa
- Degradation: Evap. fouling = 0.00032 m²K/W, Bearing wear = 12.5%, Charge = 93.5%
- Control: Compressor speed = 72.3%, EXV opening = 43.8%
- Hamiltonian: $\mathcal{H} = 4,850$ J, Energy drift = 0.0012

**Layer 3 Derived Features (sample):**
- $\dot{Q}_{\text{evap}} = 10.2$ kW, $\dot{Q}_{\text{cond}} = 12.8$ kW, $\dot{W}_{\text{comp}} = 2.6$ kW
- $\text{COP}_{\text{Carnot}} = 5.89$, $\eta_{\text{Carnot}} = 0.68$

**Layer 4 Targets (sample):**
- $\text{COP}_{\text{heating}} = 3.99$
- Energy balance error: 0.02%
- Carnot compliance: ✓
- Physically valid: ✓
