#!/usr/bin/env python3
"""
Generate PDF for Section 3: Neural Network Surrogate Model
"""

from fpdf import FPDF
from pathlib import Path

class Section3PDF(FPDF):
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
    
    def figure_placeholder(self, caption):
        self.ln(4)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(200, 200, 200)
        self.rect(self.get_x(), self.get_y(), 180, 40, 'DF')
        self.set_font('Helvetica', 'B', 10)
        self.cell(180, 20, '[INSERT YOUR ARCHITECTURE DIAGRAM HERE]', align='C')
        self.ln(30)
        self.set_font('Helvetica', 'I', 9)
        self.multi_cell(0, 4, caption)
        self.ln(4)

# Create PDF
pdf = Section3PDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 16)
pdf.cell(0, 10, 'Section 3: Neural Network Surrogate Model', align='C')
pdf.ln(12)

# Introduction
pdf.body_text(
    "This section presents our hybrid neural network architecture that combines Hamiltonian "
    "Neural Networks (HNN) for temporal dynamics with Graph Neural Networks (GNN) for component "
    "interactions. The surrogate model learns to approximate the physics-based digital twin at "
    "significantly reduced computational cost while preserving thermodynamic constraints."
)

# 3.1 Motivation
pdf.section_title('3.1 Motivation for Neural Surrogate', 1)

pdf.body_text(
    "While our physics-based digital twin provides accurate COP predictions through rigorous "
    "thermodynamic calculations, its computational requirements limit real-time applications. "
    "The complete physics simulation—executing 17 models across 100 timesteps—requires 100-500ms "
    "per prediction, depending on operating conditions and convergence characteristics."
)

pdf.body_text(
    "Real-time heat pump monitoring and control demand inference latency below 10ms to enable "
    "rapid fault detection, adaptive setpoint optimization, and predictive maintenance triggers. "
    "This 10-50x speedup requirement motivates the development of a neural network surrogate that "
    "learns to approximate the physics while maintaining thermodynamic consistency."
)

pdf.body_text(
    "The key insight is that while individual physics calculations are expensive, the underlying "
    "relationships between operating conditions and COP follow smooth, learnable manifolds. A "
    "well-designed neural network can capture these relationships through gradient-based "
    "optimization on the physics-generated dataset, achieving similar accuracy at a fraction of "
    "the computational cost."
)

pdf.bold_start("Design Requirements: ")
pdf.body_text(
    "1) Inference latency < 10ms for real-time control; "
    "2) Mean Absolute Error (MAE) < 0.15 COP units; "
    "3) 100% Carnot limit compliance (no thermodynamically invalid predictions); "
    "4) Energy conservation preservation through architectural constraints; "
    "5) Interpretable intermediate representations connecting to physical quantities."
)

# 3.2 HNN Component
pdf.section_title('3.2 HNN Component (Temporal Dynamics)', 1)

pdf.body_text(
    "The Hamiltonian Neural Network branch processes the temporal evolution of thermodynamic "
    "states over the 100-timestep cycle. Unlike standard recurrent networks that may violate "
    "energy conservation, HNNs are architecturally constrained to preserve the Hamiltonian "
    "structure, ensuring physically consistent dynamics."
)

pdf.bold_start("Input Representation. ")
pdf.body_text(
    "The HNN receives a tensor of shape [batch_size, 100, 23] representing 23 temporal channels "
    "over 100 timesteps. These channels include:"
)
pdf.body_text(
    "- Thermodynamic trajectories: T_evap(t), T_cond(t), P_evap(t), P_cond(t), m_dot(t)\n"
    "- Heat transfer rates: Q_evap(t), Q_cond(t), W_comp(t)\n"
    "- Phase space coordinates: q_T(t), q_P(t), q_V(t), q_S(t), p_T(t), p_P(t), p_V(t), p_S(t)\n"
    "- Control signals: N_comp(t), EXV(t), fan_evap(t), fan_cond(t)\n"
    "- Derived quantities: superheat(t), subcooling(t), H(t)"
)

pdf.bold_start("Hamiltonian Formulation. ")
pdf.body_text(
    "The core innovation is modeling heat pump dynamics in Hamiltonian phase space. The network "
    "learns the Hamiltonian function H(q, p) that governs system evolution:"
)
pdf.equation("H(q, p) = T(p) + V(q) = (1/2)p^T M^(-1) p + U(q)", "38")
pdf.body_text(
    "where q = [q_T, q_P, q_V, q_S]^T are generalized coordinates (temperature, pressure, volume, "
    "entropy), p = [p_T, p_P, p_V, p_S]^T are conjugate momenta, T(p) is kinetic energy, and V(q) "
    "is potential energy. The time evolution follows Hamilton's equations:"
)
pdf.equation("dq/dt = dH/dp,  dp/dt = -dH/dq", "39")

pdf.bold_start("Architecture. ")
pdf.body_text(
    "The HNN branch consists of three components:"
)
pdf.body_text(
    "1) Input Encoder: Projects the 23-dimensional input to an 8-dimensional phase space embedding "
    "through a linear layer followed by layer normalization.\n\n"
    "2) Hamiltonian Network: A multi-layer perceptron with hidden dimensions [128, 256, 128] that "
    "learns the scalar Hamiltonian function H(q, p). The network uses Tanh activations to ensure "
    "smooth gradients for the symplectic structure.\n\n"
    "3) Symplectic Integrator: Computes phase space evolution using automatic differentiation of "
    "the learned Hamiltonian. The gradients dH/dq and dH/dp are computed via backpropagation, "
    "ensuring the dynamics exactly satisfy Hamilton's equations."
)

pdf.bold_start("Energy Conservation Guarantee. ")
pdf.body_text(
    "The key property of HNNs is that they exactly conserve the learned Hamiltonian along "
    "trajectories (dH/dt = 0), preventing energy drift that plagues standard neural network "
    "approaches. This is achieved by computing dynamics from gradients of a scalar energy "
    "function rather than directly predicting state updates."
)

pdf.body_text(
    "The HNN branch outputs: (1) phase space trajectory [batch_size, 100, 8], (2) Hamiltonian "
    "values [batch_size, 100, 1], and (3) a temporal embedding for fusion."
)

# 3.3 GNN Component
pdf.section_title('3.3 GNN Component (System Topology)', 1)

pdf.body_text(
    "The Graph Neural Network branch processes the static component structure and interactions "
    "of the heat pump system. While the HNN captures how the system evolves in time, the GNN "
    "captures how components relate in space—heat and mass flow paths, thermal coupling, and "
    "control dependencies."
)

pdf.bold_start("Graph Representation. ")
pdf.body_text(
    "The heat pump system is represented as a directed graph G = (V, E) with 10 nodes and "
    "11 edges:"
)
pdf.body_text(
    "Nodes (V): Compressor, Condenser, Evaporator, Expansion valve, Refrigerant circuit (high), "
    "Refrigerant circuit (low), Control system, Ambient environment, Building/load, System health."
)
pdf.body_text(
    "Edges (E): Represent refrigerant flow paths (compressor->condenser->valve->evaporator->compressor) "
    "and thermal coupling paths (ambient<->evaporator, building<->condenser, control->components)."
)

pdf.bold_start("Node Features. ")
pdf.body_text(
    "Each node has 35 features encoding component state:"
)
pdf.body_text(
    "- Compressor: discharge T, P, efficiency, wear, vibration, speed, Hamiltonian energy\n"
    "- Heat exchangers: T, P, effectiveness, fouling resistance, heat rate, fan speed\n"
    "- Expansion valve: position, superheat control, pressure drop\n"
    "- Refrigerant: mass flow, charge level, COP factor\n"
    "- Environment: T_ambient, humidity, wet-bulb, dew point, density, wind, solar\n"
    "- Building: load, occupancy, internal gains, load match factor\n"
    "- System health: age, efficiency loss, capacity loss, fault status"
)

pdf.bold_start("Edge Features. ")
pdf.body_text(
    "Each edge has 12 features encoding flow and coupling:"
)
pdf.body_text(
    "- Mass flow rate (refrigerant flux)\n"
    "- Pressure ratio across edge\n"
    "- Temperature difference\n"
    "- Thermal coupling strength\n"
    "- Mass coupling strength\n"
    "- Component interaction coefficients"
)

pdf.bold_start("Architecture. ")
pdf.body_text(
    "The GNN branch uses a message-passing architecture with 4 layers:"
)
pdf.body_text(
    "1) Embedding Layers: Linear projections from node (35->128), edge (12->128), and "
    "graph (50->128) features to hidden dimension.\n\n"
    "2) NNConv Layers: Edge-conditioned graph convolutions that aggregate neighbor information "
    "weighted by edge features. Each layer updates node embeddings:\n"
)
pdf.equation("h_i^(l+1) = MLP(h_i^(l) + sum_j e_ij * W * h_j^(l))", "40")
pdf.body_text(
    "where h_i is node embedding, e_ij is edge feature, and W is learned weight.\n\n"
    "3) Graph Attention: Multi-head attention (4 heads) allows nodes to selectively attend to "
    "neighbors based on learned relevance.\n\n"
    "4) Global Pooling: Aggregates node embeddings to graph-level representation using mean, "
    "max, and sum pooling, concatenated with graph features."
)

pdf.bold_start("Physics Constraint: Mass and Energy Balance at Nodes. ")
pdf.body_text(
    "The GNN architecture implicitly enforces conservation laws through message passing—the "
    "aggregation of incoming and outgoing edge messages at each node naturally represents "
    "mass and energy balance. Additionally, the graph consistency loss explicitly penalizes "
    "violations of conservation at internal nodes."
)

# 3.4 Hybrid Architecture
pdf.section_title('3.4 Hybrid Architecture', 1)

pdf.body_text(
    "The hybrid HNN-GNN architecture combines temporal dynamics (HNN) with spatial structure "
    "(GNN) through an attention-based fusion mechanism. This dual-pathway design captures both "
    "how the system evolves over a cycle and how components interact within the system topology."
)

# Architecture diagram placeholder
pdf.figure_placeholder(
    "Figure X: Hybrid HNN-GNN Architecture. Left: HNN branch processes 100-timestep temporal "
    "sequences through Hamiltonian dynamics. Center: GNN branch processes 10-node component graph "
    "through message passing. Right: Multi-head attention fusion combines embeddings for COP "
    "prediction. Physics constraints are enforced at multiple levels."
)

pdf.bold_start("Fusion Layer. ")
pdf.body_text(
    "The fusion network combines HNN and GNN outputs using multi-head cross-attention (4 heads):"
)
pdf.body_text(
    "1) Projection: HNN output (9-dim: 8 phase space + 1 Hamiltonian) and GNN output (512-dim: "
    "3x128 pooling + 128 graph) are projected to common dimension (256).\n\n"
    "2) Cross-Attention: The HNN sequence attends to GNN features, learning which component "
    "interactions are most relevant at each timestep:\n"
)
pdf.equation("Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V", "41")
pdf.body_text(
    "where Q comes from HNN, K and V from GNN.\n\n"
    "3) Residual Fusion: Attended features are concatenated with HNN projection and processed "
    "through MLP layers [512->256, 256->256] with ReLU activations and dropout (0.1).\n\n"
    "4) Layer Normalization: Stabilizes training and enables residual connections."
)

pdf.bold_start("COP Prediction Head. ")
pdf.body_text(
    "The final prediction head processes fused features through an MLP [256->128->64->1] "
    "with physics constraints:"
)
pdf.body_text(
    "- Hard constraint: Output clamped to [1.5, 8.0] (physically realistic COP range)\n"
    "- Soft constraint: Carnot limit penalty in loss function\n"
    "- Output: COP prediction per timestep [batch_size, 100, 1], with final cycle COP as mean"
)

pdf.bold_start("Model Summary. ")
pdf.body_text(
    "Total parameters: ~2.5M, distributed across HNN branch (~800K), GNN branch (~1.2M), "
    "fusion layer (~400K), and prediction head (~100K). Inference time: ~5ms on GPU, ~15ms on CPU."
)

# 3.5 Physics-Informed Loss
pdf.section_title('3.5 Physics-Informed Loss Function', 1)

pdf.body_text(
    "Training the hybrid model requires a loss function that balances data fitting with "
    "physics constraint satisfaction. We design a multi-component loss that penalizes "
    "thermodynamic violations while maintaining prediction accuracy."
)

pdf.bold_start("Total Loss Formulation. ")
pdf.body_text("The total loss combines five components:")
pdf.equation("L = lambda_data * L_data + lambda_H * L_hamiltonian + lambda_thermo * L_thermo + lambda_HP * L_heatpump + lambda_G * L_graph", "42")

pdf.bold_start("1) Data Fitting Loss (L_data). ")
pdf.body_text("Mean squared error between predicted and true COP:")
pdf.equation("L_data = (1/N) sum_i (COP_pred,i - COP_true,i)^2", "43")

pdf.bold_start("2) Hamiltonian Conservation Loss (L_hamiltonian). ")
pdf.body_text("Penalizes energy drift along trajectories:")
pdf.equation("L_hamiltonian = (1/T) sum_t (dH/dt)^2 = (1/T) sum_t ((H(t+1) - H(t)) / dt)^2", "44")
pdf.body_text("This enforces dH/dt approx 0, the defining property of Hamiltonian systems.")

pdf.bold_start("3) Thermodynamic Constraints Loss (L_thermo). ")
pdf.body_text("Enforces fundamental thermodynamic laws:")
pdf.equation("L_thermo = L_entropy + L_energy_balance + L_mass_balance + L_bounds", "45")
pdf.body_text(
    "- Entropy: Penalizes Second Law violations (dS/dt < 0)\n"
    "- Energy balance: sum(Q_in) = sum(Q_out) + sum(W)\n"
    "- Mass balance: m_dot_in = m_dot_out at all components\n"
    "- Bounds: Temperature in [200, 500]K, pressure in [0.1, 50] bar"
)

pdf.bold_start("4) Heat Pump Physics Loss (L_heatpump). ")
pdf.body_text("Enforces heat pump-specific constraints:")
pdf.equation("L_HP = L_cop_def + L_carnot + L_deltaT", "46")
pdf.body_text(
    "- COP definition: COP = Q_cond / W_comp (predicted COP matches computed)\n"
    "- Carnot limit: COP <= eta_carnot * T_cond / (T_cond - T_evap)\n"
    "- Temperature lift: T_cond - T_evap >= 10K (minimum thermodynamic lift)"
)

pdf.bold_start("5) Graph Consistency Loss (L_graph). ")
pdf.body_text("Enforces node-edge relationships:")
pdf.equation("L_graph = L_temp_consistency + L_flow_conservation", "47")
pdf.body_text(
    "- Temperature consistency: Delta_T along edge matches node temperatures\n"
    "- Flow conservation: sum(m_in) = sum(m_out) at internal nodes"
)

pdf.bold_start("Loss Weights. ")
pdf.body_text("Default weights (tuned via validation):")
pdf.body_text(
    "lambda_data = 1.0, lambda_H = 1.0, lambda_thermo = 0.5, lambda_HP = 0.8, lambda_G = 0.3"
)
pdf.body_text(
    "The relatively high weight on Hamiltonian loss ensures energy conservation is prioritized, "
    "while heat pump physics loss ensures COP predictions respect thermodynamic limits."
)

# Table: Architecture Summary
pdf.add_page()
pdf.section_title('Architecture Summary', 1)

pdf.set_font('Helvetica', 'B', 10)
pdf.multi_cell(0, 6, "Table 5: Hybrid HNN-GNN Architecture Specifications")
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(60, 6, "Component", border=1, fill=True)
pdf.cell(60, 6, "Specification", border=1, fill=True)
pdf.cell(60, 6, "Dimension", border=1, fill=True)
pdf.ln()

specs = [
    ("HNN Input", "Temporal features", "100 x 23"),
    ("HNN Phase Space", "Generalized coordinates", "8-dim (q: 4, p: 4)"),
    ("HNN Hidden Layers", "MLP dimensions", "[128, 256, 128]"),
    ("HNN Output", "Phase space + Hamiltonian", "100 x 9"),
    ("GNN Input (Nodes)", "Component features", "10 x 35"),
    ("GNN Input (Edges)", "Flow/coupling features", "11 x 12"),
    ("GNN Input (Graph)", "Global context", "50"),
    ("GNN Hidden Dim", "Node embedding", "128"),
    ("GNN Layers", "Message passing", "4"),
    ("GNN Attention", "Multi-head", "4 heads"),
    ("GNN Output", "Graph embedding", "512 (3x128 + 128)"),
    ("Fusion Dim", "Common space", "256"),
    ("Fusion Attention", "Cross-attention", "4 heads"),
    ("Prediction MLP", "Output layers", "[256, 128, 64, 1]"),
    ("Total Parameters", "Trainable weights", "~2.5M"),
    ("Inference Time", "GPU / CPU", "5ms / 15ms"),
]

pdf.set_font('Helvetica', '', 8)
for comp, spec, dim in specs:
    pdf.cell(60, 5, comp, border=1)
    pdf.cell(60, 5, spec, border=1)
    pdf.cell(60, 5, dim, border=1)
    pdf.ln()

pdf.ln(6)

# Table: Loss Components
pdf.set_font('Helvetica', 'B', 10)
pdf.multi_cell(0, 6, "Table 6: Physics-Informed Loss Components")
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(45, 6, "Loss Component", border=1, fill=True)
pdf.cell(70, 6, "Physics Principle", border=1, fill=True)
pdf.cell(20, 6, "Weight", border=1, fill=True)
pdf.cell(45, 6, "Equation", border=1, fill=True)
pdf.ln()

losses = [
    ("L_data", "Data fitting (MSE)", "1.0", "Eq. (43)"),
    ("L_hamiltonian", "Energy conservation", "1.0", "Eq. (44)"),
    ("L_thermo", "Thermodynamic laws", "0.5", "Eq. (45)"),
    ("L_heatpump", "COP/Carnot bounds", "0.8", "Eq. (46)"),
    ("L_graph", "Node-edge consistency", "0.3", "Eq. (47)"),
]

pdf.set_font('Helvetica', '', 8)
for comp, principle, weight, eq in losses:
    pdf.cell(45, 5, comp, border=1)
    pdf.cell(70, 5, principle, border=1)
    pdf.cell(20, 5, weight, border=1)
    pdf.cell(45, 5, eq, border=1)
    pdf.ln()

# Save
output_path = Path(__file__).parent / "section3_neural_network.pdf"
pdf.output(str(output_path))
print(f"PDF generated: {output_path}")
