#!/usr/bin/env python3
"""
Generate PDF for Section 2.5: Dataset Visualization and Analysis
"""

from fpdf import FPDF
from pathlib import Path

class DatasetPDF(FPDF):
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
    
    def figure_placeholder(self, fig_name, caption):
        self.ln(4)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(200, 200, 200)
        self.rect(self.get_x(), self.get_y(), 180, 30, 'DF')
        self.set_font('Helvetica', 'B', 10)
        self.cell(180, 15, f'[INSERT FIGURE: {fig_name}]', align='C')
        self.ln(20)
        self.set_font('Helvetica', 'I', 9)
        self.multi_cell(0, 4, caption)
        self.ln(4)
    
    def bold_start(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.write(5, text)
        self.set_font('Helvetica', '', 10)

# Create PDF
pdf = DatasetPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 16)
pdf.cell(0, 10, 'Section 2.5: Dataset Visualization and Analysis', align='C')
pdf.ln(12)

pdf.body_text(
    "This section presents comprehensive visualizations of the generated physics-based "
    "dataset, demonstrating thermodynamic consistency, Hamiltonian conservation, and "
    "statistical characteristics suitable for training the HNN-GNN model."
)

# 2.5.1 Hamiltonian Phase Space
pdf.section_title('2.5.1 Hamiltonian Phase Space Representation', 2)

pdf.figure_placeholder(
    'fig1_hamiltonian_phase_space.png',
    "Figure X: Hamiltonian Phase Space Analysis. (a) Heat pump cycles in generalized (q, p) "
    "coordinates showing closed orbits. Green=start, red=end points. (b) Hamiltonian energy "
    "H(t) demonstrating conservation (dH/dt ~ 0). (c) Kinetic-potential energy exchange."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "The phase space representation transforms the vapor compression cycle into a form suitable "
    "for Hamiltonian Neural Network processing. The closed orbits demonstrate that our "
    "generalized coordinate formulation correctly captures the periodic nature of the "
    "thermodynamic cycle - analogous to harmonic oscillator trajectories studied in the "
    "original HNN work [Greydanus et al., 2019]. The Hamiltonian conservation plot validates "
    "that H(t)/H(0) remains close to unity throughout each cycle. The energy component plot "
    "shows the characteristic anti-phase relationship between kinetic and potential modes."
)

# 2.5.2 Thermodynamic Diagrams
pdf.section_title('2.5.2 Thermodynamic State Diagrams', 2)

pdf.figure_placeholder(
    'fig2_thermodynamic_phase_diagrams.png',
    "Figure X+1: Thermodynamic Phase Diagrams. (a) T-s diagram showing cycles colored by "
    "COP (red=low, green=high). (b) P-h diagram with same COP color mapping."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "These classical thermodynamic diagrams validate that generated cycles follow physically "
    "correct vapor compression behavior. In the T-s diagram, higher-COP cycles (green) exhibit "
    "smaller temperature lifts and tighter cycle areas, reflecting lower irreversibility. The "
    "P-h diagram shows the expected quasi-rectangular shape with clear distinction between "
    "evaporation, compression, condensation, and expansion processes. The COP gradient across "
    "cycles confirms: smaller pressure ratios and temperature lifts yield higher efficiency."
)

# 2.5.3 COP Distribution
pdf.section_title('2.5.3 COP Distribution and Carnot Compliance', 2)

pdf.figure_placeholder(
    'fig3_cop_distribution_analysis.png',
    "Figure X+2: COP Analysis. (a) COP histogram (mu=4.68). (b) COP vs Carnot limit - all "
    "points below diagonal confirm thermodynamic validity. (c) Carnot efficiency distribution "
    "(mean=60%). (d) COP vs temperature lift colored by efficiency."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "The COP distribution demonstrates successful application of the distribution filter, "
    "achieving a roughly Gaussian shape suitable for balanced model training. Critically, "
    "the COP vs Carnot plot verifies that every generated cycle satisfies the Carnot limit - "
    "no points appear above the diagonal, confirming 100% thermodynamic compliance. Carnot "
    "efficiencies range from 25% to 95% with mean around 60%, matching published field data. "
    "The COP-temperature lift relationship shows the expected inverse correlation: low-lift "
    "conditions achieve COP > 6, while high-lift conditions drop below COP = 3."
)

# 2.5.4 Energy Landscape
pdf.section_title('2.5.4 Hamiltonian Energy Landscape', 2)

pdf.figure_placeholder(
    'fig4_hamiltonian_contour.png',
    "Figure X+3: Hamiltonian Energy Landscape. (a) Phase space colored by H energy. "
    "(b) Energy contours showing level sets that constrain trajectory evolution."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "The energy landscape visualization provides insight into the Hamiltonian structure learned "
    "by the HNN. Energy levels organize concentrically around the phase space origin, with "
    "higher-energy cycles exhibiting larger oscillation amplitudes. The contour plot presents "
    "the energy level sets that constrain trajectory evolution - in a perfectly conservative "
    "system, trajectories must remain on constant-H contours. This structure enables the HNN "
    "to learn dynamics that automatically preserve energy."
)

# 2.5.5 Energy Conservation
pdf.section_title('2.5.5 Energy Conservation Validation', 2)

pdf.figure_placeholder(
    'fig5_energy_conservation.png',
    "Figure X+4: Energy Conservation. (a) Normalized H(t)/H(0) for 10 cycles. (b) dH/dt "
    "histogram centered at zero. (c) First Law error at machine precision. (d) Energy flow."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "These plots validate thermodynamic consistency at multiple levels. Individual cycles "
    "maintain approximately constant Hamiltonian energy, with variations due to realistic "
    "system dynamics and numerical integration. The dH/dt histogram shows tight distribution "
    "around zero as required for conservative systems. The First Law energy balance "
    "(Q_cond = Q_evap + W_comp) is satisfied to machine precision (~10^-18 kW). The energy "
    "flow diagram illustrates typical dynamics during a heat pump cycle."
)

# 2.5.6 Dataset Summary
pdf.section_title('2.5.6 Dataset Summary Statistics', 2)

pdf.figure_placeholder(
    'fig7_dataset_statistics.png',
    "Figure X+5: Dataset Summary. COP distribution with N(4.68, 1.37) fit, temperature and "
    "pressure ranges, Carnot efficiency vs lift, statistics box, and feature correlations."
)

pdf.bold_start("Discussion: ")
pdf.body_text(
    "This summary figure provides complete statistical characterization. The correlation matrix "
    "reveals expected physical relationships: COP correlates positively with T_evap (+0.38) and "
    "Carnot efficiency (+0.36), while correlating negatively with T_cond (-0.37). Temperature "
    "and pressure box plots confirm realistic operating ranges matching commercial heat pumps."
)

# Comprehensive Statistics Table
pdf.add_page()
pdf.section_title('Table 4: Comprehensive Dataset Statistics', 1)

# Create table
pdf.set_font('Helvetica', 'B', 9)
pdf.set_fill_color(240, 240, 240)

# Header
pdf.cell(60, 7, 'Category', border=1, fill=True)
pdf.cell(60, 7, 'Parameter', border=1, fill=True)
pdf.cell(60, 7, 'Value', border=1, fill=True)
pdf.ln()

pdf.set_font('Helvetica', '', 8)

# Data rows
stats_data = [
    ('Dataset Size', 'Total cycles', '100,000'),
    ('', 'Timesteps per cycle', '100'),
    ('', 'Total data points', '10,000,000'),
    ('COP Statistics', 'Mean', '4.68'),
    ('', 'Standard deviation', '1.37'),
    ('', 'Minimum', '1.80'),
    ('', 'Maximum', '8.00'),
    ('', 'Median', '4.61'),
    ('', '25th percentile', '3.62'),
    ('', '75th percentile', '5.68'),
    ('Carnot Efficiency', 'Mean', '60%'),
    ('', 'Std deviation', '15%'),
    ('', 'Range', '25% - 95%'),
    ('Temperature Ranges', 'T_evap', '233-288 K (-40C to +15C)'),
    ('', 'T_cond', '298-368 K (+25C to +95C)'),
    ('', 'T_ambient', '253-318 K (-20C to +45C)'),
    ('', 'Temperature lift', '15-105 K'),
    ('Pressure Ranges', 'P_evap', '0.2-1.0 MPa'),
    ('', 'P_cond', '1.0-4.0 MPa'),
    ('', 'Pressure ratio', '2.0-8.0'),
    ('Feature Dimensions', 'HNN temporal', '100 x 23 = 2,300'),
    ('', 'GNN nodes', '10 x 35 = 350'),
    ('', 'GNN edges', '11 x 12 = 132'),
    ('', 'GNN graph', '50'),
    ('', 'TOTAL per cycle', '2,832'),
    ('Refrigerants', 'Types', 'R410A, R134a, R290'),
    ('', 'Distribution', '40%, 35%, 25%'),
    ('Operating Modes', 'Heating', '55%'),
    ('', 'Cooling', '45%'),
    ('System Age', 'Range', '0-50,000 hours'),
    ('', 'Mean', '15,000 hours'),
    ('Validation', 'Energy balance error', '< 0.001%'),
    ('', 'Carnot compliance', '100%'),
    ('', 'Mass balance', '100%'),
    ('', 'Entropy positive', '100%'),
]

for cat, param, val in stats_data:
    pdf.cell(60, 5, cat, border=1)
    pdf.cell(60, 5, param, border=1)
    pdf.cell(60, 5, val, border=1)
    pdf.ln()

# Figure placement recommendations
pdf.add_page()
pdf.section_title('Figure Placement Recommendations', 1)

pdf.body_text(
    "Use the figures from paper_figures_cyclic/ directory - these show proper closed "
    "Hamiltonian orbits and better represent the cyclic thermodynamic behavior suitable "
    "for HNN training."
)

pdf.ln(4)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "Recommended Figure Placements:", ln=True)
pdf.set_font('Helvetica', '', 9)

placements = [
    ("fig1_hamiltonian_phase_space.png", "After Section 2.2.5 (Hamiltonian) or 2.5.1", "Full width"),
    ("fig2_thermodynamic_phase_diagrams.png", "After Section 2.2.1 (Thermodynamic) or 2.5.2", "Full width"),
    ("fig3_cop_distribution_analysis.png", "After Section 2.4 or 2.5.3", "Full width"),
    ("fig4_hamiltonian_contour.png", "With Phase Space discussion, 2.5.4", "Half/Full width"),
    ("fig5_energy_conservation.png", "After Section 2.3 (Validation) or 2.5.5", "Full width"),
    ("fig7_dataset_statistics.png", "End of Section 2.5.6", "Full width"),
]

for fig, placement, size in placements:
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(70, 5, fig, border=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(70, 5, placement, border=1)
    pdf.cell(40, 5, size, border=1)
    pdf.ln()

pdf.ln(6)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, "Clip/Crop Suggestions:", ln=True)
pdf.set_font('Helvetica', '', 9)

pdf.multi_cell(0, 5,
    "- fig1: Can be split into 3 separate panels (phase space | H conservation | energy components)\n"
    "- fig2: Can be used as 2 separate T-s and P-h figures\n"
    "- fig3: Can be split into 4 individual plots (COP histogram | Carnot compliance | efficiency | COP vs lift)\n"
    "- fig5: Can be split into 4 panels (H conservation | dH/dt | energy balance | energy flow)\n"
    "- fig7: Best kept as single summary figure; alternatively extract statistics box alone"
)

# Save
output_path = Path(__file__).parent / "section25_dataset_analysis.pdf"
pdf.output(str(output_path))
print(f"PDF generated: {output_path}")
