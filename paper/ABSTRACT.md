# Abstract (IEEE Conference Format, ~200 words)

Accurate Coefficient of Performance (COP) prediction for industrial heat pump systems is challenging due to complex interactions between thermodynamic cycles, component degradation, and environmental conditions. This paper presents a hybrid approach combining comprehensive physics-based modeling with a Hamiltonian Neural Network and Graph Neural Network (HNN-GNN) architecture. We develop a modular simulation framework integrating 17 physics models including refrigerant thermodynamics via CoolProp, heat exchanger fouling, compressor wear degradation, refrigerant charge effects, ambient condition impacts, and part-load cycling behavior based on AHRI degradation curves. Each generated cycle undergoes Carnot limit enforcement and energy balance validation to ensure physical plausibility. The HNN-GNN architecture processes 2,832 physics-derived features, with the HNN learning energy-conserving temporal dynamics across 100 timesteps and the GNN modeling component relationships through a 10-node heterogeneous graph representing compressor, heat exchangers, expansion valve, and auxiliary subsystems. Training on [INSERT NUMBER] synthetic cycles generated under diverse operating conditions, the model achieves mean absolute error of [INSERT MAE] and R² of [INSERT R²], outperforming baseline methods by [INSERT %]. The physics-informed approach guarantees thermodynamically consistent predictions while capturing degradation and fault effects that purely data-driven methods miss.

---

**Word count: ~195 words**

**Statistics to insert:**
- [INSERT NUMBER] - Training cycles
- [INSERT MAE] - Mean Absolute Error  
- [INSERT R²] - Coefficient of determination
- [INSERT %] - Improvement percentage over baselines

The integration of detailed physics models with physics-informed neural network architectures offers several advantages for industrial applications: (1) predictions respect thermodynamic constraints by construction, eliminating physically impossible COP values; (2) the modular physics framework enables systematic incorporation of domain knowledge about component degradation, fault conditions, and control strategies; (3) the HNN component provides interpretable energy-based representations amenable to physical analysis; and (4) the GNN component explicitly encodes the topological structure of heat pump systems, facilitating transfer learning across different system configurations. This work demonstrates that combining rigorous physics-based simulation with modern deep learning architectures yields accurate, physically consistent, and interpretable models for complex thermodynamic systems.

---

## Key Statistics to Insert After Training:

1. **[INSERT NUMBER]** - Total number of training cycles (e.g., 10,000 or 100,000)
2. **[INSERT MAE]** - Mean Absolute Error on test set (e.g., 0.15 COP units)
3. **[INSERT RMSE]** - Root Mean Squared Error on test set (e.g., 0.22 COP units)
4. **[INSERT R2]** - Coefficient of determination (e.g., 0.94)
5. **[INSERT PERCENTAGE]** - Improvement over baseline methods
6. **[INSERT KEY FINDINGS ABOUT COMPONENT CONTRIBUTIONS]** - Results from ablation study (e.g., "removing the HNN component increases error by 15%, while removing GNN increases error by 23%")
7. **[INSERT TOP CONTRIBUTING FEATURES]** - Feature importance results (e.g., "condenser temperature, compressor efficiency degradation, and part-load ratio contribute most significantly to prediction accuracy")
