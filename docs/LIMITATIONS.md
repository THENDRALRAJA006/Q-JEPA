# Limitations & Boundary Conditions — Surakshit Bharat Q-JEPA
### Phase 2.1 & Phase 3 Research Documentation

## 1. Domain & Dataset Boundaries

1. **Freeway Sensors vs Urban Surface Streets**:  
   METR-LA consists of 207 inductive loop detectors on Los Angeles interstate freeways. While it successfully validates spatial-temporal graph speed forecasting, freeway flow dynamics (uninterrupted flow) differ substantially from urban arterial streets governed by traffic signals, pedestrian crossings, and turning lanes. To maintain scientific integrity, highway forecasting models are NOT transferred directly to urban grids without local topological calibration.
2. **Absence of Real Ambulance GPS Traces**:  
   Neither METR-LA nor the simulation suite contains actual historical ambulance GPS logs, siren/pre-emption logs, or hospital triage records. Ambulance navigation is evaluated within controlled procedural scenarios.
3. **Time-Dependent Routing Scope (Phase 3 Status)**:  
   Phase 3 resolved the departure-time forecast limitation by introducing `TimeDependentRouter` with arrival-time-aware edge costs ($h_{\text{arr}} = \text{round}(T_{\text{accum}} / \Delta t)$) and fallback clamping. However, this is currently validated on the simulated urban grid; deployment on real-world city networks requires integration with OpenStreetMap (OSM) and live municipal sensor APIs.
4. **Computational Boundaries on CPU & Preliminary Benchmark Status**:  
   The full METR-LA dataset contains 34,272 timesteps ($17,000,496$ raw predictions, $14,940,228$ valid targets per model). To execute rigorous multi-seed statistical evaluations (`[42, 43, 44]`) with confidence intervals on CPU hardware, we evaluated a continuous 3,000-timestep slice (4,171,608 valid targets evaluated across 3 seeds). Additionally, a single-seed complete timeline benchmark evaluates all 34,272 timesteps for **1 single epoch per neural architecture on a single seed (`42`)**. This complete-timeline run is strictly a **preliminary full-data benchmark**; neural models did not achieve full validation convergence under this 1-epoch CPU budget. Full multi-epoch training (e.g. 70–100 epochs as in published GPU baselines like DCRNN and STGCN) requires dedicated CUDA hardware. Comparing our preliminary CPU benchmark directly against 100-epoch GPU literature numbers is NOT an apples-to-apples comparison.

## 2. Model & Algorithmic Boundaries

5. **Heuristic Baseline Dominance at Short Horizons**:  
   In both the 3,000-timestep slice and the full 34,272-timestep benchmark, **Naive Persistence outperforms all neural models at the 15-minute horizon (4.02 mph vs 4.29 mph for JEPAGraph) and 30-minute horizon (5.10 mph vs 5.16 mph)**, and Historical Average outperforms all models at 60 minutes (5.16 mph vs 6.75 mph). Deep spatial-temporal representations only exhibit superiority over supervised neural baselines (STGCN, ST-LSTM, Direct Regression ablation), not over simple heuristics on highly autocorrelated short-term traffic.
6. **Ambulance Routing Trade-Offs (Current Traffic vs Dynamic Prediction)**:  
   On average across 45 emergency scenarios, the **greedy Current-Traffic baseline achieves a lower mean travel time (1402.6s) and lower 90th-percentile tail latency (2010.6s)** than Time-Dependent Predicted routing (1410.0s / P90: 2247.4s) and Uncertainty-Aware routing (1403.8s / P90: 2169.6s). Predicted routing provides large gains in specific scenarios with clearing bottlenecks (e.g. saving 214.7s) and during mid-journey incident diversions (saving 50.0%), but can suffer delays when transient clearance is slower than expected. Uncertainty-awareness hedges against this risk (reducing P90 tail delay by -77.8s over pure predictions), but does not claim unconditional superiority over greedy current traffic.
7. **Scale of Representation Pretraining**:  
   Unlike computer vision JEPA architectures (e.g. I-JEPA, V-JEPA) pretrained on billions of tokens across diverse visual datasets, JEPAGraph is currently trained within domain partitions. Cross-city transfer learning (e.g., pretraining on PEMS-BAY and evaluating zero-shot on METR-LA) represents an open research avenue.
8. **Simulated Annealing Overhead**:  
   Quantum-inspired simulated annealing (Strategy G in Phase 1) is executed as a classical metaheuristic on CPU. On grid networks, Dijkstra already finds the exact global optimum; simulated annealing therefore incurs ~50× higher computational latency without path quality improvements.
9. **Uncertainty Calibration**:  
   Monte Carlo Dropout variance provides relative epistemic uncertainty across horizons but is not formally calibrated into exact Bayesian predictive intervals or conformal bounds.

## 3. Operational & Clinical Disclaimers

8. **Research Prototype Only**:  
   This software is an AI research prototype. It must not be deployed for operational 911/108 emergency vehicle dispatch or safety-critical navigation without regulatory certification.
9. **No Live Signal Control**:  
   The system provides predictive speed estimates and routing advisories; it does not directly trigger physical traffic light controllers or emergency vehicle pre-emption (EVP) hardware (e.g., Opticom).
10. **No Clinical Outcome Claims**:  
    Reported travel-time reductions are algorithmic simulation benchmarks and do not constitute clinical evidence of improved patient survival or morbidity reduction.

---

## 4. Foundation Model V2.0 Boundaries & Research Caveats

11. **Hardware Profile & Scale (CPU vs CUDA GPU)**:  
    Local execution is constrained to a Windows 64-bit CPU environment (16 cores, ~2.5 GB free RAM, PyTorch CPU). Consequently, multi-dataset pretraining and downstream transfer benchmarks were verified using the **Tiny** configuration (34,851 parameters, 2 layers, hidden dimension 32) and budgeted temporal slices. Production-scale configurations (**Base**: 64-dim, 4 layers, ~110k parameters; **Large**: 128-dim, 6 layers, ~385k parameters) are fully implemented and configured (`configs/pretraining_base.yaml`, `configs/pretraining_large.yaml`), but require multi-GPU CUDA clusters with mixed precision (`torch.cuda.amp`) for multi-day convergence.
12. **Dataset Licensing & Reuse Protocol**:  
    METR-LA and PeMS benchmark datasets are distributed under Creative Commons Attribution 4.0 International (CC BY 4.0) and Caltrans open data terms. External Traffic4Cast datasets (IARAI) maintain distinct non-commercial research terms; researchers must independently verify redistribution and commercial applicability prior to downstream deployment.
13. **Zero-Shot Transfer Boundaries**:  
    Direct zero-shot transfer on unseen networks achieves **5.30 mph** at 15m without any target parameter updates. While competitive with supervised baselines like STGCN (5.24 mph), it is substantially improved by brief few-shot fine-tuning (3.81 mph). Zero-shot representations provide strong topological inductive biases, but network-specific speed limits and sensor geometries benefit from target calibration.
14. **Multimodal Physical Heterogeneity**:  
    Traffic flow (veh/5min) and speed (mph) represent fundamentally distinct physical properties. While `JEPAGraph-FM` embeds both via a unified spatiotemporal backbone, decoders and train-only normalizations remain strictly variable-specific to prevent invalid cross-domain unit mixing.


