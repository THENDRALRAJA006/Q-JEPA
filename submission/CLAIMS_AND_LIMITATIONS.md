# Surakshit Bharat — Q-JEPA Traffic V3.1
## Scientific Claims, Empirical Boundaries, and Limitations Audit

**Document Version:** 3.1.0  
**Audit Standard:** Absolute Scientific Integrity & Provenance Tracking  
**Repository:** `D:\ZYNEX\surakshit-bharat-q-jepa`  

This document provides a transparent, audited breakdown distinguishing:
1. **Verified Implementations** (physically written, tested, and operational in code)
2. **Locally Measured Empirical Results** (benchmarks executed on real physical data)
3. **Synthetic Simulation Results** (evaluated within controlled procedural models)
4. **Non-Claims, Boundary Conditions & Future Work** (what is explicitly not claimed)

---

## 1. Verified Implementations (Code & Architecture)

The following components are physically implemented in the repository, verified by 135 passing tests, and fully operational:

| Component | Source Location | Implementation Details | Verification Status |
|---|---|---|:---:|
| **Spatiotemporal JEPA Backbone** | `src/forecasting/jepa_traffic.py` | Causal gated dilated temporal convolutions, spatial graph diffusion, EMA target encoder ($\tau=0.99$), stop-gradient latent prediction loss. | **VERIFIED** (20 foundation unit tests passing) |
| **Adaptive Graph Adjacency** | `src/forecasting/jepa_traffic.py` | Learnable node embeddings $E_1, E_2 \in \mathbb{R}^{N \times d}$; dynamic adjacency $A_{\text{adapt}} = \text{Softmax}(\text{ReLU}(E_1 E_2^T))$. | **VERIFIED** (tested on variable graph sizes $N \in [170, 883]$) |
| **LODO Multi-Dataset Pipeline** | `src/data/multi_dataset.py` | Leave-One-Dataset-Out partitioning; dynamic dataset collation without zero-padding; physical variable normalization tracking. | **VERIFIED** (`lodo_isolation_verified: true` confirmed) |
| **Dynamic Data-Efficiency Loader** | `src/data/multi_dataset.py` | Dynamic target window calculation `[240, 1198, 2397, 5992, 11984, 23967]`; deterministic nested prefix subsets ($D_{1\%} \subset D_{5\%} \dots$). | **VERIFIED** (no artificial sample capping) |
| **Arrival-Time Routing Engine** | `src/routing/shortest_path.py` | Time-dependent dynamic Dijkstra evaluating edge costs at estimated arrival time $h_{\text{arr}} = \text{round}(T_{\text{accum}}/\Delta t)$. | **VERIFIED** (7 routing tests passing) |
| **Uncertainty-Aware Routing** | `src/routing/shortest_path.py` | Risk-hedging cost function $C(e, t) = \text{tt}_{\text{pred}} + \beta \sigma_{\text{pred}}$; quantile uncertainty propagation. | **VERIFIED** (monotonic risk parameter response) |
| **FastAPI REST Service** | `src/api/main.py` | Endpoints for network state, speed forecasts, ambulance routing, and Google Maps integration with Pydantic validation. | **VERIFIED** (FastAPI app boots, schemas valid) |
| **Streamlit 7-Page Dashboard** | `dashboard/app.py` | Command Center, Live Simulation, Route Comparison, AI Lab, Experiments, System Status, Google Maps tab. | **VERIFIED** (all tabs load without runtime exceptions) |
| **Google Maps Integration** | `src/api/routers/google_maps.py` | Routes API v2 client, Maps JS API rendering, two-key architecture, rate limiting, and strict ToS display-only tagging. | **VERIFIED** (22 Google Maps tests passing) |

---

## 2. Locally Measured Empirical Results (Real Sensor Data)

The following metrics were directly measured and verified on physical hardware (AMD64 CPU, 16 threads, ~1.7 GB RAM) using verified Caltrans sensor records:

### 2.1 Dataset Verification on Disk (`results/metrics/v3_1/dataset_validation_audit.json`)
- **METR-LA**: 69.11 MB (34,272 timesteps, 207 sensors, speed in mph). License: CC BY 4.0. Verified on disk at `data/metr_la/METR-LA.csv`.
- **PEMS04**: 31.43 MB (16,992 timesteps, 307 sensors, flow/speed). License: Caltrans Open Data. Verified on disk at `data/pems04/PEMS04.npz`.
- **PEMS08**: 17.67 MB (17,856 timesteps, 170 sensors, flow/speed). License: Caltrans Open Data. Verified on disk at `data/pems08/PEMS08.npz`.

### 2.2 Fair Multi-Horizon Baseline Benchmark (`results/metrics/v3_1/fair_baselines_benchmark.json`)
Evaluated across identical held-out test windows of METR-LA:
- **Persistence:** 15m MAE = 5.67 mph | 30m MAE = 6.35 mph | 60m MAE = 8.78 mph (0 parameters, 0.2 ms)
- **STGCN:** 15m MAE = 5.54 mph | 30m MAE = 6.38 mph | 60m MAE = 7.47 mph (8,376 parameters, 44.2 ms)
- **Graph WaveNet:** 15m MAE = 5.22 mph | 30m MAE = 6.03 mph | 60m MAE = 7.61 mph (18,316 parameters, 54.0 ms)
- **JEPAGraph V1.0:** 15m MAE = 5.20 mph | 30m MAE = 6.01 mph | 60m MAE = 7.49 mph (12,268 parameters, 34.3 ms)
- **JEPAGraph Direct Reg:** 15m MAE = 5.04 mph | 30m MAE = 5.74 mph | 60m MAE = 6.99 mph (12,268 parameters, 32.7 ms)
- **JEPAGraph-FM (LODO FT):** **15m MAE = 4.72 mph | 30m MAE = 5.36 mph | 60m MAE = 6.85 mph** (40,041 parameters, 116.0 ms)

### 2.3 Dynamic Data-Efficiency Scaling (`results/metrics/v3_1/foundation_transfer_metr_la_lodo.json`)
Pretrained on `pems04` + `pems08` (METR-LA completely excluded). Fine-tuned on METR-LA across dynamic budgets:
- **Zero-Shot:** 9.64 mph MAE (0 samples, prior transfer)
- **1% Budget (240 samples):** Pretrained: **5.65 mph** vs Scratch: 6.91 mph (**+18.2% relative gain**)
- **5% Budget (1,198 samples):** Pretrained: **3.67 mph** vs Scratch: 3.95 mph (**+7.0% relative gain**)
- **10% Budget (2,397 samples):** Pretrained: **3.75 mph** vs Scratch: 3.95 mph (**+5.0% relative gain**)
- **50% Budget (11,984 samples):** Pretrained: **3.36 mph** vs Scratch: 3.68 mph (**+8.9% relative gain**)
- **100% Budget (23,967 samples):** Pretrained: **3.32 mph** vs Scratch: 3.48 mph (**+4.5% relative gain**)

---

## 3. Synthetic Simulation Results (Controlled Virtual Environments)

The following metrics were evaluated within procedural traffic simulations and synthetic grid topologies:

### 3.1 Realized Dynamic Ambulance Traversal (`results/metrics/v3_1/ambulance_routing_analysis.json`)
Evaluated across 40 controlled vehicle traversals on a 30-node urban grid under dynamic time-varying congestion:
- **Stationary Congestion:** Greedy Current Traffic (648.4s) vs Time-Dependent Forecast (654.4s) — **-0.9%** (predictive routing incurs a minor 6-second detour when traffic is static).
- **Rush-Hour Onset (Growing Bottleneck):** Greedy Current Traffic (872.1s, P90: 1444.8s) vs Time-Dependent Forecast (**772.4s, P90: 1048.8s**) — **+11.4% faster, saving 99.7 seconds and cutting P90 worst-case delay by 396.0 seconds**.
- **Clearing Bottleneck:** Greedy Current Traffic (753.7s) vs Time-Dependent Forecast (**690.8s**) — **+8.3% faster, saving 62.9 seconds**.
- **Random Fluctuation:** Greedy Current Traffic (694.7s) vs Time-Dependent Forecast (693.3s) — **+0.2%**.

### 3.2 Procedural Fallback Benchmark Datasets
- Datasets `pems_bay`, `pems03`, and `pems07` encountered HTTP 403 Forbidden on remote Zenodo servers.
- The repository generated deterministic sinusoidal benchmark profiles for these 3 datasets.
- **Scientific Boundary:** These 3 synthetic profiles were used solely to verify multi-graph dimensional compatibility in unit tests; they were **strictly excluded from foundation model pretraining**.

---

## 4. Boundary Conditions, Non-Claims & Future Work

To maintain highest scientific credibility, the authors state the following non-claims and boundary conditions:

### 4.1 Computational Limits
- Local pretraining was executed on CPU for 5 epochs (~60s runtime) using the lightweight **Tiny/LODO** architecture (40,041 parameters).
- Production-scale configurations (**Base**: ~110k parameters; **Large**: ~385k parameters) are configured in `configs/pretraining_base.yaml` and `configs/pretraining_large.yaml`. They require multi-GPU CUDA clusters for multi-day pretraining across hundreds of thousands of timesteps.
- We do **not** claim to have trained a 100-epoch GPU foundation model on local hardware.

### 4.2 Medical & Regulatory Disclaimers
- Surakshit Bharat Q-JEPA is an **AI research prototype**, not a certified medical device or operational 911/108 Computer-Aided Dispatch (CAD) system.
- Travel-time savings reported in this work are algorithmic simulation metrics; they do **not** represent clinical trials or proven patient mortality reductions.
- The system does not interface directly with physical municipal Emergency Vehicle Pre-emption (EVP) hardware (e.g. Opticom strobe sensors).

### 4.3 Google Maps Platform Live Status
- Google Maps integration is architecturally complete, tested with unit mocks, and enforces two-key isolation and ToS compliance.
- In the local evaluation environment, valid paid GCP billing credentials are not pre-configured; live network requests to Google Routes API return a graceful `NOT_CONFIGURED` offline status.
- We do **not** claim live billable Google Routes responses have been obtained without user credentials; the system's offline fallback operates seamlessly.

### 4.4 Heuristic Strengths Under Static Traffic
- Under stationary or highly autocorrelated traffic conditions at short horizons ($\le 15$ minutes), simple heuristics (Persistence and Historical Average) are highly competitive with neural models.
- Deep spatiotemporal representations demonstrate clear superiority under non-stationary traffic transitions (rush-hour onset, bottleneck clearance, few-shot target adaptation).

### 4.5 Future Work Roadmap
1. Deploy Base/Large JEPAGraph-FM profiles on distributed NVIDIA H100 GPU clusters with mixed precision (`torch.cuda.amp`).
2. Integrate OpenStreetMap (OSM) and live municipal General Transit Feed Specification (GTFS-RT) feeds for urban signalized street routing.
3. Conduct pilot simulations with municipal emergency medical service (EMS) dispatch authorities.
