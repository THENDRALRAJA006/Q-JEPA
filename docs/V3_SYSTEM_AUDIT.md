# Surakshit Bharat Q-JEPA Master Model V3.0 — Comprehensive System Audit
### Scientific Integrity Audit, Data Provenance, Artifact Verification, and Architectural Blueprint
**Date**: October 14, 2026  
**Auditor**: Principal AI Research Engineer & Systems Architect  
**Project**: `D:\ZYNEX\surakshit-bharat-q-jepa`  
**Latest Git Commit**: `760d371`  

---

## 1. Executive Summary & Audit Mandate

This document establishes the empirical ground truth of the **Surakshit Bharat — Q-JEPA Traffic** repository prior to V3.0 foundation model development. In accordance with strict scientific auditing standards, no capability is assumed merely from the presence of scripts, configuration files, checkpoint names, or passing unit tests. Every claim is verified against physical filesystem artifacts, dataset checksums, checkpoint inspection metadata, and optimizer trajectories.

---

## 2. Hardware Environment & Computational Profile

An inspection of the local runtime environment reveals the exact hardware budget:

| Resource | Specification | Notes |
|---|---|---|
| **Operating System** | Windows 10/11 Enterprise (64-bit AMD64) | SP0, Build 26200 |
| **Python Runtime** | Python 3.11.0 MSC v.1933 64-bit | Standard CPython |
| **PyTorch Version** | 2.9.0+cpu | **CPU-only build** |
| **CUDA Acceleration** | **Unavailable (`torch.cuda.is_available() == False`)** | **0 GPU devices accessible** |
| **Processor** | 8 physical cores / 16 logical threads | Modern x86-64 multithreaded |
| **RAM Total** | 15.23 GB | System memory |
| **RAM Available** | **~1.73 GB free** (high background memory pressure) | Requires strict batch sizing & GC |
| **Storage (D: Drive)** | 292.97 GB total / **145.37 GB free** | Ample disk space for models/data |

> [!IMPORTANT]
> **Hardware Realism Constraint**:
> Because execution is restricted to PyTorch CPU with ~1.7 GB available RAM, claims of multi-day GPU convergence across billions of tokens or 100-epoch runs on 883-sensor graphs are factually impossible on this workstation. High-capacity GPU training configurations (`configs/pretraining_base.yaml`, `configs/pretraining_large.yaml`) must be prepared for cluster deployment, while local models must be rigorously profiled on verified temporal slices and lightweight parameterizations without deceptive inflation.

---

## 3. Dataset Availability, Licenses & Provenance Audit

| Dataset ID | Expected Sensors | Expected Timesteps | Physical Variable | Units | License | Status on Disk | Actual Ingestion Reality |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **METR-LA** | 207 | 34,272 | Speed | mph | CC BY 4.0 | **PRESENT** (72.5 MB CSV + 680 KB pkl) | **GENUINE REAL DATA**: Verified 34,272 timesteps, 207 loop sensors on LA freeways. Mean=58.58 mph, Std=12.82 mph. |
| **PEMS-BAY** | 325 | 52,116 | Speed | mph | CC BY 4.0 / Caltrans | **ABSENT** (raw .h5 missing) | **SYNTHETIC FALLBACK**: Zenodo API returned HTTP 403 Forbidden. Adapter generated synthetic cosine/noise profile. |
| **PEMS03** | 358 | 26,208 | Flow | veh/5min | CC BY 4.0 / Caltrans | **ABSENT** (raw .npz missing) | **SYNTHETIC FALLBACK**: Download blocked; adapter generated synthetic profile. |
| **PEMS04** | 307 | 16,992 | Flow / Speed | veh/5min | CC BY 4.0 / Caltrans | **ABSENT** (raw .npz missing) | **SYNTHETIC FALLBACK**: Download blocked; adapter generated synthetic profile. |
| **PEMS07** | 883 | 28,224 | Flow | veh/5min | CC BY 4.0 / Caltrans | **ABSENT** (raw .npz missing) | **SYNTHETIC FALLBACK**: Download blocked; adapter generated synthetic profile. |
| **PEMS08** | 170 | 17,856 | Flow | veh/5min | CC BY 4.0 / Caltrans | **ABSENT** (raw .npz missing) | **SYNTHETIC FALLBACK**: Download blocked; adapter generated synthetic profile. |
| **Synthetic Grid** | 30 | 288 | Speed | km/h | MIT | **PRESENT** (Parquet + CSV) | **PROCEDURAL SIMULATION**: 5×6 road network for ambulance routing benchmarks. |

### Dataset Audit Conclusions:
1. **Only METR-LA is a real downloaded dataset.**
2. Remote download scripts (`scripts/download_traffic_datasets.py`) encounter `HTTP 403 Forbidden` on Zenodo records (`records/5724362`).
3. Adapters for `pems_bay`, `pems03`, `pems04`, `pems07`, `pems08` automatically fall back to procedural synthetic sinusoidal profiles (`_generate_caltrans_benchmark_profile`).
4. **V3 Rule**: Datasets that are absent on disk must be explicitly marked as **BLOCKED / ABSENT**, and synthetic profiles must never be described as real Caltrans observations in scientific benchmarks.

---

## 4. Verification of Previous V2.0 Claims

The audit investigated seven critical questions regarding the reported V2.0 results:

### Question 1: Which datasets contributed to pretraining?
- **Finding**: The pretraining configuration (`configs/pretraining_tiny.yaml`) configured a mixture of `['metr_la', 'pems_bay', 'pems04', 'pems08']`.
- **Reality**: Because raw files for `pems_bay`, `pems04`, and `pems08` were not present on disk, the training run mixed genuine METR-LA traffic with three procedural synthetic sinusoidal profiles. Only METR-LA contained real sensor physics.

### Question 2: Which checkpoint generated each reported result?
- **`models/foundation/jepagraph_fm_tiny.pt`** (355 KB, 34,851 params): Generated the pretraining loss history and zero-shot transfer numbers.
- **`models/foundation/jepagraph_fm_ft_metr_la.pt`** (162 KB): Generated the downstream fine-tuning numbers on METR-LA.
- **`models/metr_la/jepagraph_full_metr_la.pt`** (94 KB): Generated the 1-epoch full-data METR-LA benchmark.
- **`models/metr_la/jepagraph_seed_42.pt`, `43.pt`, `44.pt`** (94 KB each): Generated the 3,000-timestep multi-seed slice results.
- **`models/checkpoints/jepa.pt` & `lstm.pt`**: Generated the 30-node synthetic grid simulation numbers.

### Question 3: How many optimizer steps did each training run complete?
- Inspection of `models/foundation/jepagraph_fm_tiny.pt` optimizer state:
  - `step`: **96 optimizer steps total**.
  - `trained_epochs`: **3 epochs**.
  - `batch_size`: 8, `max_samples_per_dataset`: 128 (32 batches per epoch $\times$ 3 epochs = 96 steps).
  - Total training duration: **30.97 seconds**.
- Inspection of `models/foundation/jepagraph_fm_ft_metr_la.pt`:
  - `train_samples`: 256.
  - `train_duration_seconds`: **10.46 seconds** (32 batches of 8 for 3 epochs = **96 optimizer steps**).
- **Conclusion**: The V2.0 foundation model experiments were **fast CPU smoke tests** (96 optimizer steps), not converged multi-epoch pretraining runs.

### Question 4: Did the 1%, 5%, 10%, 25%, 50%, and 100% data budgets use genuinely different sample sizes?
- **Finding**: **NO.**
- In `scripts/run_transfer_benchmark.py` (line 64), `SingleDatasetWindowDataset` was initialized with `max_samples=256`.
- For METR-LA train split (23,967 windows):
  - 1% budget: $23,967 \times 0.01 = 239$ windows $\to$ capped at **239 samples**.
  - 5% budget: $23,967 \times 0.05 = 1,198$ windows $\to$ capped by `max_samples=256` at **256 samples**.
  - 10% budget: $23,967 \times 0.10 = 2,396$ windows $\to$ capped at **256 samples**.
  - 25% budget: $23,967 \times 0.25 = 5,991$ windows $\to$ capped at **256 samples**.
  - 50% budget: $23,967 \times 0.50 = 11,983$ windows $\to$ capped at **256 samples**.
  - 100% budget: $23,967 \times 1.00 = 23,967$ windows $\to$ capped at **256 samples**.
- **Audit Truth**: The reported scaling curve evaluated 239 samples for 1% and the identical 256 samples for 5%, 10%, 25%, 50%, and 100%. The budgets did not scale.

### Question 5: Did the transfer experiment exclude the target dataset from pretraining?
- **Finding**: **NO.**
- The target dataset for transfer was METR-LA (`target_dataset="metr_la"`).
- However, METR-LA was explicitly listed as the first dataset in `configs/pretraining_tiny.yaml` (`datasets.mixture: ['metr_la', 'pems_bay', 'pems04', 'pems08']`).
- The model evaluated in `run_transfer_benchmark.py` was loaded from `models/foundation/jepagraph_fm_tiny.pt`, which had already seen METR-LA during pretraining.
- **Audit Truth**: Calling this evaluation "zero-shot transfer on an unseen network" was invalid. Genuine zero-shot transfer requires strict Leave-One-Dataset-Out (LODO) where the target dataset is completely excluded from source pretraining.

### Question 6: Did forecasting improvements translate into better routing outcomes?
- **Finding**: **NO.**
- In `results/metrics/foundation_ambulance_routing.json`:
  - `2_current_traffic` (greedy baseline): **1130.30s mean journey time** (P90: 1789.57s).
  - `4_time_dependent_predicted` (JEPAGraph-FM): **1324.14s mean journey time** (P90: 2151.56s) — **slower by +193.84s (-17.15% relative improvement)**.
  - `5_time_dependent_uncertainty`: **1316.45s mean journey time** (P90: 2126.50s) — **slower by +186.15s (-16.47% relative improvement)**.
- **Audit Truth**: In global uncurated procedural simulations, greedy current traffic outperformed predictive routing on average. Predictive routing provided benefits only in specific dynamically clearing bottleneck scenarios (e.g. saving 214.7s), but performed worse on steady-state or misestimated links.

### Question 7: Baseline Dominance at Short Horizons
- In `results/metrics/metr_la_full_data_benchmark_results.json`:
  - 15-minute horizon: **Naive Persistence (4.02 mph MAE)** outperformed JEPAGraph (4.29 mph), STGCN (4.58 mph), and ST-LSTM (6.08 mph).
  - 30-minute horizon: **Naive Persistence (5.10 mph MAE)** outperformed JEPAGraph (5.16 mph) and STGCN (5.49 mph).
  - 60-minute horizon: **Historical Average (5.16 mph MAE)** outperformed JEPAGraph (6.75 mph) and STGCN (6.96 mph).
- **Audit Truth**: Highly autocorrelated short-term traffic is dominated by persistence and historical heuristics on 1-epoch CPU training. Deep architectures only demonstrate value over supervised neural baselines, not heuristic baselines under low compute budgets.

---

## 5. Security & Google Maps Integration Audit

The audit performed in `docs/GOOGLE_MAPS_VERIFICATION.md` confirms:
1. **Zero Secret Leakage**: No Google API keys (`AIzaSy*`) exist in tracked files, git history, or environment variables.
2. **Two-Key Architecture**: Server key is isolated to FastAPI backend; browser key is isolated to Streamlit frontend with HTTP referrer restriction.
3. **ToS Compliance Enforced**: All Google Routes responses are tagged `_display_only: True` with explicit prohibitions against ingestion into ML datasets.
4. **Live Request Status**: **BLOCKED** due to absence of configured local GCP credentials; offline mode operates gracefully.

---

## 6. V3.0 Engineering & Architectural Action Plan

To transition from the audited state to the complete V3.0 system, the following actions are mandated:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   V3.0 ARCHITECTURAL PIPELINE                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Unified Real Data Pipeline (src/data/multi_dataset.py)              │
│    - True variable-budget sampling: [1%, 5%, 10%, 25%, 50%, 100%]       │
│    - Remove artificial max_samples=256 bottleneck                      │
│    - Clear labeling of verified real vs absent/synthetic datasets       │
│                                                                        │
│ 2. Q-JEPA Master Spatiotemporal Encoder                                │
│    - Add Adaptive Learnable Adjacency (Softmax(ReLU(E1 E2^T)))         │
│    - Add Validity Mask Channel [x_norm, mask_x, tod, dow] (4 channels) │
│    - Multi-head self-attention / gated causal temporal convolutions    │
│    - Add Epistemic / Aleatoric Uncertainty Head (quantile prediction)  │
│                                                                        │
│ 3. True Leave-One-Dataset-Out (LODO) Transfer Protocol                 │
│    - Target METR-LA: Pretrain ONLY on non-target datasets              │
│    - Target PEMS: Pretrain on METR-LA, evaluate zero-shot on PEMS       │
│                                                                        │
│ 4. Dynamic Time-Dependent Ambulance Router                             │
│    - Edge cost indexing: h_arr = round(T_accum / Delta_t)              │
│    - Uncertainty risk penalty: cost = tt_pred + beta * sigma           │
│    - Dynamic rerouting threshold gate to prevent route oscillation     │
│                                                                        │
│ 5. Automated Verification & Test Suite                                 │
│    - Preserve all 129 existing unit tests                              │
│    - Add V3 tests covering adaptive adjacency, LODO, and scaling curves│
└────────────────────────────────────────────────────────────────────────┘
```
