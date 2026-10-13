# Datasets — Surakshit Bharat Q-JEPA Traffic

## 1. Real-World Dataset: METR-LA (Acquired, Audited & Benchmarked)

**Source**: Los Angeles Metropolitan Transportation Authority loop detector network (Li et al., ICLR 2018; DCRNN)  
**Distribution / Archive**: Zenodo [DOI: 10.5281/zenodo.5724362](https://zenodo.org/records/5724362)  
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Acquisition Status**: ✅ Downloaded, parsed, and verified in `data/metr_la/`  

### Dataset Inventory & Specifications
| File | Format | Size | Dimensions | Description |
|------|--------|------|------------|-------------|
| `data/metr_la/METR-LA.csv` | CSV | 69.11 MB | 34,272 × 207 | 4 months (March 1, 2012 – June 27, 2012) of 5-minute average vehicle speed readings (mph) |
| `data/metr_la/adj_mx_METR-LA.pkl` | Pickle | 680 KB | (207, 207) | Directed sensor graph adjacency matrix with 1,722 directed edges, sensor ID list, and index mapping |

### Graph Topology & Adjacency
- **Nodes ($N$)**: 207 traffic sensors located across Los Angeles freeways.
- **Edges ($E$)**: 1,722 non-zero directed connections (4.02% matrix density).
- **Adjacency Computation**: Normalized random-walk Laplacian matrix $\tilde{A} = D^{-1}(A + I_N)$ with self-loops, normalized such that each active sensor's outgoing weights sum to 1.0.

### Experiment Provenance Audit & Evaluation Slices
To ensure scientific transparency, the repository explicitly separates exploratory smoke tests, controlled statistical benchmarks, and full-timeline evaluations:

| Execution Tier | Timesteps ($T$) | Fraction of Data | Seeds Evaluated | Usable Windows (Train / Val / Test) | Raw Predictions per Seed | Valid Unmasked Targets per Seed | Masked Dropouts ($y \le 0$) | Total Predictions Evaluated (All Seeds) |
| **Preliminary Full Dataset Benchmark** | 34,272 | 100.0% | `[42]` (1 Epoch, CPU) | 23,967 / 3,416 / 6,844 | 17,000,496 | 14,940,228 (87.88%) | 2,060,268 (12.12%) | 14,940,228 valid (17,000,496 raw) |
| **Controlled Multi-Seed Benchmark** | 3,000 | 8.75% | `[42, 43, 44]` | 2,077 / 289 / 589 | 1,463,076 | 1,390,536 (95.04%) | 72,540 (4.96%) | 4,171,608 valid (4,389,228 raw across 3 seeds) |
| **Exploratory Smoke Test** | 2,000 | 5.84% | `[42]` | 1,377 / 189 / 389 | 966,276 | 920,832 (95.30%) | 45,444 (4.70%) | 920,832 valid (966,276 raw) |

### Arithmetic Reconciliation & Prediction Count Audit (Task 1 Resolution)
In earlier drafts of the project report, the prediction total was cited as:
$$\text{Reported Arithmetic Error: } 589 \text{ test windows} \times 12 \text{ horizons} \times 207 \text{ sensors} \times 3 \text{ seeds} = 1,390,536 \text{ predictions}$$
Our audit reconciled the exact arithmetic and data provenance:
1. **Raw Predictions per Seed**:
   $$589 \times 12 \times 207 = 1,463,076 \text{ predictions per seed}$$
   Across three seeds, the raw prediction count equals $1,463,076 \times 3 = 4,389,228$.
2. **Missing-Value and Sensor-Dropout Masking**:
   In accordance with Li et al. (2018), non-positive speeds and missing sensor readings ($y_{\text{true}} \le 0.0$ or NaN) are excluded from metric evaluation.
   On the 589-window test slice, exactly **72,540 predictions per seed** (4.96%) correspond to sensor dropouts or zero readings.
3. **Valid Evaluated Predictions per Seed**:
   $$1,463,076 - 72,540 = 1,390,536 \text{ valid predictions per seed (95.04\%)}$$
4. **Conclusion**:
   The figure **1,390,536** is the exact number of **valid, unmasked ground-truth targets evaluated in a single seed**, not across 3 seeds. Across 3 seeds, the total valid evaluated targets equal **4,171,608** (out of 4,389,228 raw predictions). The previous report conflated the single-seed valid count with the 3-seed total formula. All documentation and dashboard cards now report raw, valid unmasked, and masked dropout counts explicitly.

### Chronological Splitting & Leakage Prevention
To prevent data leakage, the continuous time series is partitioned chronologically:
- **Train Split (70%)**: First 70% of continuous timestamps ($t = 0 \dots 2,100$ in 3k slice; $t = 0 \dots 23,990$ in full 34k dataset).
- **Validation Split (10%)**: Subsequent 10% of timestamps, used strictly for checkpoint selection.
- **Test Split (20%)**: Final 20% of timestamps, strictly held out for final evaluation.
- **Normalization (Leakage-Free)**:
  - 3,000-timestep slice: $\mu_{\text{train}} = 58.99\text{ mph}, \quad \sigma_{\text{train}} = 12.41\text{ mph}$
  - Full 34,272 dataset: $\mu_{\text{train}} = 58.58\text{ mph}, \quad \sigma_{\text{train}} = 12.82\text{ mph}$
  Computed exclusively over non-zero speeds in the **Train Split**. Validation and Test sets are normalized using these fixed training constants.
- **Checkpoint Selection Rule**: Checkpoints are saved when validation loss reaches a new minimum during training. The test set is never accessed during training, learning rate adjustment, or model selection.

### Evaluation Masking & Metric Calculations
In accordance with established traffic benchmarking standards (Li et al., 2018):
- Missing readings, sensor dropouts, and zero speeds ($y \le 0.0$ or NaN) are masked during metric computation.
- Metrics (MAE, RMSE, MAPE) are calculated exclusively over physical valid readings:
  $$\text{MAE} = \frac{1}{\sum m_i} \sum_{i=1}^N m_i |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{\sum m_i} \sum_{i=1}^N m_i (y_i - \hat{y}_i)^2}$$
  where $m_i = \mathbb{I}(y_i > 0 \land \text{isfinite}(y_i))$.
- Models output predictions normalized by training z-scores; these outputs are transformed back to physical units ($\hat{y}_{\text{mph}} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$) before metric calculation. Correctness is verified via exact unit test fixtures (`tests/test_real_graph_forecasting.py`).

---

## 2. Synthetic Simulation Dataset (Emergency Routing)

**Source**: `TrafficSimulator` (`src/simulation/traffic_simulator.py`)  
**Generator**: `scripts/generate_demo_data.py`  
**License**: MIT (Synthetic)  

### Dataset Inventory
| File | Rows | Edges | Steps | Size |
|------|------|-------|-------|------|
| `synthetic_traffic_normal.parquet` | 28,224 | 98 | 288 | ~1 MB |
| `synthetic_traffic_rush_hour.parquet` | 9,408 | 98 | 96 | ~320 KB |
| `synthetic_traffic_road_blockage.parquet` | 9,408 | 98 | 96 | ~317 KB |

### Features
- `step` — simulation time step (5-min intervals)
- `time_minutes` — elapsed minutes
- `edge_id` — road segment identifier ($u \to v$)
- `baseline_speed_kmh` — free-flow speed
- `gt_speed_kmh` — simulated ground-truth speed (not accessible during real-time inference)
- `observed_speed_kmh` — sensor observation (subject to noise/dropout)
- `flow_veh_per_hour` — simulated vehicle throughput
- `congestion_level` — normalized congestion index $[0.0, 1.0]$

---

## 3. Data Provenance & Domain Boundary Protocol

| Dataset | Nature | Domain | Primary Purpose | Ambulance Routing Validation? |
|---------|--------|--------|-----------------|-------------------------------|
| **METR-LA** | Real Physical Sensors | Los Angeles Freeways | Multi-horizon graph speed forecasting (15m, 30m, 60m) | ❌ No (Highway sensors only, no urban emergency O-D data) |
| **Synthetic Simulator** | Procedural Simulation | Urban City Grid | End-to-end ambulance dispatch, routing, and dynamic rerouting | ✅ Yes (Within controlled simulated scenarios) |

---

## 4. Multi-Dataset Benchmark Foundation Suite (V2.0)

To support cross-network representation pretraining and transfer learning across California highway systems, `JEPAGraph-FM` integrates 6 canonical spatiotemporal benchmarks:

| Dataset ID | Official Benchmark Name | Target Variable | Units | Sensors ($N$) | Timesteps ($T$) | Edges | Graph Density | License | Provenance / Repository |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`metr_la`** | METR-LA Highway Speed | Speed | mph | 207 | 34,272 | 1,515 | 0.0355 | CC BY 4.0 | [DCRNN (Li et al., ICLR 2018)](https://github.com/liyaguang/DCRNN) |
| **`pems_bay`** | PEMS-BAY Highway Speed | Speed | mph | 325 | 52,116 | 33,208 | 0.3154 | CC BY 4.0 / Open Data | [DCRNN (Li et al., ICLR 2018)](https://github.com/liyaguang/DCRNN) |
| **`pems03`** | PEMS03 Freeway Flow | Flow | veh/5min | 358 | 26,208 | 40,486 | 0.3168 | CC BY 4.0 / Open Data | [ASTGCN (Guo et al., AAAI 2019)](https://github.com/Davidham3/ASTGCN) |
| **`pems04`** | PEMS04 Freeway Flow & Speed | Flow | veh/5min | 307 | 16,992 | 29,680 | 0.3159 | CC BY 4.0 / Open Data | [ASTGCN (Guo et al., AAAI 2019)](https://github.com/Davidham3/ASTGCN) |
| **`pems07`** | PEMS07 Arterial & Freeway Flow | Flow | veh/5min | 883 | 28,224 | 248,094 | 0.3186 | CC BY 4.0 / Open Data | [ASTGCN (Guo et al., AAAI 2019)](https://github.com/Davidham3/ASTGCN) |
| **`pems08`** | PEMS08 Freeway Flow | Flow | veh/5min | 170 | 17,856 | 8,990 | 0.3129 | CC BY 4.0 / Open Data | [ASTGCN (Guo et al., AAAI 2019)](https://github.com/Davidham3/ASTGCN) |

### Integrity Verification & Ingestion Protocols
- **Train-Only Z-Score Standardization:** Normalization constants ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on the chronological training splits ($70\%$ or $60\%$) over physically valid entries ($y > 0$).
- **Missing Value & Sensor Dropout Handling:** Sensor outages and missing observations ($y \le 0$ or NaN) are explicitly masked ($M_{t, i} = 0$) and excluded from all pretraining reconstruction losses and evaluation metrics.
- **Heterogeneous Graph Support:** Graph sizes range from $N=170$ up to $N=883$. No model layers hardcode sensor identifiers.


