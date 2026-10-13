# Standardized Benchmark Protocol: METR-LA Spatio-Temporal Forecasting

**Project:** Surakshit Bharat — Q-JEPA Traffic  
**Repository:** `D:\ZYNEX\surakshit-bharat-q-jepa`  
**Protocol Version:** 2.0 (Audited October 2026)  
**Standardized Reference Dataset:** METR-LA (Los Angeles Highway Traffic Speed Sensors)  

---

## 1. Overview and Core Principles

To ensure complete scientific integrity and prevent experimental bias or data leakage, this document defines the standardized evaluation protocol applied to all traffic forecasting experiments in this repository.

### Key Tenets
1. **No Data Leakage:** All normalization statistics, time features, and data transformations are fitted strictly on the chronological training split.
2. **Consistent Missing-Value Handling:** Ground-truth observations with non-positive values ($y \le 0.0\text{ mph}$) or $\text{NaN}$ values represent sensor outages or transmission errors and are strictly masked during metric evaluation.
3. **Apples-to-Apples Comparisons:** Every local neural model receives identical input features, graph adjacency matrices, loss objectives (or explicit ablations), training time horizons, and validation-based checkpoint selection.
4. **Strict Separation of Literature and Local Runs:** Published results from external research papers are never mixed with locally executed experiments into a single ranking.

---

## 2. Dataset Specifications: METR-LA

* **Data Source:** Caltrans Performance Measurement System (PeMS), Los Angeles County highway network.
* **Archival Citation:** Zenodo Repository (DOI: `10.5281/zenodo.5724362`), distributed under Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Sensor Nodes:** $N = 207$ loop detector stations.
* **Graph Topology:** Directed graph with 1,722 directed edges. Edge weights represent spatial proximity computed via thresholded Gaussian kernel:
  $$W_{ij} = \begin{cases} \exp\left(-\frac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right) & \text{if } \text{dist}(v_i, v_j) \le \kappa \\ 0 & \text{otherwise} \end{cases}$$
* **Graph Normalization:** Random-walk transition matrix with self-loops:
  $$\tilde{A} = D^{-1}(A + I_N) \quad \text{where } D_{ii} = \sum_j (A_{ij} + \delta_{ij})$$
* **Sampling Frequency:** 5-minute aggregation intervals (288 observations per day).
* **Total Usable Timesteps:** 34,272 continuous timesteps spanning March 1, 2012 to June 27, 2012 (approximately 4 months).

---

## 3. Data Splitting & Sliding Windows

### 3.1 Chronological Split Ratios
To preserve temporal causality and prevent future-to-past information leakage, a strict chronological split is enforced:
* **Training Set:** First 70% of continuous timeline.
* **Validation Set:** Middle 10% of continuous timeline (used solely for checkpoint selection and hyperparameter monitoring).
* **Test Set:** Final 20% of continuous timeline (held-out evaluation).

### 3.2 Sliding Window Generation
Given sequence of continuous speed matrices $S \in \mathbb{R}^{T \times N}$:
* **Input Window ($T_{\text{in}}$):** 12 steps (60 minutes historical lookback).
* **Output Horizon ($T_{\text{out}}$):** 12 steps (60 minutes future forecast).
* **Window Dimensions:**
  - Sample Input $X_i \in \mathbb{R}^{T_{\text{in}} \times N \times F}$ (where $F=2$: feature 0 = normalized speed, feature 1 = normalized time-of-day in $[0, 1)$).
  - Target Output $Y_i \in \mathbb{R}^{T_{\text{out}} \times N}$ in physical units (mph).
* **Split Boundary Handling:** To avoid lost windows at split boundaries, validation and test splits include a $T_{\text{in}}$-step lead-in buffer from the preceding partition, ensuring valid sliding windows start immediately at the true partition index.

---

## 4. Normalization and Zero-Value Policy

### 4.1 Train-Only Standardization
Normalization parameters are computed strictly from the training partition:
$$\mu_{\text{train}} = \frac{1}{|V_{\text{train}}|} \sum_{(t, n) \in V_{\text{train}}} S_{t, n}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{|V_{\text{train}}|} \sum_{(t, n) \in V_{\text{train}}} (S_{t, n} - \mu_{\text{train}})^2}$$
where $V_{\text{train}} = \{(t, n) \mid t < T_{\text{train\_end}}, S_{t, n} > 0.0\text{ and } \text{isfinite}(S_{t, n})\}$.

* Full dataset training mean: $\mu_{\text{train}} = 58.58\text{ mph}$
* Full dataset training standard deviation: $\sigma_{\text{train}} = 12.82\text{ mph}$
* Inputs normalized as: $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$
* Model predictions mapped back to physical units: $\hat{y} = \hat{z} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$

### 4.2 Missing-Value and Sensor Dropout Masking
Loop detector sensors occasionally record $0.0\text{ mph}$ or missing values due to communication faults, power drops, or maintenance shutdowns.
* **Masking Condition:**
  $$\mathcal{M}_{i, h, n} = \mathbb{I}\left(Y_{i, h, n} > 0.0 \land \text{isfinite}(Y_{i, h, n}) \land \text{isfinite}(\hat{Y}_{i, h, n})\right)$$
* Metrics are accumulated exclusively over valid entries where $\mathcal{M}_{i, h, n} = 1$.
* Across the 34,272-step full test split ($N_{\text{raw}} = 17,000,496$ targets), $14,940,228$ targets (87.88%) are valid unmasked observations; $2,060,268$ targets (12.12%) are masked dropouts.

---

## 5. Evaluation Metrics & Horizon Mapping

The METR-LA sampling interval is 5 minutes. The 12 forecast horizons correspond to:
* **Step 3:** 15 minutes ahead ($3 \times 5 = 15\text{ min}$, index 2)
* **Step 6:** 30 minutes ahead ($6 \times 5 = 30\text{ min}$, index 5)
* **Step 12:** 60 minutes ahead ($12 \times 5 = 60\text{ min}$, index 11)
* **Steps 1 through 12:** All individual intermediate horizons.
* **Overall Average:** Mean across all 12 forecast steps.

### Metric Formulas
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{\sum \mathcal{M}} \sum_{i, h, n} \mathcal{M}_{i, h, n} |\hat{Y}_{i, h, n} - Y_{i, h, n}| \quad [\text{mph}]$$
2. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{\sum \mathcal{M}} \sum_{i, h, n} \mathcal{M}_{i, h, n} (\hat{Y}_{i, h, n} - Y_{i, h, n})^2} \quad [\text{mph}]$$
3. **Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{100\%}{\sum \mathcal{M}} \sum_{i, h, n} \mathcal{M}_{i, h, n} \left|\frac{\hat{Y}_{i, h, n} - Y_{i, h, n}}{Y_{i, h, n}}\right| \quad [\%]$$
4. **Relative MAE Improvement:**
   $$\Delta \text{MAE}_{\text{rel}} = 100 \times \frac{\text{MAE}_{\text{baseline}} - \text{MAE}_{\text{JEPAGraph}}}{\text{MAE}_{\text{baseline}}} \quad [\%]$$
   - Positive values indicate JEPAGraph has lower error (better performance).
   - Negative values indicate JEPAGraph has higher error (worse performance).

---

## 6. Training & Checkpoint Selection Protocol

* **Optimization:** Adam optimizer with initial learning rate $\eta = 10^{-3}$, weight decay $10^{-5}$, gradient clipping at $\|\mathbf{g}\|_2 \le 5.0$.
* **Batch Size:** 64 sliding windows per mini-batch.
* **Validation Monitoring:** At the conclusion of each epoch, validation MSE is evaluated on the held-out validation set.
* **Checkpoint Selection:** The exact network state dictionary yielding the lowest validation loss ($\arg\min \mathcal{L}_{\text{val}}$) is preserved and restored prior to evaluating the held-out test set.
* **Random Seeds:** Multi-seed evaluations execute over independent pseudo-random seeds: $\{42, 43, 44\}$, reporting mean, standard deviation, and 95% confidence intervals.
* **Ablation Symmetry:** JEPAGraph and its Direct-Regression ablation share identical online encoder layers, graph predictor dimensions, and speed decoder weights. The ablation sets $\lambda_{\text{rep}} = 0.0$ and freezes target-encoder EMA updates.

---

## 7. Foundation Model Multi-Dataset Benchmarking Protocol (V2.0)

For large-scale spatiotemporal pretraining and cross-dataset evaluation of `JEPAGraph-FM`:

### Protocol A: Multi-Dataset Balanced Pretraining
* **Dataset Mixture:** Training partitions of METR-LA, PEMS-BAY, PEMS04, PEMS08.
* **Temperature Sampling:** $p_d \propto S_d^{0.5}$, balancing datasets with varying time spans and sensor counts.
* **Heterogeneous Graph Collation:** Mini-batches contain samples exclusively from a single dataset, eliminating padding while supporting $N \in [170, 883]$.

### Protocol B: Target-Specific Fine-Tuning
* Supervised fine-tuning on downstream target training partitions using train-only standardization.
* Model selection guided strictly by target validation split.

### Protocol C: Leave-One-Dataset-Out (LODO) Transfer
* Target dataset $D_{\text{target}}$ is completely held out during source pretraining.
* **Zero-Shot Evaluation:** Direct forward pass on $D_{\text{target}}$ without target gradient updates.
* **Few-Shot Fine-Tuning:** Evaluated on target training subsets ($1\%, 5\%, 10\%, 25\%$).

### Protocol D: Data Efficiency Scaling
* Evaluates label-efficiency gain: comparing sample fractions $[1\%, 5\%, 10\%, 25\%, 50\%, 100\%]$ between pretrained foundation representations vs random initialization.

### Protocol E: Ambulance Emergency Routing Evaluation
* Time-dependent arrival-time edge evaluation:
  $$h_{\text{arrival}} = \text{round}\left(\frac{t_{\text{elapsed}}}{\Delta t_{\text{sample}}}\right)$$
* Evaluates 5 strategies on identical simulated dispatch trips: Shortest Distance, Current Traffic, Historical Average, Forecast-Aware, and Uncertainty-Aware Routing.

