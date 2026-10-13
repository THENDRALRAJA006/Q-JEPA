# Methodology — Surakshit Bharat Q-JEPA Traffic

## Research Questions & Hypotheses

1. **Graph Representation Learning Hypothesis**: Does predicting future traffic representations using a Joint Embedding Predictive Architecture (JEPA) provide an empirical advantage over direct future-speed regression, recurrent baselines (LSTM), and spatial-temporal graph baselines (STGCN) across multiple prediction horizons (15m, 30m, 60m) on real-world sensor data?
2. **Forecast-Aware Routing Hypothesis**: Does feeding multi-step traffic speed forecasts and uncertainty estimates into routing cost functions reduce ambulance travel times and improve journey-time reliability under congestion and incident scenarios?

These hypotheses are investigated experimentally without presupposing positive findings.

---

## Part 1: Real-World Graph Forecasting Methodology (METR-LA)

### 1. Data Ingestion & Chronological Splitting
- **Dataset**: METR-LA (207 sensors, 5-minute sampling rate, 34,272 timestamps).
- **Graph Adjacency**: Directed graph with 1,722 directed edges. Normalized random-walk Laplacian matrix:
  $$\tilde{A} = D^{-1}(A + I_N)$$
- **Chronological Partitioning**:
  - Train Split (70%): First 70% of continuous timestamps.
  - Validation Split (10%): Middle 10% (used strictly for model selection and early stopping).
  - Test Split (20%): Final 20% (strictly held out).
- **Leakage-Free Normalization**:
  $$\mu_{\text{train}} = 59.41\text{ mph}, \quad \sigma_{\text{train}} = 12.38\text{ mph}$$
  Computed exclusively over non-zero speeds in the train split.
- **Evaluation Masking**:
  All metrics mask missing values and sensor faults ($y_{\text{true}} \le 0.0$ or NaN), evaluating strictly over physical observations:
  $$\text{MAE} = \frac{1}{\sum m_i} \sum_{i=1}^N m_i |y_i - \hat{y}_i|, \quad m_i = \mathbb{I}(y_i > 0)$$

### 2. Evaluated Model Architectures

#### A. Persistence Baseline
Vectorized baseline copying the latest observed sensor speed into all future horizons:
$$\hat{Y}_{t+h, v} = X_{t, v}, \quad \forall h \in \{1, \dots, 12\}$$

#### B. Historical Average (HA) Baseline
Estimates speed based on the historical mean speed for the exact 5-minute time-of-day slot ($288$ daily slots), fit exclusively on the train split:
$$\hat{Y}_{t+h, v} = \bar{Y}_{\text{slot}(t+h), v}$$

#### C. Spatial-Temporal LSTM
- 2-layer LSTM with 64 hidden units and linear input/output spatial projections across all $207$ sensor channels.
- Direct multi-step regression trained via MSE loss.

#### D. STGCN (Spatial-Temporal Graph Convolutional Network)
- Stack of 2 Spatio-Temporal Blocks:
  1. Spatial Graph Convolution: $H_{\text{spatial}} = \text{GELU}(\tilde{A} X W)$
  2. 1D Temporal Convolution across history window ($T_{\text{in}} = 12$)
- Linear projection head mapping pooled temporal features to 12 future horizons.

#### E. JEPAGraph (Proposed Representation-Learning Architecture)
```
Context Window [B, 12, 207, 2] ──▶ Online Encoder E_θ ──▶ Z_context [B, 207, D]
                                                                │
                                                         Graph Predictor P_ϕ
                                                                │
                                                              Z_hat [B, 207, D] ──▶ Latent Loss L_rep
                                                                │                        ▲
Future Window [B, 12, 207, 1] ──▶ Target Encoder E_ξ ──▶ Z_target [B, 207, D] ─────────┘
                                  (stop_gradient, EMA)          │
                                                         Speed Decoder D_ψ
                                                                │
                                                         Y_pred [B, 12, 207] ──▶ Speed Loss L_speed
```

**Key Architectural Properties:**
1. **Target Encoder Stop-Gradient**: Gradients never backpropagate through $E_\xi$:
   $$Z_{\text{target}} = \text{stop\_gradient}(E_\xi(Y_{\text{future}}))$$
2. **Exponential Moving Average (EMA)**: Target encoder weights $\xi$ update smoothly from online weights $\theta$:
   $$\xi \leftarrow \tau \xi + (1 - \tau) \theta, \quad \tau = 0.99$$
3. **Latent Graph Predictor**: $P_\phi$ performs message-passing across graph neighbors in latent representation space:
    $$\hat{Z} = \text{LayerNorm}(Z_{\text{context}} + \text{MLP}(\text{GCN}(Z_{\text{context}}, \tilde{A})))$$
4. **Multi-Horizon Speed Decoder**: $D_\psi$ decodes predicted representations to physical speeds:
   $$\hat{Y} = D_\psi(\hat{Z}) \in \mathbb{R}^{B \times 12 \times 207}$$
5. **Multi-Task Objective**:
   $$\mathcal{L} = \mathcal{L}_{\text{speed}}(\hat{Y}, Y_{\text{true}}) + \lambda_{\text{rep}} \text{SmoothL1}(\hat{Z}, Z_{\text{target}})$$

#### F. JEPAGraph Direct Future-Speed Regression Ablation
To strictly isolate whether representation learning provides an empirical advantage over direct speed regression, the ablation removes the latent representation predictor and EMA target encoder:
- **Architecture**: Identical online encoder $E_\theta$ and projection head $D_\psi$.
- **Loss Function**: Pure supervised speed regression ($\lambda_{\text{rep}} = 0.0$):
  $$\mathcal{L}_{\text{ablation}} = \frac{1}{B \cdot 12 \cdot N} \sum_{b, t, v} |Y_{\text{pred}} - Y_{\text{true}}|$$
- **Parameters & Hyperparameters**: Exactly matches JEPAGraph (12,268 parameters, batch size 64, Adam optimizer, lr = 0.001).

### 3. Horizon Definitions & Multi-Seed Protocol (Phase 2.1)
- **Dataset Sampling Interval**: $\Delta t = 5\text{ minutes}$.
- **Evaluation Conventions**:
  - **15-minute Horizon**: Prediction error evaluated specifically at prediction step $h = 3$ ($t + 15\text{ min}$).
  - **30-minute Horizon**: Prediction error evaluated specifically at prediction step $h = 6$ ($t + 30\text{ min}$).
  - **60-minute Horizon**: Prediction error evaluated specifically at prediction step $h = 12$ ($t + 60\text{ min}$).
  - **Average Future Horizon**: Unweighted arithmetic mean across all future prediction steps $h \in \{1, \dots, 12\}$.
- **Checkpoint Selection**: Saved whenever validation loss reaches a new global minimum. Test splits are strictly evaluated once after training is complete.
- **Multi-Seed Protocol**:
  - Models are trained and evaluated across at least 3 distinct random seeds: `[42, 43, 44]`.
  - For each horizon, we report mean, sample standard deviation, and Student's $t$-distribution 95% two-sided confidence intervals ($\text{df} = 2, t_{\text{crit}} = 4.303$):
    $$\text{CI}_{95\%} = \bar{x} \pm t_{0.025, 2} \frac{s}{\sqrt{3}}$$
- **Relative Improvement Formulation**:
### 4. Comparison and Diagnostic Audit vs Published METR-LA Literature
Published benchmark literature (Li et al., ICLR 2018; Yu et al., IJCAI 2018) reports METR-LA MAE figures in the range of 2.7–3.6 mph for deep spatial-temporal models. Our scientific integrity audit conducted a diagnostic breakdown across six key dimensions:

1. **Training Epochs & Convergence Budget**:
   - *Published Studies*: Train complex models (e.g., DCRNN, STGCN) for 70–100 epochs using learning rate schedulers, gradient clipping, scheduled sampling, and curriculum learning over multi-GPU compute (NVIDIA Tesla K80/V100).
   - *Our Controlled Evaluation*: Trained for 2 epochs on CPU across 3 seeds (`[42, 43, 44]`) on a 3,000-timestep slice.
   - *Our Complete Dataset Run*: Evaluated on all 34,272 timesteps for **1 single epoch on a single seed (Seed 42) on CPU**. This run must be treated as a **Preliminary Full-Timeline Benchmark**, not a fully converged research conclusion. Comparing a 1-epoch CPU run directly against a 100-epoch GPU baseline from published literature is NOT an apples-to-apples comparison.
2. **Model Parameterization & Complexity**:
   - *DCRNN (Li et al.)*: 370,000+ parameters utilizing multi-layer recurrent sequence-to-sequence cells with bidirectional random-walk diffusion convolutions.
   - *JEPAGraph (Ours)*: Lightweight 12,268 parameters designed for sub-100ms inference on commodity edge CPUs for real-time dispatch operations.
3. **Historical Average, Persistence & ARIMA Baselines (Corrected Literature Attribution)**:
   - *Published ARIMA (Li et al. 2018, Table 1)*: 15m MAE = 3.99 mph, 30m MAE = 5.15 mph, 60m MAE = 6.90 mph. *(Earlier project notes erroneously labeled these values as "Published Persistence"; Table 1 of Li et al. 2018 establishes them as the published ARIMA baseline).*
   - *Our Naive Persistence Baseline*:
     - *Full 34,272 Dataset*: 15m MAE = **4.02 mph**, 30m MAE = **5.10 mph**, 60m MAE = **6.80 mph**, Avg 1–12 MAE = **5.14 mph**.
     - *Controlled 3,000-Step Slice*: 15m MAE = **3.64 mph**, 30m MAE = **4.52 mph**, 60m MAE = **6.07 mph**, Avg 1–12 MAE = **4.58 mph**.
     - *Critical Empirical Finding*: Naive Persistence closely tracks the published ARIMA baseline because highway speeds exhibit very high short-term temporal autocorrelation. **Persistence outperforms JEPAGraph at 15m (4.02 vs 4.29 mph) and 30m (5.10 vs 5.16 mph), and on the overall 1–12 average (5.14 vs 5.23 mph) on the full dataset, and outperforms ALL neural baselines across all horizons on the 3,000-step slice.** JEPAGraph only marginally outperforms Persistence at the 60-minute horizon on the full dataset (6.75 vs 6.80 mph, a 0.76% / 0.05 mph gain).
   - *Published Historical Average (Li et al. 2018, Table 1)*: 15m MAE = 4.16 mph, 30m MAE = 4.16 mph, 60m MAE = 4.16 mph.
   - *Our Historical Average Baseline*: 15m MAE = 5.15 mph, 30m MAE = 5.15 mph, 60m MAE = **5.16 mph** on the full 34,272 dataset. At the 60-minute horizon on the full dataset, Historical Average outperforms all neural models (5.16 vs 6.75 mph) as traffic dynamics revert to daily seasonal means over long prediction horizons.
4. **Why Results Differ Between 3,000-Timestep and Full-Timeline Benchmarks**:
   - *Evaluation Period & Temporal Diversity*: The 3,000-step slice spans ~10.4 continuous days (March 2012) exhibiting high autocorrelation (favoring persistence: 3.64 mph at 15m). The full 34,272 timesteps span 4 months (March–June 2012), incorporating seasonal variations and broader traffic patterns where historical daily means become more competitive.
   - *Training Duration & Convergence*: The 3,000-step run trained for 2 epochs on 2,077 windows (4,154 sample updates); the full-timeline run executed 1 single epoch on 23,967 windows (~24,000 sample updates). Neither model reached full validation convergence due to CPU runtime limits (~10–15 min per model).
   - *Model Initialization & Checkpoint Selection*: The 3,000-step experiment evaluated 3 random seeds (`[42, 43, 44]`) with validation monitoring; the full-timeline run used a single seed (`42`) evaluated directly at the end of epoch 1.
   - *Normalization*: Train-only normalization parameters slightly differ ($\mu = 58.99, \sigma = 12.41$ on 3k slice vs $\mu = 58.58, \sigma = 12.82$ on full dataset), both strictly leakage-free.
5. **Missing-Value Evaluation Masking**:
   - Consistent with Li et al. (2018), missing sensors and zero-speed dropouts ($y_{\text{true}} \le 0.0$ or NaN) are masked across all models (4.96% masked in 3k slice; 12.12% masked in full dataset).
   - Verified via unit test fixture `test_hand_calculated_metric_fixture`.
6. **Chronological Splitting & Leakage Prevention**:
   - Chronological 70% train / 10% validation / 20% test split guarantees zero future data leakage.

### 5. Four-Tier Scientific Validation Framework
To prevent premature operational claims based merely on passing automated tests, this project strictly delineates four tiers of research maturity:

| Tier | Objective | Evaluated Domain | Validation Standard | Current Status |
|------|-----------|------------------|---------------------|:--------------:|
| **1. Engineering Verification** | Code correctness, mathematical invariants, metric calculations, and zero data leakage | Codebase unit & integration test suite | 90 automated pytest tests passing (`exit code 0`) | ✅ **Verified** |
| **2. Forecasting Validation** | Multi-horizon graph traffic forecasting accuracy on real sensor data | Real METR-LA highway loop detectors (207 sensors) | Controlled 3-seed benchmark (3k steps) + Preliminary 1-epoch full timeline | ⚠️ **Preliminary Benchmark** (Requires multi-epoch GPU scaling) |
| **3. Simulated Routing Evaluation** | Time-dependent dynamic routing & uncertainty-aware dispatch | Synthetic urban road network (6×6 grid, 45 scenarios) | Paired journey-time comparisons against 4 baselines | 🔬 **Simulated Validation** (Non-oracle adaptation confirmed) |
| **4. Real-World Deployment Readiness** | Operational deployment in municipal emergency ambulances | Real city street topology (e.g. OpenStreetMap) & live sensor feeds | Hardware-in-the-loop, GPS tracking, clinical trials | ❌ **Not Ready** (Academic research prototype only) |

---

## Part 2: Time-Dependent Ambulance Routing Methodology (Phase 3)

### 1. The Dynamic Routing Challenge
Ambulance journeys take physical time to traverse road segments. Applying a single static forecast evaluated at departure time $t_{\text{dep}}$ across all downstream segments creates a temporal mismatch: a segment reached 25 minutes into the journey will experience traffic corresponding to $t_{\text{dep}} + 25\text{ min}$, not $t_{\text{dep}}$.

### 2. Arrival-Time-Aware Edge Cost Formulation
Let $t_{\text{dep}}$ be the ambulance departure time in seconds. For any candidate path $P = (v_0, v_1, \dots, v_k)$, the accumulated arrival time at segment $e = (v_i, v_{i+1})$ is:
$$T_{\text{accumulated}} = \sum_{j=0}^{i-1} T(v_j, v_{j+1})$$
$$t_{\text{arr}}(e) = t_{\text{dep}} + T_{\text{accumulated}}$$

The corresponding discrete prediction horizon index $h_{\text{arr}}$ is computed using the model sampling rate $\Delta t = 300\text{ s}$ (5 minutes):
$$h_{\text{step}} = \text{round}\left(\frac{T_{\text{accumulated}}}{\Delta t}\right)$$

### 3. Horizon Fallback Protocol
If the estimated arrival step exceeds the model's supported horizon range ($H_{\max} = 12$, i.e. 60 minutes):
1. **Documented Fallback Mode `CLAMP_MAX_HORIZON` (Default)**:
   $$h_{\text{arr}} = \min(H_{\max}, \max(1, h_{\text{step}}))$$
   The engine uses the longest available forecast (step 12) to capture sustained downstream congestion trends.
2. **Documented Fallback Mode `HISTORICAL_AVERAGE`**:
   The engine reverts to the time-of-day seasonal historical average speed for arrival timestamps beyond 60 minutes.

### 4. Edge Traversal & Uncertainty-Aware Cost
The predicted speed at arrival is retrieved from the multi-horizon forecast matrix:
$$v_{\text{pred}}(e, h_{\text{arr}}) = \hat{Y}[e, h_{\text{arr}}]$$
The expected traversal time across physical segment length $L(e)$ is:
$$T(e) = \frac{L(e)}{v_{\text{pred}}(e, h_{\text{arr}})}$$

Under uncertainty-aware routing, the edge weight incorporates prediction variance:
$$w(u, v) = T(u, v) + \lambda_{\text{risk}} \cdot \sigma(u, v, h_{\text{arr}})$$
where $\sigma(u, v, h_{\text{arr}})$ is the epistemic uncertainty at the arrival horizon and $\lambda_{\text{risk}} = 0.5$.

### 5. Time-Dependent Dijkstra Algorithm
Unlike standard static Dijkstra, the search maintains a priority queue of state tuples $(c_{\text{accum}}, T_{\text{accum}}, u)$ where:
- $c_{\text{accum}}$ is the total cumulative routing cost.
- $T_{\text{accum}}$ is the physical elapsed journey time in seconds.
- Edge costs are dynamically evaluated at $T_{\text{accum}}$ for each outgoing edge $(u, v)$.
- Guarantees FIFO property (First-In-First-Out) since physical traversal times are strictly positive.

### 6. Evaluated Routing Strategies (5-Strategy Suite)
| Strategy ID | Strategy Name | Edge Cost Function $w(u, v)$ | Temporal Awareness |
|:-----------:|:--------------|:-----------------------------|:-------------------|
| **1** | Shortest Distance | Road length $L(u, v)$ | Static geometry only |
| **2** | Current Traffic | $\frac{L(u, v)}{v_{\text{current}}}$ | Static departure snapshot |
| **3** | Static Historical | $\frac{L(u, v)}{v_{\text{historical}}}$ | Static historical average |
| **4** | Time-Dependent Predicted | $\frac{L(u, v)}{v_{\text{pred}}(h_{\text{arr}})}$ | Arrival-time-aware dynamic forecast |
| **5** | Time-Dependent Uncertainty | $\frac{L(u, v)}{v_{\text{pred}}(h_{\text{arr}})} + \lambda \sigma(h_{\text{arr}})$ | Dynamic forecast + risk penalty |

### 7. En-Route Dynamic Reroute & No-Oracle Guarantee
- During vehicle transit, if real-time observations indicate downstream incident occurrence or heavy congestion deviation exceeding $\tau_{\text{reroute}} = 20\%$, the router re-computes an optimal path from the ambulance's current location $v_{\text{curr}}$ to the destination.
- **Zero Future Leakage**: At departure $t_0$, the router only has access to observations up to $t_0$ and forecasts generated from $t_0$. Future ground-truth incident injections at $t > t_0$ are completely inaccessible until the simulation clock reaches the incident timestamp.

### 8. Routing Integrity Audit Diagnoses & Empirical Resolution (Priority 3 Audit)
A rigorous scenario-by-scenario paired research audit was performed on `results/metrics/time_dependent_routing_results.json` across 45 emergency journeys (seeds `[42, 43, 44]`, 15 scenarios per seed):

#### A. Paired Journey-Time Difference Audit (Current Traffic vs Dynamic Methods)
The aggregate routing metrics and paired journey-time differences reveal crucial operational insights:
- **Mean Journey Times**:
  - *Current Traffic*: **1402.6 s** (Fastest overall mean)
  - *Time-Dependent Uncertainty-Aware*: **1403.8 s** (+1.2s difference, -0.09%)
  - *Time-Dependent Predicted*: **1410.0 s** (+7.4s difference, -0.53%)
  - *Static Historical*: **1571.6 s** (+169.0s slower)
  - *Shortest Distance*: **1614.0 s** (+211.4s slower)
- **90th Percentile Tail Latency (P90)**:
  - *Current Traffic*: **2010.6 s**
  - *Time-Dependent Uncertainty-Aware*: **2169.6 s**
  - *Time-Dependent Predicted*: **2247.4 s**
  - *Static Historical*: **2507.9 s**
  - *Shortest Distance*: **2674.4 s**

#### B. Paired Scenario Analysis Breakdown
Calculating paired journey-time differences ($TT_{\text{Current}} - TT_{\text{Strategy}}$) across all 45 individual journeys:
1. **Current Traffic vs Time-Dependent Predicted**:
   - Mean Difference: **-7.37 s** (Current Traffic is faster by 7.37s on average)
   - Median Difference: **0.00 s**
   - Standard Deviation: **71.46 s**
   - Scenario Outcomes: **2 Wins for Predicted / 41 Ties / 2 Losses for Predicted**
   - In 91.1% of journeys (41/45), Predicted and Current Traffic choose identical paths.
2. **Current Traffic vs Time-Dependent Uncertainty-Aware**:
   - Mean Difference: **-1.22 s** (Current Traffic is faster by only 1.22s on average)
   - Median Difference: **0.00 s**
   - Standard Deviation: **51.96 s**
   - Scenario Outcomes: **1 Win for Uncertainty / 43 Ties / 1 Loss for Uncertainty**
3. **Time-Dependent Predicted vs Uncertainty-Aware**:
   - Mean Difference: **+6.14 s** (Uncertainty-Aware is 6.14s faster than pure prediction on average)
   - P90 Tail Latency Reduction: Uncertainty-aware saves **-77.8 s** on the 90th percentile tail (2169.6s vs 2247.4s).

#### C. Where Predictions Help vs Where Current Traffic Wins
- **Where Predicted Traffic Helps**:
  - *Anticipated Bottleneck Clearance*: In Scenario `SEED_43_SCEN_14` (`N2_4` $\to$ `N3_0`), Current Traffic took an indirect 3019.7s detour due to a temporary incident. Predicted and Uncertainty-Aware routing correctly forecasted that the direct corridor would clear before arrival, achieving 2805.0s — **saving 214.7 seconds (~3.5 minutes)**!
  - *Dynamic Mid-Journey Incident Rerouting*: In controlled en-route incident injection, dynamic adaptation averted a blocked corridor upon downstream observation, cutting journey time by 50.0% (83.1s vs 166.2s).
  - *Vs Static Baselines*: Predicted routing beats Shortest Distance in 20 out of 45 scenarios (saving an average of **204.07 s / 3.4 minutes** per journey) and beats Static Historical Average by **161.6 s**.
- **Where Current Traffic Wins**:
  - *Stochastic Clearance Delays*: In Scenario `SEED_42_SCEN_04` (`N0_3` $\to$ `N3_0`), pure Predicted routing anticipated clearance along a corridor that cleared slower than expected in ground truth, suffering 2210.8s vs Current Traffic's safe 1885.5s (a 325.4s delay).
  - *Instantaneous Greediness*: On short-to-medium trips during stable congestion, the current instantaneous speed is an exceptionally strong predictor.
- **Why Uncertainty-Aware Routing Matters**:
  - In Scenario `SEED_42_SCEN_04`, because the clearing corridor exhibited high prediction variance, Uncertainty-Aware routing penalized the corridor and selected the safe current-traffic route (1885.5s), **completely averting the 325.4s delay suffered by pure prediction**!
  - Uncertainty-awareness acts as a risk hedge: it preserves large gains when confidence is high while avoiding catastrophic tail delays when prediction variance is elevated.

#### D. Controlled Uncertainty Route Selection Unit Tests
To mathematically verify route selection mechanics, two unit tests were implemented in `tests/test_time_dependent_routing.py`:
1. `test_controlled_uncertainty_changes_route`: Corridor A has travel time 50.0s with high uncertainty $\sigma = 40.0\text{s}$ (risk-adjusted cost = 62.0s with $\lambda = 0.3$). Corridor B has travel time 55.0s with low uncertainty $\sigma = 5.0\text{s}$ (risk-adjusted cost = 56.5s). When $\lambda_{\text{risk}} = 0.0$, Strategy 4 chooses Corridor A. When $\lambda_{\text{risk}} = 0.3$, Strategy 5 correctly flips to Corridor B.
2. `test_controlled_uncertainty_preserves_route`: Corridor A has travel time 30.0s with uncertainty $\sigma = 40.0\text{s}$ (cost = 42.0s). Corridor B has travel time 55.0s with low uncertainty $\sigma = 5.0\text{s}$ (cost = 56.5s). Because the speed advantage of Corridor A outweighs the risk penalty ($42.0 < 56.5$), Strategy 5 correctly preserves Corridor A.

#### E. Geographic Domain Boundary Protocol
METR-LA sensors represent loop detectors on multi-lane California freeways. They do NOT represent urban street intersections, and their numerical IDs (`773869`, `767541`, etc.) must never be treated as urban ambulance road segments. Emergency routing is evaluated strictly within procedural urban grid simulations with explicit graph topologies and road segment distances.

