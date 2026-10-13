# Multi-Dataset Self-Supervised Traffic Pretraining (JEPAGraph-FM)

**Project:** Surakshit Bharat — Q-JEPA Traffic  
**Repository:** `D:\ZYNEX\surakshit-bharat-q-jepa`  
**Date:** 13 October 2026  
**Document Version:** 2.0.0  

---

## 1. Overview & Training Protocols

To learn generalizable spatiotemporal representations that transfer across diverse road networks, `JEPAGraph-FM` implements a unified multi-dataset pretraining framework across 6 real-world benchmark datasets:
- **Speed Benchmarks:** METR-LA ($N=207$), PEMS-BAY ($N=325$)
- **Flow Benchmarks:** PEMS03 ($N=358$), PEMS04 ($N=307$), PEMS07 ($N=883$), PEMS08 ($N=170$)

---

## 2. The Four Training Protocols

### Protocol A: Multi-Dataset Balanced Pretraining
Pretraining is executed strictly on the chronological training splits ($70\%$ or $60\%$) of source datasets.
- **Dataset-Aware Temperature Sampling:** Because datasets differ in length ($T \in [16992, 52116]$) and sensor count ($N \in [170, 883]$), uniform batch sampling would bias representations toward larger networks. We sample batches across datasets according to temperature-smoothed probabilities:
$$p_d = \frac{S_d^\alpha}{\sum_{k=1}^K S_k^\alpha}, \quad \text{with } \alpha = 0.5$$
where $S_d$ is the number of samples in dataset $d$.
- **Batch Collation Invariance:** Each mini-batch is sampled homogeneously from one dataset graph ($B \times T \times N_d \times C$). This eliminates tensor padding overhead, preserves exact spatial adjacency operations, and guarantees linear memory scaling.

### Protocol B: Target-Specific Fine-Tuning
A pretrained checkpoint is fine-tuned on the chronological training partition of a specific downstream target network.
- Learning rate is reduced ($5 \times 10^{-4} \rightarrow 1 \times 10^{-4}$).
- Supervised Huber loss is applied on valid target observations.
- Validation loss triggers early stopping and checkpoint selection.

### Protocol C: Leave-One-Dataset-Out (LODO) Transfer
To evaluate true zero-shot and few-shot transfer:
1. Target dataset $D_{\text{target}}$ is completely excluded from source pretraining.
2. The foundation model pretrains on the remaining mixture.
3. **Zero-Shot Evaluation:** Direct inference on $D_{\text{target}}$ using its spatial graph adjacency without target gradient updates.
4. **Few-Shot Fine-Tuning:** Fine-tuning with restricted target training budgets ($1\%, 5\%, 10\%, 25\%$).

### Protocol D: Data Efficiency Scaling
Compares sample efficiency between:
- **Pretrained Foundation Model** (initialized with multi-dataset SSL weights)
- **Scratch Baseline** (randomly initialized identical architecture)
Evaluated across fixed fractions of target training data: $\mathcal{F} \in \{1\%, 5\%, 10\%, 25\%, 50\%, 100\%\}$.

---

## 3. Loss Formulations & Multi-Objective Suite

The total pretraining objective balances four complementary signals:
$$\mathcal{L}_{\text{total}} = w_{\text{jepa}} \mathcal{L}_{\text{jepa}} + w_{\text{mask}} \mathcal{L}_{\text{mask}} + w_{\text{sup}} \mathcal{L}_{\text{sup}} + w_{\text{temp}} \mathcal{L}_{\text{temp}}$$

### Objective 1: Future Latent Representation Prediction ($\mathcal{L}_{\text{jepa}}$)
$$\mathcal{L}_{\text{jepa}} = \frac{1}{B \cdot T_{\text{out}} \cdot N} \sum_{b, t, n} \text{SmoothL1}\left( \hat{Z}_{b, t, n}, \; \text{sg}[Z^{\text{tgt}}_{b, t, n}] \right)$$
where $\text{sg}[\cdot]$ denotes stop-gradient and $\hat{Z}$ is the output of the latent trajectory predictor.

### Objective 2: Spatiotemporal Masked Reconstruction ($\mathcal{L}_{\text{mask}}$)
A subset of observations is randomly masked ($\rho_{\text{mask}} = 15\% - 20\%$). The reconstruction loss is evaluated **strictly on masked valid observations**:
$$\mathcal{L}_{\text{mask}} = \frac{\sum_{b, t, n} M^{\text{mask}}_{b, t, n} \cdot M^{\text{valid}}_{b, t, n} \cdot \text{SmoothL1}(\hat{X}_{b, t, n}, X_{b, t, n})}{\sum_{b, t, n} M^{\text{mask}}_{b, t, n} \cdot M^{\text{valid}}_{b, t, n} + \epsilon}$$

### Objective 3: Supervised Multi-Horizon Forecasting ($\mathcal{L}_{\text{sup}}$)
Evaluated on valid future physical targets:
$$\mathcal{L}_{\text{sup}} = \frac{\sum_{b, t, n} M^{\text{valid}}_{b, t, n} \cdot \text{SmoothL1}(\hat{Y}_{b, t, n}, Y_{b, t, n})}{\sum_{b, t, n} M^{\text{valid}}_{b, t, n} + \epsilon}$$

### Objective 4: Multiscale Temporal Smoothness ($\mathcal{L}_{\text{temp}}$)
$$\mathcal{L}_{\text{temp}} = \frac{1}{B \cdot (T_{\text{out}}-1) \cdot N} \sum_{b, t, n} \left( \hat{Y}_{b, t+1, n} - \hat{Y}_{b, t, n} \right)^2$$

---

## 4. Empirical Pretraining Results (Tiny Profile, CPU)

Execution of `scripts/pretrain_jepagraph.py` across `[metr_la, pems_bay, pems04, pems08]`:
- **Epoch 1:** Train Loss: `0.7250` (JEPA: 0.2560, Mask: 0.2073, Sup: 0.3650) | Val Loss: `0.5166`
- **Epoch 2:** Train Loss: `0.5015` (JEPA: 0.1552, Mask: 0.1443, Sup: 0.2732) | Val Loss: `0.4372`
- **Epoch 3:** Train Loss: `0.4449` (JEPA: 0.1316, Mask: 0.1310, Sup: 0.2466) | Val Loss: `0.4000`
- **Total Duration:** 31.0 seconds (10.3s / epoch on CPU)
- **Best Model Checkpoint:** `models/foundation/jepagraph_fm_tiny.pt`
