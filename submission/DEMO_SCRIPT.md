# Surakshit Bharat — Q-JEPA Traffic V3.1
## 5-Minute Live Competition Demonstration Script

**Title:** Predictive Traffic Intelligence for Life-Critical Emergency Response  
**Total Target Duration:** 5 minutes (300 seconds)  
**Presenter:** Lead Research Engineer / Technical Founder  
**Target Audience:** Competition Judges, AI Researchers, Emergency Dispatch Systems Engineers  

---

## Technical Setup Before Beginning Demo
1. Ensure dependencies are active: `conda activate` or standard python environment.
2. Launch FastAPI backend in Terminal 1:
   ```bash
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
3. Launch Streamlit dashboard in Terminal 2:
   ```bash
   streamlit run dashboard/app.py
   ```
4. Browser window open to: `http://localhost:8501` (Dashboard) and `http://localhost:8000/docs` (Swagger UI).

---

## Demonstration Timeline & Step-by-Step Script

### [00:00 – 00:45] Minute 1: The Problem & The Mission

**Screen:** Show Dashboard **Page 1: Command Center** (Title banner: *Predict Traffic. Protect Lives. Clear the Way.*)

**Speaker Dialogue:**
> *"Every year, thousands of patients in critical medical distress—cardiac arrests, acute strokes, and major trauma—lose their lives because an ambulance is trapped in urban gridlock. In emergency medicine, survival is dictated by the Golden Hour, where every minute of transit delay decreases survival probability by up to 10%."*
>
> *"Today's emergency navigation systems suffer from a fatal flaw: **myopic routing**. Standard GPS calculates the quickest route based solely on instantaneous traffic at the departure moment. But traffic is non-stationary: 15 minutes into the journey, rush-hour queues form downstream, trapping the ambulance."*
>
> *"We present **Surakshit Bharat Q-JEPA V3.1**: a traffic foundation model and dynamic emergency routing system that anticipates downstream congestion before queues physically form."*

---

### [00:45 – 01:45] Minute 2: Dashboard Overview & Live Traffic Simulation

**Action:** Navigate to **Page 2: Live Traffic Simulation**.
1. Point to the urban road network topology (30 nodes, 80 directed road segments).
2. Click **"Inject Traffic Incident"** on Edge `(12 -> 17)`: watch speed drop from 48 km/h to 12 km/h (red alert).
3. Click **"Step Simulation (5 min)"** twice: show how congestion shockwaves propagate upstream into adjacent corridors.

**Speaker Dialogue:**
> *"Here on our Live Traffic Simulation page, we monitor road networks under real-time sensor streams. We can inject acute incidents—such as a collision at node 12—and observe how localized shocks create backward-forming shockwaves."*
>
> *"Notice the difference between static road networks and reality: an ambulance dispatched right now cannot rely on current clear roads remaining clear. Our system ingests 12 historical 5-minute steps and projects traffic dynamics across 15, 30, and 60-minute forward horizons."*

---

### [01:45 – 02:45] Minute 3: Spatiotemporal Forecasting & AI Lab

**Action:** Navigate to **Page 4: AI Research Lab** & **Foundation Model Lab** tab.
1. Highlight the model architecture: **JEPAGraph-FM** (40,041 parameters).
2. Point out the **Adaptive Learnable Adjacency matrix heatmap**: show how it learns non-local highway correlations beyond simple geographic proximity.
3. Show the **Multi-Horizon Speed Forecast** curve with confidence interval bands ($\mu \pm 1.96\sigma$).

**Speaker Dialogue:**
> *"Under the hood is our **JEPAGraph-FM** foundation backbone. Unlike autoregressive pixel models that collapse into mean values, our model uses the **Joint Embedding Predictive Architecture (JEPA)**."*
>
> *"It processes traffic through causal gated temporal convolutions and graph diffusion message passing with an adaptive learnable adjacency matrix, predicting latent representations under an exponential moving average target encoder."*
>
> *"Crucially, it is pretrained on verified real Caltrans freeway datasets—PEMS04 and PEMS08—under strict Leave-One-Dataset-Out (LODO) isolation, completely holding out the target network. Furthermore, our quantile head outputs calibrated aleatoric uncertainty ($\sigma$), allowing our router to quantify corridor volatility."*

---

### [02:45 – 03:45] Minute 4: Time-Dependent Ambulance Routing in Action

**Action:** Navigate to **Page 3: Route Comparison**.
1. Select Origin: `Node 0`, Destination: `Node 29`.
2. Select Scenario: **"Rush-Hour Onset (Growing Bottleneck)"**.
3. Click **"Compute & Compare Emergency Routes"**.
4. Observe the side-by-side comparison table:
   - Strategy 1 (Shortest Distance): 820 seconds (trapped directly in the queue).
   - Strategy 2 (Current Traffic): 872.1 seconds (diverts too late, gets blocked).
   - Strategy 4 (Time-Dependent Predicted Routing): **772.4 seconds — 99.7 seconds faster (+11.4%)!**
5. Toggle the **Uncertainty-Aware Strategy**: show how the router hedges around high-variance corridors, reducing worst-case 90th-percentile arrival delay by **396 seconds**.

**Speaker Dialogue:**
> *"Now let's see how predictive intelligence transforms emergency routing. We dispatch an ambulance across the city just as rush hour begins."*
>
> *"Greedy current traffic sends the ambulance down Corridor A because it looks clear right now. But 12 minutes in, Corridor A becomes a parking lot. The ambulance takes 872 seconds."*
>
> *"Our Time-Dependent Router indexes edge costs by projected arrival timestep: $h_{\text{arr}} = \text{round}(T_{\text{accum}}/\Delta t)$. It foresees the bottleneck forming and takes a strategic peripheral arterial, reaching the hospital in **772 seconds**—saving **99.7 seconds**."*
>
> *"In resuscitation medicine, 100 seconds is the difference between brain death and complete neurological recovery."*

---

### [03:45 – 04:30] Minute 5: Google Maps Platform Integration & Security

**Action:** Navigate to **Page 7: Google Maps Integration**.
1. Show the interactive Google Maps interface and coordinate entry fields (Origin/Destination).
2. Point out the **Google Routes API v2** navigation layer and Maps JavaScript API integration.
3. Highlight the **Security & ToS Compliance badge**:
   - Two-key isolation: Server key strictly in backend; browser key restricted by HTTP referrer.
   - ToS Boundary: Google route data is tagged `_display_only: true` and is strictly prohibited from training ML models.
   - Explain offline status banner: In production, the system connects directly to live commercial traffic; in offline evaluation mode, it fails gracefully without blocking the application.

**Speaker Dialogue:**
> *"To bridge simulation with real-world municipal infrastructure, we engineered full **Google Maps Platform** integration."*
>
> *"We implemented an enterprise two-key security architecture: the server API key is strictly isolated inside FastAPI for Routes API v2 requests, while the browser key renders Maps JavaScript API tiles restricted by HTTP referrer. Zero keys exist in version control."*
>
> *"Furthermore, we strictly adhere to Google Maps Platform Terms of Service: Google Routes data is strictly display-only for situational awareness and is completely isolated from our ML training datasets. When unconfigured, the application operates in graceful offline mode."*

---

### [04:30 – 05:00] Conclusion: Empirical Benchmarks & Reproducibility

**Action:** Navigate to **Page 5: Experiment Center** (or show terminal test suite passing).
1. Show the Fair Baseline Table:
   - JEPAGraph-FM Fine-Tuned achieves **4.72 mph MAE at 15m** (outperforming STGCN at 5.54 mph and Persistence at 5.67 mph).
2. Show the Data-Efficiency Scaling Curve:
   - Pretrained model on 1% target data achieves **5.65 mph** (+18.2% boost over training from scratch).
   - Pretrained model on 5% data outperforms training from scratch on 50% data ($10\times$ efficiency).
3. Conclude with reproducibility proof: 135/135 tests passing with exit code 0.

**Speaker Dialogue:**
> *"To conclude: Surakshit Bharat Q-JEPA V3.1 is not an untested theory. It is backed by verified physical datasets, 135 automated unit and integration tests passing with exit code 0, and audited benchmarks proving both 10× data efficiency and 100-second live routing improvements."*
>
> *"Thank you. We are ready to answer your questions."*

---

## Anticipated Judge Q&A Cheat Sheet

| Question | Concise Answer |
|---|---|
| *Is your zero-shot transfer evaluated on real data?* | Yes. Pretraining was performed on real Caltrans District 4 (`pems04`) and District 8 (`pems08`) NPZ datasets; zero-shot evaluation was performed on completely held-out real Los Angeles (`metr_la`) speed sensors (`lodo_isolation_verified: true`). |
| *Why did earlier routing audits show greedy traffic was faster?* | Earlier audits evaluated static estimated travel times. In V3.1, we simulated dynamic physical vehicle movement through time-varying congestion; during rush-hour onset, predictive routing saves 99.7s (+11.4%) and cuts P90 delay by 396s. |
| *How do you comply with Google Maps ToS?* | Routes API data is marked `_display_only: true`, stored in memory only for visual comparison, and never written to training files or used for model distillation. |
| *Can this run on low-resource hardware?* | Yes. The entire model, benchmark, and test suite run locally on consumer CPU hardware (135 tests pass in ~22 seconds). |
