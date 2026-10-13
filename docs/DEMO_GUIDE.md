# Demo Guide — Surakshit Bharat Q-JEPA Traffic

## Step-by-Step Demonstration Script

### Prerequisites
All dependencies installed, synthetic data generated, models trained.

---

### Step 1: Start the API Backend
```bash
cd surakshit-bharat-q-jepa
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify: Open http://localhost:8000/docs — you should see the Swagger UI.

---

### Step 2: Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open: http://localhost:8501

---

### Step 3: Command Center Tour
1. Open browser to http://localhost:8501
2. You see: **network stats**, **road map**, **ambulance origin/destination selector**
3. Click **"🔄 Simulate Step"** — watch congestion update
4. The route box shows the fastest current-traffic route and ETA

---

### Step 4: Live Traffic Simulation
1. Click **"🗺️ Live Traffic Simulation"** in the sidebar
2. Select scenario: **"Rush Hour"**
3. Click **"🔄 Reset Simulation"**
4. Click **"▶️ Run Steps"** (set to 10 steps)
5. Watch congestion increase in the histogram
6. Select an edge → click **"💥 Add Incident"**
7. Observe the incident marker on the map

---

### Step 5: Route Comparison
1. Click **"🔀 Route Comparison"**
2. Origin: N0_0 (top-left), Destination: N4_5 (bottom-right)
3. Adjust **Risk Penalty** slider to 0.5
4. Click **"🔀 Compare All Routes"**
5. Observe the comparison table showing all 5 routers
6. The bar chart shows travel time differences

---

### Step 6: AI Research Lab
1. Click **"🔬 AI Research Lab"**
2. See the model comparison table with actual measured MAE/RMSE
3. Expand "JEPA-Traffic Architecture" to see the design details
4. Adjust the forecast horizon slider to see the demo forecast plot

---

### Step 7: Run an Experiment
1. Click **"⚗️ Experiment Center"**
2. Select Scenario: **"road_blockage"**, Seed: 42, Steps: 50
3. Click **"▶️ Run Experiment"**
4. Watch progress bar complete
5. See: congestion-over-time plot + route comparison table
6. Click **"📥 Export JSON"** to download results

---

### Step 8: System Status Check
1. Click **"⚙️ System & Data"**
2. Verify: synthetic datasets ✅ available
3. Verify: LSTM ✅, JEPA ✅ checkpoints present
4. Review known limitations section

---

## Key Things to Highlight

| Feature | What to Show |
|---------|-------------|
| Simulation data | "SIMULATED DATA" badge on every page |
| JEPA novelty | Architecture expander in AI Lab |
| Honest results | Model comparison shows LSTM ≈ JEPA — not fabricated |
| Routing comparison | All 5 algorithms side by side |
| Safety label | "ADVISORY — Not for operational use" on routes |
| Reproducibility | Commands in System page |

## Common Questions

**Q: Is this real data?**
A: No — all data is synthetically generated. The dashboard clearly labels this everywhere.

**Q: Does the quantum optimizer outperform Dijkstra?**
A: On small networks, no — results are equivalent. This is documented honestly in the report.

**Q: Has JEPA been proven better than LSTM?**
A: On this synthetic dataset, LSTM slightly outperforms JEPA. Real-world comparison
   requires validation on METR-LA or PeMS-BAY datasets — future work.
