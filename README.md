# Surakshit Bharat — Q-JEPA Traffic (V3.1 Master Model)

> **Predict Traffic. Protect Lives. Clear the Way.**

[![Tests](https://img.shields.io/badge/tests-135%2F135%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11-blue)](#requirements)
[![Data](https://img.shields.io/badge/data-real%20METR--LA%20%2B%20PEMS04%20%2B%20PEMS08-green)](#data)

An advanced traffic intelligence and emergency ambulance routing research platform combining the **Q-JEPA Master Foundation Model (V3.1)**, spatial graph representation learning, time-dependent routing, and Google Maps Platform integration.

---

## Features

- **Q-JEPA Master Foundation Model (V3.1)**: Spatiotemporal representation learning with causal gated dilated temporal convolutions, learnable adaptive graph adjacency, self-supervised EMA target prediction, and calibrated aleatoric uncertainty estimation
- **Verified Multi-Dataset Real Traffic**: Authentic Caltrans highway sensors on disk: METR-LA (207 sensors, 34,272 timesteps, speed in mph), PEMS04 (307 sensors, 16,992 timesteps, flow/speed), and PEMS08 (170 sensors, 17,856 timesteps, flow/speed)
- **Leave-One-Dataset-Out (LODO) Pretraining**: Pretrained on real PEMS04 and PEMS08 while strictly holding out METR-LA (`lodo_isolation_verified: true`)
- **Dynamic Time-Dependent Emergency Routing**: Arrival-time-aware edge cost indexing ($h_{\text{arr}} = \text{round}(T_{\text{accum}}/\Delta t)$), risk penalty hedging ($C(e, t) = \text{tt} + \beta \sigma$), and oscillation-dampened dynamic rerouting
- **7-Page Research Dashboard**: Command Center, Live Map, Route Comparison, AI Lab, Experiments, System Status, and Google Maps Integration
- **Google Maps Platform Runtime Integration**: Routes API v2 navigation and Maps JS API visualization with strict two-key separation, rate limiting, and graceful offline fallback
- **FastAPI REST Backend**: Comprehensive endpoints with schemas, secret redaction, and rate guards
- **135 Automated Tests** — all passing with exit code 0
- **Audited Reproducibility**: See [docs/V3_SYSTEM_AUDIT.md](docs/V3_SYSTEM_AUDIT.md), [docs/MULTI_DATASET_PRETRAINING.md](docs/MULTI_DATASET_PRETRAINING.md), and [docs/GOOGLE_MAPS_VERIFICATION.md](docs/GOOGLE_MAPS_VERIFICATION.md)

---

## Quick Start

### 1. Install dependencies
```bash
pip install streamlit plotly folium streamlit-folium fastapi uvicorn httpx pytest pyyaml pyarrow torch numpy pandas scikit-learn networkx opencv-python matplotlib
```

### 2. Generate synthetic data
```bash
python scripts/generate_demo_data.py
```

### 3. Train forecasting models
```bash
python scripts/train_baselines.py
```

### 4. Run benchmark experiments
```bash
python scripts/run_benchmarks.py
```

### 5. Start the API backend
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Run all tests
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
surakshit-bharat-q-jepa/
├── configs/              YAML configuration
├── data/
│   ├── synthetic/        Generated traffic datasets (CSV + Parquet)
│   └── raw/              External datasets (METR-LA etc. — not included)
├── models/checkpoints/   Saved model weights
├── src/
│   ├── common/           Config, logging, reproducibility
│   ├── simulation/       Road network, traffic simulator, scenarios, incidents
│   ├── data/             Loaders, splits, normalisation, window builders
│   ├── forecasting/      Persistence, HistAvg, LSTM, JEPA-Traffic, training
│   ├── routing/          5 routing algorithms + dynamic rerouter
│   ├── evaluation/       Forecasting & routing metrics
│   └── api/              FastAPI backend
├── dashboard/            Streamlit 6-page dashboard
├── tests/                33 automated tests
├── scripts/              generate_demo_data, train_baselines, run_benchmarks
├── results/              Experiment outputs, figures, reports
└── docs/                 Architecture, Methodology, Limitations, API docs
```

---

## API Reference (brief)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/system/status` | Network + model status |
| GET | `/network` | Full road network data |
| GET | `/traffic/current` | Current edge speeds |
| POST | `/traffic/forecast` | Traffic forecasts |
| POST | `/routes/compare` | Compare all 5 routers |
| POST | `/routes/ambulance` | Single ambulance route |
| POST | `/simulation/start` | Reset simulation |
| POST | `/simulation/step` | Advance one step |
| POST | `/simulation/incident` | Add incident |
| POST | `/experiments/run` | Run routing experiment |
| GET | `/metrics` | Forecasting metrics |
| GET | `/integrations/google/status` | Google Maps Platform configuration and offline status |
| POST | `/integrations/google/routes` | Compute traffic-aware routes via Google Routes API v2 |

Full API docs at `http://localhost:8000/docs` when backend is running.

---

## Google Maps Platform Integration

Surakshit Bharat Q-JEPA includes optional Google Maps Platform integration for real-world geographic route visualization and commercial traffic-aware baseline comparisons.

- **Two-Key Architecture**:
  - `GOOGLE_MAPS_SERVER_API_KEY`: Used strictly by FastAPI backend for the Routes API.
  - `GOOGLE_MAPS_BROWSER_API_KEY`: Restricted by HTTP referrer, used only by Streamlit for Maps JavaScript API tiles.
- **Offline / Fallback Resilience**: The entire application and test suite execute without internet access or API credentials. Unconfigured instances gracefully present an offline banner.
- **ToS & Intellectual Property Boundary**: In compliance with Google Maps Platform Terms of Service, route data (polylines, durations, distances) is **strictly display-only** and is never ingested into datasets or used to train or evaluate `JEPAGraph-FM`.
- **Setup Guide**: See [docs/GOOGLE_MAPS_SETUP.md](docs/GOOGLE_MAPS_SETUP.md) for step-by-step GCP configuration instructions.

---

## Scientific Research Question

> Can JEPA-style predictive traffic representations improve future traffic-state prediction and ambulance travel-time reliability compared with well-tuned conventional forecasting models, and does uncertainty-aware routing produce measurable benefits?

This is investigated as a **hypothesis** — results are measured from actual experiments, not assumed.

---

## Model Architecture: JEPA-Traffic

```
Observed History (T steps)  ──▶  [Online Encoder] ──▶  z_ctx
                                                           │
                                                    [Predictor]
                                                           │
                                                      z_pred  ──── MSE Loss ────┐
                                                                                │
Future Window (H steps)  ──▶  [Target Encoder] ──▶  z_tgt ─── stop_grad ───────┘

Target Encoder updated by EMA (τ=0.99), NOT by backprop.
Decoding Head trained separately: z_ctx ──▶ speed forecast.
```

---

## Known Limitations

- All traffic data is synthetic — not validated against real-world sensors
- JEPA model architecture is research-grade, not production-tested
- Quantum-inspired optimizer is classical simulated annealing — no quantum speedup
- Road network is a synthetic grid, not a real city
- No real signal pre-emption, dispatch integration, or GPS
- ETA estimates are advisory only — not for operational deployment

---

## Data Sources
 
| Dataset | Status on Disk | Type | Notes |
|---------|:--------------:|:----:|-------|
| **METR-LA** | ✅ Verified (69.11 MB) | Real Sensor | 207 loop detectors, 34,272 timesteps, speed in mph (CC BY 4.0) |
| **PEMS04** | ✅ Verified (31.43 MB) | Real Sensor | 307 loop detectors, 16,992 timesteps, flow/speed (Open Data) |
| **PEMS08** | ✅ Verified (17.67 MB) | Real Sensor | 170 loop detectors, 17,856 timesteps, flow/speed (Open Data) |
| **PEMS-BAY** | ⚠️ Synthetic Profile | Procedural | Download blocked by remote Zenodo 403; unit testing only |
| **PEMS03** | ⚠️ Synthetic Profile | Procedural | Download blocked by remote Zenodo 403; unit testing only |
| **PEMS07** | ⚠️ Synthetic Profile | Procedural | Download blocked by remote Zenodo 403; unit testing only |
| **Synthetic Grid** | ✅ Generated (Parquet/CSV)| Simulation | 30-node urban road network for ambulance routing benchmarks |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Surakshit Bharat Q-JEPA Traffic — Research Prototype v3.1.0*
