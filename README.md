# Surakshit Bharat — Q-JEPA Traffic

> **Predict Traffic. Protect Lives. Clear the Way.**

[![Tests](https://img.shields.io/badge/tests-33%2F33%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11-blue)](#requirements)
[![Data](https://img.shields.io/badge/data-synthetic-orange)](#data)

An AI-powered ambulance routing research platform using JEPA-style predictive representations, uncertainty-aware routing, and classical quantum-inspired optimization.

> ⚠️ **All data is synthetically generated. This is a research prototype. Not for operational use.**

---

## Features

- **30-node synthetic road network** (5×6 grid) with time-varying traffic
- **7 emergency scenarios**: normal, rush-hour, blockage, queue spillback, missing data, noisy sensors, dynamic rerouting
- **5 routing algorithms**: Shortest Distance, Current Traffic, Predicted Traffic, Uncertainty-Aware, Quantum-Inspired (SA)
- **4 forecasting models**: Persistence, Historical Average, LSTM (with MC-Dropout uncertainty), JEPA-Traffic
- **JEPA-style representation learning**: online/target encoder with EMA updates
- **6-page Streamlit dashboard**: Command Center, Live Map, Route Comparison, AI Lab, Experiments, System Status
- **FastAPI REST backend** with full endpoint coverage
- **33 automated tests** — all passing

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

Full API docs at `http://localhost:8000/docs` when backend is running.

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

| Dataset | Status | Notes |
|---------|--------|-------|
| Synthetic (this project) | ✅ Generated | 30-node grid, 288 time steps |
| METR-LA | ❌ Not downloaded | Uber/Li et al. 2018 |
| PeMS-BAY | ❌ Not downloaded | Caltrans/DCRNN |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Surakshit Bharat Q-JEPA Traffic — Research Prototype v0.1.0*
