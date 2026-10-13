# Architecture — Surakshit Bharat Q-JEPA Traffic

## Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SURAKSHIT BHARAT Q-JEPA TRAFFIC                   │
│                  Predict Traffic. Protect Lives. Clear the Way.      │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐    ┌───────────────┐    ┌────────────────────────┐
  │  Traffic    │    │  Forecasting  │    │   Routing Engine       │
  │  Simulator  │───▶│  Pipeline     │───▶│                        │
  │             │    │               │    │  A: Shortest Distance  │
  │  Scenarios  │    │  Persistence  │    │  B: Current Traffic    │
  │  Incidents  │    │  Hist. Avg    │    │  C: Predicted Traffic  │
  │  Road Graph │    │  LSTM         │    │  D: Uncertainty-Aware  │
  └─────────────┘    │  JEPA-Traffic │    │  E: Quantum-Inspired   │
                     └───────────────┘    └────────────────────────┘
                              │                       │
                              ▼                       ▼
                     ┌────────────────────────────────────────┐
                     │           FastAPI Backend               │
                     │   /health  /network  /routes/compare   │
                     │   /traffic/forecast  /experiments/run  │
                     └────────────────────────────────────────┘
                                        │
                                        ▼
                     ┌────────────────────────────────────────┐
                     │       Streamlit Dashboard (6 pages)     │
                     │  Command Center | Live Map | Routes     │
                     │  AI Lab | Experiments | System Status   │
                     └────────────────────────────────────────┘
```

## Data Flow

1. **Synthetic traffic generation** → CSV/Parquet files in `data/synthetic/`
2. **Temporal split** (70/15/15 by step) → train/val/test — no leakage
3. **Normalisation** — SpeedNormalizer fit on training data only
4. **Window building** — (N, T) context → (N, H) target for each edge
5. **Model training** — four forecasters independently trained
6. **Inference** — predicted travel times written back to road network edges
7. **Routing** — five algorithms query the road graph with current/predicted times
8. **Evaluation** — metrics computed on held-out test scenarios

## Key Design Decisions

- **Ground truth vs observations**: Simulator maintains `gt_speed_kmh` separate from `observed_speed_kmh`. Models only see observed data during training.
- **Modular forecasters**: All share `BaseForecaster` interface — swap without rewriting routing.
- **EMA target encoder**: JEPA target encoder updated after each optimiser step, never by backprop.
- **Graceful degradation**: Every component falls back gracefully when optional dependencies are missing.
- **No fabricated results**: Metrics files are generated from actual training runs.

## Module Map

```
src/
  common/        config.py  logging.py  reproducibility.py
  simulation/    road_network.py  traffic_simulator.py  incident_generator.py  ambulance_scenarios.py
  data/          loaders.py  (splits, normalisation, window builders)
  forecasting/   persistence.py  lstm_model.py  jepa_encoder.py  train.py
  routing/       shortest_path.py  (all 5 routers + DynamicRouter)
  evaluation/    prediction_metrics.py  (MAE, RMSE, routing metrics)
  api/           main.py  (FastAPI endpoints)
dashboard/       app.py  (Streamlit 6-page app)
tests/           test_all.py  (33 tests)
scripts/         generate_demo_data.py  train_baselines.py  run_benchmarks.py
```
