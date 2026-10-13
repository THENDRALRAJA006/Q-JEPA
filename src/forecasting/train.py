"""
Training pipeline for all forecasting models.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.data.loaders import (
    load_synthetic_traffic, temporal_split,
    SpeedNormalizer, build_all_edges_windows,
)
from src.forecasting.persistence import PersistenceForecaster, HistoricalAverageForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.jepa_encoder import JEPAForecaster
from src.common.config import config, abs_path
from src.common.reproducibility import set_seed
from src.common.logging import get_logger

logger = get_logger(__name__)


def train_all(seed: int = 42, epochs_override: int = None) -> dict:
    """
    Train all available forecasters. Returns a dict of metrics.
    """
    cfg = config()
    set_seed(seed)

    data_path = abs_path(cfg.data.synthetic_dir) / "synthetic_traffic_normal.parquet"
    if not data_path.exists():
        logger.error("Dataset not found: %s — run scripts/generate_demo_data.py first", data_path)
        return {}

    logger.info("Loading dataset: %s", data_path)
    df = load_synthetic_traffic(data_path)

    train_df, val_df, test_df = temporal_split(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
    )

    # Fit normalizer on training data only
    norm = SpeedNormalizer().fit(train_df)

    window_size = cfg.forecasting.window_size
    horizon = cfg.forecasting.horizons[0]  # primary horizon
    epochs = epochs_override or cfg.forecasting.lstm_epochs

    def normalize_df(split_df):
        out = split_df.copy()
        out["observed_speed_kmh"] = norm.transform(out["observed_speed_kmh"].values)
        out["gt_speed_kmh"] = norm.transform(out["gt_speed_kmh"].values)
        return out

    train_norm = normalize_df(train_df)
    val_norm = normalize_df(val_df)
    test_norm = normalize_df(test_df)

    X_tr, y_tr = build_all_edges_windows(train_norm, window_size=window_size, horizon=horizon)
    X_val, y_val = build_all_edges_windows(val_norm, window_size=window_size, horizon=horizon)
    X_te, y_te = build_all_edges_windows(test_norm, window_size=window_size, horizon=horizon)

    logger.info("Windows — train: %s, val: %s, test: %s", X_tr.shape, X_val.shape, X_te.shape)

    ckpt_dir = abs_path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── Persistence ──
    logger.info("--- Persistence baseline ---")
    perf = PersistenceForecaster().fit(X_tr, y_tr)
    preds = perf.predict_horizon(X_te, horizon)
    mae, rmse = _metrics(y_te, preds)
    results["persistence"] = {"MAE": mae, "RMSE": rmse}
    logger.info("Persistence: MAE=%.4f RMSE=%.4f", mae, rmse)

    # ── Historical Average ──
    logger.info("--- Historical Average baseline ---")
    hist = HistoricalAverageForecaster().fit(X_tr, y_tr)
    preds = hist.predict_horizon(X_te, horizon)
    mae, rmse = _metrics(y_te, preds)
    results["historical_average"] = {"MAE": mae, "RMSE": rmse}
    logger.info("Historical Average: MAE=%.4f RMSE=%.4f", mae, rmse)

    # ── LSTM ──
    logger.info("--- LSTM ---")
    lstm = LSTMForecaster(
        window_size=window_size, horizon=horizon,
        hidden=cfg.forecasting.lstm_hidden,
        layers=cfg.forecasting.lstm_layers,
        dropout=cfg.forecasting.lstm_dropout,
        epochs=epochs, lr=cfg.forecasting.lstm_lr,
        batch_size=cfg.forecasting.lstm_batch,
    )
    lstm.fit(X_tr, y_tr, X_val, y_val)
    preds = lstm.predict(X_te)
    mae, rmse = _metrics(y_te, preds)
    results["lstm"] = {
        "MAE": mae, "RMSE": rmse,
        "train_losses": lstm.train_losses, "val_losses": lstm.val_losses,
    }
    lstm.save(ckpt_dir / "lstm.pt")
    logger.info("LSTM: MAE=%.4f RMSE=%.4f", mae, rmse)

    # ── JEPA ──
    logger.info("--- JEPA-Traffic ---")
    jepa = JEPAForecaster(
        window_size=window_size, horizon=horizon,
        embed_dim=cfg.forecasting.jepa_embed_dim,
        pred_dim=cfg.forecasting.jepa_pred_dim,
        ema_tau=cfg.forecasting.jepa_ema_tau,
        epochs=epochs, lr=cfg.forecasting.jepa_lr,
        batch_size=cfg.forecasting.jepa_batch,
    )
    jepa.fit(X_tr, y_tr, X_val, y_val)
    preds = jepa.predict(X_te)
    mae, rmse = _metrics(y_te, preds)
    results["jepa"] = {
        "MAE": mae, "RMSE": rmse,
        "jepa_losses": jepa.jepa_losses, "head_losses": jepa.head_losses,
        "val_losses": jepa.val_losses,
    }
    jepa.save(ckpt_dir / "jepa.pt")
    logger.info("JEPA: MAE=%.4f RMSE=%.4f", mae, rmse)

    # Save results
    out_dir = abs_path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": seed,
        "window_size": window_size,
        "horizon": horizon,
        "epochs": epochs,
        "normalizer": {"min": norm.speed_min, "max": norm.speed_max},
        "note": "All metrics on normalised speeds. Denormalize for km/h.",
        "results": {k: {m: v for m, v in vals.items() if isinstance(v, float)}
                    for k, vals in results.items()},
    }
    with open(out_dir / "forecasting_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metrics saved to %s", out_dir / "forecasting_metrics.json")
    return results


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_true - y_pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return mae, rmse


if __name__ == "__main__":
    results = train_all()
    print("\n=== Forecasting Results ===")
    for model, metrics in results.items():
        print(f"  {model}: MAE={metrics.get('MAE', 'N/A'):.4f}  RMSE={metrics.get('RMSE', 'N/A'):.4f}")
