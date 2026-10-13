"""
Evaluation metrics for forecasting and routing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────
# Forecasting metrics
# ─────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float((np.abs(y_true - y_pred) / (np.abs(y_true) + eps)).mean() * 100.0)


def per_horizon_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> List[Dict[str, float]]:
    """Compute MAE and RMSE for each forecast horizon separately."""
    horizon = y_true.shape[1] if y_true.ndim > 1 else 1
    results = []
    for h in range(horizon):
        yt = y_true[:, h] if y_true.ndim > 1 else y_true
        yp = y_pred[:, h] if y_pred.ndim > 1 else y_pred
        results.append({"horizon": h + 1, "MAE": mae(yt, yp), "RMSE": rmse(yt, yp)})
    return results


def compute_all_forecast_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
) -> Dict:
    return {
        "model": model_name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "per_horizon": per_horizon_metrics(y_true, y_pred),
    }


# ─────────────────────────────────────────────────
# Routing metrics
# ─────────────────────────────────────────────────

def routing_metrics(journey_times_s: List[float]) -> Dict[str, float]:
    arr = np.array(journey_times_s)
    feasible = arr[np.isfinite(arr)]
    return {
        "mean_s": float(feasible.mean()) if len(feasible) else float("nan"),
        "median_s": float(np.median(feasible)) if len(feasible) else float("nan"),
        "p90_s": float(np.percentile(feasible, 90)) if len(feasible) else float("nan"),
        "unreachable_rate": float((~np.isfinite(arr)).sum() / max(len(arr), 1)),
        "n_scenarios": len(arr),
    }


def eta_error(predicted_s: float, actual_s: float) -> float:
    return abs(predicted_s - actual_s)


def journey_time_improvement_pct(baseline_s: float, improved_s: float) -> float:
    if baseline_s <= 0 or not np.isfinite(baseline_s):
        return float("nan")
    return 100.0 * (baseline_s - improved_s) / baseline_s
