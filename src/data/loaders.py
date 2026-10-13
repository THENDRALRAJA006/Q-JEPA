"""
Data loading, validation, and train/val/test splitting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────

def load_synthetic_traffic(
    path: Path, fill_missing: bool = True
) -> pd.DataFrame:
    """Load synthetic traffic CSV or Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required_cols = ["step", "edge_id", "observed_speed_kmh", "gt_speed_kmh"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    if fill_missing:
        # Fill missing speed observations with group-mean imputation
        df["observed_speed_kmh"] = df.groupby("edge_id")["observed_speed_kmh"].transform(
            lambda x: x.fillna(x.mean())
        )
        df["flow_veh_per_hour"] = df.groupby("edge_id")["flow_veh_per_hour"].transform(
            lambda x: x.fillna(x.mean())
        )
        df["congestion_level"] = df["congestion_level"].fillna(0.0)

    logger.info(
        "Loaded dataset: %d rows, %d edges, steps %d–%d",
        len(df), df["edge_id"].nunique(), df["step"].min(), df["step"].max(),
    )
    return df


# ─────────────────────────────────────────────────
# Temporal splits (no leakage)
# ─────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by step index (temporal order) to prevent data leakage.
    All preprocessing parameters MUST be fit on train split only.
    """
    steps = sorted(df["step"].unique())
    n = len(steps)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_steps = steps[:n_train]
    val_steps = steps[n_train:n_train + n_val]
    test_steps = steps[n_train + n_val:]

    train_df = df[df["step"].isin(train_steps)].copy()
    val_df = df[df["step"].isin(val_steps)].copy()
    test_df = df[df["step"].isin(test_steps)].copy()

    logger.info(
        "Temporal split: train=%d steps, val=%d steps, test=%d steps",
        len(train_steps), len(val_steps), len(test_steps),
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────
# Normalization (fit on train only)
# ─────────────────────────────────────────────────

class SpeedNormalizer:
    """Min-max normalizer for edge speeds. Fit on training data only."""

    def __init__(self):
        self.speed_min: float = 0.0
        self.speed_max: float = 1.0

    def fit(self, df: pd.DataFrame) -> "SpeedNormalizer":
        self.speed_min = float(df["observed_speed_kmh"].min())
        self.speed_max = float(df["observed_speed_kmh"].max())
        logger.info("Normalizer fit: speed [%.2f, %.2f] km/h", self.speed_min, self.speed_max)
        return self

    def transform(self, speeds: np.ndarray) -> np.ndarray:
        denom = max(self.speed_max - self.speed_min, 1e-6)
        return (speeds - self.speed_min) / denom

    def inverse(self, normed: np.ndarray) -> np.ndarray:
        return normed * (self.speed_max - self.speed_min) + self.speed_min


# ─────────────────────────────────────────────────
# Sequence window builder
# ─────────────────────────────────────────────────

def build_windows(
    df: pd.DataFrame,
    edge_id: str,
    window_size: int = 12,
    horizon: int = 3,
    feature_col: str = "observed_speed_kmh",
    target_col: str = "gt_speed_kmh",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding windows for a single edge.
    X shape: (N, window_size), y shape: (N, horizon)
    No future information leaks into X.
    """
    edge_df = df[df["edge_id"] == edge_id].sort_values("step")
    speeds = edge_df[feature_col].values.astype(np.float32)
    targets = edge_df[target_col].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(len(speeds) - window_size - horizon + 1):
        X_list.append(speeds[i:i + window_size])
        y_list.append(targets[i + window_size:i + window_size + horizon])

    if not X_list:
        return np.empty((0, window_size)), np.empty((0, horizon))

    return np.array(X_list), np.array(y_list)


def build_all_edges_windows(
    df: pd.DataFrame,
    window_size: int = 12,
    horizon: int = 3,
    feature_col: str = "observed_speed_kmh",
    target_col: str = "gt_speed_kmh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windows across all edges."""
    X_parts, y_parts = [], []
    for eid in df["edge_id"].unique():
        X, y = build_windows(df, eid, window_size, horizon, feature_col, target_col)
        if len(X) > 0:
            X_parts.append(X)
            y_parts.append(y)
    if not X_parts:
        return np.empty((0, window_size)), np.empty((0, horizon))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ─────────────────────────────────────────────────
# Dataset readiness check
# ─────────────────────────────────────────────────

def check_dataset_readiness(data_dir: Path) -> Dict[str, str]:
    """Check which datasets are available and report readiness."""
    results = {}
    checks = {
        "synthetic_normal": [data_dir / "synthetic" / "synthetic_traffic_normal.parquet"],
        "synthetic_rush_hour": [data_dir / "synthetic" / "synthetic_traffic_rush_hour.parquet"],
        "synthetic_road_blockage": [data_dir / "synthetic" / "synthetic_traffic_road_blockage.parquet"],
        "metr_la": [
            data_dir / "metr_la" / "METR-LA.csv",
            data_dir / "metr_la" / "metr-la.h5",
            data_dir / "raw" / "metr-la.h5",
        ],
        "pems_bay": [
            data_dir / "raw" / "pems-bay.h5",
            data_dir / "pems_bay" / "pems-bay.h5",
        ],
    }
    for name, paths in checks.items():
        found = False
        for path in paths:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                results[name] = f"AVAILABLE ({size_kb:.0f} KB: {path.name})"
                found = True
                break
        if not found:
            results[name] = "NOT FOUND"
    return results

