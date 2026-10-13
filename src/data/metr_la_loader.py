"""
METR-LA Dataset Loader and Preprocessing Pipeline.

Supports:
- Sensor metadata and graph adjacency loading from adj_mx_METR-LA.pkl
- Speed matrix loading from METR-LA.csv
- Chronological train/val/test split (70% / 10% / 20%)
- Train-only standard scaling (Z-score) to prevent data leakage
- Sliding window tensor generation: T_in=12 (60 min) -> T_out=12 (60 min)
- Horizons: 3 steps (15m), 6 steps (30m), 12 steps (60m)
- Masked metric evaluation ignoring missing values (speed <= 0 or NaN)
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    train_x: np.ndarray  # [N_train, T_in, num_nodes, num_features]
    train_y: np.ndarray  # [N_train, T_out, num_nodes]
    val_x: np.ndarray    # [N_val, T_in, num_nodes, num_features]
    val_y: np.ndarray    # [N_val, T_out, num_nodes]
    test_x: np.ndarray   # [N_test, T_in, num_nodes, num_features]
    test_y: np.ndarray   # [N_test, T_out, num_nodes]
    mean: float
    std: float
    sensor_ids: List[str]
    adj_matrix: np.ndarray


class TrafficWindowDataset(Dataset):
    """PyTorch Dataset for multi-horizon traffic forecasting."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: shape [N, T_in, num_nodes, num_features] or [N, T_in, num_nodes]
            y: shape [N, T_out, num_nodes]
        """
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class METRLALoader:
    """Loader for the Los Angeles highway traffic sensor dataset (METR-LA)."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/metr_la",
        input_window: int = 12,
        output_window: int = 12,
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
    ):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "METR-LA.csv"
        self.adj_path = self.data_dir / "adj_mx_METR-LA.pkl"

        self.input_window = input_window
        self.output_window = output_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.sensor_ids: List[str] = []
        self.sensor_id_to_ind: Dict[str, int] = {}
        self.adj_matrix: Optional[np.ndarray] = None
        self.mean: float = 0.0
        self.std: float = 1.0

    def load_graph(self) -> Tuple[List[str], Dict[str, int], np.ndarray]:
        """Load sensor list, ID-to-index mapping, and adjacency matrix."""
        if not self.adj_path.exists():
            raise FileNotFoundError(f"Adjacency file not found at {self.adj_path}")

        with open(self.adj_path, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            sensor_ids, sensor_id_to_ind, adj_matrix = u.load()

        self.sensor_ids = [str(sid) for sid in sensor_ids]
        self.sensor_id_to_ind = {str(k): int(v) for k, v in sensor_id_to_ind.items()}
        self.adj_matrix = adj_matrix.astype(np.float32)
        logger.info(
            "Loaded graph: %d sensors, adj matrix shape %s, non-zero edges: %d",
            len(self.sensor_ids),
            self.adj_matrix.shape,
            int(np.count_nonzero(self.adj_matrix)),
        )
        return self.sensor_ids, self.sensor_id_to_ind, self.adj_matrix

    def load_raw_speeds(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load speed data from CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path, index_col=0, nrows=nrows)
        df.index = pd.to_datetime(df.index)
        logger.info(
            "Loaded METR-LA speed DataFrame: shape %s, time range %s to %s",
            df.shape,
            df.index[0],
            df.index[-1],
        )
        return df

    def compute_normalized_adj(self, adj_type: str = "random_walk") -> torch.Tensor:
        """Compute normalized graph Laplacian or random-walk matrix for GCN/GNN."""
        if self.adj_matrix is None:
            self.load_graph()

        adj = self.adj_matrix.copy()
        n = adj.shape[0]

        if adj_type == "random_walk":
            # D^{-1} * A with self-loops
            adj = adj + np.eye(n, dtype=np.float32)
            d = np.sum(adj, axis=1)
            d_inv = np.power(d, -1, where=d > 0)
            d_inv[d == 0] = 0.0
            norm_adj = np.diag(d_inv) @ adj
        elif adj_type == "symmetric":
            # D^{-1/2} * A * D^{-1/2} with self-loops
            adj = adj + np.eye(n, dtype=np.float32)
            d = np.sum(adj, axis=1)
            d_inv_sqrt = np.power(d, -0.5, where=d > 0)
            d_inv_sqrt[d == 0] = 0.0
            norm_adj = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
        else:
            norm_adj = adj

        return torch.from_numpy(norm_adj).float()

    def generate_sliding_windows(
        self,
        speed_data: np.ndarray,
        time_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sliding windows (X, Y) from a continuous time series array.

        Args:
            speed_data: array of shape [T, num_nodes]
            time_features: optional array of shape [T, num_features] (e.g. time of day)

        Returns:
            X: [N, T_in, num_nodes, num_features]
            Y: [N, T_out, num_nodes]
        """
        t_total, num_nodes = speed_data.shape
        num_windows = t_total - self.input_window - self.output_window + 1

        if num_windows <= 0:
            raise ValueError(f"Time series length {t_total} too short for windows {self.input_window}+{self.output_window}")

        x_list = []
        y_list = []

        for i in range(num_windows):
            # Input window
            x_speed = speed_data[i : i + self.input_window]  # [T_in, num_nodes]
            if time_features is not None:
                x_time = time_features[i : i + self.input_window]  # [T_in, f]
                # Broadcast time features across all nodes: [T_in, num_nodes, f]
                x_time_expanded = np.repeat(x_time[:, np.newaxis, :], num_nodes, axis=1)
                x_combined = np.concatenate([x_speed[..., np.newaxis], x_time_expanded], axis=-1)
            else:
                x_combined = x_speed[..., np.newaxis]  # [T_in, num_nodes, 1]

            # Output window (ground truth target speeds)
            y_speed = speed_data[i + self.input_window : i + self.input_window + self.output_window]  # [T_out, num_nodes]

            x_list.append(x_combined)
            y_list.append(y_speed)

        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def prepare_dataset(
        self,
        max_timesteps: Optional[int] = None,
        include_time_of_day: bool = True,
    ) -> DatasetSplits:
        """Load, validate, split, normalize, and construct window datasets.

        Preserves strict chronological order without data leakage:
        - Train: first 70%
        - Val: next 10%
        - Test: final 20%
        - Normalization parameters (mean, std) calculated exclusively on Train!
        """
        self.load_graph()
        df = self.load_raw_speeds(nrows=max_timesteps)

        # Align columns with graph sensor order
        col_order = [str(sid) for sid in self.sensor_ids if str(sid) in df.columns]
        if len(col_order) != len(self.sensor_ids):
            logger.warning(
                "Sensor alignment warning: DataFrame has %d matched sensors out of %d graph sensors",
                len(col_order),
                len(self.sensor_ids),
            )
        df = df[col_order]

        speeds = df.values.astype(np.float32)
        total_len = len(speeds)

        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))

        # Time of day feature: normalized [0, 1) by minute of day (0 to 1439 / 1440)
        time_feat = None
        if include_time_of_day:
            minute_of_day = (df.index.hour * 60 + df.index.minute) / 1440.0
            time_feat = minute_of_day.values.reshape(-1, 1).astype(np.float32)

        # Strict train-only normalization
        train_speeds_raw = speeds[:train_end]
        # Only non-zero / positive speeds contribute to mean/std to avoid zero-skew
        valid_train = train_speeds_raw[train_speeds_raw > 0.0]
        self.mean = float(np.mean(valid_train)) if len(valid_train) > 0 else float(np.mean(train_speeds_raw))
        self.std = float(np.std(valid_train)) if len(valid_train) > 0 else float(np.std(train_speeds_raw))
        if self.std < 1e-5:
            self.std = 1.0

        logger.info(
            "Train-only normalization: mean=%.2f mph, std=%.2f mph (leakage-free)",
            self.mean,
            self.std,
        )

        # Normalize speeds: (x - mean) / std
        speeds_norm = (speeds - self.mean) / self.std

        # Chronological splits of raw continuous time series
        # To avoid boundary gaps between splits, each split retains enough lead-in
        # for input_window:
        train_norm = speeds_norm[:train_end]
        train_time = time_feat[:train_end] if time_feat is not None else None

        val_start = max(0, train_end - self.input_window)
        val_norm = speeds_norm[val_start:val_end]
        val_time = time_feat[val_start:val_end] if time_feat is not None else None

        test_start = max(0, val_end - self.input_window)
        test_norm = speeds_norm[test_start:]
        test_time = time_feat[test_start:] if time_feat is not None else None

        # Targets Y are in unnormalized physical units (mph) for direct evaluation
        train_raw = speeds[:train_end]
        val_raw = speeds[val_start:val_end]
        test_raw = speeds[test_start:]

        logger.info("Generating sliding windows: T_in=%d, T_out=%d...", self.input_window, self.output_window)
        train_x, train_y_norm = self.generate_sliding_windows(train_norm, train_time)
        val_x, val_y_norm = self.generate_sliding_windows(val_norm, val_time)
        test_x, test_y_norm = self.generate_sliding_windows(test_norm, test_time)

        # Targets in physical mph:
        train_y = train_y_norm * self.std + self.mean
        val_y = val_y_norm * self.std + self.mean
        test_y = test_y_norm * self.std + self.mean

        logger.info(
            "Splits generated: Train X=%s Y=%s | Val X=%s Y=%s | Test X=%s Y=%s",
            train_x.shape, train_y.shape,
            val_x.shape, val_y.shape,
            test_x.shape, test_y.shape,
        )

        return DatasetSplits(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            mean=self.mean,
            std=self.std,
            sensor_ids=self.sensor_ids,
            adj_matrix=self.adj_matrix,
        )


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
) -> Dict[str, float]:
    """Calculate standard traffic forecasting metrics (MAE, RMSE, MAPE).

    Masks missing / sensor fault values (where y_true <= null_val or NaN).
    """
    mask = (y_true > null_val) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    diff = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    # MAPE with division by zero avoidance
    mape = float(np.mean(np.abs(diff / y_true[mask])) * 100.0)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
