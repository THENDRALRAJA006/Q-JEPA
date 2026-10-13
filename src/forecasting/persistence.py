"""
Forecasting baselines: Persistence, Historical Average, LSTM, JEPA-style.
All baselines share a common interface: fit(X_train, y_train) / predict(X).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class BaseForecaster(ABC):
    """Common interface for all forecasting models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions shape (N, horizon)."""
        ...

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) both shape (N, horizon). Default: zero uncertainty."""
        preds = self.predict(X)
        return preds, np.zeros_like(preds)

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────────
# Baseline A: Persistence
# ─────────────────────────────────────────────────

class PersistenceForecaster(BaseForecaster):
    """
    Naive persistence: predict the last observed value for all horizons.
    Strong baseline for short horizons.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        return self  # no training needed

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, window_size) — last value repeated
        last = X[:, -1:]          # (N, 1)
        horizon = 1               # will be broadcast
        return np.repeat(last, 1, axis=1)  # (N, 1) — caller repeats for multi-horizon

    def predict_horizon(self, X: np.ndarray, horizon: int) -> np.ndarray:
        last = X[:, -1:]
        return np.repeat(last, horizon, axis=1)  # (N, horizon)


# ─────────────────────────────────────────────────
# Baseline B: Historical / seasonal average
# ─────────────────────────────────────────────────

class HistoricalAverageForecaster(BaseForecaster):
    """
    Predict the mean speed for each time-of-day slot seen in training.
    Uses step index mod slots as the temporal key.
    """

    def __init__(self, slots: int = 288):
        self.slots = slots
        self._slot_means: Optional[np.ndarray] = None
        self._global_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistoricalAverageForecaster":
        # X: (N, window), y: (N, horizon)
        # Approximate slot from window mean — simple slot-based estimate
        window_means = X.mean(axis=1)
        self._global_mean = float(window_means.mean())
        # Bin by normalized position within dataset (crude slot estimate)
        n = len(X)
        slot_sums = np.zeros(self.slots, dtype=np.float64)
        slot_counts = np.zeros(self.slots, dtype=np.int64)
        for i, wm in enumerate(window_means):
            slot = int((i / max(n, 1)) * self.slots) % self.slots
            slot_sums[slot] += wm
            slot_counts[slot] += 1
        with np.errstate(invalid="ignore"):
            self._slot_means = np.where(slot_counts > 0, slot_sums / slot_counts, self._global_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._slot_means is None:
            return np.full((len(X), 1), self._global_mean)
        n = len(X)
        preds = np.zeros((n, 1))
        for i in range(n):
            slot = int((i / max(n, 1)) * self.slots) % self.slots
            preds[i, 0] = self._slot_means[slot]
        return preds

    def predict_horizon(self, X: np.ndarray, horizon: int) -> np.ndarray:
        base = self.predict(X)  # (N, 1)
        return np.repeat(base, horizon, axis=1)
