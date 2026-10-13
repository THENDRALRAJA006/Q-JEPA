"""
LSTM traffic forecaster (Baseline C).
Standard sequence-to-multi-step model with dropout regularisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.persistence import BaseForecaster
from src.common.logging import get_logger

logger = get_logger(__name__)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.2, horizon: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (B, horizon)


class LSTMForecaster(BaseForecaster):
    """
    Conventional LSTM forecaster.
    Trained to directly regress future speed values.
    """

    def __init__(
        self,
        window_size: int = 12,
        horizon: int = 3,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model: Optional[LSTMModel] = None
        self.train_losses: list = []
        self.val_losses: list = []

    @property
    def name(self) -> str:
        return "LSTM"

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        """Train LSTM on (N, window_size) -> (N, horizon) data."""
        self.model = LSTMModel(
            input_size=1, hidden=self.hidden,
            layers=self.layers, dropout=self.dropout, horizon=self.horizon,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Tensor datasets — add feature dim
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # (N, T, 1)
        y_t = torch.FloatTensor(y).to(self.device)                # (N, H)
        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_v = torch.FloatTensor(X_val).unsqueeze(-1).to(self.device)
            y_v = torch.FloatTensor(y_val).to(self.device)
        else:
            X_v, y_v = None, None

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X)
            self.train_losses.append(epoch_loss)

            if X_v is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_v)
                    val_loss = criterion(val_pred, y_v).item()
                self.val_losses.append(val_loss)
                self.model.train()
                if (epoch + 1) % 5 == 0:
                    logger.info("LSTM epoch %d/%d — train=%.4f val=%.4f",
                                epoch + 1, self.epochs, epoch_loss, val_loss)
            elif (epoch + 1) % 5 == 0:
                logger.info("LSTM epoch %d/%d — train=%.4f", epoch + 1, self.epochs, epoch_loss)

        logger.info("LSTM training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        self.model.eval()
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()
        return preds  # (N, horizon)

    def predict_with_uncertainty(self, X: np.ndarray, n_passes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo dropout uncertainty estimate."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        self.model.train()  # keep dropout active
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        samples = []
        with torch.no_grad():
            for _ in range(n_passes):
                samples.append(self.model(X_t).cpu().numpy())
        self.model.eval()
        samples_arr = np.stack(samples, axis=0)  # (passes, N, H)
        return samples_arr.mean(0), samples_arr.std(0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "config": {
                "window_size": self.window_size,
                "horizon": self.horizon,
                "hidden": self.hidden,
                "layers": self.layers,
                "dropout": self.dropout,
            },
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
        logger.info("LSTM checkpoint saved: %s", path)

    def load(self, path: Path) -> "LSTMForecaster":
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        self.model = LSTMModel(
            input_size=1, hidden=cfg["hidden"],
            layers=cfg["layers"], dropout=cfg["dropout"],
            horizon=cfg["horizon"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        logger.info("LSTM checkpoint loaded: %s", path)
        return self
