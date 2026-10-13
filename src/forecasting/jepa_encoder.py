"""
JEPA-style predictive representation learning for traffic forecasting.

Architecture:
  - OnlineEncoder: encodes observed traffic history into a latent context z_ctx
  - TargetEncoder: encodes future traffic windows into target representations z_tgt
    (protected by stop-gradient; updated via EMA of online encoder weights)
  - Predictor: maps z_ctx -> predicted z_tgt

Training objective: MSE between Predictor(z_ctx) and sg(z_tgt)

IMPORTANT NOTES:
  1. This predicts REPRESENTATIONS produced by a target encoder, not raw speeds.
  2. The target encoder is updated by Exponential Moving Average (EMA), not backprop.
  3. To produce numerical speed forecasts, a lightweight linear head is trained on
     frozen JEPA representations (separate from the JEPA objective).
  4. This is a JEPA-inspired research experiment, not a production-tested JEPA system.
  5. No pretrained JEPA weights are claimed or loaded.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.persistence import BaseForecaster
from src.common.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────
# Neural network components
# ─────────────────────────────────────────────────

class TrafficEncoder(nn.Module):
    """Encodes a traffic speed window into a latent representation."""

    def __init__(self, input_size: int, embed_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, num_layers=1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, features)
        _, (h, _) = self.lstm(x)
        return self.proj(h[-1])  # (B, embed_dim)


class RepresentationPredictor(nn.Module):
    """Maps context representation to predicted target representation."""

    def __init__(self, embed_dim: int = 64, pred_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, pred_dim * 2),
            nn.GELU(),
            nn.Linear(pred_dim * 2, pred_dim),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.net(z_ctx)


class DecodingHead(nn.Module):
    """Lightweight linear head: latent -> speed forecast. Trained separately."""

    def __init__(self, embed_dim: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


# ─────────────────────────────────────────────────
# JEPA Forecaster
# ─────────────────────────────────────────────────

class JEPAForecaster(BaseForecaster):
    """
    JEPA-inspired traffic forecaster.

    Phase 1: Train JEPA objective (online encoder + predictor vs target encoder).
    Phase 2: Freeze encoders, train linear decoding head for speed regression.
    """

    def __init__(
        self,
        window_size: int = 12,
        horizon: int = 3,
        embed_dim: int = 64,
        pred_dim: int = 64,
        ema_tau: float = 0.99,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.ema_tau = ema_tau
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Network components
        self.online_encoder: Optional[TrafficEncoder] = None
        self.target_encoder: Optional[TrafficEncoder] = None
        self.predictor: Optional[RepresentationPredictor] = None
        self.head: Optional[DecodingHead] = None

        self.jepa_losses: list = []
        self.head_losses: list = []
        self.val_losses: list = []

    @property
    def name(self) -> str:
        return "JEPA-Traffic"

    def _init_models(self):
        self.online_encoder = TrafficEncoder(1, self.embed_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        # Target encoder is never updated by backprop
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = RepresentationPredictor(self.embed_dim, self.pred_dim).to(self.device)
        self.head = DecodingHead(self.embed_dim, self.horizon).to(self.device)

    def _ema_update(self):
        """Update target encoder with EMA of online encoder weights."""
        tau = self.ema_tau
        for p_online, p_target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data = tau * p_target.data + (1 - tau) * p_online.data

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    ) -> "JEPAForecaster":
        """
        X: (N, window_size) — observed context
        y: (N, horizon)    — future target speeds (used as target encoder input)
        """
        self._init_models()
        jepa_params = list(self.online_encoder.parameters()) + list(self.predictor.parameters())
        jepa_opt = torch.optim.Adam(jepa_params, lr=self.lr)
        jepa_loss_fn = nn.MSELoss()

        # Convert to tensors
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)   # (N, T, 1)
        y_t = torch.FloatTensor(y).unsqueeze(-1).to(self.device)   # (N, H, 1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        # ── Phase 1: JEPA representation learning ──
        logger.info("JEPA Phase 1: Representation learning (%d epochs)", self.epochs)
        self.online_encoder.train()
        self.predictor.train()

        for epoch in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                jepa_opt.zero_grad()

                # Context representation (online encoder, gradients flow)
                z_ctx = self.online_encoder(xb)                # (B, embed_dim)

                # Target representation (target encoder, stop-gradient)
                with torch.no_grad():
                    z_tgt = self.target_encoder(yb)            # (B, embed_dim)

                # Predict target representation from context
                z_pred = self.predictor(z_ctx)                 # (B, pred_dim)

                # JEPA objective: minimise representation prediction error
                loss = jepa_loss_fn(z_pred, z_tgt)
                loss.backward()
                jepa_opt.step()

                # EMA update of target encoder (after optimiser step)
                self._ema_update()
                ep_loss += loss.item() * len(xb)

            ep_loss /= len(X)
            self.jepa_losses.append(ep_loss)
            if (epoch + 1) % 5 == 0:
                logger.info("JEPA epoch %d/%d — repr_loss=%.6f", epoch + 1, self.epochs, ep_loss)

        # ── Phase 2: Train decoding head (frozen encoder) ──
        logger.info("JEPA Phase 2: Training decoding head")
        self.online_encoder.eval()
        for p in self.online_encoder.parameters():
            p.requires_grad_(False)

        head_opt = torch.optim.Adam(self.head.parameters(), lr=self.lr)
        head_loss_fn = nn.MSELoss()
        y_speed_t = torch.FloatTensor(y).to(self.device)  # (N, H)

        head_loader = DataLoader(TensorDataset(X_t, y_speed_t), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            ep_loss = 0.0
            self.head.train()
            for xb, yb in head_loader:
                head_opt.zero_grad()
                with torch.no_grad():
                    z = self.online_encoder(xb)
                pred = self.head(z)
                loss = head_loss_fn(pred, yb)
                loss.backward()
                head_opt.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= len(X)
            self.head_losses.append(ep_loss)

            if X_val is not None and y_val is not None and (epoch + 1) % 5 == 0:
                val_loss = self._eval_head(X_val, y_val)
                self.val_losses.append(val_loss)
                logger.info("JEPA head epoch %d/%d — head_loss=%.4f val=%.4f",
                            epoch + 1, self.epochs, ep_loss, val_loss)

        logger.info("JEPA training complete.")
        return self

    def _eval_head(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        self.head.eval()
        X_t = torch.FloatTensor(X_val).unsqueeze(-1).to(self.device)
        y_t = torch.FloatTensor(y_val).to(self.device)
        with torch.no_grad():
            z = self.online_encoder(X_t)
            pred = self.head(z)
            loss = nn.MSELoss()(pred, y_t).item()
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.online_encoder is None or self.head is None:
            raise RuntimeError("Call fit() before predict().")
        self.online_encoder.eval()
        self.head.eval()
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            z = self.online_encoder(X_t)
            preds = self.head(z).cpu().numpy()
        return preds  # (N, horizon)

    def get_representations(self, X: np.ndarray) -> np.ndarray:
        """Return latent representations for analysis."""
        if self.online_encoder is None:
            raise RuntimeError("Call fit() before get_representations().")
        self.online_encoder.eval()
        X_t = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            z = self.online_encoder(X_t).cpu().numpy()
        return z

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_encoder": self.online_encoder.state_dict() if self.online_encoder else None,
            "target_encoder": self.target_encoder.state_dict() if self.target_encoder else None,
            "predictor": self.predictor.state_dict() if self.predictor else None,
            "head": self.head.state_dict() if self.head else None,
            "config": {
                "window_size": self.window_size, "horizon": self.horizon,
                "embed_dim": self.embed_dim, "pred_dim": self.pred_dim,
                "ema_tau": self.ema_tau,
            },
            "jepa_losses": self.jepa_losses,
            "head_losses": self.head_losses,
            "val_losses": self.val_losses,
        }, path)
        logger.info("JEPA checkpoint saved: %s", path)

    def load(self, path: Path) -> "JEPAForecaster":
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        self.embed_dim = cfg["embed_dim"]
        self.pred_dim = cfg["pred_dim"]
        self.horizon = cfg["horizon"]
        self._init_models()
        self.online_encoder.load_state_dict(ckpt["online_encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.head.load_state_dict(ckpt["head"])
        self.jepa_losses = ckpt.get("jepa_losses", [])
        self.head_losses = ckpt.get("head_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        logger.info("JEPA checkpoint loaded: %s", path)
        return self
