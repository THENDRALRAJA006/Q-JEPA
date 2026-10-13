"""
Neural architectures for real-data graph traffic forecasting on METR-LA.

Includes:
1. PersistenceBaseline
2. HistoricalAverageBaseline
3. SpatialTemporalLSTM
4. STGCNModel (Spatial-Temporal Graph Convolutional Network)
5. JEPAGraphModel (Joint Embedding Predictive Architecture for Graphs)
   - Online encoder (GCN + TemporalConv)
   - Target encoder (EMA updated, stop-gradient)
   - Graph Predictor (representation-level future state prediction)
   - Multi-horizon speed decoder
   - Clean ablation switches
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.logging import get_logger

logger = get_logger(__name__)


# ─── GRAPH CONVOLUTION LAYER ──────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Vectorized Spatial Graph Convolution Layer for Batched Spatio-Temporal Tensors.

    Supports inputs of shape:
        - [B, T, N, D_in] -> [B, T, N, D_out]
        - [B, N, D_in]    -> [B, N, D_out]
    Formula: H' = A * (H * W) + Bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, N, D_in] or [B, N, D_in]
            adj: Normalized adjacency matrix [N, N] (2D)
        """
        has_time = (x.dim() == 4)
        if has_time:
            B, T, N, D = x.shape
            x_flat = x.view(B * T, N, D)
        else:
            B, N, D = x.shape
            x_flat = x

        # 1. Feature transformation: [B*T, N, D_out]
        h = self.linear(x_flat)

        # 2. Graph propagation: adj @ h -> [B*T, N, D_out]
        # (N, N) @ (B*T, N, D_out) -> (B*T, N, D_out)
        out = torch.matmul(adj, h)

        if has_time:
            out = out.view(B, T, N, self.out_features)

        return out


# ─── SPATIO-TEMPORAL CONVOLUTIONAL BLOCK ─────────────────────────────────────

class SpatialTemporalBlock(nn.Module):
    """Spatio-Temporal block combining Graph Convolution and 1D Temporal Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_graph: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.gcn = GraphConvLayer(in_channels, out_channels)
        self.tconv = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, in_channels]
            adj: [N, N]
        """
        res = self.residual(x)

        # Spatial step
        if self.use_graph:
            h_spatial = F.gelu(self.gcn(x, adj))
        else:
            # Identity graph ablation
            identity_adj = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            h_spatial = F.gelu(self.gcn(x, identity_adj))

        # Temporal step: permute to [B * N, out_channels, T]
        B, T, N, C = h_spatial.shape
        h_temp = h_spatial.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        h_temp = F.gelu(self.tconv(h_temp))
        h_temp = h_temp.view(B, N, C, T).permute(0, 3, 1, 2)  # [B, T, N, C]

        out = self.norm(h_temp + res)
        out = self.dropout(out)
        return out


# ─── BASELINES ───────────────────────────────────────────────────────────────

class PersistenceBaseline:
    """Predicts future speeds as equal to the latest observed speed."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def predict(self, x: np.ndarray, num_horizons: int = 12) -> np.ndarray:
        """
        Args:
            x: [N_samples, T_in, num_nodes, num_features] (normalized speeds at index 0)
        Returns:
            y_pred: [N_samples, num_horizons, num_nodes] (physical mph)
        """
        # Latest observed normalized speed: [N_samples, num_nodes]
        latest_speed_norm = x[:, -1, :, 0]
        latest_speed_mph = latest_speed_norm * self.std + self.mean
        # Broadcast across all future horizons
        y_pred = np.repeat(latest_speed_mph[:, np.newaxis, :], num_horizons, axis=1)
        return y_pred


class HistoricalAverageBaseline:
    """Predicts future speed based on historical mean speed for the matching time-of-day slot."""

    def __init__(self, num_slots: int = 288, mean: float = 0.0, std: float = 1.0):
        self.num_slots = num_slots
        self.mean = mean
        self.std = std
        self.table: Optional[np.ndarray] = None  # [num_slots, num_nodes] in physical units

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        """Fit slot-level historical averages on the training split."""
        num_nodes = train_y.shape[2]
        num_horizons = train_y.shape[1]
        counts = np.zeros((self.num_slots, num_nodes), dtype=np.float32)
        sums = np.zeros((self.num_slots, num_nodes), dtype=np.float32)

        for i in range(len(train_y)):
            base_time = train_x[i, -1, 0, 1] if train_x.shape[-1] > 1 else 0.0
            for h in range(num_horizons):
                target_time = (base_time + (h + 1) / self.num_slots) % 1.0
                slot = int(np.clip(round(target_time * self.num_slots), 0, self.num_slots - 1)) % self.num_slots
                speeds = train_y[i, h]
                valid = (speeds > 0.0) & np.isfinite(speeds)
                sums[slot, valid] += speeds[valid]
                counts[slot, valid] += 1.0

        # Replace empty slots with overall dataset mean
        self.table = np.where(counts > 0, sums / np.maximum(counts, 1.0), self.mean)

    def predict(self, x: np.ndarray, num_horizons: int = 12) -> np.ndarray:
        """Predict using historical slot average."""
        n_samples, t_in, num_nodes, _ = x.shape
        preds = np.zeros((n_samples, num_horizons, num_nodes), dtype=np.float32)

        for i in range(n_samples):
            base_time = x[i, -1, 0, 1] if x.shape[-1] > 1 else 0.0
            for h in range(num_horizons):
                target_time = (base_time + (h + 1) / self.num_slots) % 1.0
                slot = int(np.clip(round(target_time * self.num_slots), 0, self.num_slots - 1)) % self.num_slots
                preds[i, h] = self.table[slot] if self.table is not None else self.mean

        return preds



# ─── SPATIAL-TEMPORAL LSTM ────────────────────────────────────────────────────

class SpatialTemporalLSTM(nn.Module):
    """LSTM model operating across sensor features."""

    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_window: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_window = output_window
        # Project node features: [B, T, N * F] -> [B, T, hidden_dim]
        self.input_proj = nn.Linear(num_nodes * in_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_window * num_nodes)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, num_nodes, in_features]
        Returns:
            out: [B, output_window, num_nodes] (normalized speed)
        """
        B, T, N, feat_dim = x.shape
        x_flat = x.view(B, T, N * feat_dim)
        h = F.gelu(self.input_proj(x_flat))
        lstm_out, _ = self.lstm(h)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_dim]
        pred = self.head(last_hidden)     # [B, output_window * num_nodes]
        return pred.view(B, self.output_window, N)


# ─── STGCN MODEL (Graph-Temporal Baseline) ────────────────────────────────────

class STGCNModel(nn.Module):
    """Spatio-Temporal Graph Convolutional Network baseline."""

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 2,
        hidden_dim: int = 32,
        output_window: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_window = output_window

        self.block1 = SpatialTemporalBlock(in_channels, hidden_dim, dropout=dropout)
        self.block2 = SpatialTemporalBlock(hidden_dim, hidden_dim, dropout=dropout)
        # Prediction head: projects hidden states over time to future horizons
        self.out_conv = nn.Conv1d(hidden_dim, output_window, kernel_size=1)
        self.out_proj = nn.Linear(hidden_dim, output_window)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, num_nodes, in_channels]
            adj: [num_nodes, num_nodes]
        Returns:
            pred: [B, output_window, num_nodes] (normalized speed)
        """
        B, T, N, C = x.shape
        h = self.block1(x, adj)
        h = self.block2(h, adj)  # [B, T, N, hidden_dim]

        # Temporal pooling over input window: mean over T -> [B, N, hidden_dim]
        h_pool = torch.mean(h, dim=1)
        # Linear projection to future horizons: [B, N, output_window]
        pred = self.out_proj(h_pool)
        # Permute to [B, output_window, N]
        return pred.permute(0, 2, 1)


# ─── GRAPH WAVENET BASELINE ──────────────────────────────────────────────────

class GraphWaveNetBlock(nn.Module):
    """WaveNet layer combining Dilated Gated Temporal Convolutions and Graph Convolutions."""

    def __init__(self, c_in: int, c_out: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.tconv_filter = nn.Conv1d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.tconv_gate = nn.Conv1d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.gconv = GraphConvLayer(c_out, c_out)
        self.norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
        self.skip = nn.Linear(c_out, c_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, N, c_in]
            adj: [N, N]
        Returns:
            out: [B, T, N, c_out], skip: [B, T, N, c_out]
        """
        B, T, N, C = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        filter_out = torch.tanh(self.tconv_filter(x_perm))
        gate_out = torch.sigmoid(self.tconv_gate(x_perm))
        t_out = filter_out * gate_out
        t_out = t_out.view(B, N, -1, T).permute(0, 3, 1, 2).contiguous()

        s_out = F.gelu(self.gconv(t_out, adj))
        out = self.norm(s_out + self.res(x))
        out = self.dropout(out)
        skip = self.skip(out)
        return out, skip


class GraphWaveNetModel(nn.Module):
    """Graph WaveNet for Deep Spatial-Temporal Graph Modeling (Wu et al., IJCAI 2019)."""

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 2,
        hidden_dim: int = 32,
        output_window: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_window = output_window
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.block1 = GraphWaveNetBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.block2 = GraphWaveNetBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_window),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, num_nodes, in_channels]
            adj: [num_nodes, num_nodes]
        Returns:
            pred: [B, output_window, num_nodes] (normalized speed)
        """
        h = self.input_proj(x)
        h, skip1 = self.block1(h, adj)
        h, skip2 = self.block2(h, adj)
        skip_total = skip1 + skip2
        h_pool = torch.mean(skip_total, dim=1)  # [B, N, hidden_dim]
        pred = self.head(h_pool)                # [B, N, output_window]
        return pred.permute(0, 2, 1)            # [B, output_window, N]


# ─── SPATIAL-TEMPORAL TRANSFORMER BASELINE ────────────────────────────────────

class SpatialTemporalTransformerModel(nn.Module):
    """Contemporary Spatio-Temporal Transformer baseline for traffic forecasting."""

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 2,
        hidden_dim: int = 32,
        num_heads: int = 2,
        output_window: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_window = output_window
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_window),
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T_in, num_nodes, in_channels]
            adj: Optional adjacency matrix
        Returns:
            pred: [B, output_window, num_nodes] (normalized speed)
        """
        B, T, N, _ = x.shape
        h = self.input_proj(x)

        # Spatial Self-Attention across nodes
        h_s = h.view(B * T, N, -1)
        attn_s, _ = self.spatial_attn(h_s, h_s, h_s)
        h_s = self.spatial_norm(h_s + attn_s).view(B, T, N, -1)

        # Temporal Self-Attention across time steps
        h_t = h_s.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        attn_t, _ = self.temporal_attn(h_t, h_t, h_t)
        h_t = self.temporal_norm(h_t + attn_t).view(B, N, T, -1)

        # Temporal pooling and projection head
        h_pool = torch.mean(h_t, dim=2)  # [B, N, hidden_dim]
        pred = self.head(h_pool)         # [B, N, output_window]
        return pred.permute(0, 2, 1)     # [B, output_window, N]


# ─── JEPAGraph MODEL ──────────────────────────────────────────────────────────

class JEPAGraphModel(nn.Module):
    """Joint Embedding Predictive Architecture for Spatio-Temporal Graph Forecasting.

    Key Architectural Components:
    1. Online Encoder E_theta:
       Encodes historical observations X -> Z_context in latent space.
    2. Target Encoder E_xi:
       Encodes future trajectory Y -> Z_target in latent space.
       Guaranteed stop-gradient protection (no backward pass).
       Updated strictly via Exponential Moving Average (EMA).
    3. Graph Predictor P_phi:
       Predicts future latent representation Z_hat from Z_context and graph structure.
    4. Multi-Horizon Speed Decoder D_psi:
       Projects latent representations to physical speed forecasts across horizons.
    """

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 2,
        latent_dim: int = 32,
        output_window: int = 12,
        dropout: float = 0.1,
        use_graph: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.output_window = output_window
        self.use_graph = use_graph
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self.in_channels = in_channels

        # ── 1. Online Encoder (Context) ──────────────────────────────
        self.online_encoder = nn.Sequential(
            SpatialTemporalBlock(in_channels, latent_dim, dropout=dropout, use_graph=use_graph),
            SpatialTemporalBlock(latent_dim, latent_dim, dropout=dropout, use_graph=use_graph),
        )

        # ── 2. Target Encoder (Future) ───────────────────────────────
        # In JEPA, target encoder has identical architecture to online encoder
        self.target_encoder = nn.Sequential(
            SpatialTemporalBlock(in_channels, latent_dim, dropout=dropout, use_graph=use_graph),
            SpatialTemporalBlock(latent_dim, latent_dim, dropout=dropout, use_graph=use_graph),
        )

        # Initialize target encoder identical to online and freeze gradients
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # ── 3. Latent Graph Predictor ────────────────────────────────
        # Predicts Z_target from Z_context using spatial graph propagation
        self.predictor_gcn = GraphConvLayer(latent_dim, latent_dim)
        self.predictor_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.predictor_norm = nn.LayerNorm(latent_dim)

        # ── 4. Multi-Horizon Speed Decoder ───────────────────────────
        # Decodes predicted representations to physical speeds: [B, N, latent_dim] -> [B, output_window, N]
        self.speed_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_window),
        )

    def encode_context(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Online encoder: [B, T_in, N, C] -> Z_context [B, N, latent_dim]."""
        h = x
        for block in self.online_encoder:
            h = block(h, adj)
        # Temporal pooling over history
        z_context = torch.mean(h, dim=1)  # [B, N, latent_dim]
        return z_context

    def encode_target(self, y_future: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Target encoder with stop-gradient: [B, T_out, N] or [B, T_out, N, C] -> Z_target [B, N, latent_dim]."""
        with torch.no_grad():
            if y_future.dim() == 3:
                B, T, N = y_future.shape
                if self.in_channels == 1:
                    y_in = y_future.unsqueeze(-1)
                else:
                    zeros = torch.zeros(B, T, N, self.in_channels - 1, device=y_future.device, dtype=y_future.dtype)
                    y_in = torch.cat([y_future.unsqueeze(-1), zeros], dim=-1)
            else:
                y_in = y_future

            h = y_in
            for block in self.target_encoder:
                h = block(h, adj)
            z_target = torch.mean(h, dim=1)  # [B, N, latent_dim]
            return z_target.detach()

    def predict_latent(self, z_context: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Predicts future representations in latent space."""
        if self.use_graph:
            h_spatial = F.gelu(self.predictor_gcn(z_context, adj))
        else:
            ident_adj = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            h_spatial = F.gelu(self.predictor_gcn(z_context, ident_adj))

        z_hat = self.predictor_mlp(h_spatial)
        z_hat = self.predictor_norm(z_hat + z_context)
        return z_hat

    def decode_speed(self, z_pred: torch.Tensor) -> torch.Tensor:
        """Decodes latent representation [B, N, latent_dim] -> [B, output_window, N]."""
        # [B, N, output_window]
        speeds = self.speed_decoder(z_pred)
        # Permute to [B, output_window, N]
        return speeds.permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y_future: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for both inference and training.

        Returns:
            dict containing:
                - 'speed_pred': [B, output_window, N] (normalized predicted speed)
                - 'z_pred': [B, N, latent_dim]
                - 'z_target': [B, N, latent_dim] (if y_future is provided)
        """
        z_context = self.encode_context(x, adj)
        z_pred = self.predict_latent(z_context, adj)
        speed_pred = self.decode_speed(z_pred)

        result = {
            "speed_pred": speed_pred,
            "z_pred": z_pred,
        }

        if y_future is not None:
            z_target = self.encode_target(y_future, adj)
            result["z_target"] = z_target

        return result

    def update_target_ema(self):
        """Update target encoder parameters using Exponential Moving Average (EMA)."""
        if not self.use_ema:
            return

        with torch.no_grad():
            for p_online, p_target in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                p_target.data.mul_(self.ema_decay).add_(
                    p_online.data, alpha=1.0 - self.ema_decay
                )
