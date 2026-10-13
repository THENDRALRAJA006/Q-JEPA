"""
Q-JEPA Traffic Foundation Model Architecture (JEPAGraph-FM V2.0).

Key Architectural Components:
1. Spatiotemporal Backbone: Causal Gated Dilated TCN + Spatial Graph Diffusion
2. Online Context Encoder: Maps observed traffic history to contextual representations
3. EMA Target Encoder: Computes stable future representations without gradient leakage
4. Latent Graph Predictor: Multi-horizon latent trajectory forecasting in latent space
5. Masked Spatiotemporal Reconstruction Head: Self-supervised sensor and patch completion
6. Variable-Specific Forecast Decoders: Modular prediction heads for Speed and Flow
7. Foundation Scaling Profiles: Tiny (CPU/smoke), Base (GPU standard), Large (Research capacity)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.logging import get_logger
from src.forecasting.graph_temporal_encoder import SpatiotemporalBlock

logger = get_logger(__name__)


@dataclass
class FoundationModelConfig:
    """Configuration dataclass for JEPAGraph Foundation Model."""

    name: str = "base"
    in_channels: int = 3  # [value, time_of_day, day_of_week]
    hidden_dim: int = 64
    num_layers: int = 4
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 2
    input_window: int = 12
    output_window: int = 12
    predictor_layers: int = 2
    dropout: float = 0.1
    ema_decay: float = 0.99
    mask_ratio: float = 0.15

    @classmethod
    def tiny(cls) -> FoundationModelConfig:
        """Lightweight configuration for CPU smoke testing and fast unit verification."""
        return cls(
            name="tiny",
            in_channels=3,
            hidden_dim=32,
            num_layers=2,
            dilations=[1, 2],
            kernel_size=2,
            input_window=12,
            output_window=12,
            predictor_layers=1,
            dropout=0.05,
            ema_decay=0.99,
            mask_ratio=0.15,
        )

    @classmethod
    def base(cls) -> FoundationModelConfig:
        """Standard GPU configuration for multi-dataset pretraining."""
        return cls(
            name="base",
            in_channels=3,
            hidden_dim=64,
            num_layers=4,
            dilations=[1, 2, 4, 8],
            kernel_size=2,
            input_window=12,
            output_window=12,
            predictor_layers=2,
            dropout=0.1,
            ema_decay=0.995,
            mask_ratio=0.20,
        )

    @classmethod
    def large(cls) -> FoundationModelConfig:
        """High-capacity research configuration."""
        return cls(
            name="large",
            in_channels=3,
            hidden_dim=128,
            num_layers=6,
            dilations=[1, 2, 4, 8, 1, 2],
            kernel_size=2,
            input_window=12,
            output_window=12,
            predictor_layers=3,
            dropout=0.15,
            ema_decay=0.999,
            mask_ratio=0.25,
        )


class SpatiotemporalEncoder(nn.Module):
    """
    Backbone encoder processing spatiotemporal observation sequences.
    Maps [B, T, N, in_channels] to [B, T, N, hidden_dim].
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.in_channels, config.hidden_dim)

        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            d = config.dilations[i % len(config.dilations)]
            self.blocks.append(
                SpatiotemporalBlock(
                    hidden_dim=config.hidden_dim,
                    dilation=d,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
            )
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, in_channels]
            adj_matrix: [N, N]
        Returns:
            [B, T, N, hidden_dim]
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, adj_matrix)
        return self.final_norm(h)


class LatentPredictor(nn.Module):
    """
    JEPA Predictor mapping context representation to predicted future representations.
    Maps [B, N, hidden_dim] -> [B, T_out, N, hidden_dim].
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config
        self.output_window = config.output_window
        self.hidden_dim = config.hidden_dim

        # Learnable temporal horizon query embeddings
        self.horizon_embeddings = nn.Parameter(
            torch.empty(config.output_window, config.hidden_dim)
        )
        nn.init.normal_(self.horizon_embeddings, std=0.02)

        layers = []
        for _ in range(config.predictor_layers):
            layers.extend([
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_latent: [B, N, hidden_dim]
        Returns:
            predicted future latents: [B, T_out, N, hidden_dim]
        """
        B, N, D = context_latent.shape
        T_out = self.output_window

        # Expand context latent across time: [B, T_out, N, D]
        ctx_expanded = context_latent.unsqueeze(1).expand(-1, T_out, -1, -1)
        # Expand horizon queries: [1, T_out, 1, D] -> [B, T_out, N, D]
        h_expanded = self.horizon_embeddings.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1)

        combined = torch.cat([ctx_expanded, h_expanded], dim=-1)  # [B, T_out, N, 2*D]
        pred_latents = self.mlp(combined)
        return self.out_proj(pred_latents)


class ForecastDecoder(nn.Module):
    """Prediction head mapping predicted representations to physical scalar values."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, T_out, N, hidden_dim]
        Returns:
            predictions: [B, T_out, N]
        """
        out = self.net(latent)  # [B, T_out, N, 1]
        return out.squeeze(-1)


class MaskedReconstructionHead(nn.Module):
    """Self-supervised head reconstructing masked input observations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representations: [B, T_in, N, hidden_dim]
        Returns:
            reconstructed inputs: [B, T_in, N, 1]
        """
        return self.net(representations)


class JEPAGraphFoundationModel(nn.Module):
    """
    Q-JEPA Spatiotemporal Traffic Foundation Model (JEPAGraph-FM).

    Supports:
    - Pretraining via joint JEPA representation matching and masked reconstruction
    - Fine-tuning for downstream multi-horizon forecasting
    - Zero-shot / few-shot transfer to unseen networks
    - Heterogeneous graph sizes N
    """

    def __init__(self, config: Optional[FoundationModelConfig] = None):
        super().__init__()
        self.config = config or FoundationModelConfig.base()

        # 1. Online Context Encoder
        self.online_encoder = SpatiotemporalEncoder(self.config)

        # 2. Target Encoder (EMA updated, gradients detached)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 3. Latent Predictor
        self.predictor = LatentPredictor(self.config)

        # 4. Masked Reconstruction Head
        self.recon_head = MaskedReconstructionHead(self.config.hidden_dim)

        # 5. Forecast Decoders (Modular per variable type)
        self.speed_decoder = ForecastDecoder(self.config.hidden_dim, dropout=self.config.dropout)
        self.flow_decoder = ForecastDecoder(self.config.hidden_dim, dropout=self.config.dropout)

        self.ema_decay = self.config.ema_decay

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        """
        Exponential moving average update of target encoder weights:
        theta_target = momentum * theta_target + (1 - momentum) * theta_online
        """
        m = momentum if momentum is not None else self.ema_decay
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(m).add_(online_param.data, alpha=1.0 - m)

    def _apply_spatiotemporal_mask(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking on input observations for reconstruction objective.

        Args:
            x: [B, T_in, N, in_channels]
        Returns:
            masked_x: tensor with masked values zeroed
            binary_mask: [B, T_in, N] (1 = masked, 0 = unmasked)
        """
        B, T, N, C = x.shape
        # Create random mask on value channel
        rand = torch.rand(B, T, N, device=x.device)
        binary_mask = (rand < mask_ratio).float()

        masked_x = x.clone()
        # Zero out the observation value channel for masked tokens
        masked_x[..., 0] = masked_x[..., 0] * (1.0 - binary_mask)
        return masked_x, binary_mask

    def forward_pretrain(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adj_matrix: torch.Tensor,
        variable_type: str = "speed",
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for self-supervised pretraining.

        Args:
            x: [B, T_in, N, 3] historical context observations
            y: [B, T_out, N] ground truth future observations
            adj_matrix: [N, N] normalized spatial graph adjacency
            variable_type: 'speed' or 'flow'
            mask_ratio: ratio of inputs to mask for reconstruction

        Returns:
            Dictionary containing:
            - pred_latent: [B, T_out, N, hidden_dim]
            - target_latent: [B, T_out, N, hidden_dim]
            - recon_x: [B, T_in, N, 1]
            - mask_patch: [B, T_in, N]
            - pred_forecast: [B, T_out, N]
        """
        ratio = mask_ratio if mask_ratio is not None else self.config.mask_ratio

        # 1. Masked Spatiotemporal inputs
        masked_x, mask_patch = self._apply_spatiotemporal_mask(x, ratio)

        # 2. Online Context Encoding
        # h_ctx: [B, T_in, N, hidden_dim]
        h_ctx = self.online_encoder(masked_x, adj_matrix)

        # Summary context latent (take last historical causal step)
        context_latent = h_ctx[:, -1, :, :]  # [B, N, hidden_dim]

        # 3. Latent Prediction into future
        pred_latents = self.predictor(context_latent)  # [B, T_out, N, hidden_dim]

        # 4. Target Encoder representation (stop gradient)
        with torch.no_grad():
            # Build pseudo-input for future target: [B, T_out, N, in_channels]
            # Use y as values, with default time tokens
            B, T_out, N = y.shape
            dummy_time = torch.zeros(B, T_out, N, 2, device=y.device)
            future_input = torch.cat([y.unsqueeze(-1), dummy_time], dim=-1)
            target_latents = self.target_encoder(future_input, adj_matrix)

        # 5. Masked Reconstruction
        recon_x = self.recon_head(h_ctx)  # [B, T_in, N, 1]

        # 6. Forecasting projection
        decoder = self.speed_decoder if variable_type == "speed" else self.flow_decoder
        pred_forecast = decoder(pred_latents)  # [B, T_out, N]

        return {
            "pred_latent": pred_latents,
            "target_latent": target_latents,
            "recon_x": recon_x,
            "mask_patch": mask_patch,
            "pred_forecast": pred_forecast,
        }

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        variable_type: str = "speed",
    ) -> torch.Tensor:
        """
        Inference forward pass: produces multi-horizon forecasts.

        Args:
            x: [B, T_in, N, in_channels]
            adj_matrix: [N, N]
            variable_type: 'speed' or 'flow'
        Returns:
            predictions: [B, T_out, N] in normalized units
        """
        h_ctx = self.online_encoder(x, adj_matrix)
        context_latent = h_ctx[:, -1, :, :]
        pred_latents = self.predictor(context_latent)

        decoder = self.speed_decoder if variable_type == "speed" else self.flow_decoder
        return decoder(pred_latents)

    def forecast(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        variable_type: str = "speed",
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        """
        High-level inference returning predictions in physical units.
        """
        self.eval()
        with torch.no_grad():
            pred_norm = self.forward(x, adj_matrix, variable_type=variable_type)
            pred_phys = pred_norm * std + mean
            return pred_phys

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        online = sum(p.numel() for p in self.online_encoder.parameters())
        target = sum(p.numel() for p in self.target_encoder.parameters())
        predictor = sum(p.numel() for p in self.predictor.parameters())
        decoders = sum(p.numel() for p in self.speed_decoder.parameters()) + sum(
            p.numel() for p in self.flow_decoder.parameters()
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "online_encoder": online,
            "target_encoder": target,
            "predictor": predictor,
            "decoders": decoders,
        }
