"""
Multi-Objective Self-Supervised Loss Suite for JEPAGraph Foundation Model.

Provides:
- Objective 1: Future latent prediction loss (Smooth L1 in embedding space)
- Objective 2: Masked spatiotemporal reconstruction loss (reconstructing masked sensors/patches)
- Objective 3: Supervised forecasting loss (Huber/MAE on valid physical observations)
- Objective 4: Multiscale temporal consistency loss (optional smoothness penalty)
- ObjectiveSuite: Configurable weighted loss computation and metric decomposition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeightsConfig:
    """Weights configuration for pretraining and fine-tuning objectives."""

    w_jepa: float = 1.0  # Future latent representation prediction
    w_mask: float = 0.5  # Masked spatiotemporal reconstruction
    w_sup: float = 1.0  # Supervised future forecasting
    w_temp: float = 0.05  # Temporal consistency / smoothness
    huber_delta: float = 1.0


class RepresentationObjectives(nn.Module):
    """
    Multi-task loss module coordinating JEPA latent prediction,
    masked sensor reconstruction, and supervised forecasting.
    """

    def __init__(self, config: Optional[LossWeightsConfig] = None):
        super().__init__()
        self.config = config or LossWeightsConfig()

    def future_latent_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Objective 1: Smooth L1 loss between predictor output and target encoder latent.
        
        Args:
            pred_latent: [B, T_out, N, D]
            target_latent: [B, T_out, N, D]
        """
        return F.smooth_l1_loss(pred_latent, target_latent, beta=1.0)

    def masked_reconstruction_loss(
        self,
        recon_x: torch.Tensor,
        orig_x: torch.Tensor,
        mask_patch: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Objective 2: Reconstruct masked observations.
        Computed ONLY on entries that were masked AND originally valid (non-null).

        Args:
            recon_x: [B, T_in, N, 1]
            orig_x: [B, T_in, N, C] (value is channel 0)
            mask_patch: [B, T_in, N] (1 = masked by model)
            valid_mask: [B, T_in, N] (1 = physically valid observation)
        """
        target_val = orig_x[..., 0:1]  # [B, T_in, N, 1]
        loss_mask = (mask_patch.unsqueeze(-1) * valid_mask.unsqueeze(-1)).bool()

        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=recon_x.device)

        diff = F.smooth_l1_loss(
            recon_x[loss_mask], target_val[loss_mask], beta=1.0, reduction="mean"
        )
        return diff

    def supervised_forecast_loss(
        self,
        pred_y: torch.Tensor,
        target_y: torch.Tensor,
        valid_mask_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Objective 3: Supervised forecasting loss on valid future observations.

        Args:
            pred_y: [B, T_out, N]
            target_y: [B, T_out, N]
            valid_mask_y: [B, T_out, N] (1 = valid observation)
        """
        mask = valid_mask_y.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_y.device)

        return F.smooth_l1_loss(
            pred_y[mask], target_y[mask], beta=self.config.huber_delta, reduction="mean"
        )

    def temporal_consistency_loss(self, pred_y: torch.Tensor) -> torch.Tensor:
        """
        Objective 4: Temporal smoothness penalty penalizing erratic frame-to-frame jumps.
        """
        # 1st-order temporal differences across T_out: [B, T_out - 1, N]
        diffs = pred_y[:, 1:, :] - pred_y[:, :-1, :]
        return torch.mean(torch.square(diffs))

    def compute_total_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        valid_mask_x: torch.Tensor,
        valid_mask_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total weighted loss and return isolated component metrics.

        Returns:
            - total_loss: scalar torch.Tensor for backpropagation
            - metrics: dictionary of float values for monitoring
        """
        l_jepa = self.future_latent_loss(
            model_outputs["pred_latent"], model_outputs["target_latent"]
        )

        l_mask = self.masked_reconstruction_loss(
            recon_x=model_outputs["recon_x"],
            orig_x=batch_x,
            mask_patch=model_outputs["mask_patch"],
            valid_mask=valid_mask_x,
        )

        l_sup = self.supervised_forecast_loss(
            pred_y=model_outputs["pred_forecast"],
            target_y=batch_y,
            valid_mask_y=valid_mask_y,
        )

        l_temp = self.temporal_consistency_loss(model_outputs["pred_forecast"])

        total = (
            self.config.w_jepa * l_jepa
            + self.config.w_mask * l_mask
            + self.config.w_sup * l_sup
            + self.config.w_temp * l_temp
        )

        metrics = {
            "loss_total": float(total.item()),
            "loss_jepa": float(l_jepa.item()),
            "loss_mask": float(l_mask.item()),
            "loss_sup": float(l_sup.item()),
            "loss_temp": float(l_temp.item()),
        }

        return total, metrics
