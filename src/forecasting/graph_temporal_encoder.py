"""
Spatiotemporal Graph and Dilated Causal Temporal Convolutions for JEPAGraph-FM V3.0.

Components:
- GatedDilatedTemporalConv: Causal 1D convolution with exponential dilation receptive fields
- AdaptiveGraphLearning: Dynamic spatial correlations learned from features without memorizing node IDs
- GraphDiffusionConvolution: Spatial graph message passing supporting variable node counts N
- SpatiotemporalBlock: Modular residual ST block fusing temporal and spatial dynamics with adaptive graph learning
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D causal convolution with explicit left-padding to prevent future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, C, T]
        Returns:
            shape [B, C, T]
        """
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class GatedDilatedTemporalConv(nn.Module):
    """
    Gated Causal Dilated Temporal Convolution (Gated TCN).
    Split into filter (tanh) and gate (sigmoid) paths for stable gradient flow.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.filter_conv = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, C, T]
        Returns:
            shape [B, C, T]
        """
        filter_act = torch.tanh(self.filter_conv(x))
        gate_act = torch.sigmoid(self.gate_conv(x))
        return filter_act * gate_act


class AdaptiveGraphLearning(nn.Module):
    """
    Learns dynamic spatial inter-sensor correlations for arbitrary N nodes.
    Computes A_adp = Softmax(ReLU(Q(h) K(h)^T) / sqrt(d)).
    Does NOT memorize fixed sensor IDs; works seamlessly across heterogeneous N.
    """

    def __init__(self, hidden_dim: int, adaptive_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, adaptive_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, adaptive_dim, bias=False)
        self.scale = 1.0 / math.sqrt(adaptive_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B*T, N, hidden_dim] or [N, hidden_dim]
        Returns:
            A_adp: [N, N] normalized adaptive adjacency
        """
        if h.dim() == 2:
            q = self.q_proj(h)
            k = self.k_proj(h)
            sim = torch.mm(q, k.t()) * self.scale
            return F.softmax(F.relu(sim), dim=-1)
        elif h.dim() == 3:
            # Aggregate over batch*time to maintain a single [N, N] topology per batch
            h_mean = torch.mean(h, dim=0)  # [N, hidden_dim]
            q = self.q_proj(h_mean)
            k = self.k_proj(h_mean)
            sim = torch.mm(q, k.t()) * self.scale
            return F.softmax(F.relu(sim), dim=-1)
        else:
            raise ValueError(f"Unexpected shape {h.shape} for AdaptiveGraphLearning")


class GraphDiffusionConvolution(nn.Module):
    """
    Spatial graph diffusion convolution operating on arbitrary sensor graphs [N, N].
    Supports 1st-order and 2nd-order diffusion polynomials without hardcoded N.
    """

    def __init__(self, in_channels: int, out_channels: int, max_diffusion_step: int = 2):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        num_matrices = max_diffusion_step + 1  # I, A, A^2
        self.weights = nn.Parameter(torch.empty(num_matrices * in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in = self.weights.size(0)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, T, N, C] or [B, N, C]
            adj_matrix: normalized adjacency [N, N]
        Returns:
            shape with same spatial dimensions, projected to out_channels
        """
        has_time_dim = x.dim() == 4
        if has_time_dim:
            B, T, N, C = x.shape
            x_reshaped = x.reshape(B * T, N, C)
        else:
            B, N, C = x.shape
            x_reshaped = x

        # Diffusion supports: [X, A X, A^2 X]
        supports = [x_reshaped]
        curr_support = x_reshaped
        for _ in range(self.max_diffusion_step):
            # Matmul: [N, N] x [BT, N, C] -> einsum 'nm, bmc -> bnc'
            curr_support = torch.einsum("nm,bmc->bnc", adj_matrix, curr_support)
            supports.append(curr_support)

        # Concatenate diffusion steps along feature dimension: [BT, N, (K+1)*C]
        concat_supports = torch.cat(supports, dim=-1)
        out = torch.matmul(concat_supports, self.weights) + self.bias

        if has_time_dim:
            out = out.reshape(B, T, N, -1)
        return out


class SpatiotemporalBlock(nn.Module):
    """
    Unified Spatiotemporal Block combining:
    1. Gated causal dilated temporal convolution
    2. Optional adaptive graph learning combined with physical graph diffusion
    3. Residual skip connection and LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        dilation: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.1,
        use_adaptive_graph: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_adaptive_graph = use_adaptive_graph
        self.temporal_conv = GatedDilatedTemporalConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.spatial_conv = GraphDiffusionConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            max_diffusion_step=2,
        )
        if use_adaptive_graph:
            self.adaptive_graph = AdaptiveGraphLearning(hidden_dim=hidden_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.adaptive_graph = None
            self.alpha = None

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, T, N, hidden_dim]
            adj_matrix: shape [N, N]
        Returns:
            shape [B, T, N, hidden_dim]
        """
        residual = x
        B, T, N, C = x.shape

        # 1. Temporal Conv: reshape to [B*N, C, T]
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        t_out = self.temporal_conv(x_t)  # [B*N, C, T]
        t_out = t_out.reshape(B, N, C, T).permute(0, 3, 1, 2)  # [B, T, N, C]

        # 2. Determine effective graph adjacency
        if self.use_adaptive_graph and self.adaptive_graph is not None:
            # Extract features for adaptive topology: [B*T, N, C]
            h_flat = t_out.reshape(B * T, N, C)
            a_adp = self.adaptive_graph(h_flat)
            a_weight = torch.sigmoid(self.alpha)
            effective_adj = a_weight * adj_matrix + (1.0 - a_weight) * a_adp
        else:
            effective_adj = adj_matrix

        # 3. Spatial Graph Diffusion
        s_out = self.spatial_conv(t_out, effective_adj)  # [B, T, N, C]

        # 4. Residual and Norm
        out = self.norm(residual + self.dropout(s_out))
        return out
