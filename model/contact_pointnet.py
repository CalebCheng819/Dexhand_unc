"""Lightweight PointNet encoder for heatmap-weighted object point clouds.

Input:  (B, N, 4)  — N points, each with (x, y, z, heatmap_weight)
Output: (B, out_dim) — global feature vector

Architecture:
  Per-point MLP:  4 → 32 → 64 → 128  (shared weights, BN + ReLU)
  Global max pool: (B, N, 128) → (B, 128)
  Global MLP:     128 → 64 → out_dim  (BN + ReLU, no final activation)

Total params: ~24 K (negligible vs 5.7 M diffusion model).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContactPointNet(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim

        # Per-point shared MLP (operates on each point independently)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Global MLP (operates on pooled feature)
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) or (N, in_dim) for single sample
        Returns:
            (B, out_dim) global feature
        """
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True

        B, N, C = x.shape
        # Apply per-point MLP: reshape to (B*N, C) for BatchNorm1d
        h = x.reshape(B * N, C)
        h = self.point_mlp(h)           # (B*N, 128)
        h = h.reshape(B, N, 128)

        # Global max pool
        h = h.max(dim=1).values         # (B, 128)

        # Global MLP
        out = self.global_mlp(h)        # (B, out_dim)

        if squeeze:
            out = out.squeeze(0)
        return out
