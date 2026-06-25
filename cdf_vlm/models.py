from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, feature_name: str, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_name = feature_name
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(features[self.feature_name]).squeeze(-1)


class ConcatFusion(nn.Module):
    def __init__(self, feature_names: list[str], input_dims: dict[str, int], hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.feature_names = feature_names
        total_dim = sum(input_dims[name] for name in feature_names)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([features[name] for name in self.feature_names], dim=-1)
        return self.classifier(x).squeeze(-1)


class GatedFusion(nn.Module):
    """Two-feature gated fusion intended for BiomedCLIP and Mammo-CLIP features."""

    def __init__(
        self,
        biomed_name: str,
        mammo_name: str,
        biomed_dim: int,
        mammo_dim: int,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.biomed_name = biomed_name
        self.mammo_name = mammo_name
        self.biomed_proj = nn.Sequential(nn.Linear(biomed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.mammo_proj = nn.Sequential(nn.Linear(mammo_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def fuse(self, features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        bio = F.normalize(self.biomed_proj(features[self.biomed_name]), dim=-1)
        mammo = F.normalize(self.mammo_proj(features[self.mammo_name]), dim=-1)
        gate = self.gate(torch.cat([bio, mammo], dim=-1))
        fused = gate * bio + (1.0 - gate) * mammo
        return fused, gate

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        fused, _ = self.fuse(features)
        return self.classifier(fused).squeeze(-1)


def build_model(
    model_type: str,
    feature_names: list[str],
    input_dims: dict[str, int],
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> nn.Module:
    if model_type.endswith("_probe"):
        feature_name = model_type.removesuffix("_probe")
        if feature_name not in input_dims:
            raise ValueError(f"Probe feature {feature_name!r} not found in {list(input_dims)}")
        return LinearProbe(feature_name, input_dims[feature_name], dropout=dropout)
    if model_type == "concat":
        return ConcatFusion(feature_names, input_dims, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == "gated":
        if len(feature_names) != 2:
            raise ValueError("gated fusion expects exactly two features, e.g. biomed and mammo")
        biomed_name, mammo_name = feature_names
        return GatedFusion(
            biomed_name=biomed_name,
            mammo_name=mammo_name,
            biomed_dim=input_dims[biomed_name],
            mammo_dim=input_dims[mammo_name],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type}")

