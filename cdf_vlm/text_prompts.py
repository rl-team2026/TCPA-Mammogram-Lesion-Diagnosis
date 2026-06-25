from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn


DEFAULT_KEYWORDS = [
    "mass",
    "calcification",
    "irregular",
    "spiculated",
    "circumscribed",
    "obscured",
    "microlobulated",
    "architectural distortion",
    "clustered",
    "pleomorphic",
    "linear",
    "segmental",
    "left breast",
    "right breast",
    "cc view",
    "mlo view",
]


KEYWORD_ALIASES = {
    "mass": [r"\bmass\b", r"\btumou?r\b"],
    "calcification": [r"\bcalcification\w*\b", r"\bmicrocalcification\w*\b"],
    "irregular": [r"\birregular\b"],
    "spiculated": [r"\bspiculated\b", r"\bspiculation\b"],
    "circumscribed": [r"\bcircumscribed\b"],
    "obscured": [r"\bobscured\b"],
    "microlobulated": [r"\bmicrolobulated\b"],
    "architectural distortion": [r"\barchitectural distortion\b"],
    "clustered": [r"\bclustered\b", r"\bcluster\b"],
    "pleomorphic": [r"\bpleomorphic\b"],
    "linear": [r"\blinear\b"],
    "segmental": [r"\bsegmental\b"],
    "left breast": [r"\bleft breast\b", r"\bleft\b"],
    "right breast": [r"\bright breast\b", r"\bright\b"],
    "cc view": [r"\bcc view\b", r"\b cc\b"],
    "mlo view": [r"\bmlo view\b", r"\bmlo\b"],
}


def extract_pathology_keywords(text: str, keywords: list[str] | None = None) -> list[str]:
    text_l = str(text).lower()
    out = []
    for keyword in keywords or DEFAULT_KEYWORDS:
        patterns = KEYWORD_ALIASES.get(keyword, [re.escape(keyword)])
        if any(re.search(pattern, text_l) for pattern in patterns):
            out.append(keyword)
    return out


def keyword_multihot(texts: list[str], keywords: list[str] | None = None) -> torch.Tensor:
    vocab = keywords or DEFAULT_KEYWORDS
    rows = []
    for text in texts:
        present = set(extract_pathology_keywords(text, vocab))
        rows.append([1.0 if keyword in present else 0.0 for keyword in vocab])
    return torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class PromptBatch:
    multihot: torch.Tensor
    keywords: list[list[str]]


class PathologyPromptEncoder(nn.Module):
    """Keyword prompt encoder for pathology-text-guided visual attention."""

    def __init__(self, embed_dim: int = 512, keywords: list[str] | None = None):
        super().__init__()
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.proj = nn.Sequential(
            nn.Linear(len(self.keywords), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_texts(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        x = keyword_multihot(texts, self.keywords)
        if device is not None:
            x = x.to(device)
        return self.forward(x)

    def forward(self, keyword_features: torch.Tensor) -> torch.Tensor:
        return self.proj(keyword_features.float())

