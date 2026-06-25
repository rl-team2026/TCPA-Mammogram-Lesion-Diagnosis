from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0, init_scale: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.lora_b.weight.data.mul_(init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


@dataclass(frozen=True)
class LoRAInjectionReport:
    vision_modules: int
    text_modules: int


def _replace_linear_modules(
    module: nn.Module,
    rank: int,
    alpha: int,
    target_keywords: tuple[str, ...],
    dropout: float,
    init_scale: float,
) -> int:
    count = 0
    for name, child in list(module.named_children()):
        lowered = name.lower()
        if isinstance(child, nn.Linear) and any(keyword in lowered for keyword in target_keywords):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, init_scale=init_scale))
            count += 1
        else:
            count += _replace_linear_modules(child, rank, alpha, target_keywords, dropout, init_scale)
    return count


def freeze_all_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def mark_lora_trainable(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.lora_a.weight.requires_grad = True
            child.lora_b.weight.requires_grad = True


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA adapter weights for lightweight saving."""
    return {
        key: value
        for key, value in module.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }


def inject_decoupled_lora(
    model: nn.Module,
    vision_rank: int = 64,
    text_rank: int = 16,
    vision_alpha: int | None = None,
    text_alpha: int | None = None,
    dropout: float = 0.05,
    lesion_mask_prior: float | None = None,
) -> LoRAInjectionReport:
    """Inject high-rank LoRA into visual branch and low-rank LoRA into text branch.

    lesion_mask_prior can be a mean lesion coverage value in [0, 1]. It scales LoRA
    initialization, giving lesion-heavy datasets slightly larger adapter variance.
    """
    freeze_all_parameters(model)
    init_scale = 1.0 + float(lesion_mask_prior or 0.0)
    vision_alpha = vision_alpha or vision_rank * 2
    text_alpha = text_alpha or text_rank * 2
    vision_count = 0
    text_count = 0
    if hasattr(model, "visual"):
        vision_count = _replace_linear_modules(
            model.visual,
            rank=vision_rank,
            alpha=vision_alpha,
            target_keywords=("q", "k", "v", "proj", "fc", "linear"),
            dropout=dropout,
            init_scale=init_scale,
        )
    text_branch = getattr(model, "text", None) or getattr(model, "transformer", None)
    if text_branch is not None:
        text_count = _replace_linear_modules(
            text_branch,
            rank=text_rank,
            alpha=text_alpha,
            target_keywords=("q", "k", "v", "proj", "fc", "linear"),
            dropout=dropout,
            init_scale=1.0,
        )
    mark_lora_trainable(model)
    return LoRAInjectionReport(vision_modules=vision_count, text_modules=text_count)
