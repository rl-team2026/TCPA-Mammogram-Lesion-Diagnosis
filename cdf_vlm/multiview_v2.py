"""Improved multi-view fusion modules (v2).

Key improvements over v1:
  - CrossViewFeatureAligner: feature-space alignment replaces pixel-space STN warping.
  - LightweightViewFusion: per-sample view weighting (~16K params) replaces
    the broken ViewAwareBilateralFusion whose single-token cross-attention was
    a no-op and whose ~3-4M parameters caused overfitting.
  - Learnable aligner mixing weight replaces hardcoded 0.3.
  - Optional cross-view consistency loss for multi-view regularization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cdf_vlm.text_prompts import PathologyPromptEncoder


# ═══════════════════════════════════════════════════════════════════════
# V1 modules kept for reuse (PromptAttentionGate, PathologyPromptEncoder)
# ═══════════════════════════════════════════════════════════════════════

class PromptAttentionGate(nn.Module):
    """Inject pathology prompt vector into visual features via feature-wise attention."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.shift = nn.Linear(embed_dim, embed_dim)

    def forward(self, visual_feature: torch.Tensor, prompt_feature: torch.Tensor) -> torch.Tensor:
        gate = self.gate(prompt_feature)
        shift = self.shift(prompt_feature)
        return visual_feature * (1.0 + gate) + shift


# ═══════════════════════════════════════════════════════════════════════
# V2: Feature-space cross-view alignment (replaces ImageSTN)
# ═══════════════════════════════════════════════════════════════════════

class CrossViewFeatureAligner(nn.Module):
    """Learn a shared latent space where CC and MLO features of the same lesion align.

    Instead of trying to warp pixels (ill-posed 2D→2D projection of 3D structure),
    this module:
      1. Projects CC and MLO features into a shared subspace via learnable MLPs.
      2. Applies a symmetric projection loss encouraging same-lesion features to be similar.
      3. The alignment is bidirectional: CC→MLO and MLO→CC.

    The loss is designed to be a meaningful training signal even with frozen backbones.
    """

    def __init__(self, embed_dim: int = 512, bottleneck_dim: int = 256):
        super().__init__()
        self.cc_proj = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.mlo_proj = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        # Confidence gate: model learns when alignment is reliable
        self.confidence = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, cc_feature: torch.Tensor, mlo_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment loss and return aligned features.

        Returns:
            cc_aligned: CC features projected toward MLO space.
            mlo_aligned: MLO features projected toward CC space.
            align_loss: symmetric MSE + cosine embedding loss.
        """
        # Bidirectional projection
        cc_in_mlo_space = self.cc_proj(cc_feature)
        mlo_in_cc_space = self.mlo_proj(mlo_feature)

        # Symmetric MSE loss (detach targets to avoid collapse)
        mse_loss = (
            F.mse_loss(cc_in_mlo_space, mlo_feature.detach())
            + F.mse_loss(mlo_in_cc_space, cc_feature.detach())
        ) * 0.5

        # Cosine embedding loss: pull same-lesion features together
        cosine_loss = (1.0 - F.cosine_similarity(
            F.normalize(cc_in_mlo_space, dim=-1),
            F.normalize(mlo_feature.detach(), dim=-1),
            dim=-1,
        )).mean() + (1.0 - F.cosine_similarity(
            F.normalize(mlo_in_cc_space, dim=-1),
            F.normalize(cc_feature.detach(), dim=-1),
            dim=-1,
        )).mean()

        # Confidence-weighted combination
        conf = self.confidence(torch.cat([cc_feature, mlo_feature], dim=-1))
        align_loss = conf.mean() * (mse_loss + 0.1 * cosine_loss)

        return cc_in_mlo_space, mlo_in_cc_space, align_loss


# ═══════════════════════════════════════════════════════════════════════
# V2: Lightweight view fusion (replaces ViewAwareBilateralFusion)
# ═══════════════════════════════════════════════════════════════════════
#
# The original ViewAwareBilateralFusion had two critical flaws:
#   1. Single-token cross-attention is a no-op: softmax(scalar) ≡ 1.0,
#      so the MHA output is just the value vector unchanged.
#   2. ~3–4M parameters on a small dataset → overfitting.
#
# LightweightViewFusion uses a tiny bottleneck (16-dim) to predict a
# per-sample scalar CC weight. It is initialized to output ~0.5 so
# training starts from the known-good simple-average baseline.
# Total: ~33K params (100× fewer than the original).

class LightweightViewFusion(nn.Module):
    """Per-sample learnable view weighting with minimal parameters (~33K).

    Replaces ViewAwareBilateralFusion whose single-token cross-attention was a
    no-op (softmax of a scalar ≡ 1.0) and whose ~3–4M params caused overfitting.

    Uses a 16-dim bottleneck to predict a per-sample CC-vs-MLO weight.
    Initialized so training starts from equal-weight averaging.
    """

    def __init__(self, embed_dim: int = 512, bottleneck: int = 16):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.constant_(self.weight_net[-1].bias, 0.0)  # sigmoid(0) = 0.5

    def forward(
        self, cc_feature: torch.Tensor, mlo_feature: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.weight_net(
            torch.cat([cc_feature, mlo_feature], dim=-1)
        ))
        return alpha * cc_feature + (1.0 - alpha) * mlo_feature


ViewAwareBilateralFusion = LightweightViewFusion


# ═══════════════════════════════════════════════════════════════════════
# V2 Model: DDSMGeometryPromptModelV2
# ═══════════════════════════════════════════════════════════════════════

class DDSMGeometryPromptModelV2(nn.Module):
    """Improved multi-view model with feature-space alignment and lightweight fusion.

    V2 changes:
      - CrossViewFeatureAligner replaces pixel-space STN warping.
      - LightweightViewFusion (~33K params) replaces the ~3M-param bilateral fusion.
      - Learnable aligner mixing weight replaces hardcoded 0.3.
      - Optional cross-view consistency loss for multi-view regularization.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        prompt_keywords: list[str] | None = None,
        use_aligner: bool = True,
        use_bilateral: bool = True,
        use_text_prompts: bool = True,
        use_consistency: bool = False,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.use_aligner = use_aligner
        self.use_bilateral = use_bilateral
        self.use_text_prompts = use_text_prompts
        self.use_consistency = use_consistency
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.prompt_encoder = PathologyPromptEncoder(embed_dim=embed_dim, keywords=prompt_keywords)
        self.prompt_gate = PromptAttentionGate(embed_dim=embed_dim)

        # V2 modules
        self.aligner = CrossViewFeatureAligner(embed_dim=embed_dim)
        self.bilateral_fusion = LightweightViewFusion(embed_dim=embed_dim)

        # Learnable aligner mixing weight (replaces hardcoded 0.3)
        self.aligner_weight = nn.Parameter(torch.tensor(0.3))

        self.local_global_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.clip_model.encode_image(image)
        return F.normalize(feature.float(), dim=-1)

    def _masked_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
        return image * mask

    def _combine_global_local(
        self, global_feature: torch.Tensor, local_feature: torch.Tensor
    ) -> torch.Tensor:
        gate = self.local_global_gate(
            torch.cat([global_feature, local_feature], dim=-1)
        )
        return gate * local_feature + (1.0 - gate) * global_feature

    def forward(
        self,
        cc_image: torch.Tensor,
        cc_mask: torch.Tensor,
        cc_description: list[str],
        mlo_image: torch.Tensor,
        mlo_mask: torch.Tensor,
        mlo_description: list[str],
        side_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        # Global + local features
        cc_global = self.encode_image(cc_image)
        mlo_global = self.encode_image(mlo_image)
        cc_masked = self._masked_image(cc_image, cc_mask)
        mlo_masked = self._masked_image(mlo_image, mlo_mask)
        cc_local = self.encode_image(cc_masked)
        mlo_local = self.encode_image(mlo_masked)

        cc_feature = self._combine_global_local(cc_global, cc_local)
        mlo_feature = self._combine_global_local(mlo_global, mlo_local)

        # V2: Feature-space alignment with learnable mixing weight
        if self.use_aligner:
            cc_aligned, mlo_aligned, align_loss = self.aligner(cc_feature, mlo_feature)
            w = torch.sigmoid(self.aligner_weight)  # constrain to (0, 1)
            cc_feature = cc_feature + w * cc_aligned
            mlo_feature = mlo_feature + w * mlo_aligned
        else:
            align_loss = cc_image.new_tensor(0.0)

        # Text prompts
        if self.use_text_prompts:
            cc_prompt = self.prompt_encoder.encode_texts(cc_description, device=cc_image.device)
            mlo_prompt = self.prompt_encoder.encode_texts(mlo_description, device=mlo_image.device)
            cc_feature = self.prompt_gate(cc_feature, cc_prompt)
            mlo_feature = self.prompt_gate(mlo_feature, mlo_prompt)

        # V2: Lightweight view fusion
        if self.use_bilateral:
            fused = self.bilateral_fusion(cc_feature, mlo_feature)
        else:
            fused = 0.5 * (cc_feature + mlo_feature)

        logits = self.classifier(fused).squeeze(-1)

        # Cross-view consistency loss (MSE between CC-only and MLO-only predictions)
        if self.use_consistency:
            cc_logits = self.classifier(cc_feature).squeeze(-1)
            mlo_logits = self.classifier(mlo_feature).squeeze(-1)
            consistency_loss = F.mse_loss(cc_logits, mlo_logits.detach()) + F.mse_loss(
                mlo_logits, cc_logits.detach()
            )
        else:
            consistency_loss = cc_image.new_tensor(0.0)

        return {
            "logits": logits,
            "projection_loss": align_loss,
            "consistency_loss": consistency_loss,
            "cc_feature": cc_feature,
            "mlo_feature": mlo_feature,
        }


# ═══════════════════════════════════════════════════════════════════════
# V2 Single-view model (same as V1, reused for joint training)
# ═══════════════════════════════════════════════════════════════════════

class DDSMSingleViewPromptModelV2(nn.Module):
    """Single-view branch for joint training (same architecture as V1)."""

    def __init__(
        self,
        clip_model: nn.Module,
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        prompt_keywords: list[str] | None = None,
        use_text_prompts: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.use_text_prompts = use_text_prompts
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.prompt_encoder = PathologyPromptEncoder(embed_dim=embed_dim, keywords=prompt_keywords)
        self.prompt_gate = PromptAttentionGate(embed_dim=embed_dim)
        self.local_global_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip_model.encode_image(image).float(), dim=-1)

    def _masked_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
        return image * mask

    def forward(
        self, image: torch.Tensor, mask: torch.Tensor, description: list[str]
    ) -> dict[str, torch.Tensor]:
        global_feature = self.encode_image(image)
        local_feature = self.encode_image(self._masked_image(image, mask))
        gate = self.local_global_gate(
            torch.cat([global_feature, local_feature], dim=-1)
        )
        feature = gate * local_feature + (1.0 - gate) * global_feature
        if self.use_text_prompts:
            prompt = self.prompt_encoder.encode_texts(description, device=image.device)
            feature = self.prompt_gate(feature, prompt)
        logits = self.classifier(feature).squeeze(-1)
        return {"logits": logits, "feature": feature}
