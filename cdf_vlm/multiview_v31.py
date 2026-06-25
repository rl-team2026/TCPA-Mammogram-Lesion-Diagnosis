"""V3.1 multi-view modules — improved over V3.

V3 learnings:
  - BiomedCLIP text encoder: massive improvement (+0.0246 AUROC) — keep
  - Spatial attention consistency: no gain (crude attention extraction) — drop
  - Mask-guided cross-attention: harmful (single-token MHA = no-op) — drop

V3.1 new modules:
  1. Contrastive cross-view loss (batch-level, SimCLR-style)
  2. Channel-wise attention fusion (per-dimension weights, ~30K params)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ══════════════════════════════════════════════════════════════
# V3.1 Module 1: Contrastive Cross-View Loss
# ══════════════════════════════════════════════════════════════

class ContrastiveCrossViewLoss(nn.Module):
    """Batch-level contrastive loss: pull same-lesion CC/MLO pairs together,
    push different-lesion pairs apart.

    For each sample i in a batch of size B:
      - Positive pair: (f_cc_i, f_mlo_i) — same lesion
      - Negative pairs: (f_cc_i, f_mlo_j) for j ≠ i — different lesions

    Uses symmetric InfoNCE loss with learnable temperature.
    """

    def __init__(self, embed_dim: int = 512, temperature: float = 0.07):
        super().__init__()
        # Learnable temperature (initialized small for hard contrast)
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / temperature).log()
        )

    def forward(self, cc_features: torch.Tensor,
                mlo_features: torch.Tensor) -> torch.Tensor:
        """Compute symmetric contrastive loss.

        Args:
            cc_features: (B, D) normalized CC features
            mlo_features: (B, D) normalized MLO features

        Returns:
            scalar contrastive loss
        """
        B = cc_features.shape[0]
        if B < 2:
            return cc_features.new_tensor(0.0)

        # L2 normalize
        cc = F.normalize(cc_features, dim=-1)
        mlo = F.normalize(mlo_features, dim=-1)

        # Temperature scaling
        scale = self.logit_scale.exp()

        # CC→MLO: for each CC, find its MLO pair among all MLOs
        logits_cc_to_mlo = scale * (cc @ mlo.T)  # (B, B)
        labels = torch.arange(B, device=cc.device)

        # MLO→CC: symmetric
        logits_mlo_to_cc = scale * (mlo @ cc.T)  # (B, B)

        loss_cc = F.cross_entropy(logits_cc_to_mlo, labels)
        loss_mlo = F.cross_entropy(logits_mlo_to_cc, labels)

        return (loss_cc + loss_mlo) * 0.5


# ══════════════════════════════════════════════════════════════
# V3.1 Module 2: Channel-wise Attention Fusion
# ══════════════════════════════════════════════════════════════

class ChannelWiseAttentionFusion(nn.Module):
    """Per-dimension CC/MLO fusion weights via squeeze-excitation style attention.

    Instead of α·CC + (1-α)·MLO (scalar weight), this learns a 512-dim weight
    vector w, enabling fine-grained per-channel blending of CC and MLO features.

    Architecture: Linear(1024→64) → GELU → Linear(64→512) → Sigmoid
    Total: ~66K params (still lightweight)
    """

    def __init__(self, embed_dim: int = 512, bottleneck: int = 64):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, embed_dim),
        )
        # Initialize to output ~0.5 (equal-weight average)
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.constant_(self.weight_net[-1].bias, 0.0)

    def forward(self, cc_feature: torch.Tensor,
                mlo_feature: torch.Tensor) -> torch.Tensor:
        """Fuse CC and MLO with per-channel learned weights.

        Args:
            cc_feature: (B, D)
            mlo_feature: (B, D)

        Returns:
            fused: (B, D)
        """
        w = torch.sigmoid(self.weight_net(
            torch.cat([cc_feature, mlo_feature], dim=-1)
        ))
        return w * cc_feature + (1.0 - w) * mlo_feature


# ══════════════════════════════════════════════════════════════
# V3.1 BiomedCLIP Text Prompt Encoder (same as V3, kept)
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# V3.1 Module 3: Attention Regularization
# ══════════════════════════════════════════════════════════════

class MaskAttentionRegularization(nn.Module):
    """Penalize mismatch between model spatial attention and target mask.

    Uses ViT feature norms as a proxy for spatial attention.
    Target can be: binary mask, Gaussian-blurred mask, or dilated mask.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.grid_size = img_size // patch_size  # 14

    def _extract_sensitivity(self, clip_model, image):
        """Use patch feature L2 norm as spatial sensitivity proxy."""
        try:
            visual = clip_model.visual
            # Get patch embeddings (before transformer)
            if hasattr(visual, 'conv1'):
                x = visual.conv1(image)
            elif hasattr(visual, 'patch_embed'):
                x = visual.patch_embed(image)
            else:
                return None
            B, C, H, W = x.shape
            sensitivity = x.abs().mean(dim=1)  # (B, H, W)
            sensitivity = sensitivity / (sensitivity.max() + 1e-8)
            return sensitivity
        except Exception:
            return None

    def _prepare_target(self, mask, mode="binary", sigma=3, radius=3):
        """Convert lesion mask to attention target."""
        import torch.nn.functional as F
        B, _, H_m, W_m = mask.shape
        target = F.interpolate(mask, size=(self.grid_size, self.grid_size),
                               mode='bilinear', align_corners=False)
        target = target.squeeze(1)  # (B, H, W)

        if mode == "binary":
            pass
        elif mode == "gaussian":
            from torchvision.transforms.functional import gaussian_blur
            target = target.unsqueeze(1)
            ks = int(sigma * 3) | 1  # odd kernel
            target = gaussian_blur(target, kernel_size=ks, sigma=sigma)
            target = target.squeeze(1)
        elif mode == "dilated":
            target = F.max_pool2d(
                F.pad(target.unsqueeze(1), [radius]*4, mode='reflect'),
                kernel_size=2*radius+1, stride=1
            ).squeeze(1)

        # Normalize
        target = target / (target.max() + 1e-8)
        return target

    def forward(self, clip_model, cc_image, cc_mask, mlo_image, mlo_mask,
                mode="binary", sigma=3, radius=3):
        sens_cc = self._extract_sensitivity(clip_model, cc_image)
        sens_mlo = self._extract_sensitivity(clip_model, mlo_image)

        if sens_cc is None or sens_mlo is None:
            return cc_image.new_tensor(0.0)

        target_cc = self._prepare_target(cc_mask, mode, sigma, radius)
        target_mlo = self._prepare_target(mlo_mask, mode, sigma, radius)

        loss_cc = F.mse_loss(sens_cc, target_cc.detach())
        loss_mlo = F.mse_loss(sens_mlo, target_mlo.detach())
        return (loss_cc + loss_mlo) * 0.5


class BiomedCLIPTextPromptEncoder(nn.Module):
    """Use BiomedCLIP's own PubMedBERT text encoder for rich prompt vectors."""

    def __init__(self, clip_model: nn.Module, tokenizer,
                 embed_dim: int = 512, freeze: bool = True):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if freeze:
            text_encoder = getattr(clip_model, 'text', None) or \
                           getattr(clip_model, 'transformer', None)
            if text_encoder is not None:
                for p in text_encoder.parameters():
                    p.requires_grad = False

    def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        if isinstance(tokens, dict):
            tokens = {k: v.to(device) for k, v in tokens.items()}
        else:
            tokens = tokens.to(device)
        with torch.set_grad_enabled(not all(
            p.requires_grad == False
            for p in self.clip_model.parameters()
        )):
            text_features = self.clip_model.encode_text(tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        return self.text_proj(text_features)


# ══════════════════════════════════════════════════════════════
# V3.1 Model: DDSMGeometryPromptModelV31
# ══════════════════════════════════════════════════════════════

class DDSMGeometryPromptModelV31(nn.Module):
    """V3.1 multi-view model: BiomedCLIP prompts + contrastive loss + channel fusion."""

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        use_text_prompts: bool = True,
        use_aligner: bool = False,      # default off per V3 findings
        use_consistency: bool = True,
        use_contrastive: bool = True,
        use_channel_fusion: bool = True,
        use_attn_reg: bool = False,
        attn_reg_mode: str = "binary",
        attn_reg_sigma: float = 3.0,
        attn_reg_radius: int = 3,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.use_text_prompts = use_text_prompts
        self.use_aligner = use_aligner
        self.use_consistency = use_consistency
        self.use_contrastive = use_contrastive
        self.use_channel_fusion = use_channel_fusion
        self.use_attn_reg = use_attn_reg
        self.attn_reg_mode = attn_reg_mode
        self.attn_reg_sigma = attn_reg_sigma
        self.attn_reg_radius = attn_reg_radius
        if self.use_attn_reg:
            self.attn_reg = MaskAttentionRegularization()

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # BiomedCLIP text prompt encoder
        self.text_prompt_encoder = BiomedCLIPTextPromptEncoder(
            clip_model, tokenizer, embed_dim=embed_dim, freeze=freeze_backbone
        )
        from cdf_vlm.multiview_v2 import PromptAttentionGate
        self.prompt_gate = PromptAttentionGate(embed_dim=embed_dim)

        # Contrastive cross-view loss (V3.1 new)
        self.contrastive_loss_fn = ContrastiveCrossViewLoss(embed_dim=embed_dim)

        # Channel-wise fusion (V3.1 new)
        self.channel_fusion = ChannelWiseAttentionFusion(embed_dim=embed_dim)

        # Optional aligner
        if self.use_aligner:
            from cdf_vlm.multiview_v2 import CrossViewFeatureAligner
            self.aligner = CrossViewFeatureAligner(embed_dim=embed_dim)
            self.aligner_weight = nn.Parameter(torch.tensor(0.3))

        # Global-local gate (same as V2)
        self.local_global_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )

        # Classifier
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

    def _combine_global_local(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        gate = self.local_global_gate(torch.cat([g, l], dim=-1))
        return gate * l + (1.0 - gate) * g

    def forward(
        self,
        cc_image, cc_mask, cc_description,
        mlo_image, mlo_mask, mlo_description,
        side_ids,
    ) -> dict[str, torch.Tensor]:
        # Global + local features
        cc_feature = self._combine_global_local(
            self.encode_image(cc_image),
            self.encode_image(self._masked_image(cc_image, cc_mask)),
        )
        mlo_feature = self._combine_global_local(
            self.encode_image(mlo_image),
            self.encode_image(self._masked_image(mlo_image, mlo_mask)),
        )

        # Optional aligner
        align_loss = cc_image.new_tensor(0.0)
        if self.use_aligner:
            cc_aligned, mlo_aligned, align_loss = self.aligner(cc_feature, mlo_feature)
            w = torch.sigmoid(self.aligner_weight)
            cc_feature = cc_feature + w * cc_aligned
            mlo_feature = mlo_feature + w * mlo_aligned

        # Text prompts via BiomedCLIP
        if self.use_text_prompts:
            cc_prompt = self.text_prompt_encoder.encode_texts(
                cc_description, device=cc_image.device
            )
            mlo_prompt = self.text_prompt_encoder.encode_texts(
                mlo_description, device=mlo_image.device
            )
            cc_feature = self.prompt_gate(cc_feature, cc_prompt)
            mlo_feature = self.prompt_gate(mlo_feature, mlo_prompt)

        # V3.1: Contrastive cross-view loss
        contrastive_loss = cc_image.new_tensor(0.0)
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss_fn(cc_feature, mlo_feature)

        # V3.1: Channel-wise attention fusion
        if self.use_channel_fusion:
            fused = self.channel_fusion(cc_feature, mlo_feature)
        else:
            fused = 0.5 * (cc_feature + mlo_feature)

        logits = self.classifier(fused).squeeze(-1)

        # Cross-view consistency loss
        consistency_loss = cc_image.new_tensor(0.0)
        if self.use_consistency:
            cc_logits = self.classifier(cc_feature).squeeze(-1)
            mlo_logits = self.classifier(mlo_feature).squeeze(-1)
            consistency_loss = (
                F.mse_loss(cc_logits, mlo_logits.detach())
                + F.mse_loss(mlo_logits, cc_logits.detach())
            )

        # Attention regularization loss
        attn_reg_loss = cc_image.new_tensor(0.0)
        if self.use_attn_reg:
            attn_reg_loss = self.attn_reg(
                self.clip_model, cc_image, cc_mask, mlo_image, mlo_mask,
                mode=self.attn_reg_mode, sigma=self.attn_reg_sigma,
                radius=self.attn_reg_radius,
            )

        return {
            "logits": logits,
            "projection_loss": align_loss,
            "consistency_loss": consistency_loss,
            "contrastive_loss": contrastive_loss,
            "attn_reg_loss": attn_reg_loss,
            "cc_feature": cc_feature,
            "mlo_feature": mlo_feature,
        }


# ══════════════════════════════════════════════════════════════
# V3.1 Single-view model
# ══════════════════════════════════════════════════════════════

class DDSMSingleViewPromptModelV31(nn.Module):
    """Single-view branch for V3.1 joint training."""

    def __init__(self, clip_model, tokenizer, embed_dim=512,
                 freeze_backbone=True, use_text_prompts=True):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.use_text_prompts = use_text_prompts
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.text_prompt_encoder = BiomedCLIPTextPromptEncoder(
            clip_model, tokenizer, embed_dim=embed_dim, freeze=freeze_backbone
        )
        from cdf_vlm.multiview_v2 import PromptAttentionGate
        self.prompt_gate = PromptAttentionGate(embed_dim=embed_dim)
        self.local_global_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def encode_image(self, image):
        return F.normalize(self.clip_model.encode_image(image).float(), dim=-1)

    def _masked_image(self, image, mask):
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask, size=image.shape[-2:], mode="nearest")
        return image * mask

    def forward(self, image, mask, description):
        g = self.encode_image(image)
        l = self.encode_image(self._masked_image(image, mask))
        gate = self.local_global_gate(torch.cat([g, l], dim=-1))
        feature = gate * l + (1.0 - gate) * g
        if self.use_text_prompts:
            prompt = self.text_prompt_encoder.encode_texts(description, device=image.device)
            feature = self.prompt_gate(feature, prompt)
        logits = self.classifier(feature).squeeze(-1)
        return {"logits": logits, "feature": feature}
