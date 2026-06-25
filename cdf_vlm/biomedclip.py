from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


def load_local_biomedclip(
    model_dir: str | Path = "external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    device: str | torch.device | None = None,
):
    """Load local BiomedCLIP model, preprocess transform, and tokenizer."""
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS

    model_dir = Path(model_dir)
    config_path = model_dir / "open_clip_config.json"
    weights_path = model_dir / "open_clip_pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"BiomedCLIP config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"BiomedCLIP weights not found: {weights_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    local_bert_config = model_dir / "pubmedbert_config"
    if local_bert_config.exists():
        config["model_cfg"]["text_cfg"]["hf_model_name"] = str(local_bert_config)
        config["model_cfg"]["text_cfg"]["hf_model_pretrained"] = False
        config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(model_dir)

    model_name = "biomedclip_local_cdf_vlm"
    _MODEL_CONFIGS[model_name] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weights_path),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    tokenizer = get_tokenizer(model_name)
    if device is not None:
        model = model.to(device)
    return model, preprocess, tokenizer


class TinyVisualEncoder(nn.Module):
    """Small image encoder used only for smoke tests and CI."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(image))


class TinyTextEncoder(nn.Module):
    """Small text encoder used only for smoke tests and CI."""

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 128, embed_dim: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).float().unsqueeze(-1)
        embedded = self.token_embed(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class TinyBiomedCLIP(nn.Module):
    """BiomedCLIP-compatible tiny backbone for fast end-to-end smoke runs."""

    def __init__(self, embed_dim: int = 512, vocab_size: int = 4096):
        super().__init__()
        self.visual = TinyVisualEncoder(embed_dim=embed_dim)
        self.text = TinyTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text(tokens)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = torch.nn.functional.normalize(self.encode_image(image), dim=-1)
        text_features = torch.nn.functional.normalize(self.encode_text(text), dim=-1)
        return image_features, text_features, self.logit_scale.exp()


class SimpleTokenizer:
    """Deterministic local tokenizer for smoke tests.

    It intentionally does not emulate PubMedBERT tokenization; it only provides a
    BiomedCLIP-compatible callable returning integer tensors.
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size

    def __call__(self, texts, context_length: int = 256) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        tokens = torch.zeros((len(texts), context_length), dtype=torch.long)
        for row_idx, text in enumerate(texts):
            encoded = str(text).lower().encode("utf-8")[:context_length]
            for col_idx, value in enumerate(encoded):
                tokens[row_idx, col_idx] = int(value % (self.vocab_size - 1)) + 1
        return tokens


def _mock_preprocess(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def load_mock_biomedclip(
    device: str | torch.device | None = None,
    image_size: int = 224,
    embed_dim: int = 512,
):
    """Return a tiny BiomedCLIP-compatible model for fast smoke tests."""
    model = TinyBiomedCLIP(embed_dim=embed_dim)
    if device is not None:
        model = model.to(device)
    return model, _mock_preprocess(image_size=image_size), SimpleTokenizer()


def encode_biomedclip_image(model, image: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        return model.encode_image(image)
    if hasattr(model, "visual"):
        return model.visual(image)
    raise AttributeError("BiomedCLIP model does not expose encode_image or visual.")


def encode_biomedclip_text(model, tokens: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_text"):
        return model.encode_text(tokens)
    raise AttributeError("BiomedCLIP model does not expose encode_text.")
