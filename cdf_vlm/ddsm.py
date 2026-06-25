from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


def resolve_data_path(root: str | Path | None, path_value: str) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def load_mask_tensor(path: str | Path, target_hw: tuple[int, int] = (1520, 912)) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        mask = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
    else:
        mask_img = Image.open(path).convert("L").resize(target_hw[::-1])
        mask = torch.as_tensor(list(mask_img.getdata()), dtype=torch.float32).reshape(target_hw)
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[0] != 1:
        mask = mask[:1]
    mask = (mask > 0).float()
    if tuple(mask.shape[-2:]) != target_hw:
        mask = F.interpolate(mask.unsqueeze(0), size=target_hw, mode="nearest").squeeze(0)
    return mask


class DDSMSingleViewDataset(Dataset):
    """Single-view DDSM CSV: image + mask + description + label."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path | None,
        image_transform: Any,
        target_hw: tuple[int, int] = (1520, 912),
    ):
        self.table = pd.read_csv(csv_path)
        self.data_root = Path(data_root) if data_root else None
        self.image_transform = image_transform
        self.target_hw = target_hw
        required = {"image_path", "mask_path", "description", "label", "side_id", "view"}
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(f"Single-view CSV missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.table.iloc[idx]
        image_path = resolve_data_path(self.data_root, row["image_path"])
        mask_path = resolve_data_path(self.data_root, row["mask_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        mask = load_mask_tensor(mask_path, target_hw=tuple(image_tensor.shape[-2:]))
        return {
            "image": image_tensor,
            "mask": mask,
            "description": str(row["description"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "side_id": str(row["side_id"]),
            "view": str(row["view"]),
            "image_path": str(image_path),
        }


class DDSMPairedViewDataset(Dataset):
    """Paired DDSM CSV produced by preprocess_ddsm.py."""

    def __init__(
        self,
        pair_csv_path: str | Path,
        data_root: str | Path | None,
        image_transform: Any,
        target_hw: tuple[int, int] = (1520, 912),
    ):
        self.table = pd.read_csv(pair_csv_path)
        self.data_root = Path(data_root) if data_root else None
        self.image_transform = image_transform
        self.target_hw = target_hw
        required = {
            "side_id",
            "cc_image_path",
            "cc_mask_path",
            "cc_description",
            "mlo_image_path",
            "mlo_mask_path",
            "mlo_description",
            "label",
        }
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(f"Paired CSV missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.table)

    def _load_image_mask(self, image_col: str, mask_col: str, row) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = resolve_data_path(self.data_root, row[image_col])
        mask_path = resolve_data_path(self.data_root, row[mask_col])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        mask = load_mask_tensor(mask_path, target_hw=tuple(image_tensor.shape[-2:]))
        return image_tensor, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.table.iloc[idx]
        cc_image, cc_mask = self._load_image_mask("cc_image_path", "cc_mask_path", row)
        mlo_image, mlo_mask = self._load_image_mask("mlo_image_path", "mlo_mask_path", row)
        return {
            "cc_image": cc_image,
            "cc_mask": cc_mask,
            "cc_description": str(row["cc_description"]),
            "mlo_image": mlo_image,
            "mlo_mask": mlo_mask,
            "mlo_description": str(row["mlo_description"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "side_id": str(row["side_id"]),
        }

