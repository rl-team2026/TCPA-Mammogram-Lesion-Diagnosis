from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_feature_npz(
    output_path: str | Path,
    image_ids: list[str],
    features: np.ndarray,
    feature_name: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray(image_ids, dtype=str),
        features=np.asarray(features, dtype=np.float32),
        feature_name=np.asarray(feature_name),
    )


def load_feature_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].astype(str)
    features = data["features"].astype(np.float32)
    if len(ids) != len(features):
        raise ValueError(f"Feature file has mismatched ids/features: {path}")
    return ids, features


def align_features(
    manifest: pd.DataFrame,
    feature_paths: dict[str, str | Path],
    image_id_col: str = "image_id",
) -> dict[str, np.ndarray]:
    """Align multiple feature matrices to manifest row order."""
    row_ids = manifest[image_id_col].astype(str).tolist()
    aligned: dict[str, np.ndarray] = {}
    for name, path in feature_paths.items():
        ids, feats = load_feature_npz(path)
        index = {image_id: i for i, image_id in enumerate(ids)}
        missing = [image_id for image_id in row_ids if image_id not in index]
        if missing:
            raise ValueError(
                f"{name} is missing {len(missing)} manifest ids. First missing: {missing[:5]}"
            )
        aligned[name] = np.stack([feats[index[image_id]] for image_id in row_ids]).astype(
            np.float32
        )
    return aligned


def parse_feature_args(items: list[str]) -> dict[str, Path]:
    """Parse CLI feature arguments in name=/path/file.npz format."""
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Feature argument must be name=path, got: {item}")
        name, path = item.split("=", 1)
        parsed[name.strip()] = Path(path)
    return parsed

