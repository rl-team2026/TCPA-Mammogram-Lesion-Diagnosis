from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self, features: dict[str, np.ndarray], labels: np.ndarray, indices: np.ndarray):
        self.features = {name: values[indices].astype(np.float32) for name, values in features.items()}
        self.labels = labels[indices].astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "features": {name: torch.from_numpy(values[idx]) for name, values in self.features.items()},
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float


def _move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in features.items()}


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        features = _move_features(batch["features"], device)
        labels = batch["label"].to(device)
        logits = model(features)
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def train_binary_classifier(
    model: nn.Module,
    train_dataset: FeatureDataset,
    val_dataset: FeatureDataset,
    device: torch.device,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    patience: int = 8,
    num_workers: int = 0,
) -> TrainResult:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    labels = train_dataset.labels
    pos = max(float((labels == 1).sum()), 1.0)
    neg = max(float((labels == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = _move_features(batch["features"], device)
            labels_t = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features = _move_features(batch["features"], device)
                labels_t = batch["label"].to(device)
                logits = model(features)
                loss = criterion(logits, labels_t)
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_epoch=best_epoch, best_val_loss=best_loss)

