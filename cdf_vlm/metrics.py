from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def threshold_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> ThresholdMetrics:
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return ThresholdMetrics(
        threshold=threshold,
        sensitivity=tp / max(tp + fn, 1),
        specificity=tn / max(tn + fp, 1),
        ppv=tp / max(tp + fp, 1),
        npv=tn / max(tn + fn, 1),
    )


def fixed_specificity_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    target_specificity: float = 0.90,
) -> ThresholdMetrics:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    candidate_thresholds = np.unique(np.r_[0.0, probs, 1.0])
    best = None
    for threshold in candidate_thresholds:
        tm = threshold_metrics(labels, probs, float(threshold))
        if tm.specificity >= target_specificity:
            if best is None or tm.sensitivity > best.sensitivity:
                best = tm
    return best or threshold_metrics(labels, probs, 1.0)


def fixed_sensitivity_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    target_sensitivity: float = 0.90,
) -> ThresholdMetrics:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    candidate_thresholds = np.unique(np.r_[0.0, probs, 1.0])
    best = None
    for threshold in candidate_thresholds:
        tm = threshold_metrics(labels, probs, float(threshold))
        if tm.sensitivity >= target_sensitivity:
            if best is None or tm.specificity > best.specificity:
                best = tm
    return best or threshold_metrics(labels, probs, 0.0)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": expected_calibration_error(labels, probs),
    }
    if len(np.unique(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, probs))
        out["auprc"] = float(average_precision_score(labels, probs))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")

    tm = threshold_metrics(labels, probs, threshold)
    out.update(
        {
            "threshold": tm.threshold,
            "sensitivity": tm.sensitivity,
            "specificity": tm.specificity,
            "ppv": tm.ppv,
            "npv": tm.npv,
        }
    )
    spec90 = fixed_specificity_threshold(labels, probs, 0.90)
    sens90 = fixed_sensitivity_threshold(labels, probs, 0.90)
    out["sensitivity_at_spec90"] = spec90.sensitivity
    out["threshold_at_spec90"] = spec90.threshold
    out["specificity_at_sens90"] = sens90.specificity
    out["threshold_at_sens90"] = sens90.threshold
    return out


def bootstrap_metric_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_name: str = "auroc",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    values = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        values.append(binary_metrics(labels[idx], probs[idx])[metric_name])
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

