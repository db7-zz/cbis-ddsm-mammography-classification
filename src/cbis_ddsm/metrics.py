"""Threshold-free binary metrics used for validation during phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    """Validation metrics that do not require selecting a decision threshold."""

    roc_auc: float
    pr_auc: float


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for one-dimensional NumPy logits."""
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    probabilities[~positive] = negative_exp / (1.0 + negative_exp)
    return probabilities


def compute_binary_metrics(labels: np.ndarray, logits: np.ndarray) -> BinaryMetrics:
    """Compute ROC AUC and average precision from labels and raw logits."""
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape:
        raise ValueError("Labels and logits must have the same shape.")
    if targets.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set.")
    unique = set(np.unique(targets).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Binary labels must be 0 or 1, got {sorted(unique)}.")
    if unique != {0, 1}:
        raise ValueError("ROC AUC requires both classes in the evaluated split.")
    probabilities = sigmoid_numpy(scores)
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(targets, probabilities)),
        pr_auc=float(average_precision_score(targets, probabilities)),
    )
