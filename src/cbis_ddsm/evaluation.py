"""Leakage-safe threshold selection and final evaluation utilities."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader

from .config import ConfigurationError
from .dataset import MammogramDataset


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated stage 4 settings."""

    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    mixed_precision: bool = True
    deterministic: bool = True
    threshold_method: str = "youden_j"
    threshold_tie_break: str = "higher_threshold"
    expected_validation_samples: int = 241
    expected_test_samples: int = 284
    output_dir: str = "artifacts/evaluation/baseline_seed42"
    reference_threshold: float = 0.4515

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "EvaluationConfig":
        evaluation = document.get("evaluation")
        experiment = document.get("experiment")
        if not isinstance(evaluation, Mapping) or not isinstance(experiment, Mapping):
            raise ConfigurationError(
                "Evaluation configuration requires experiment and evaluation mappings."
            )
        values = dict(evaluation)
        values["reference_threshold"] = experiment.get("reference_threshold", 0.4515)
        known = set(cls.__dataclass_fields__)
        unexpected = sorted(set(values).difference(known))
        if unexpected:
            raise ConfigurationError(
                f"Unexpected evaluation configuration keys: {', '.join(unexpected)}"
            )
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigurationError("evaluation.seed must be non-negative.")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ConfigurationError("batch_size must be positive and num_workers non-negative.")
        if self.prefetch_factor <= 0:
            raise ConfigurationError("prefetch_factor must be positive.")
        if self.threshold_method != "youden_j":
            raise ConfigurationError("The locked threshold method is youden_j.")
        if self.threshold_tie_break != "higher_threshold":
            raise ConfigurationError("Youden J ties must use the higher threshold.")
        if self.expected_validation_samples <= 0 or self.expected_test_samples <= 0:
            raise ConfigurationError("Expected split sizes must be positive.")
        if not 0.0 <= float(self.reference_threshold) <= 1.0:
            raise ConfigurationError("reference_threshold must be in [0, 1].")
        if not self.output_dir:
            raise ConfigurationError("evaluation.output_dir must not be empty.")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    youden_j: float
    sensitivity: float
    specificity: float
    false_positive_rate: float
    candidate_count: int
    tie_count: int


@dataclass(frozen=True)
class EvaluationMetrics:
    samples: int
    positives: int
    negatives: int
    threshold: float
    roc_auc: float
    pr_auc: float
    sensitivity: float
    specificity: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass(frozen=True)
class InferenceOutput:
    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    labels: np.ndarray
    logits: np.ndarray


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Evaluation configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ConfigurationError("Evaluation configuration root must be a mapping.")
    return EvaluationConfig.from_document(document)


def _validated_binary_arrays(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape or targets.size == 0:
        raise ValueError("Labels and probabilities must be non-empty and have equal shape.")
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes.")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return targets, scores


def select_youden_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> ThresholdSelection:
    """Select validation threshold by maximum Youden J; exact ties choose higher."""
    targets, scores = _validated_binary_arrays(labels, probabilities)
    fpr, tpr, thresholds = roc_curve(targets, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("ROC curve produced no finite threshold candidates.")
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    youden = tpr - fpr
    best_j = float(np.max(youden))
    best_indices = np.flatnonzero(youden == best_j)
    selected = int(best_indices[np.argmax(thresholds[best_indices])])
    return ThresholdSelection(
        threshold=float(thresholds[selected]),
        youden_j=best_j,
        sensitivity=float(tpr[selected]),
        specificity=float(1.0 - fpr[selected]),
        false_positive_rate=float(fpr[selected]),
        candidate_count=int(thresholds.size),
        tie_count=int(best_indices.size),
    )


def compute_evaluation_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> EvaluationMetrics:
    targets, scores = _validated_binary_arrays(labels, probabilities)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be finite and in [0, 1].")
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return EvaluationMetrics(
        samples=int(targets.size),
        positives=int(targets.sum()),
        negatives=int(targets.size - targets.sum()),
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(targets, scores)),
        pr_auc=float(average_precision_score(targets, scores)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        accuracy=float(accuracy_score(targets, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(targets, predictions)),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        f1=float(f1_score(targets, predictions, zero_division=0)),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def ensure_new_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Evaluation output path is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def freeze_threshold(
    output_dir: str | Path,
    selection: ThresholdSelection,
    *,
    provenance: Mapping[str, Any],
    reference_threshold: float,
) -> Path:
    """Persist the validation-only decision before any test manifest may be read."""
    output = Path(output_dir).expanduser().resolve()
    threshold_path = output / "threshold.json"
    if threshold_path.exists():
        raise FileExistsError(f"Frozen threshold already exists: {threshold_path}")
    payload = {
        "format_version": 1,
        "frozen": True,
        "selection_split": "validation",
        "method": "youden_j",
        "tie_break": "higher_threshold",
        "selection": asdict(selection),
        "reference_threshold": float(reference_threshold),
        "reference_threshold_used_for_predictions": False,
        "test_split_loaded_at_freeze_time": False,
        "provenance": dict(provenance),
    }
    write_json(threshold_path, payload)
    return threshold_path


def read_frozen_threshold(path: str | Path) -> dict[str, Any]:
    threshold_path = Path(path).expanduser().resolve()
    if not threshold_path.is_file():
        raise PermissionError("Test loading is blocked until threshold.json exists.")
    with threshold_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format_version") != 1 or document.get("frozen") is not True:
        raise PermissionError("Test loading requires a valid frozen threshold record.")
    if document.get("selection_split") != "validation":
        raise PermissionError("The frozen threshold must have been selected on validation.")
    return document


def validate_split_manifest(
    frame: pd.DataFrame,
    *,
    expected_split: str,
    expected_samples: int,
) -> None:
    required = {"sample_id", "split", "patient_id", "label", "image_path"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{expected_split} manifest is missing: {', '.join(missing)}")
    if len(frame) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} {expected_split} samples, found {len(frame)}."
        )
    if set(frame["split"].astype(str).unique()) != {expected_split}:
        raise ValueError(f"Manifest is not exclusively split={expected_split}.")
    if set(pd.to_numeric(frame["label"], errors="raise").astype(int).unique()) != {0, 1}:
        raise ValueError(f"{expected_split} manifest must contain both classes.")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{expected_split} manifest has duplicate sample IDs.")


def load_test_manifest_after_threshold(
    data_output_dir: str | Path,
    threshold_path: str | Path,
    *,
    expected_samples: int,
) -> pd.DataFrame:
    """The only stage 4 test loader; threshold validation deliberately comes first."""
    read_frozen_threshold(threshold_path)
    test_path = Path(data_output_dir).expanduser().resolve() / "test.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"Test manifest does not exist: {test_path}")
    frame = pd.read_csv(test_path, dtype={"patient_id": str, "sample_id": str})
    validate_split_manifest(
        frame, expected_split="test", expected_samples=expected_samples
    )
    return frame


def validate_no_patient_or_sample_overlap(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    frames = {"train": train, "val": val, "test": test}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        patient_overlap = set(frames[left]["patient_id"].astype(str)).intersection(
            frames[right]["patient_id"].astype(str)
        )
        sample_overlap = set(frames[left]["sample_id"].astype(str)).intersection(
            frames[right]["sample_id"].astype(str)
        )
        if patient_overlap or sample_overlap:
            raise ValueError(
                f"Leakage between {left}/{right}: patients={len(patient_overlap)}, "
                f"samples={len(sample_overlap)}"
            )


def build_test_dataloader(
    manifest: pd.DataFrame,
    *,
    preprocessing: Mapping[str, Any],
    config: EvaluationConfig,
) -> DataLoader[dict[str, Any]]:
    dataset = MammogramDataset(
        manifest, split="test", preprocessing=dict(preprocessing)
    )
    options: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        options["persistent_workers"] = config.persistent_workers
        options["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **options)


def run_inference(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    mixed_precision: bool,
) -> InferenceOutput:
    model.to(device)
    model.eval()
    amp_enabled = mixed_precision and device.type == "cuda"
    sample_ids: list[str] = []
    patient_ids: list[str] = []
    image_paths: list[str] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                output = model(images).reshape(-1)
            values = output.detach().float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite test logits encountered.")
            logits.append(values)
            labels.append(batch["label"].detach().cpu().numpy().reshape(-1))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            patient_ids.extend(str(value) for value in batch["patient_id"])
            image_paths.extend(str(value) for value in batch["image_path"])
    if not labels:
        raise RuntimeError("Test data loader produced no samples.")
    return InferenceOutput(
        sample_ids=tuple(sample_ids),
        patient_ids=tuple(patient_ids),
        image_paths=tuple(image_paths),
        labels=np.concatenate(labels).astype(np.int64),
        logits=np.concatenate(logits).astype(np.float64),
    )


def save_evaluation_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: EvaluationMetrics,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics.roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {metrics.pr_auc:.4f}")
    ax.axhline(labels.mean(), linestyle="--", color="gray", label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "pr_curve.png", dpi=160)
    plt.close(fig)

    matrix = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
    ax.set(xlabel="Predicted", ylabel="True", title=f"Test confusion matrix (t={metrics.threshold:.4f})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
