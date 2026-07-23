"""Configuration and manifest gates shared by phase 5 entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .config import ConfigurationError
from .model import ModelConfig
from .training import (
    TrainingConfig,
    load_training_manifests,
    sha256_file,
)


@dataclass(frozen=True)
class Phase5ExperimentConfig:
    experiment_id: str
    document: dict[str, Any]
    preprocessing: dict[str, Any]
    model: ModelConfig
    training: TrainingConfig
    source_path: Path


def _mapping(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing phase 5 configuration section: {name}")
    return value


def _validate_safe_augmentation(preprocessing: Mapping[str, Any]) -> None:
    for name in (
        "train_affine_probability",
        "train_gamma_probability",
        "train_contrast_probability",
    ):
        value = float(preprocessing.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError(f"{name} must be in [0, 1].")

    if float(preprocessing.get("train_horizontal_flip_probability", 0.0)) != 0.0:
        raise ConfigurationError("Phase 5 safe augmentation does not include horizontal flips.")
    if float(preprocessing.get("train_rotation_degrees", 0.0)) != 0.0:
        raise ConfigurationError("Use the approved combined affine transform in phase 5.")

    degrees = float(preprocessing.get("train_affine_degrees", 0.0))
    translation = float(preprocessing.get("train_affine_translate_fraction", 0.0))
    scale_min = float(preprocessing.get("train_affine_scale_min", 1.0))
    scale_max = float(preprocessing.get("train_affine_scale_max", 1.0))
    gamma_min = float(preprocessing.get("train_gamma_min", 1.0))
    gamma_max = float(preprocessing.get("train_gamma_max", 1.0))
    contrast_min = float(preprocessing.get("train_contrast_min", 1.0))
    contrast_max = float(preprocessing.get("train_contrast_max", 1.0))
    if not 0.0 <= degrees <= 5.0:
        raise ConfigurationError("train_affine_degrees must be in [0, 5].")
    if not 0.0 <= translation <= 0.03:
        raise ConfigurationError("train_affine_translate_fraction must be in [0, 0.03].")
    if not 0.95 <= scale_min <= scale_max <= 1.05:
        raise ConfigurationError("Approved affine scale range is within [0.95, 1.05].")
    if not 0.90 <= gamma_min <= gamma_max <= 1.10:
        raise ConfigurationError("Approved gamma range is within [0.90, 1.10].")
    if not 0.90 <= contrast_min <= contrast_max <= 1.10:
        raise ConfigurationError("Approved contrast range is within [0.90, 1.10].")


def load_phase5_experiment(path: str | Path) -> Phase5ExperimentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Phase 5 configuration does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Phase 5 configuration root must be a mapping.")
    document = dict(loaded)
    experiment = _mapping(document, "experiment")
    experiment_id = str(experiment.get("id", "")).strip()
    if not experiment_id:
        raise ConfigurationError("experiment.id must not be empty.")
    if int(experiment.get("stage", 0)) != 5:
        raise ConfigurationError("Phase 5 entry points require experiment.stage=5.")

    preprocessing = dict(_mapping(document, "preprocessing"))
    if preprocessing.get("spatial_mode") != "breast_crop_letterbox":
        raise ConfigurationError("Phase 5 requires breast_crop_letterbox preprocessing.")
    _validate_safe_augmentation(preprocessing)
    image_size = int(preprocessing.get("image_size", 0))
    model = ModelConfig.from_mapping(_mapping(document, "model"))
    training = TrainingConfig.from_document(document)
    if image_size != model.input_size:
        raise ConfigurationError("Preprocessing and model input sizes must match.")
    if training.batch_size * training.gradient_accumulation_steps != 16:
        raise ConfigurationError("Phase 5 effective batch size must equal 16.")
    expected_prefix = f"artifacts/experiments/phase5/{experiment_id}"
    normalized_output = training.output_dir.replace("\\", "/").rstrip("/")
    if normalized_output != expected_prefix:
        raise ConfigurationError(
            f"Phase 5 output_dir must be the unique path {expected_prefix}."
        )
    return Phase5ExperimentConfig(
        experiment_id=experiment_id,
        document=document,
        preprocessing=preprocessing,
        model=model,
        training=training,
        source_path=source,
    )


def verify_locked_training_manifests(
    manifest_dir: str | Path,
    lock_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load only train/validation and enforce their approved phase 4 hashes."""
    root = Path(manifest_dir).expanduser().resolve()
    lock = Path(lock_path).expanduser().resolve()
    with lock.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    expected = document.get("manifests") if isinstance(document, Mapping) else None
    if not isinstance(expected, Mapping) or set(expected) != {"train.csv", "val.csv"}:
        raise ConfigurationError("Manifest lock must contain only train.csv and val.csv.")
    paths = {name: root / name for name in ("train.csv", "val.csv")}
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != {name: str(value) for name, value in expected.items()}:
        raise ValueError(
            f"Phase 5 manifest hashes do not match the lock: {observed}"
        )
    train, val = load_training_manifests(root)
    return train, val, observed
