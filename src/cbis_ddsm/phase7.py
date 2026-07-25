"""Configuration and train/validation-only gates for approved phase 7 work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .config import ConfigurationError
from .phase5 import verify_locked_training_manifests
from .phase7_model import Phase7ModelConfig
from .training import TrainingConfig


@dataclass(frozen=True)
class Phase7ExperimentConfig:
    experiment_id: str
    variant: str
    document: dict[str, Any]
    preprocessing: dict[str, Any]
    model: Phase7ModelConfig
    training: TrainingConfig
    source_path: Path


def _mapping(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing phase 7 configuration section: {name}")
    return value


def _validate_preprocessing(preprocessing: Mapping[str, Any]) -> None:
    if int(preprocessing.get("image_size", 0)) != 384:
        raise ConfigurationError("Phase 7 locks preprocessing.image_size to 384.")
    if preprocessing.get("spatial_mode") != "breast_crop_letterbox":
        raise ConfigurationError("Phase 7 requires breast_crop_letterbox preprocessing.")
    locked_values = {
        "breast_margin_fraction": 0.02,
        "min_foreground_fraction": 0.15,
        "max_foreground_fraction": 0.95,
        "min_breast_width_fraction": 0.25,
        "min_breast_height_fraction": 0.25,
        "train_horizontal_flip_probability": 0.0,
        "train_rotation_degrees": 0.0,
        "train_affine_probability": 0.0,
        "train_gamma_probability": 0.0,
        "train_contrast_probability": 0.0,
    }
    for name, expected in locked_values.items():
        observed = float(preprocessing.get(name, 0.0))
        if observed != expected:
            raise ConfigurationError(
                f"Phase 7 locks preprocessing.{name} to {expected}."
            )


def _validate_variant(variant: str, model: Phase7ModelConfig) -> None:
    expected = {
        "v2s": (False, 0, False),
        "v2s_bifpn": (True, 1, False),
        "v2s_cbam": (False, 0, True),
        "v2s_bifpn_cbam": (True, 1, True),
    }
    if variant not in expected:
        raise ConfigurationError(f"Unsupported phase 7 variant: {variant}")
    observed = (
        model.bifpn_enabled,
        model.bifpn_repeats,
        model.cbam_enabled,
    )
    if observed != expected[variant]:
        raise ConfigurationError(
            f"Variant {variant} requires module settings {expected[variant]}, got {observed}."
        )


def load_phase7_experiment(path: str | Path) -> Phase7ExperimentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Phase 7 configuration does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Phase 7 configuration root must be a mapping.")
    document = dict(loaded)
    experiment = _mapping(document, "experiment")
    experiment_id = str(experiment.get("id", "")).strip()
    variant = str(experiment.get("variant", "")).strip()
    if not experiment_id:
        raise ConfigurationError("experiment.id must not be empty.")
    if int(experiment.get("stage", 0)) != 7:
        raise ConfigurationError("Phase 7 entry points require experiment.stage=7.")

    preprocessing = dict(_mapping(document, "preprocessing"))
    _validate_preprocessing(preprocessing)
    model = Phase7ModelConfig.from_mapping(_mapping(document, "model"))
    _validate_variant(variant, model)
    training = TrainingConfig.from_document(document)
    if training.batch_size * training.gradient_accumulation_steps != 16:
        raise ConfigurationError("Phase 7 effective batch size must equal 16.")
    expected_output = f"artifacts/experiments/phase7/{experiment_id}"
    normalized_output = training.output_dir.replace("\\", "/").rstrip("/")
    if normalized_output != expected_output:
        raise ConfigurationError(
            f"Phase 7 output_dir must be the unique path {expected_output}."
        )
    return Phase7ExperimentConfig(
        experiment_id=experiment_id,
        variant=variant,
        document=document,
        preprocessing=preprocessing,
        model=model,
        training=training,
        source_path=source,
    )


def verify_phase7_manifests(
    manifest_dir: str | Path,
    lock_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Expose only the locked train and validation manifests."""
    train, val, hashes = verify_locked_training_manifests(manifest_dir, lock_path)
    if set(hashes) != {"train.csv", "val.csv"}:
        raise ConfigurationError("Phase 7 manifest gate may expose only train and validation.")
    if set(train["patient_id"]) & set(val["patient_id"]):
        raise ValueError("Phase 7 train and validation patients overlap.")
    return train, val, hashes
