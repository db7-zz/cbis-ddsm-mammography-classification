"""Configuration loading and validation for phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the phase 1 configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigurationError("The configuration root must be a mapping.")
    for section in ("experiment", "data", "preprocessing"):
        if not isinstance(config.get(section), dict):
            raise ConfigurationError(f"Missing configuration section: {section}")
    return config


def apply_path_overrides(
    config: dict[str, Any],
    *,
    data_root: str | None = None,
    train_csv: str | None = None,
    test_csv: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a detached config with optional CLI path overrides applied."""
    copied = json.loads(json.dumps(config))
    overrides = {
        "data_root": data_root,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "output_dir": output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            copied["data"][key] = value
    return copied


def validate_data_config(config: dict[str, Any]) -> None:
    data = config["data"]
    missing = [key for key in ("data_root", "train_csv", "test_csv") if not data.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(
            f"Data paths are not configured ({joined}). Edit configs/data.yaml "
            "or pass --data-root, --train-csv, and --test-csv."
        )

    val_fraction = float(data.get("val_fraction", 0.2))
    if not 0.0 < val_fraction < 1.0:
        raise ConfigurationError("data.val_fraction must be strictly between 0 and 1.")

    preprocessing = config["preprocessing"]
    if int(preprocessing.get("image_size", 0)) <= 0:
        raise ConfigurationError("preprocessing.image_size must be positive.")
    lower = float(preprocessing.get("lower_percentile", 0.5))
    upper = float(preprocessing.get("upper_percentile", 99.5))
    if not 0.0 <= lower < upper <= 100.0:
        raise ConfigurationError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    spatial_mode = str(preprocessing.get("spatial_mode", "stretch"))
    if spatial_mode not in {"stretch", "breast_crop_letterbox"}:
        raise ConfigurationError(
            "preprocessing.spatial_mode must be stretch or breast_crop_letterbox."
        )
    if spatial_mode == "breast_crop_letterbox":
        margin = float(preprocessing.get("breast_margin_fraction", 0.02))
        minimum = float(preprocessing.get("min_foreground_fraction", 0.15))
        maximum = float(preprocessing.get("max_foreground_fraction", 0.95))
        min_width = float(preprocessing.get("min_breast_width_fraction", 0.25))
        min_height = float(preprocessing.get("min_breast_height_fraction", 0.25))
        if not 0.0 <= margin < 0.5:
            raise ConfigurationError("breast_margin_fraction must be in [0, 0.5).")
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ConfigurationError(
                "Foreground fractions must satisfy 0 <= minimum < maximum <= 1."
            )
        if not 0.0 < min_width <= 1.0 or not 0.0 < min_height <= 1.0:
            raise ConfigurationError(
                "Minimum breast dimension fractions must be in (0, 1]."
            )
