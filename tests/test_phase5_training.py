from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.model import ModelConfig
from cbis_ddsm.phase5 import (
    load_phase5_experiment,
    verify_locked_training_manifests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_CONFIGS = PROJECT_ROOT / "configs" / "phase5"


@pytest.mark.parametrize(
    ("name", "size", "batch_size", "accumulation"),
    [
        ("P5-E01_seed42.yaml", 256, 16, 1),
        ("P5-E02_seed42.yaml", 384, 8, 2),
        ("P5-E03_seed42.yaml", 512, 4, 4),
        ("P5-E04_seed42.yaml", 384, 8, 2),
        ("P5-E02_seed2026.yaml", 384, 8, 2),
        ("P5-E02_seed3407.yaml", 384, 8, 2),
    ],
)
def test_phase5_experiment_matrix_is_locked_and_valid(
    name: str,
    size: int,
    batch_size: int,
    accumulation: int,
) -> None:
    experiment = load_phase5_experiment(PHASE5_CONFIGS / name)

    assert experiment.model.input_size == size
    assert experiment.preprocessing["image_size"] == size
    assert experiment.training.batch_size == batch_size
    assert experiment.training.gradient_accumulation_steps == accumulation
    assert batch_size * accumulation == 16
    assert experiment.model.backbone == "efficientnet_b0"
    assert experiment.model.projection_dim == 512
    assert experiment.model.transformer_layers == 4
    assert experiment.model.attention_heads == 8
    assert experiment.model.feedforward_dim == 2048


def test_p5_e04_uses_only_the_approved_safe_augmentation() -> None:
    experiment = load_phase5_experiment(PHASE5_CONFIGS / "P5-E04_seed42.yaml")
    preprocessing = experiment.preprocessing

    assert preprocessing["train_horizontal_flip_probability"] == 0.0
    assert preprocessing["train_rotation_degrees"] == 0.0
    assert preprocessing["train_affine_probability"] == 0.5
    assert preprocessing["train_affine_degrees"] == 5.0
    assert preprocessing["train_affine_translate_fraction"] == 0.03
    assert preprocessing["train_affine_scale_min"] == 0.95
    assert preprocessing["train_affine_scale_max"] == 1.05
    assert preprocessing["train_gamma_probability"] == 0.3
    assert preprocessing["train_gamma_min"] == 0.90
    assert preprocessing["train_gamma_max"] == 1.10
    assert preprocessing["train_contrast_probability"] == 0.3
    assert preprocessing["train_contrast_min"] == 0.90
    assert preprocessing["train_contrast_max"] == 1.10


def test_gate_d_configs_only_change_seed_and_unique_output() -> None:
    reference = load_phase5_experiment(PHASE5_CONFIGS / "P5-E02_seed42.yaml")
    for seed in (2026, 3407):
        candidate = load_phase5_experiment(
            PHASE5_CONFIGS / f"P5-E02_seed{seed}.yaml"
        )
        assert candidate.preprocessing == reference.preprocessing
        assert candidate.model == reference.model
        assert candidate.training.seed == seed
        assert candidate.training.output_dir == (
            f"artifacts/experiments/phase5/P5-E02_seed{seed}"
        )
        reference_training = {
            key: value
            for key, value in reference.document["training"].items()
            if key not in {"seed", "output_dir"}
        }
        candidate_training = {
            key: value
            for key, value in candidate.document["training"].items()
            if key not in {"seed", "output_dir"}
        }
        assert candidate_training == reference_training


def test_only_approved_phase5_input_sizes_are_allowed() -> None:
    for size in (256, 384, 512):
        replace(ModelConfig(), input_size=size).validate()
    with pytest.raises(ConfigurationError):
        replace(ModelConfig(), input_size=768).validate()


def test_phase5_manifest_gate_loads_only_locked_train_and_validation() -> None:
    train, val, hashes = verify_locked_training_manifests(
        PROJECT_ROOT / "artifacts" / "data",
        PHASE5_CONFIGS / "manifest_lock.yaml",
    )

    assert len(train) == 986
    assert len(val) == 241
    assert set(hashes) == {"train.csv", "val.csv"}
    assert set(train["patient_id"]).isdisjoint(set(val["patient_id"]))
