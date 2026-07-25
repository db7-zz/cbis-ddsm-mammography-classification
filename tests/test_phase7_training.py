from __future__ import annotations

from pathlib import Path

import pytest

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase7 import load_phase7_experiment, verify_phase7_manifests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE7_CONFIGS = PROJECT_ROOT / "configs" / "phase7"


def test_phase7_seed42_configs_match_the_approved_matrix() -> None:
    expected = {
        "P7-E01_V2S_seed42.yaml": ("v2s", False, 0, False),
        "P7-E02_V2S_BiFPN_R1_C128_seed42.yaml": (
            "v2s_bifpn",
            True,
            1,
            False,
        ),
        "P7-E03_V2S_CBAM_seed42.yaml": (
            "v2s_cbam",
            False,
            0,
            True,
        ),
        "P7-E04_V2S_BiFPN_CBAM_seed42.yaml": (
            "v2s_bifpn_cbam",
            True,
            1,
            True,
        ),
    }
    for name, settings in expected.items():
        experiment = load_phase7_experiment(PHASE7_CONFIGS / name)
        variant, bifpn, repeats, cbam = settings
        assert experiment.variant == variant
        assert experiment.model.backbone == "efficientnet_v2_s"
        assert experiment.model.bifpn_enabled is bifpn
        assert experiment.model.bifpn_repeats == repeats
        assert experiment.model.cbam_enabled is cbam
        assert experiment.model.input_size == 384
        assert experiment.training.seed == 42
        assert experiment.training.batch_size == 8
        assert experiment.training.gradient_accumulation_steps == 2
        assert (
            experiment.training.batch_size
            * experiment.training.gradient_accumulation_steps
            == 16
        )
        assert experiment.training.output_dir == (
            f"artifacts/experiments/phase7/{experiment.experiment_id}"
        )


def test_phase7_configs_disable_all_random_augmentation() -> None:
    for path in PHASE7_CONFIGS.glob("P7-*.yaml"):
        preprocessing = load_phase7_experiment(path).preprocessing
        assert preprocessing["train_horizontal_flip_probability"] == 0.0
        assert preprocessing["train_rotation_degrees"] == 0.0
        assert preprocessing["train_affine_probability"] == 0.0
        assert preprocessing["train_gamma_probability"] == 0.0
        assert preprocessing["train_contrast_probability"] == 0.0


def test_phase7_manifest_gate_exposes_only_train_and_validation() -> None:
    train, val, hashes = verify_phase7_manifests(
        PROJECT_ROOT / "artifacts" / "data",
        PHASE7_CONFIGS / "manifest_lock.yaml",
    )
    assert len(train) == 986
    assert len(val) == 241
    assert set(hashes) == {"train.csv", "val.csv"}
    assert set(train["patient_id"]).isdisjoint(set(val["patient_id"]))


def test_phase7_rejects_variant_and_module_mismatch(tmp_path: Path) -> None:
    source = PHASE7_CONFIGS / "P7-E01_V2S_seed42.yaml"
    text = source.read_text(encoding="utf-8")
    text = text.replace("cbam_enabled: false", "cbam_enabled: true", 1)
    path = tmp_path / "mismatch.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_phase7_experiment(path)
