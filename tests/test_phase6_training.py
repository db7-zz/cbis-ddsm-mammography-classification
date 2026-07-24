from __future__ import annotations

from pathlib import Path

from cbis_ddsm.phase6 import load_phase6_experiment, verify_phase6_manifests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE6_CONFIGS = PROJECT_ROOT / "configs" / "phase6"


def test_phase6_screening_configs_are_locked() -> None:
    expected = {
        "P6-E01A_BiFPN_R1_C128_seed42.yaml": ("bifpn_r1", True, 1, False),
        "P6-E01B_BiFPN_R2_C128_seed42.yaml": ("bifpn_r2", True, 2, False),
        "P6-E02_CBAM_seed42.yaml": ("cbam", False, 0, True),
    }
    for name, settings in expected.items():
        experiment = load_phase6_experiment(PHASE6_CONFIGS / name)
        variant, bifpn, repeats, cbam = settings
        assert experiment.variant == variant
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
            f"artifacts/experiments/phase6/{experiment.experiment_id}"
        )


def test_phase6_configs_disable_all_random_augmentation() -> None:
    for path in PHASE6_CONFIGS.glob("P6-*.yaml"):
        preprocessing = load_phase6_experiment(path).preprocessing
        assert preprocessing["train_horizontal_flip_probability"] == 0.0
        assert preprocessing["train_rotation_degrees"] == 0.0
        assert preprocessing["train_affine_probability"] == 0.0
        assert preprocessing["train_gamma_probability"] == 0.0
        assert preprocessing["train_contrast_probability"] == 0.0


def test_phase6_manifest_gate_exposes_only_train_and_validation() -> None:
    train, val, hashes = verify_phase6_manifests(
        PROJECT_ROOT / "artifacts" / "data",
        PHASE6_CONFIGS / "manifest_lock.yaml",
    )
    assert len(train) == 986
    assert len(val) == 241
    assert set(hashes) == {"train.csv", "val.csv"}
    assert set(train["patient_id"]).isdisjoint(set(val["patient_id"]))
