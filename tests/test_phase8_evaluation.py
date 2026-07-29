from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase8 import (
    PHASE8_MODEL_IDS,
    PHASE8_OFFICIAL_TEST_MODEL_IDS,
    ensure_phase8_official_test_outputs_absent,
    freeze_phase8_thresholds,
    load_phase8_artifact_lock,
    load_phase8_official_test_policy,
    load_phase8_test_manifest,
    paired_patient_bootstrap,
    read_phase8_validation_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE8_CONFIGS = PROJECT_ROOT / "configs" / "phase8"


def test_phase8_lock_contains_only_the_three_frozen_seed42_models() -> None:
    lock = load_phase8_artifact_lock(PHASE8_CONFIGS / "artifact_lock.yaml")
    assert tuple(lock.models) == PHASE8_MODEL_IDS
    assert lock.models["P5-E02_seed42"].official_test_allowed is False
    assert tuple(
        model_id
        for model_id, artifact in lock.models.items()
        if artifact.official_test_allowed
    ) == PHASE8_OFFICIAL_TEST_MODEL_IDS
    assert all(artifact.seed == 42 for artifact in lock.models.values())


def test_phase8_policy_locks_two_test_passes_and_no_training() -> None:
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    assert policy.authorized_model_ids == PHASE8_OFFICIAL_TEST_MODEL_IDS
    assert policy.project_test_inference_passes_before_phase8 == 1
    assert policy.maximum_new_test_inference_passes == 2
    assert policy.project_test_inference_passes_after_success == 3
    assert all(policy.forbidden.values())


def test_phase8_policy_rejects_an_additional_test_model(tmp_path: Path) -> None:
    source = PHASE8_CONFIGS / "official_test.yaml"
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "  - P7-E04_V2S_BiFPN_CBAM_seed42",
        "  - P7-E04_V2S_BiFPN_CBAM_seed42\n  - P7-E02_V2S_BiFPN_R1_C128_seed42",
        1,
    )
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Only P7-E01 and P7-E04"):
        load_phase8_official_test_policy(path)


def test_phase8_validation_evidence_matches_locked_predictions() -> None:
    lock = load_phase8_artifact_lock(PHASE8_CONFIGS / "artifact_lock.yaml")
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    expected = {
        "P7-E01_V2S_seed42": (0.717926689576174, 0.6056954281153017),
        "P7-E04_V2S_BiFPN_CBAM_seed42": (
            0.6679553264604811,
            0.5554048312928487,
        ),
    }
    for model_id, metrics in expected.items():
        evidence = read_phase8_validation_evidence(
            PROJECT_ROOT, lock, policy, model_id
        )
        assert evidence.samples == 241
        assert evidence.roc_auc == pytest.approx(metrics[0], abs=1e-12)
        assert evidence.pr_auc == pytest.approx(metrics[1], abs=1e-12)


def test_phase8_thresholds_freeze_without_test_access(tmp_path: Path) -> None:
    lock = load_phase8_artifact_lock(PHASE8_CONFIGS / "artifact_lock.yaml")
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    policy = policy.__class__(
        **{
            **policy.__dict__,
            "threshold_output_dirs": {
                model_id: str(tmp_path / model_id)
                for model_id in policy.authorized_model_ids
            },
        }
    )
    results = freeze_phase8_thresholds(PROJECT_ROOT, lock, policy)
    assert tuple(results) == PHASE8_OFFICIAL_TEST_MODEL_IDS
    for model_id, result in results.items():
        document = json.loads(
            Path(result["threshold_path"]).read_text(encoding="utf-8")
        )
        assert document["test_split_loaded_at_freeze_time"] is False
        assert document["provenance"]["model_id"] == model_id
        assert document["provenance"]["official_test_loaded"] is False
    with pytest.raises(FileExistsError):
        freeze_phase8_thresholds(PROJECT_ROOT, lock, policy)


def test_phase8_test_loader_requires_separate_authorization(tmp_path: Path) -> None:
    lock = load_phase8_artifact_lock(PHASE8_CONFIGS / "artifact_lock.yaml")
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    with pytest.raises(PermissionError, match="blocked"):
        load_phase8_test_manifest(
            PROJECT_ROOT,
            lock,
            policy,
            authorization_path=tmp_path / "missing.json",
        )


def test_phase8_test_loader_requires_both_matching_thresholds(tmp_path: Path) -> None:
    lock = load_phase8_artifact_lock(PHASE8_CONFIGS / "artifact_lock.yaml")
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    policy = policy.__class__(
        **{
            **policy.__dict__,
            "threshold_output_dirs": {
                model_id: str(tmp_path / "thresholds" / model_id)
                for model_id in policy.authorized_model_ids
            },
        }
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": 8,
                "official_test_reuse_authorized": True,
                "approved_by_user": True,
                "authorized_model_ids": list(policy.authorized_model_ids),
                "maximum_new_inference_passes": 2,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="threshold"):
        load_phase8_test_manifest(
            PROJECT_ROOT,
            lock,
            policy,
            authorization_path=authorization,
        )


def test_phase8_validation_manifest_still_has_patient_ids() -> None:
    frame = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "data" / "val.csv",
        dtype={"patient_id": str},
    )
    assert frame["patient_id"].nunique() == 120


def test_phase8_policy_locks_official_test_manifest_hash() -> None:
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    assert (
        policy.expected_test_manifest_sha256
        == "ad631667905ec737f6d60efffd608eb6aaf280f3d2bc85864ed43da1b56cf8a0"
    )


def test_phase8_official_test_rejects_any_existing_output_path(
    tmp_path: Path,
) -> None:
    policy = load_phase8_official_test_policy(
        PHASE8_CONFIGS / "official_test.yaml"
    )
    outputs = {
        model_id: f"official_test/{model_id}"
        for model_id in policy.authorized_model_ids
    }
    policy = replace(policy, official_test_output_dirs=outputs)
    ensure_phase8_official_test_outputs_absent(tmp_path, policy)
    (tmp_path / outputs[policy.authorized_model_ids[0]]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing output path"):
        ensure_phase8_official_test_outputs_absent(tmp_path, policy)


def test_paired_patient_bootstrap_is_deterministic_and_patient_clustered() -> None:
    first = pd.DataFrame(
        {
            "sample_id": ["a1", "a2", "b1", "c1", "d1", "d2"],
            "patient_id": ["a", "a", "b", "c", "d", "d"],
            "label": [0, 0, 0, 1, 1, 1],
            "probability": [0.1, 0.2, 0.4, 0.9, 0.8, 0.7],
        }
    )
    second = first.copy()
    second["probability"] = [0.2, 0.3, 0.6, 0.8, 0.7, 0.4]
    arguments = {
        "first_model_id": "first",
        "second_model_id": "second",
        "valid_replicates": 100,
        "seed": 42,
        "confidence_level": 0.95,
    }
    result = paired_patient_bootstrap(first, second, **arguments)
    repeated = paired_patient_bootstrap(first, second, **arguments)
    assert result == repeated
    assert result["unit"] == "patient_id"
    assert result["patients"] == 4
    assert result["valid_replicates"] == 100
    assert result["roc_auc"]["difference"] > 0.0
    assert result["pr_auc"]["difference"] > 0.0


def test_paired_patient_bootstrap_rejects_misaligned_samples() -> None:
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "patient_id": ["p1", "p2"],
            "label": [0, 1],
            "probability": [0.1, 0.9],
        }
    )
    second = first.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sample order"):
        paired_patient_bootstrap(
            first,
            second,
            first_model_id="first",
            second_model_id="second",
            valid_replicates=10,
            seed=42,
            confidence_level=0.95,
        )
