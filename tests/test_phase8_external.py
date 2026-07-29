from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.phase8_external import (
    audit_inbreast_xls,
    candidate_birads_label,
    deterministic_external_sample_id,
    load_inbreast_external_policy,
    patient_cluster_metric_bootstrap,
    parse_inbreast_dicom_name,
    validate_inbreast_authorization,
    validate_inbreast_route_authorization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "configs" / "phase8" / "inbreast_external.yaml"


def test_inbreast_policy_is_audit_first_and_freezes_two_models() -> None:
    policy = load_inbreast_external_policy(POLICY_PATH)
    assert policy.mapping_status == (
        "exploratory_only_requires_post_audit_user_confirmation"
    )
    assert policy.authorized_model_ids == (
        "P7-E01_V2S_seed42",
        "P7-E04_V2S_BiFPN_CBAM_seed42",
    )
    assert all(policy.forbidden.values())
    assert policy.audit_lock["expected_candidate_images"] == 314
    assert policy.runtime["batch_size"] == 8


def test_inbreast_dicom_filename_recovers_patient_and_view() -> None:
    cc = parse_inbreast_dicom_name(
        "20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm"
    )
    mlo = parse_inbreast_dicom_name(
        "20586960_6c613a14b80a8591_MG_L_ML_ANON.dcm"
    )
    assert cc == {
        "file_id": "20586908",
        "patient_id": "6c613a14b80a8591",
        "laterality": "R",
        "view": "CC",
    }
    assert mlo["view"] == "MLO"


@pytest.mark.parametrize(
    ("birads", "expected"),
    [("1", 0), ("2", 0), ("3", 0), ("4a", 1), ("4C", 1), ("5", 1), ("6", 1)],
)
def test_birads_mapping_is_explicitly_candidate_only(
    birads: str,
    expected: int,
) -> None:
    policy = load_inbreast_external_policy(POLICY_PATH)
    assert candidate_birads_label(birads, policy) == expected


def test_birads_mapping_rejects_unknown_values() -> None:
    policy = load_inbreast_external_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="Unsupported"):
        candidate_birads_label("4", policy)


def test_external_sample_id_is_deterministic() -> None:
    values = {
        "file_id": "20586908",
        "patient_id": "6c613a14b80a8591",
        "laterality": "R",
        "view": "CC",
    }
    assert deterministic_external_sample_id(**values) == deterministic_external_sample_id(
        **values
    )


def test_gate_d_authorization_locks_dataset_and_requires_route_confirmation(
    tmp_path: Path,
) -> None:
    policy = load_inbreast_external_policy(POLICY_PATH)
    root = tmp_path / "inbreast"
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": 8,
                "gate": "D",
                "approved_by_user": True,
                "data_access_authorized": True,
                "full_gate_d_workflow_requested": True,
                "dataset_root": str(root),
                "authorized_model_ids": list(policy.authorized_model_ids),
                "post_audit_label_route": "requires_user_confirmation",
                "raw_data_mutation_authorized": False,
                "dependency_installation_authorized": False,
                "commit_or_push_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    validated = validate_inbreast_authorization(
        authorization,
        dataset_root=root,
        policy=policy,
    )
    assert validated["approved_by_user"] is True
    with pytest.raises(PermissionError, match="approved scope"):
        validate_inbreast_authorization(
            authorization,
            dataset_root=tmp_path / "other",
            policy=policy,
        )


def test_policy_rejects_a_preselected_birads_route(tmp_path: Path) -> None:
    text = POLICY_PATH.read_text(encoding="utf-8").replace(
        "exploratory_only_requires_post_audit_user_confirmation",
        "exploratory_birads_transfer",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must remain unapproved"):
        load_inbreast_external_policy(changed)


def test_xls_audit_helper_exists() -> None:
    helper = PROJECT_ROOT / "scripts" / "audit_inbreast_xls.ps1"
    assert helper.is_file()
    assert callable(audit_inbreast_xls)


def test_gate_d_route1_authorization_locks_exploratory_claims(
    tmp_path: Path,
) -> None:
    policy = load_inbreast_external_policy(POLICY_PATH)
    root = tmp_path / "inbreast"
    authorization = tmp_path / "route.json"
    authorization.write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": 8,
                "gate": "D",
                "approved_by_user": True,
                "selected_route": "exploratory_birads_transfer",
                "research_use_authorized_for_this_run": True,
                "dataset_root": str(root),
                "label_equivalence_claimed": False,
                "mapping": {
                    "negative": list(policy.negative_birads),
                    "positive": list(policy.positive_birads),
                },
                "authorized_model_ids": list(policy.authorized_model_ids),
                "maximum_external_inference_passes": 2,
                "training_authorized": False,
                "threshold_refit_authorized": False,
                "official_test_reload_authorized": False,
                "commit_or_push_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    validated = validate_inbreast_route_authorization(
        authorization,
        dataset_root=root,
        policy=policy,
    )
    assert validated["label_equivalence_claimed"] is False


def test_patient_cluster_metric_bootstrap_is_deterministic() -> None:
    predictions = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "c", "d", "d"],
            "label": [0, 0, 0, 1, 1, 1],
            "probability": [0.1, 0.2, 0.4, 0.9, 0.8, 0.7],
        }
    )
    arguments = {
        "threshold": 0.5,
        "valid_replicates": 100,
        "seed": 42,
        "confidence_level": 0.95,
    }
    first = patient_cluster_metric_bootstrap(predictions, **arguments)
    second = patient_cluster_metric_bootstrap(predictions, **arguments)
    assert first == second
    assert first["patients"] == 4
    assert first["metrics"]["roc_auc"]["point"] == pytest.approx(1.0)
