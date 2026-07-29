"""Artifact, threshold, and official-test gates for phase 8."""

from __future__ import annotations

import gc
import json
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .config import ConfigurationError
from .evaluation import (
    ensure_new_output_directory,
    freeze_threshold,
    load_test_manifest_after_threshold,
    read_frozen_threshold,
    select_youden_threshold,
)
from .model import build_model
from .phase5 import load_phase5_experiment
from .phase7 import load_phase7_experiment
from .phase7_model import build_phase7_model
from .training import sha256_file


PHASE8_SOURCE_COMMIT = "a534698a68de836dc89340c5c52bd535289aca7e"
PHASE8_MODEL_IDS = (
    "P5-E02_seed42",
    "P7-E01_V2S_seed42",
    "P7-E04_V2S_BiFPN_CBAM_seed42",
)
PHASE8_OFFICIAL_TEST_MODEL_IDS = (
    "P7-E01_V2S_seed42",
    "P7-E04_V2S_BiFPN_CBAM_seed42",
)


@dataclass(frozen=True)
class Phase8ArtifactSpec:
    model_id: str
    family: str
    role: str
    seed: int
    config_path: str
    checkpoint_path: str
    checkpoint_sha256: str
    validation_predictions_path: str
    validation_predictions_sha256: str
    expected_best_epoch: int
    expected_val_roc_auc: float
    expected_val_pr_auc: float
    official_test_allowed: bool


@dataclass(frozen=True)
class Phase8ArtifactLock:
    source_path: Path
    source_commit: str
    manifests: dict[str, str]
    models: dict[str, Phase8ArtifactSpec]


@dataclass(frozen=True)
class Phase8OfficialTestPolicy:
    source_path: Path
    expected_validation_samples: int
    expected_test_samples: int
    expected_test_patients: int
    expected_test_manifest_sha256: str
    threshold_method: str
    threshold_tie_break: str
    authorized_model_ids: tuple[str, ...]
    threshold_output_dirs: dict[str, str]
    official_test_output_dirs: dict[str, str]
    project_test_inference_passes_before_phase8: int
    maximum_new_test_inference_passes: int
    project_test_inference_passes_after_success: int
    bootstrap: dict[str, Any]
    forbidden: dict[str, bool]


@dataclass(frozen=True)
class ValidationEvidence:
    model_id: str
    samples: int
    best_epoch: int
    roc_auc: float
    pr_auc: float
    predictions_sha256: str
    labels: np.ndarray
    probabilities: np.ndarray


def _mapping(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing phase 8 configuration section: {name}")
    return value


def _relative_path(value: object, name: str) -> str:
    text = str(value).replace("\\", "/").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{name} must be a project-relative path.")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ConfigurationError(f"{name} must be a SHA-256 hexadecimal digest.")
    return text


def load_phase8_artifact_lock(path: str | Path) -> Phase8ArtifactLock:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Phase 8 artifact lock root must be a mapping.")
    if int(loaded.get("schema_version", 0)) != 1 or int(loaded.get("stage", 0)) != 8:
        raise ConfigurationError("Phase 8 artifact lock requires schema_version=1 and stage=8.")
    source_commit = str(loaded.get("source_commit", "")).strip().lower()
    if source_commit != PHASE8_SOURCE_COMMIT:
        raise ConfigurationError("Phase 8 artifact lock source commit is not phase-7-final.")

    manifest_values = _mapping(loaded, "manifests")
    if set(manifest_values) != {"train.csv", "val.csv"}:
        raise ConfigurationError("Phase 8 artifact lock may contain only train.csv and val.csv.")
    manifests = {
        str(name): _sha256(value, f"manifests.{name}")
        for name, value in manifest_values.items()
    }

    model_values = _mapping(loaded, "models")
    if tuple(model_values) != PHASE8_MODEL_IDS:
        raise ConfigurationError("Phase 8 artifact lock model order or membership changed.")
    models: dict[str, Phase8ArtifactSpec] = {}
    required = {
        "family",
        "role",
        "seed",
        "config_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "validation_predictions_path",
        "validation_predictions_sha256",
        "expected_best_epoch",
        "expected_val_roc_auc",
        "expected_val_pr_auc",
        "official_test_allowed",
    }
    for model_id, raw in model_values.items():
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ConfigurationError(f"Artifact fields changed for {model_id}.")
        spec = Phase8ArtifactSpec(
            model_id=str(model_id),
            family=str(raw["family"]),
            role=str(raw["role"]),
            seed=int(raw["seed"]),
            config_path=_relative_path(raw["config_path"], f"{model_id}.config_path"),
            checkpoint_path=_relative_path(
                raw["checkpoint_path"], f"{model_id}.checkpoint_path"
            ),
            checkpoint_sha256=_sha256(
                raw["checkpoint_sha256"], f"{model_id}.checkpoint_sha256"
            ),
            validation_predictions_path=_relative_path(
                raw["validation_predictions_path"],
                f"{model_id}.validation_predictions_path",
            ),
            validation_predictions_sha256=_sha256(
                raw["validation_predictions_sha256"],
                f"{model_id}.validation_predictions_sha256",
            ),
            expected_best_epoch=int(raw["expected_best_epoch"]),
            expected_val_roc_auc=float(raw["expected_val_roc_auc"]),
            expected_val_pr_auc=float(raw["expected_val_pr_auc"]),
            official_test_allowed=bool(raw["official_test_allowed"]),
        )
        if spec.seed != 42:
            raise ConfigurationError("Phase 8 locks all model artifacts to seed 42.")
        models[model_id] = spec

    if models["P5-E02_seed42"].family != "phase5":
        raise ConfigurationError("P5-E02 must use the phase 5 model family.")
    if models["P5-E02_seed42"].official_test_allowed:
        raise ConfigurationError("P5-E02 is Grad-CAM-only in phase 8.")
    for model_id in PHASE8_OFFICIAL_TEST_MODEL_IDS:
        if models[model_id].family != "phase7" or not models[model_id].official_test_allowed:
            raise ConfigurationError(f"{model_id} must be an allowed phase 7 test model.")
    return Phase8ArtifactLock(source, source_commit, manifests, models)


def load_phase8_official_test_policy(path: str | Path) -> Phase8OfficialTestPolicy:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Phase 8 official-test policy root must be a mapping.")
    if int(loaded.get("schema_version", 0)) != 1 or int(loaded.get("stage", 0)) != 8:
        raise ConfigurationError("Phase 8 official-test policy requires stage=8.")
    authorized = tuple(str(value) for value in loaded.get("authorized_model_ids", ()))
    if authorized != PHASE8_OFFICIAL_TEST_MODEL_IDS:
        raise ConfigurationError("Only P7-E01 and P7-E04 seed 42 may use official test.")
    thresholds = {
        str(name): _relative_path(value, f"threshold_output_dirs.{name}")
        for name, value in _mapping(loaded, "threshold_output_dirs").items()
    }
    outputs = {
        str(name): _relative_path(value, f"official_test_output_dirs.{name}")
        for name, value in _mapping(loaded, "official_test_output_dirs").items()
    }
    if tuple(thresholds) != authorized or tuple(outputs) != authorized:
        raise ConfigurationError("Phase 8 output mappings must match authorized model order.")
    before = int(loaded.get("project_test_inference_passes_before_phase8", -1))
    maximum = int(loaded.get("maximum_new_test_inference_passes", -1))
    after = int(loaded.get("project_test_inference_passes_after_success", -1))
    if (before, maximum, after) != (1, 2, 3):
        raise ConfigurationError("Phase 8 official-test pass budget must remain 1 + 2 = 3.")
    if loaded.get("threshold_method") != "youden_j":
        raise ConfigurationError("Phase 8 locks threshold_method=youden_j.")
    if loaded.get("threshold_tie_break") != "higher_threshold":
        raise ConfigurationError("Phase 8 locks exact threshold ties to the higher value.")
    forbidden = {str(key): bool(value) for key, value in _mapping(loaded, "forbidden").items()}
    if not forbidden or not all(forbidden.values()):
        raise ConfigurationError("All phase 8 forbidden test behaviors must remain enabled.")
    bootstrap = dict(_mapping(loaded, "bootstrap"))
    if bootstrap != {
        "unit": "patient_id",
        "valid_replicates": 2000,
        "seed": 42,
        "confidence_level": 0.95,
    }:
        raise ConfigurationError("Phase 8 bootstrap policy changed.")
    expected_test_manifest_sha256 = _sha256(
        loaded.get("expected_test_manifest_sha256"),
        "expected_test_manifest_sha256",
    )
    return Phase8OfficialTestPolicy(
        source_path=source,
        expected_validation_samples=int(loaded.get("expected_validation_samples", 0)),
        expected_test_samples=int(loaded.get("expected_test_samples", 0)),
        expected_test_patients=int(loaded.get("expected_test_patients", 0)),
        expected_test_manifest_sha256=expected_test_manifest_sha256,
        threshold_method="youden_j",
        threshold_tie_break="higher_threshold",
        authorized_model_ids=authorized,
        threshold_output_dirs=thresholds,
        official_test_output_dirs=outputs,
        project_test_inference_passes_before_phase8=before,
        maximum_new_test_inference_passes=maximum,
        project_test_inference_passes_after_success=after,
        bootstrap=bootstrap,
        forbidden=forbidden,
    )


def _is_read_only(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    if os.name == "nt":
        return bool(attributes & stat.FILE_ATTRIBUTE_READONLY)
    return not os.access(path, os.W_OK)


def audit_phase8_artifacts(
    project_root: str | Path,
    lock: Phase8ArtifactLock,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    manifest_results: dict[str, dict[str, Any]] = {}
    for name, expected_hash in lock.manifests.items():
        path = root / "artifacts" / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Locked phase 8 manifest is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ValueError(f"Locked phase 8 manifest hash mismatch: {name}")
        manifest_results[name] = {
            "path": str(path),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
            "read_only": _is_read_only(path),
        }

    model_results: dict[str, dict[str, Any]] = {}
    for model_id, spec in lock.models.items():
        paths = {
            "config": root / spec.config_path,
            "checkpoint": root / spec.checkpoint_path,
            "validation_predictions": root / spec.validation_predictions_path,
        }
        for artifact_type, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Locked {model_id} {artifact_type} is missing: {path}"
                )
        checkpoint_hash = sha256_file(paths["checkpoint"])
        prediction_hash = sha256_file(paths["validation_predictions"])
        if checkpoint_hash != spec.checkpoint_sha256:
            raise ValueError(f"Locked checkpoint hash mismatch for {model_id}.")
        if prediction_hash != spec.validation_predictions_sha256:
            raise ValueError(f"Locked validation prediction hash mismatch for {model_id}.")
        model_results[model_id] = {
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "config_path": str(paths["config"]),
            "checkpoint_path": str(paths["checkpoint"]),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": paths["checkpoint"].stat().st_size,
            "checkpoint_read_only": _is_read_only(paths["checkpoint"]),
            "validation_predictions_path": str(paths["validation_predictions"]),
            "validation_predictions_sha256": prediction_hash,
            "validation_predictions_read_only": _is_read_only(
                paths["validation_predictions"]
            ),
            "official_test_allowed": spec.official_test_allowed,
        }
    return {
        "status": "PASS",
        "source_commit": lock.source_commit,
        "manifests": manifest_results,
        "models": model_results,
        "official_test_loaded": False,
        "test_split_loaded": False,
    }


def read_phase8_validation_evidence(
    project_root: str | Path,
    lock: Phase8ArtifactLock,
    policy: Phase8OfficialTestPolicy,
    model_id: str,
) -> ValidationEvidence:
    if model_id not in lock.models:
        raise ConfigurationError(f"Unknown phase 8 model: {model_id}")
    root = Path(project_root).expanduser().resolve()
    spec = lock.models[model_id]
    validation_path = root / "artifacts" / "data" / "val.csv"
    if sha256_file(validation_path) != lock.manifests["val.csv"]:
        raise ValueError("Phase 8 validation manifest hash mismatch.")
    predictions_path = root / spec.validation_predictions_path
    if sha256_file(predictions_path) != spec.validation_predictions_sha256:
        raise ValueError(f"Phase 8 validation prediction hash mismatch for {model_id}.")

    validation = pd.read_csv(
        validation_path, dtype={"sample_id": str, "patient_id": str}
    )
    predictions = pd.read_csv(predictions_path, dtype={"sample_id": str})
    required = {"sample_id", "label", "logit", "probability", "best_epoch"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Validation predictions for {model_id} lack required columns.")
    if len(validation) != policy.expected_validation_samples:
        raise ValueError("Locked validation sample count changed.")
    if len(predictions) != policy.expected_validation_samples:
        raise ValueError(f"Validation predictions for {model_id} must contain 241 rows.")
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate validation sample IDs for {model_id}.")
    expected_ids = validation["sample_id"].astype(str).to_numpy()
    observed_ids = predictions["sample_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError(f"Validation sample order or membership changed for {model_id}.")
    labels = pd.to_numeric(predictions["label"], errors="raise").astype(int).to_numpy()
    expected_labels = pd.to_numeric(validation["label"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Validation labels changed for {model_id}.")
    logits = pd.to_numeric(predictions["logit"], errors="raise").to_numpy(dtype=np.float64)
    probabilities = pd.to_numeric(
        predictions["probability"], errors="raise"
    ).to_numpy(dtype=np.float64)
    recomputed_probabilities = 1.0 / (1.0 + np.exp(-logits))
    if not np.allclose(probabilities, recomputed_probabilities, rtol=0.0, atol=1e-7):
        raise ValueError(f"Stored probabilities do not match logits for {model_id}.")
    epochs = set(pd.to_numeric(predictions["best_epoch"], errors="raise").astype(int))
    if epochs != {spec.expected_best_epoch}:
        raise ValueError(f"Best epoch changed for {model_id}.")
    roc_auc = float(roc_auc_score(labels, probabilities))
    pr_auc = float(average_precision_score(labels, probabilities))
    if not np.isclose(roc_auc, spec.expected_val_roc_auc, rtol=0.0, atol=1e-12):
        raise ValueError(f"Validation ROC AUC changed for {model_id}.")
    if not np.isclose(pr_auc, spec.expected_val_pr_auc, rtol=0.0, atol=1e-12):
        raise ValueError(f"Validation PR AUC changed for {model_id}.")
    return ValidationEvidence(
        model_id=model_id,
        samples=len(predictions),
        best_epoch=spec.expected_best_epoch,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        predictions_sha256=spec.validation_predictions_sha256,
        labels=labels,
        probabilities=probabilities,
    )


def freeze_phase8_thresholds(
    project_root: str | Path,
    lock: Phase8ArtifactLock,
    policy: Phase8OfficialTestPolicy,
) -> dict[str, dict[str, Any]]:
    root = Path(project_root).expanduser().resolve()
    planned: list[tuple[str, Path, ValidationEvidence]] = []
    for model_id in policy.authorized_model_ids:
        spec = lock.models[model_id]
        if not spec.official_test_allowed:
            raise PermissionError(f"{model_id} is not authorized for official test.")
        output_dir = root / policy.threshold_output_dirs[model_id]
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Threshold output directory is not empty: {output_dir}")
        evidence = read_phase8_validation_evidence(root, lock, policy, model_id)
        planned.append((model_id, output_dir, evidence))

    results: dict[str, dict[str, Any]] = {}
    for model_id, output_dir, evidence in planned:
        ensure_new_output_directory(output_dir)
        selection = select_youden_threshold(evidence.labels, evidence.probabilities)
        threshold_path = freeze_threshold(
            output_dir,
            selection,
            provenance={
                "stage": 8,
                "model_id": model_id,
                "checkpoint_sha256": lock.models[model_id].checkpoint_sha256,
                "validation_predictions_sha256": evidence.predictions_sha256,
                "validation_samples": evidence.samples,
                "validation_roc_auc": evidence.roc_auc,
                "validation_pr_auc": evidence.pr_auc,
                "official_test_loaded": False,
                "test_split_loaded": False,
            },
            reference_threshold=0.4515,
        )
        results[model_id] = {
            "threshold_path": str(threshold_path),
            "threshold_sha256": sha256_file(threshold_path),
            "threshold": selection.threshold,
            "youden_j": selection.youden_j,
            "validation_roc_auc": evidence.roc_auc,
            "validation_pr_auc": evidence.pr_auc,
            "official_test_loaded": False,
            "test_split_loaded": False,
        }
    return results


def validate_phase8_test_authorization(
    path: str | Path,
    policy: Phase8OfficialTestPolicy,
) -> dict[str, Any]:
    authorization_path = Path(path).expanduser().resolve()
    if not authorization_path.is_file():
        raise PermissionError("Official test is blocked until a phase 8 authorization exists.")
    with authorization_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    expected_models = list(policy.authorized_model_ids)
    if (
        document.get("format_version") != 1
        or document.get("stage") != 8
        or document.get("official_test_reuse_authorized") is not True
        or document.get("approved_by_user") is not True
        or document.get("authorized_model_ids") != expected_models
        or document.get("maximum_new_inference_passes")
        != policy.maximum_new_test_inference_passes
    ):
        raise PermissionError("Invalid or incomplete phase 8 official-test authorization.")
    return dict(document)


def load_phase8_test_manifest(
    project_root: str | Path,
    lock: Phase8ArtifactLock,
    policy: Phase8OfficialTestPolicy,
    *,
    authorization_path: str | Path,
) -> pd.DataFrame:
    validate_phase8_test_authorization(authorization_path, policy)
    root = Path(project_root).expanduser().resolve()
    threshold_paths: dict[str, Path] = {}
    for model_id in policy.authorized_model_ids:
        threshold_path = root / policy.threshold_output_dirs[model_id] / "threshold.json"
        document = read_frozen_threshold(threshold_path)
        provenance = document.get("provenance", {})
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("model_id") != model_id
            or provenance.get("checkpoint_sha256")
            != lock.models[model_id].checkpoint_sha256
            or provenance.get("official_test_loaded") is not False
        ):
            raise PermissionError(f"Frozen threshold provenance mismatch for {model_id}.")
        evidence = read_phase8_validation_evidence(root, lock, policy, model_id)
        expected_selection = select_youden_threshold(
            evidence.labels,
            evidence.probabilities,
        )
        observed_selection = document.get("selection")
        if (
            document.get("method") != policy.threshold_method
            or document.get("tie_break") != policy.threshold_tie_break
            or not isinstance(observed_selection, Mapping)
            or not np.isclose(
                float(observed_selection.get("threshold", float("nan"))),
                expected_selection.threshold,
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(
                float(observed_selection.get("youden_j", float("nan"))),
                expected_selection.youden_j,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise PermissionError(f"Frozen threshold value mismatch for {model_id}.")
        threshold_paths[model_id] = threshold_path
    first_threshold = threshold_paths[policy.authorized_model_ids[0]]
    test_manifest_path = root / "artifacts" / "data" / "test.csv"
    if sha256_file(test_manifest_path) != policy.expected_test_manifest_sha256:
        raise ValueError("Official test manifest hash mismatch.")
    test = load_test_manifest_after_threshold(
        root / "artifacts" / "data",
        first_threshold,
        expected_samples=policy.expected_test_samples,
    )
    if test["patient_id"].astype(str).nunique() != policy.expected_test_patients:
        raise ValueError("Official test patient count changed.")
    return test


def ensure_phase8_official_test_outputs_absent(
    project_root: str | Path,
    policy: Phase8OfficialTestPolicy,
) -> None:
    """Refuse Gate B before test access if any formal output path already exists."""

    root = Path(project_root).expanduser().resolve()
    paths = [
        root / policy.official_test_output_dirs[model_id]
        for model_id in policy.authorized_model_ids
    ]
    comparison_dir = paths[0].parent / "paired_comparison"
    for path in [*paths, comparison_dir]:
        if path.exists():
            raise FileExistsError(f"Gate B refuses an existing output path: {path}")


def paired_patient_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_model_id: str,
    second_model_id: str,
    valid_replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Estimate paired metric differences by resampling whole patients."""

    required = {"sample_id", "patient_id", "label", "probability"}
    for model_id, frame in ((first_model_id, first), (second_model_id, second)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Predictions for {model_id} are missing: {', '.join(missing)}"
            )
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"Predictions for {model_id} contain duplicate sample IDs.")
    if valid_replicates <= 0:
        raise ValueError("valid_replicates must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    first_ids = first["sample_id"].astype(str).to_numpy()
    second_ids = second["sample_id"].astype(str).to_numpy()
    first_patients = first["patient_id"].astype(str).to_numpy()
    second_patients = second["patient_id"].astype(str).to_numpy()
    first_labels = pd.to_numeric(first["label"], errors="raise").to_numpy(np.int64)
    second_labels = pd.to_numeric(second["label"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(first_ids, second_ids):
        raise ValueError("Paired predictions must have identical sample order.")
    if not np.array_equal(first_patients, second_patients):
        raise ValueError("Paired predictions must have identical patient assignments.")
    if not np.array_equal(first_labels, second_labels):
        raise ValueError("Paired predictions must have identical labels.")
    if set(np.unique(first_labels).tolist()) != {0, 1}:
        raise ValueError("Paired bootstrap requires both binary classes.")
    if (first_patients == "").any():
        raise ValueError("Paired bootstrap requires non-empty patient IDs.")

    first_probabilities = pd.to_numeric(
        first["probability"], errors="raise"
    ).to_numpy(np.float64)
    second_probabilities = pd.to_numeric(
        second["probability"], errors="raise"
    ).to_numpy(np.float64)
    for model_id, probabilities in (
        (first_model_id, first_probabilities),
        (second_model_id, second_probabilities),
    ):
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise ValueError(f"Predictions for {model_id} contain invalid probabilities.")

    patient_ids = np.unique(first_patients)
    patient_indices = {
        patient_id: np.flatnonzero(first_patients == patient_id)
        for patient_id in patient_ids
    }
    rng = np.random.default_rng(seed)
    roc_differences: list[float] = []
    pr_differences: list[float] = []
    attempts = 0
    maximum_attempts = max(valid_replicates * 100, 10_000)
    while len(roc_differences) < valid_replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled_indices = np.concatenate(
            [patient_indices[patient_id] for patient_id in sampled_patients]
        )
        labels = first_labels[sampled_indices]
        if np.unique(labels).size != 2:
            continue
        first_scores = first_probabilities[sampled_indices]
        second_scores = second_probabilities[sampled_indices]
        roc_differences.append(
            float(
                roc_auc_score(labels, first_scores)
                - roc_auc_score(labels, second_scores)
            )
        )
        pr_differences.append(
            float(
                average_precision_score(labels, first_scores)
                - average_precision_score(labels, second_scores)
            )
        )
    if len(roc_differences) != valid_replicates:
        raise RuntimeError(
            f"Only {len(roc_differences)} valid paired bootstrap replicates were produced."
        )

    alpha = 1.0 - confidence_level

    def summarize(
        first_scores: np.ndarray,
        second_scores: np.ndarray,
        differences: list[float],
        metric: str,
    ) -> dict[str, float]:
        if metric == "roc_auc":
            scorer = roc_auc_score
        elif metric == "pr_auc":
            scorer = average_precision_score
        else:
            raise ValueError(f"Unsupported paired metric: {metric}")
        values = np.asarray(differences, dtype=np.float64)
        first_value = float(scorer(first_labels, first_scores))
        second_value = float(scorer(first_labels, second_scores))
        return {
            first_model_id: first_value,
            second_model_id: second_value,
            "difference": first_value - second_value,
            "ci_lower": float(np.quantile(values, alpha / 2.0)),
            "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
        }

    return {
        "format_version": 1,
        "unit": "patient_id",
        "direction": f"{first_model_id}_minus_{second_model_id}",
        "patients": int(len(patient_ids)),
        "samples": int(len(first)),
        "valid_replicates": valid_replicates,
        "attempts": attempts,
        "seed": seed,
        "confidence_level": confidence_level,
        "roc_auc": summarize(
            first_probabilities,
            second_probabilities,
            roc_differences,
            "roc_auc",
        ),
        "pr_auc": summarize(
            first_probabilities,
            second_probabilities,
            pr_differences,
            "pr_auc",
        ),
    }


def load_phase8_frozen_model(
    project_root: str | Path,
    lock: Phase8ArtifactLock,
    model_id: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if model_id not in lock.models:
        raise ConfigurationError(f"Unknown phase 8 model: {model_id}")
    root = Path(project_root).expanduser().resolve()
    spec = lock.models[model_id]
    checkpoint_path = root / spec.checkpoint_path
    if sha256_file(checkpoint_path) != spec.checkpoint_sha256:
        raise ValueError(f"Checkpoint hash mismatch for {model_id}.")

    if spec.family == "phase5":
        experiment = load_phase5_experiment(root / spec.config_path)
        if experiment.experiment_id != model_id:
            raise ConfigurationError("Phase 5 experiment ID does not match artifact lock.")
        expected_model_config = asdict(experiment.model)
        model = build_model(replace(experiment.model, pretrained=False))
    elif spec.family == "phase7":
        experiment = load_phase7_experiment(root / spec.config_path)
        if experiment.experiment_id != model_id:
            raise ConfigurationError("Phase 7 experiment ID does not match artifact lock.")
        expected_model_config = asdict(experiment.model)
        model = build_phase7_model(replace(experiment.model, pretrained=False))
    else:
        raise ConfigurationError(f"Unsupported phase 8 model family: {spec.family}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format for {model_id}.")
    if checkpoint.get("model_config") != expected_model_config:
        raise ValueError(f"Checkpoint model configuration mismatch for {model_id}.")
    training_config = checkpoint.get("training_config", {})
    if not isinstance(training_config, Mapping) or int(training_config.get("seed", -1)) != 42:
        raise ValueError(f"Checkpoint seed mismatch for {model_id}.")
    if checkpoint.get("manifest_hashes") != lock.manifests:
        raise ValueError(f"Checkpoint manifest hashes mismatch for {model_id}.")
    if int(checkpoint.get("best_epoch", -1)) != spec.expected_best_epoch:
        raise ValueError(f"Checkpoint best epoch mismatch for {model_id}.")
    if not np.isclose(
        float(checkpoint.get("best_metric", float("nan"))),
        spec.expected_val_roc_auc,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Checkpoint best metric mismatch for {model_id}.")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    gc.collect()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return model, {
        "model_id": model_id,
        "family": spec.family,
        "role": spec.role,
        "seed": 42,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "strict_load_pass": True,
        "total_parameters": total_parameters,
        "official_test_allowed": spec.official_test_allowed,
        "test_split_loaded": False,
    }
