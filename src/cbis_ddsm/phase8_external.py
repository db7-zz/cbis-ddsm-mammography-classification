"""INbreast audit and external-validation gates for phase 8."""

from __future__ import annotations

import hashlib
import json
import plistlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pydicom
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)

from .config import ConfigurationError
from .evaluation import compute_evaluation_metrics
from .phase8 import PHASE8_OFFICIAL_TEST_MODEL_IDS
from .training import sha256_file


_DICOM_NAME = re.compile(
    r"^(?P<file_id>\d+)_(?P<patient_id>[0-9a-f]+)_MG_"
    r"(?P<laterality>[RL])_(?P<view>CC|ML|MLO|FB)_ANON$",
    re.IGNORECASE,
)
_POINT = re.compile(
    r"^\(\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y>[-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class InbreastExternalPolicy:
    source_path: Path
    dataset_id: str
    layout: dict[str, str]
    expected: dict[str, Any]
    negative_birads: tuple[str, ...]
    positive_birads: tuple[str, ...]
    mapping_status: str
    audit_lock: dict[str, Any]
    authorized_model_ids: tuple[str, ...]
    bootstrap: dict[str, Any]
    runtime: dict[str, Any]
    audit_dir: str
    formal_dir: str
    forbidden: dict[str, bool]


def _mapping(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing INbreast policy section: {name}")
    return value


def _relative_path(value: object, name: str) -> str:
    text = str(value).replace("\\", "/").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{name} must be project-relative.")
    return text


def load_inbreast_external_policy(path: str | Path) -> InbreastExternalPolicy:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ConfigurationError("INbreast policy root must be a mapping.")
    if (
        int(document.get("schema_version", 0)) != 1
        or int(document.get("stage", 0)) != 8
        or str(document.get("gate", "")).upper() != "D"
        or str(document.get("dataset_id", "")) != "INbreast_Release_1.0"
    ):
        raise ConfigurationError("INbreast policy identity changed.")

    layout_raw = _mapping(document, "layout")
    required_layout = {
        "raw_subdir",
        "archive_name",
        "dicom_subdir",
        "xml_subdir",
        "roi_subdir",
        "reports_subdir",
        "summary_csv",
        "summary_xls",
        "readme",
        "publication_pdf",
    }
    if set(layout_raw) != required_layout:
        raise ConfigurationError("INbreast layout fields changed.")
    layout = {str(key): str(value).strip() for key, value in layout_raw.items()}
    if any(not value or Path(value).is_absolute() or ".." in Path(value).parts for value in layout.values()):
        raise ConfigurationError("INbreast layout entries must be safe relative paths.")

    expected = dict(_mapping(document, "expected"))
    archive_sha = str(expected.get("archive_sha256", "")).lower()
    if len(archive_sha) != 64 or any(char not in "0123456789abcdef" for char in archive_sha):
        raise ConfigurationError("INbreast archive SHA-256 is invalid.")
    if expected.get("supported_views") != ["CC", "MLO"]:
        raise ConfigurationError("INbreast external views must remain CC and MLO.")
    if [str(value) for value in expected.get("known_birads_values", [])] != [
        "1",
        "2",
        "3",
        "4a",
        "4b",
        "4c",
        "5",
        "6",
    ]:
        raise ConfigurationError("INbreast known BI-RADS values changed.")

    mapping = _mapping(document, "candidate_birads_mapping")
    negative = tuple(str(value) for value in mapping.get("negative", ()))
    positive = tuple(str(value) for value in mapping.get("positive", ()))
    if negative != ("1", "2", "3") or positive != ("4a", "4b", "4c", "5", "6"):
        raise ConfigurationError("Candidate BI-RADS mapping changed.")
    mapping_status = str(mapping.get("status", ""))
    if mapping_status != "exploratory_only_requires_post_audit_user_confirmation":
        raise ConfigurationError("Candidate BI-RADS mapping must remain unapproved.")

    models = tuple(str(value) for value in document.get("authorized_model_ids", ()))
    if models != PHASE8_OFFICIAL_TEST_MODEL_IDS:
        raise ConfigurationError("Gate D may use only P7-E01 and P7-E04 seed 42.")
    bootstrap = dict(_mapping(document, "bootstrap"))
    if bootstrap != {
        "unit": "patient_id",
        "valid_replicates": 2000,
        "seed": 42,
        "confidence_level": 0.95,
    }:
        raise ConfigurationError("Gate D bootstrap policy changed.")
    audit_lock = dict(_mapping(document, "audit_lock"))
    if set(audit_lock) != {
        "outcome_path",
        "outcome_sha256",
        "candidate_manifest_path",
        "candidate_manifest_sha256",
        "expected_candidate_images",
        "expected_candidate_patients",
        "expected_candidate_negatives",
        "expected_candidate_positives",
    }:
        raise ConfigurationError("Gate D audit lock fields changed.")
    for name in ("outcome_path", "candidate_manifest_path"):
        audit_lock[name] = _relative_path(audit_lock[name], f"audit_lock.{name}")
    for name in ("outcome_sha256", "candidate_manifest_sha256"):
        digest = str(audit_lock[name]).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ConfigurationError(f"audit_lock.{name} must be a SHA-256 digest.")
        audit_lock[name] = digest
    expected_audit_counts = {
        "expected_candidate_images": 314,
        "expected_candidate_patients": 91,
        "expected_candidate_negatives": 227,
        "expected_candidate_positives": 87,
    }
    if {
        name: int(audit_lock[name])
        for name in expected_audit_counts
    } != expected_audit_counts:
        raise ConfigurationError("Gate D locked candidate counts changed.")
    runtime = dict(_mapping(document, "runtime"))
    if runtime != {
        "seed": 42,
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "mixed_precision": True,
        "deterministic": True,
    }:
        raise ConfigurationError("Gate D runtime policy changed.")
    output = _mapping(document, "output")
    if set(output) != {"audit_dir", "formal_dir"}:
        raise ConfigurationError("Gate D output fields changed.")
    forbidden = {
        str(key): bool(value)
        for key, value in _mapping(document, "forbidden").items()
    }
    if not forbidden or not all(forbidden.values()):
        raise ConfigurationError("All Gate D forbidden behaviors must remain enabled.")
    return InbreastExternalPolicy(
        source_path=source,
        dataset_id="INbreast_Release_1.0",
        layout=layout,
        expected=expected,
        negative_birads=negative,
        positive_birads=positive,
        mapping_status=mapping_status,
        audit_lock=audit_lock,
        authorized_model_ids=models,
        bootstrap=bootstrap,
        runtime=runtime,
        audit_dir=_relative_path(output["audit_dir"], "output.audit_dir"),
        formal_dir=_relative_path(output["formal_dir"], "output.formal_dir"),
        forbidden=forbidden,
    )


def validate_inbreast_authorization(
    path: str | Path,
    *,
    dataset_root: str | Path,
    policy: InbreastExternalPolicy,
) -> dict[str, Any]:
    authorization_path = Path(path).expanduser().resolve()
    if not authorization_path.is_file():
        raise PermissionError("Gate D data access requires a separate authorization record.")
    with authorization_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    expected_root = Path(dataset_root).expanduser().resolve()
    authorized_root = Path(str(document.get("dataset_root", ""))).expanduser().resolve()
    if (
        document.get("format_version") != 1
        or document.get("stage") != 8
        or str(document.get("gate", "")).upper() != "D"
        or document.get("approved_by_user") is not True
        or document.get("data_access_authorized") is not True
        or document.get("full_gate_d_workflow_requested") is not True
        or document.get("raw_data_mutation_authorized") is not False
        or document.get("dependency_installation_authorized") is not False
        or document.get("commit_or_push_authorized") is not False
        or authorized_root != expected_root
        or document.get("authorized_model_ids") != list(policy.authorized_model_ids)
    ):
        raise PermissionError("Gate D authorization record does not match the approved scope.")
    if document.get("post_audit_label_route") != "requires_user_confirmation":
        raise PermissionError("Gate D label route must remain pending until audit completes.")
    return document


def validate_inbreast_route_authorization(
    path: str | Path,
    *,
    dataset_root: str | Path,
    policy: InbreastExternalPolicy,
) -> dict[str, Any]:
    authorization_path = Path(path).expanduser().resolve()
    if not authorization_path.is_file():
        raise PermissionError("Gate D route 1 requires a separate authorization record.")
    with authorization_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    expected_root = Path(dataset_root).expanduser().resolve()
    authorized_root = Path(str(document.get("dataset_root", ""))).expanduser().resolve()
    expected_mapping = {
        "negative": list(policy.negative_birads),
        "positive": list(policy.positive_birads),
    }
    if (
        document.get("format_version") != 1
        or document.get("stage") != 8
        or str(document.get("gate", "")).upper() != "D"
        or document.get("approved_by_user") is not True
        or document.get("selected_route") != "exploratory_birads_transfer"
        or document.get("research_use_authorized_for_this_run") is not True
        or document.get("label_equivalence_claimed") is not False
        or document.get("mapping") != expected_mapping
        or document.get("authorized_model_ids") != list(policy.authorized_model_ids)
        or document.get("maximum_external_inference_passes") != 2
        or document.get("training_authorized") is not False
        or document.get("threshold_refit_authorized") is not False
        or document.get("official_test_reload_authorized") is not False
        or document.get("commit_or_push_authorized") is not False
        or authorized_root != expected_root
    ):
        raise PermissionError("Gate D route authorization does not match route 1 scope.")
    return document


def parse_inbreast_dicom_name(path: str | Path) -> dict[str, str]:
    stem = Path(path).stem
    match = _DICOM_NAME.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected INbreast DICOM filename: {Path(path).name}")
    values = {key: value for key, value in match.groupdict().items()}
    view = values["view"].upper()
    values["view"] = "MLO" if view in {"ML", "MLO"} else view
    values["laterality"] = values["laterality"].upper()
    values["patient_id"] = values["patient_id"].lower()
    return values


def normalize_birads(value: object) -> str:
    return str(value).strip().lower().replace(" ", "")


def candidate_birads_label(
    value: object,
    policy: InbreastExternalPolicy,
) -> int:
    normalized = normalize_birads(value)
    if normalized in policy.negative_birads:
        return 0
    if normalized in policy.positive_birads:
        return 1
    raise ValueError(f"Unsupported INbreast BI-RADS value: {value}")


def _roi_names_and_points(path: Path) -> tuple[list[str], list[tuple[float, float]]]:
    with path.open("rb") as handle:
        document = plistlib.load(handle)
    images = document.get("Images", [])
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"INbreast XML must describe exactly one image: {path}")
    rois = images[0].get("ROIs", [])
    if not isinstance(rois, list):
        raise ValueError(f"INbreast XML ROI list is invalid: {path}")
    names: list[str] = []
    points: list[tuple[float, float]] = []
    for roi in rois:
        if not isinstance(roi, Mapping):
            raise ValueError(f"INbreast XML contains a non-mapping ROI: {path}")
        names.append(str(roi.get("Name", "")).strip())
        raw_points = roi.get("Point_px", [])
        if not isinstance(raw_points, list):
            raise ValueError(f"INbreast XML Point_px is invalid: {path}")
        for raw_point in raw_points:
            match = _POINT.match(str(raw_point))
            if match is None:
                raise ValueError(f"INbreast XML point cannot be parsed: {path}")
            points.append((float(match.group("x")), float(match.group("y"))))
    return names, points


def deterministic_external_sample_id(
    *,
    file_id: str,
    patient_id: str,
    laterality: str,
    view: str,
) -> str:
    value = f"inbreast|{file_id}|{patient_id}|{laterality}|{view}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def audit_inbreast_xls(path: str | Path) -> dict[str, Any]:
    workbook = Path(path).expanduser().resolve()
    script = Path(__file__).resolve().parents[2] / "scripts" / "audit_inbreast_xls.ps1"
    powershell = (
        Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").is_file()
        else Path("powershell.exe")
    )
    if not script.is_file():
        raise FileNotFoundError(f"Missing INbreast XLS audit helper: {script}")
    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Path",
            str(workbook),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    document = json.loads(result.stdout)
    if (
        document.get("status") != "PASS"
        or document.get("workbook_sha256") != sha256_file(workbook)
        or document.get("usable_image_rows") != 410
        or document.get("image_level_pathology_link_supported") is not False
    ):
        raise ValueError("INbreast XLS audit result does not satisfy the label gate.")
    return document


def load_locked_inbreast_candidate_manifest(
    project_root: str | Path,
    policy: InbreastExternalPolicy,
) -> pd.DataFrame:
    root = Path(project_root).expanduser().resolve()
    outcome_path = root / policy.audit_lock["outcome_path"]
    manifest_path = root / policy.audit_lock["candidate_manifest_path"]
    if sha256_file(outcome_path) != policy.audit_lock["outcome_sha256"]:
        raise ValueError("Gate D audit outcome SHA-256 mismatch.")
    if sha256_file(manifest_path) != policy.audit_lock["candidate_manifest_sha256"]:
        raise ValueError("Gate D candidate manifest SHA-256 mismatch.")
    with outcome_path.open("r", encoding="utf-8") as handle:
        outcome = json.load(handle)
    if (
        outcome.get("status") != "AUDIT_PASS_ROUTE_CONFIRMATION_REQUIRED"
        or outcome.get("external_inference_performed") is not False
        or outcome.get("label_audit", {}).get(
            "formal_cbis_ddsm_pathology_equivalence_supported"
        )
        is not False
        or outcome.get("label_audit", {}).get("xls", {}).get(
            "image_level_pathology_link_supported"
        )
        is not False
    ):
        raise ValueError("Gate D audit outcome does not permit route 1.")
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "patient_id": str, "file_id": str},
        keep_default_na=False,
    )
    required = {
        "sample_id",
        "patient_id",
        "file_id",
        "laterality",
        "view",
        "birads",
        "candidate_birads_label",
        "candidate_label_source",
        "image_path",
        "inclusion_candidate",
        "classification_route",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Gate D candidate manifest lacks required columns.")
    inclusion = frame["inclusion_candidate"]
    if inclusion.dtype != bool:
        inclusion = inclusion.astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    if inclusion.isna().any():
        raise ValueError("Gate D candidate inclusion flags are invalid.")
    selected = frame[inclusion].copy().reset_index(drop=True)
    selected["candidate_birads_label"] = pd.to_numeric(
        selected["candidate_birads_label"],
        errors="raise",
    ).astype(int)
    if len(selected) != int(policy.audit_lock["expected_candidate_images"]):
        raise ValueError("Gate D candidate image count changed.")
    if selected["patient_id"].nunique() != int(
        policy.audit_lock["expected_candidate_patients"]
    ):
        raise ValueError("Gate D candidate patient count changed.")
    counts = selected["candidate_birads_label"].value_counts().to_dict()
    if counts != {
        0: int(policy.audit_lock["expected_candidate_negatives"]),
        1: int(policy.audit_lock["expected_candidate_positives"]),
    }:
        raise ValueError("Gate D candidate label counts changed.")
    if selected["sample_id"].duplicated().any():
        raise ValueError("Gate D candidate sample IDs are not unique.")
    if selected["patient_id"].eq("").any():
        raise ValueError("Gate D candidate patient IDs must not be empty.")
    if set(selected["view"]) != {"CC", "MLO"}:
        raise ValueError("Gate D candidate views changed.")
    if any(not Path(value).is_file() for value in selected["image_path"]):
        raise FileNotFoundError("Gate D candidate manifest contains a missing image.")
    runtime = selected.copy()
    runtime["split"] = "test"
    runtime["label"] = runtime["candidate_birads_label"]
    return runtime


def patient_cluster_metric_bootstrap(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    valid_replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    required = {"patient_id", "label", "probability"}
    if not required.issubset(predictions.columns):
        raise ValueError("External predictions lack bootstrap columns.")
    patients = predictions["patient_id"].astype(str).to_numpy()
    labels = pd.to_numeric(
        predictions["label"], errors="raise"
    ).to_numpy(np.int64)
    probabilities = pd.to_numeric(
        predictions["probability"],
        errors="raise",
    ).to_numpy(np.float64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("External bootstrap requires both classes.")
    patient_ids = np.asarray(sorted(set(patients.tolist())), dtype=object)
    indices = {
        patient: np.flatnonzero(patients == patient) for patient in patient_ids
    }
    rng = np.random.default_rng(seed)
    metric_names = (
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "f1",
    )
    distributions: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    attempts = 0
    maximum_attempts = max(valid_replicates * 100, valid_replicates)
    while len(distributions["roc_auc"]) < valid_replicates:
        if attempts >= maximum_attempts:
            raise RuntimeError("Unable to produce enough valid external bootstraps.")
        attempts += 1
        sampled_patients = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [indices[patient] for patient in sampled_patients]
        )
        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[sampled_indices]
        if set(np.unique(sampled_labels).tolist()) != {0, 1}:
            continue
        predicted = (sampled_probabilities >= threshold).astype(np.int64)
        tn = int(((sampled_labels == 0) & (predicted == 0)).sum())
        fp = int(((sampled_labels == 0) & (predicted == 1)).sum())
        fn = int(((sampled_labels == 1) & (predicted == 0)).sum())
        tp = int(((sampled_labels == 1) & (predicted == 1)).sum())
        distributions["roc_auc"].append(
            float(roc_auc_score(sampled_labels, sampled_probabilities))
        )
        distributions["pr_auc"].append(
            float(average_precision_score(sampled_labels, sampled_probabilities))
        )
        distributions["sensitivity"].append(
            tp / (tp + fn) if tp + fn else 0.0
        )
        distributions["specificity"].append(
            tn / (tn + fp) if tn + fp else 0.0
        )
        distributions["accuracy"].append(
            float(accuracy_score(sampled_labels, predicted))
        )
        distributions["balanced_accuracy"].append(
            float(balanced_accuracy_score(sampled_labels, predicted))
        )
        distributions["precision"].append(
            float(precision_score(sampled_labels, predicted, zero_division=0))
        )
        distributions["f1"].append(
            float(f1_score(sampled_labels, predicted, zero_division=0))
        )
    point = compute_evaluation_metrics(labels, probabilities, threshold)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "unit": "patient_id",
        "patients": int(len(patient_ids)),
        "samples": int(len(predictions)),
        "valid_replicates": valid_replicates,
        "attempts": attempts,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": {
            name: {
                "point": float(getattr(point, name)),
                "ci_low": float(np.quantile(distributions[name], alpha)),
                "ci_high": float(
                    np.quantile(distributions[name], 1.0 - alpha)
                ),
            }
            for name in metric_names
        },
    }


def build_inbreast_audit(
    dataset_root: str | Path,
    policy: InbreastExternalPolicy,
    *,
    decode_pixels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    raw = root / policy.layout["raw_subdir"]
    required_paths = {
        "archive": root / policy.layout["archive_name"],
        "dicom_dir": raw / policy.layout["dicom_subdir"],
        "xml_dir": raw / policy.layout["xml_subdir"],
        "roi_dir": raw / policy.layout["roi_subdir"],
        "reports_dir": raw / policy.layout["reports_subdir"],
        "summary_csv": raw / policy.layout["summary_csv"],
        "summary_xls": raw / policy.layout["summary_xls"],
        "readme": raw / policy.layout["readme"],
        "publication_pdf": raw / policy.layout["publication_pdf"],
    }
    for name, required in required_paths.items():
        expected_directory = name.endswith("_dir")
        if expected_directory and not required.is_dir():
            raise FileNotFoundError(f"Missing INbreast directory: {required}")
        if not expected_directory and not required.is_file():
            raise FileNotFoundError(f"Missing INbreast file: {required}")
    observed_archive_sha = sha256_file(required_paths["archive"])
    if observed_archive_sha != policy.expected["archive_sha256"]:
        raise ValueError("INbreast release archive SHA-256 mismatch.")

    dicom_paths = sorted(required_paths["dicom_dir"].glob("*.dcm"))
    xml_paths = sorted(required_paths["xml_dir"].glob("*.xml"))
    roi_paths = sorted(required_paths["roi_dir"].glob("*.roi"))
    report_paths = sorted(required_paths["reports_dir"].glob("*.txt"))
    expected_counts = {
        "dicom_images": len(dicom_paths),
        "annotation_xml_files": len(xml_paths),
        "osirix_roi_files": len(roi_paths),
        "medical_report_files": len(report_paths),
    }
    for key, observed in expected_counts.items():
        if observed != int(policy.expected[key]):
            raise ValueError(
                f"INbreast {key} count mismatch: {observed} != {policy.expected[key]}"
            )

    dicom_by_file_id: dict[str, tuple[Path, dict[str, str]]] = {}
    dicom_rows: list[dict[str, Any]] = []
    for path in dicom_paths:
        identity = parse_inbreast_dicom_name(path)
        file_id = identity["file_id"]
        if file_id in dicom_by_file_id:
            raise ValueError(f"Duplicate INbreast file ID: {file_id}")
        dataset = pydicom.dcmread(path)
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        pixel_shape: tuple[int, ...] | None = None
        pixel_finite = None
        pixel_min = None
        pixel_max = None
        if decode_pixels:
            pixels = np.asarray(dataset.pixel_array)
            pixel_shape = tuple(int(value) for value in pixels.shape)
            finite = np.isfinite(pixels)
            pixel_finite = bool(finite.all())
            if not finite.any():
                raise ValueError(f"INbreast image has no finite pixels: {path}")
            pixel_min = float(pixels[finite].min())
            pixel_max = float(pixels[finite].max())
            if pixel_shape != (rows, columns):
                raise ValueError(f"INbreast decoded pixel shape mismatch: {path}")
        dicom_by_file_id[file_id] = (path, identity)
        dicom_rows.append(
            {
                **identity,
                "image_path": str(path),
                "rows": rows,
                "columns": columns,
                "photometric_interpretation": str(
                    getattr(dataset, "PhotometricInterpretation", "")
                ),
                "bits_stored": int(getattr(dataset, "BitsStored", 0)),
                "transfer_syntax_uid": str(dataset.file_meta.TransferSyntaxUID),
                "decoded": decode_pixels,
                "pixel_shape": "x".join(str(value) for value in pixel_shape or ()),
                "pixel_all_finite": pixel_finite,
                "pixel_min": pixel_min,
                "pixel_max": pixel_max,
            }
        )
        del dataset

    xml_by_file_id = {path.stem: path for path in xml_paths}
    if len(xml_by_file_id) != len(xml_paths):
        raise ValueError("Duplicate INbreast XML file IDs.")
    xml_audit: dict[str, dict[str, Any]] = {}
    calcification_names = {
        str(value).strip().casefold()
        for value in policy.expected["calcification_roi_names"]
    }
    all_roi_names: set[str] = set()
    out_of_bounds_points = 0
    for file_id, path in xml_by_file_id.items():
        if file_id not in dicom_by_file_id:
            raise ValueError(f"INbreast XML lacks a matching DICOM: {path}")
        names, points = _roi_names_and_points(path)
        all_roi_names.update(names)
        dicom_path, _ = dicom_by_file_id[file_id]
        dicom_row = next(row for row in dicom_rows if row["file_id"] == file_id)
        columns = int(dicom_row["columns"])
        rows = int(dicom_row["rows"])
        invalid = [
            (x, y)
            for x, y in points
            if not (0.0 <= x < columns and 0.0 <= y < rows)
        ]
        out_of_bounds_points += len(invalid)
        normalized_names = {name.casefold() for name in names}
        xml_audit[file_id] = {
            "annotation_path": str(path),
            "roi_count": len(names),
            "point_count": len(points),
            "roi_names": "|".join(sorted(set(names))),
            "has_calcification_annotation": bool(
                normalized_names.intersection(calcification_names)
            ),
            "out_of_bounds_points": len(invalid),
            "dicom_path": str(dicom_path),
        }

    frame = pd.read_csv(
        required_paths["summary_csv"],
        sep=";",
        dtype=str,
        keep_default_na=False,
    )
    required_columns = {
        "Patient ID",
        "Patient age",
        "Laterality",
        "View",
        "Acquisition date",
        "File Name",
        "ACR",
        "Bi-Rads",
    }
    if set(frame.columns) != required_columns:
        raise ValueError("INbreast summary CSV columns changed.")
    if len(frame) != int(policy.expected["summary_rows"]):
        raise ValueError("INbreast summary CSV row count changed.")
    if frame["File Name"].duplicated().any():
        raise ValueError("INbreast summary CSV contains duplicate file IDs.")

    manifest_rows: list[dict[str, Any]] = []
    mismatched_identity = 0
    for row in frame.to_dict("records"):
        file_id = str(row["File Name"])
        if file_id not in dicom_by_file_id:
            raise ValueError(f"INbreast summary row lacks DICOM: {file_id}")
        image_path, identity = dicom_by_file_id[file_id]
        csv_laterality = str(row["Laterality"]).strip().upper()
        csv_view = str(row["View"]).strip().upper()
        identity_matches = (
            csv_laterality == identity["laterality"]
            and csv_view == identity["view"]
        )
        mismatched_identity += int(not identity_matches)
        birads = normalize_birads(row["Bi-Rads"])
        if birads not in set(policy.expected["known_birads_values"]):
            raise ValueError(f"Unexpected INbreast BI-RADS value: {birads}")
        annotation = xml_audit.get(file_id)
        has_calcification = bool(
            annotation and annotation["has_calcification_annotation"]
        )
        supported_view = csv_view in set(policy.expected["supported_views"])
        inclusion_candidate = has_calcification and supported_view and identity_matches
        reasons: list[str] = []
        if not has_calcification:
            reasons.append("no_calcification_annotation")
        if not supported_view:
            reasons.append("unsupported_view")
        if not identity_matches:
            reasons.append("csv_filename_identity_mismatch")
        candidate_label = candidate_birads_label(birads, policy)
        manifest_rows.append(
            {
                "sample_id": deterministic_external_sample_id(
                    file_id=file_id,
                    patient_id=identity["patient_id"],
                    laterality=csv_laterality,
                    view=csv_view,
                ),
                "split": "external",
                "dataset": policy.dataset_id,
                "patient_id": identity["patient_id"],
                "file_id": file_id,
                "laterality": csv_laterality,
                "view": csv_view,
                "acquisition_date": str(row["Acquisition date"]),
                "acr": str(row["ACR"]),
                "birads": birads,
                "candidate_birads_label": candidate_label,
                "candidate_label_source": "INbreast_BI-RADS_not_pathology",
                "image_path": str(image_path),
                "annotation_path": (
                    str(annotation["annotation_path"]) if annotation else ""
                ),
                "roi_names": annotation["roi_names"] if annotation else "",
                "roi_count": int(annotation["roi_count"]) if annotation else 0,
                "point_count": int(annotation["point_count"]) if annotation else 0,
                "out_of_bounds_points": (
                    int(annotation["out_of_bounds_points"]) if annotation else 0
                ),
                "has_calcification_annotation": has_calcification,
                "supported_view": supported_view,
                "identity_matches": identity_matches,
                "inclusion_candidate": inclusion_candidate,
                "exclusion_reason": "|".join(reasons),
                "formal_pathology_label_available": False,
                "classification_route": "requires_post_audit_user_confirmation",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    dicom_audit = pd.DataFrame(dicom_rows).sort_values("file_id").reset_index(drop=True)
    candidate = manifest[manifest["inclusion_candidate"]]
    readme_text = required_paths["readme"].read_text(
        encoding="latin-1",
        errors="replace",
    )
    xls_audit = audit_inbreast_xls(required_paths["summary_xls"])
    summary = {
        "status": "AUDIT_PASS_ROUTE_CONFIRMATION_REQUIRED",
        "dataset_id": policy.dataset_id,
        "dataset_root": str(root),
        "archive_path": str(required_paths["archive"]),
        "archive_sha256": observed_archive_sha,
        "raw_data_mutated": False,
        "counts": {
            **expected_counts,
            "summary_rows": len(frame),
            "filename_derived_patients": int(manifest["patient_id"].nunique()),
            "xml_matched_images": len(xml_audit),
            "images_without_xml": int(manifest["annotation_path"].eq("").sum()),
            "calcification_related_images": int(
                manifest["has_calcification_annotation"].sum()
            ),
            "candidate_cc_mlo_images": len(candidate),
            "candidate_patients": int(candidate["patient_id"].nunique()),
            "candidate_birads_negative_images": int(
                (candidate["candidate_birads_label"] == 0).sum()
            ),
            "candidate_birads_positive_images": int(
                (candidate["candidate_birads_label"] == 1).sum()
            ),
            "unsupported_view_images": int((~manifest["supported_view"]).sum()),
            "identity_mismatches": mismatched_identity,
            "xml_points_out_of_bounds": out_of_bounds_points,
        },
        "dicom": {
            "all_decoded": bool(decode_pixels and dicom_audit["decoded"].all()),
            "all_pixel_values_finite": bool(
                decode_pixels and dicom_audit["pixel_all_finite"].all()
            ),
            "photometric_interpretations": sorted(
                dicom_audit["photometric_interpretation"].unique().tolist()
            ),
            "transfer_syntax_uids": sorted(
                dicom_audit["transfer_syntax_uid"].unique().tolist()
            ),
            "rows": sorted(int(value) for value in dicom_audit["rows"].unique()),
            "columns": sorted(int(value) for value in dicom_audit["columns"].unique()),
        },
        "annotations": {
            "observed_roi_names": sorted(all_roi_names),
            "calcification_name_policy": sorted(calcification_names),
            "coordinates_checked_against_dicom_grid": True,
        },
        "label_audit": {
            "summary_csv_label_field": "Bi-Rads",
            "readme_describes_birads_summary": "BIRADS classification" in readme_text,
            "xls": xls_audit,
            "uniform_machine_readable_pathology_field": False,
            "formal_cbis_ddsm_pathology_equivalence_supported": False,
            "candidate_mapping": {
                "negative": list(policy.negative_birads),
                "positive": list(policy.positive_birads),
                "status": policy.mapping_status,
            },
            "allowed_next_routes": [
                "exploratory_birads_transfer",
                "annotation_localization_only",
            ],
            "selected_route": None,
        },
        "license_audit": {
            "local_release_readme_present": True,
            "explicit_license_file_present": False,
            "dataset_access_confirmed_by_user": True,
            "formal_use_permission_claimed": False,
            "status": "USER_CONFIRMATION_REQUIRED_FOR_LICENSE_OR_ACCESS_TERMS",
        },
        "external_inference_performed": False,
        "training_performed": False,
        "fine_tuned": False,
        "threshold_refit": False,
        "official_test_loaded": False,
        "test_split_loaded": False,
    }
    return manifest, dicom_audit, summary
