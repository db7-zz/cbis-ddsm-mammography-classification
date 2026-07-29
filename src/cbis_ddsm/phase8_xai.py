"""ROI geometry, deterministic panels, and clustered statistics for phase 8 XAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pydicom

from .gradcam_pp import PHASE8_GRADCAM_MODEL_IDS
from .indexing import PATHOLOGY_TO_LABEL
from .training import sha256_file


ROI_AUDIT_PASS = "geometry_pass"
ROI_AUDIT_SHAPE_MISMATCH = "not_comparable_roi_geometry"


def normalize_inference_preprocessing(
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only inference-relevant fields and reject active augmentation."""

    augmentation_fields = (
        "train_horizontal_flip_probability",
        "train_rotation_degrees",
        "train_affine_probability",
        "train_gamma_probability",
        "train_contrast_probability",
    )
    for field in augmentation_fields:
        if float(preprocessing.get(field, 0.0)) != 0.0:
            raise ValueError(f"Gate C requires zero inference augmentation: {field}")
    required = (
        "image_size",
        "dicom_apply_modality_lut",
        "dicom_apply_voi_lut",
        "lower_percentile",
        "upper_percentile",
        "spatial_mode",
        "breast_margin_fraction",
        "min_foreground_fraction",
        "max_foreground_fraction",
        "min_breast_width_fraction",
        "min_breast_height_fraction",
        "imagenet_mean",
        "imagenet_std",
    )
    missing = [field for field in required if field not in preprocessing]
    if missing:
        raise ValueError(
            "Gate C preprocessing is missing inference fields: " + ", ".join(missing)
        )
    return {field: preprocessing[field] for field in required}


def normalize_official_path(value: object) -> str:
    cleaned = str(value).strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def validate_gradcam_authorization(path: str | Path) -> dict[str, Any]:
    authorization_path = Path(path).expanduser().resolve()
    if not authorization_path.is_file():
        raise PermissionError("Gate C requires a separate Grad-CAM++ authorization.")
    with authorization_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if (
        document.get("format_version") != 1
        or document.get("stage") != 8
        or document.get("gate") != "C"
        or document.get("approved_by_user") is not True
        or document.get("official_test_images_authorized") is not True
        or document.get("official_test_roi_masks_authorized") is not True
        or tuple(document.get("authorized_model_ids", ()))
        != PHASE8_GRADCAM_MODEL_IDS
        or document.get("training_authorized") is not False
        or document.get("gate_d_authorized") is not False
    ):
        raise PermissionError("Invalid or incomplete phase 8 Gate C authorization.")
    return dict(document)


def _candidate_roi_directories(
    *,
    data_root: Path,
    official_roi_path: str,
    metadata_row: pd.Series,
) -> list[tuple[str, Path]]:
    parts = normalize_official_path(official_roi_path).split("/")
    if len(parts) < 4:
        raise ValueError("Official ROI path does not contain study and series UIDs.")
    prefix, study_uid, series_uid = parts[0], parts[-3], parts[-2]
    names = [str(metadata_row["PatientID"]), prefix]
    if prefix.startswith("Calc-Test_"):
        names.append(prefix[len("Calc-Test_") :] + ".dcm")
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for name in names:
        directory = (
            data_root
            / name
            / study_uid
            / f"{metadata_row.get('Modality', 'MG')}_{series_uid}"
        )
        key = str(directory.resolve(strict=False)).lower()
        if key not in seen:
            candidates.append(("derived", directory))
            seen.add(key)
    fallback_text = str(metadata_row.get("S5cmdManifestPath", "")).strip()
    if fallback_text:
        fallback = Path(fallback_text)
        key = str(fallback.resolve(strict=False)).lower()
        if key not in seen:
            candidates.append(("metadata_fallback", fallback))
    return candidates


def _resolve_roi_mask(
    *,
    data_root: Path,
    official_roi_path: str,
    metadata_row: pd.Series,
    expected_shape: tuple[int, int],
    expected_view: str,
) -> dict[str, Any]:
    discovered: list[tuple[str, Path, Any]] = []
    for method, directory in _candidate_roi_directories(
        data_root=data_root,
        official_roi_path=official_roi_path,
        metadata_row=metadata_row,
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.dcm")):
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            discovered.append((method, path.resolve(), dataset))

    by_sop: dict[str, list[tuple[str, Path, Any]]] = {}
    for item in discovered:
        sop_uid = str(getattr(item[2], "SOPInstanceUID", "")).strip()
        if not sop_uid:
            raise ValueError(f"ROI candidate lacks SOPInstanceUID: {item[1]}")
        by_sop.setdefault(sop_uid, []).append(item)
    unique_candidates = [
        sorted(items, key=lambda item: str(item[1]).lower())[0]
        for items in by_sop.values()
    ]
    duplicate_copies = sum(max(0, len(items) - 1) for items in by_sop.values())
    matching = [
        item
        for item in unique_candidates
        if (int(item[2].Rows), int(item[2].Columns)) == expected_shape
    ]
    binary_headers = [
        item for item in unique_candidates if int(getattr(item[2], "BitsAllocated", 0)) == 8
    ]
    base = {
        "candidate_files": len(discovered),
        "unique_sop_instances": len(unique_candidates),
        "duplicate_copies": duplicate_copies,
        "expected_rows": expected_shape[0],
        "expected_columns": expected_shape[1],
    }
    if len(matching) != 1:
        mismatch = binary_headers[0] if len(binary_headers) == 1 else None
        return {
            **base,
            "status": ROI_AUDIT_SHAPE_MISMATCH,
            "reason": (
                f"expected one full-size ROI mask, found {len(matching)}; "
                f"unique series instances={len(unique_candidates)}"
            ),
            "roi_mask_path": str(mismatch[1]) if mismatch else "",
            "roi_rows": int(mismatch[2].Rows) if mismatch else None,
            "roi_columns": int(mismatch[2].Columns) if mismatch else None,
            "roi_sop_instance_uid": (
                str(mismatch[2].SOPInstanceUID) if mismatch else ""
            ),
        }

    method, path, header = matching[0]
    observed_view = str(getattr(header, "PatientOrientation", "")).strip().upper()
    if observed_view != expected_view:
        return {
            **base,
            "status": ROI_AUDIT_SHAPE_MISMATCH,
            "reason": (
                f"ROI PatientOrientation={observed_view!r} does not match "
                f"official view={expected_view!r}"
            ),
            "roi_mask_path": str(path),
            "roi_rows": int(header.Rows),
            "roi_columns": int(header.Columns),
            "roi_sop_instance_uid": str(header.SOPInstanceUID),
        }
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array)
    unique_values = np.unique(pixels)
    if (
        pixels.ndim != 2
        or unique_values.size != 2
        or int(unique_values.min()) != 0
        or int(unique_values.max()) <= 0
        or not np.count_nonzero(pixels)
    ):
        return {
            **base,
            "status": ROI_AUDIT_SHAPE_MISMATCH,
            "reason": "Resolved full-size ROI object is not a non-empty binary mask.",
            "roi_mask_path": str(path),
            "roi_rows": int(header.Rows),
            "roi_columns": int(header.Columns),
            "roi_sop_instance_uid": str(header.SOPInstanceUID),
        }
    return {
        **base,
        "status": ROI_AUDIT_PASS,
        "reason": "",
        "resolution_method": method,
        "roi_mask_path": str(path),
        "roi_mask_sha256": sha256_file(path),
        "roi_rows": int(header.Rows),
        "roi_columns": int(header.Columns),
        "roi_sop_instance_uid": str(header.SOPInstanceUID),
        "roi_view": observed_view,
        "roi_bits_allocated": int(header.BitsAllocated),
        "roi_min": int(unique_values.min()),
        "roi_max": int(unique_values.max()),
        "roi_nonzero_pixels": int(np.count_nonzero(pixels)),
        "roi_nonzero_fraction": float(np.count_nonzero(pixels) / pixels.size),
    }


def audit_roi_geometry(
    test_manifest: pd.DataFrame,
    *,
    official_csv_path: str | Path,
    metadata_csv_path: str | Path,
    data_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a row-level official-record-to-whole-image ROI audit."""

    official_path = Path(official_csv_path).expanduser().resolve()
    metadata_path = Path(metadata_csv_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not official_path.is_file() or not metadata_path.is_file() or not root.is_dir():
        raise FileNotFoundError("Gate C source CSVs and CBIS-DDSM data root are required.")

    official = pd.read_csv(official_path, dtype=str, keep_default_na=False)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    required_official = {
        "patient_id",
        "left or right breast",
        "image view",
        "abnormality id",
        "pathology",
        "image file path",
        "ROI mask file path",
    }
    missing = sorted(required_official.difference(official.columns))
    if missing:
        raise ValueError(f"Official calcification CSV is missing: {', '.join(missing)}")
    required_metadata = {
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "SeriesDescription",
        "S5cmdManifestPath",
    }
    missing = sorted(required_metadata.difference(metadata.columns))
    if missing:
        raise ValueError(f"IDC metadata is missing: {', '.join(missing)}")
    if metadata["SeriesInstanceUID"].duplicated().any():
        raise ValueError("IDC metadata contains duplicate SeriesInstanceUID values.")
    metadata_by_series = metadata.set_index("SeriesInstanceUID")

    manifest = test_manifest.copy()
    manifest["_official_path"] = manifest["image_file_path_raw"].map(
        normalize_official_path
    )
    if manifest["_official_path"].duplicated().any():
        raise ValueError("Test manifest contains duplicate official image paths.")
    manifest_by_path = manifest.set_index("_official_path")
    whole_headers: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for source_row, row in official.iterrows():
        image_key = normalize_official_path(row["image file path"])
        if image_key not in manifest_by_path.index:
            raise ValueError(f"Official row is absent from test manifest: {image_key}")
        sample = manifest_by_path.loc[image_key]
        patient_id = str(row["patient_id"]).strip()
        laterality = str(row["left or right breast"]).strip().upper()
        view = str(row["image view"]).strip().upper()
        pathology = str(row["pathology"]).strip().upper()
        if pathology not in PATHOLOGY_TO_LABEL:
            raise ValueError(f"Unknown pathology in official CSV: {pathology}")
        if (
            str(sample["patient_id"]) != patient_id
            or str(sample["laterality"]).upper() != laterality
            or str(sample["view"]).upper() != view
        ):
            raise ValueError(f"Official/manifest identity mismatch for {image_key}")
        prefix = normalize_official_path(row["ROI mask file path"]).split("/")[0]
        identity_token = f"_{laterality}_{view}_"
        if patient_id not in prefix or identity_token not in prefix:
            raise ValueError(f"ROI path identity mismatch for official row {source_row}.")

        whole_path = Path(str(sample["image_path"]))
        header = whole_headers.get(str(whole_path))
        if header is None:
            header = pydicom.dcmread(whole_path, stop_before_pixels=True)
            whole_headers[str(whole_path)] = header
        whole_shape = (int(header.Rows), int(header.Columns))
        whole_view = str(getattr(header, "PatientOrientation", "")).strip().upper()
        if whole_view != view:
            raise ValueError(
                f"Whole mammogram PatientOrientation mismatch for {sample['sample_id']}."
            )

        roi_parts = normalize_official_path(row["ROI mask file path"]).split("/")
        study_uid, series_uid = roi_parts[-3], roi_parts[-2]
        if series_uid not in metadata_by_series.index:
            raise ValueError(f"ROI series is absent from IDC metadata: {series_uid}")
        metadata_row = metadata_by_series.loc[series_uid]
        if str(metadata_row["StudyInstanceUID"]) != study_uid:
            raise ValueError(f"ROI study UID mismatch for series {series_uid}.")
        if str(metadata_row["SeriesDescription"]).strip().lower() not in {
            "cropped images",
            "roi mask images",
        }:
            raise ValueError(f"Unexpected ROI series description for {series_uid}.")
        resolved = _resolve_roi_mask(
            data_root=root,
            official_roi_path=row["ROI mask file path"],
            metadata_row=metadata_row,
            expected_shape=whole_shape,
            expected_view=view,
        )
        records.append(
            {
                "source_row": int(source_row),
                "sample_id": str(sample["sample_id"]),
                "patient_id": patient_id,
                "laterality": laterality,
                "view": view,
                "image_label": int(sample["label"]),
                "abnormality_id": str(row["abnormality id"]).strip(),
                "abnormality_pathology": pathology,
                "abnormality_label": PATHOLOGY_TO_LABEL[pathology],
                "whole_image_path": str(whole_path),
                "whole_rows": whole_shape[0],
                "whole_columns": whole_shape[1],
                "whole_view": whole_view,
                "official_image_file_path": image_key,
                "official_roi_mask_file_path": normalize_official_path(
                    row["ROI mask file path"]
                ),
                "roi_study_instance_uid": study_uid,
                "roi_series_instance_uid": series_uid,
                **resolved,
            }
        )

    row_audit = pd.DataFrame.from_records(records)
    image_records: list[dict[str, Any]] = []
    for sample_id, group in row_audit.groupby("sample_id", sort=True):
        statuses = group["status"].astype(str)
        image_label = int(group["image_label"].iloc[0])
        if int(group["abnormality_label"].max()) != image_label:
            raise ValueError(f"Official pathology aggregation mismatch for {sample_id}.")
        image_records.append(
            {
                "sample_id": sample_id,
                "patient_id": group["patient_id"].iloc[0],
                "laterality": group["laterality"].iloc[0],
                "view": group["view"].iloc[0],
                "label": image_label,
                "official_roi_rows": int(len(group)),
                "resolved_roi_rows": int((statuses == ROI_AUDIT_PASS).sum()),
                "roi_geometry_complete": bool((statuses == ROI_AUDIT_PASS).all()),
                "quantitative_localization_status": (
                    "eligible"
                    if (statuses == ROI_AUDIT_PASS).all()
                    else ROI_AUDIT_SHAPE_MISMATCH
                ),
            }
        )
    image_audit = pd.DataFrame.from_records(image_records)
    if set(image_audit["sample_id"]) != set(manifest["sample_id"].astype(str)):
        raise ValueError("ROI audit image membership does not match official test manifest.")
    summary = {
        "format_version": 1,
        "stage": 8,
        "gate": "C",
        "status": "PASS_WITH_DECLARED_EXCLUSIONS",
        "official_test_samples": int(len(manifest)),
        "official_roi_rows": int(len(row_audit)),
        "resolved_roi_rows": int((row_audit["status"] == ROI_AUDIT_PASS).sum()),
        "excluded_roi_rows": int((row_audit["status"] != ROI_AUDIT_PASS).sum()),
        "complete_roi_images": int(image_audit["roi_geometry_complete"].sum()),
        "excluded_roi_images": int((~image_audit["roi_geometry_complete"]).sum()),
        "malignant_images": int((image_audit["label"] == 1).sum()),
        "eligible_malignant_images": int(
            ((image_audit["label"] == 1) & image_audit["roi_geometry_complete"]).sum()
        ),
        "excluded_malignant_images": int(
            ((image_audit["label"] == 1) & ~image_audit["roi_geometry_complete"]).sum()
        ),
        "exclusion_reason": ROI_AUDIT_SHAPE_MISMATCH,
        "official_csv_sha256": sha256_file(official_path),
        "metadata_csv_sha256": sha256_file(metadata_path),
        "test_manifest_sha256": sha256_file(
            Path(test_manifest.attrs["source_path"])
            if "source_path" in test_manifest.attrs
            else Path("artifacts/data/test.csv")
        ),
        "coordinate_claim": (
            "Official row identity and full-size binary ROI pixel matrix match the "
            "whole mammogram rows, columns, and CC/MLO orientation."
        ),
    }
    return row_audit, image_audit, summary


def load_union_roi_mask(
    row_audit: pd.DataFrame,
    *,
    sample_id: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    rows = row_audit[row_audit["sample_id"].astype(str) == str(sample_id)]
    if rows.empty or not (rows["status"] == ROI_AUDIT_PASS).all():
        raise ValueError(f"ROI geometry is incomplete for sample {sample_id}.")
    union = np.zeros(expected_shape, dtype=bool)
    for path in rows["roi_mask_path"].astype(str):
        pixels = np.asarray(pydicom.dcmread(path).pixel_array)
        if tuple(pixels.shape) != expected_shape:
            raise ValueError(f"ROI mask shape changed for sample {sample_id}.")
        union |= pixels > 0
    if not union.any():
        raise ValueError(f"Union ROI mask is empty for sample {sample_id}.")
    return union


def clustered_mean_bootstrap(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    valid_replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if frame.empty or "patient_id" not in frame:
        raise ValueError("Cluster bootstrap requires non-empty patient_id data.")
    patient_ids = np.unique(frame["patient_id"].astype(str).to_numpy())
    groups = {
        patient_id: frame.index[frame["patient_id"].astype(str) == patient_id].to_numpy()
        for patient_id in patient_ids
    }
    rng = np.random.default_rng(seed)
    replicates = {column: [] for column in value_columns}
    for _ in range(valid_replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        indices = np.concatenate([groups[patient_id] for patient_id in sampled])
        for column in value_columns:
            values = pd.to_numeric(frame.loc[indices, column], errors="raise").to_numpy(
                np.float64
            )
            replicates[column].append(float(values.mean()))
    alpha = 1.0 - confidence_level
    metrics: dict[str, dict[str, float]] = {}
    for column in value_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        samples = np.asarray(replicates[column], dtype=np.float64)
        metrics[column] = {
            "mean": float(values.mean()),
            "ci_lower": float(np.quantile(samples, alpha / 2.0)),
            "ci_upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        }
    return {
        "unit": "patient_id",
        "patients": int(len(patient_ids)),
        "images": int(len(frame)),
        "valid_replicates": int(valid_replicates),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "metrics": metrics,
    }


def paired_clustered_difference_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_model_id: str,
    second_model_id: str,
    value_columns: Sequence[str],
    valid_replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    required = {"sample_id", "patient_id", *value_columns}
    for model_id, frame in ((first_model_id, first), (second_model_id, second)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Localization results for {model_id} are missing: {', '.join(missing)}"
            )
    merged = first[list(required)].merge(
        second[list(required)],
        on=["sample_id", "patient_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_first", "_second"),
    )
    if merged.empty:
        raise ValueError("Paired localization comparison has no common images.")
    patient_ids = np.unique(merged["patient_id"].astype(str).to_numpy())
    groups = {
        patient_id: merged.index[
            merged["patient_id"].astype(str) == patient_id
        ].to_numpy()
        for patient_id in patient_ids
    }
    rng = np.random.default_rng(seed)
    replicate_differences = {column: [] for column in value_columns}
    for _ in range(valid_replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        indices = np.concatenate([groups[patient_id] for patient_id in sampled])
        for column in value_columns:
            differences = (
                pd.to_numeric(
                    merged.loc[indices, f"{column}_first"],
                    errors="raise",
                ).to_numpy(np.float64)
                - pd.to_numeric(
                    merged.loc[indices, f"{column}_second"],
                    errors="raise",
                ).to_numpy(np.float64)
            )
            replicate_differences[column].append(float(differences.mean()))
    alpha = 1.0 - confidence_level
    metrics: dict[str, dict[str, float]] = {}
    for column in value_columns:
        differences = (
            pd.to_numeric(merged[f"{column}_first"], errors="raise").to_numpy(
                np.float64
            )
            - pd.to_numeric(merged[f"{column}_second"], errors="raise").to_numpy(
                np.float64
            )
        )
        samples = np.asarray(replicate_differences[column], dtype=np.float64)
        metrics[column] = {
            "mean_difference": float(differences.mean()),
            "ci_lower": float(np.quantile(samples, alpha / 2.0)),
            "ci_upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        }
    return {
        "unit": "patient_id",
        "direction": f"{first_model_id}_minus_{second_model_id}",
        "patients": int(len(patient_ids)),
        "images": int(len(merged)),
        "valid_replicates": int(valid_replicates),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "metrics": metrics,
    }
