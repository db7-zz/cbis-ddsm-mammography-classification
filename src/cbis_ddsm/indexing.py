"""Official CSV indexing, patient-level splitting, and data integrity checks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd


PATHOLOGY_TO_LABEL = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}

REQUIRED_COLUMNS = {
    "patient_id",
    "left or right breast",
    "image view",
    "pathology",
    "image file path",
}

MANIFEST_COLUMNS = [
    "sample_id",
    "split",
    "official_split",
    "patient_id",
    "laterality",
    "view",
    "label",
    "pathology_values",
    "study_instance_uid",
    "series_instance_uid",
    "idc_series_description",
    "image_path",
    "image_file_path_raw",
    "source_row_count",
    "image_exists",
]


class DataIntegrityError(RuntimeError):
    """Raised when source metadata violates the locked data protocol."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower() for column in frame.columns}
    result = frame.rename(columns=renamed)
    missing = sorted(REQUIRED_COLUMNS.difference(result.columns))
    if missing:
        raise DataIntegrityError(f"CSV is missing required columns: {', '.join(missing)}")
    return result


def _normalize_relative_path(raw_path: str) -> str:
    cleaned = str(raw_path).strip().strip('"').replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise DataIntegrityError("Encountered an empty whole-mammogram image file path.")
    path = PurePosixPath(cleaned)
    if path.is_absolute() or ".." in path.parts:
        raise DataIntegrityError(f"Unsafe image path in official CSV: {raw_path!r}")
    return path.as_posix()


def _resolve_image_path(
    data_root: Path,
    relative_path: str,
    allowed_extensions: set[str],
) -> tuple[Path, bool]:
    candidate = data_root.joinpath(*PurePosixPath(relative_path).parts).resolve(strict=False)
    root = data_root.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DataIntegrityError(f"Resolved image path escapes data root: {relative_path}") from exc

    if candidate.is_file():
        return candidate, True
    if candidate.is_dir():
        files = sorted(
            path
            for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed_extensions
        )
        if len(files) == 1:
            return files[0].resolve(), True
        if len(files) > 1:
            raise DataIntegrityError(
                f"Image path resolves to a directory with multiple candidate images: {candidate}"
            )
    return candidate, False


def _official_uids(relative_path: str) -> tuple[str, str]:
    parts = PurePosixPath(relative_path).parts
    if len(parts) < 4:
        return "", ""
    return str(parts[-3]), str(parts[-2])


def _load_idc_metadata(metadata_csv: str | Path) -> dict[str, dict[str, str]]:
    path = Path(metadata_csv).expanduser().resolve()
    if not path.is_file():
        raise DataIntegrityError(f"IDC metadata CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "SeriesDescription",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataIntegrityError("IDC metadata is missing columns: " + ", ".join(missing))
    if frame["SeriesInstanceUID"].duplicated().any():
        raise DataIntegrityError("IDC metadata contains duplicate SeriesInstanceUID values.")
    return {
        str(row["SeriesInstanceUID"]): {column: str(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    }


def _resolve_idc_image_path(
    collection_root: Path,
    relative_path: str,
    idc_by_series: dict[str, dict[str, str]],
) -> tuple[Path, bool, str, str, str]:
    study_uid, series_uid = _official_uids(relative_path)
    if not study_uid or not series_uid:
        raise DataIntegrityError(
            f"Official image path does not contain Study/Series UIDs: {relative_path}"
        )
    if series_uid not in idc_by_series:
        raise DataIntegrityError(f"Official SeriesInstanceUID is absent from IDC metadata: {series_uid}")
    row = idc_by_series[series_uid]
    if row["StudyInstanceUID"] != study_uid:
        raise DataIntegrityError(
            f"Study UID mismatch for series {series_uid}: official={study_uid}, "
            f"IDC={row['StudyInstanceUID']}"
        )
    description = row.get("SeriesDescription", "").strip()
    if description.lower() in {"cropped images", "roi mask images"}:
        raise DataIntegrityError(
            f"Official whole-mammogram UID mapped to forbidden IDC series type: {description}"
        )

    series_dir = (
        collection_root
        / row["PatientID"]
        / study_uid
        / f"{row.get('Modality', 'MG')}_{series_uid}"
    )
    candidates = sorted(series_dir.glob("*.dcm")) if series_dir.is_dir() else []
    if len(candidates) > 1:
        raise DataIntegrityError(f"Multiple DICOM files found for IDC series: {series_dir}")
    if len(candidates) == 1:
        return candidates[0].resolve(), True, study_uid, series_uid, description

    fallback_text = row.get("S5cmdManifestPath", "").strip()
    if fallback_text:
        fallback = Path(fallback_text)
        fallback_candidates = sorted(fallback.glob("*.dcm")) if fallback.is_dir() else []
        if len(fallback_candidates) > 1:
            raise DataIntegrityError(f"Multiple DICOM files found in IDC fallback path: {fallback}")
        if len(fallback_candidates) == 1:
            return fallback_candidates[0].resolve(), True, study_uid, series_uid, description
    return series_dir.resolve(strict=False), False, study_uid, series_uid, description


def _sample_id(official_split: str, patient_id: str, relative_path: str) -> str:
    payload = f"{official_split}|{patient_id}|{relative_path}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def read_official_csv(csv_path: str | Path, official_split: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise DataIntegrityError(f"Official {official_split} CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = _canonicalize_columns(frame)
    frame["official_split"] = official_split

    if "abnormality type" in frame.columns:
        types = {value.strip().upper() for value in frame["abnormality type"] if value.strip()}
        unexpected = sorted(types.difference({"CALCIFICATION"}))
        if unexpected:
            raise DataIntegrityError(
                "Non-calcification rows found in calcification CSV: " + ", ".join(unexpected)
            )
    return frame


def build_manifest(
    train_csv: str | Path,
    test_csv: str | Path,
    data_root: str | Path,
    *,
    allowed_extensions: Iterable[str] = (
        ".dcm",
        ".dicom",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    ),
    verify_images: bool = True,
    strict_missing_images: bool = True,
    idc_metadata_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build one sample per whole-mammogram view from official calc CSV rows.

    Multiple abnormalities may refer to the same whole image.  Such rows are
    collapsed.  If any abnormality on that image is malignant, the image label is
    malignant, avoiding duplicate samples with contradictory labels.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise DataIntegrityError(f"Data root does not exist or is not a directory: {root}")

    frames = [
        read_official_csv(train_csv, "train"),
        read_official_csv(test_csv, "test"),
    ]
    source = pd.concat(frames, ignore_index=True)
    source["patient_id"] = source["patient_id"].str.strip()
    if (source["patient_id"] == "").any():
        raise DataIntegrityError("One or more rows have an empty patient_id.")

    source["pathology_normalized"] = source["pathology"].str.strip().str.upper()
    unknown = sorted(set(source["pathology_normalized"]).difference(PATHOLOGY_TO_LABEL))
    if unknown:
        raise DataIntegrityError("Unsupported pathology values: " + ", ".join(unknown))
    source["row_label"] = source["pathology_normalized"].map(PATHOLOGY_TO_LABEL).astype(int)
    source["image_rel_path"] = source["image file path"].map(_normalize_relative_path)

    train_patients = set(source.loc[source["official_split"] == "train", "patient_id"])
    test_patients = set(source.loc[source["official_split"] == "test", "patient_id"])
    overlap = sorted(train_patients.intersection(test_patients))
    if overlap:
        preview = ", ".join(overlap[:10])
        raise DataIntegrityError(
            f"Official train/test patient overlap detected ({len(overlap)}): {preview}"
        )

    extensions = {
        extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
        for extension in allowed_extensions
    }
    idc_by_series = _load_idc_metadata(idc_metadata_csv) if idc_metadata_csv else None
    records: list[dict[str, Any]] = []
    grouping = ["official_split", "patient_id", "image_rel_path"]
    for (official_split, patient_id, relative_path), group in source.groupby(
        grouping, sort=True, dropna=False
    ):
        lateralities = {value.strip().upper() for value in group["left or right breast"]}
        views = {value.strip().upper() for value in group["image view"]}
        if len(lateralities) != 1 or "" in lateralities:
            raise DataIntegrityError(
                f"Inconsistent laterality for patient {patient_id}, image {relative_path}"
            )
        if len(views) != 1 or "" in views:
            raise DataIntegrityError(
                f"Inconsistent image view for patient {patient_id}, image {relative_path}"
            )
        invalid_views = views.difference({"CC", "MLO"})
        if invalid_views:
            raise DataIntegrityError(f"Unsupported image view(s): {', '.join(sorted(invalid_views))}")

        if idc_by_series is not None:
            resolved, exists, study_uid, series_uid, idc_description = _resolve_idc_image_path(
                root, relative_path, idc_by_series
            )
        else:
            resolved, exists = _resolve_image_path(root, relative_path, extensions)
            study_uid, series_uid = _official_uids(relative_path)
            idc_description = ""
        if verify_images and strict_missing_images and not exists:
            raise DataIntegrityError(f"Whole-mammogram image does not exist: {resolved}")

        pathologies = sorted(set(group["pathology_normalized"]))
        label = int(group["row_label"].max())
        records.append(
            {
                "sample_id": _sample_id(str(official_split), str(patient_id), str(relative_path)),
                "split": "test" if official_split == "test" else "unassigned",
                "official_split": str(official_split),
                "patient_id": str(patient_id),
                "laterality": next(iter(lateralities)),
                "view": next(iter(views)),
                "label": label,
                "pathology_values": "|".join(pathologies),
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "idc_series_description": idc_description,
                "image_path": str(resolved),
                "image_file_path_raw": str(relative_path),
                "source_row_count": int(len(group)),
                "image_exists": bool(exists),
            }
        )

    manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        raise DataIntegrityError("No samples were produced from the official CSV files.")
    if manifest["sample_id"].duplicated().any():
        raise DataIntegrityError("Internal error: duplicate sample IDs were generated.")
    return manifest


def _validation_patients_for_stratum(
    patients: list[str], val_fraction: float, rng: random.Random
) -> set[str]:
    ordered = sorted(patients)
    rng.shuffle(ordered)
    if len(ordered) < 2:
        return set()
    count = round(len(ordered) * val_fraction)
    count = min(max(int(count), 1), len(ordered) - 1)
    return set(ordered[:count])


def split_official_train_by_patient(
    manifest: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    result = manifest.copy()
    official_train = result[result["official_split"] == "train"]
    if official_train.empty:
        raise DataIntegrityError("Official training subset is empty.")

    patient_labels = official_train.groupby("patient_id")["label"].max().astype(int)
    rng = random.Random(seed)
    val_patients: set[str] = set()
    for label in sorted(patient_labels.unique()):
        patients = patient_labels[patient_labels == label].index.astype(str).tolist()
        val_patients.update(_validation_patients_for_stratum(patients, val_fraction, rng))

    if not val_patients:
        raise DataIntegrityError(
            "Could not create a validation split. At least two official-train patients "
            "are required in one patient-label stratum."
        )
    train_mask = result["official_split"] == "train"
    result.loc[train_mask, "split"] = "train"
    result.loc[train_mask & result["patient_id"].isin(val_patients), "split"] = "val"
    result.loc[result["official_split"] == "test", "split"] = "test"
    assert_no_patient_overlap(result)
    return result.sort_values(["split", "patient_id", "sample_id"]).reset_index(drop=True)


def patient_overlaps(manifest: pd.DataFrame) -> dict[str, list[str]]:
    patients = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"].astype(str))
        for split in ("train", "val", "test")
    }
    return {
        "train_val": sorted(patients["train"].intersection(patients["val"])),
        "train_test": sorted(patients["train"].intersection(patients["test"])),
        "val_test": sorted(patients["val"].intersection(patients["test"])),
    }


def assert_no_patient_overlap(manifest: pd.DataFrame) -> None:
    overlaps = patient_overlaps(manifest)
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        details = "; ".join(f"{name}={values[:10]}" for name, values in nonempty.items())
        raise DataIntegrityError(f"Patient leakage across splits: {details}")


def manifest_statistics(manifest: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"splits": {}, "patient_overlap": patient_overlaps(manifest)}
    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        positives = int((subset["label"] == 1).sum())
        negatives = int((subset["label"] == 0).sum())
        stats["splits"][split] = {
            "samples": int(len(subset)),
            "patients": int(subset["patient_id"].nunique()),
            "positive": positives,
            "negative": negatives,
            "positive_fraction": positives / len(subset) if len(subset) else None,
            "negative_to_positive_ratio": negatives / positives if positives else None,
            "missing_images": int((~subset["image_exists"].astype(bool)).sum()),
        }
    return stats


def prepare_manifests(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = config["data"]
    manifest = build_manifest(
        data["train_csv"],
        data["test_csv"],
        data["data_root"],
        allowed_extensions=data.get("allowed_extensions", []),
        verify_images=bool(data.get("verify_images", True)),
        strict_missing_images=bool(data.get("strict_missing_images", True)),
        idc_metadata_csv=data.get("idc_metadata_csv") or None,
    )
    manifest = split_official_train_by_patient(
        manifest,
        val_fraction=float(data.get("val_fraction", 0.2)),
        seed=int(data.get("split_seed", 42)),
    )
    assert_no_patient_overlap(manifest)

    output_dir = Path(data.get("output_dir", "artifacts/data")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    for split in ("train", "val", "test"):
        manifest[manifest["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    stats = manifest_statistics(manifest)
    summary = {
        "protocol": {
            "subset": "calcification only",
            "image_source_column": "image file path",
            "sample_unit": "independent CC or MLO whole-mammogram view",
            "positive_label": "MALIGNANT",
            "negative_labels": ["BENIGN", "BENIGN_WITHOUT_CALLBACK"],
            "split_policy": "official test; official train split by patient_id",
            "reproduction_assumption": config["experiment"]["reproduction_assumption"],
        },
        "parameters": {
            "val_fraction": float(data.get("val_fraction", 0.2)),
            "split_seed": int(data.get("split_seed", 42)),
            "verify_images": bool(data.get("verify_images", True)),
        },
        "preprocessing": config["preprocessing"],
        "provenance": {
            "data_root": str(Path(data["data_root"]).expanduser().resolve()),
            "train_csv": str(Path(data["train_csv"]).expanduser().resolve()),
            "test_csv": str(Path(data["test_csv"]).expanduser().resolve()),
            "train_csv_sha256": _sha256_file(data["train_csv"]),
            "test_csv_sha256": _sha256_file(data["test_csv"]),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        **stats,
    }
    if data.get("idc_metadata_csv"):
        summary["provenance"]["idc_metadata_csv"] = str(
            Path(data["idc_metadata_csv"]).expanduser().resolve()
        )
        summary["provenance"]["idc_metadata_csv_sha256"] = _sha256_file(
            data["idc_metadata_csv"]
        )
        summary["idc_mapping"] = {
            "exact_series_uid_matches": int(len(manifest)),
            "missing_series_descriptions": int(
                (manifest["idc_series_description"].astype(str).str.strip() == "").sum()
            ),
            "forbidden_crop_or_roi_series": 0,
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return manifest, summary
