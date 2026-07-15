from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from cbis_ddsm.indexing import (
    DataIntegrityError,
    assert_no_patient_overlap,
    build_manifest,
    manifest_statistics,
    prepare_manifests,
    split_official_train_by_patient,
)


def _write_image(root: Path, relative_path: str, value: int = 100) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 18), color=value).save(path)


def _row(patient: str, image_path: str, pathology: str, view: str = "CC") -> dict[str, str]:
    return {
        "patient_id": patient,
        "left or right breast": "LEFT",
        "image view": view,
        "abnormality type": "calcification",
        "pathology": pathology,
        "image file path": image_path,
        "cropped image file path": "must/not/be/used.png",
        "ROI mask file path": "must/not/be/used-mask.png",
    }


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "images"
    train_rows: list[dict[str, str]] = []
    for patient_index in range(8):
        patient = f"P_{patient_index:05d}"
        pathology = "MALIGNANT" if patient_index % 2 else "BENIGN"
        for view in ("CC", "MLO"):
            relative = f"train/{patient}/{view}.png"
            _write_image(data_root, relative, value=40 + patient_index)
            train_rows.append(_row(patient, relative, pathology, view))

    # A second lesion row on the same image must be collapsed, with malignant winning.
    train_rows.append(_row("P_00000", "train/P_00000/CC.png", "MALIGNANT", "CC"))

    test_rows: list[dict[str, str]] = []
    for patient_index in range(2):
        patient = f"T_{patient_index:05d}"
        relative = f"test/{patient}/CC.png"
        _write_image(data_root, relative, value=150 + patient_index)
        pathology = "BENIGN_WITHOUT_CALLBACK" if patient_index == 0 else "MALIGNANT"
        test_rows.append(_row(patient, relative, pathology))

    train_csv = tmp_path / "calc_case_description_train_set.csv"
    test_csv = tmp_path / "calc_case_description_test_set.csv"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(test_rows).to_csv(test_csv, index=False)
    return data_root, train_csv, test_csv


def test_manifest_uses_whole_image_collapses_lesions_and_has_patient_safe_splits(
    tmp_path: Path,
) -> None:
    data_root, train_csv, test_csv = _source_files(tmp_path)
    manifest = build_manifest(train_csv, test_csv, data_root)
    assert len(manifest) == 18
    collapsed = manifest[
        (manifest["patient_id"] == "P_00000") & (manifest["view"] == "CC")
    ].iloc[0]
    assert collapsed["source_row_count"] == 2
    assert collapsed["label"] == 1
    assert collapsed["pathology_values"] == "BENIGN|MALIGNANT"
    assert "must/not/be/used" not in collapsed["image_path"]

    split = split_official_train_by_patient(manifest, val_fraction=0.25, seed=7)
    assert_no_patient_overlap(split)
    assert set(split.loc[split["official_split"] == "test", "split"]) == {"test"}
    assert set(split["split"]) == {"train", "val", "test"}
    stats = manifest_statistics(split)
    assert stats["patient_overlap"] == {"train_val": [], "train_test": [], "val_test": []}


def test_split_is_deterministic(tmp_path: Path) -> None:
    data_root, train_csv, test_csv = _source_files(tmp_path)
    manifest = build_manifest(train_csv, test_csv, data_root)
    first = split_official_train_by_patient(manifest, val_fraction=0.25, seed=42)
    second = split_official_train_by_patient(manifest, val_fraction=0.25, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_official_train_test_patient_overlap_is_fatal(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    _write_image(data_root, "train/a.png")
    _write_image(data_root, "test/b.png")
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame([_row("P_SHARED", "train/a.png", "BENIGN")]).to_csv(train_csv, index=False)
    pd.DataFrame([_row("P_SHARED", "test/b.png", "MALIGNANT")]).to_csv(test_csv, index=False)
    with pytest.raises(DataIntegrityError, match="overlap"):
        build_manifest(train_csv, test_csv, data_root)


def test_missing_whole_image_is_fatal_by_default(tmp_path: Path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame([_row("P_1", "missing/train.png", "BENIGN")]).to_csv(train_csv, index=False)
    pd.DataFrame([_row("T_1", "missing/test.png", "MALIGNANT")]).to_csv(test_csv, index=False)
    with pytest.raises(DataIntegrityError, match="does not exist"):
        build_manifest(train_csv, test_csv, data_root)


def test_prepare_manifests_writes_auditable_outputs(tmp_path: Path) -> None:
    data_root, train_csv, test_csv = _source_files(tmp_path)
    output_dir = tmp_path / "artifacts"
    config = {
        "experiment": {"reproduction_assumption": "test assumption"},
        "data": {
            "data_root": str(data_root),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "output_dir": str(output_dir),
            "val_fraction": 0.25,
            "split_seed": 42,
            "verify_images": True,
            "strict_missing_images": True,
            "allowed_extensions": [".png"],
        },
        "preprocessing": {"image_size": 256},
    }
    manifest, summary = prepare_manifests(config)
    assert len(manifest) == 18
    assert (output_dir / "manifest.csv").is_file()
    assert (output_dir / "train.csv").is_file()
    assert (output_dir / "val.csv").is_file()
    assert (output_dir / "test.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert summary["patient_overlap"] == {"train_val": [], "train_test": [], "val_test": []}
    assert len(summary["provenance"]["manifest_sha256"]) == 64


def test_idc_metadata_resolves_official_series_uid_to_default_hierarchy(tmp_path: Path) -> None:
    collection_root = tmp_path / "cbis_ddsm"
    study_uid = "1.2.3.4"
    train_series_uid = "1.2.3.4.5"
    test_series_uid = "1.2.3.4.6"
    train_patient = "Calc-Training_P_00001_LEFT_CC"
    test_patient = "Calc-Test_P_00002_RIGHT_MLO"
    for patient, series_uid in ((train_patient, train_series_uid), (test_patient, test_series_uid)):
        directory = collection_root / patient / study_uid / f"MG_{series_uid}"
        directory.mkdir(parents=True)
        (directory / f"{series_uid}.dcm").write_bytes(b"test")

    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "PatientID": train_patient,
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": train_series_uid,
                "Modality": "MG",
                "SeriesDescription": "full mammogram images",
            },
            {
                "PatientID": test_patient,
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": test_series_uid,
                "Modality": "MG",
                "SeriesDescription": "",
            },
        ]
    ).to_csv(metadata_csv, index=False)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        [
            _row(
                "P_00001",
                f"{train_patient}/{study_uid}/{train_series_uid}/000000.dcm",
                "BENIGN",
            )
        ]
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        [
            _row(
                "P_00002",
                f"{test_patient}/{study_uid}/{test_series_uid}/000000.dcm",
                "MALIGNANT",
                "MLO",
            )
        ]
    ).to_csv(test_csv, index=False)

    manifest = build_manifest(
        train_csv,
        test_csv,
        collection_root,
        idc_metadata_csv=metadata_csv,
    )
    assert set(manifest["series_instance_uid"]) == {train_series_uid, test_series_uid}
    assert manifest["image_exists"].all()
    assert set(Path(path).suffix for path in manifest["image_path"]) == {".dcm"}
