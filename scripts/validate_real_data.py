"""Decode every manifest DICOM and verify real-data tensor samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.config import load_config
from cbis_ddsm.dataset import MammogramDataset
from cbis_ddsm.indexing import assert_no_patient_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["image_path"])
    dataset = pydicom.dcmread(path)
    if str(dataset.SeriesInstanceUID) != str(row["series_instance_uid"]):
        raise ValueError(f"SeriesInstanceUID mismatch: {path}")
    if str(dataset.StudyInstanceUID) != str(row["study_instance_uid"]):
        raise ValueError(f"StudyInstanceUID mismatch: {path}")
    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"Expected 2D mammogram, got {pixels.shape}: {path}")
    return {
        "rows": int(pixels.shape[0]),
        "columns": int(pixels.shape[1]),
        "bits_allocated": int(getattr(dataset, "BitsAllocated", 0)),
        "photometric_interpretation": str(
            getattr(dataset, "PhotometricInterpretation", "UNKNOWN")
        ),
    }


def sample_indices(length: int) -> list[int]:
    if length <= 3:
        return list(range(length))
    return sorted({0, length // 3, (2 * length) // 3, length - 1})


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "sample_id": str})
    assert_no_patient_overlap(frame)
    records = frame.to_dict(orient="records")

    failures: list[dict[str, str]] = []
    decoded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_record = {executor.submit(validate_row, record): record for record in records}
        for completed, future in enumerate(as_completed(future_to_record), start=1):
            record = future_to_record[future]
            try:
                decoded.append(future.result())
            except Exception as exc:  # collect all corruptions in one audit report
                failures.append(
                    {
                        "sample_id": str(record["sample_id"]),
                        "image_path": str(record["image_path"]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if completed % 100 == 0 or completed == len(records):
                print(f"Decoded {completed}/{len(records)} DICOM files", flush=True)

    config = load_config(args.config)
    tensor_checks: list[dict[str, Any]] = []
    if not failures:
        for split in ("train", "val", "test"):
            dataset = MammogramDataset(
                frame,
                split=split,
                preprocessing=config["preprocessing"],
            )
            for index in sample_indices(len(dataset)):
                item = dataset[index]
                tensor_checks.append(
                    {
                        "split": split,
                        "sample_id": item["sample_id"],
                        "shape": list(item["image"].shape),
                        "dtype": str(item["image"].dtype),
                        "label_shape": list(item["label"].shape),
                    }
                )
                if list(item["image"].shape) != [3, 256, 256]:
                    raise ValueError(f"Unexpected tensor shape: {item['image'].shape}")

    summary = {
        "manifest_samples": int(len(frame)),
        "decoded_dicom": int(len(decoded)),
        "decode_failures": failures,
        "row_range": [min(x["rows"] for x in decoded), max(x["rows"] for x in decoded)]
        if decoded
        else None,
        "column_range": [
            min(x["columns"] for x in decoded),
            max(x["columns"] for x in decoded),
        ]
        if decoded
        else None,
        "bits_allocated": dict(Counter(str(x["bits_allocated"]) for x in decoded)),
        "photometric_interpretation": dict(
            Counter(x["photometric_interpretation"] for x in decoded)
        ),
        "tensor_checks": tensor_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "tensor_checks"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
