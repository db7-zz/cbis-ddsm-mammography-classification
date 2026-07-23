"""Create a deterministic, train/validation-only phase 5 preprocessing audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.dataset import load_mammogram
from cbis_ddsm.phase5 import load_phase5_experiment, verify_locked_training_manifests
from cbis_ddsm.preprocessing import apply_spatial_preprocessing, letterbox_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit phase 5 crop geometry without loading the official test split."
    )
    parser.add_argument(
        "--experiment-config",
        default=str(PROJECT_ROOT / "configs" / "phase5" / "P5-E01_seed42.yaml"),
    )
    parser.add_argument(
        "--manifest-dir", default=str(PROJECT_ROOT / "artifacts" / "data")
    )
    parser.add_argument(
        "--manifest-lock",
        default=str(PROJECT_ROOT / "configs" / "phase5" / "manifest_lock.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase5"
            / "preprocessing_audit_v1"
        ),
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Decode every locked train/validation image while saving panels only for the audit subset.",
    )
    return parser.parse_args()


def ensure_new_directory(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Audit output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def read_dimensions(path: str) -> tuple[int, int]:
    header = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        specific_tags=["Rows", "Columns"],
    )
    return int(header.Columns), int(header.Rows)


def select_audit_samples(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("sample_id").copy()
    coverage = (
        ordered.groupby(["split", "view", "laterality", "label"], sort=True)
        .head(1)
        .copy()
    )
    dimension_rows: list[pd.Series] = []
    for split, group in ordered.groupby("split", sort=True):
        ranked = group.sort_values(["pixel_count", "sample_id"])
        for index in (0, len(ranked) // 2, len(ranked) - 1):
            dimension_rows.append(ranked.iloc[index])
    dimensions = pd.DataFrame(dimension_rows)
    selected = pd.concat([coverage, dimensions], ignore_index=True)
    return selected.drop_duplicates("sample_id").sort_values("sample_id").reset_index(drop=True)


def preview(image: Image.Image, size: int = 512) -> Image.Image:
    return letterbox_image(image.convert("RGB"), size, fill=(0, 0, 0))


def compose_audit_image(
    original: Image.Image,
    crop_box: tuple[int, int, int, int],
    processed: Image.Image,
) -> Image.Image:
    overlay = original.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(max(original.size) / 500))
    draw.rectangle(crop_box, outline=(255, 64, 64), width=line_width)
    panels = [preview(original), preview(overlay), preview(processed)]
    canvas = Image.new("RGB", (512 * 3, 512), color=(0, 0, 0))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    return canvas


def main() -> int:
    args = parse_args()
    experiment = load_phase5_experiment(args.experiment_config)
    train, val, hashes = verify_locked_training_manifests(
        args.manifest_dir, args.manifest_lock
    )
    frame = pd.concat([train, val], ignore_index=True)
    dimensions = [read_dimensions(path) for path in frame["image_path"]]
    frame["original_width"] = [width for width, _ in dimensions]
    frame["original_height"] = [height for _, height in dimensions]
    frame["pixel_count"] = frame["original_width"] * frame["original_height"]
    selected = select_audit_samples(frame)
    selected_ids = set(selected["sample_id"])
    processing_frame = (
        frame.sort_values("sample_id").reset_index(drop=True)
        if args.scan_all
        else selected
    )
    output_dir = ensure_new_directory(Path(args.output_dir))
    images_dir = output_dir / "images"
    images_dir.mkdir()

    records: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for index, row in enumerate(processing_frame.to_dict(orient="records"), start=1):
        original = load_mammogram(
            row["image_path"],
            dicom_apply_modality_lut=bool(
                experiment.preprocessing.get("dicom_apply_modality_lut", True)
            ),
            dicom_apply_voi_lut=bool(
                experiment.preprocessing.get("dicom_apply_voi_lut", True)
            ),
            lower_percentile=float(
                experiment.preprocessing.get("lower_percentile", 0.5)
            ),
            upper_percentile=float(
                experiment.preprocessing.get("upper_percentile", 99.5)
            ),
        )
        result = apply_spatial_preprocessing(original, experiment.preprocessing)
        reason = result.fallback_reason or ""
        if result.fallback:
            reasons[reason or "unspecified"] += 1
        filename = ""
        if row["sample_id"] in selected_ids or (args.scan_all and result.fallback):
            filename = f"{row['split']}_{row['sample_id']}.png"
            compose_audit_image(original, result.crop_box, result.image).save(
                images_dir / filename
            )
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "view": row["view"],
                "laterality": row["laterality"],
                "label": int(row["label"]),
                "original_width": int(row["original_width"]),
                "original_height": int(row["original_height"]),
                "crop_left": result.crop_box[0],
                "crop_top": result.crop_box[1],
                "crop_right": result.crop_box[2],
                "crop_bottom": result.crop_box[3],
                "foreground_fraction": result.foreground_fraction,
                "used_crop": result.used_crop,
                "fallback": result.fallback,
                "fallback_reason": reason,
                "audit_image": str(Path("images") / filename) if filename else "",
            }
        )
        if args.scan_all and index % 100 == 0:
            print(f"scanned {index}/{len(processing_frame)}", flush=True)

    pd.DataFrame(records).to_csv(output_dir / "audit_records.csv", index=False)
    summary = {
        "experiment_id": experiment.experiment_id,
        "selected_samples": len(selected),
        "scanned_samples": len(records),
        "source_samples": len(frame),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "fallback_count": sum(reasons.values()),
        "fallback_reasons": dict(sorted(reasons.items())),
        "manifest_hashes": hashes,
        "test_split_loaded": False,
        "panel_order": ["original", "crop_box_overlay", "processed"],
        "manual_review_complete": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
