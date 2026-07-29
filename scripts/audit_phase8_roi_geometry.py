"""Audit CBIS-DDSM whole-mammogram/ROI geometry before formal Gate C metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.dataset import load_mammogram
from cbis_ddsm.evaluation import validate_split_manifest, write_json
from cbis_ddsm.gradcam_pp import (
    load_gradcam_pp_config,
    select_qualitative_samples,
    transform_binary_mask,
)
from cbis_ddsm.phase8 import load_phase8_official_test_policy
from cbis_ddsm.phase8_xai import (
    audit_roi_geometry,
    load_union_roi_mask,
    validate_gradcam_authorization,
)
from cbis_ddsm.preprocessing import apply_spatial_preprocessing
from cbis_ddsm.training import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase 8 ROI geometry gate.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "gradcam_pp.yaml"),
    )
    parser.add_argument(
        "--policy",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "official_test.yaml"),
    )
    parser.add_argument(
        "--authorization",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase8"
            / "gradcam_pp_authorization.json"
        ),
    )
    parser.add_argument(
        "--test-manifest",
        default=str(PROJECT_ROOT / "artifacts" / "data" / "test.csv"),
    )
    parser.add_argument("--official-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_gradcam_pp_config(args.config)
    policy = load_phase8_official_test_policy(args.policy)
    authorization = validate_gradcam_authorization(args.authorization)
    test_path = Path(args.test_manifest).expanduser().resolve()
    if sha256_file(test_path) != policy.expected_test_manifest_sha256:
        raise ValueError("Gate C official test manifest hash mismatch.")
    test = pd.read_csv(test_path, dtype={"patient_id": str, "sample_id": str})
    test.attrs["source_path"] = str(test_path)
    validate_split_manifest(
        test,
        expected_split="test",
        expected_samples=policy.expected_test_samples,
    )
    output_dir = (PROJECT_ROOT / config.geometry_audit_output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Geometry audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    row_audit, image_audit, summary = audit_roi_geometry(
        test,
        official_csv_path=args.official_csv,
        metadata_csv_path=args.metadata_csv,
        data_root=args.data_root,
    )
    row_path = output_dir / "roi_row_audit.csv"
    image_path = output_dir / "roi_image_audit.csv"
    row_audit.to_csv(row_path, index=False, encoding="utf-8")
    image_audit.to_csv(image_path, index=False, encoding="utf-8")

    panel = select_qualitative_samples(
        test,
        samples_per_stratum=int(config.qualitative_panel["samples_per_stratum"]),
    )
    panel_path = output_dir / "qualitative_panel_selection.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8")

    preprocessing = {
        "image_size": 384,
        "dicom_apply_modality_lut": True,
        "dicom_apply_voi_lut": True,
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
        "spatial_mode": "breast_crop_letterbox",
        "breast_margin_fraction": 0.02,
        "min_foreground_fraction": 0.15,
        "max_foreground_fraction": 0.95,
        "min_breast_width_fraction": 0.25,
        "min_breast_height_fraction": 0.25,
    }
    figure, axes = plt.subplots(4, 3, figsize=(12, 16))
    selected_audit: list[dict[str, object]] = []
    for axis, (_, sample) in zip(axes.flat, panel.iterrows()):
        mammogram = load_mammogram(
            sample["image_path"],
            dicom_apply_modality_lut=True,
            dicom_apply_voi_lut=True,
            lower_percentile=0.5,
            upper_percentile=99.5,
        )
        spatial = apply_spatial_preprocessing(mammogram, preprocessing)
        display = np.asarray(spatial.image.convert("RGB"), dtype=np.uint8)
        sample_id = str(sample["sample_id"])
        geometry = image_audit[
            image_audit["sample_id"].astype(str) == sample_id
        ].iloc[0]
        roi_available = bool(geometry["roi_geometry_complete"])
        roi_pixels = 0
        if roi_available:
            raw_roi = load_union_roi_mask(
                row_audit,
                sample_id=sample_id,
                expected_shape=(mammogram.height, mammogram.width),
            )
            roi = transform_binary_mask(
                raw_roi,
                crop_box=spatial.crop_box,
                output_size=config.input_size,
            )
            roi_pixels = int(roi.sum())
            axis.contour(roi.astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=1.2)
        axis.imshow(display)
        axis.set_title(
            f"{sample['panel_stratum']} #{int(sample['panel_rank'])}\n"
            f"{sample_id} | ROI={'pass' if roi_available else 'excluded'}",
            fontsize=8,
        )
        axis.axis("off")
        selected_audit.append(
            {
                "sample_id": sample_id,
                "panel_stratum": sample["panel_stratum"],
                "panel_rank": int(sample["panel_rank"]),
                "roi_geometry_complete": roi_available,
                "transformed_roi_pixels": roi_pixels,
                "crop_box": list(spatial.crop_box),
                "preprocessing_fallback": spatial.fallback,
                "preprocessing_fallback_reason": spatial.fallback_reason,
            }
        )
    figure.suptitle(
        "Gate C ROI geometry audit — green contour is the transformed union ROI",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    audit_panel_path = output_dir / "roi_geometry_audit_panel.png"
    figure.savefig(audit_panel_path, dpi=160)
    plt.close(figure)

    summary.update(
        {
            "authorization_sha256": sha256_file(Path(args.authorization)),
            "gradcam_config_sha256": sha256_file(config.source_path),
            "official_test_policy_sha256": sha256_file(policy.source_path),
            "row_audit_path": str(row_path),
            "row_audit_sha256": sha256_file(row_path),
            "image_audit_path": str(image_path),
            "image_audit_sha256": sha256_file(image_path),
            "qualitative_selection_path": str(panel_path),
            "qualitative_selection_sha256": sha256_file(panel_path),
            "visual_audit_panel_path": str(audit_panel_path),
            "visual_audit_panel_sha256": sha256_file(audit_panel_path),
            "selected_visual_audit": selected_audit,
            "official_test_images_loaded": True,
            "official_test_roi_masks_loaded": True,
            "classification_test_inference_passes_added": 0,
            "formal_gradcam_generated": False,
            "training_performed": False,
            "gate_d_loaded": False,
            "approved_by_user": authorization["approved_by_user"],
        }
    )
    write_json(output_dir / "outcome.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
