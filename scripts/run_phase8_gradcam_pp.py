"""Run formal Gate C Grad-CAM++ on frozen models and official-test images."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.dataset import build_transform, load_mammogram
from cbis_ddsm.evaluation import validate_split_manifest, write_json
from cbis_ddsm.gradcam_pp import (
    PHASE8_GRADCAM_MODEL_IDS,
    compute_localization_metrics,
    generate_gradcam_pp,
    load_gradcam_pp_config,
    select_phase8_target_layer,
    transform_binary_mask,
)
from cbis_ddsm.phase5 import load_phase5_experiment
from cbis_ddsm.phase7 import load_phase7_experiment
from cbis_ddsm.phase8 import (
    audit_phase8_artifacts,
    load_phase8_artifact_lock,
    load_phase8_frozen_model,
    load_phase8_official_test_policy,
)
from cbis_ddsm.phase8_xai import (
    ROI_AUDIT_PASS,
    clustered_mean_bootstrap,
    load_union_roi_mask,
    normalize_inference_preprocessing,
    paired_clustered_difference_bootstrap,
    validate_gradcam_authorization,
)
from cbis_ddsm.preprocessing import (
    apply_spatial_preprocessing,
    breast_foreground_mask,
)
from cbis_ddsm.training import set_global_seed, sha256_file


LOCALIZATION_COLUMNS = ("iou", "dice", "lesion_hit", "energy_inside_roi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-model formal phase 8 Grad-CAM++ comparison."
    )
    parser.add_argument(
        "--lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
    )
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def model_preprocessing(lock: Any, model_id: str) -> dict[str, Any]:
    spec = lock.models[model_id]
    source = PROJECT_ROOT / spec.config_path
    if spec.family == "phase5":
        return load_phase5_experiment(source).preprocessing
    if spec.family == "phase7":
        return load_phase7_experiment(source).preprocessing
    raise ValueError(f"Unsupported Gate C model family: {spec.family}")


def make_roi_overlay(image: np.ndarray, roi: np.ndarray | None) -> np.ndarray:
    result = image.copy()
    if roi is None or not roi.any():
        return result
    boundary = roi & ~ndimage.binary_erosion(roi)
    result[boundary] = np.array([0, 255, 0], dtype=np.uint8)
    return result


def colorize_heatmap(
    heatmap: np.ndarray,
    *,
    valid_mask: np.ndarray,
    colormap: str,
) -> np.ndarray:
    colored = matplotlib.colormaps[colormap](heatmap)[..., :3]
    output = np.round(colored * 255.0).astype(np.uint8)
    output[~valid_mask] = 0
    return output


def overlay_heatmap(
    image: np.ndarray,
    colored_heatmap: np.ndarray,
    *,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    result = image.copy()
    blended = np.round(
        image.astype(np.float32) * (1.0 - alpha)
        + colored_heatmap.astype(np.float32) * alpha
    ).astype(np.uint8)
    result[valid_mask] = blended[valid_mask]
    return result


def prepare_samples(
    test: pd.DataFrame,
    panel: pd.DataFrame,
    row_audit: pd.DataFrame,
    image_audit: pd.DataFrame,
    *,
    preprocessing: dict[str, Any],
    input_size: int,
) -> list[dict[str, Any]]:
    panel_ids = set(panel["sample_id"].astype(str))
    selected = test[
        (pd.to_numeric(test["label"], errors="raise").astype(int) == 1)
        | test["sample_id"].astype(str).isin(panel_ids)
    ].copy()
    transform = build_transform(
        train=False,
        image_size=input_size,
        imagenet_mean=tuple(preprocessing["imagenet_mean"]),
        imagenet_std=tuple(preprocessing["imagenet_std"]),
        resize=False,
    )
    prepared: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        mammogram = load_mammogram(
            row["image_path"],
            dicom_apply_modality_lut=bool(
                preprocessing["dicom_apply_modality_lut"]
            ),
            dicom_apply_voi_lut=bool(preprocessing["dicom_apply_voi_lut"]),
            lower_percentile=float(preprocessing["lower_percentile"]),
            upper_percentile=float(preprocessing["upper_percentile"]),
        )
        spatial = apply_spatial_preprocessing(mammogram, preprocessing)
        display = np.asarray(spatial.image.convert("RGB"), dtype=np.uint8)
        foreground = breast_foreground_mask(mammogram)
        valid_mask = transform_binary_mask(
            foreground,
            crop_box=spatial.crop_box,
            output_size=input_size,
        )
        if not valid_mask.any():
            raise ValueError(f"Valid breast mask is empty for {row['sample_id']}.")
        sample_id = str(row["sample_id"])
        geometry = image_audit[
            image_audit["sample_id"].astype(str) == sample_id
        ].iloc[0]
        roi_geometry_complete = bool(geometry["roi_geometry_complete"])
        roi_mask = None
        if roi_geometry_complete:
            raw_roi = load_union_roi_mask(
                row_audit,
                sample_id=sample_id,
                expected_shape=(mammogram.height, mammogram.width),
            )
            roi_mask = transform_binary_mask(
                raw_roi,
                crop_box=spatial.crop_box,
                output_size=input_size,
            )
            if not (roi_mask & valid_mask).any():
                raise ValueError(
                    f"Transformed ROI has no valid breast pixels for {sample_id}."
                )
        prepared.append(
            {
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                "view": str(row["view"]),
                "laterality": str(row["laterality"]),
                "image_path": str(row["image_path"]),
                "input": transform(spatial.image),
                "display": display,
                "valid_mask": valid_mask,
                "roi_mask": roi_mask,
                "roi_geometry_complete": roi_geometry_complete,
                "is_qualitative_panel": sample_id in panel_ids,
                "crop_box": spatial.crop_box,
                "preprocessing_fallback": spatial.fallback,
                "preprocessing_fallback_reason": spatial.fallback_reason,
            }
        )
    return prepared


def save_common_panel_artifacts(
    prepared: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    common = output_dir / "qualitative" / "common"
    common.mkdir(parents=True, exist_ok=False)
    for sample in prepared:
        if not sample["is_qualitative_panel"]:
            continue
        sample_id = sample["sample_id"]
        Image.fromarray(sample["display"], mode="RGB").save(
            common / f"{sample_id}_original.png"
        )
        roi = sample["roi_mask"]
        roi_image = (
            np.zeros_like(sample["valid_mask"], dtype=np.uint8)
            if roi is None
            else roi.astype(np.uint8) * 255
        )
        Image.fromarray(roi_image, mode="L").save(
            common / f"{sample_id}_roi_union_mask.png"
        )
        Image.fromarray(
            make_roi_overlay(sample["display"], roi),
            mode="RGB",
        ).save(common / f"{sample_id}_roi_overlay.png")


def formal_model_run(
    *,
    model_id: str,
    lock: Any,
    config: Any,
    prepared: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    model, model_audit = load_phase8_frozen_model(PROJECT_ROOT, lock, model_id)
    model = model.to(device)
    target_layer = select_phase8_target_layer(model_id, model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    heatmaps = np.zeros(
        (len(prepared), config.input_size, config.input_size),
        dtype=np.float32,
    )
    logits = np.zeros(len(prepared), dtype=np.float64)
    degenerate = np.zeros(len(prepared), dtype=bool)
    records: list[dict[str, Any]] = []
    qualitative_overlays: dict[str, np.ndarray] = {}
    model_visual_dir = output_dir / "qualitative" / model_id
    model_visual_dir.mkdir(parents=True, exist_ok=False)

    for start in range(0, len(prepared), config.batch_size):
        batch_samples = prepared[start : start + config.batch_size]
        images = torch.stack([sample["input"] for sample in batch_samples]).to(device)
        valid = torch.from_numpy(
            np.stack([sample["valid_mask"] for sample in batch_samples])
        ).to(device)
        result = generate_gradcam_pp(
            model,
            target_layer,
            images,
            valid_mask=valid,
        )
        expected_shape = (
            len(batch_samples),
            *config.expected_shapes[model_id],
        )
        if result.target_shape != expected_shape:
            raise RuntimeError(
                f"Gate C target shape mismatch for {model_id}: {result.target_shape}"
            )
        batch_heatmaps = result.heatmaps.cpu().numpy().astype(np.float32)
        batch_logits = result.logits.cpu().numpy().astype(np.float64)
        for offset, sample in enumerate(batch_samples):
            index = start + offset
            heatmap = batch_heatmaps[offset]
            is_degenerate = bool(result.degenerate[offset])
            heatmaps[index] = heatmap
            logits[index] = batch_logits[offset]
            degenerate[index] = is_degenerate
            record: dict[str, Any] = {
                "model_id": model_id,
                "sample_id": sample["sample_id"],
                "patient_id": sample["patient_id"],
                "label": sample["label"],
                "view": sample["view"],
                "laterality": sample["laterality"],
                "logit": logits[index],
                "degenerate_heatmap": is_degenerate,
                "roi_geometry_complete": sample["roi_geometry_complete"],
                "valid_pixels": int(sample["valid_mask"].sum()),
                "roi_pixels": (
                    int((sample["roi_mask"] & sample["valid_mask"]).sum())
                    if sample["roi_mask"] is not None
                    else 0
                ),
                "is_qualitative_panel": sample["is_qualitative_panel"],
            }
            if sample["label"] != 1:
                record["metric_status"] = "not_applicable_benign"
            elif not sample["roi_geometry_complete"]:
                record["metric_status"] = "not_comparable_roi_geometry"
            elif is_degenerate:
                record["metric_status"] = "degenerate_heatmap"
            else:
                metrics = compute_localization_metrics(
                    heatmap,
                    sample["roi_mask"],
                    valid_mask=sample["valid_mask"],
                    top_fraction=config.binary_top_fraction,
                )
                record.update(asdict(metrics))
                record["lesion_hit"] = float(metrics.lesion_hit)
                record["metric_status"] = "complete"
            records.append(record)

            if sample["is_qualitative_panel"]:
                colored = colorize_heatmap(
                    heatmap,
                    valid_mask=sample["valid_mask"],
                    colormap=config.visualization["colormap"],
                )
                overlay = overlay_heatmap(
                    sample["display"],
                    colored,
                    valid_mask=sample["valid_mask"],
                    alpha=float(config.visualization["overlay_alpha"]),
                )
                sample_id = sample["sample_id"]
                Image.fromarray(colored, mode="RGB").save(
                    model_visual_dir / f"{sample_id}_heatmap.png"
                )
                Image.fromarray(overlay, mode="RGB").save(
                    model_visual_dir / f"{sample_id}_overlay.png"
                )
                qualitative_overlays[sample_id] = overlay
        del result, images, valid

    arrays_path = output_dir / f"{model_id}_heatmaps.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray([sample["sample_id"] for sample in prepared]),
        logits=logits,
        heatmaps=heatmaps,
        degenerate=degenerate,
    )
    frame = pd.DataFrame.from_records(records)
    complete = frame[frame["metric_status"] == "complete"].copy().reset_index(drop=True)
    bootstrap_settings = config.bootstrap
    bootstrap = (
        clustered_mean_bootstrap(
            complete,
            value_columns=LOCALIZATION_COLUMNS,
            valid_replicates=int(bootstrap_settings["valid_replicates"]),
            seed=int(bootstrap_settings["seed"]),
            confidence_level=float(bootstrap_settings["confidence_level"]),
        )
        if not complete.empty
        else None
    )
    summary = {
        **model_audit,
        "test_split_loaded": True,
        "official_test_images_loaded": True,
        "target_layer": config.target_layers[model_id],
        "expected_target_shape_per_image": list(config.expected_shapes[model_id]),
        "formal_images": len(prepared),
        "malignant_images": int((frame["label"] == 1).sum()),
        "complete_localization_images": int(len(complete)),
        "degenerate_heatmaps": int(frame["degenerate_heatmap"].sum()),
        "geometry_excluded_images": int(
            (frame["metric_status"] == "not_comparable_roi_geometry").sum()
        ),
        "qualitative_panel_images": int(frame["is_qualitative_panel"].sum()),
        "machine_arrays_path": str(arrays_path),
        "machine_arrays_sha256": sha256_file(arrays_path),
        "clustered_bootstrap": bootstrap,
        "peak_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else None
        ),
        "precision": "float32",
        "amp_enabled": False,
    }
    model.to("cpu")
    del target_layer, model, heatmaps, logits, degenerate
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame, summary, qualitative_overlays


def write_master_panel(
    *,
    panel: pd.DataFrame,
    prepared: list[dict[str, Any]],
    overlays: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> Path:
    prepared_by_id = {sample["sample_id"]: sample for sample in prepared}
    figure, axes = plt.subplots(len(panel), 5, figsize=(15, 36))
    model_columns = list(PHASE8_GRADCAM_MODEL_IDS)
    for row_index, (_, panel_row) in enumerate(panel.iterrows()):
        sample_id = str(panel_row["sample_id"])
        sample = prepared_by_id[sample_id]
        images = [
            sample["display"],
            make_roi_overlay(sample["display"], sample["roi_mask"]),
            *[overlays[model_id][sample_id] for model_id in model_columns],
        ]
        for column_index, image in enumerate(images):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{panel_row['panel_stratum']} #{int(panel_row['panel_rank'])}\n{sample_id}",
            fontsize=8,
        )
    titles = [
        "Original",
        "Union ROI",
        "P5-E02 B0",
        "P7-E01 V2S",
        "P7-E04 V2S+BiFPN+CBAM",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=10)
    figure.suptitle(
        "Gate C fixed qualitative panel — malignant-logit Grad-CAM++",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.995))
    path = output_dir / "qualitative" / "qualitative_panel.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    lock = load_phase8_artifact_lock(args.lock)
    config = load_gradcam_pp_config(args.config)
    policy = load_phase8_official_test_policy(args.policy)
    authorization = validate_gradcam_authorization(args.authorization)
    audit_phase8_artifacts(PROJECT_ROOT, lock)
    test_path = Path(args.test_manifest).expanduser().resolve()
    if sha256_file(test_path) != policy.expected_test_manifest_sha256:
        raise ValueError("Formal Gate C test manifest hash mismatch.")
    test = pd.read_csv(test_path, dtype={"patient_id": str, "sample_id": str})
    validate_split_manifest(
        test,
        expected_split="test",
        expected_samples=policy.expected_test_samples,
    )

    geometry_dir = (PROJECT_ROOT / config.geometry_audit_output_dir).resolve()
    geometry_outcome_path = geometry_dir / "outcome.json"
    with geometry_outcome_path.open("r", encoding="utf-8") as handle:
        geometry_outcome = json.load(handle)
    if (
        geometry_outcome.get("status") != "PASS_WITH_DECLARED_EXCLUSIONS"
        or geometry_outcome.get("test_manifest_sha256")
        != policy.expected_test_manifest_sha256
        or geometry_outcome.get("formal_gradcam_generated") is not False
    ):
        raise PermissionError("Formal Gate C requires the passed geometry audit.")
    row_audit_path = Path(geometry_outcome["row_audit_path"])
    image_audit_path = Path(geometry_outcome["image_audit_path"])
    panel_path = Path(geometry_outcome["qualitative_selection_path"])
    if (
        sha256_file(row_audit_path) != geometry_outcome["row_audit_sha256"]
        or sha256_file(image_audit_path) != geometry_outcome["image_audit_sha256"]
        or sha256_file(panel_path) != geometry_outcome["qualitative_selection_sha256"]
    ):
        raise ValueError("Gate C geometry audit artifacts changed.")
    row_audit = pd.read_csv(row_audit_path, dtype={"patient_id": str, "sample_id": str})
    image_audit = pd.read_csv(
        image_audit_path,
        dtype={"patient_id": str, "sample_id": str},
    )
    panel = pd.read_csv(panel_path, dtype={"patient_id": str, "sample_id": str})

    preprocessings = {
        model_id: normalize_inference_preprocessing(
            model_preprocessing(lock, model_id)
        )
        for model_id in PHASE8_GRADCAM_MODEL_IDS
    }
    preprocessing = preprocessings[PHASE8_GRADCAM_MODEL_IDS[0]]
    if any(value != preprocessing for value in preprocessings.values()):
        raise ValueError("Gate C models do not share the locked preprocessing.")
    prepared = prepare_samples(
        test,
        panel,
        row_audit,
        image_audit,
        preprocessing=preprocessing,
        input_size=config.input_size,
    )
    if (
        len(prepared) != 125
        or sum(sample["label"] == 1 for sample in prepared) != 119
        or sum(sample["is_qualitative_panel"] for sample in prepared) != 12
    ):
        raise ValueError("Gate C formal image selection changed.")

    output_dir = (PROJECT_ROOT / config.formal_output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Formal Gate C output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    device = select_device(args.device)
    set_global_seed(42, deterministic=True)
    preflight = {
        "format_version": 1,
        "stage": 8,
        "gate": "C",
        "status": "formal_gradcam_started",
        "started_at_utc": started_at,
        "device": str(device),
        "models": list(PHASE8_GRADCAM_MODEL_IDS),
        "formal_images": len(prepared),
        "malignant_images": 119,
        "eligible_malignant_localization_images": 117,
        "geometry_excluded_malignant_images": 2,
        "qualitative_panel_images": 12,
        "batch_size": config.batch_size,
        "precision": "float32",
        "amp_enabled": False,
        "authorization_sha256": sha256_file(Path(args.authorization)),
        "geometry_outcome_sha256": sha256_file(geometry_outcome_path),
        "classification_test_inference_passes_added": 0,
        "training_performed": False,
    }
    write_json(output_dir / "started.json", preflight)
    save_common_panel_artifacts(prepared, output_dir)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

    try:
        all_frames: list[pd.DataFrame] = []
        model_summaries: dict[str, Any] = {}
        all_overlays: dict[str, dict[str, np.ndarray]] = {}
        for model_id in PHASE8_GRADCAM_MODEL_IDS:
            frame, summary, overlays = formal_model_run(
                model_id=model_id,
                lock=lock,
                config=config,
                prepared=prepared,
                output_dir=output_dir,
                device=device,
            )
            all_frames.append(frame)
            model_summaries[model_id] = summary
            all_overlays[model_id] = overlays
            print(json.dumps({model_id: summary}, ensure_ascii=False, indent=2), flush=True)

        metrics_frame = pd.concat(all_frames, ignore_index=True)
        metrics_path = output_dir / "per_image_results.csv"
        metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8")
        complete_by_model = {
            model_id: metrics_frame[
                (metrics_frame["model_id"] == model_id)
                & (metrics_frame["metric_status"] == "complete")
            ].copy()
            for model_id in PHASE8_GRADCAM_MODEL_IDS
        }
        comparisons = {
            "P7-E01_minus_P5-E02": paired_clustered_difference_bootstrap(
                complete_by_model["P7-E01_V2S_seed42"],
                complete_by_model["P5-E02_seed42"],
                first_model_id="P7-E01_V2S_seed42",
                second_model_id="P5-E02_seed42",
                value_columns=LOCALIZATION_COLUMNS,
                valid_replicates=int(config.bootstrap["valid_replicates"]),
                seed=int(config.bootstrap["seed"]),
                confidence_level=float(config.bootstrap["confidence_level"]),
            ),
            "P7-E01_minus_P7-E04": paired_clustered_difference_bootstrap(
                complete_by_model["P7-E01_V2S_seed42"],
                complete_by_model["P7-E04_V2S_BiFPN_CBAM_seed42"],
                first_model_id="P7-E01_V2S_seed42",
                second_model_id="P7-E04_V2S_BiFPN_CBAM_seed42",
                value_columns=LOCALIZATION_COLUMNS,
                valid_replicates=int(config.bootstrap["valid_replicates"]),
                seed=int(config.bootstrap["seed"]),
                confidence_level=float(config.bootstrap["confidence_level"]),
            ),
        }
        comparison_path = output_dir / "paired_localization_comparisons.json"
        write_json(comparison_path, comparisons)
        panel_output = write_master_panel(
            panel=panel,
            prepared=prepared,
            overlays=all_overlays,
            output_dir=output_dir,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        outcome = {
            **preflight,
            "status": "complete",
            "completed_at_utc": completed_at,
            "model_summaries": model_summaries,
            "paired_localization_comparisons": comparisons,
            "per_image_results_path": str(metrics_path),
            "per_image_results_sha256": sha256_file(metrics_path),
            "paired_comparisons_path": str(comparison_path),
            "paired_comparisons_sha256": sha256_file(comparison_path),
            "qualitative_panel_path": str(panel_output),
            "qualitative_panel_sha256": sha256_file(panel_output),
            "gradcam_image_model_pairs": len(prepared)
            * len(PHASE8_GRADCAM_MODEL_IDS),
            "gradcam_backward_batches": math.ceil(len(prepared) / config.batch_size)
            * len(PHASE8_GRADCAM_MODEL_IDS),
            "official_test_images_loaded": True,
            "official_test_roi_masks_loaded": True,
            "formal_localization_metrics_claimed": True,
            "geometry_exclusions_preserved": True,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "git_status_short": git_value("status", "--short"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "gate_d_loaded": False,
        }
        write_json(output_dir / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                **preflight,
                "status": "failed_no_automatic_rerun",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
