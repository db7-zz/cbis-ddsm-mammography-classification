"""Run Gate D route 1 as an exploratory INbreast BI-RADS transfer."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.evaluation import (
    build_test_dataloader,
    compute_evaluation_metrics,
    load_evaluation_config,
    read_frozen_threshold,
    run_inference,
    write_json,
)
from cbis_ddsm.metrics import sigmoid_numpy
from cbis_ddsm.phase7 import load_phase7_experiment
from cbis_ddsm.phase8 import (
    audit_phase8_artifacts,
    load_phase8_artifact_lock,
    load_phase8_frozen_model,
    load_phase8_official_test_policy,
    paired_patient_bootstrap,
)
from cbis_ddsm.phase8_external import (
    load_inbreast_external_policy,
    load_locked_inbreast_candidate_manifest,
    patient_cluster_metric_bootstrap,
    validate_inbreast_route_authorization,
)
from cbis_ddsm.training import set_global_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run P7-E01 and P7-E04 once on the locked exploratory INbreast "
            "BI-RADS transfer task."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--external-policy",
        default=str(
            PROJECT_ROOT / "configs" / "phase8" / "inbreast_external.yaml"
        ),
    )
    parser.add_argument(
        "--artifact-lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
    )
    parser.add_argument(
        "--official-policy",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "official_test.yaml"),
    )
    parser.add_argument(
        "--route-authorization",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "experiments"
            / "phase8"
            / "external"
            / "inbreast_route1_authorization.json"
        ),
    )
    parser.add_argument(
        "--evaluation-config",
        default=str(PROJECT_ROOT / "configs" / "eval.yaml"),
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


def locked_threshold(model_id: str, official_policy: Any) -> tuple[float, Path, str]:
    path = (
        PROJECT_ROOT
        / official_policy.threshold_output_dirs[model_id]
        / "threshold.json"
    )
    document = read_frozen_threshold(path)
    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Frozen validation threshold is missing for {model_id}.")
    threshold = float(selection["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Frozen validation threshold is invalid for {model_id}.")
    if document.get("selection_split") != "validation":
        raise ValueError(f"Threshold source changed for {model_id}.")
    return threshold, path, sha256_file(path)


def save_external_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: Any,
    output_dir: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC AUC = {metrics.roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="INbreast exploratory BI-RADS ROC",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"PR AUC = {metrics.pr_auc:.4f}")
    axis.axhline(labels.mean(), linestyle="--", color="gray", label="Prevalence")
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="INbreast exploratory BI-RADS precision-recall",
    )
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(labels, probabilities >= metrics.threshold, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
            )
    axis.set_xticks([0, 1], labels=["BI-RADS 1-3", "BI-RADS 4a-6"])
    axis.set_yticks([0, 1], labels=["BI-RADS 1-3", "BI-RADS 4a-6"])
    axis.set(
        xlabel="Predicted",
        ylabel="Mapped BI-RADS target",
        title=f"Frozen validation threshold (t={metrics.threshold:.4f})",
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def run_one_model(
    *,
    model_id: str,
    output_dir: Path,
    lock: Any,
    external_policy: Any,
    official_policy: Any,
    manifest: pd.DataFrame,
    preprocessing: dict[str, Any],
    evaluation_config: Any,
    device: torch.device,
    route_authorization_sha256: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=False, exist_ok=False)
    threshold, threshold_path, threshold_sha = locked_threshold(
        model_id,
        official_policy,
    )
    started = {
        "format_version": 1,
        "stage": 8,
        "gate": "D",
        "route": "exploratory_birads_transfer",
        "status": "started",
        "model_id": model_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_samples": len(manifest),
        "external_patients": int(manifest["patient_id"].nunique()),
        "threshold_source": "CBIS-DDSM_validation_youden_j_frozen",
        "threshold": threshold,
        "threshold_sha256": threshold_sha,
        "route_authorization_sha256": route_authorization_sha256,
        "official_test_loaded": False,
        "test_split_loaded": False,
    }
    write_json(output_dir / "external_evaluation_started.json", started)
    try:
        model, model_audit = load_phase8_frozen_model(
            PROJECT_ROOT,
            lock,
            model_id,
        )
        loader = build_test_dataloader(
            manifest,
            preprocessing=preprocessing,
            config=evaluation_config,
        )
        inference = run_inference(
            model,
            loader,
            device,
            mixed_precision=evaluation_config.mixed_precision,
        )
        expected_ids = manifest["sample_id"].astype(str).tolist()
        if list(inference.sample_ids) != expected_ids:
            raise RuntimeError("Gate D external sample order changed.")
        probabilities = sigmoid_numpy(inference.logits)
        predicted_labels = (probabilities >= threshold).astype(np.int64)
        predictions = manifest[
            [
                "sample_id",
                "patient_id",
                "file_id",
                "laterality",
                "view",
                "birads",
                "candidate_label_source",
                "image_path",
            ]
        ].copy()
        predictions["label"] = inference.labels
        predictions["logit"] = inference.logits
        predictions["probability"] = probabilities
        predictions["predicted_label"] = predicted_labels
        predictions["threshold"] = threshold
        predictions["route"] = "exploratory_birads_transfer"
        predictions["pathology_label_equivalence_claimed"] = False
        predictions_path = output_dir / "external_predictions.csv"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8")

        metrics = compute_evaluation_metrics(
            inference.labels,
            probabilities,
            threshold,
        )
        bootstrap = patient_cluster_metric_bootstrap(
            predictions,
            threshold=threshold,
            valid_replicates=int(
                external_policy.bootstrap["valid_replicates"]
            ),
            seed=int(external_policy.bootstrap["seed"]),
            confidence_level=float(
                external_policy.bootstrap["confidence_level"]
            ),
        )
        write_json(output_dir / "patient_bootstrap.json", bootstrap)
        save_external_plots(
            inference.labels,
            probabilities,
            metrics,
            output_dir,
        )
        metrics_payload = {
            "format_version": 1,
            "stage": 8,
            "gate": "D",
            "route": "exploratory_birads_transfer",
            "claim_scope": (
                "exploratory INbreast BI-RADS task transfer; not pathology-equivalent "
                "external validation"
            ),
            "model_id": model_id,
            "split": "external_inbreast",
            "label_mapping": {
                "negative": list(external_policy.negative_birads),
                "positive": list(external_policy.positive_birads),
            },
            "threshold_source": "CBIS-DDSM_validation_youden_j_frozen",
            "metrics": asdict(metrics),
            "patient_bootstrap": bootstrap,
            "audit": {
                **started,
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "checkpoint_sha256": lock.models[model_id].checkpoint_sha256,
                "predictions_sha256": sha256_file(predictions_path),
                "artifact_lock_sha256": sha256_file(lock.source_path),
                "external_policy_sha256": sha256_file(
                    external_policy.source_path
                ),
                "git_commit": git_value("rev-parse", "HEAD"),
                "git_branch": git_value("branch", "--show-current"),
                "git_status_short": git_value("status", "--short"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
                "strict_checkpoint_load": model_audit["strict_load_pass"],
                "external_inference_passes": 1,
                "official_test_inference_passes_added": 0,
                "training_performed": False,
                "fine_tuned": False,
                "threshold_refit": False,
                "test_time_augmentation": False,
                "ensemble": False,
            },
        }
        write_json(output_dir / "metrics.json", metrics_payload)
        outcome = {
            "format_version": 1,
            "stage": 8,
            "gate": "D",
            "status": "exploratory_external_evaluation_complete",
            "model_id": model_id,
            "route": "exploratory_birads_transfer",
            "pathology_label_equivalence_claimed": False,
            "threshold": threshold,
            "threshold_path": str(threshold_path),
            "external_samples": len(predictions),
            "external_patients": int(predictions["patient_id"].nunique()),
            "external_inference_passes": 1,
            "official_test_loaded": False,
            "test_split_loaded": False,
            "metrics": asdict(metrics),
            "artifacts": {
                "predictions": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "metrics": str(output_dir / "metrics.json"),
                "patient_bootstrap": str(
                    output_dir / "patient_bootstrap.json"
                ),
            },
        }
        write_json(output_dir / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
        return predictions
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                **started,
                "status": "failed_no_automatic_rerun",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "reason": str(error),
                "raw_data_mutated": False,
                "official_test_loaded": False,
                "test_split_loaded": False,
            },
        )
        raise
    finally:
        if "model" in locals():
            model.to("cpu")
            del model
        if "loader" in locals():
            del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    external_policy = load_inbreast_external_policy(args.external_policy)
    lock = load_phase8_artifact_lock(args.artifact_lock)
    official_policy = load_phase8_official_test_policy(args.official_policy)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    route_authorization = validate_inbreast_route_authorization(
        args.route_authorization,
        dataset_root=dataset_root,
        policy=external_policy,
    )
    route_authorization_sha = sha256_file(Path(args.route_authorization))
    if external_policy.authorized_model_ids != official_policy.authorized_model_ids:
        raise ValueError("Gate D and Gate B model order must match.")
    audit_phase8_artifacts(PROJECT_ROOT, lock)
    manifest = load_locked_inbreast_candidate_manifest(
        PROJECT_ROOT,
        external_policy,
    )
    if {
        Path(path).resolve().parents[1]
        for path in manifest["image_path"]
    } != {dataset_root / external_policy.layout["raw_subdir"]}:
        raise ValueError("Gate D candidate paths do not belong to the authorized root.")

    train = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "data" / "train.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    validation = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "data" / "val.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    external_patients = set(manifest["patient_id"].astype(str))
    external_samples = set(manifest["sample_id"].astype(str))
    if external_patients.intersection(train["patient_id"].astype(str)):
        raise ValueError("INbreast patient IDs overlap CBIS-DDSM train.")
    if external_patients.intersection(validation["patient_id"].astype(str)):
        raise ValueError("INbreast patient IDs overlap CBIS-DDSM validation.")
    if external_samples.intersection(train["sample_id"].astype(str)):
        raise ValueError("INbreast sample IDs overlap CBIS-DDSM train.")
    if external_samples.intersection(validation["sample_id"].astype(str)):
        raise ValueError("INbreast sample IDs overlap CBIS-DDSM validation.")

    experiments = {
        model_id: load_phase7_experiment(
            PROJECT_ROOT / lock.models[model_id].config_path
        )
        for model_id in external_policy.authorized_model_ids
    }
    preprocessing = experiments[
        external_policy.authorized_model_ids[0]
    ].preprocessing
    if any(
        experiment.preprocessing != preprocessing
        for experiment in experiments.values()
    ):
        raise ValueError("Gate D models do not share frozen preprocessing.")

    for model_id in external_policy.authorized_model_ids:
        locked_threshold(model_id, official_policy)
    output_root = (PROJECT_ROOT / external_policy.formal_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Gate D formal output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    base_evaluation = load_evaluation_config(args.evaluation_config)
    evaluation_config = replace(
        base_evaluation,
        seed=int(external_policy.runtime["seed"]),
        batch_size=int(external_policy.runtime["batch_size"]),
        num_workers=int(external_policy.runtime["num_workers"]),
        pin_memory=bool(external_policy.runtime["pin_memory"]),
        persistent_workers=bool(
            external_policy.runtime["persistent_workers"]
        ),
        prefetch_factor=int(external_policy.runtime["prefetch_factor"]),
        mixed_precision=bool(external_policy.runtime["mixed_precision"]),
        deterministic=bool(external_policy.runtime["deterministic"]),
        expected_test_samples=len(manifest),
        output_dir=external_policy.formal_dir,
    )
    device = select_device(args.device)
    set_global_seed(evaluation_config.seed, deterministic=True)
    preflight = {
        "format_version": 1,
        "stage": 8,
        "gate": "D",
        "status": "route1_preflight_pass",
        "route": route_authorization["selected_route"],
        "claim_scope": "exploratory_birads_transfer_not_pathology_equivalent",
        "models": list(external_policy.authorized_model_ids),
        "external_samples": len(manifest),
        "external_patients": int(manifest["patient_id"].nunique()),
        "mapped_negatives": int((manifest["label"] == 0).sum()),
        "mapped_positives": int((manifest["label"] == 1).sum()),
        "device": str(device),
        "official_test_loaded": False,
        "test_split_loaded": False,
        "training_performed": False,
        "threshold_refit": False,
    }
    write_json(output_root / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

    predictions: list[pd.DataFrame] = []
    for model_id, directory_name in zip(
        external_policy.authorized_model_ids,
        ("P7-E01_seed42", "P7-E04_seed42"),
    ):
        predictions.append(
            run_one_model(
                model_id=model_id,
                output_dir=output_root / directory_name,
                lock=lock,
                external_policy=external_policy,
                official_policy=official_policy,
                manifest=manifest,
                preprocessing=preprocessing,
                evaluation_config=evaluation_config,
                device=device,
                route_authorization_sha256=route_authorization_sha,
            )
        )

    paired_dir = output_root / "paired_comparison"
    paired_dir.mkdir(parents=False, exist_ok=False)
    paired = paired_patient_bootstrap(
        predictions[0],
        predictions[1],
        first_model_id=external_policy.authorized_model_ids[0],
        second_model_id=external_policy.authorized_model_ids[1],
        valid_replicates=int(external_policy.bootstrap["valid_replicates"]),
        seed=int(external_policy.bootstrap["seed"]),
        confidence_level=float(external_policy.bootstrap["confidence_level"]),
    )
    write_json(paired_dir / "paired_bootstrap.json", paired)
    comparison_rows = []
    for model_id, frame in zip(
        external_policy.authorized_model_ids,
        predictions,
    ):
        threshold, _, _ = locked_threshold(model_id, official_policy)
        metrics = compute_evaluation_metrics(
            frame["label"].to_numpy(np.int64),
            frame["probability"].to_numpy(np.float64),
            threshold,
        )
        comparison_rows.append(
            {
                "model_id": model_id,
                "role": lock.models[model_id].role,
                "route": "exploratory_birads_transfer",
                "pathology_label_equivalence_claimed": False,
                **asdict(metrics),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = paired_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    summary = {
        "format_version": 1,
        "stage": 8,
        "gate": "D",
        "status": "gate_d_route1_complete",
        "route": "exploratory_birads_transfer",
        "claim_scope": (
            "exploratory INbreast BI-RADS task transfer; not pathology-equivalent "
            "external validation"
        ),
        "external_samples": len(manifest),
        "external_patients": int(manifest["patient_id"].nunique()),
        "models": comparison_rows,
        "paired_bootstrap": paired,
        "external_inference_passes": 2,
        "official_test_inference_passes_added": 0,
        "official_test_loaded": False,
        "test_split_loaded": False,
        "training_performed": False,
        "fine_tuned": False,
        "threshold_refit": False,
        "test_time_augmentation": False,
        "ensemble": False,
        "raw_data_mutated": False,
        "artifacts": {
            "comparison": str(comparison_path),
            "comparison_sha256": sha256_file(comparison_path),
            "paired_bootstrap": str(paired_dir / "paired_bootstrap.json"),
            "paired_bootstrap_sha256": sha256_file(
                paired_dir / "paired_bootstrap.json"
            ),
        },
    }
    write_json(output_root / "outcome.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
