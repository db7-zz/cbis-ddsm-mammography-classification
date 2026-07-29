"""Run the two authorized phase 8 official-test inference passes exactly once."""

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

import numpy as np
import pandas as pd
import torch

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
    save_evaluation_plots,
    validate_no_patient_or_sample_overlap,
    validate_split_manifest,
    write_json,
)
from cbis_ddsm.metrics import sigmoid_numpy
from cbis_ddsm.phase7 import load_phase7_experiment
from cbis_ddsm.phase8 import (
    audit_phase8_artifacts,
    ensure_phase8_official_test_outputs_absent,
    load_phase8_artifact_lock,
    load_phase8_frozen_model,
    load_phase8_official_test_policy,
    load_phase8_test_manifest,
    paired_patient_bootstrap,
    validate_phase8_test_authorization,
)
from cbis_ddsm.training import set_global_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked P7-E01 and P7-E04 seed-42 official-test passes in order."
        )
    )
    parser.add_argument(
        "--artifact-lock",
        default=str(PROJECT_ROOT / "configs" / "phase8" / "artifact_lock.yaml"),
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
            / "official_test_authorization.json"
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


def validate_runtime_config(config: Any, policy: Any) -> None:
    if (
        config.seed != 42
        or not config.deterministic
        or config.threshold_method != policy.threshold_method
        or config.threshold_tie_break != policy.threshold_tie_break
        or config.expected_validation_samples != policy.expected_validation_samples
        or config.expected_test_samples != policy.expected_test_samples
    ):
        raise ValueError("Phase 8 runtime evaluation settings are not locked.")


def load_locked_threshold(
    model_id: str,
    policy: Any,
) -> tuple[float, Path, str]:
    threshold_path = (
        PROJECT_ROOT / policy.threshold_output_dirs[model_id] / "threshold.json"
    )
    document = read_frozen_threshold(threshold_path)
    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Frozen threshold selection is missing for {model_id}.")
    threshold = float(selection["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Frozen threshold is invalid for {model_id}.")
    return threshold, threshold_path, sha256_file(threshold_path)


def validate_image_paths(test_manifest: pd.DataFrame) -> None:
    missing = [
        str(value)
        for value in test_manifest["image_path"].astype(str)
        if not Path(value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} official-test image paths are missing; first={missing[0]}"
        )


def run_one_model(
    *,
    model_id: str,
    lock: Any,
    policy: Any,
    evaluation_config: Any,
    test_manifest: pd.DataFrame,
    preprocessing: dict[str, Any],
    device: torch.device,
    authorization_sha256: str,
) -> pd.DataFrame:
    output_dir = PROJECT_ROOT / policy.official_test_output_dirs[model_id]
    output_dir.mkdir(parents=True, exist_ok=False)
    threshold, threshold_path, threshold_sha256 = load_locked_threshold(model_id, policy)
    started_at = datetime.now(timezone.utc).isoformat()
    model_index = list(policy.authorized_model_ids).index(model_id)
    started_payload = {
        "format_version": 1,
        "stage": 8,
        "gate": "B",
        "status": "started",
        "model_id": model_id,
        "started_at_utc": started_at,
        "ordered_pass_number_within_gate_b": model_index + 1,
        "project_test_inference_pass_number": (
            policy.project_test_inference_passes_before_phase8 + model_index + 1
        ),
        "test_split_loaded": True,
        "test_manifest_sha256": policy.expected_test_manifest_sha256,
        "test_samples": len(test_manifest),
        "test_patients": int(test_manifest["patient_id"].astype(str).nunique()),
        "threshold": threshold,
        "threshold_sha256": threshold_sha256,
        "authorization_sha256": authorization_sha256,
    }
    write_json(output_dir / "test_evaluation_started.json", started_payload)

    try:
        model, model_audit = load_phase8_frozen_model(PROJECT_ROOT, lock, model_id)
        loader = build_test_dataloader(
            test_manifest,
            preprocessing=preprocessing,
            config=evaluation_config,
        )
        inference = run_inference(
            model,
            loader,
            device,
            mixed_precision=evaluation_config.mixed_precision,
        )
        if len(inference.labels) != policy.expected_test_samples:
            raise RuntimeError("Official-test inference returned an unexpected sample count.")
        expected_ids = test_manifest["sample_id"].astype(str).tolist()
        if list(inference.sample_ids) != expected_ids:
            raise RuntimeError("Official-test inference order changed.")

        probabilities = sigmoid_numpy(inference.logits)
        predicted_labels = (probabilities >= threshold).astype(np.int64)
        predictions = pd.DataFrame(
            {
                "sample_id": inference.sample_ids,
                "patient_id": inference.patient_ids,
                "image_path": inference.image_paths,
                "label": inference.labels,
                "logit": inference.logits,
                "probability": probabilities,
                "predicted_label": predicted_labels,
                "threshold": threshold,
            }
        )
        predictions_path = output_dir / "test_predictions.csv"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8")
        metrics = compute_evaluation_metrics(
            inference.labels,
            probabilities,
            threshold,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        metrics_payload = {
            "format_version": 1,
            "stage": 8,
            "gate": "B",
            "model_id": model_id,
            "split": "official_test",
            "threshold_source": "locked_validation_youden_j",
            "metrics": asdict(metrics),
            "audit": {
                **started_payload,
                "status": "completed",
                "completed_at_utc": completed_at,
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "checkpoint_sha256": lock.models[model_id].checkpoint_sha256,
                "test_predictions_sha256": sha256_file(predictions_path),
                "artifact_lock_sha256": sha256_file(lock.source_path),
                "official_test_policy_sha256": sha256_file(policy.source_path),
                "git_commit": git_value("rev-parse", "HEAD"),
                "git_branch": git_value("branch", "--show-current"),
                "git_status_short": git_value("status", "--short"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
                "strict_checkpoint_load": model_audit["strict_load_pass"],
                "test_inference_passes": 1,
                "test_time_augmentation": False,
                "ensemble": False,
                "training_performed": False,
            },
        }
        write_json(output_dir / "metrics.json", metrics_payload)
        save_evaluation_plots(
            inference.labels,
            probabilities,
            metrics,
            output_dir,
        )
        outcome = {
            "format_version": 1,
            "stage": 8,
            "gate": "B",
            "status": "official_test_evaluation_complete",
            "model_id": model_id,
            "threshold": threshold,
            "threshold_path": str(threshold_path),
            "test_split_loaded": True,
            "test_inference_completed": True,
            "test_inference_passes": 1,
            "test_samples": len(predictions),
            "test_patients": int(predictions["patient_id"].nunique()),
            "metrics": asdict(metrics),
            "artifacts": {
                "predictions": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "metrics": str(output_dir / "metrics.json"),
            },
        }
        write_json(output_dir / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
        return predictions
    except Exception as error:
        write_json(
            output_dir / "failure.json",
            {
                **started_payload,
                "status": "failed_no_automatic_rerun",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "reason": str(error),
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


def write_paired_outputs(
    *,
    first: pd.DataFrame,
    second: pd.DataFrame,
    lock: Any,
    policy: Any,
) -> dict[str, Any]:
    comparison_dir = (
        PROJECT_ROOT
        / policy.official_test_output_dirs[policy.authorized_model_ids[0]]
    ).parent / "paired_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    bootstrap = paired_patient_bootstrap(
        first,
        second,
        first_model_id=policy.authorized_model_ids[0],
        second_model_id=policy.authorized_model_ids[1],
        valid_replicates=int(policy.bootstrap["valid_replicates"]),
        seed=int(policy.bootstrap["seed"]),
        confidence_level=float(policy.bootstrap["confidence_level"]),
    )
    write_json(comparison_dir / "paired_bootstrap.json", bootstrap)

    rows: list[dict[str, Any]] = []
    for model_id, predictions in zip(policy.authorized_model_ids, (first, second)):
        threshold, _, _ = load_locked_threshold(model_id, policy)
        metrics = compute_evaluation_metrics(
            predictions["label"].to_numpy(np.int64),
            predictions["probability"].to_numpy(np.float64),
            threshold,
        )
        spec = lock.models[model_id]
        rows.append(
            {
                "model_id": model_id,
                "role": spec.role,
                "seed": 42,
                "validation_roc_auc": spec.expected_val_roc_auc,
                "validation_pr_auc": spec.expected_val_pr_auc,
                "official_test_roc_auc": metrics.roc_auc,
                "official_test_pr_auc": metrics.pr_auc,
                "validation_to_test_roc_auc_change": (
                    metrics.roc_auc - spec.expected_val_roc_auc
                ),
                "validation_to_test_pr_auc_change": (
                    metrics.pr_auc - spec.expected_val_pr_auc
                ),
            }
        )
    comparison_path = comparison_dir / "validation_test_comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False, encoding="utf-8")
    summary = {
        "format_version": 1,
        "stage": 8,
        "gate": "B",
        "status": "complete",
        "official_test_manifest_sha256": policy.expected_test_manifest_sha256,
        "authorized_model_ids": list(policy.authorized_model_ids),
        "new_test_inference_passes": policy.maximum_new_test_inference_passes,
        "project_test_inference_passes_after_success": (
            policy.project_test_inference_passes_after_success
        ),
        "paired_bootstrap": bootstrap,
        "comparison_csv": str(comparison_path),
        "comparison_csv_sha256": sha256_file(comparison_path),
        "test_split_loaded": True,
        "training_performed": False,
        "threshold_refit_on_test": False,
        "test_time_augmentation": False,
        "ensemble": False,
    }
    write_json(comparison_dir / "outcome.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    lock = load_phase8_artifact_lock(args.artifact_lock)
    policy = load_phase8_official_test_policy(args.policy)
    authorization = validate_phase8_test_authorization(args.authorization, policy)
    authorization_sha256 = sha256_file(Path(args.authorization))
    evaluation_config = load_evaluation_config(args.evaluation_config)
    validate_runtime_config(evaluation_config, policy)
    ensure_phase8_official_test_outputs_absent(PROJECT_ROOT, policy)
    audit_phase8_artifacts(PROJECT_ROOT, lock)

    experiments = {
        model_id: load_phase7_experiment(PROJECT_ROOT / lock.models[model_id].config_path)
        for model_id in policy.authorized_model_ids
    }
    preprocessing = experiments[policy.authorized_model_ids[0]].preprocessing
    if any(
        experiment.preprocessing != preprocessing
        for experiment in experiments.values()
    ):
        raise ValueError("Authorized phase 8 models do not share preprocessing.")

    train_manifest = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "data" / "train.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    val_manifest = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "data" / "val.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    validate_split_manifest(
        train_manifest,
        expected_split="train",
        expected_samples=986,
    )
    validate_split_manifest(
        val_manifest,
        expected_split="val",
        expected_samples=policy.expected_validation_samples,
    )

    # This is the only point at which Gate B reads the official test manifest.
    test_manifest = load_phase8_test_manifest(
        PROJECT_ROOT,
        lock,
        policy,
        authorization_path=args.authorization,
    )
    validate_no_patient_or_sample_overlap(
        train_manifest,
        val_manifest,
        test_manifest,
    )
    validate_image_paths(test_manifest)

    device = select_device(args.device)
    set_global_seed(evaluation_config.seed, deterministic=True)
    print(
        json.dumps(
            {
                "status": "gate_b_preflight_pass",
                "approved_by_user": authorization["approved_by_user"],
                "device": str(device),
                "models": list(policy.authorized_model_ids),
                "test_samples": len(test_manifest),
                "test_patients": int(test_manifest["patient_id"].nunique()),
                "test_manifest_sha256": policy.expected_test_manifest_sha256,
                "test_split_loaded": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    predictions: list[pd.DataFrame] = []
    for model_id in policy.authorized_model_ids:
        predictions.append(
            run_one_model(
                model_id=model_id,
                lock=lock,
                policy=policy,
                evaluation_config=replace(
                    evaluation_config,
                    output_dir=policy.official_test_output_dirs[model_id],
                ),
                test_manifest=test_manifest,
                preprocessing=preprocessing,
                device=device,
                authorization_sha256=authorization_sha256,
            )
        )
    summary = write_paired_outputs(
        first=predictions[0],
        second=predictions[1],
        lock=lock,
        policy=policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
