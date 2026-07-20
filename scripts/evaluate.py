"""Freeze a validation threshold, then run the one-time final test evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.config import load_config
from cbis_ddsm.evaluation import (
    build_test_dataloader,
    compute_evaluation_metrics,
    ensure_new_output_directory,
    freeze_threshold,
    load_evaluation_config,
    load_test_manifest_after_threshold,
    run_inference,
    save_evaluation_plots,
    select_youden_threshold,
    validate_no_patient_or_sample_overlap,
    validate_split_manifest,
    write_json,
)
from cbis_ddsm.metrics import sigmoid_numpy
from cbis_ddsm.model import build_model, load_model_config
from cbis_ddsm.training import set_global_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and freeze a validation-only Youden threshold before allowing "
            "the final test split to be loaded."
        )
    )
    parser.add_argument(
        "--data-config", default=str(PROJECT_ROOT / "configs" / "data.local.yaml")
    )
    parser.add_argument(
        "--model-config", default=str(PROJECT_ROOT / "configs" / "model.yaml")
    )
    parser.add_argument(
        "--eval-config", default=str(PROJECT_ROOT / "configs" / "eval.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "artifacts" / "training" / "baseline_seed42" / "best.pt"),
    )
    parser.add_argument(
        "--validation-predictions",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "training"
            / "baseline_seed42"
            / "val_predictions_best.csv"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--threshold-only",
        action="store_true",
        help="Freeze and audit the validation threshold without reading test.csv.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
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
    return result.stdout.strip()


def load_checkpoint_audit(
    checkpoint_path: Path,
    *,
    model_config: object,
    train_manifest_path: Path,
    val_manifest_path: Path,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    expected_hashes = {
        "train.csv": sha256_file(train_manifest_path),
        "val.csv": sha256_file(val_manifest_path),
    }
    if dict(checkpoint.get("manifest_hashes", {})) != expected_hashes:
        raise ValueError("Checkpoint train/validation manifest hashes do not match.")
    recorded_model_config = dict(checkpoint.get("model_config", {}))
    if recorded_model_config != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match configs/model.yaml.")
    epoch = int(checkpoint.get("epoch", -1))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    if epoch <= 0 or epoch != best_epoch:
        raise ValueError("Final evaluation requires the numerical-best checkpoint.")
    return checkpoint


def load_and_audit_validation_predictions(
    path: Path,
    val_manifest: pd.DataFrame,
    checkpoint: dict[str, object],
    *,
    expected_samples: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation predictions do not exist: {path}")
    predictions = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "label", "logit", "probability", "best_epoch"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Validation predictions are missing: {', '.join(missing)}")
    if len(predictions) != expected_samples or predictions["sample_id"].duplicated().any():
        raise ValueError("Validation prediction sample count or identifiers are invalid.")
    recorded_epochs = set(pd.to_numeric(predictions["best_epoch"], errors="raise").astype(int))
    if recorded_epochs != {int(checkpoint["best_epoch"])}:
        raise ValueError("Validation prediction epoch does not match the checkpoint.")

    manifest_labels = val_manifest.set_index("sample_id")["label"].astype(int)
    prediction_labels = predictions.set_index("sample_id")["label"].astype(int)
    if set(manifest_labels.index) != set(prediction_labels.index):
        raise ValueError("Validation prediction IDs do not match val.csv.")
    prediction_labels = prediction_labels.reindex(manifest_labels.index)
    if not np.array_equal(manifest_labels.to_numpy(), prediction_labels.to_numpy()):
        raise ValueError("Validation labels do not match val.csv.")

    logits = pd.to_numeric(predictions["logit"], errors="raise").to_numpy(np.float64)
    probabilities = pd.to_numeric(
        predictions["probability"], errors="raise"
    ).to_numpy(np.float64)
    recomputed_probabilities = sigmoid_numpy(logits)
    if not np.allclose(probabilities, recomputed_probabilities, rtol=0.0, atol=1e-7):
        raise ValueError("Saved validation probabilities do not match their logits.")
    labels = pd.to_numeric(predictions["label"], errors="raise").to_numpy(np.int64)
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }
    if not np.isclose(metrics["roc_auc"], float(checkpoint["best_metric"]), atol=1e-12):
        raise ValueError("Recomputed validation ROC AUC does not match the checkpoint.")
    return predictions, metrics


def main() -> int:
    args = parse_args()
    data_document = load_config(args.data_config)
    model_config = load_model_config(args.model_config)
    evaluation_config = load_evaluation_config(args.eval_config)
    if args.output_dir:
        evaluation_config = replace(evaluation_config, output_dir=args.output_dir)
        evaluation_config.validate()
    if int(data_document["preprocessing"]["image_size"]) != model_config.input_size:
        raise ValueError("Data preprocessing size does not match model input size.")

    output_dir = ensure_new_output_directory(
        resolve_project_path(evaluation_config.output_dir)
    )
    data_output_dir = resolve_project_path(data_document["data"]["output_dir"])
    train_manifest_path = data_output_dir / "train.csv"
    val_manifest_path = data_output_dir / "val.csv"
    checkpoint_path = resolve_project_path(args.checkpoint)
    validation_predictions_path = resolve_project_path(args.validation_predictions)

    if not train_manifest_path.is_file() or not val_manifest_path.is_file():
        raise FileNotFoundError("Stage 1 train.csv and val.csv are required.")
    val_manifest = pd.read_csv(
        val_manifest_path, dtype={"patient_id": str, "sample_id": str}
    )
    validate_split_manifest(
        val_manifest,
        expected_split="val",
        expected_samples=evaluation_config.expected_validation_samples,
    )
    checkpoint = load_checkpoint_audit(
        checkpoint_path,
        model_config=model_config,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
    )
    validation_predictions, validation_metrics = load_and_audit_validation_predictions(
        validation_predictions_path,
        val_manifest,
        checkpoint,
        expected_samples=evaluation_config.expected_validation_samples,
    )
    val_labels = validation_predictions["label"].to_numpy(np.int64)
    val_probabilities = validation_predictions["probability"].to_numpy(np.float64)
    selection = select_youden_threshold(val_labels, val_probabilities)
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_metric": float(checkpoint["best_metric"]),
        "validation_predictions_path": str(validation_predictions_path),
        "validation_predictions_sha256": sha256_file(validation_predictions_path),
        "validation_manifest_sha256": sha256_file(val_manifest_path),
        "validation_samples": len(validation_predictions),
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_pr_auc": validation_metrics["pr_auc"],
    }
    threshold_path = freeze_threshold(
        output_dir,
        selection,
        provenance=provenance,
        reference_threshold=evaluation_config.reference_threshold,
    )
    write_json(
        output_dir / "validation_selection.json",
        {
            "validation_metrics": validation_metrics,
            "selection": asdict(selection),
            "reference_threshold": evaluation_config.reference_threshold,
            "absolute_difference_from_reference": abs(
                selection.threshold - evaluation_config.reference_threshold
            ),
        },
    )
    print(
        json.dumps(
            {
                "threshold_frozen": selection.threshold,
                "threshold_path": str(threshold_path),
                "test_split_loaded": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.threshold_only:
        write_json(
            output_dir / "outcome.json",
            {
                "status": "threshold_only_complete",
                "threshold": selection.threshold,
                "test_split_loaded": False,
                "test_inference_completed": False,
            },
        )
        return 0

    # This is the first point at which test.csv is allowed to be read.
    test_manifest = load_test_manifest_after_threshold(
        data_output_dir,
        threshold_path,
        expected_samples=evaluation_config.expected_test_samples,
    )
    train_manifest = pd.read_csv(
        train_manifest_path, dtype={"patient_id": str, "sample_id": str}
    )
    validate_no_patient_or_sample_overlap(train_manifest, val_manifest, test_manifest)
    test_manifest_hash = sha256_file(data_output_dir / "test.csv")

    device = select_device(args.device)
    set_global_seed(
        evaluation_config.seed, deterministic=evaluation_config.deterministic
    )
    inference_model_config = replace(model_config, pretrained=False)
    model = build_model(inference_model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader = build_test_dataloader(
        test_manifest,
        preprocessing=data_document["preprocessing"],
        config=evaluation_config,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        output_dir / "test_evaluation_started.json",
        {
            "started_at_utc": started_at,
            "test_split_loaded": True,
            "test_manifest_sha256": test_manifest_hash,
            "test_samples": len(test_manifest),
            "threshold": selection.threshold,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        },
    )
    inference = run_inference(
        model,
        loader,
        device,
        mixed_precision=evaluation_config.mixed_precision,
    )
    if len(inference.labels) != evaluation_config.expected_test_samples:
        raise RuntimeError("Test inference did not return the expected number of samples.")
    expected_ids = test_manifest["sample_id"].astype(str).tolist()
    if list(inference.sample_ids) != expected_ids:
        raise RuntimeError("Test inference order does not match test.csv.")

    probabilities = sigmoid_numpy(inference.logits)
    predicted_labels = (probabilities >= selection.threshold).astype(np.int64)
    predictions = pd.DataFrame(
        {
            "sample_id": inference.sample_ids,
            "patient_id": inference.patient_ids,
            "image_path": inference.image_paths,
            "label": inference.labels,
            "logit": inference.logits,
            "probability": probabilities,
            "predicted_label": predicted_labels,
            "threshold": selection.threshold,
        }
    )
    predictions_path = output_dir / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    metrics = compute_evaluation_metrics(
        inference.labels, probabilities, selection.threshold
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    metrics_payload = {
        "format_version": 1,
        "split": "official_test",
        "threshold_source": "validation_youden_j",
        "metrics": asdict(metrics),
        "validation_selection": asdict(selection),
        "reference_results": {
            "threshold": evaluation_config.reference_threshold,
            "roc_auc": 0.7869,
            "pr_auc": 0.6673,
            "sensitivity": 0.806,
            "used_for_selection_or_predictions": False,
        },
        "audit": {
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "test_manifest_sha256": test_manifest_hash,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "test_predictions_sha256": sha256_file(predictions_path),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_short": git_value("status", "--short"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    save_evaluation_plots(inference.labels, probabilities, metrics, output_dir)
    outcome = {
        "status": "final_test_evaluation_complete",
        "threshold": selection.threshold,
        "test_split_loaded": True,
        "test_inference_completed": True,
        "test_inference_passes": 1,
        "test_samples": len(inference.labels),
        "metrics": asdict(metrics),
        "artifacts": {
            "threshold": str(threshold_path),
            "predictions": str(predictions_path),
            "metrics": str(output_dir / "metrics.json"),
        },
    }
    write_json(output_dir / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
