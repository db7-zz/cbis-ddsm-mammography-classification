"""Formal train/validation-only entry point for approved phase 7 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torchvision.models import EfficientNet_V2_S_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.phase7 import load_phase7_experiment, verify_phase7_manifests
from cbis_ddsm.phase7_model import build_phase7_model, count_phase7_parameters
from cbis_ddsm.training import (
    build_dataloaders,
    set_global_seed,
    sha256_file,
    train_model,
    write_run_metadata,
)


REGISTRY_FIELDS = (
    "experiment_id",
    "started_at",
    "finished_at",
    "status",
    "config_path",
    "output_dir",
    "failure_type",
    "reason",
    "peak_memory_mib",
    "test_split_loaded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one approved phase 7 experiment without loading test data."
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--manifest-dir",
        default=str(PROJECT_ROOT / "artifacts" / "data"),
    )
    parser.add_argument(
        "--manifest-lock",
        default=str(PROJECT_ROOT / "configs" / "phase7" / "manifest_lock.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def git_value(arguments: list[str]) -> str | None:
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


def pretrained_weight_metadata() -> dict[str, object]:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    filename = Path(weights.url).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Locked V2S pretrained weights are missing: {path}")
    return {
        "enum": "EfficientNet_V2_S_Weights.DEFAULT",
        "url": weights.url,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def append_registry(row: dict[str, Any]) -> None:
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "experiments"
        / "phase7"
        / "experiment_registry.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    normalized = {field: row.get(field, "") for field in REGISTRY_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def peak_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)


def main() -> int:
    args = parse_args()
    experiment = load_phase7_experiment(args.experiment_config)
    started_at = datetime.now().astimezone().isoformat()
    output_dir = (PROJECT_ROOT / experiment.training.output_dir).resolve()
    device = torch.device("cpu")
    try:
        train_manifest, val_manifest, manifest_hashes = verify_phase7_manifests(
            args.manifest_dir,
            args.manifest_lock,
        )
        weight_metadata = pretrained_weight_metadata()
        device = select_device(args.device)
        resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
        model_config = (
            replace(experiment.model, pretrained=False)
            if resume_path is not None
            else experiment.model
        )
        set_global_seed(
            experiment.training.seed,
            deterministic=experiment.training.deterministic,
        )
        train_loader, val_loader = build_dataloaders(
            train_manifest,
            val_manifest,
            preprocessing=experiment.preprocessing,
            config=experiment.training,
        )
        model = build_phase7_model(model_config)
        total_parameters, trainable_parameters = count_phase7_parameters(model)
        source_paths = (
            experiment.source_path,
            PROJECT_ROOT / "configs" / "phase7" / "manifest_lock.yaml",
            PROJECT_ROOT / "scripts" / "run_phase7_experiment.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "dataset.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "phase7.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "phase7_model.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "phase6_model.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "preprocessing.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "training.py",
        )
        audit_metadata = {
            "experiment_id": experiment.experiment_id,
            "variant": experiment.variant,
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_branch": git_value(["branch", "--show-current"]),
            "git_status_short": git_value(["status", "--short"]),
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
                for path in source_paths
            },
            "pretrained_weights": weight_metadata,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "physical_batch_size": experiment.training.batch_size,
            "gradient_accumulation_steps": (
                experiment.training.gradient_accumulation_steps
            ),
            "effective_batch_size": (
                experiment.training.batch_size
                * experiment.training.gradient_accumulation_steps
            ),
            "test_split_loaded": False,
        }
        print(
            json.dumps(
                {
                    "experiment_id": experiment.experiment_id,
                    "variant": experiment.variant,
                    "device": str(device),
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "manifest_hashes": manifest_hashes,
                    "pretrained_weights": weight_metadata,
                    "output_dir": str(output_dir),
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "test_split_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        outcome = train_model(
            model,
            train_loader,
            val_loader,
            config=experiment.training,
            device=device,
            output_dir=output_dir,
            model_config=asdict(experiment.model),
            manifest_hashes=manifest_hashes,
            resume_checkpoint=resume_path,
            audit_metadata=audit_metadata,
        )
        result = {
            "experiment_id": experiment.experiment_id,
            "variant": experiment.variant,
            "status": "completed",
            "best_epoch": outcome.best_epoch,
            "best_val_roc_auc": outcome.best_val_roc_auc,
            "completed_epochs": outcome.completed_epochs,
            "stopped_early": outcome.stopped_early,
            "output_dir": str(outcome.output_dir),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "pretrained_weights": weight_metadata,
            "peak_memory_mib": peak_memory_mib(device),
            "test_split_loaded": False,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
            },
        }
        write_run_metadata(outcome.output_dir / "outcome.json", result)
        append_registry(
            {
                "experiment_id": experiment.experiment_id,
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(),
                "status": "completed",
                "config_path": str(experiment.source_path),
                "output_dir": str(outcome.output_dir),
                "peak_memory_mib": peak_memory_mib(device),
                "test_split_loaded": False,
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        failure_type = (
            "oom"
            if isinstance(error, torch.OutOfMemoryError)
            else "nan"
            if isinstance(error, FloatingPointError)
            else "pretrained_weight"
            if "weight" in str(error).lower()
            else "exception"
        )
        result = {
            "experiment_id": experiment.experiment_id,
            "status": "failed",
            "failure_type": failure_type,
            "reason": f"{type(error).__name__}: {error}",
            "output_dir": str(output_dir),
            "peak_memory_mib": peak_memory_mib(device),
            "test_split_loaded": False,
        }
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
        failure_path = output_dir / "outcome.json"
        if not failure_path.exists():
            write_run_metadata(failure_path, result)
        append_registry(
            {
                "experiment_id": experiment.experiment_id,
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(),
                "status": "failed",
                "config_path": str(experiment.source_path),
                "output_dir": str(output_dir),
                "failure_type": failure_type,
                "reason": result["reason"],
                "peak_memory_mib": peak_memory_mib(device),
                "test_split_loaded": False,
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
