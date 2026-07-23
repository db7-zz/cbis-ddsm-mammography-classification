"""Formal train/validation-only entry point for approved phase 5 experiments."""

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
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.model import build_model, count_parameters
from cbis_ddsm.phase5 import load_phase5_experiment, verify_locked_training_manifests
from cbis_ddsm.training import (
    build_dataloaders,
    set_global_seed,
    sha256_file,
    train_model,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one approved phase 5 experiment without loading test data."
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--manifest-dir", default=str(PROJECT_ROOT / "artifacts" / "data")
    )
    parser.add_argument(
        "--manifest-lock",
        default=str(PROJECT_ROOT / "configs" / "phase5" / "manifest_lock.yaml"),
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


def append_registry(row: Mapping[str, Any]) -> None:
    path = PROJECT_ROOT / "artifacts" / "experiments" / "phase5" / "experiment_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    experiment = load_phase5_experiment(args.experiment_config)
    started_at = datetime.now().astimezone().isoformat()
    registry_base = {
        "experiment_id": experiment.experiment_id,
        "started_at": started_at,
        "config_path": str(experiment.source_path),
        "test_split_loaded": False,
    }
    try:
        train_manifest, val_manifest, manifest_hashes = verify_locked_training_manifests(
            args.manifest_dir, args.manifest_lock
        )
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
        model = build_model(model_config)
        total_parameters, trainable_parameters = count_parameters(model)
        output_dir = (PROJECT_ROOT / experiment.training.output_dir).resolve()
        git_status = git_value(["status", "--short"])
        source_paths = (
            experiment.source_path,
            PROJECT_ROOT / "configs" / "phase5" / "manifest_lock.yaml",
            PROJECT_ROOT / "scripts" / "run_phase5_experiment.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "dataset.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "model.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "phase5.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "preprocessing.py",
            PROJECT_ROOT / "src" / "cbis_ddsm" / "training.py",
        )
        audit_metadata = {
            "experiment_id": experiment.experiment_id,
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_status_short": git_status,
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
                for path in source_paths
            },
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
                    "device": str(device),
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "manifest_hashes": manifest_hashes,
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
            "status": "completed",
            "best_epoch": outcome.best_epoch,
            "best_val_roc_auc": outcome.best_val_roc_auc,
            "completed_epochs": outcome.completed_epochs,
            "stopped_early": outcome.stopped_early,
            "output_dir": str(outcome.output_dir),
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
                **registry_base,
                "finished_at": datetime.now().astimezone().isoformat(),
                "status": "completed",
                "output_dir": str(outcome.output_dir),
                "reason": "",
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        append_registry(
            {
                **registry_base,
                "finished_at": datetime.now().astimezone().isoformat(),
                "status": "failed",
                "output_dir": experiment.training.output_dir,
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
