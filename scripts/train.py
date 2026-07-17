"""CLI for leakage-safe phase 3 training on train and validation only."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cbis_ddsm.config import load_config
from cbis_ddsm.model import build_model, count_parameters, load_model_config
from cbis_ddsm.training import (
    build_dataloaders,
    load_training_config,
    load_training_manifests,
    set_global_seed,
    sha256_file,
    train_model,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the phase 3 baseline without loading the test split."
    )
    parser.add_argument(
        "--data-config", default=str(PROJECT_ROOT / "configs" / "data.local.yaml")
    )
    parser.add_argument(
        "--model-config", default=str(PROJECT_ROOT / "configs" / "model.yaml")
    )
    parser.add_argument(
        "--train-config", default=str(PROJECT_ROOT / "configs" / "train.yaml")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet initialization. Intended only for smoke tests.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event writing for a smoke test.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_status() -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_hashes() -> dict[str, str]:
    relative_paths = (
        "configs/model.yaml",
        "configs/train.yaml",
        "scripts/train.py",
        "src/cbis_ddsm/dataset.py",
        "src/cbis_ddsm/metrics.py",
        "src/cbis_ddsm/model.py",
        "src/cbis_ddsm/training.py",
    )
    return {
        relative: sha256_file(PROJECT_ROOT / relative) for relative in relative_paths
    }


def positive_limit(value: int | None, name: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")
    return value


def main() -> int:
    args = parse_args()
    max_train_batches = positive_limit(args.max_train_batches, "--max-train-batches")
    max_val_batches = positive_limit(args.max_val_batches, "--max-val-batches")
    max_epochs = positive_limit(args.max_epochs, "--max-epochs")

    data_document = load_config(args.data_config)
    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.train_config)
    if max_epochs is not None:
        training_config = replace(training_config, max_epochs=max_epochs)
    if args.output_dir:
        training_config = replace(training_config, output_dir=args.output_dir)
    if args.no_tensorboard:
        training_config = replace(training_config, tensorboard=False)
    training_config.validate()

    if int(data_document["preprocessing"]["image_size"]) != model_config.input_size:
        raise ValueError("Data preprocessing size does not match the model input size.")

    data_output_dir = resolve_project_path(data_document["data"]["output_dir"])
    train_path = data_output_dir / "train.csv"
    val_path = data_output_dir / "val.csv"
    manifests = load_training_manifests(data_output_dir)
    manifest_hashes = {
        "train.csv": sha256_file(train_path),
        "val.csv": sha256_file(val_path),
    }

    resume_path = resolve_project_path(args.resume) if args.resume else None
    build_config = model_config
    if args.no_pretrained or resume_path is not None:
        build_config = replace(model_config, pretrained=False)

    device = select_device(args.device)
    set_global_seed(training_config.seed, deterministic=training_config.deterministic)
    train_loader, val_loader = build_dataloaders(
        *manifests,
        preprocessing=data_document["preprocessing"],
        config=training_config,
    )
    model = build_model(build_config)
    total_parameters, trainable_parameters = count_parameters(model)
    output_dir = resolve_project_path(training_config.output_dir)

    startup = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_split_loaded": False,
        "pretrained_initialization": build_config.pretrained,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "output_dir": str(output_dir),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "resume": str(resume_path) if resume_path else None,
    }
    print(json.dumps(startup, ensure_ascii=False, indent=2), flush=True)

    recorded_model_config = model_config if resume_path is not None else build_config
    current_git_status = git_status()
    audit_metadata = {
        "git_commit": git_commit(),
        "git_status_short": current_git_status,
        "source_hashes": source_hashes(),
    }
    outcome = train_model(
        model,
        train_loader,
        val_loader,
        config=training_config,
        device=device,
        output_dir=output_dir,
        model_config=asdict(recorded_model_config),
        manifest_hashes=manifest_hashes,
        resume_checkpoint=resume_path,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        audit_metadata=audit_metadata,
    )
    result = {
        "best_epoch": outcome.best_epoch,
        "best_val_roc_auc": outcome.best_val_roc_auc,
        "completed_epochs": outcome.completed_epochs,
        "stopped_early": outcome.stopped_early,
        "output_dir": str(outcome.output_dir),
        "test_split_loaded": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "git_commit": audit_metadata["git_commit"],
            "git_dirty": bool(current_git_status),
        },
    }
    write_run_metadata(outcome.output_dir / "outcome.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
