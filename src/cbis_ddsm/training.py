"""Leakage-safe phase 3 training utilities for the locked baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import ConfigurationError
from .dataset import MammogramDataset
from .metrics import BinaryMetrics, compute_binary_metrics, sigmoid_numpy


@dataclass(frozen=True)
class TrainingConfig:
    """Validated phase 3 settings assembled from ``configs/train.yaml``."""

    seed: int = 42
    max_epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    mixed_precision: bool = True
    deterministic: bool = True
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    output_dir: str = "artifacts/training/baseline_seed42"
    loss_name: str = "BCEWithLogitsLoss"
    pos_weight: float = 1.84
    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_name: str = "ReduceLROnPlateau"
    scheduler_monitor: str = "val_roc_auc"
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_learning_rate: float = 1e-6
    early_stopping_monitor: str = "val_roc_auc"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    tensorboard: bool = True
    save_validation_predictions: bool = True

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "TrainingConfig":
        required_sections = (
            "training",
            "loss",
            "optimizer",
            "scheduler",
            "early_stopping",
            "logging",
        )
        sections: dict[str, Mapping[str, Any]] = {}
        for name in required_sections:
            value = document.get(name)
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"Missing training configuration section: {name}")
            sections[name] = value

        training = sections["training"]
        loss = sections["loss"]
        optimizer = sections["optimizer"]
        scheduler = sections["scheduler"]
        early = sections["early_stopping"]
        logging = sections["logging"]
        config = cls(
            seed=int(training.get("seed", 42)),
            max_epochs=int(training.get("max_epochs", 50)),
            batch_size=int(training.get("batch_size", 16)),
            num_workers=int(training.get("num_workers", 4)),
            pin_memory=bool(training.get("pin_memory", True)),
            persistent_workers=bool(training.get("persistent_workers", True)),
            prefetch_factor=int(training.get("prefetch_factor", 2)),
            mixed_precision=bool(training.get("mixed_precision", True)),
            deterministic=bool(training.get("deterministic", True)),
            gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 1)
            ),
            output_dir=str(training.get("output_dir", "artifacts/training/baseline_seed42")),
            loss_name=str(loss.get("name", "BCEWithLogitsLoss")),
            pos_weight=float(loss.get("pos_weight", 1.84)),
            optimizer_name=str(optimizer.get("name", "AdamW")),
            learning_rate=float(optimizer.get("learning_rate", 1e-4)),
            weight_decay=float(optimizer.get("weight_decay", 1e-4)),
            scheduler_name=str(scheduler.get("name", "ReduceLROnPlateau")),
            scheduler_monitor=str(scheduler.get("monitor", "val_roc_auc")),
            scheduler_mode=str(scheduler.get("mode", "max")),
            scheduler_factor=float(scheduler.get("factor", 0.5)),
            scheduler_patience=int(scheduler.get("patience", 3)),
            min_learning_rate=float(scheduler.get("min_learning_rate", 1e-6)),
            early_stopping_monitor=str(early.get("monitor", "val_roc_auc")),
            early_stopping_mode=str(early.get("mode", "max")),
            early_stopping_patience=int(early.get("patience", 10)),
            early_stopping_min_delta=float(early.get("min_delta", 0.001)),
            tensorboard=bool(logging.get("tensorboard", True)),
            save_validation_predictions=bool(
                logging.get("save_validation_predictions", True)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigurationError("training.seed must be non-negative.")
        if self.max_epochs <= 0 or self.batch_size <= 0:
            raise ConfigurationError("max_epochs and batch_size must be positive.")
        if self.num_workers < 0 or self.prefetch_factor <= 0:
            raise ConfigurationError("num_workers must be non-negative and prefetch_factor positive.")
        if self.gradient_clip_norm <= 0:
            raise ConfigurationError("gradient_clip_norm must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigurationError("gradient_accumulation_steps must be positive.")
        if not self.output_dir:
            raise ConfigurationError("training.output_dir must not be empty.")
        if self.loss_name != "BCEWithLogitsLoss":
            raise ConfigurationError("The locked baseline loss is BCEWithLogitsLoss.")
        if not np.isclose(self.pos_weight, 1.84):
            raise ConfigurationError("The locked baseline pos_weight is 1.84.")
        if self.optimizer_name != "AdamW":
            raise ConfigurationError("The phase 3 reproduction assumption uses AdamW.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigurationError("Optimizer learning rate must be positive.")
        if self.scheduler_name != "ReduceLROnPlateau":
            raise ConfigurationError(
                "The phase 3 reproduction assumption uses ReduceLROnPlateau."
            )
        if self.scheduler_monitor != "val_roc_auc" or self.scheduler_mode != "max":
            raise ConfigurationError("The scheduler must maximize val_roc_auc.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ConfigurationError("scheduler.factor must be between 0 and 1.")
        if self.scheduler_patience < 0 or self.min_learning_rate < 0:
            raise ConfigurationError("Scheduler patience and minimum LR must be non-negative.")
        if (
            self.early_stopping_monitor != "val_roc_auc"
            or self.early_stopping_mode != "max"
        ):
            raise ConfigurationError("Early stopping must maximize val_roc_auc.")
        if self.early_stopping_patience <= 0 or self.early_stopping_min_delta < 0:
            raise ConfigurationError("Early-stopping patience must be positive.")


@dataclass(frozen=True)
class EpochOutput:
    """Loss and per-sample outputs collected from one loader pass."""

    loss: float
    labels: np.ndarray
    logits: np.ndarray
    sample_ids: tuple[str, ...]
    batches: int
    samples: int
    preprocessing_fallback_count: int
    preprocessing_fallback_reasons: dict[str, int]


@dataclass(frozen=True)
class TrainingOutcome:
    """Summary of a completed or early-stopped training run."""

    best_epoch: int
    best_val_roc_auc: float
    completed_epochs: int
    stopped_early: bool
    output_dir: Path


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Training configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ConfigurationError("Training configuration root must be a mapping.")
    return TrainingConfig.from_document(document)


def set_global_seed(seed: int, *, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_manifests(data_output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only train and validation manifests; test is intentionally untouched."""
    root = Path(data_output_dir).expanduser().resolve()
    train_path = root / "train.csv"
    val_path = root / "val.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"Expected train.csv and val.csv under {root}")
    dtype = {"patient_id": str, "sample_id": str}
    train = pd.read_csv(train_path, dtype=dtype)
    val = pd.read_csv(val_path, dtype=dtype)
    _validate_training_manifests(train, val)
    return train, val


def _validate_training_manifests(train: pd.DataFrame, val: pd.DataFrame) -> None:
    required = {"sample_id", "split", "patient_id", "label", "image_path"}
    for name, frame, expected_split in (("train", train, "train"), ("val", val, "val")):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} manifest is missing columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError(f"{name} manifest must not be empty.")
        observed_splits = set(frame["split"].astype(str).unique())
        if observed_splits != {expected_split}:
            raise ValueError(f"{name} manifest contains unexpected splits: {observed_splits}")
        labels = set(pd.to_numeric(frame["label"], errors="raise").astype(int).unique())
        if labels != {0, 1}:
            raise ValueError(f"{name} manifest must contain both binary classes.")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} manifest contains duplicate sample IDs.")
    patient_overlap = set(train["patient_id"]).intersection(val["patient_id"])
    sample_overlap = set(train["sample_id"]).intersection(val["sample_id"])
    if patient_overlap:
        raise ValueError(f"Train/validation patient leakage detected: {sorted(patient_overlap)[:5]}")
    if sample_overlap:
        raise ValueError(f"Train/validation sample leakage detected: {sorted(sample_overlap)[:5]}")


def build_dataloaders(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    *,
    preprocessing: Mapping[str, Any],
    config: TrainingConfig,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    _validate_training_manifests(train_manifest, val_manifest)
    train_dataset = MammogramDataset(
        train_manifest,
        split="train",
        preprocessing=dict(preprocessing),
    )
    val_dataset = MammogramDataset(
        val_manifest,
        split="val",
        preprocessing=dict(preprocessing),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    shared: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "worker_init_fn": seed_worker,
    }
    if config.num_workers > 0:
        shared["persistent_workers"] = config.persistent_workers
        shared["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **shared,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **shared,
    )
    return train_loader, val_loader


def build_optimizer_scheduler(
    model: nn.Module, config: TrainingConfig
) -> tuple[Optimizer, ReduceLROnPlateau]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    return optimizer, scheduler


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    gradient_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> EpochOutput:
    training = optimizer is not None
    model.train(training)
    amp_enabled = mixed_precision and device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    fallback_reasons: Counter[str] = Counter()
    batches = 0
    accumulated_samples = 0

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    available_batches = len(loader)
    target_batches = (
        available_batches if max_batches is None else min(available_batches, max_batches)
    )
    if target_batches <= 0:
        raise RuntimeError("The data loader produced no samples.")
    if training:
        optimizer.zero_grad(set_to_none=True)

    batches_to_run = loader if max_batches is None else islice(loader, max_batches)
    for batch_index, batch in enumerate(batches_to_run):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().reshape(-1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
        batch_size = int(labels.numel())
        if training:
            accumulated_samples += batch_size
            backward_loss = loss * batch_size
            if scaler is not None and scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == target_batches
            )
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0

        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        batches += 1
        all_labels.append(labels.detach().cpu().numpy())
        all_logits.append(logits.detach().float().cpu().numpy())
        identifiers = batch.get("sample_id")
        if identifiers is None:
            all_sample_ids.extend(str(total_samples - batch_size + index) for index in range(batch_size))
        else:
            all_sample_ids.extend(str(value) for value in identifiers)
        fallback_values = batch.get("preprocessing_fallback")
        reason_values = batch.get("preprocessing_fallback_reason")
        if fallback_values is not None:
            if isinstance(fallback_values, Tensor):
                observed_fallbacks = [bool(value) for value in fallback_values.tolist()]
            else:
                observed_fallbacks = [bool(value) for value in fallback_values]
            observed_reasons = (
                [str(value) for value in reason_values]
                if reason_values is not None
                else [""] * len(observed_fallbacks)
            )
            for fallback, reason in zip(observed_fallbacks, observed_reasons, strict=True):
                if fallback:
                    fallback_reasons[reason or "unspecified"] += 1

    if total_samples == 0:
        raise RuntimeError("The data loader produced no samples.")
    return EpochOutput(
        loss=total_loss / total_samples,
        labels=np.concatenate(all_labels),
        logits=np.concatenate(all_logits),
        sample_ids=tuple(all_sample_ids),
        batches=batches,
        samples=total_samples,
        preprocessing_fallback_count=sum(fallback_reasons.values()),
        preprocessing_fallback_reasons=dict(sorted(fallback_reasons.items())),
    )


def ensure_run_directory(output_dir: str | Path, *, resume: bool) -> Path:
    path = Path(output_dir).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Training output path is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()) and not resume:
        raise FileExistsError(
            f"Training output directory is not empty: {path}. Use an explicit resume checkpoint."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    training_config: TrainingConfig,
    model_config: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    early_stopping_reference: float | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "early_stopping_reference": (
            best_metric if early_stopping_reference is None else early_stopping_reference
        ),
        "epochs_without_improvement": epochs_without_improvement,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "training_config": asdict(training_config),
        "model_config": dict(model_config),
        "manifest_hashes": dict(manifest_hashes),
        "rng_state": _rng_state(),
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    expected_manifest_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    if expected_manifest_hashes is not None and dict(
        checkpoint.get("manifest_hashes", {})
    ) != dict(expected_manifest_hashes):
        raise ValueError("Checkpoint manifest hashes do not match the current data split.")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _append_epoch_csv(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_validation_predictions(path: Path, epoch: int, output: EpochOutput) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": output.sample_ids,
            "label": output.labels.astype(int),
            "logit": output.logits,
            "probability": sigmoid_numpy(output.logits),
            "best_epoch": epoch,
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8")


def train_model(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    *,
    config: TrainingConfig,
    device: torch.device,
    output_dir: str | Path,
    model_config: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    resume_checkpoint: str | Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> TrainingOutcome:
    run_dir = ensure_run_directory(output_dir, resume=resume_checkpoint is not None)
    write_run_metadata(
        run_dir / "run_config.json",
        {
            "training_config": asdict(config),
            "model_config": dict(model_config),
            "manifest_hashes": dict(manifest_hashes),
            "test_split_loaded": False,
            "audit_metadata": dict(audit_metadata or {}),
        },
    )
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_epoch = 1
    best_epoch = 0
    best_metric = float("-inf")
    early_stopping_reference = float("-inf")
    epochs_without_improvement = 0

    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_manifest_hashes=manifest_hashes,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        early_stopping_reference = float(
            checkpoint.get("early_stopping_reference", best_metric)
        )
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])

    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=start_epoch)

    stopped_early = False
    completed_epochs = start_epoch - 1
    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            train_output = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                mixed_precision=config.mixed_precision,
                gradient_clip_norm=config.gradient_clip_norm,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_batches=max_train_batches,
            )
            with torch.inference_mode():
                val_output = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    mixed_precision=config.mixed_precision,
                    gradient_clip_norm=config.gradient_clip_norm,
                    max_batches=max_val_batches,
                )
            val_metrics: BinaryMetrics = compute_binary_metrics(
                val_output.labels, val_output.logits
            )
            scheduler.step(val_metrics.roc_auc)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            new_best = val_metrics.roc_auc > best_metric
            meaningful_improvement = val_metrics.roc_auc > (
                early_stopping_reference + config.early_stopping_min_delta
            )
            if new_best:
                best_metric = val_metrics.roc_auc
                best_epoch = epoch
            if meaningful_improvement:
                early_stopping_reference = val_metrics.roc_auc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            elapsed = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "train_loss": train_output.loss,
                "val_loss": val_output.loss,
                "val_roc_auc": val_metrics.roc_auc,
                "val_pr_auc": val_metrics.pr_auc,
                "learning_rate": learning_rate,
                "epoch_seconds": elapsed,
                "train_samples": train_output.samples,
                "val_samples": val_output.samples,
                "best_epoch": best_epoch,
                "best_val_roc_auc": best_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": (
                    config.batch_size * config.gradient_accumulation_steps
                ),
                "train_preprocessing_fallbacks": (
                    train_output.preprocessing_fallback_count
                ),
                "val_preprocessing_fallbacks": val_output.preprocessing_fallback_count,
                "train_preprocessing_fallback_reasons": json.dumps(
                    train_output.preprocessing_fallback_reasons, sort_keys=True
                ),
                "val_preprocessing_fallback_reasons": json.dumps(
                    val_output.preprocessing_fallback_reasons, sort_keys=True
                ),
                "cuda_peak_memory_mib": (
                    round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
                    if device.type == "cuda"
                    else None
                ),
            }
            _append_epoch_csv(run_dir / "metrics.csv", row)
            if writer is not None:
                for key in (
                    "train_loss",
                    "val_loss",
                    "val_roc_auc",
                    "val_pr_auc",
                    "learning_rate",
                    "epoch_seconds",
                ):
                    writer.add_scalar(key, row[key], epoch)
                writer.flush()

            if new_best:
                save_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    epochs_without_improvement=epochs_without_improvement,
                    training_config=config,
                    model_config=model_config,
                    manifest_hashes=manifest_hashes,
                    early_stopping_reference=early_stopping_reference,
                )
                if config.save_validation_predictions:
                    _save_validation_predictions(
                        run_dir / "val_predictions_best.csv", epoch, val_output
                    )
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                training_config=config,
                model_config=model_config,
                manifest_hashes=manifest_hashes,
                early_stopping_reference=early_stopping_reference,
            )
            completed_epochs = epoch
            print(json.dumps(row, ensure_ascii=False), flush=True)

            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break
    finally:
        if writer is not None:
            writer.close()

    if best_epoch == 0:
        raise RuntimeError("Training completed without producing a best checkpoint.")
    return TrainingOutcome(
        best_epoch=best_epoch,
        best_val_roc_auc=best_metric,
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
        output_dir=run_dir,
    )


def write_run_metadata(path: str | Path, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, ensure_ascii=False, indent=2)
