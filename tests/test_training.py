from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import cbis_ddsm.training as training_module
from cbis_ddsm.metrics import BinaryMetrics, compute_binary_metrics, sigmoid_numpy
from cbis_ddsm.training import (
    TrainingConfig,
    build_optimizer_scheduler,
    ensure_run_directory,
    load_checkpoint,
    load_training_config,
    load_training_manifests,
    run_epoch,
    save_checkpoint,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyDataset(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [-1.0, -0.5, 0.0, 0.5],
                [1.0, 0.5, 0.0, -0.5],
                [-0.8, -0.2, 0.1, 0.4],
                [0.8, 0.2, -0.1, -0.4],
            ]
        )
        self.labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "sample_id": f"sample-{index}",
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images).squeeze(-1)


def tiny_loader(*, shuffle: bool = False) -> DataLoader[dict[str, object]]:
    return DataLoader(TinyDataset(), batch_size=2, shuffle=shuffle)


def test_locked_training_config_matches_phase3_specification() -> None:
    config = load_training_config(PROJECT_ROOT / "configs" / "train.yaml")

    assert config.seed == 42
    assert config.max_epochs == 50
    assert config.batch_size == 16
    assert config.pos_weight == pytest.approx(1.84)
    assert config.optimizer_name == "AdamW"
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.scheduler_monitor == "val_roc_auc"
    assert config.early_stopping_patience == 10


def test_binary_metrics_use_logits_and_require_both_classes() -> None:
    labels = np.array([0, 0, 1, 1])
    logits = np.array([-3.0, -1.0, 1.0, 3.0])
    metrics = compute_binary_metrics(labels, logits)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert np.all((sigmoid_numpy(logits) > 0) & (sigmoid_numpy(logits) < 1))
    with pytest.raises(ValueError, match="both classes"):
        compute_binary_metrics(np.zeros(4), logits)


def manifest_frame(split: str, patients: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{split}-{index}" for index in range(len(patients))],
            "split": [split] * len(patients),
            "patient_id": patients,
            "label": [index % 2 for index in range(len(patients))],
            "image_path": [f"unused-{index}.dcm" for index in range(len(patients))],
        }
    )


def test_training_manifest_loader_never_requires_test_csv(tmp_path: Path) -> None:
    manifest_frame("train", ["P1", "P2", "P3", "P4"]).to_csv(
        tmp_path / "train.csv", index=False
    )
    manifest_frame("val", ["P5", "P6", "P7", "P8"]).to_csv(
        tmp_path / "val.csv", index=False
    )
    (tmp_path / "test.csv").mkdir()

    train, val = load_training_manifests(tmp_path)
    assert len(train) == 4
    assert len(val) == 4


def test_training_manifest_loader_rejects_patient_leakage(tmp_path: Path) -> None:
    manifest_frame("train", ["P1", "P2"]).to_csv(tmp_path / "train.csv", index=False)
    manifest_frame("val", ["P1", "P3"]).to_csv(tmp_path / "val.csv", index=False)

    with pytest.raises(ValueError, match="patient leakage"):
        load_training_manifests(tmp_path)


def test_run_epoch_updates_parameters_and_returns_all_samples() -> None:
    torch.manual_seed(42)
    model = TinyModel()
    config = replace(load_training_config(PROJECT_ROOT / "configs" / "train.yaml"), num_workers=0)
    optimizer, _ = build_optimizer_scheduler(model, config)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.84]))
    before = model.classifier.weight.detach().clone()

    output = run_epoch(
        model,
        tiny_loader(shuffle=False),
        criterion,
        torch.device("cpu"),
        optimizer=optimizer,
        mixed_precision=False,
    )

    assert output.samples == 4
    assert output.batches == 2
    assert output.labels.shape == (4,)
    assert output.logits.shape == (4,)
    assert not torch.equal(before, model.classifier.weight.detach())


def test_checkpoint_round_trip_restores_model_and_training_state(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = TinyModel()
    config = load_training_config(PROJECT_ROOT / "configs" / "train.yaml")
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "checkpoint.pt"
    hashes = {"train.csv": "train-hash", "val.csv": "val-hash"}
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        best_epoch=2,
        best_metric=0.75,
        epochs_without_improvement=1,
        training_config=config,
        model_config={"backbone": "tiny"},
        manifest_hashes=hashes,
    )
    with torch.no_grad():
        model.classifier.weight.add_(10.0)

    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_manifest_hashes=hashes,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_epoch"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_nonempty_run_directory_requires_explicit_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        ensure_run_directory(run_dir, resume=False)
    assert ensure_run_directory(run_dir, resume=True) == run_dir.resolve()


def test_tiny_end_to_end_training_writes_auditable_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(42)
    base = load_training_config(PROJECT_ROOT / "configs" / "train.yaml")
    config = replace(
        base,
        max_epochs=2,
        batch_size=2,
        num_workers=0,
        mixed_precision=False,
        tensorboard=False,
        early_stopping_patience=2,
    )
    model = TinyModel()
    hashes = {"train.csv": "train-hash", "val.csv": "val-hash"}
    metric_sequence = iter(
        [BinaryMetrics(roc_auc=0.6, pr_auc=0.5), BinaryMetrics(roc_auc=0.6005, pr_auc=0.5)]
    )
    monkeypatch.setattr(
        training_module,
        "compute_binary_metrics",
        lambda _labels, _logits: next(metric_sequence),
    )
    outcome = train_model(
        model,
        tiny_loader(shuffle=True),
        tiny_loader(shuffle=False),
        config=config,
        device=torch.device("cpu"),
        output_dir=tmp_path / "training",
        model_config={"backbone": "tiny"},
        manifest_hashes=hashes,
    )

    assert outcome.completed_epochs == 2
    assert outcome.best_epoch == 2
    assert outcome.best_val_roc_auc == pytest.approx(0.6005)
    for name in (
        "run_config.json",
        "metrics.csv",
        "best.pt",
        "last.pt",
        "val_predictions_best.csv",
    ):
        assert (outcome.output_dir / name).is_file()
    run_config = __import__("json").loads(
        (outcome.output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["test_split_loaded"] is False
    assert run_config["manifest_hashes"] == hashes
    best_checkpoint = torch.load(
        outcome.output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["epochs_without_improvement"] == 1
    assert best_checkpoint["early_stopping_reference"] == pytest.approx(0.6)
