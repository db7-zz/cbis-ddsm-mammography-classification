from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cbis_ddsm.evaluation import (
    EvaluationConfig,
    compute_evaluation_metrics,
    freeze_threshold,
    load_test_manifest_after_threshold,
    read_frozen_threshold,
    run_inference,
    select_youden_threshold,
    validate_no_patient_or_sample_overlap,
)


def manifest(split: str, patients: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{split}-{index}" for index in range(len(patients))],
            "split": split,
            "patient_id": patients,
            "label": [index % 2 for index in range(len(patients))],
            "image_path": [f"unused-{index}.dcm" for index in range(len(patients))],
        }
    )


def test_evaluation_config_enforces_locked_threshold_policy() -> None:
    EvaluationConfig().validate()
    with pytest.raises(ValueError, match="higher threshold"):
        EvaluationConfig(threshold_tie_break="lower_threshold").validate()
    with pytest.raises(ValueError, match="youden_j"):
        EvaluationConfig(threshold_method="fixed").validate()


def test_youden_selection_uses_higher_threshold_on_exact_tie() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6])
    selection = select_youden_threshold(labels, probabilities)
    assert selection.youden_j == pytest.approx(0.0)
    assert selection.threshold == pytest.approx(0.8)
    assert selection.tie_count == 2
    assert selection.candidate_count == 4


def test_evaluation_metrics_use_frozen_threshold() -> None:
    metrics = compute_evaluation_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.7, 0.9]),
        threshold=0.75,
    )
    assert (metrics.tn, metrics.fp, metrics.fn, metrics.tp) == (1, 1, 1, 1)
    assert metrics.sensitivity == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.pr_auc == pytest.approx(5 / 6)


def test_test_manifest_is_blocked_until_threshold_is_frozen(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest("test", ["p1", "p2", "p3", "p4"]).to_csv(
        data_dir / "test.csv", index=False
    )
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    threshold_path = output_dir / "threshold.json"
    with pytest.raises(PermissionError, match="blocked"):
        load_test_manifest_after_threshold(
            data_dir, threshold_path, expected_samples=4
        )

    selection = select_youden_threshold([0, 1], [0.2, 0.8])
    freeze_threshold(
        output_dir,
        selection,
        provenance={"validation_samples": 2},
        reference_threshold=0.4515,
    )
    frozen = read_frozen_threshold(threshold_path)
    assert frozen["frozen"] is True
    assert frozen["test_split_loaded_at_freeze_time"] is False
    loaded = load_test_manifest_after_threshold(
        data_dir, threshold_path, expected_samples=4
    )
    assert len(loaded) == 4
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_threshold(
            output_dir,
            selection,
            provenance={},
            reference_threshold=0.4515,
        )


def test_invalid_threshold_document_does_not_unlock_test(tmp_path: Path) -> None:
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"frozen": False}), encoding="utf-8")
    with pytest.raises(PermissionError, match="valid frozen"):
        read_frozen_threshold(threshold_path)


def test_all_split_overlap_is_rejected() -> None:
    train = manifest("train", ["p1", "p2"])
    val = manifest("val", ["p3", "p4"])
    test = manifest("test", ["p5", "p1"])
    with pytest.raises(ValueError, match="train/test"):
        validate_no_patient_or_sample_overlap(train, val, test)


class TinyInferenceDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": torch.full((3, 2, 2), float(index)),
            "label": torch.tensor(float(index % 2)),
            "sample_id": f"sample-{index}",
            "patient_id": f"patient-{index}",
            "image_path": f"image-{index}.dcm",
        }


class MeanModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))


def test_run_inference_preserves_order_and_metadata() -> None:
    loader = DataLoader(TinyInferenceDataset(), batch_size=2, shuffle=False)
    output = run_inference(
        MeanModel(), loader, torch.device("cpu"), mixed_precision=True
    )
    assert output.sample_ids == ("sample-0", "sample-1", "sample-2")
    assert output.patient_ids == ("patient-0", "patient-1", "patient-2")
    assert np.array_equal(output.labels, np.array([0, 1, 0]))
    assert np.allclose(output.logits, np.array([0.0, 1.0, 2.0]))
