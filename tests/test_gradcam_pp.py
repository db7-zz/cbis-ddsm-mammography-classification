from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cbis_ddsm.config import ConfigurationError
from cbis_ddsm.gradcam_pp import (
    compute_localization_metrics,
    generate_gradcam_pp,
    load_gradcam_pp_config,
    select_qualitative_samples,
    select_phase8_target_layer,
    transform_binary_mask,
)
from cbis_ddsm.phase8_xai import (
    clustered_mean_bootstrap,
    normalize_inference_preprocessing,
    paired_clustered_difference_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(
                torch.arange(108, dtype=torch.float32).reshape(4, 3, 3, 3)
                / 108.0
            )
            self.classifier.weight.fill_(0.25)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(images))
        return self.classifier(features.mean(dim=(-2, -1))).squeeze(-1)


def test_gradcam_pp_is_finite_normalized_and_deterministic() -> None:
    model = TinyClassifier()
    images = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
    first = generate_gradcam_pp(model, model.features, images)
    second = generate_gradcam_pp(model, model.features, images)
    assert first.target_shape == (1, 4, 8, 8)
    assert first.heatmaps.shape == (1, 8, 8)
    assert torch.isfinite(first.heatmaps).all()
    assert first.heatmaps.min().item() == pytest.approx(0.0)
    assert first.heatmaps.max().item() == pytest.approx(1.0)
    assert first.degenerate == (False,)
    assert torch.equal(first.heatmaps, second.heatmaps)


def test_gradcam_pp_excludes_invalid_padding_from_normalization() -> None:
    model = TinyClassifier()
    images = torch.ones(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 2:6, 2:6] = True
    result = generate_gradcam_pp(model, model.features, images, valid_mask=mask)
    assert torch.count_nonzero(result.heatmaps[~mask]) == 0


def test_localization_uses_exact_top_ten_percent_and_peak_hit() -> None:
    heatmap = np.arange(100, dtype=np.float64).reshape(10, 10)
    roi = np.zeros((10, 10), dtype=bool)
    roi[9, 5:10] = True
    metrics = compute_localization_metrics(heatmap, roi, top_fraction=0.10)
    assert metrics.selected_pixels == 10
    assert metrics.valid_pixels == 100
    assert metrics.lesion_hit is True
    assert metrics.iou == pytest.approx(0.5)
    assert metrics.dice == pytest.approx(2 * 5 / 15)
    assert 0.0 < metrics.energy_inside_roi < 1.0


def test_localization_rejects_degenerate_or_missing_roi() -> None:
    heatmap = np.ones((4, 4), dtype=np.float64)
    roi = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="no valid lesion"):
        compute_localization_metrics(heatmap, roi)
    roi[0, 0] = True
    with pytest.raises(ValueError, match="Degenerate"):
        compute_localization_metrics(heatmap, roi)


def test_phase8_target_layers_are_locked() -> None:
    projection = nn.Conv2d(4, 4, 1)
    mixer = nn.Conv2d(4, 4, 1)
    p5 = SimpleNamespace(channel_projection=projection)
    p7_e04 = SimpleNamespace(
        token_compressor=SimpleNamespace(mixer=mixer)
    )
    assert select_phase8_target_layer("P5-E02_seed42", p5) is projection
    assert (
        select_phase8_target_layer(
            "P7-E04_V2S_BiFPN_CBAM_seed42", p7_e04
        )
        is mixer
    )
    with pytest.raises(ConfigurationError):
        select_phase8_target_layer("P7-E02_V2S_BiFPN_R1_C128_seed42", p5)


def test_gradcam_pp_config_locks_models_layers_and_top_fraction() -> None:
    config = load_gradcam_pp_config(
        PROJECT_ROOT / "configs" / "phase8" / "gradcam_pp.yaml"
    )
    assert config.input_size == 384
    assert config.binary_top_fraction == pytest.approx(0.10)
    assert config.batch_size == 2
    assert config.bootstrap == {
        "unit": "patient_id",
        "valid_replicates": 2000,
        "seed": 42,
        "confidence_level": 0.95,
    }
    assert config.target_layers == {
        "P5-E02_seed42": "channel_projection",
        "P7-E01_V2S_seed42": "channel_projection",
        "P7-E04_V2S_BiFPN_CBAM_seed42": "token_compressor.mixer",
    }


def test_transform_binary_mask_uses_nearest_letterbox_geometry() -> None:
    mask = np.zeros((4, 2), dtype=bool)
    mask[1:3, 1] = True
    transformed = transform_binary_mask(
        mask,
        crop_box=(0, 0, 2, 4),
        output_size=4,
    )
    assert transformed.shape == (4, 4)
    assert transformed[:, 0].sum() == 0
    assert transformed[:, 3].sum() == 0
    assert transformed[1:3, 2].all()
    assert transformed.sum() == 2


def test_qualitative_panel_selection_is_hash_ordered_per_stratum() -> None:
    rows = []
    for label in (0, 1):
        for view in ("CC", "MLO"):
            for index in range(5):
                rows.append(
                    {
                        "sample_id": f"{label}-{view}-{index}",
                        "label": label,
                        "view": view,
                    }
                )
    manifest = pd.DataFrame(rows)
    selected = select_qualitative_samples(manifest, samples_per_stratum=3)
    assert len(selected) == 12
    assert set(selected.groupby(["label", "view"]).size()) == {3}
    repeated = select_qualitative_samples(manifest, samples_per_stratum=3)
    assert selected["sample_id"].tolist() == repeated["sample_id"].tolist()


def test_clustered_localization_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "c"],
            "iou": [0.1, 0.2, 0.6, 0.9],
            "lesion_hit": [0.0, 1.0, 1.0, 1.0],
        }
    )
    arguments = {
        "value_columns": ["iou", "lesion_hit"],
        "valid_replicates": 100,
        "seed": 42,
        "confidence_level": 0.95,
    }
    first = clustered_mean_bootstrap(frame, **arguments)
    second = clustered_mean_bootstrap(frame, **arguments)
    assert first == second
    assert first["unit"] == "patient_id"
    assert first["patients"] == 3
    assert first["metrics"]["iou"]["mean"] == pytest.approx(0.45)


def test_paired_localization_bootstrap_uses_common_images() -> None:
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "patient_id": ["p1", "p1", "p2"],
            "iou": [0.4, 0.6, 0.8],
            "lesion_hit": [1.0, 1.0, 0.0],
        }
    )
    second = first.copy()
    second["iou"] = [0.2, 0.3, 0.5]
    second["lesion_hit"] = [0.0, 1.0, 0.0]
    result = paired_clustered_difference_bootstrap(
        first,
        second,
        first_model_id="first",
        second_model_id="second",
        value_columns=["iou", "lesion_hit"],
        valid_replicates=100,
        seed=42,
        confidence_level=0.95,
    )
    assert result["images"] == 3
    assert result["patients"] == 2
    assert result["metrics"]["iou"]["mean_difference"] == pytest.approx(
        (0.2 + 0.3 + 0.3) / 3
    )


def test_inference_preprocessing_normalizes_missing_zero_train_fields() -> None:
    base = {
        "image_size": 384,
        "dicom_apply_modality_lut": True,
        "dicom_apply_voi_lut": True,
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
        "spatial_mode": "breast_crop_letterbox",
        "breast_margin_fraction": 0.02,
        "min_foreground_fraction": 0.15,
        "max_foreground_fraction": 0.95,
        "min_breast_width_fraction": 0.25,
        "min_breast_height_fraction": 0.25,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    with_explicit_zeros = {
        **base,
        "train_affine_probability": 0.0,
        "train_gamma_probability": 0.0,
        "train_contrast_probability": 0.0,
    }
    assert normalize_inference_preprocessing(base) == (
        normalize_inference_preprocessing(with_explicit_zeros)
    )
    with pytest.raises(ValueError, match="zero inference augmentation"):
        normalize_inference_preprocessing(
            {**base, "train_gamma_probability": 0.1}
        )
