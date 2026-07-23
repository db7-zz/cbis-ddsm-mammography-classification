from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from cbis_ddsm.dataset import MammogramDataset, build_transform
from cbis_ddsm.preprocessing import (
    apply_spatial_preprocessing,
    crop_breast_region,
    letterbox_image,
)


def synthetic_mammogram() -> Image.Image:
    image = Image.new("L", (100, 80), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 79, 69), fill=180)
    draw.rectangle((2, 2, 3, 3), fill=255)
    return image.convert("RGB")


def phase5_preprocessing(image_size: int = 256) -> dict[str, object]:
    return {
        "image_size": image_size,
        "spatial_mode": "breast_crop_letterbox",
        "breast_margin_fraction": 0.02,
        "min_foreground_fraction": 0.15,
        "max_foreground_fraction": 0.95,
        "min_breast_width_fraction": 0.25,
        "min_breast_height_fraction": 0.25,
    }


def test_crop_uses_largest_component_and_preserves_margin() -> None:
    result = crop_breast_region(synthetic_mammogram())

    assert result.fallback is False
    assert result.used_crop is True
    assert result.crop_box == (18, 8, 82, 72)
    assert result.image.size == (64, 64)
    assert result.foreground_fraction == 3600 / 8000


def test_empty_mask_falls_back_to_complete_image_with_reason() -> None:
    image = Image.new("RGB", (40, 60), color=0)
    result = crop_breast_region(image)

    assert result.fallback is True
    assert result.used_crop is False
    assert result.fallback_reason == "empty_foreground_mask"
    assert result.crop_box == (0, 0, 40, 60)
    assert result.image.size == image.size


def test_dense_scanner_frame_does_not_force_a_full_image_crop() -> None:
    image = Image.new("L", (100, 80), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 99, 79), outline=255, width=1)
    draw.rectangle((20, 10, 99, 69), fill=160)

    result = crop_breast_region(image.convert("RGB"))

    assert result.fallback is False
    assert result.used_crop is True
    assert result.crop_box == (18, 8, 100, 72)


def test_letterbox_preserves_aspect_ratio_and_centers_padding() -> None:
    image = Image.new("RGB", (40, 20), color=(200, 200, 200))
    output = letterbox_image(image, 100)
    pixels = np.asarray(output)

    assert output.size == (100, 100)
    assert np.all(pixels[:25] == 0)
    assert np.all(pixels[25:75] == 200)
    assert np.all(pixels[75:] == 0)


def test_phase5_preprocessing_is_deterministic() -> None:
    config = phase5_preprocessing(384)
    first = apply_spatial_preprocessing(synthetic_mammogram(), config)
    second = apply_spatial_preprocessing(synthetic_mammogram(), config)

    assert first.crop_box == second.crop_box
    assert first.fallback_reason == second.fallback_reason
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
    assert first.image.size == (384, 384)


def test_dataset_exposes_auditable_phase5_preprocessing_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "mammogram.png"
    synthetic_mammogram().save(image_path)
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "split": "val",
                "patient_id": "P_00001",
                "label": 1,
                "image_path": str(image_path),
            }
        ]
    )
    dataset = MammogramDataset(
        manifest,
        split="val",
        preprocessing=phase5_preprocessing(256),
    )

    first = dataset[0]
    second = dataset[0]
    assert first["image"].shape == (3, 256, 256)
    assert torch.equal(first["image"], second["image"])
    assert first["preprocessing_crop_box"].tolist() == [18, 8, 82, 72]
    assert first["preprocessing_fallback"] is False
    assert first["preprocessing_fallback_reason"] == ""


def test_safe_augmentation_is_reproducible_and_train_only() -> None:
    augmentation = {
        "affine_probability": 1.0,
        "affine_degrees": 5.0,
        "affine_translate_fraction": 0.03,
        "affine_scale_min": 0.95,
        "affine_scale_max": 1.05,
        "gamma_probability": 1.0,
        "gamma_min": 0.90,
        "gamma_max": 1.10,
        "contrast_probability": 1.0,
        "contrast_min": 0.90,
        "contrast_max": 1.10,
    }
    train_transform = build_transform(train=True, image_size=128, **augmentation)
    val_transform = build_transform(train=False, image_size=128, **augmentation)
    plain_transform = build_transform(train=False, image_size=128)

    torch.manual_seed(42)
    first = train_transform(synthetic_mammogram())
    torch.manual_seed(42)
    second = train_transform(synthetic_mammogram())

    assert torch.equal(first, second)
    assert not torch.equal(first, plain_transform(synthetic_mammogram()))
    assert torch.equal(
        val_transform(synthetic_mammogram()), plain_transform(synthetic_mammogram())
    )
