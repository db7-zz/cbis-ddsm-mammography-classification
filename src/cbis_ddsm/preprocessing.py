"""Deterministic spatial preprocessing for phase 5 whole mammograms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class SpatialPreprocessingResult:
    """Processed image and auditable geometry decisions for one sample."""

    image: Image.Image
    crop_box: tuple[int, int, int, int]
    foreground_fraction: float
    used_crop: bool
    fallback: bool
    fallback_reason: str | None


def _clear_dense_border_lines(
    foreground: np.ndarray,
    *,
    border_fraction: float = 0.02,
    dense_fraction: float = 0.90,
) -> np.ndarray:
    """Remove scanner-frame lines confined to the safety-margin band."""
    cleaned = foreground.copy()
    height, width = cleaned.shape
    border_rows = max(1, math.ceil(height * border_fraction))
    border_columns = max(1, math.ceil(width * border_fraction))
    for row in list(range(border_rows)) + list(range(height - border_rows, height)):
        if foreground[row].mean() >= dense_fraction:
            cleaned[row] = False
    for column in list(range(border_columns)) + list(
        range(width - border_columns, width)
    ):
        if foreground[:, column].mean() >= dense_fraction:
            cleaned[:, column] = False
    return cleaned


def _largest_foreground_component(grayscale: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected nonzero component."""
    if grayscale.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")
    foreground = _clear_dense_border_lines(np.asarray(grayscale) > 0)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=bool)
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(foreground.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labels == largest_label


def letterbox_image(
    image: Image.Image,
    size: int,
    *,
    fill: int | tuple[int, int, int] = 0,
) -> Image.Image:
    """Resize the long edge to ``size`` and center-pad without distortion."""
    target = int(size)
    if target <= 0:
        raise ValueError("Letterbox size must be positive.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot letterbox an empty image.")
    scale = target / max(width, height)
    resized_width = max(1, min(target, int(round(width * scale))))
    resized_height = max(1, min(target, int(round(height * scale))))
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new(image.mode, (target, target), color=fill)
    left = (target - resized_width) // 2
    top = (target - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def crop_breast_region(
    image: Image.Image,
    *,
    margin_fraction: float = 0.02,
    min_foreground_fraction: float = 0.15,
    max_foreground_fraction: float = 0.95,
    min_width_fraction: float = 0.25,
    min_height_fraction: float = 0.25,
) -> SpatialPreprocessingResult:
    """Crop the largest foreground component or explicitly fall back to the full image."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    full_box = (0, 0, width, height)
    if width <= 0 or height <= 0:
        raise ValueError("Mammogram dimensions must be positive.")
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError("margin_fraction must be in [0, 0.5).")
    if not 0.0 <= min_foreground_fraction < max_foreground_fraction <= 1.0:
        raise ValueError("Foreground fractions must satisfy 0 <= min < max <= 1.")
    if not 0.0 < min_width_fraction <= 1.0 or not 0.0 < min_height_fraction <= 1.0:
        raise ValueError("Minimum dimension fractions must be in (0, 1].")

    grayscale = np.asarray(rgb.convert("L"), dtype=np.uint8)
    component = _largest_foreground_component(grayscale)
    foreground_pixels = int(component.sum())
    if foreground_pixels == 0:
        return SpatialPreprocessingResult(
            image=rgb,
            crop_box=full_box,
            foreground_fraction=0.0,
            used_crop=False,
            fallback=True,
            fallback_reason="empty_foreground_mask",
        )

    foreground_fraction = foreground_pixels / float(width * height)
    if foreground_fraction < min_foreground_fraction:
        reason = "foreground_fraction_below_minimum"
    elif foreground_fraction > max_foreground_fraction:
        reason = "foreground_fraction_above_maximum"
    else:
        reason = None

    rows, columns = np.nonzero(component)
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    detected_width = right - left
    detected_height = bottom - top
    if reason is None and detected_width < math.ceil(width * min_width_fraction):
        reason = "foreground_width_below_minimum"
    if reason is None and detected_height < math.ceil(height * min_height_fraction):
        reason = "foreground_height_below_minimum"
    if reason is not None:
        return SpatialPreprocessingResult(
            image=rgb,
            crop_box=full_box,
            foreground_fraction=foreground_fraction,
            used_crop=False,
            fallback=True,
            fallback_reason=reason,
        )

    horizontal_margin = math.ceil(width * margin_fraction)
    vertical_margin = math.ceil(height * margin_fraction)
    crop_box = (
        max(0, left - horizontal_margin),
        max(0, top - vertical_margin),
        min(width, right + horizontal_margin),
        min(height, bottom + vertical_margin),
    )
    if not (
        crop_box[0] <= left
        and crop_box[1] <= top
        and crop_box[2] >= right
        and crop_box[3] >= bottom
    ):
        raise RuntimeError("Expanded crop does not contain the detected foreground.")
    return SpatialPreprocessingResult(
        image=rgb.crop(crop_box),
        crop_box=crop_box,
        foreground_fraction=foreground_fraction,
        used_crop=crop_box != full_box,
        fallback=False,
        fallback_reason=None,
    )


def apply_spatial_preprocessing(
    image: Image.Image,
    config: Mapping[str, Any],
) -> SpatialPreprocessingResult:
    """Apply the configured baseline stretch or phase 5 crop-and-letterbox path."""
    size = int(config.get("image_size", 256))
    mode = str(config.get("spatial_mode", "stretch"))
    rgb = image.convert("RGB")
    full_box = (0, 0, rgb.width, rgb.height)
    if mode == "stretch":
        return SpatialPreprocessingResult(
            image=rgb.resize((size, size), resample=Image.Resampling.BILINEAR),
            crop_box=full_box,
            foreground_fraction=1.0,
            used_crop=False,
            fallback=False,
            fallback_reason=None,
        )
    if mode != "breast_crop_letterbox":
        raise ValueError(f"Unsupported spatial preprocessing mode: {mode}")
    cropped = crop_breast_region(
        rgb,
        margin_fraction=float(config.get("breast_margin_fraction", 0.02)),
        min_foreground_fraction=float(config.get("min_foreground_fraction", 0.15)),
        max_foreground_fraction=float(config.get("max_foreground_fraction", 0.95)),
        min_width_fraction=float(config.get("min_breast_width_fraction", 0.25)),
        min_height_fraction=float(config.get("min_breast_height_fraction", 0.25)),
    )
    return SpatialPreprocessingResult(
        image=letterbox_image(cropped.image, size, fill=(0, 0, 0)),
        crop_box=cropped.crop_box,
        foreground_fraction=cropped.foreground_fraction,
        used_crop=cropped.used_crop,
        fallback=cropped.fallback,
        fallback_reason=cropped.fallback_reason,
    )
