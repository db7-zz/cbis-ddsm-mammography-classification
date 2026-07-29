"""Dependency-free Grad-CAM++ utilities for the frozen phase 8 models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor, nn

from .config import ConfigurationError


PHASE8_GRADCAM_MODEL_IDS = (
    "P5-E02_seed42",
    "P7-E01_V2S_seed42",
    "P7-E04_V2S_BiFPN_CBAM_seed42",
)


@dataclass(frozen=True)
class GradCAMPPResult:
    logits: Tensor
    heatmaps: Tensor
    degenerate: tuple[bool, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class LocalizationMetrics:
    iou: float
    dice: float
    lesion_hit: bool
    energy_inside_roi: float
    selected_pixels: int
    valid_pixels: int


@dataclass(frozen=True)
class GradCAMPPConfig:
    source_path: Path
    input_size: int
    binary_top_fraction: float
    batch_size: int
    geometry_audit_output_dir: str
    formal_output_dir: str
    bootstrap: dict[str, Any]
    visualization: dict[str, Any]
    target_layers: dict[str, str]
    expected_shapes: dict[str, tuple[int, int, int]]
    qualitative_panel: dict[str, Any]
    localization_metrics: tuple[str, ...]


def load_gradcam_pp_config(path: str | Path) -> GradCAMPPConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ConfigurationError("Phase 8 Grad-CAM++ configuration must be a mapping.")
    if int(loaded.get("schema_version", 0)) != 1 or int(loaded.get("stage", 0)) != 8:
        raise ConfigurationError("Phase 8 Grad-CAM++ requires schema_version=1 and stage=8.")
    if loaded.get("target_logit") != "malignant":
        raise ConfigurationError("Phase 8 Grad-CAM++ must target the malignant logit.")
    if loaded.get("precision") != "float32":
        raise ConfigurationError("Phase 8 Grad-CAM++ locks precision to float32.")
    if loaded.get("normalization_region") != "valid_breast_without_letterbox_padding":
        raise ConfigurationError("Phase 8 Grad-CAM++ normalization region changed.")
    input_size = int(loaded.get("input_size", 0))
    fraction = float(loaded.get("binary_top_fraction", 0.0))
    if input_size != 384 or fraction != 0.10:
        raise ConfigurationError("Phase 8 Grad-CAM++ locks 384 input and top 10%.")
    if loaded.get("formal_split") != "official_test":
        raise ConfigurationError("Phase 8 Grad-CAM++ formal split must be official_test.")
    if int(loaded.get("localization_label", -1)) != 1:
        raise ConfigurationError("Phase 8 Grad-CAM++ localizes malignant images only.")
    batch_size = int(loaded.get("batch_size", 0))
    if batch_size != 2:
        raise ConfigurationError("Phase 8 Grad-CAM++ locks batch_size=2.")
    geometry_output = str(loaded.get("geometry_audit_output_dir", "")).strip()
    formal_output = str(loaded.get("formal_output_dir", "")).strip()
    if (
        geometry_output
        != "artifacts/experiments/phase8/gradcam_pp_geometry_audit_attempt2"
        or formal_output != "artifacts/experiments/phase8/gradcam_pp_official_test"
    ):
        raise ConfigurationError("Phase 8 Grad-CAM++ output directories changed.")
    bootstrap = loaded.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or dict(bootstrap) != {
        "unit": "patient_id",
        "valid_replicates": 2000,
        "seed": 42,
        "confidence_level": 0.95,
    }:
        raise ConfigurationError("Phase 8 Grad-CAM++ bootstrap policy changed.")
    visualization = loaded.get("visualization")
    if not isinstance(visualization, Mapping) or dict(visualization) != {
        "colormap": "magma",
        "overlay_alpha": 0.45,
    }:
        raise ConfigurationError("Phase 8 Grad-CAM++ visualization policy changed.")
    raw_models = loaded.get("models")
    if not isinstance(raw_models, Mapping) or tuple(raw_models) != PHASE8_GRADCAM_MODEL_IDS:
        raise ConfigurationError("Phase 8 Grad-CAM++ model matrix changed.")
    expected_targets = {
        "P5-E02_seed42": "channel_projection",
        "P7-E01_V2S_seed42": "channel_projection",
        "P7-E04_V2S_BiFPN_CBAM_seed42": "token_compressor.mixer",
    }
    target_layers: dict[str, str] = {}
    expected_shapes: dict[str, tuple[int, int, int]] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, Mapping) or set(raw) != {"target_layer", "expected_shape"}:
            raise ConfigurationError(f"Grad-CAM++ fields changed for {model_id}.")
        target_layer = str(raw["target_layer"])
        shape = tuple(int(value) for value in raw["expected_shape"])
        if target_layer != expected_targets[model_id] or shape != (512, 12, 12):
            raise ConfigurationError(f"Grad-CAM++ target changed for {model_id}.")
        target_layers[model_id] = target_layer
        expected_shapes[model_id] = shape
    qualitative = loaded.get("qualitative_panel")
    if not isinstance(qualitative, Mapping) or dict(qualitative) != {
        "strata": ["label", "view"],
        "samples_per_stratum": 3,
        "ordering": "sha256_sample_id_ascending",
    }:
        raise ConfigurationError("Phase 8 qualitative panel selection changed.")
    metrics = tuple(str(value) for value in loaded.get("localization_metrics", ()))
    if metrics != ("iou", "dice", "lesion_hit_rate", "energy_inside_roi"):
        raise ConfigurationError("Phase 8 localization metrics changed.")
    return GradCAMPPConfig(
        source_path=source,
        input_size=input_size,
        binary_top_fraction=fraction,
        batch_size=batch_size,
        geometry_audit_output_dir=geometry_output,
        formal_output_dir=formal_output,
        bootstrap=dict(bootstrap),
        visualization=dict(visualization),
        target_layers=target_layers,
        expected_shapes=expected_shapes,
        qualitative_panel=dict(qualitative),
        localization_metrics=metrics,
    )


def transform_binary_mask(
    mask: Tensor | np.ndarray,
    *,
    crop_box: tuple[int, int, int, int],
    output_size: int,
) -> np.ndarray:
    """Apply the locked crop and nearest-neighbor letterbox geometry to a mask."""

    values = np.asarray(
        mask.detach().cpu().numpy() if isinstance(mask, Tensor) else mask,
        dtype=bool,
    )
    if values.ndim != 2:
        raise ValueError("Binary mask geometry expects a two-dimensional array.")
    left, top, right, bottom = (int(value) for value in crop_box)
    height, width = values.shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Crop box is outside the binary mask.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")
    cropped = values[top:bottom, left:right]
    cropped_height, cropped_width = cropped.shape
    scale = output_size / max(cropped_width, cropped_height)
    resized_width = max(1, min(output_size, int(round(cropped_width * scale))))
    resized_height = max(1, min(output_size, int(round(cropped_height * scale))))
    image = Image.fromarray(cropped.astype(np.uint8) * 255, mode="L")
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )
    transformed = np.zeros((output_size, output_size), dtype=bool)
    offset_left = (output_size - resized_width) // 2
    offset_top = (output_size - resized_height) // 2
    transformed[
        offset_top : offset_top + resized_height,
        offset_left : offset_left + resized_width,
    ] = np.asarray(resized, dtype=np.uint8) > 0
    return transformed


def select_qualitative_samples(
    manifest: pd.DataFrame,
    *,
    samples_per_stratum: int = 3,
) -> pd.DataFrame:
    """Select the fixed label-by-view panel using SHA-256(sample_id) ordering."""

    required = {"sample_id", "label", "view"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Qualitative selection is missing: {', '.join(missing)}")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive.")
    selected: list[pd.DataFrame] = []
    for label in (0, 1):
        for view in ("CC", "MLO"):
            subset = manifest[
                (pd.to_numeric(manifest["label"], errors="raise").astype(int) == label)
                & (manifest["view"].astype(str).str.upper() == view)
            ].copy()
            subset["_sample_hash"] = subset["sample_id"].astype(str).map(
                lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
            )
            subset = subset.sort_values(
                ["_sample_hash", "sample_id"],
                kind="stable",
            ).head(samples_per_stratum)
            subset["panel_stratum"] = f"label_{label}_{view}"
            subset["panel_rank"] = np.arange(1, len(subset) + 1)
            selected.append(subset)
    result = pd.concat(selected, ignore_index=True)
    return result.drop(columns=["_sample_hash"])


def select_phase8_target_layer(model_id: str, model: nn.Module) -> nn.Module:
    if model_id in {"P5-E02_seed42", "P7-E01_V2S_seed42"}:
        layer = getattr(model, "channel_projection", None)
    elif model_id == "P7-E04_V2S_BiFPN_CBAM_seed42":
        compressor = getattr(model, "token_compressor", None)
        layer = getattr(compressor, "mixer", None)
    else:
        raise ConfigurationError(f"Unsupported phase 8 Grad-CAM++ model: {model_id}")
    if not isinstance(layer, nn.Module):
        raise ConfigurationError(f"Grad-CAM++ target layer is unavailable for {model_id}.")
    return layer


def _normalize_heatmaps(
    heatmaps: Tensor,
    valid_mask: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, tuple[bool, ...]]:
    normalized = torch.zeros_like(heatmaps)
    degenerate: list[bool] = []
    for index in range(heatmaps.shape[0]):
        mask = valid_mask[index]
        values = heatmaps[index][mask]
        if values.numel() == 0:
            raise ValueError("Grad-CAM++ valid mask contains no pixels.")
        minimum = values.min()
        maximum = values.max()
        is_degenerate = bool((maximum - minimum).abs().item() <= epsilon)
        degenerate.append(is_degenerate)
        if not is_degenerate:
            normalized[index][mask] = (values - minimum) / (maximum - minimum)
    return normalized, tuple(degenerate)


def generate_gradcam_pp(
    model: nn.Module,
    target_layer: nn.Module,
    images: Tensor,
    *,
    valid_mask: Tensor | None = None,
    epsilon: float = 1e-8,
) -> GradCAMPPResult:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Grad-CAM++ expects images shaped [batch, 3, height, width].")
    if not images.is_floating_point():
        raise ValueError("Grad-CAM++ images must be floating point.")
    if epsilon <= 0.0:
        raise ValueError("Grad-CAM++ epsilon must be positive.")
    batch, _, height, width = images.shape
    if valid_mask is None:
        mask = torch.ones(
            (batch, height, width), dtype=torch.bool, device=images.device
        )
    else:
        if tuple(valid_mask.shape) != (batch, height, width):
            raise ValueError("Grad-CAM++ valid mask shape does not match the images.")
        mask = valid_mask.to(device=images.device, dtype=torch.bool)

    captured: dict[str, Tensor] = {}

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("Grad-CAM++ target layer must produce a 4D tensor.")
        captured["activation"] = output

    handle = target_layer.register_forward_hook(capture_activation)
    model.eval()
    inputs = images.detach().to(dtype=torch.float32).requires_grad_(True)
    try:
        outputs = model(inputs)
        logits = outputs.reshape(-1)
        if logits.shape != (batch,):
            raise RuntimeError("Grad-CAM++ requires one malignant logit per image.")
        activation = captured.get("activation")
        if activation is None:
            raise RuntimeError("Grad-CAM++ target layer was not reached.")
        gradients = torch.autograd.grad(
            logits.sum(),
            activation,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    finally:
        handle.remove()

    gradient_squared = gradients.pow(2)
    gradient_cubed = gradient_squared * gradients
    denominator = 2.0 * gradient_squared + (
        activation * gradient_cubed.sum(dim=(-2, -1), keepdim=True)
    )
    safe_denominator = torch.where(
        denominator.abs() > epsilon,
        denominator,
        torch.ones_like(denominator),
    )
    alpha = gradient_squared / safe_denominator
    alpha = torch.where(denominator.abs() > epsilon, alpha, torch.zeros_like(alpha))
    weights = (alpha * gradients.clamp_min(0.0)).sum(dim=(-2, -1), keepdim=True)
    coarse = (weights * activation).sum(dim=1).clamp_min(0.0)
    upsampled = F.interpolate(
        coarse.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    normalized, degenerate = _normalize_heatmaps(
        upsampled, mask, epsilon=epsilon
    )
    if not torch.isfinite(normalized).all() or not torch.isfinite(logits).all():
        raise FloatingPointError("Grad-CAM++ produced non-finite values.")
    return GradCAMPPResult(
        logits=logits.detach(),
        heatmaps=normalized.detach(),
        degenerate=degenerate,
        target_shape=tuple(activation.shape),
    )


def compute_localization_metrics(
    heatmap: Tensor | np.ndarray,
    roi_mask: Tensor | np.ndarray,
    *,
    valid_mask: Tensor | np.ndarray | None = None,
    top_fraction: float = 0.10,
) -> LocalizationMetrics:
    values = np.asarray(
        heatmap.detach().cpu().numpy() if isinstance(heatmap, Tensor) else heatmap,
        dtype=np.float64,
    )
    roi = np.asarray(
        roi_mask.detach().cpu().numpy() if isinstance(roi_mask, Tensor) else roi_mask,
        dtype=bool,
    )
    if values.ndim != 2 or roi.shape != values.shape:
        raise ValueError("Heatmap and ROI mask must be equally shaped 2D arrays.")
    if valid_mask is None:
        valid = np.ones_like(roi, dtype=bool)
    else:
        valid = np.asarray(
            valid_mask.detach().cpu().numpy()
            if isinstance(valid_mask, Tensor)
            else valid_mask,
            dtype=bool,
        )
        if valid.shape != values.shape:
            raise ValueError("Valid mask shape does not match the heatmap.")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    if not np.isfinite(values[valid]).all() or not valid.any():
        raise ValueError("Heatmap must be finite on a non-empty valid region.")
    if not (roi & valid).any():
        raise ValueError("ROI mask contains no valid lesion pixels.")
    valid_values = values[valid]
    if float(valid_values.max() - valid_values.min()) <= 1e-12:
        raise ValueError("Degenerate heatmap cannot produce localization metrics.")

    valid_indices = np.flatnonzero(valid.reshape(-1))
    count = max(1, int(np.ceil(top_fraction * valid_indices.size)))
    order = np.argsort(-values.reshape(-1)[valid_indices], kind="stable")
    selected_indices = valid_indices[order[:count]]
    selected = np.zeros(values.size, dtype=bool)
    selected[selected_indices] = True
    selected = selected.reshape(values.shape)
    effective_roi = roi & valid
    intersection = int((selected & effective_roi).sum())
    union = int((selected | effective_roi).sum())
    selected_count = int(selected.sum())
    roi_count = int(effective_roi.sum())
    maximum_index = valid_indices[
        int(np.argmax(values.reshape(-1)[valid_indices]))
    ]
    lesion_hit = bool(effective_roi.reshape(-1)[maximum_index])
    denominator = float(values[valid].sum())
    energy = (
        float(values[effective_roi].sum()) / denominator
        if denominator > 0.0
        else 0.0
    )
    return LocalizationMetrics(
        iou=float(intersection / union) if union else 0.0,
        dice=float(2 * intersection / (selected_count + roi_count)),
        lesion_hit=lesion_hit,
        energy_inside_roi=energy,
        selected_pixels=selected_count,
        valid_pixels=int(valid.sum()),
    )
