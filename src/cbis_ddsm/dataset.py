"""Whole-mammogram loading and transforms for phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _scale_to_uint8(array: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    pixels = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError("Image contains no finite pixels.")
    values = pixels[finite]
    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    if upper <= lower:
        lower = float(values.min())
        upper = float(values.max())
    if upper <= lower:
        return np.zeros(pixels.shape, dtype=np.uint8)
    pixels = np.nan_to_num(pixels, nan=lower, posinf=upper, neginf=lower)
    pixels = np.clip(pixels, lower, upper)
    pixels = (pixels - lower) / (upper - lower)
    return np.round(pixels * 255.0).astype(np.uint8)


def _load_dicom(
    path: Path,
    *,
    apply_modality: bool,
    apply_voi: bool,
    lower_percentile: float,
    upper_percentile: float,
) -> Image.Image:
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError as exc:
        raise RuntimeError(
            "DICOM input requires pydicom. Install the project dependencies in the surf environment."
        ) from exc

    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    if apply_modality:
        pixels = apply_modality_lut(pixels, dataset)
    if apply_voi:
        try:
            pixels = apply_voi_lut(pixels, dataset)
        except (IndexError, TypeError, ValueError):
            # Some files have incomplete VOI metadata; raw modality pixels remain valid.
            pass
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim > 2:
        pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D mammogram, got shape {pixels.shape} from {path}")
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        finite = np.isfinite(pixels)
        maximum = float(pixels[finite].max()) if finite.any() else 0.0
        minimum = float(pixels[finite].min()) if finite.any() else 0.0
        pixels = maximum + minimum - pixels
    image = _scale_to_uint8(pixels, lower_percentile, upper_percentile)
    return Image.fromarray(image, mode="L").convert("RGB")


def load_mammogram(
    path: str | Path,
    *,
    dicom_apply_modality_lut: bool = True,
    dicom_apply_voi_lut: bool = True,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Image.Image:
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Mammogram does not exist: {image_path}")
    if image_path.suffix.lower() in {".dcm", ".dicom"}:
        return _load_dicom(
            image_path,
            apply_modality=dicom_apply_modality_lut,
            apply_voi=dicom_apply_voi_lut,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
    with Image.open(image_path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def build_transform(
    *,
    train: bool,
    image_size: int = 256,
    horizontal_flip_probability: float = 0.0,
    rotation_degrees: float = 0.0,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Image.Image], torch.Tensor]:
    operations: list[Callable[..., Any]] = [
        transforms.Resize(
            (int(image_size), int(image_size)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    ]
    if train and horizontal_flip_probability > 0.0:
        operations.append(transforms.RandomHorizontalFlip(p=horizontal_flip_probability))
    if train and rotation_degrees > 0.0:
        operations.append(
            transforms.RandomRotation(rotation_degrees, interpolation=InterpolationMode.BILINEAR)
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)])
    return transforms.Compose(operations)


class MammogramDataset(Dataset[dict[str, Any]]):
    """Dataset backed by the phase 1 manifest."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        split: str | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        preprocessing: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
        else:
            frame = pd.read_csv(Path(manifest), dtype={"patient_id": str, "sample_id": str})
        required = {"sample_id", "split", "patient_id", "label", "image_path"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unsupported split: {split}")
            frame = frame[frame["split"] == split]
        self.frame = frame.reset_index(drop=True)
        self.preprocessing = preprocessing or {}
        self.transform = transform or build_transform(
            train=split == "train",
            image_size=int(self.preprocessing.get("image_size", 256)),
            horizontal_flip_probability=float(
                self.preprocessing.get("train_horizontal_flip_probability", 0.0)
            ),
            rotation_degrees=float(self.preprocessing.get("train_rotation_degrees", 0.0)),
            imagenet_mean=tuple(self.preprocessing.get("imagenet_mean", (0.485, 0.456, 0.406))),
            imagenet_std=tuple(self.preprocessing.get("imagenet_std", (0.229, 0.224, 0.225))),
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = load_mammogram(
            row["image_path"],
            dicom_apply_modality_lut=bool(
                self.preprocessing.get("dicom_apply_modality_lut", True)
            ),
            dicom_apply_voi_lut=bool(self.preprocessing.get("dicom_apply_voi_lut", True)),
            lower_percentile=float(self.preprocessing.get("lower_percentile", 0.5)),
            upper_percentile=float(self.preprocessing.get("upper_percentile", 99.5)),
        )
        return {
            "image": self.transform(image),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "image_path": str(row["image_path"]),
        }
