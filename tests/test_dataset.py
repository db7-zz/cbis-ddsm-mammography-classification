from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from cbis_ddsm.dataset import MammogramDataset, build_transform, load_mammogram


def test_dataset_returns_single_logit_compatible_label_and_256_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "mammogram.png"
    Image.new("L", (30, 50), color=128).save(image_path)
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "split": "train",
                "patient_id": "P_00001",
                "label": 1,
                "image_path": str(image_path),
            }
        ]
    )
    dataset = MammogramDataset(manifest, split="train")
    item = dataset[0]
    assert item["image"].shape == (3, 256, 256)
    assert item["image"].dtype == torch.float32
    assert item["label"].shape == torch.Size([])
    assert item["label"].dtype == torch.float32
    assert item["label"].item() == 1.0
    assert item["patient_id"] == "P_00001"


def test_default_train_transform_is_deterministic(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 30), color=(10, 20, 30))
    transform = build_transform(train=True, image_size=256)
    assert torch.equal(transform(image), transform(image))


def test_uncompressed_monochrome_dicom_loads_as_rgb(tmp_path: Path) -> None:
    path = tmp_path / "mammogram.dcm"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MG"
    dataset.ContentDate = datetime.now().strftime("%Y%m%d")
    dataset.ContentTime = datetime.now().strftime("%H%M%S")
    pixels = np.arange(80, dtype=np.uint16).reshape(8, 10)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME1"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)

    image = load_mammogram(path)
    assert image.mode == "RGB"
    assert image.size == (10, 8)
    converted = np.asarray(image)
    assert converted.shape == (8, 10, 3)
    assert converted[0, 0, 0] > converted[-1, -1, 0]
