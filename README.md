# CBIS-DDSM calcification baseline reproduction

This repository is a clean-room reproduction of the approved calcification-only,
whole-mammogram CBIS-DDSM binary classification baseline. Phase 1 contains the
patient-safe data pipeline. Phase 2 adds the structurally verified
EfficientNet-B0 plus Transformer classifier. Training remains intentionally out
of scope until phase 3.

The real-data phase 1 audit is recorded in
[`docs/phase1_data_audit.md`](docs/phase1_data_audit.md).
The locked model and explicit reproduction assumptions are recorded in
[`docs/phase2_model_spec.md`](docs/phase2_model_spec.md).

## Locked data policy

- Use only the official calcification description CSV files.
- Use `image file path` (whole mammogram), never `cropped image file path` or
  `ROI mask file path`.
- Treat each CC/MLO view as an independent sample.
- Map `MALIGNANT` to 1 and both `BENIGN` and
  `BENIGN_WITHOUT_CALLBACK` to 0.
- Keep official test patients in test.  Split official train patients into train
  and validation, with no patient appearing in more than one split.
- Use validation only for model selection and threshold selection.  Test is for
  the final evaluation only.

The choice "calcification-only + whole mammogram" is a **Reproduction
assumption** because the original experiment manifest and prediction CSV are not
available.  It is nevertheless the user-approved protocol for this reproduction.

## Environment

Create or activate a Conda environment named `surf`. From PowerShell:

```powershell
conda activate surf
python -m pytest
```

## Prepare manifests

Edit `configs/data.yaml`, or pass paths on the command line:

```powershell
python scripts/prepare_data.py `
  --config configs/data.yaml `
  --data-root "D:\path\to\CBIS-DDSM" `
  --train-csv "D:\path\to\calc_case_description_train_set.csv" `
  --test-csv "D:\path\to\calc_case_description_test_set.csv"
```

The command writes a unified `manifest.csv`, one CSV per split, and
`summary.json` under `artifacts/data`.  Missing or ambiguous image paths and any
patient overlap are fatal by default.

## Phase 1 reproduction assumptions

The target baseline specifies a 256 x 256 input but does not preserve the exact
historical preprocessing recipe.  Phase 1 therefore records these assumptions
explicitly:

- DICOM modality/VOI LUTs are applied when available.
- MONOCHROME1 images are inverted.
- Intensities are clipped to configurable per-image percentiles and scaled to
  8-bit before conversion to three channels.
- Images are resized directly to 256 x 256 and ImageNet-normalized.
- No random training augmentation is enabled by default because none is stated
  in the locked baseline specification.

These settings are centralized in `configs/data.yaml`; they are not evidence
about the unavailable original preprocessing.

## Model architecture smoke test

The smoke test defaults to random initialization and therefore performs no
network download:

```powershell
python scripts/smoke_model.py --device auto --batch-size 1
```

It verifies the `1280 x 8 x 8` B0 feature map, `512 x 8 x 8` projection, 64
spatial tokens, four-layer/eight-head Transformer, one-logit output, and a finite
backward pass. Pass `--pretrained` only when an ImageNet weight download is
intended. No optimizer step, checkpoint or metric evaluation occurs in phase 2.
