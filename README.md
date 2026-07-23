# CBIS-DDSM phase 5 final validation baseline

This branch is the completed phase 5 result for the calcification-only,
whole-mammogram CBIS-DDSM binary classification project. It inherits the frozen
phase 1-4 reference baseline and adds validation-only breast-region
preprocessing, input-resolution ablation, safe-augmentation screening, and
multi-seed confirmation.

The real-data phase 1 audit is recorded in
[`docs/phase1_data_audit.md`](docs/phase1_data_audit.md).
The locked model and explicit reproduction assumptions are recorded in
[`docs/phase2_model_spec.md`](docs/phase2_model_spec.md).
The leakage-safe training protocol is recorded in
[`docs/phase3_training_spec.md`](docs/phase3_training_spec.md).
The completed seed-42 run is audited in
[`docs/phase3_training_report.md`](docs/phase3_training_report.md).
The locked final-evaluation protocol is recorded in
[`docs/phase4_evaluation_spec.md`](docs/phase4_evaluation_spec.md).
The completed one-time test result is audited in
[`docs/phase4_evaluation_report.md`](docs/phase4_evaluation_report.md).

## Branch role

Phases 1-4 remain frozen on `main` and are tagged `v1.0-baseline`. This
`phase-5-final` branch is the immutable starting point for phase 6 work.

The baseline tag must not be moved or modified. Phase 5 did not repeat or use
official test inference for model selection. All phase 5 conclusions are based
only on the locked training and validation manifests.

## Phase 5 completed work

- Completed a deterministic breast-region crop-and-letterbox pipeline with
  explicit fallback reasons.
- Audited all 1,227 locked train/validation images: 1,217 used the detected
  breast crop and 10 safely fell back to the complete image.
- Completed CUDA memory smoke tests for 256, 384, and 512-pixel inputs while
  keeping the effective batch size at 16.
- Completed P5-E01, P5-E02, and P5-E03 resolution screening. Validation ROC AUC
  selected the 384-pixel P5-E02 configuration.
- Completed P5-E04 safe-augmentation screening. It did not improve on P5-E02,
  so the final candidate uses no random augmentation.
- Completed three-seed confirmation using seeds 42, 2026, and 3407.
- Verified validation prediction IDs and recomputed ROC AUC and PR AUC from the
  saved predictions.
- Completed the project test suite with 59 passing tests and preserved the
  frozen phase 1-4 artifacts.
- Confirmed `test_split_loaded=false` throughout phase 5.

| Seed | Validation ROC AUC | Validation PR AUC |
|---:|---:|---:|
| 42 | 0.699563 | 0.616203 |
| 2026 | 0.684779 | 0.580666 |
| 3407 | 0.669709 | 0.547121 |
| Mean | 0.684684 | 0.581330 |
| Sample standard deviation | 0.014927 | 0.034546 |

The final phase 5 candidate is therefore the 384-pixel, crop-and-letterbox,
no-random-augmentation configuration. No new official test metric is claimed
for this branch.

The approved protocol is documented in
[`docs/phase5_preprocessing_resolution_spec.md`](docs/phase5_preprocessing_resolution_spec.md),
and the completed validation-only results are summarized in
[`docs/phase5_progress_report.md`](docs/phase5_progress_report.md).

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

## Phase 3 training

The training entry point reads only `artifacts/data/train.csv` and `val.csv`;
the test split is reserved for phase 4. A full configured run is:

```powershell
python scripts/train.py `
  --data-config configs/data.local.yaml `
  --model-config configs/model.yaml `
  --train-config configs/train.yaml `
  --device cuda
```

The first run can download the official torchvision EfficientNet-B0 ImageNet
weights. Training artifacts are written below `artifacts/training` and are not
committed to Git.

## Phase 4 final evaluation

First, the validation-only workflow can be rehearsed without loading test data:

```powershell
python scripts/evaluate.py `
  --threshold-only `
  --output-dir artifacts/evaluation/threshold_rehearsal
```

The formal command is run only once, after tests and the threshold-only rehearsal
pass. It freezes Youden's J threshold before the program can read `test.csv`:

```powershell
python scripts/evaluate.py --device cuda
```
