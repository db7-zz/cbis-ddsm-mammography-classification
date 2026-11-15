# CBIS-DDSM calcification baseline reproduction

Completed (2026-11-12): the approved [V2S CNN-only supplement](docs/phase9_cnn_only_control_spec.md)
ran seeds 42/2026/3407 under the frozen phase 7 train/validation protocol.
Validation ROC AUC is **0.70404 ± 0.00365**; PR AUC is **0.57453 ± 0.01151** (sample SD).
Transformer ROC AUC gains were inconsistent across seeds, while its PR AUC was higher in all three.
See the [completed report](docs/phase9_progress_report.md). Phase 8 is complete;
this supplement does not access official test data or add test inference passes.
Entry points: `scripts/audit_phase9_cnn_only.py`, `scripts/run_phase9_experiment.py`
and `scripts/summarize_phase9_controls.py`. Existing P7-E01/P7-E04 results are reused.

This repository is a clean-room reproduction of the approved calcification-only,
whole-mammogram CBIS-DDSM binary classification baseline. Phase 1 contains the
patient-safe data pipeline. Phase 2 adds the structurally verified
EfficientNet-B0 plus Transformer classifier. Phase 3 adds leakage-safe,
checkpointed training and validation. Phase 4 adds a validation-frozen threshold
and one-time official test evaluation. Phase 5 adds validation-only
breast-region preprocessing, input-resolution ablation, safe-augmentation
screening, and multi-seed confirmation.

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

## Versioning and experimental development

Phases 1-4 form the immutable reference baseline and are tagged
`v1.0-baseline`. All later research changes are developed on dedicated
experimental branches; phase 5 is published on
`codex/phase-5-experiments`.

The baseline tag must not be moved or modified. Phase 5 does not repeat or use
the official test inference for model selection. Its conclusions are based only
on the locked training and validation manifests.

## Phase 5 validation-only experiments

Phase 5 evaluates deterministic breast-region crop plus letterbox
preprocessing, 256/384/512-pixel inputs, a conservative augmentation policy,
and three model seeds. The selected candidate is a 384-pixel input without
random augmentation.

| Seed | Validation ROC AUC | Validation PR AUC |
|---:|---:|---:|
| 42 | 0.699563 | 0.616203 |
| 2026 | 0.684779 | 0.580666 |
| 3407 | 0.669709 | 0.547121 |
| Mean | 0.684684 | 0.581330 |
| Sample standard deviation | 0.014927 | 0.034546 |

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
