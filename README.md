# CBIS-DDSM calcification baseline reproduction

This repository is a clean-room reproduction of the approved calcification-only,
whole-mammogram CBIS-DDSM binary classification baseline. Phase 1 contains the
patient-safe data pipeline. Phase 2 adds the structurally verified
EfficientNet-B0 plus Transformer classifier. Phase 3 adds leakage-safe,
checkpointed training and validation. Phase 4 adds a validation-frozen threshold
and one-time official test evaluation.

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

## 当前对话迁移交接入口

阶段一至阶段四已经完成，阶段四正式产物已经冻结。阶段五目前只有待用户审核的规划，**尚未批准实施**；接手后不要直接训练、安装依赖或再次运行官方测试集。

新的 Codex 对话应按顺序阅读和核验：

1. [`AGENTS.md`](AGENTS.md)：仓库级操作边界；
2. [`docs/HANDOFF.md`](docs/HANDOFF.md)：完整中文交接说明；
3. [`docs/handoff_state.yaml`](docs/handoff_state.yaml)：机器可读的真实状态、指标与哈希；
4. [`docs/research_plan_v2_post_baseline.md`](docs/research_plan_v2_post_baseline.md)：后基线研究路线；
5. [`docs/phase5_preprocessing_resolution_spec.md`](docs/phase5_preprocessing_resolution_spec.md)：待审核的阶段五 spec。

只读交接核验命令：

```powershell
& 'C:\nonsystem\DevTools\conda\envs\surf\python.exe' scripts\verify_handoff.py
```

详细接手步骤、正式指标、冻结策略、已知问题和未提交文件边界均以交接文档与状态文件为准。

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
