# Phase 4 final evaluation specification

## Scope

Phase 4 evaluates the frozen phase 3 numerical-best checkpoint on the official
CBIS-DDSM calcification test split. It does not train, tune, select an epoch, or
change preprocessing. The reference threshold and metrics are comparison values
only and cannot affect predictions.

## Validation-only threshold selection

The input is `val_predictions_best.csv` from phase 3 (241 samples, epoch 12).
The script recomputes probabilities from logits, verifies sample IDs and labels
against `val.csv`, verifies the validation ROC AUC against `best.pt`, and checks
the checkpoint's train/validation manifest hashes.

Threshold selection uses `sklearn.metrics.roc_curve` with
`drop_intermediate=False`. Infinite thresholds are excluded. Youden's J is
`sensitivity - false_positive_rate`; the maximum is selected. An exact tie is
resolved by taking the higher threshold, a rule declared before test access.

The selected value and its complete provenance are written to
`threshold.json`. This file states that the test split had not been loaded. The
only test-manifest loader refuses access unless this frozen record exists and is
valid. The reference threshold 0.4515 is recorded but never substituted.

## One-time official test inference

After threshold freezing, the script loads `test.csv`, requires exactly 284
samples and both classes, and rechecks patient and sample separation across
train, validation, and test. It then loads the phase 3 model state into the
locked EfficientNet-B0 plus Transformer architecture and performs one ordered,
non-augmenting inference pass.

Predicted malignant labels use `probability >= frozen_threshold`. The report
contains ROC AUC, average precision (PR AUC), sensitivity, specificity,
accuracy, balanced accuracy, precision, F1, and TN/FP/FN/TP. It also saves every
test prediction and ROC, precision-recall, and confusion-matrix figures.

The output directory must be new or empty. This prevents silently overwriting a
formal evaluation. Once formal test results are visible, there is no further
model, preprocessing, threshold, or epoch adjustment in this reproduction.

## Artifacts

The formal run writes under `artifacts/evaluation/baseline_seed42`:

- `threshold.json`
- `validation_selection.json`
- `test_evaluation_started.json`
- `test_predictions.csv`
- `metrics.json`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `outcome.json`

Runtime artifacts are excluded from Git. The final stage report records hashes
and metrics without committing private local data paths or model weights.
