# Phase 3 training specification

## Scope and leakage boundary

Phase 3 trains the locked phase 2 architecture using only the 986-image train
split and the 241-image validation split produced in phase 1. The training CLI
does not open `test.csv`; test evaluation and Youden threshold selection remain
out of scope until phase 4.

The train split performs gradient updates. Validation ROC AUC selects the best
epoch, drives learning-rate reduction, and controls early stopping. No weighted
sampler or duplicated samples are used. The fixed loss is
`BCEWithLogitsLoss(pos_weight=1.84)` even though the empirical train class ratio
differs, because 1.84 is part of the locked target baseline.

## Training configuration

The complete auditable configuration is in `configs/train.yaml`:

- seed 42 and deterministic PyTorch execution;
- maximum 50 epochs, batch size 16, four data-loader workers;
- CUDA automatic mixed precision;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- global gradient clipping at norm 1.0;
- ReduceLROnPlateau on validation ROC AUC, factor 0.5, patience 3;
- early stopping on validation ROC AUC, patience 10, minimum delta 0.001;
- ImageNet-pretrained EfficientNet-B0 with every backbone parameter trainable.

The optimizer, learning rate, scheduler, clipping and stopping policy are
explicit **Reproduction assumptions** because the historical training
configuration is unavailable.

## Outputs and recovery

The formal run writes only under `artifacts/training/baseline_seed42`, which Git
ignores. A non-empty directory is never overwritten without an explicit resume
checkpoint. Outputs include:

- `best.pt` and `last.pt`, containing model, optimizer, scheduler, AMP scaler,
  RNG state, configuration and train/validation manifest hashes;
- `metrics.csv` and TensorBoard events;
- `val_predictions_best.csv`, containing validation labels, logits and
  probabilities for the best epoch;
- `run_config.json` and `outcome.json` with environment and audit metadata.

The validation prediction file is intended for phase 4 threshold selection. It
contains no test predictions.

Every numerical increase in validation ROC AUC updates `best.pt`. The configured
`min_delta=0.001` applies only to resetting early-stopping patience; it never
suppresses a numerically better checkpoint.
