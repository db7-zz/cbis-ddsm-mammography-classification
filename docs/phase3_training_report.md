# Phase 3 training report

Run date: 2026-07-16

## Outcome

The formal seed-42 run completed successfully on an NVIDIA GeForce RTX 5070 Ti
Laptop GPU. It stopped after epoch 12 when validation ROC AUC had not improved
by the configured `min_delta=0.001` for ten epochs. Every numerical ROC AUC
increase still updated the best checkpoint, independent of the early-stopping
delta.

| Selected epoch | Train loss | Validation loss | Validation ROC AUC | Validation PR AUC | Learning rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.158105 | 2.705716 | 0.658684 | 0.533763 | 0.000025 |

Epoch 12 exceeded the previous numerical ROC AUC best from epoch 3
(`0.658434`) by approximately `0.000251`, so `best.pt` was updated. The increase
was smaller than the early-stopping delta relative to its reference, so the run
correctly stopped after writing the new best checkpoint.

The 12 epochs processed all 986 train images and 241 validation images each
time. Summed epoch time was 2,343.21 seconds (39.05 minutes). Falling train loss
and rising validation loss after the early epochs indicate substantial
overfitting; no post-hoc epoch substitution was performed.

## Independent verification

- `metrics.csv` contains exactly 12 epoch records.
- `val_predictions_best.csv` contains 241 unique validation samples, all marked
  as epoch 12.
- Recomputing metrics from the saved probabilities gives ROC AUC `0.6586841352`
  and PR AUC `0.5337626574`, exactly matching the epoch log.
- `best.pt` and `last.pt` both identify epoch 12 and have identical content,
  because the final epoch was also the numerical best.
- The CLI and run metadata state `test_split_loaded=false`; phase 3 never loaded
  the 284-image official test split.

This is a validation result, not the final test result, and is not directly
comparable to the reference test ROC AUC `0.7869`. Threshold selection and the
single final test evaluation remain phase 4 work.

## Provenance

- Train manifest SHA-256:
  `3443c220d76ff8e50e63d17d3a6c46681e28b4fe2510eb56b41ee3682a595716`
- Validation manifest SHA-256:
  `15e3611a765db01b1765cfcd45067046f7b81e41ea827179e1a81f501dae43d0`
- Torchvision EfficientNet-B0 weight SHA-256:
  `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`
- Best/last checkpoint SHA-256:
  `9b471183ea5d5b34365f2be0a5e5d401ef73dc56987847310679f364f1f3aabb`
- Metrics CSV SHA-256:
  `dd739a5d73b74854b04047a0ec5d4a37ab288acbc656540763384d2799b175cb`
- Best validation predictions SHA-256:
  `ee93f8b758b7ad325dd12435350dfadd3edee6a3b70e0c9ec3e4ad94d6fdaa95`

The run was based on Git commit
`946b22b875cfbfcdf9927517369d5ed59fe0167e` with uncommitted phase 3 source
changes. `run_config.json` therefore records SHA-256 hashes for every relevant
phase 3 source and configuration file.

## Artifacts

Formal artifacts are under `artifacts/training/baseline_seed42` and are excluded
from Git. An earlier three-epoch run that exposed the best-checkpoint/min-delta
logic defect was stopped and preserved under
`artifacts/training/baseline_seed42_pre_fix_20260716_epoch3`; it is not used as
an experimental result.
