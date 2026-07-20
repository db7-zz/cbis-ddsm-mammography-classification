# Phase 4 final evaluation report

## Outcome

The phase 3 numerical-best checkpoint (epoch 12) was evaluated once on all 284
samples in the approved official calcification test manifest. The evaluation
completed successfully on CUDA without a retry. No training, epoch selection,
threshold adjustment, or preprocessing change occurred after test access.

The validation-only Youden threshold was frozen at `0.0482263415881329` before
`test.csv` was loaded. On validation it gave Youden J 0.2949, sensitivity
0.6907, and specificity 0.6042. The historical reference threshold 0.4515 was
recorded for comparison only and was not used for predictions.

## Official test results

| Metric | This reproduction | Historical reference | Difference |
|---|---:|---:|---:|
| ROC AUC | 0.7414 | 0.7869 | -0.0455 |
| PR AUC (average precision) | 0.6729 | 0.6673 | +0.0056 |
| Sensitivity | 70.59% | 80.60% | -10.01 pp |
| Specificity | 63.64% | not reported | - |
| Accuracy | 66.55% | not reported | - |
| Balanced accuracy | 67.11% | not reported | - |
| Precision | 58.33% | not reported | - |
| F1 | 0.6388 | not reported | - |

The confusion matrix is TN=105, FP=60, FN=35, TP=84. Test prevalence is
119/284 (41.90%). The model falls short of the historical ROC AUC and
sensitivity targets, while its PR AUC is slightly higher. This is an honest
reproduction result, not evidence that the historical result was reproduced
exactly.

The historical result materials describe N=326 and a confusion matrix of
TN=132, FP=65, FN=25, TP=104. The approved official calcification-only test
manifest in this project has N=284. Therefore the two metric sets do not use the
same observable sample population, which limits direct comparison and remains a
known reproduction gap.

## Integrity verification

An independent calculation from the saved prediction CSV exactly reproduced all
reported metrics and TN/FP/FN/TP. It also verified:

- 284 rows and 284 unique sample IDs;
- prediction order and labels exactly match `artifacts/data/test.csv`;
- one identical frozen threshold on every row;
- every predicted label equals `probability >= frozen_threshold`;
- train, validation, and test patient/sample overlap is zero;
- the test inference pass count recorded by the evaluation is one.

Key SHA-256 hashes:

- phase 3 checkpoint: `9b471183ea5d5b34365f2be0a5e5d401ef73dc56987847310679f364f1f3aabb`
- frozen threshold: `d45b45ffd8032a372e913606f3a4ba2bda29685fd18f09726e16fffa4f168056`
- test predictions: `678fd357e37043eac59fa73cd6c45d60bc39074d072a49e16098e3b942eaaa82`
- metrics: `8960339437a3ea04332af897adc9bb8f6beac716b414ed824bb8a89f4745bf30`

Formal runtime artifacts are under
`artifacts/evaluation/baseline_seed42` and are intentionally Git-ignored. They
must be retained unchanged if this evaluation is to remain auditable.

## Interpretation

Phase 3 validation ROC AUC was 0.6587, and the final test ROC AUC is 0.7414. The
very low selected threshold indicates that output probabilities are poorly
calibrated relative to the historical reference run. Youden selection remains
valid for classification at the declared operating point, but it does not
calibrate probabilities. Threshold-independent ROC/PR metrics are therefore the
cleanest measures for comparing ranking performance.

Because the official test results are now known, any subsequent architecture,
training, preprocessing, calibration, or threshold experiment must be labeled a
new experiment. This completed run must not be retrospectively tuned against
the test set.
