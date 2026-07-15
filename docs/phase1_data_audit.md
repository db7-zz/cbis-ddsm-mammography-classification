# Phase 1 real-data audit

Audit date: 2026-07-15

## Locked protocol

- CBIS-DDSM calcification subset only.
- Whole mammogram from the official `image file path` column.
- One unique CC/MLO image view per sample.
- `MALIGNANT=1`; `BENIGN` and `BENIGN_WITHOUT_CALLBACK=0`.
- Official test retained as test.
- Official train split into train/validation by `patient_id`, seed 42.
- No patient overlap between train, validation, and test.

The calcification-only plus whole-mammogram choice remains a **Reproduction
assumption** because the original experiment manifest and predictions are not
available.

## Source integrity

| Source | SHA-256 |
| --- | --- |
| `calc_case_description_train_set.csv` | `3c7b07e8202f66709f446b9695e96c5d17484a65fda6a24d0170a0c1b81d56e3` |
| `calc_case_description_test_set.csv` | `03a2f72efb6ed29f417e07f4598950e6a1e02e6b11250ff516ebe7924ace3689` |
| IDC `metadata.csv` | `8fc12b4d70e514d96b03073e12cf0624dfec6b98fd11eb1265db1346c38c8614` |
| Generated `manifest.csv` | `f1c2a8709228087251bbd1b1d4a2ea55b2049c66458f3c8045096f460fb19758` |

The official description files contain 1,546 train abnormality rows and 326
test abnormality rows. Collapsing rows that reference the same official whole
mammogram produces 1,227 official-train images and 284 official-test images,
for 1,511 unique views in total.

There are 222 images referenced by more than one abnormality row. Eleven have
mixed benign/malignant pathology values; the locked image-level policy assigns
these images the malignant label.

## IDC mapping

- Official CSV Study and Series UIDs matched IDC metadata exactly for all
  1,511 unique images.
- All 1,511 mapped paths exist in the IDC default hierarchy.
- 1,265 metadata rows state `full mammogram images`.
- The remaining 246 rows (official test) have an empty IDC
  `SeriesDescription`; their official Study/Series UIDs still match exactly.
- No selected UID maps to `cropped images` or `ROI mask images`.

The empty test descriptions are treated as an IDC metadata limitation, not as
evidence for crop/ROI use. The official whole-mammogram UID and decoded image
dimensions are the mapping authority.

## Patient-level split

| Split | Samples | Patients | Negative | Positive | Positive fraction | Negative/positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 986 | 482 | 574 | 412 | 0.41785 | 1.39320 |
| Validation | 241 | 120 | 144 | 97 | 0.40249 | 1.48454 |
| Test | 284 | 151 | 165 | 119 | 0.41901 | 1.38655 |

Patient intersections are empty for train/validation, train/test, and
validation/test.

| Split | CC | MLO |
| --- | ---: | ---: |
| Train | 469 | 517 |
| Validation | 118 | 123 |
| Test | 130 | 154 |

## DICOM and tensor validation

- 1,511/1,511 DICOM files decoded successfully with no pixel failures.
- Every DICOM Study/Series UID matches its manifest row.
- All selected images are 16-bit `MONOCHROME2`.
- Row range: 3,256 to 7,111 pixels.
- Column range: 1,531 to 5,251 pixels.
- Twelve deterministic samples across all splits produced `float32` tensors of
  shape `3 x 256 x 256` and scalar `float32` labels.
- Nine automated phase 1 tests pass and `pip check` reports no broken
  requirements.

## Known differences from the reference material

1. The reference confusion matrix uses `N=326`, equal to the number of
   official test abnormality rows. The approved protocol uses one independent
   whole-mammogram view per sample, yielding 284 unique test images. Duplicating
   the same image for multiple abnormalities would violate the selected sample
   unit and is not done merely to reproduce `N=326`.
2. The empirical train negative/positive ratio is 1.3932, not 1.84. The locked
   baseline still specifies `BCEWithLogitsLoss(pos_weight=1.84)` for the later
   training phase, but this mismatch is evidence that the unavailable original
   manifest or weighting rationale differed. The value must not be presented
   as an empirical weight from this split.
3. Exact historical intensity preprocessing is unavailable. DICOM LUT,
   percentile scaling, direct 256 x 256 resize, and ImageNet normalization are
   explicit reproduction assumptions in `configs/data.yaml`.

No model, checkpoint, training log, threshold, prediction, or performance
metric was produced in phase 1.
