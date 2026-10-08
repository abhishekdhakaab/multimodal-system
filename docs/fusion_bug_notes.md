# Bug: fusion.py's NUM_CLASSES silently stuck at 4

## What happened

After switching the dataset from 4 hand-drawn synthetic shapes to ModelNet10
(10 real object categories, see `docs/dataset_rework_notes.md`), the fused
model's validation accuracy plateaued around 33-35% no matter what training
fix was tried (data augmentation, class-weighted loss, weight decay, LR
decay, higher image resolution). Each fix moved the number by a point or
two, never more — a sign the real bottleneck wasn't in any of those things.

## How it was found

Ran an ablation: trained a vision-only classifier and a lidar-only
classifier (same encoders, a plain linear head, same optimizer/regularization
setup) directly on the real shards, bypassing the fusion module entirely.

- Vision only: **68.9%** val accuracy
- Lidar only: **87.7%** val accuracy
- Fused (the real model): **35.1%** val accuracy

A fused model scoring *worse than either input alone* is not a tuning
problem — if the fusion module were doing nothing useful, it should
score roughly like the better of the two inputs, not dramatically worse.
That's what made this a "go find the bug" moment rather than "keep tuning."

## The bug

`model/fusion.py` had:

```python
NUM_CLASSES = 4
```

— a leftover constant from the original 4-shape synthetic dataset, never
updated when the dataset changed to ModelNet10's 10 classes. This set the
classification head's output layer (`head_w`) to shape `(96, 4)` instead of
`(96, 10)`. The model was **structurally incapable of predicting 6 of the
10 real classes** — not a training/optimization issue at all, a wrong
output shape baked into the architecture.

The separate ablation script didn't have this bug because it built its own
classification head directly from `model.train.NUM_CLASSES` (correctly 10),
which is exactly why it caught the discrepancy.

## The fix

- `model/fusion.py`: `NUM_CLASSES` is now `len(CLASSES)` imported from
  `data_pipeline.modelnet_loader` — the dataset's class list — instead of a
  separately hardcoded number.
- `model/train.py`: same change, same reasoning.
- Added `model/tests/test_model.py::test_fusion_num_classes_matches_real_dataset`,
  which explicitly asserts `fusion.NUM_CLASSES == len(CLASSES)`, so this
  specific class of bug (a stale hardcoded constant drifting out of sync with
  the dataset) fails loudly in the test suite instead of silently capping
  accuracy again if the dataset changes in the future.

## Lesson for the writeup

This is a legitimate, tellable debugging story: multi-modal fusion model
underperforms every single modality alone -> ablate each modality
independently -> find the fused model is structurally capped, not just
undertrained -> fix a one-line constant -> rerun. Worth including in the
final report as a real example of using ablations to localize a bug, not
just to report "it's better together" numbers.
