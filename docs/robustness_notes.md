# Sensor-Failure Robustness — Testing an Assumption, Not Just Stating It

## The question

Multimodal fusion is often implicitly assumed to be more robust than any
single sensor — "if the camera fails, the lidar can still carry you."
This was never actually tested in this project until now. It's exactly
the question a robotics/perception interviewer would ask, so it's worth
having a real answer instead of an assumption.

## The real result — fusion is NOT robust to sensor failure here

| Scenario | Accuracy | vs. clean baseline (86.01%) |
|---|---|---|
| Clean (both sensors) | 86.01% | — |
| Lidar dropped (zeroed) | 20.15% | **-65.86 points** |
| Camera dropped (zeroed) | 61.34% | -24.67 points |
| Lidar heavy noise | 17.18% | **-68.83 points** |
| Camera heavy noise | 69.27% | -16.74 points |
| Both sensors, moderate noise | 31.72% | -54.30 points |

**This is a genuine negative finding, reported honestly rather than
glossed over:** losing lidar is catastrophic (worse than random
10-class guessing would even predict is "safe," though still well above
the 10% floor), while losing the camera hurts less. This is consistent
with the earlier single-modality ablation (lidar-only: 87.7%, vision-
only: 68.9%, see `docs/fusion_bug_notes.md`) — the model leans heavily
on lidar, so losing it costs more than losing the camera.

## The genuinely surprising part

**Noisy lidar (17.18%) is worse than zeroed lidar (20.15%).** Zeros are
at least a consistent, uninformative signal; random noise actively feeds
the lidar encoder plausible-looking but wrong geometry, which misleads
it more than having nothing. The opposite happens for the camera: noisy
images (69.27%) hurt *less* than zeroed images (61.34%) — the underlying
silhouette contrast partially survives additive noise, where zeroing
destroys it completely. Two different failure modes for two different
modalities, not a single clean story — worth stating precisely rather
than collapsing into "sensors are fragile."

## Why this isn't robust, honestly diagnosed, not just reported

The model was trained with both sensors reliably present on every
example. It never saw a dropped or corrupted modality during training —
there was no reason for it to learn a fallback behavior, so it didn't.
This is a known, real phenomenon (models don't generalize to failure
modes they were never trained on) with a known, real fix: **modality
dropout during training** (randomly zero out one modality some fraction
of the time while training), which forces the model to maintain usable
features in the other modality rather than leaning on whichever one is
more informative on average.

## Honest status

This document reports the measurement and the diagnosis. The fix
(modality-dropout training) was **not** implemented in this pass — it's
a legitimate next step, not claimed as already done. If pursued, the
right experiment is: retrain with modality dropout, re-run this exact
robustness suite, and report whether the dropped/noisy-sensor numbers
improve and by how much — the same diagnose-then-fix-then-measure
pattern used for the overfitting and token-pruning findings elsewhere in
this project.
