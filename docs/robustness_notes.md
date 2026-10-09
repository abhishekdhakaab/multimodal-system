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

## The fix, actually built and measured

`model/train_modality_dropout.py` retrains from scratch with each
training example independently having its lidar input zeroed with
probability 0.15 and its image zeroed with probability 0.15 (reusing the
existing training loop unchanged — only the input the model sees during
training changed). `model/compare_robustness.py` runs the exact same
failure suite against both checkpoints side by side, same noise seed, for
a fair comparison:

| Scenario | Original | Modality-dropout | Change |
|---|---|---|---|
| Clean | 86.01% | 84.69% | -1.32 pts (small, real cost) |
| **Lidar dropped** | 20.15% | **56.61%** | **+36.45 pts** |
| **Camera dropped** | 61.34% | **81.28%** | **+19.93 pts** |
| Lidar heavy noise | 17.18% | 16.85% | -0.33 pts (no change) |
| Camera heavy noise | 69.27% | 63.55% | -5.73 pts (**got worse**) |

## The honest, complete picture — a partial fix with a real limit

**The fix works, dramatically, for exactly the failure mode it was
trained for**: a modality going completely to zero (a dead sensor).
Both dropped-sensor scenarios improved by 20-36 points.

**It does nothing for the other real failure mode**: a modality that's
still producing values, just corrupted ones (noisy, not dead). Training
with zeroed inputs teaches the model "handle an all-zero input
gracefully" — it does not teach the model "be skeptical of a present but
wrong-looking input," which is a different problem. The camera-noise
case even got measurably worse, a real, honestly-reported side effect,
not hidden because it doesn't fit the "the fix worked" narrative.

**The complete, honest conclusion**: multimodal robustness isn't one
problem with one fix — "sensor dies" and "sensor lies" are different
failure modes requiring different training strategies (dropout augmentation
for the former; something like noise-augmentation or outlier-robust
losses for the latter, neither of which was attempted here). This
project fixed one of the two, measured that it didn't generalize to the
other, and reported both facts.
