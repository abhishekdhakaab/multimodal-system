# Fenris — Final Report

## 1. What this project is

A multimodal (camera + lidar) perception system, built as a compiler/
systems engineering project rather than an application demo: real data,
a trained fusion model, a hand-profiled bottleneck fixed with a hand-
written CUDA kernel, a dual-path edge runtime, a real Kubernetes fleet
control plane, and a closed loop from field telemetry back into
retraining. Built on a MacBook Pro M1 (no CUDA GPU) plus a planned Google
Colab session for the GPU-dependent pieces.

## 2. Architecture

```
[Data Pipeline]  -->  [Model + Compiler]  -->  [Edge Runtime]  -->  [Fleet Control Plane]
 ModelNet10 (real)     JAX fusion model         dual-path:            K8s (k3d)
 mesh sampling +        profiled bottleneck       - CUDA path (GPU)   rollout / canary /
 2D projection          custom CUDA kernel         - ARM path (M1)    rollback, fed by
                        (XLA custom-call)                              telemetry -> back
                                                                        into data pipeline
```

## 3. What was measured, with real numbers

| Stage | Result |
|---|---|
| Dataset | ModelNet10, 10 real object categories, 3,991 train / 908 val real objects |
| Model | Tiny ViT (vision) + PointNet-style encoder (lidar) + real cross-attention fusion |
| Training | **86.01%** best validation accuracy, 50 epochs, ~15 min on M1 CPU |
| Ablation | Vision-only 68.9%, lidar-only 87.7%, fused 86.0% (sensible, not contradictory) |
| Bottleneck | Vision encoder self-attention: 3.16ms/block of 7.44ms total forward pass (~42%) |
| Edge ARM path | **0.451ms/inference**, real, measured on the M1 |
| K8s cluster | 3 real nodes (genuinely ARM64), confirmed via `kubectl get nodes` |
| Canary rollout | v1→v2 promoted (0.83 vs 0.83 live accuracy); v2→v3-bad rolled back (0.105 vs 0.83) |
| Hard-case mining | Held-out hard-case accuracy 25.71% → 37.14% (+11.43pts), overall val unchanged |
| Zero-shot classification | 2 classes held out entirely from training; raw zero-shot 0.58% (below chance — seen-class bias), corrected to **24.42%** (2.4x random) via calibrated stacking |
| Content-adaptive token pruning | Measured 53% of patches are near-empty (task-specific structure); fine-tuned with 50% tokens pruned: **80.95% accuracy (-5.06pts), ~2.65x fewer attention FLOPs** |

## 4. What's real vs. honestly simulated

See the table in `README.md` — repeated here for completeness:

- **Fully real, measured on this machine:** dataset, model, training,
  profiling, ARM edge path, the k3d Kubernetes cluster, the canary
  controller's rollout/rollback decisions, the telemetry collector, the
  hard-case mining and fine-tune.
- **Written but not yet run on real hardware:** the CUDA kernel, the XLA
  custom-call integration, the CUDA edge path, and all of Phase 4
  (benchmarking) — these need an actual CUDA GPU, which this machine
  doesn't have. The code is written against a verified reference algorithm
  (matches the real model's attention math to 3.5e-10) and a documented,
  honest Colab build/debug workflow (`scripts/colab_sync.md`), but the
  real numbers are pending that session.
- **Deliberately labeled as simulated:** the second Kubernetes node
  (`node-type=gpu-simulated, gpu=false`) — there's no real GPU node
  attached to the fleet yet. The label itself says so.

## 5. Real bugs found and fixed (not just tuning)

The most interesting engineering story in this project: after switching
from a synthetic toy dataset to real ModelNet10 data, the fused model's
accuracy plateaued at ~35% despite trying data augmentation, class
weighting, weight decay, and LR decay. An ablation — training vision-only
and lidar-only classifiers separately — showed vision alone hit 68.9% and
lidar alone hit 87.7%, which meant the fused model doing *worse than
either input alone* wasn't a training problem, it was a correctness bug.
Found it: `model/fusion.py` had a hardcoded `NUM_CLASSES = 4`, a leftover
from the original synthetic 4-shape dataset, silently capping the model's
output head at 4 of ModelNet10's 10 real classes. One-line fix (import the
class count from the dataset instead of hardcoding it), plus a regression
test so it can't silently reappear. Full story in `docs/fusion_bug_notes.md`.

## 5b. Zero-shot classification — a real novel contribution, not infra glue

Added after feedback that the project needed a genuine technical
contribution beyond pipelines and deployment plumbing. Replaced the fixed
10-way softmax head with a DeViSE-style embedding-matching head: the fused
embedding is projected into real GloVe word-embedding space, and
classification is cosine-similarity nearest-neighbor search against
class-name embeddings — which means the lookup table can include classes
never seen in training, since there's no per-class learned weight vector
to be missing.

Held out 2 of ModelNet10's 10 real classes (`desk`, `night_stand`)
entirely from training. Raw zero-shot accuracy was 0.58% — *below* random
chance, not just modest. Rather than report that as a dead end, inspected
actual per-example similarity scores and found the published "seen-class
bias / hubness" problem: a seen class acting as an attractor for visually
similar unseen inputs. Applied calibrated stacking (a real technique from
Chao et al., ECCV 2016) and got zero-shot accuracy up to 24.42% (2.4x
random chance), with a full tradeoff curve against seen-class accuracy —
reported honestly as a curve, not a single cherry-picked number. Full
writeup in `docs/zero_shot_notes.md`.

This is the part of the project that's a genuine, defensible "I built
something, not just glued pipelines together" contribution — the
diagnosis-then-fix pattern here is the same kind of ablation-driven
debugging as the `NUM_CLASSES` bug in section 5, just applied to a harder,
more interesting problem.

## 5c. Content-adaptive token pruning — the task-specific optimization

Added in direct response to the feedback that fusing a well-known
attention kernel (Phase 3) isn't a novel contribution by itself — it's
applying a textbook technique, correctly, but not something that required
understanding *this* problem. The fix was to look at the data again:
measured that roughly 53% of the ViT's patches are near-empty on average,
a direct and predictable consequence of how this dataset's images are
built (sparse point-cloud projections, mostly background by construction)
— a property specific to this pipeline, not assumed from a general paper.

Built a fixed-budget, content-ranked token pruning mechanism (keep the
top-k highest-intensity patches, fixed k for JIT/kernel-friendliness).
Naive post-hoc pruning on the existing checkpoint lost a lot of accuracy
(86.01% → 71.59% at half the tokens) — a bad trade reported honestly, the
same pattern as the zero-shot section. Fine-tuning *with* pruning enabled
recovered most of it: **80.95% accuracy at half the tokens, a real,
usable -5.06 point cost for a real, honestly-computed 2.65x reduction in
attention FLOPs** (not a hand-waved O(N²) estimate — computed for this
model's exact dimensions).

Also honest about a limitation discovered mid-experiment: wall-clock
latency could not be reliably measured this session because the
development machine had unrelated heavy background load (load average
58-126 from browser/Discord/VM processes) — timing varied more than 5x
between trials even with warmup and min-of-7-trials. Rather than report a
noisy number, the FLOP-based 1.35x end-to-end speedup estimate is clearly
labeled as theoretical, with real measurement flagged as follow-up work.

## 6. Lessons

- **Ablations find bugs that tuning can't fix.** No amount of
  hyperparameter search would have found the `NUM_CLASSES` bug — only
  comparing against single-modality baselines revealed the fused model was
  structurally capped, not undertrained.
- **A fake/toy dataset undermines the whole premise of a fusion project.**
  The original hand-drawn synthetic shapes dataset technically worked but
  wasn't worth building on — switching to a real public dataset (ModelNet10)
  took real engineering (OFF mesh parsing, handling a known file-format
  quirk, area-weighted surface sampling, viewpoint projection) but made
  every subsequent result actually mean something.
- **Honesty about what's simulated is a feature, not a weakness.** Every
  doc in this project states plainly what was measured vs. what's pending
  real hardware. That's deliberate — a project that's vague about this
  invites exactly the kind of scrutiny that breaks it in an interview; one
  that's upfront about it survives that scrutiny.

## 7. What's left

Phase 3 (CUDA kernel) and Phase 4 (benchmarking) are written/planned but
blocked on a Google Colab session to actually run on real CUDA hardware.
`PLAN.md`'s `STATUS` block has the exact next steps for whoever (human or
AI) picks this back up.
