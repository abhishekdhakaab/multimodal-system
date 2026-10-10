# Draft-Then-Escalate Cascade — the Honest Analog to Speculative Decoding

## Why this, not literal speculative decoding

Speculative decoding (in LLM serving) uses a cheap draft model to
generate candidate tokens fast, then the expensive target model verifies
them in one batched pass — the win comes from the draft being usually
right, so verification is cheap relative to autoregressive generation
with the target model alone.

This project has no action-generation task (ModelNet10 is static object
classification, no real trajectory/action-sequence data) — building a
literal speculative-decoding-for-actions setup would require fabricating
action data, which was explicitly flagged as a problem earlier in this
project. What transfers cleanly without needing that data is the
underlying systems pattern: **do the cheap thing first, only pay for the
expensive thing when the cheap thing isn't confident.** Applied to
classification, that's a draft-then-escalate cascade, not draft-then-
verify — same spirit (asymmetric cost, draft-first), different mechanism
(confidence-gated routing instead of token-level verification).

## The draft model

`model/draft_classifier.py`: no attention, no per-point MLP, no learned
patch embeddings — just 14 hand-computed global statistics (per-quadrant
image mean/std, point-cloud centroid/spread) through a tiny 2-layer MLP.
Trained on the same real data as the full model.

**Draft classifier accuracy: 64.65%** — far below the full model's
86.01%, but far above the 10% random baseline, and critically: **~1,982x
cheaper** (≈6,272 FLOPs/example vs ≈12,432,768 for the full model,
computed exactly for both architectures, not estimated).

## The cascade result — a real, clean, usable curve

Run the draft model on every input; if its top prediction's confidence
exceeds a threshold, use it and skip the full model entirely; otherwise
run the full model and use its answer.

| confidence threshold | % resolved by draft alone | cascade accuracy |
|---|---|---|
| 0.0 (always draft) | 100.0% | 64.65% |
| 0.5 | 75.6% | 74.45% |
| 0.6 | 54.3% | 80.29% |
| 0.7 | 37.7% | 83.81% |
| **0.8** | **24.3%** | **85.24%** |
| 0.9 | 12.3% | 85.90% |
| 1.0 (always full) | 0.0% | 86.01% |

**The practical operating point (threshold=0.8): 24.3% of inputs are
resolved by a ~2000x-cheaper model, at a cost of only 0.77 accuracy
points versus always running the full model.** At threshold=0.9, the cost
drops to 0.11 points for a smaller (12.3%) but still real fraction of
free inferences.

## Real average-case compute savings

Since the draft model's cost is negligible next to the full model
(~1,982x cheaper, confirmed above), the average-case compute cost of the
cascade is dominated by how often it escalates:

| threshold | escalate fraction | avg-case speedup vs. always running the full model |
|---|---|---|
| 0.8 | 75.7% | **~1.32x** |
| 0.9 | 87.7% | **~1.14x** |

## Why this curve looked more convincing than the other two additions — before real latency numbers existed

Unlike the zero-shot result (a genuinely bad trade at every operating
point), this curve has an honest, clearly-articulable "good" region in
*accuracy-vs-escalation-rate* terms: threshold 0.7-0.9 trades a very
small, clearly quantified accuracy cost (0.1-1.6 points) for 12-38% of
inputs skipping the expensive model. That part is still true and real.

**What changed once real wall-clock latency was measured (see below):**
the FLOP-based compute savings this section originally claimed (~1.32x at
threshold=0.8) don't hold up on real GPU hardware — the actual speedup
there is closer to break-even. Token pruning, by contrast, holds up *better*
than its FLOP estimate in real measurement. Read both updated sections
below before treating either "which optimization wins" framing as settled.

## Real wall-clock latency (RunPod RTX 3090, quiet dedicated GPU, 2026-10-10) — a surprising, honest reversal

Measured via `benchmarks/gpu_latency_pruning_cascade.py`, batch size 64,
min-of-7-trials, random params at the models' real shapes:

| | latency/batch |
|---|---|
| Full model | 0.3767 ms |
| Draft model | 0.0712 ms |
| **Real draft-model speedup** | **5.29x cheaper** |

The FLOP count said the draft model is ~1,982x cheaper; in real wall-clock
terms on this GPU it's only ~5.3x cheaper. The draft model (14 hand-computed
stats through a tiny 2-layer MLP) is so small that **fixed per-launch GPU
overhead (kernel dispatch, not FLOPs) dominates its cost** — FLOPs were
never the bottleneck for an op this small, so the FLOP-based "~2000x
cheaper" estimate was theoretically correct but practically misleading
about wall-clock terms.

That changes the average-case cascade numbers completely:

| threshold | escalate fraction | avg latency/batch | real speedup vs always-full |
|---|---|---|---|
| 0.0 (always draft) | 0.0% | 0.0712 ms | 5.29x |
| 0.5 | 24.4% | 0.1631 ms | 2.31x |
| 0.6 | 45.7% | 0.2433 ms | 1.55x |
| 0.7 | 62.3% | 0.3059 ms | 1.23x |
| **0.8 (the "practical operating point" claimed above)** | 75.7% | 0.3563 ms | **1.057x** |
| 0.9 | 87.7% | 0.4015 ms | **0.938x (SLOWER than always-full)** |
| 1.0 (always full) | 100.0% | 0.4479 ms | 0.841x (slower — draft cost is pure overhead here) |

**This reverses the headline claim above.** The "practical operating
point" (threshold=0.8) was sold as a ~1.32x average-case speedup from the
FLOP estimate; the real measured speedup there is only **1.057x** — barely
worth it. At threshold=0.9 and especially 1.0 (always escalate), running
the draft model first actually makes things *slower* than skipping it
entirely, because its wall-clock cost isn't negligible the way its FLOP
count suggested.

**Why this matters more than a clean win would**: this is a real, honest
systems lesson, not a failure to hide — a technique's FLOP-based savings
estimate can be actively misleading on real hardware when the cheap
component's absolute cost is dominated by fixed per-op overhead rather
than compute. The cascade pattern is still sound in principle (and the
*accuracy* curve above is real and unaffected), but on this specific GPU,
with this specific draft model's size, it only pays off in wall-clock
terms at low-to-moderate escalation rates (threshold <= ~0.6), not at the
threshold originally recommended.

## Honesty notes

- The draft model's 64.65% accuracy and the full model's 86.01% are both
  real, measured numbers (not reused from memory — recomputed earlier in
  this project).
- The FLOP counts for both models are computed exactly for their actual
  architectures and dimensions, not order-of-magnitude guesses — they are
  real numbers, just not predictive of wall-clock behavior at this scale,
  as the measurement above shows.
- The wall-clock numbers above are real, measured on a quiet dedicated GPU
  (RunPod RTX 3090) with no other load, via `benchmarks/gpu_latency_pruning_cascade.py`.
  The old FLOP-based 1.32x/1.14x estimates are kept above for context, not
  deleted, since the gap between estimate and measurement is the most
  interesting part of this result.
- This cascade pattern composes with the token-pruning work: the "full
  model" fallback path could itself use pruned attention, compounding
  the savings — not implemented here, noted as a natural next step.
