# Phase 4 — Benchmark Results

## Scope, honestly stated

The original Phase 4 plan was a head-to-head: stock JAX attention vs. the
custom CUDA kernel, both run through the same JAX forward pass on a real
GPU. That comparison is currently **not possible** — see
`scripts/runpod_sync.md` for the full evidence trail showing JAX's own
pip-distributed `jax[cuda12]==0.4.34` CUDA plugin fails to fully initialize
its custom-call registry, independent of anything in this project's code.
What follows is the real data that IS available: the kernel's own measured
cost on real hardware, from a standalone CUDA harness, plus the real ARM/M1
CPU baseline for the full model.

## Kernel-only latency (RunPod RTX 3090, 2026-10-10)

Via `kernels/tests/standalone_cuda_test.cu` (calls `launch_fused_attention`
directly, no JAX/XLA in the loop):

| Metric | Value |
|---|---|
| Shape | batch=2, heads=4, N=101 (seq len), head_dim=12 |
| Correctness (max abs diff vs CPU reference) | 2.98e-08 (fp32-noise level) |
| Avg latency | 65.59 us/launch (200 iterations, after 10 warmup) |

This is the real, measured cost of one fused-attention kernel launch at
this project's actual model shape, on real Ampere hardware. It is not yet
a JAX-vs-kernel comparison (see above).

## Full-model CPU baseline (M1, real edge path)

From `docs/edge_dual_path_notes.md`:

| Path | Latency |
|---|---|
| ARM/M1 CPU, full model forward pass | 0.451 ms/inference |

Note these two numbers aren't directly comparable (one is a single fused op
on a discrete GPU, the other is the entire model's forward pass on a CPU)
— they're reported separately and honestly, not combined into a misleading
ratio.

## Token pruning and cascade: real wall-clock latency (RunPod RTX 3090)

Both were previously FLOP-based theoretical estimates only (the M1 was too
noisy to trust timing — see `docs/token_pruning_notes.md` and
`docs/speculative_cascade_notes.md`). Measured for real on the same quiet
RunPod GPU via `benchmarks/gpu_latency_pruning_cascade.py`:

- **Token pruning held up better than its estimate**: real speedup at
  k=50 is **1.738x** vs. the FLOP-based estimate of 1.35x.
- **The cascade did not hold up**: the draft model's FLOP count suggested
  it's ~1,982x cheaper than the full model; in real wall-clock terms it's
  only ~5.3x cheaper (fixed GPU per-launch overhead dominates a model this
  small). The "practical operating point" (threshold=0.8) that looked like
  a 1.32x average-case win is actually only ~1.057x in real measurement,
  and threshold>=0.9 is measurably *slower* than always running the full
  model. Full numbers and discussion in `docs/speculative_cascade_notes.md`.

## What's still missing, and why

A real JAX-vs-kernel speedup ratio for the vision encoder's self-attention
specifically requires the kernel to be reachable as a JAX custom call,
which is blocked by the upstream bug documented in `scripts/runpod_sync.md`.
If revisited, the next thing worth trying is a jaxlib version well outside
the two already ruled out (0.4.34 and 0.11.1), or building jaxlib from
source to get a plugin whose Python glue actually matches its compiled
native code.
