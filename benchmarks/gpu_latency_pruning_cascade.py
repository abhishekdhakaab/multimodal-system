"""
Real wall-clock GPU latency for the two optimizations that only ever had
FLOP-based theoretical estimates: content-adaptive token pruning
(docs/token_pruning_notes.md) and the draft-then-escalate cascade
(docs/speculative_cascade_notes.md). Both were deferred on the M1 because
the dev machine's load average (58-126, other processes competing for
CPU) made timing results untrustworthy -- a quiet, dedicated GPU box
(see scripts/runpod_sync.md) is exactly the environment needed instead.

Uses randomly-initialized params at the model's real shapes, not a
trained checkpoint -- latency depends on the compute graph (shapes, which
ops run), not on weight values or which patches top-k selects, so this
is a legitimate way to measure latency without needing a checkpoint file
or the full dataset on the machine running this. The real ACCURACY
numbers already exist (measured on the M1, see the two docs above) and
are not recomputed here -- this script's only job is the missing
wall-clock number.

Usage: python -m benchmarks.gpu_latency_pruning_cascade
"""

import time

import jax
import jax.numpy as jnp
from jax import random

from model.full_model import init_params, forward
from model.vision_encoder import NUM_PATCHES
from model.draft_classifier import init_params as draft_init_params, forward as draft_forward
from data_pipeline.modelnet_loader import NUM_POINTS, IMAGE_SIZE

BATCH_SIZE = 64
N_TIMING_RUNS = 50
N_TIMING_TRIALS = 7  # min-of-N, standard practice for latency microbenchmarks -- scheduler
# noise only ever adds delay, never removes it, so the min across trials is the most
# trustworthy single number
K_VALUES = [100, 80, 60, 50, 40, 30, 20]  # matches docs/token_pruning_notes.md exactly


def make_inputs(key, batch_size):
    k1, k2 = random.split(key)
    images = random.normal(k1, (batch_size, IMAGE_SIZE, IMAGE_SIZE))
    points = random.normal(k2, (batch_size, NUM_POINTS, 3))
    return images, points


def time_fn(fn, args, label):
    out = fn(*args)
    jax.block_until_ready(out)  # warmup / compile, excluded from timing

    trial_latencies = []
    for _ in range(N_TIMING_TRIALS):
        start = time.perf_counter()
        for _ in range(N_TIMING_RUNS):
            out = fn(*args)
        jax.block_until_ready(out)
        trial_latencies.append((time.perf_counter() - start) / N_TIMING_RUNS * 1000)
    latency_ms = min(trial_latencies)
    print(f"{label:>40}: {latency_ms:8.4f} ms/batch  (trials: {[round(t, 4) for t in trial_latencies]})")
    return latency_ms


def main():
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    key = random.PRNGKey(0)
    params = init_params(key)
    draft_params = draft_init_params(random.PRNGKey(1))
    images, points = make_inputs(random.PRNGKey(2), BATCH_SIZE)

    print("\n=== Token pruning: real wall-clock latency per k ===")
    results = {}
    for k in K_VALUES:
        prune_k = None if k == NUM_PATCHES else k
        fn = jax.jit(lambda p, img, pts, pk=prune_k: forward(p, img, pts, prune_k=pk))
        label = "no pruning" if prune_k is None else f"k={k}/{NUM_PATCHES}"
        results[k] = time_fn(fn, (params, images, points), label)

    baseline = results[NUM_PATCHES] if NUM_PATCHES in results else results[100]
    print("\nspeedup vs no-pruning baseline:")
    for k in K_VALUES:
        print(f"  k={k:>3}: {baseline / results[k]:.3f}x")

    print("\n=== Cascade: draft model vs full model, real wall-clock latency ===")
    full_fn = jax.jit(lambda p, img, pts: forward(p, img, pts))
    draft_fn = jax.jit(lambda p, img, pts: draft_forward(p, img, pts))
    full_latency = time_fn(full_fn, (params, images, points), "full model")
    draft_latency = time_fn(draft_fn, (draft_params, images, points), "draft model")
    print(f"\ndraft-model speedup vs full model: {full_latency / draft_latency:.2f}x")

    print("\n=== Cascade: real average-case latency at each confidence threshold ===")
    # escalate fractions straight from docs/speculative_cascade_notes.md's
    # already-measured (on real val accuracy data) operating points -- this
    # section only adds the real latency side, not new escalate-fraction data
    escalate_fractions = {0.0: 0.0, 0.5: 0.244, 0.6: 0.457, 0.7: 0.623, 0.8: 0.757, 0.9: 0.877, 1.0: 1.0}
    for threshold, escalate_frac in escalate_fractions.items():
        avg_latency = draft_latency + escalate_frac * full_latency
        speedup = full_latency / avg_latency
        print(f"  threshold={threshold:.1f}: avg latency {avg_latency:7.4f} ms/batch, speedup {speedup:.3f}x vs always-full")


if __name__ == "__main__":
    main()
