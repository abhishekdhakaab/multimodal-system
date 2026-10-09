"""
The real experiment: take the already-trained checkpoint (model.pkl, 86.01%
val accuracy, trained with NO pruning) and test content-adaptive token
pruning at inference time only -- no retraining. Reports both the real
accuracy at each pruning level AND the real measured latency, so this is
an honest accuracy-vs-speed curve, not a cherry-picked point.

Usage: python -m model.eval_token_pruning
"""

import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from model.full_model import forward
from model.vision_encoder import NUM_PATCHES

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")

K_VALUES = [100, 80, 60, 50, 40, 30, 20]  # 100 = no pruning (every patch), the original baseline
BATCH_SIZE = 64
N_TIMING_RUNS = 30
N_TIMING_TRIALS = 7  # repeat the whole timing measurement this many times, report the min --
# this machine has other real load (Docker/k3d, browser) competing for CPU, so a single
# mean is noisy; min-of-many-trials is standard practice for latency microbenchmarks
# because scheduler noise only ever adds delay, never removes it


def load_all_val():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return np.concatenate(images), np.concatenate(points), np.concatenate(labels)


def main():
    with open(CHECKPOINT_PATH, "rb") as f:
        params = pickle.load(f)

    images, points, labels = load_all_val()
    images, points, labels = jnp.array(images), jnp.array(points), jnp.array(labels)

    print(f"evaluating checkpoint trained WITHOUT pruning (86.01% baseline) under test-time pruning")
    print(f"{'k':>5} {'tokens kept':>12} {'accuracy':>10} {'latency/batch':>15}")

    results = []
    for k in K_VALUES:
        prune_k = None if k == NUM_PATCHES else k

        infer_fn = jax.jit(lambda p, img, pts: forward(p, img, pts, prune_k=prune_k))

        # accuracy over the full val set
        logits = infer_fn(params, images, points)
        preds = jnp.argmax(logits, axis=-1)
        acc = float(jnp.mean(preds == labels))

        # latency on one batch: repeat the whole timed block N_TIMING_TRIALS
        # times, report the min -- robust to this machine's background load
        batch_images, batch_points = images[:BATCH_SIZE], points[:BATCH_SIZE]
        out = infer_fn(params, batch_images, batch_points)
        jax.block_until_ready(out)  # warmup / JIT compile, excluded from timing

        trial_latencies = []
        for _ in range(N_TIMING_TRIALS):
            start = time.perf_counter()
            for _ in range(N_TIMING_RUNS):
                out = infer_fn(params, batch_images, batch_points)
            jax.block_until_ready(out)
            trial_latencies.append((time.perf_counter() - start) / N_TIMING_RUNS * 1000)
        latency_ms = min(trial_latencies)

        label = "no pruning" if prune_k is None else f"{k}/{NUM_PATCHES}"
        print(f"{k:>5} {label:>12} {acc:>10.4f} {latency_ms:>13.3f}ms  (trials: {[round(t,2) for t in trial_latencies]})")
        results.append((k, acc, latency_ms))

    return results


if __name__ == "__main__":
    main()
