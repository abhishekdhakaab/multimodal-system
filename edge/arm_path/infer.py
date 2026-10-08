"""
The ARM inference path: runs the trained model using plain JAX (CPU
backend), no CUDA kernel involved. This is the real edge path for this
project -- the M1 itself IS the ARM device, not a simulation of one.

Usage: python -m edge.arm_path.infer
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from model.full_model import forward

HERE = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(HERE, "..", "..", "model", "checkpoints", "model.pkl")
VAL_SHARD_PATH = os.path.join(HERE, "..", "..", "data_pipeline", "shards", "val", "shard_000.npz")


def load_checkpoint():
    with open(CHECKPOINT_PATH, "rb") as f:
        return pickle.load(f)


def load_one_example():
    data = np.load(VAL_SHARD_PATH, allow_pickle=True)
    image = jnp.array(data["images"][0:1])
    points = jnp.array(data["pointclouds"][0:1])
    label = int(data["labels"][0])
    classes = data["classes"]
    return image, points, label, classes


def run(n_timing_runs=30):
    params = load_checkpoint()
    image, points, label, classes = load_one_example()

    infer_fn = jax.jit(lambda p, img, pts: forward(p, img, pts, backend="jax"))

    logits = infer_fn(params, image, points)  # warm up / trigger JIT compile
    jax.block_until_ready(logits)

    start = time.perf_counter()
    for _ in range(n_timing_runs):
        logits = infer_fn(params, image, points)
    jax.block_until_ready(logits)
    elapsed = (time.perf_counter() - start) / n_timing_runs

    pred = int(jnp.argmax(logits, axis=-1)[0])
    print(f"[ARM path / M1 CPU] true={classes[label]}  predicted={classes[pred]}  "
          f"latency={elapsed*1000:.3f} ms/inference")
    return elapsed, pred, label


if __name__ == "__main__":
    run()
