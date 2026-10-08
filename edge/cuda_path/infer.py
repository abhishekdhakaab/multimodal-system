"""
The CUDA inference path: runs the trained model with the vision encoder's
core attention computed by the hand-written CUDA kernel (kernels/fused_attention.cu)
instead of plain JAX. Only runs on a machine with a CUDA GPU and a built
kernels/fused_attention.so -- see scripts/colab_sync.md.

On the M1 (no CUDA), this will raise FileNotFoundError when it tries to load
the kernel -- that's expected and is reported clearly below rather than as a
confusing crash.

Usage (on a CUDA machine, e.g. Colab, after building fused_attention.so):
    python -m edge.cuda_path.infer
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from model.full_model import forward
from kernels.register import SO_PATH

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
    if not os.path.exists(SO_PATH):
        print(
            f"[CUDA path] {SO_PATH} not found -- this path only runs on a CUDA GPU "
            "with the kernel built. See scripts/colab_sync.md. Not a bug, just the "
            "wrong machine (expected on the M1)."
        )
        return None, None, None

    params = load_checkpoint()
    image, points, label, classes = load_one_example()

    infer_fn = jax.jit(lambda p, img, pts: forward(p, img, pts, backend="cuda_kernel"))

    logits = infer_fn(params, image, points)
    jax.block_until_ready(logits)

    start = time.perf_counter()
    for _ in range(n_timing_runs):
        logits = infer_fn(params, image, points)
    jax.block_until_ready(logits)
    elapsed = (time.perf_counter() - start) / n_timing_runs

    pred = int(jnp.argmax(logits, axis=-1)[0])
    print(f"[CUDA path] true={classes[label]}  predicted={classes[pred]}  "
          f"latency={elapsed*1000:.3f} ms/inference")
    return elapsed, pred, label


if __name__ == "__main__":
    run()
