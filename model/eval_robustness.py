"""
Tests what fusion is actually buying you: if one sensor fails or gets
corrupted at inference time, how much does accuracy degrade? This is the
real robotics-relevant question ("what happens when the lidar is occluded
by rain, or the camera is blinded by glare") that the fusion architecture
is implicitly claimed to help with but was never actually stress-tested
until now.

Usage: python -m model.eval_robustness
"""

import glob
import os
import pickle

import jax.numpy as jnp
import numpy as np

from model.full_model import forward

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")


def load_all_val():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return jnp.array(np.concatenate(images)), jnp.array(np.concatenate(points)), jnp.array(np.concatenate(labels))


def accuracy(params, images, points, labels):
    logits = forward(params, images, points)
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))


def main():
    with open(CHECKPOINT_PATH, "rb") as f:
        params = pickle.load(f)

    images, points, labels = load_all_val()
    rng = np.random.default_rng(0)

    print("sensor failure robustness (clean baseline = 86.01%)\n")

    results = {}

    results["clean (both sensors)"] = accuracy(params, images, points, labels)

    # lidar completely dropped (sensor failure -- stuck at zero)
    zero_points = jnp.zeros_like(points)
    results["lidar dropped (zeros)"] = accuracy(params, images, zero_points, labels)

    # camera completely dropped (sensor failure -- stuck at zero)
    zero_images = jnp.zeros_like(images)
    results["camera dropped (zeros)"] = accuracy(params, zero_images, points, labels)

    # lidar heavily corrupted (noisy sensor, not fully dead)
    noisy_points = points + jnp.array(rng.normal(0, 0.5, points.shape).astype(np.float32))
    results["lidar heavy noise"] = accuracy(params, images, noisy_points, labels)

    # camera heavily corrupted (e.g. glare, lens flare -- blown-out bright noise)
    noisy_images = jnp.clip(images + jnp.array(rng.normal(0, 0.5, images.shape).astype(np.float32)), 0.0, 1.0)
    results["camera heavy noise"] = accuracy(params, noisy_images, points, labels)

    # both sensors degraded moderately at once -- the realistic bad-day scenario
    mod_noisy_points = points + jnp.array(rng.normal(0, 0.2, points.shape).astype(np.float32))
    mod_noisy_images = jnp.clip(images + jnp.array(rng.normal(0, 0.2, images.shape).astype(np.float32)), 0.0, 1.0)
    results["both sensors, moderate noise"] = accuracy(params, mod_noisy_images, mod_noisy_points, labels)

    baseline = results["clean (both sensors)"]
    print(f"{'scenario':<32} {'accuracy':>10} {'vs clean':>10}")
    for name, acc in results.items():
        print(f"{name:<32} {acc:>10.4f} {acc - baseline:>+10.4f}")

    return results


if __name__ == "__main__":
    main()
