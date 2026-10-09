"""
Runs the exact same sensor-failure suite from model/eval_robustness.py
against BOTH the original checkpoint and the modality-dropout-trained
one, side by side, so the real question gets a real answer: did training
with modality dropout actually fix the robustness problem diagnosed in
docs/robustness_notes.md, and at what cost to clean accuracy?

Usage: python -m model.compare_robustness
"""

import pickle

import jax.numpy as jnp
import numpy as np

from model.eval_robustness import CHECKPOINT_PATH as ORIGINAL_CHECKPOINT_PATH
from model.eval_robustness import accuracy, load_all_val
from model.train_modality_dropout import CHECKPOINT_PATH as DROPOUT_CHECKPOINT_PATH


def run_suite(params, images, points, labels, rng):
    results = {}
    results["clean"] = accuracy(params, images, points, labels)

    zero_points = jnp.zeros_like(points)
    results["lidar dropped"] = accuracy(params, images, zero_points, labels)

    zero_images = jnp.zeros_like(images)
    results["camera dropped"] = accuracy(params, zero_images, points, labels)

    noisy_points = points + jnp.array(rng.normal(0, 0.5, points.shape).astype(np.float32))
    results["lidar heavy noise"] = accuracy(params, images, noisy_points, labels)

    noisy_images = jnp.clip(images + jnp.array(rng.normal(0, 0.5, images.shape).astype(np.float32)), 0.0, 1.0)
    results["camera heavy noise"] = accuracy(params, noisy_images, points, labels)

    return results


def main():
    images, points, labels = load_all_val()

    with open(ORIGINAL_CHECKPOINT_PATH, "rb") as f:
        original_params = pickle.load(f)
    with open(DROPOUT_CHECKPOINT_PATH, "rb") as f:
        dropout_params = pickle.load(f)

    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)  # same seed -- identical noise for a fair comparison

    original_results = run_suite(original_params, images, points, labels, rng_a)
    dropout_results = run_suite(dropout_params, images, points, labels, rng_b)

    print(f"{'scenario':<22} {'original':>10} {'modality-dropout':>18} {'change':>10}")
    for scenario in original_results:
        o, d = original_results[scenario], dropout_results[scenario]
        print(f"{scenario:<22} {o:>10.4f} {d:>18.4f} {d - o:>+10.4f}")


if __name__ == "__main__":
    main()
