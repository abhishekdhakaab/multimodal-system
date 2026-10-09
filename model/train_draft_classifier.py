"""
Trains the cheap draft classifier (model/draft_classifier.py) on the same
real shards as the full model -- same data, same classes, just a far
cheaper architecture. Used by model/cascade_infer.py.

Usage: python -m model.train_draft_classifier
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from model.draft_classifier import NUM_CLASSES, forward, init_params
from model.train import ADAM_B1, ADAM_B2, ADAM_EPS, BATCH_SIZE, compute_class_weights

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "draft_classifier.pkl")

LEARNING_RATE = 0.01
NUM_EPOCHS = 40


def load_split(split):
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, split, "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return jnp.array(np.concatenate(images)), jnp.array(np.concatenate(points)), jnp.array(np.concatenate(labels))


def loss_fn(params, images, points, labels, class_weights):
    logits = forward(params, images, points)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    per_example = -jnp.sum(one_hot * log_probs, axis=-1)
    weights = class_weights[labels]
    return jnp.sum(per_example * weights) / jnp.sum(weights)


@jax.jit
def train_step(params, adam_state, images, points, labels, class_weights):
    loss, grads = jax.value_and_grad(loss_fn)(params, images, points, labels, class_weights)
    t = adam_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, adam_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * (g**2), adam_state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2**t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: LEARNING_RATE * mh / (jnp.sqrt(vh) + ADAM_EPS), m_hat, v_hat
    )
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return params, {"m": m, "v": v, "t": t}, loss


def main():
    train_images, train_points, train_labels = load_split("train")
    val_images, val_points, val_labels = load_split("val")

    class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    key = random.PRNGKey(0)
    params = init_params(key)
    adam_state = {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }
    rng = np.random.default_rng(0)

    for epoch in range(NUM_EPOCHS):
        perm = rng.permutation(train_images.shape[0])
        losses = []
        for start in range(0, train_images.shape[0], BATCH_SIZE):
            idx = jnp.array(perm[start : start + BATCH_SIZE])
            params, adam_state, loss = train_step(
                params, adam_state, train_images[idx], train_points[idx], train_labels[idx], class_weights
            )
            losses.append(float(loss))
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            logits = forward(params, val_images, val_points)
            val_acc = float(jnp.mean(jnp.argmax(logits, axis=-1) == val_labels))
            print(f"epoch {epoch+1}/{NUM_EPOCHS}  loss={np.mean(losses):.4f}  val_acc={val_acc:.4f}")

    final_logits = forward(params, val_images, val_points)
    final_acc = float(jnp.mean(jnp.argmax(final_logits, axis=-1) == val_labels))
    print(f"\nfinal draft classifier val accuracy: {final_acc:.4f}")

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
