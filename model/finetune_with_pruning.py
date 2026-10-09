"""
model/eval_token_pruning.py showed that applying token pruning post-hoc
to a checkpoint trained WITHOUT it costs a lot of accuracy (86.01% -> 71.59%
at k=50) -- expected, since the model's attention/MLP statistics were
never trained to expect a pruned, smaller token budget. This script tests
the real question: does briefly fine-tuning WITH pruning enabled recover
most of that loss, for a model that's cheaper to run at inference time
from here on?

Usage: python -m model.finetune_with_pruning
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from model.full_model import forward
from model.train import (
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    BATCH_SIZE,
    NUM_CLASSES,
    WEIGHT_DECAY,
    compute_class_weights,
)

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")
OUT_PATH = os.path.join(HERE, "checkpoints", "model_pruned_k50.pkl")

PRUNE_K = 50
FINETUNE_EPOCHS = 15
FINETUNE_LR = 0.001


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
    logits = forward(params, images, points, prune_k=PRUNE_K)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    per_example = -jnp.sum(one_hot * log_probs, axis=-1)
    weights = class_weights[labels]
    weighted = jnp.sum(per_example * weights) / jnp.sum(weights)
    l2 = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
    return weighted + WEIGHT_DECAY * l2


def accuracy_at_k(params, images, points, labels, k):
    logits = forward(params, images, points, prune_k=k)
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))


@jax.jit
def train_step(params, adam_state, images, points, labels, class_weights):
    loss, grads = jax.value_and_grad(loss_fn)(params, images, points, labels, class_weights)
    t = adam_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, adam_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * (g**2), adam_state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2**t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: FINETUNE_LR * mh / (jnp.sqrt(vh) + ADAM_EPS), m_hat, v_hat
    )
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return params, {"m": m, "v": v, "t": t}, loss


def main():
    with open(CHECKPOINT_PATH, "rb") as f:
        params = pickle.load(f)

    train_images, train_points, train_labels = load_split("train")
    val_images, val_points, val_labels = load_split("val")

    before_acc = accuracy_at_k(params, val_images, val_points, val_labels, PRUNE_K)
    print(f"BEFORE fine-tune: accuracy at k={PRUNE_K} (post-hoc pruning, no retraining) = {before_acc:.4f}")

    class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    adam_state = {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }
    rng = np.random.default_rng(0)

    for epoch in range(FINETUNE_EPOCHS):
        perm = rng.permutation(train_images.shape[0])
        losses = []
        for start in range(0, train_images.shape[0], BATCH_SIZE):
            idx = jnp.array(perm[start : start + BATCH_SIZE])
            params, adam_state, loss = train_step(
                params, adam_state, train_images[idx], train_points[idx], train_labels[idx], class_weights
            )
            losses.append(float(loss))
        print(f"epoch {epoch+1}/{FINETUNE_EPOCHS}  loss={np.mean(losses):.4f}")

    after_acc_k50 = accuracy_at_k(params, val_images, val_points, val_labels, PRUNE_K)
    after_acc_full = accuracy_at_k(params, val_images, val_points, val_labels, None)

    print()
    print(f"AFTER fine-tune (trained WITH k={PRUNE_K} pruning):")
    print(f"  accuracy at k={PRUNE_K} (the regime it was fine-tuned for): {after_acc_k50:.4f}")
    print(f"  accuracy at full resolution (k=100, never trained at this k): {after_acc_full:.4f}")
    print(f"  recovered: {after_acc_k50 - before_acc:+.4f} vs post-hoc pruning")
    print(f"  vs original un-pruned baseline (86.01%): {after_acc_k50 - 0.8601:+.4f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
