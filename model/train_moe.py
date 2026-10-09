"""
Trains the MoE-head model and measures the real question: does the gate
learn meaningful per-example routing, or collapse onto a single expert?
Reports both soft-routed (training objective) and hard-routed (what a
real sparse MoE would run at inference) accuracy, plus the actual expert
usage distribution -- not assumed, measured.

Usage: python -m model.train_moe
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from model.full_model_moe import NUM_CLASSES, forward_hard, forward_soft, init_params
from model.moe_head import NUM_EXPERTS, load_balance_loss
from model.train import ADAM_B1, ADAM_B2, ADAM_EPS, BATCH_SIZE, WEIGHT_DECAY, compute_class_weights

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model_moe.pkl")

LEARNING_RATE = 0.003
NUM_EPOCHS = 40
LOAD_BALANCE_COEF = 0.05  # small -- a strong penalty would just force uniform routing regardless of content


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
    logits, probs = forward_soft(params, images, points)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    per_example = -jnp.sum(one_hot * log_probs, axis=-1)
    weights = class_weights[labels]
    classification_loss = jnp.sum(per_example * weights) / jnp.sum(weights)

    balance_loss = load_balance_loss(probs)
    l2 = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
    return classification_loss + LOAD_BALANCE_COEF * balance_loss + WEIGHT_DECAY * l2


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
            soft_logits, _ = forward_soft(params, val_images, val_points)
            soft_acc = float(jnp.mean(jnp.argmax(soft_logits, axis=-1) == val_labels))
            print(f"epoch {epoch+1}/{NUM_EPOCHS}  loss={np.mean(losses):.4f}  soft_val_acc={soft_acc:.4f}")

    soft_logits, _ = forward_soft(params, val_images, val_points)
    soft_acc = float(jnp.mean(jnp.argmax(soft_logits, axis=-1) == val_labels))

    hard_logits, chosen_expert = forward_hard(params, val_images, val_points)
    hard_acc = float(jnp.mean(jnp.argmax(hard_logits, axis=-1) == val_labels))

    usage = np.bincount(np.array(chosen_expert), minlength=NUM_EXPERTS)
    usage_frac = usage / usage.sum()

    print()
    print(f"soft-routed val accuracy (training objective): {soft_acc:.4f}")
    print(f"hard-routed val accuracy (real sparse-MoE inference): {hard_acc:.4f}")
    print(f"expert usage distribution: {[f'{f*100:.1f}%' for f in usage_frac]}")
    print(f"(uniform would be {100/NUM_EXPERTS:.1f}% each)")

    # does usage correlate with real class identity, or something else? check honestly
    print()
    print("expert choice vs true class (rows=class, cols=expert, counts):")
    from data_pipeline.modelnet_loader import CLASSES

    table = np.zeros((len(CLASSES), NUM_EXPERTS), dtype=int)
    for c, e in zip(np.array(val_labels), np.array(chosen_expert)):
        table[c, e] += 1
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12} {table[i].tolist()}")

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"\nsaved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
