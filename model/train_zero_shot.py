"""
The real zero-shot experiment: hold out 2 of ModelNet10's 10 classes
ENTIRELY from training (no training images from them at all), train the
embedding-matching head only on the remaining 8, then measure whether the
model can correctly identify the held-out classes' real validation images
anyway -- using nothing but the semantic structure of real GloVe word
embeddings (model/class_embeddings.npy) plus what the vision/lidar
backbone learned from the 8 seen classes.

This is only possible because the classification head is a cosine-
similarity lookup against class embeddings, not a fixed per-class softmax
weight vector -- see model/zero_shot_head.py's docstring.

Usage: python -m model.train_zero_shot
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from data_pipeline.modelnet_loader import CLASSES
from model.full_model_zero_shot import init_params, similarity_logits
from model.zero_shot_head import load_class_embeddings

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model_zero_shot.pkl")

HELD_OUT_CLASSES = ["desk", "night_stand"]  # chosen for reasonable, not-trivially-easy sample counts
SEEN_CLASSES = [c for c in CLASSES if c not in HELD_OUT_CLASSES]
HELD_OUT_IDX = jnp.array([CLASSES.index(c) for c in HELD_OUT_CLASSES])
SEEN_IDX = jnp.array([CLASSES.index(c) for c in SEEN_CLASSES])

LEARNING_RATE = 0.003
NUM_EPOCHS = 40
BATCH_SIZE = 64
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def load_split(split):
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, split, "shard_*.npz")))
    images, pointclouds, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        pointclouds.append(d["pointclouds"])
        labels.append(d["labels"])
    return np.concatenate(images), np.concatenate(pointclouds), np.concatenate(labels)


def remap_to_seen(labels):
    """Maps original class indices (0-9) to seen-only indices (0-7), for computing
    the training loss restricted to seen classes."""
    class_to_seen_idx = {CLASSES.index(c): i for i, c in enumerate(SEEN_CLASSES)}
    return np.array([class_to_seen_idx[int(l)] for l in labels])


def loss_fn(params, images, pointclouds, seen_labels, class_embeddings):
    full_logits = similarity_logits(params, images, pointclouds, class_embeddings)
    seen_logits = full_logits[:, SEEN_IDX]  # restrict to seen classes only during training
    log_probs = jax.nn.log_softmax(seen_logits)
    one_hot = jax.nn.one_hot(seen_labels, len(SEEN_CLASSES))
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def init_adam_state(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }


@jax.jit
def train_step(params, adam_state, images, pointclouds, seen_labels, class_embeddings):
    loss, grads = jax.value_and_grad(loss_fn)(params, images, pointclouds, seen_labels, class_embeddings)
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


def evaluate(params, images, pointclouds, labels, class_embeddings):
    """Full 10-class argmax (seen AND unseen embeddings both in the lookup) --
    this is what makes the held-out-class numbers genuinely zero-shot."""
    logits = similarity_logits(params, jnp.array(images), jnp.array(pointclouds), class_embeddings)
    preds = np.array(jnp.argmax(logits, axis=-1))
    return preds


def main():
    print(f"held-out classes (NEVER trained on): {HELD_OUT_CLASSES}")
    print(f"seen classes: {SEEN_CLASSES}")

    train_images, train_points, train_labels = load_split("train")
    val_images, val_points, val_labels = load_split("val")

    seen_mask = ~np.isin(train_labels, np.array([CLASSES.index(c) for c in HELD_OUT_CLASSES]))
    train_images, train_points, train_labels = (
        train_images[seen_mask],
        train_points[seen_mask],
        train_labels[seen_mask],
    )
    print(f"train set after removing held-out classes: {len(train_labels)} examples")
    train_seen_labels = remap_to_seen(train_labels)

    class_embeddings = load_class_embeddings()

    key = random.PRNGKey(0)
    params = init_params(key)
    adam_state = init_adam_state(params)
    rng = np.random.default_rng(0)

    for epoch in range(NUM_EPOCHS):
        perm = rng.permutation(len(train_labels))
        losses = []
        for start in range(0, len(train_labels), BATCH_SIZE):
            idx = jnp.array(perm[start : start + BATCH_SIZE])
            params, adam_state, loss = train_step(
                params,
                adam_state,
                jnp.array(train_images)[idx],
                jnp.array(train_points)[idx],
                jnp.array(train_seen_labels)[idx],
                class_embeddings,
            )
            losses.append(float(loss))
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"epoch {epoch+1}/{NUM_EPOCHS}  loss={np.mean(losses):.4f}")

    # final evaluation: seen-class accuracy vs. the real zero-shot number
    val_held_out_mask = np.isin(val_labels, np.array([CLASSES.index(c) for c in HELD_OUT_CLASSES]))
    val_seen_mask = ~val_held_out_mask

    preds_seen = evaluate(params, val_images[val_seen_mask], val_points[val_seen_mask], val_labels[val_seen_mask], class_embeddings)
    seen_acc = float(np.mean(preds_seen == val_labels[val_seen_mask]))

    preds_held_out = evaluate(params, val_images[val_held_out_mask], val_points[val_held_out_mask], val_labels[val_held_out_mask], class_embeddings)
    zero_shot_acc = float(np.mean(preds_held_out == val_labels[val_held_out_mask]))

    random_baseline = 1.0 / len(CLASSES)

    print()
    print(f"seen-class val accuracy (8 trained classes):      {seen_acc:.4f}")
    print(f"ZERO-SHOT accuracy (2 NEVER-trained classes):     {zero_shot_acc:.4f}")
    print(f"random baseline (10-way):                          {random_baseline:.4f}")
    print(f"held-out examples evaluated: {int(val_held_out_mask.sum())}")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")

    return seen_acc, zero_shot_acc


if __name__ == "__main__":
    main()
