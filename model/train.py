"""
Trains the multimodal fusion model on the shards from Phase 1.
Plain SGD-with-momentum-free gradient descent via jax.grad, no optimizer
library needed at this scale. Runs on CPU on the M1.
"""

import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from model.full_model import init_params, forward
from data_pipeline.modelnet_loader import CLASSES

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINTS_DIR = os.path.join(HERE, "checkpoints")

LEARNING_RATE = 0.003
LR_DECAY_AT_EPOCH = 30  # halve the LR partway through -- lets the model settle instead of oscillating near the optimum
NUM_EPOCHS = 50
BATCH_SIZE = 64
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-4  # L2 regularization, to fight overfitting on the small real dataset
NUM_CLASSES = len(CLASSES)  # single source of truth, see model/fusion.py's comment on this


def load_split(split):
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, split, "shard_*.npz")))
    images, pointclouds, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        pointclouds.append(d["pointclouds"])
        labels.append(d["labels"])
    return (
        jnp.array(np.concatenate(images)),
        jnp.array(np.concatenate(pointclouds)),
        jnp.array(np.concatenate(labels)),
    )


def compute_class_weights(labels, num_classes):
    """Inverse-frequency weights so the 8x imbalance between the biggest and
    smallest ModelNet10 class (chair vs bathtub) doesn't bias the model
    toward just predicting the majority classes."""
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return jnp.array(weights)


def loss_fn(params, images, pointclouds, labels, class_weights):
    logits = forward(params, images, pointclouds)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    per_example_loss = -jnp.sum(one_hot * log_probs, axis=-1)
    sample_weights = class_weights[labels]
    weighted_loss = jnp.sum(per_example_loss * sample_weights) / jnp.sum(sample_weights)

    l2 = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
    return weighted_loss + WEIGHT_DECAY * l2


def accuracy(params, images, pointclouds, labels):
    logits = forward(params, images, pointclouds)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


def init_adam_state(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}


@jax.jit
def train_step(params, adam_state, images, pointclouds, labels, class_weights, learning_rate):
    loss, grads = jax.value_and_grad(loss_fn)(params, images, pointclouds, labels, class_weights)

    t = adam_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, adam_state["m"], grads)
    v = jax.tree_util.tree_map(
        lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * (g**2), adam_state["v"], grads
    )
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2**t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: learning_rate * mh / (jnp.sqrt(vh) + ADAM_EPS), m_hat, v_hat
    )
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)

    return params, {"m": m, "v": v, "t": t}, loss


def iterate_batches(n, batch_size, rng):
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


def main():
    print("loading data...")
    train_images, train_points, train_labels = load_split("train")
    val_images, val_points, val_labels = load_split("val")
    print(f"train: {train_images.shape[0]} examples, val: {val_images.shape[0]} examples")

    key = random.PRNGKey(0)
    params = init_params(key)
    adam_state = init_adam_state(params)
    rng = np.random.default_rng(0)
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    print(f"class weights (inverse frequency): {np.array(class_weights).round(2)}")

    best_val_acc = 0.0
    best_params = params
    for epoch in range(NUM_EPOCHS):
        lr = LEARNING_RATE if epoch < LR_DECAY_AT_EPOCH else LEARNING_RATE * 0.3
        start = time.time()
        epoch_losses = []
        for idx in iterate_batches(train_images.shape[0], BATCH_SIZE, rng):
            idx = jnp.array(idx)
            params, adam_state, loss = train_step(
                params, adam_state, train_images[idx], train_points[idx], train_labels[idx], class_weights, lr
            )
            epoch_losses.append(float(loss))

        val_acc = float(accuracy(params, val_images, val_points, val_labels))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
        elapsed = time.time() - start
        print(
            f"epoch {epoch+1}/{NUM_EPOCHS}  "
            f"train_loss={np.mean(epoch_losses):.4f}  val_acc={val_acc:.4f}  "
            f"(best={best_val_acc:.4f})  ({elapsed:.1f}s)"
        )

    # keep the checkpoint from the epoch with the best validation accuracy,
    # not necessarily the last one -- the model can overfit past its peak
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "model.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, best_params), f)
    print(f"saved checkpoint to {ckpt_path}")
    print(f"best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
