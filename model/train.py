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

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINTS_DIR = os.path.join(HERE, "checkpoints")

LEARNING_RATE = 0.003
NUM_EPOCHS = 50
BATCH_SIZE = 64
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8


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


def loss_fn(params, images, pointclouds, labels):
    logits = forward(params, images, pointclouds)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def accuracy(params, images, pointclouds, labels):
    logits = forward(params, images, pointclouds)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


def init_adam_state(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}


@jax.jit
def train_step(params, adam_state, images, pointclouds, labels):
    loss, grads = jax.value_and_grad(loss_fn)(params, images, pointclouds, labels)

    t = adam_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, adam_state["m"], grads)
    v = jax.tree_util.tree_map(
        lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * (g**2), adam_state["v"], grads
    )
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2**t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: LEARNING_RATE * mh / (jnp.sqrt(vh) + ADAM_EPS), m_hat, v_hat
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

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        epoch_losses = []
        for idx in iterate_batches(train_images.shape[0], BATCH_SIZE, rng):
            idx = jnp.array(idx)
            params, adam_state, loss = train_step(
                params, adam_state, train_images[idx], train_points[idx], train_labels[idx]
            )
            epoch_losses.append(float(loss))

        val_acc = float(accuracy(params, val_images, val_points, val_labels))
        elapsed = time.time() - start
        print(
            f"epoch {epoch+1}/{NUM_EPOCHS}  "
            f"train_loss={np.mean(epoch_losses):.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "model.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"saved checkpoint to {ckpt_path}")

    final_val_acc = float(accuracy(params, val_images, val_points, val_labels))
    print(f"final val accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
