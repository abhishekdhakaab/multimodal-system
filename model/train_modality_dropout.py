"""
Fixes what model/eval_robustness.py only diagnosed: the model was never
trained with a missing/corrupted modality, so it had no reason to learn
a fallback. This retrains from scratch with modality dropout -- each
training example independently has its lidar input zeroed with
probability P_DROP_LIDAR and its image zeroed with probability
P_DROP_IMAGE -- so the model is forced to keep usable signal in each
modality alone, not just lean on whichever is more informative on average.

Reuses model.train's train_step/loss_fn unchanged -- modality dropout is
applied to the batch BEFORE it reaches train_step, so the only change
here is what data the existing training loop sees, not the training
mechanics themselves.

Usage: python -m model.train_modality_dropout
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from model.full_model import forward, init_params
from model.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    LR_DECAY_AT_EPOCH,
    NUM_CLASSES,
    NUM_EPOCHS,
    compute_class_weights,
    init_adam_state,
    train_step,
)

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model_modality_dropout.pkl")

P_DROP_LIDAR = 0.15
P_DROP_IMAGE = 0.15


def load_split(split):
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, split, "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return np.concatenate(images), np.concatenate(points), np.concatenate(labels)


def accuracy(params, images, points, labels):
    logits = forward(params, images, points)
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))


def main():
    train_images, train_points, train_labels = load_split("train")
    val_images_np, val_points_np, val_labels_np = load_split("val")
    val_images, val_points, val_labels = jnp.array(val_images_np), jnp.array(val_points_np), jnp.array(val_labels_np)

    class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    key = random.PRNGKey(0)
    params = init_params(key)
    adam_state = init_adam_state(params)
    rng = np.random.default_rng(0)

    best_val_acc = 0.0
    best_params = params

    for epoch in range(NUM_EPOCHS):
        lr = LEARNING_RATE if epoch < LR_DECAY_AT_EPOCH else LEARNING_RATE * 0.3
        perm = rng.permutation(train_images.shape[0])
        losses = []
        for start in range(0, train_images.shape[0], BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            batch_images = train_images[idx].copy()
            batch_points = train_points[idx].copy()

            drop_lidar = rng.random(len(idx)) < P_DROP_LIDAR
            drop_image = rng.random(len(idx)) < P_DROP_IMAGE
            batch_points[drop_lidar] = 0.0
            batch_images[drop_image] = 0.0

            params, adam_state, loss = train_step(
                params,
                adam_state,
                jnp.array(batch_images),
                jnp.array(batch_points),
                jnp.array(train_labels[idx]),
                class_weights,
                lr,
            )
            losses.append(float(loss))

        val_acc = accuracy(params, val_images, val_points, val_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"epoch {epoch+1}/{NUM_EPOCHS}  loss={np.mean(losses):.4f}  val_acc={val_acc:.4f}  (best={best_val_acc:.4f})")

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, best_params), f)
    print(f"\nclean val accuracy with modality-dropout training: {best_val_acc:.4f}")
    print(f"saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
