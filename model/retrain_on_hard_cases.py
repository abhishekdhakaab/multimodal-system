"""
Closes the loop: runs the current best checkpoint on the full validation
pool (all 908 held-out real objects, across all 5 val shards -- a broader
pool than the single shard any one live pod serves, so there's enough
signal to split into "mined for retraining" vs "held-out eval").

  1. Find hard cases (low-confidence predictions) in the val pool.
  2. Split them in half: one half gets added to the training set and the
     model is briefly fine-tuned; the other half is held out and NEVER
     trained on -- it's what "before" and "after" are measured against.
  3. Report the real, honest accuracy change on that held-out hard-case
     set. No forced win -- whatever the number is, that's what gets
     written to docs/hard_case_mining_notes.md.

Usage: python -m model.retrain_on_hard_cases
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from data_pipeline.hard_case_miner import find_hard_cases
from model.full_model import forward, init_params
from model.train import (
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    BATCH_SIZE,
    NUM_CLASSES,
    WEIGHT_DECAY,
    compute_class_weights,
    init_adam_state,
    iterate_batches,
    loss_fn,
    train_step,
)

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")
FINETUNED_CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model_finetuned.pkl")

FINETUNE_EPOCHS = 8
FINETUNE_LR = 0.0005  # small -- this is a brief fine-tune on a checkpoint, not training from scratch


def load_all_val():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    images, pointclouds, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        pointclouds.append(d["pointclouds"])
        labels.append(d["labels"])
    return (
        np.concatenate(images),
        np.concatenate(pointclouds),
        np.concatenate(labels),
    )


def load_train_shards():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "train", "shard_*.npz")))
    images, pointclouds, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        pointclouds.append(d["pointclouds"])
        labels.append(d["labels"])
    return (
        np.concatenate(images),
        np.concatenate(pointclouds),
        np.concatenate(labels),
    )


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accuracy_on(params, images, points, labels):
    logits = forward(params, jnp.array(images), jnp.array(points))
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == jnp.array(labels)))


def main():
    print("loading checkpoint and full validation pool...")
    params = load_checkpoint(CHECKPOINT_PATH)
    val_images, val_points, val_labels = load_all_val()
    print(f"validation pool: {len(val_labels)} real held-out objects")

    logits = forward(params, jnp.array(val_images), jnp.array(val_points))
    probs = np.array(jax.nn.softmax(logits, axis=-1))
    hard_idx = find_hard_cases(probs, val_labels, threshold=0.6)
    print(f"found {len(hard_idx)} hard cases ({len(hard_idx)/len(val_labels)*100:.1f}% of the pool)")

    if len(hard_idx) < 10:
        print("too few hard cases to split meaningfully -- try a higher confidence threshold")
        return

    rng = np.random.default_rng(0)
    hard_idx = rng.permutation(hard_idx)
    mine_idx, eval_idx = np.array_split(hard_idx, 2)
    print(f"mining {len(mine_idx)} hard cases for retraining, holding out {len(eval_idx)} for evaluation")

    before_acc = accuracy_on(params, val_images[eval_idx], val_points[eval_idx], val_labels[eval_idx])
    print(f"BEFORE: accuracy on held-out hard-case eval set = {before_acc:.4f}")

    print("fine-tuning on train shards + mined hard cases...")
    train_images, train_points, train_labels = load_train_shards()
    # oversample the mined hard cases a few times so they carry real weight
    # in the fine-tune batch mix, not get diluted into near-nothing
    mined_images = np.tile(val_images[mine_idx], (5, 1, 1))
    mined_points = np.tile(val_points[mine_idx], (5, 1, 1))
    mined_labels = np.tile(val_labels[mine_idx], 5)

    ft_images = jnp.array(np.concatenate([train_images, mined_images]))
    ft_points = jnp.array(np.concatenate([train_points, mined_points]))
    ft_labels = jnp.array(np.concatenate([train_labels, mined_labels]))

    class_weights = compute_class_weights(ft_labels, NUM_CLASSES)
    adam_state = init_adam_state(params)
    rng2 = np.random.default_rng(1)

    for epoch in range(FINETUNE_EPOCHS):
        for idx in iterate_batches(ft_images.shape[0], BATCH_SIZE, rng2):
            idx = jnp.array(idx)
            params, adam_state, loss = train_step(
                params, adam_state, ft_images[idx], ft_points[idx], ft_labels[idx], class_weights, FINETUNE_LR
            )
        print(f"  fine-tune epoch {epoch+1}/{FINETUNE_EPOCHS}  loss={float(loss):.4f}")

    after_acc = accuracy_on(params, val_images[eval_idx], val_points[eval_idx], val_labels[eval_idx])
    print(f"AFTER:  accuracy on held-out hard-case eval set = {after_acc:.4f}")
    print(f"change: {after_acc - before_acc:+.4f}")

    overall_acc = accuracy_on(params, val_images, val_points, val_labels)
    print(f"sanity check -- overall val accuracy after fine-tune: {overall_acc:.4f}")

    with open(FINETUNED_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"saved fine-tuned checkpoint to {FINETUNED_CHECKPOINT_PATH}")

    return before_acc, after_acc, overall_acc


if __name__ == "__main__":
    main()
