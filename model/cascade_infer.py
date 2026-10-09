"""
The draft-then-escalate cascade: run the cheap draft classifier first
(model/draft_classifier.py); if it's confident, use its answer and skip
the expensive model entirely; if not, fall back to the full fused model
(model/full_model.py). This is the draft-then-verify pattern from
speculative decoding, applied to classification latency rather than
autoregressive token generation (see model/draft_classifier.py's
docstring for why the literal action-generation version wasn't built).

Usage: python -m model.cascade_infer
"""

import glob
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import model.draft_classifier as draft
from model.full_model import forward as full_forward

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
FULL_CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")
DRAFT_CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "draft_classifier.pkl")

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 0.0 = always use draft, 1.0 = always escalate


def load_all_val():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return jnp.array(np.concatenate(images)), jnp.array(np.concatenate(points)), jnp.array(np.concatenate(labels))


def main():
    with open(FULL_CHECKPOINT_PATH, "rb") as f:
        full_params = pickle.load(f)
    with open(DRAFT_CHECKPOINT_PATH, "rb") as f:
        draft_params = pickle.load(f)

    images, points, labels = load_all_val()

    draft_logits = draft.forward(draft_params, images, points)
    draft_probs = jax.nn.softmax(draft_logits, axis=-1)
    draft_preds = jnp.argmax(draft_logits, axis=-1)
    draft_confidence = jnp.max(draft_probs, axis=-1)

    full_logits = full_forward(full_params, images, points)
    full_preds = jnp.argmax(full_logits, axis=-1)

    draft_only_acc = float(jnp.mean(draft_preds == labels))
    full_only_acc = float(jnp.mean(full_preds == labels))
    print(f"draft-only accuracy:  {draft_only_acc:.4f} (cheap, always used)")
    print(f"full-only accuracy:   {full_only_acc:.4f} (expensive, always used)")
    print()
    print(f"{'threshold':>10} {'draft-path %':>13} {'cascade accuracy':>18}")

    for threshold in THRESHOLDS:
        use_draft = draft_confidence >= threshold
        cascade_preds = jnp.where(use_draft, draft_preds, full_preds)
        cascade_acc = float(jnp.mean(cascade_preds == labels))
        draft_fraction = float(jnp.mean(use_draft))
        print(f"{threshold:>10.1f} {draft_fraction*100:>12.1f}% {cascade_acc:>18.4f}")


if __name__ == "__main__":
    main()
