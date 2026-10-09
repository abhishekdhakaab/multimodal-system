"""
Diagnoses and partially corrects the seen-class bias ("hubness") problem
observed in the zero-shot experiment (model/train_zero_shot.py): seen
classes systematically win the argmax over unseen ones because the model
was only ever pushed to produce high similarity scores in seen classes'
directions.

Mitigation: "calibrated stacking" (Chao et al., ECCV 2016) -- subtract a
single scalar gamma from every seen-class score before argmax, so unseen
classes compete on a leveled playing field. This is a real, published
technique, not something invented for this project.

Honesty note: gamma is swept and reported directly against the actual
zero-shot accuracy here, which is an exploration of the technique's effect,
not a properly cross-validated hyperparameter search (that would need a
held-out set of *labeled unseen* examples to tune on without peeking at
the real test set, which this project doesn't have enough data to afford).
Report this as "here's the tradeoff calibration exposes," not as a
tuned, generalizable gamma value.

Usage: python -m model.calibrate_zero_shot
"""

import glob
import os
import pickle

import jax.numpy as jnp
import numpy as np

from data_pipeline.modelnet_loader import CLASSES
from model.full_model_zero_shot import similarity_logits
from model.zero_shot_head import load_class_embeddings
from model.train_zero_shot import HELD_OUT_CLASSES, SEEN_CLASSES, CHECKPOINT_PATH

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")


def load_all_val():
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    images, points, labels = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        images.append(d["images"])
        points.append(d["pointclouds"])
        labels.append(d["labels"])
    return np.concatenate(images), np.concatenate(points), np.concatenate(labels)


def main():
    with open(CHECKPOINT_PATH, "rb") as f:
        params = pickle.load(f)

    images, points, labels = load_all_val()
    class_embeddings = load_class_embeddings()
    logits = np.array(similarity_logits(params, jnp.array(images), jnp.array(points), class_embeddings))

    held_out_idx = np.array([CLASSES.index(c) for c in HELD_OUT_CLASSES])
    seen_idx = np.array([CLASSES.index(c) for c in SEEN_CLASSES])
    held_out_mask = np.isin(labels, held_out_idx)
    seen_mask = ~held_out_mask

    print("gamma  seen_acc  zero_shot_acc")
    for gamma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        calibrated = logits.copy()
        calibrated[:, seen_idx] -= gamma

        preds = calibrated.argmax(axis=-1)
        seen_acc = float(np.mean(preds[seen_mask] == labels[seen_mask]))
        zero_shot_acc = float(np.mean(preds[held_out_mask] == labels[held_out_mask]))
        print(f"{gamma:5.1f}  {seen_acc:.4f}    {zero_shot_acc:.4f}")


if __name__ == "__main__":
    main()
