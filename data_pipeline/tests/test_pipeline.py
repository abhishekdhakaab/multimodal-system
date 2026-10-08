import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
SHARDS_TRAIN_DIR = os.path.join(HERE, "..", "shards", "train")


def test_shards_exist():
    assert os.path.isdir(SHARDS_TRAIN_DIR), "run build_shards.py before running tests"
    shard_files = [f for f in os.listdir(SHARDS_TRAIN_DIR) if f.endswith(".npz")]
    assert len(shard_files) > 0


def test_shard_roundtrip_shapes():
    shard_files = sorted(f for f in os.listdir(SHARDS_TRAIN_DIR) if f.endswith(".npz"))
    data = np.load(os.path.join(SHARDS_TRAIN_DIR, shard_files[0]), allow_pickle=True)

    images, pointclouds, labels, classes = (
        data["images"],
        data["pointclouds"],
        data["labels"],
        data["classes"],
    )

    n = len(labels)
    assert images.shape == (n, 32, 32)
    assert pointclouds.shape == (n, 64, 3)
    assert labels.shape == (n,)
    assert set(np.unique(labels)).issubset(set(range(len(classes))))


def test_classes_are_visually_distinct():
    # sanity check: each class's average image should differ noticeably from the others.
    # this catches a broken generator that accidentally draws the same shape for every class.
    import sys

    sys.path.insert(0, os.path.join(HERE, ".."))
    from generate import CLASSES, generate_dataset

    images, _, labels = generate_dataset(n_per_class=20, seed=42)
    means = []
    for i in range(len(CLASSES)):
        means.append(images[labels == i].mean(axis=0))

    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            diff = np.abs(means[i] - means[j]).mean()
            assert diff > 0.01, f"classes {CLASSES[i]} and {CLASSES[j]} look too similar"


def test_hard_case_miner_flags_low_confidence():
    from hard_case_miner import find_hard_cases

    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    labels = np.array([0, 1, 1])
    hard = find_hard_cases(probs, labels, threshold=0.6)
    assert hard == [1]
