"""
Splits the generated dataset into train/val shards.

Kept deliberately simple: a "shard" here is just an .npz file with a slice
of the full dataset. No need for a real distributed shard format at this
scale (a few thousand examples total).
"""

import os

import numpy as np

HERE = os.path.dirname(__file__)
GENERATED_PATH = os.path.join(HERE, "generated", "dataset.npz")
SHARDS_DIR = os.path.join(HERE, "shards")


def build_shards(val_fraction=0.15, examples_per_shard=500, seed=1):
    data = np.load(GENERATED_PATH, allow_pickle=True)
    images, pointclouds, labels, classes = (
        data["images"],
        data["pointclouds"],
        data["labels"],
        data["classes"],
    )

    n = len(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    images, pointclouds, labels = images[perm], pointclouds[perm], labels[perm]

    n_val = int(n * val_fraction)
    splits = {
        "train": (images[n_val:], pointclouds[n_val:], labels[n_val:]),
        "val": (images[:n_val], pointclouds[:n_val], labels[:n_val]),
    }

    os.makedirs(SHARDS_DIR, exist_ok=True)
    for split_name, (imgs, pts, lbls) in splits.items():
        split_dir = os.path.join(SHARDS_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        n_split = len(lbls)
        n_shards = max(1, (n_split + examples_per_shard - 1) // examples_per_shard)
        for shard_idx in range(n_shards):
            start = shard_idx * examples_per_shard
            end = min(start + examples_per_shard, n_split)
            shard_path = os.path.join(split_dir, f"shard_{shard_idx:03d}.npz")
            np.savez(
                shard_path,
                images=imgs[start:end],
                pointclouds=pts[start:end],
                labels=lbls[start:end],
                classes=classes,
            )
        print(f"{split_name}: {n_split} examples -> {n_shards} shards in {split_dir}")


if __name__ == "__main__":
    build_shards()
