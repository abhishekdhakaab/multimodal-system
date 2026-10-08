"""
Builds train/val shards directly from ModelNet10 (.off meshes), using
ModelNet10's own official train/test split (not one we invented).
For each mesh we sample a point cloud off the real surface and render a
2D projection of that same geometry as the "camera" image.

Kept simple: a "shard" here is just an .npz file holding a slice of
examples. No need for a real distributed shard format at this scale.
"""

import os

import numpy as np

from data_pipeline.modelnet_loader import (
    CLASSES,
    IMAGE_SIZE,
    NUM_POINTS,
    build_example,
    list_examples,
)

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "shards")
EXAMPLES_PER_SHARD = 200


def build_split(split_name, modelnet_split, seed, views_per_mesh=1):
    """views_per_mesh > 1 renders each mesh multiple times with a different
    random viewpoint/point sample each time -- cheap data augmentation that
    helps the model learn the object's geometry instead of memorizing one
    fixed view. Used for the train split only; val should see each real
    object once, like a normal evaluation set."""
    examples = list_examples(modelnet_split)
    rng = np.random.default_rng(seed)

    augmented = [(path, cls) for (path, cls) in examples for _ in range(views_per_mesh)]
    perm = rng.permutation(len(augmented))
    augmented = [augmented[i] for i in perm]

    split_dir = os.path.join(SHARDS_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    n = len(augmented)
    n_shards = max(1, (n + EXAMPLES_PER_SHARD - 1) // EXAMPLES_PER_SHARD)

    for shard_idx in range(n_shards):
        start = shard_idx * EXAMPLES_PER_SHARD
        end = min(start + EXAMPLES_PER_SHARD, n)
        images, pointclouds, labels = [], [], []
        for off_path, cls in augmented[start:end]:
            try:
                img, pts, label = build_example(off_path, cls, rng)
            except Exception as e:
                print(f"skipping {off_path}: {e}")
                continue
            images.append(img)
            pointclouds.append(pts)
            labels.append(label)

        shard_path = os.path.join(split_dir, f"shard_{shard_idx:03d}.npz")
        np.savez(
            shard_path,
            images=np.stack(images),
            pointclouds=np.stack(pointclouds),
            labels=np.array(labels, dtype=np.int64),
            classes=np.array(CLASSES),
        )
        print(f"{split_name} shard {shard_idx+1}/{n_shards}: {len(labels)} examples -> {shard_path}")

    print(f"{split_name}: {len(examples)} real objects, {n} examples total ({views_per_mesh}x views) -> {n_shards} shards in {split_dir}")


if __name__ == "__main__":
    build_split("train", "train", seed=1, views_per_mesh=5)
    build_split("val", "test", seed=2, views_per_mesh=1)  # ModelNet10's official "test" split, one view each
