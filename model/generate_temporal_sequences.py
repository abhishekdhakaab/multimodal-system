"""
Builds short multi-view "sequences" for the temporal/bucket-KV-cache
experiment: a robot circling a real object over time, capturing T frames
from different (random) viewpoints of the SAME real mesh each time --
using the exact same per-object rendering machinery already used for
training augmentation (data_pipeline.modelnet_loader.build_example),
just called T times per object with different seeds instead of once.
This is NOT synthetic/fabricated identity -- every frame is a real
object, same honesty standard as the rest of this project.

Each frame is immediately run through the FROZEN, already-trained
single-frame backbone (model.pkl) to get its 96-dim fused embedding
(vision_encoder -> lidar_encoder -> fusion.cross_attend, the exact
pre-classification-head representation fusion.forward() uses) -- the
backbone itself is not modified or retrained here, only its output is
cached per frame. This keeps the temporal experiment's own training
(model/train_temporal.py) fast: it only ever sees pre-computed
embeddings, not raw images/point clouds.

Usage: python -m model.generate_temporal_sequences
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from data_pipeline.modelnet_loader import build_example, list_examples
from model import fusion, lidar_encoder, vision_encoder

HERE = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "model.pkl")
OUT_DIR = os.path.join(HERE, "..", "data_pipeline", "temporal")

FRAMES_PER_SEQUENCE = 8
N_TRAIN_OBJECTS = 400
N_VAL_OBJECTS = 150


@jax.jit
def _embed_frame_jit(params, image, points):
    vision_embed = vision_encoder.forward(params["vision"], image[None, :, :])
    lidar_points = lidar_encoder.forward_per_point(params["lidar"], points[None, :, :])
    return fusion.cross_attend(params["fusion"], vision_embed, lidar_points)[0]


def embed_frame(params, image, points):
    """Runs one frame through the frozen backbone, returns the 96-dim
    fused embedding BEFORE the classification head -- exactly what
    fusion.forward() feeds into head_w/head_b."""
    return np.array(_embed_frame_jit(params, image, points))


def generate_split(params, modelnet_split, n_objects, seed):
    examples = list_examples(modelnet_split)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(examples), size=min(n_objects, len(examples)), replace=False)

    all_embeddings, all_labels = [], []
    for i, idx in enumerate(chosen):
        off_path, cls = examples[idx]
        frame_embeddings = []
        for frame_idx in range(FRAMES_PER_SEQUENCE):
            frame_rng = np.random.default_rng(seed * 100000 + idx * 100 + frame_idx)
            image, points, label = build_example(off_path, cls, frame_rng)
            frame_embeddings.append(embed_frame(params, jnp.array(image), jnp.array(points)))
        all_embeddings.append(np.stack(frame_embeddings))  # [T, 96]
        all_labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chosen)} objects done")

    return np.stack(all_embeddings), np.array(all_labels)  # [N, T, 96], [N]


def main():
    with open(CHECKPOINT_PATH, "rb") as f:
        params = pickle.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"generating {N_TRAIN_OBJECTS} train sequences ({FRAMES_PER_SEQUENCE} frames each)...")
    train_embeddings, train_labels = generate_split(params, "train", N_TRAIN_OBJECTS, seed=10)
    np.savez(os.path.join(OUT_DIR, "train.npz"), embeddings=train_embeddings, labels=train_labels)
    print(f"saved train: {train_embeddings.shape}")

    print(f"generating {N_VAL_OBJECTS} val sequences ({FRAMES_PER_SEQUENCE} frames each)...")
    val_embeddings, val_labels = generate_split(params, "test", N_VAL_OBJECTS, seed=20)
    np.savez(os.path.join(OUT_DIR, "val.npz"), embeddings=val_embeddings, labels=val_labels)
    print(f"saved val: {val_embeddings.shape}")


if __name__ == "__main__":
    main()
