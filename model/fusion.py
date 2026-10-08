"""
Fuses the vision and lidar modalities with one real cross-attention layer:
the vision CLS embedding is the query, and the lidar per-point features
(before pooling) are the keys/values — so the model learns which lidar
points to attend to for a given visual embedding. The fused output is
concatenated with the vision embedding and passed through a small linear
head to predict the shape class.
"""

import jax.numpy as jnp
from jax import random

from model.vision_encoder import EMBED_DIM as VISION_DIM
from model.lidar_encoder import EMBED_DIM as LIDAR_DIM
from data_pipeline.modelnet_loader import CLASSES

FUSION_DIM = 48
NUM_CLASSES = len(CLASSES)  # single source of truth -- this used to be a hardcoded 4,
# a leftover from the old synthetic-shapes dataset, which silently capped the model's
# output head at 4 of ModelNet10's 10 classes and was the real cause of the fused
# model badly underperforming single-modality ablations (see docs/fusion_bug_notes.md)

assert VISION_DIM == LIDAR_DIM == FUSION_DIM, "encoder dims must match for this simple fusion"


def init_params(key):
    keys = random.split(key, 4)
    return {
        "q_proj": random.normal(keys[0], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "k_proj": random.normal(keys[1], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "v_proj": random.normal(keys[2], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "head_w": random.normal(keys[3], (FUSION_DIM * 2, NUM_CLASSES)) * 0.1,
        "head_b": jnp.zeros(NUM_CLASSES),
    }


def _softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def forward(params, vision_embed, lidar_points):
    """
    vision_embed: [B, FUSION_DIM]        (one query token, the vision CLS embedding)
    lidar_points: [B, N, FUSION_DIM]     (N key/value tokens, per-point lidar features)
    -> logits: [B, NUM_CLASSES]
    """
    q = vision_embed @ params["q_proj"]  # [B, D]
    k = lidar_points @ params["k_proj"]  # [B, N, D]
    v = lidar_points @ params["v_proj"]  # [B, N, D]

    scores = jnp.einsum("bd,bnd->bn", q, k) / jnp.sqrt(FUSION_DIM)  # [B, N]
    weights = _softmax(scores)  # [B, N], attention over lidar points
    fused = jnp.einsum("bn,bnd->bd", weights, v)  # [B, D]

    combined = jnp.concatenate([vision_embed, fused], axis=-1)  # [B, 2D]
    logits = combined @ params["head_w"] + params["head_b"]
    return logits
