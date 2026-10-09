"""
A deliberately cheap "draft" classifier -- the speculative-decoding idea
(do the cheap thing first, only pay for the expensive thing when
necessary) applied to classification latency instead of autoregressive
token generation, which is what the user actually meant by "speculative
decoding" here: there's no real action-sequence data in this project to
decode, but the draft-then-verify systems pattern transfers cleanly to
classification as a draft-then-escalate cascade (model/cascade_infer.py).

No attention, no per-point MLP -- just hand-computed global statistics
of the image and point cloud, fed through a tiny 2-layer MLP. This is
intentionally far cheaper than the full ViT+PointNet+cross-attention
model (model/full_model.py), which is the whole point: the draft only
earns its place in the cascade if it's both cheap AND usually right.
"""

import jax.numpy as jnp
from jax import random

from data_pipeline.modelnet_loader import CLASSES

NUM_CLASSES = len(CLASSES)
NUM_FEATURES = 14  # 8 image quadrant stats + 6 point-cloud centroid/spread stats
HIDDEN_DIM = 32


def extract_features(images, pointclouds):
    """images: [B, H, W], pointclouds: [B, N, 3] -> [B, NUM_FEATURES]

    No learned parameters here at all -- these are hand-computed global
    statistics, as cheap as inference gets (a handful of means/stds over
    the raw input, no matrix multiplies beyond what jnp.mean/std already do).
    """
    b, h, w = images.shape
    h_mid, w_mid = h // 2, w // 2

    quadrants = [
        images[:, :h_mid, :w_mid],
        images[:, :h_mid, w_mid:],
        images[:, h_mid:, :w_mid],
        images[:, h_mid:, w_mid:],
    ]
    image_features = jnp.concatenate(
        [jnp.stack([q.mean(axis=(1, 2)), q.std(axis=(1, 2))], axis=-1) for q in quadrants], axis=-1
    )  # [B, 8]

    centroid = pointclouds.mean(axis=1)  # [B, 3]
    spread = pointclouds.std(axis=1)  # [B, 3]
    point_features = jnp.concatenate([centroid, spread], axis=-1)  # [B, 6]

    return jnp.concatenate([image_features, point_features], axis=-1)  # [B, 14]


def init_params(key):
    k1, k2 = random.split(key, 2)
    return {
        "w1": random.normal(k1, (NUM_FEATURES, HIDDEN_DIM)) * 0.1,
        "b1": jnp.zeros(HIDDEN_DIM),
        "w2": random.normal(k2, (HIDDEN_DIM, NUM_CLASSES)) * 0.1,
        "b2": jnp.zeros(NUM_CLASSES),
    }


def forward(params, images, pointclouds):
    """-> logits: [B, NUM_CLASSES]"""
    features = extract_features(images, pointclouds)
    h = jnp.maximum(features @ params["w1"] + params["b1"], 0.0)
    return h @ params["w2"] + params["b2"]
