"""
Simple PointNet-style point cloud encoder: a per-point MLP shared across all
points, followed by a max-pool to get one permutation-invariant embedding
per point cloud. This is the standard minimal approach for point cloud
encoding and avoids anything requiring a custom CUDA op at this stage.
"""

import jax.numpy as jnp
from jax import random

POINT_DIM = 3  # x, y, z
HIDDEN_DIM = 48
EMBED_DIM = 48  # must match vision_encoder.EMBED_DIM for fusion


def init_params(key):
    keys = random.split(key, 4)
    return {
        "w1": random.normal(keys[0], (POINT_DIM, HIDDEN_DIM)) * 0.1,
        "b1": jnp.zeros(HIDDEN_DIM),
        "w2": random.normal(keys[1], (HIDDEN_DIM, EMBED_DIM)) * 0.1,
        "b2": jnp.zeros(EMBED_DIM),
    }


def forward_per_point(params, pointclouds):
    """pointclouds: [B, N, 3] -> per-point features: [B, N, EMBED_DIM] (used as cross-attention tokens)"""
    h = jnp.maximum(pointclouds @ params["w1"] + params["b1"], 0.0)  # [B, N, HIDDEN_DIM]
    h = jnp.maximum(h @ params["w2"] + params["b2"], 0.0)  # [B, N, EMBED_DIM]
    return h


def forward(params, pointclouds):
    """pointclouds: [B, N, 3] -> pooled embedding: [B, EMBED_DIM], permutation-invariant"""
    return jnp.max(forward_per_point(params, pointclouds), axis=1)
