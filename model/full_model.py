"""Wires vision encoder + lidar encoder + fusion head into one model."""

from jax import random

from model import vision_encoder, lidar_encoder, fusion


def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    return {
        "vision": vision_encoder.init_params(k1),
        "lidar": lidar_encoder.init_params(k2),
        "fusion": fusion.init_params(k3),
    }


def forward(params, images, pointclouds, backend="jax", prune_k=None):
    """images: [B,IMAGE_SIZE,IMAGE_SIZE], pointclouds: [B,N,3] -> logits: [B, NUM_CLASSES]

    backend: "jax" (runs anywhere) or "cuda_kernel" (vision encoder's core
    attention runs through the hand-written CUDA kernel -- GPU only, see
    edge/cuda_path/).
    prune_k: if set, the vision encoder only attends over its top-k most
    informative patches instead of all of them -- see
    model/vision_encoder.py's forward() and docs/token_pruning_notes.md.
    """
    vision_embed = vision_encoder.forward(params["vision"], images, backend=backend, prune_k=prune_k)
    lidar_points = lidar_encoder.forward_per_point(params["lidar"], pointclouds)
    logits = fusion.forward(params["fusion"], vision_embed, lidar_points)
    return logits
