"""
Same vision+lidar+cross-attention backbone as model/full_model.py, but
with a Mixture-of-Experts classification head (model/moe_head.py)
instead of one fixed linear head. See model/train_moe.py for the real
measured question this exists to answer: does the gate learn meaningful
specialization, or collapse onto one expert?
"""

from jax import random

from data_pipeline.modelnet_loader import CLASSES
from model import vision_encoder, lidar_encoder, fusion, moe_head

NUM_CLASSES = len(CLASSES)


def init_params(key):
    k1, k2, k3, k4 = random.split(key, 4)
    return {
        "vision": vision_encoder.init_params(k1),
        "lidar": lidar_encoder.init_params(k2),
        "fusion": fusion.init_params(k3),  # only q_proj/k_proj/v_proj used; head_w/head_b unused here
        "moe": moe_head.init_params(k4, NUM_CLASSES),
    }


def _combined_embedding(params, images, pointclouds):
    vision_embed = vision_encoder.forward(params["vision"], images)
    lidar_points = lidar_encoder.forward_per_point(params["lidar"], pointclouds)
    return fusion.cross_attend(params["fusion"], vision_embed, lidar_points)


def forward_soft(params, images, pointclouds):
    """-> (logits [B, NUM_CLASSES], gate_probs [B, NUM_EXPERTS]). Differentiable, used for training."""
    combined = _combined_embedding(params, images, pointclouds)
    return moe_head.forward_soft(params["moe"], combined)


def forward_hard(params, images, pointclouds):
    """-> (logits [B, NUM_CLASSES], chosen_expert [B]). What real sparse MoE inference would run."""
    combined = _combined_embedding(params, images, pointclouds)
    return moe_head.forward_hard(params["moe"], combined)
