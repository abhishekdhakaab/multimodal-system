"""
Same vision+lidar+cross-attention backbone as model/full_model.py, but
with the zero-shot embedding-matching head (model/zero_shot_head.py)
instead of a fixed softmax classifier. See model/train_zero_shot.py for
the held-out-class experiment this exists for.
"""

from jax import random

from model import vision_encoder, lidar_encoder, fusion, zero_shot_head


def init_params(key):
    k1, k2, k3, k4 = random.split(key, 4)
    return {
        "vision": vision_encoder.init_params(k1),
        "lidar": lidar_encoder.init_params(k2),
        "fusion": fusion.init_params(k3),  # only q_proj/k_proj/v_proj are used; head_w/head_b are unused here
        "zero_shot": zero_shot_head.init_params(k4),
    }


def similarity_logits(params, images, pointclouds, class_embeddings):
    """-> [B, NUM_CLASSES] cosine-similarity logits against class_embeddings
    (which may include classes never seen during training)."""
    vision_embed = vision_encoder.forward(params["vision"], images)
    lidar_points = lidar_encoder.forward_per_point(params["lidar"], pointclouds)
    combined = fusion.cross_attend(params["fusion"], vision_embed, lidar_points)
    return zero_shot_head.similarity_logits(params["zero_shot"], combined, class_embeddings)
