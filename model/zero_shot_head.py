"""
DeViSE-style zero-shot classification head: instead of a fixed softmax
over a trained class count, project the fused embedding into a real
pretrained word-embedding space (GloVe) and classify by cosine similarity
against class-name embeddings.

Why this is genuinely zero-shot (not just a relabeled classifier): the
embedding lookup table (model/class_embeddings.npy) can include classes
the model was NEVER shown a single training image of -- held-out classes
get the exact same treatment as seen ones, since "prediction" is just
nearest-neighbor search in embedding space, not a per-class learned
weight vector the way a plain linear softmax head works. See
model/train_zero_shot.py for the actual held-out-class experiment and
docs/zero_shot_notes.md for the real measured result.
"""

import os

import jax.numpy as jnp
import numpy as np
from jax import random

from data_pipeline.modelnet_loader import CLASSES

FUSION_DIM = 96  # vision_embed (48) concatenated with attended lidar (48), see model/fusion.py
EMBED_DIM = 50  # GloVe dimensionality
TEMPERATURE = 10.0  # scales cosine similarity before cross-entropy, standard in embedding-matching heads

CLASS_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "class_embeddings.npy")


def load_class_embeddings():
    """Returns [NUM_CLASSES, EMBED_DIM], L2-normalized."""
    embeddings = np.load(CLASS_EMBEDDINGS_PATH)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return jnp.array(embeddings)


def init_params(key):
    return {
        "proj_w": random.normal(key, (FUSION_DIM, EMBED_DIM)) * 0.1,
        "proj_b": jnp.zeros(EMBED_DIM),
    }


def project(params, fused_embedding):
    """fused_embedding: [B, FUSION_DIM] -> [B, EMBED_DIM], L2-normalized."""
    projected = fused_embedding @ params["proj_w"] + params["proj_b"]
    return projected / (jnp.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8)


def similarity_logits(params, fused_embedding, class_embeddings):
    """fused_embedding: [B, FUSION_DIM], class_embeddings: [C, EMBED_DIM] (both
    normalized) -> [B, C] cosine similarities scaled by TEMPERATURE, usable
    directly with softmax/cross-entropy or argmax for prediction."""
    projected = project(params, fused_embedding)
    return TEMPERATURE * (projected @ class_embeddings.T)
