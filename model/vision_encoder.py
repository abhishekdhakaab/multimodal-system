"""
Tiny Vision Transformer for 32x32 grayscale images.

Kept intentionally small (a few hundred thousand params) since this trains
on M1 CPU: 4x4 patches -> 64 patches, small embed dim, 2 transformer blocks.
"""

import jax.numpy as jnp
from jax import random

PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
EMBED_DIM = 48
NUM_HEADS = 4
NUM_LAYERS = 2
MLP_DIM = 96


def init_params(key):
    keys = random.split(key, 10)
    patch_dim = PATCH_SIZE * PATCH_SIZE

    params = {
        "patch_proj": random.normal(keys[0], (patch_dim, EMBED_DIM)) * 0.02,
        "pos_embed": random.normal(keys[1], (NUM_PATCHES, EMBED_DIM)) * 0.02,
        "cls_token": random.normal(keys[2], (1, EMBED_DIM)) * 0.02,
        "blocks": [],
    }

    for i in range(NUM_LAYERS):
        bk = random.split(keys[3 + i], 8)
        block = {
            "qkv": random.normal(bk[0], (EMBED_DIM, 3 * EMBED_DIM)) * 0.02,
            "out_proj": random.normal(bk[1], (EMBED_DIM, EMBED_DIM)) * 0.02,
            "ln1_scale": jnp.ones(EMBED_DIM),
            "ln1_bias": jnp.zeros(EMBED_DIM),
            "mlp_w1": random.normal(bk[2], (EMBED_DIM, MLP_DIM)) * 0.02,
            "mlp_b1": jnp.zeros(MLP_DIM),
            "mlp_w2": random.normal(bk[3], (MLP_DIM, EMBED_DIM)) * 0.02,
            "mlp_b2": jnp.zeros(EMBED_DIM),
            "ln2_scale": jnp.ones(EMBED_DIM),
            "ln2_bias": jnp.zeros(EMBED_DIM),
        }
        params["blocks"].append(block)

    return params


def _layernorm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _patchify(images):
    """images: [B, 32, 32] -> patches: [B, NUM_PATCHES, patch_dim]"""
    b = images.shape[0]
    n_per_side = IMAGE_SIZE // PATCH_SIZE
    x = images.reshape(b, n_per_side, PATCH_SIZE, n_per_side, PATCH_SIZE)
    x = x.transpose(0, 1, 3, 2, 4)  # [B, n, n, P, P]
    x = x.reshape(b, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE)
    return x


def _attention(x, block):
    b, n, d = x.shape
    qkv = x @ block["qkv"]  # [B, N, 3D]
    q, k, v = jnp.split(qkv, 3, axis=-1)

    head_dim = d // NUM_HEADS
    q = q.reshape(b, n, NUM_HEADS, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, n, NUM_HEADS, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, n, NUM_HEADS, head_dim).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    weights = jax_softmax(scores)
    out = weights @ v  # [B, H, N, head_dim]

    out = out.transpose(0, 2, 1, 3).reshape(b, n, d)
    return out @ block["out_proj"]


def jax_softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def _mlp(x, block):
    h = jnp.maximum(x @ block["mlp_w1"] + block["mlp_b1"], 0.0)  # ReLU
    return h @ block["mlp_w2"] + block["mlp_b2"]


def forward(params, images):
    """images: [B, 32, 32] float32 in [0,1] -> embedding: [B, EMBED_DIM] (the CLS token output)"""
    patches = _patchify(images)  # [B, NUM_PATCHES, patch_dim]
    x = patches @ params["patch_proj"]  # [B, NUM_PATCHES, EMBED_DIM]
    x = x + params["pos_embed"][None, :, :]

    cls = jnp.broadcast_to(params["cls_token"], (x.shape[0], 1, EMBED_DIM))
    x = jnp.concatenate([cls, x], axis=1)  # [B, NUM_PATCHES+1, EMBED_DIM]

    for block in params["blocks"]:
        attn_out = _attention(_layernorm(x, block["ln1_scale"], block["ln1_bias"]), block)
        x = x + attn_out
        mlp_out = _mlp(_layernorm(x, block["ln2_scale"], block["ln2_bias"]), block)
        x = x + mlp_out

    return x[:, 0, :]  # CLS token embedding
