"""
Tiny Vision Transformer for 32x32 grayscale images.

Kept intentionally small (a few hundred thousand params) since this trains
on M1 CPU: 4x4 patches -> 64 patches, small embed dim, 2 transformer blocks.
"""

import jax
import jax.numpy as jnp
from jax import random

PATCH_SIZE = 4
IMAGE_SIZE = 40  # must match data_pipeline.modelnet_loader.IMAGE_SIZE
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 100
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


def _sink_patch_indices():
    """The 4 corner patches, in raster order. See _select_top_k_patches's
    docstring for why these are forced into every pruned selection
    regardless of content."""
    n_per_side = IMAGE_SIZE // PATCH_SIZE
    return jnp.array(
        [0, n_per_side - 1, (n_per_side - 1) * n_per_side, n_per_side * n_per_side - 1]
    )


def _select_top_k_patches(patches, k, use_sink_tokens=False):
    """patches: [B, NUM_PATCHES, patch_dim] -> (selected: [B, k, patch_dim], indices: [B, k])

    Content-adaptive token pruning: on this project's data (silhouettes
    rendered from sparse point-cloud projections, see
    data_pipeline/modelnet_loader.project_to_image), roughly half of all
    patches are near-empty background on average -- measured directly on
    the real training data, not assumed. Self-attention cost is O(N^2) in
    token count, so running it over a fixed, smaller budget of the most
    "informative" patches (by total pixel intensity -- a direct, cheap
    proxy for "this patch contains part of the object, not background")
    instead of all NUM_PATCHES is a real, content-driven compute reduction
    specific to this kind of sparse-render input, not a generic trick
    applied without looking at the data. See docs/token_pruning_notes.md
    for the measured accuracy/latency tradeoff.

    k is fixed across the batch (not a per-example dynamic count) so this
    stays a static shape under jax.jit -- which patches are chosen still
    varies per example, only the COUNT is fixed.

    use_sink_tokens: if True, the 4 corner patches are ALWAYS included
    regardless of content (adapting the "attention sink" idea from
    StreamingLLM-style sliding-window attention: a few fixed anchor
    positions attended to irrespective of their own importance, which
    stabilizes attention across a varying, content-dependent selection).
    This is a real adaptation to test, not an assumed win -- see
    docs/token_pruning_notes.md for whether it actually helps here, since
    a single-frame ViT doesn't have the same softmax-stability motivation
    sink tokens were originally introduced for in streaming LLM decoding.
    """
    importance = jnp.sum(patches, axis=-1)  # [B, NUM_PATCHES], total intensity per patch

    if not use_sink_tokens:
        _, top_idx = jax.lax.top_k(importance, k)  # [B, k]
        selected = jnp.take_along_axis(patches, top_idx[:, :, None], axis=1)
        return selected, top_idx

    sink_idx = _sink_patch_indices()  # [4]
    b = patches.shape[0]
    sink_idx_batched = jnp.broadcast_to(sink_idx[None, :], (b, sink_idx.shape[0]))

    # exclude sink positions from the content-ranked pool, then fill the
    # remaining budget from the highest-importance non-sink patches
    masked_importance = importance.at[:, sink_idx].set(-jnp.inf)
    _, rest_idx = jax.lax.top_k(masked_importance, k - sink_idx.shape[0])

    top_idx = jnp.concatenate([sink_idx_batched, rest_idx], axis=1)  # [B, k]
    selected = jnp.take_along_axis(patches, top_idx[:, :, None], axis=1)
    return selected, top_idx


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


def _attention_cuda_kernel(x, block):
    """Same math as _attention, but the core attention computation (the part
    profiled as the bottleneck -- see docs/phase2_profiling_notes.md) runs
    through the hand-written CUDA kernel from kernels/fused_attention.cu
    instead of plain JAX ops. Only usable on a machine with a built
    kernels/fused_attention.so (a CUDA GPU) -- see edge/cuda_path/."""
    from kernels.register import fused_attention  # imported lazily: this module must
    # still import cleanly on the M1, where kernels.register's ctypes.CDLL call would
    # fail at import time if this were a top-level import

    b, n, d = x.shape
    qkv = x @ block["qkv"]
    q, k, v = jnp.split(qkv, 3, axis=-1)

    attn_out = fused_attention(q, k, v, NUM_HEADS)  # [B, N, D], core attention only
    return attn_out @ block["out_proj"]


def _mlp(x, block):
    h = jnp.maximum(x @ block["mlp_w1"] + block["mlp_b1"], 0.0)  # ReLU
    return h @ block["mlp_w2"] + block["mlp_b2"]


def forward(params, images, backend="jax", prune_k=None, use_sink_tokens=False):
    """images: [B, IMAGE_SIZE, IMAGE_SIZE] float32 in [0,1] -> embedding: [B, EMBED_DIM] (the CLS token output)

    backend: "jax" (default, runs anywhere, including the M1) or
        "cuda_kernel" (uses the hand-written fused CUDA kernel for the core
        attention computation -- only works on a machine with a built
        kernels/fused_attention.so, see edge/cuda_path/).
    prune_k: if set, runs attention over only the top-k most "informative"
        patches (by pixel intensity) instead of all NUM_PATCHES -- see
        _select_top_k_patches's docstring and docs/token_pruning_notes.md.
        None (default) uses every patch, unchanged from the original model.
    use_sink_tokens: if prune_k is set, forces the 4 corner patches into
        every selection regardless of content -- see
        _select_top_k_patches's docstring.
    """
    attention_fn = _attention if backend == "jax" else _attention_cuda_kernel

    patches = _patchify(images)  # [B, NUM_PATCHES, patch_dim]

    if prune_k is not None:
        patches, patch_idx = _select_top_k_patches(patches, prune_k, use_sink_tokens=use_sink_tokens)
        pos_embed = jnp.take_along_axis(
            jnp.broadcast_to(params["pos_embed"][None, :, :], (images.shape[0], NUM_PATCHES, EMBED_DIM)),
            patch_idx[:, :, None],
            axis=1,
        )  # [B, k, EMBED_DIM], gathered to match the selected patches
    else:
        pos_embed = params["pos_embed"][None, :, :]

    x = patches @ params["patch_proj"]  # [B, k_or_NUM_PATCHES, EMBED_DIM]
    x = x + pos_embed

    cls = jnp.broadcast_to(params["cls_token"], (x.shape[0], 1, EMBED_DIM))
    x = jnp.concatenate([cls, x], axis=1)  # [B, (k_or_NUM_PATCHES)+1, EMBED_DIM]

    for block in params["blocks"]:
        attn_out = attention_fn(_layernorm(x, block["ln1_scale"], block["ln1_bias"]), block)
        x = x + attn_out
        mlp_out = _mlp(_layernorm(x, block["ln2_scale"], block["ln2_bias"]), block)
        x = x + mlp_out

    return x[:, 0, :]  # CLS token embedding
