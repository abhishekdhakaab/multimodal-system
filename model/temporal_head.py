"""
The bucket/group-old-tokens mechanism: given T per-frame embeddings from
a short multi-view sequence (data_pipeline via
model/generate_temporal_sequences.py), bound the memory used to
aggregate them, the same real idea behind KV-cache eviction/bucketing in
long-context LLM serving (e.g. H2O, StreamingLLM) -- keep recent context
at full resolution, compress older context into fewer summary tokens --
applied here to a sequence of per-frame object embeddings instead of a
sequence of per-token language-model activations.

Three conditions, compared honestly in model/train_temporal.py:
  (a) no memory  -- classify from the latest frame alone (the existing,
      already-trained single-frame model, unmodified)
  (b) bucketed memory -- T frames compressed to a fixed M tokens via
      bucket_sequence(), regardless of how large T grows
  (c) full memory -- all T frames kept as individual tokens, no
      compression (the expensive upper bound bucketing is compared against)
"""

import jax.numpy as jnp
from jax import random

FUSION_DIM = 96  # matches model/fusion.py's combined embedding dim
RECENT_WINDOW = 2  # most recent frames kept individually, full resolution
GROUP_SIZE = 3  # older frames are averaged together in groups of this size


def bucket_sequence(frame_embeddings):
    """frame_embeddings: [B, T, FUSION_DIM] (oldest to newest) -> [B, M, FUSION_DIM]
    where M = RECENT_WINDOW + ceil((T - RECENT_WINDOW) / GROUP_SIZE).

    For this project's T=8, RECENT_WINDOW=2, GROUP_SIZE=3: the oldest 6
    frames become 2 averaged "bucket" tokens, plus the 2 most recent
    frames kept individually -> M=4 total, regardless of how large T
    might grow in a longer deployment (the whole point of bucketing).
    """
    b, t, d = frame_embeddings.shape
    n_old = t - RECENT_WINDOW
    assert n_old % GROUP_SIZE == 0, "this reference implementation assumes T divides evenly for simplicity"

    old_frames = frame_embeddings[:, :n_old, :]  # [B, n_old, D]
    recent_frames = frame_embeddings[:, n_old:, :]  # [B, RECENT_WINDOW, D]

    n_groups = n_old // GROUP_SIZE
    grouped = old_frames.reshape(b, n_groups, GROUP_SIZE, d).mean(axis=2)  # [B, n_groups, D]

    return jnp.concatenate([grouped, recent_frames], axis=1)  # [B, M, D]


def init_params(key, num_classes):
    keys = random.split(key, 4)
    return {
        "q_proj": random.normal(keys[0], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "k_proj": random.normal(keys[1], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "v_proj": random.normal(keys[2], (FUSION_DIM, FUSION_DIM)) * 0.1,
        "head_w": random.normal(keys[3], (FUSION_DIM, num_classes)) * 0.1,
        "head_b": jnp.zeros(num_classes),
    }


def _softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def forward(params, latest_embedding, memory_tokens):
    """
    latest_embedding: [B, FUSION_DIM]      (the current frame, used as the query)
    memory_tokens: [B, M, FUSION_DIM]      (bucketed OR full memory, caller's choice)
    -> logits: [B, num_classes]
    """
    q = latest_embedding @ params["q_proj"]  # [B, D]
    k = memory_tokens @ params["k_proj"]  # [B, M, D]
    v = memory_tokens @ params["v_proj"]  # [B, M, D]

    scores = jnp.einsum("bd,bmd->bm", q, k) / jnp.sqrt(FUSION_DIM)
    weights = _softmax(scores)
    fused = jnp.einsum("bm,bmd->bd", weights, v)  # [B, D]

    return fused @ params["head_w"] + params["head_b"]
