"""
Verifies kernels/reference.py's algorithm matches the real model's attention
math exactly (up to floating point noise), BEFORE trusting it as the spec
for the CUDA kernel. Runs on CPU -- no GPU needed, this just checks the math.
"""

import numpy as np
from jax import random

from model import vision_encoder as ve
from kernels.reference import multi_head_attention_reference


def test_reference_matches_model_attention():
    key = random.PRNGKey(0)
    params = ve.init_params(key)
    block = params["blocks"][0]

    b, n, d = 2, 10, ve.EMBED_DIM
    x = random.normal(random.PRNGKey(1), (b, n, d)) * 0.1

    real_out = ve._attention(x, block)

    qkv = x @ block["qkv"]
    import jax.numpy as jnp

    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = np.array(q), np.array(k), np.array(v)

    ref_core_out = multi_head_attention_reference(q, k, v, ve.NUM_HEADS)
    ref_final_out = ref_core_out @ np.array(block["out_proj"])

    diff = np.abs(np.array(real_out) - ref_final_out).max()
    assert diff < 1e-4, f"reference implementation diverges from the real model: max diff {diff}"
