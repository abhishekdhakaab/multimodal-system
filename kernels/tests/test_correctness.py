"""
GPU correctness test for the real CUDA kernel -- run this on Colab, not
locally (needs a CUDA GPU). Compares the compiled kernel's output against
(a) kernels/reference.py's algorithm (already verified against the real
model on CPU, see test_reference.py) and (b) the real model's own
vision_encoder._attention, end to end.

Run with: pytest kernels/tests/test_correctness.py -v
(after building fused_attention.so per register.py's docstring)
"""

import os

import numpy as np
import pytest
from jax import random

from model import vision_encoder as ve
from kernels.register import SO_PATH, fused_attention

KERNEL_AVAILABLE = os.path.exists(SO_PATH)

pytestmark = pytest.mark.skipif(
    not KERNEL_AVAILABLE,
    reason=f"{SO_PATH} not built yet -- build on a CUDA machine (e.g. Colab) first, see register.py docstring",
)


def test_kernel_matches_reference_random_inputs():
    rng = np.random.default_rng(0)
    b, n, num_heads, head_dim = 2, 101, ve.NUM_HEADS, 12
    d = num_heads * head_dim

    q = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)
    k = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)
    v = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)

    kernel_out = np.array(fused_attention(q, k, v, num_heads))

    from kernels.reference import multi_head_attention_reference

    ref_out = multi_head_attention_reference(q, k, v, num_heads)

    max_diff = np.abs(kernel_out - ref_out).max()
    print(f"max abs diff (kernel vs reference): {max_diff}")
    assert max_diff < 1e-3, f"kernel output diverges from reference: {max_diff}"


def test_kernel_matches_real_model_end_to_end():
    key = random.PRNGKey(0)
    params = ve.init_params(key)
    block = params["blocks"][0]

    b, n, d = 2, ve.NUM_PATCHES + 1, ve.EMBED_DIM
    x = random.normal(random.PRNGKey(1), (b, n, d)) * 0.1

    real_out = np.array(ve._attention(x, block))

    qkv = x @ block["qkv"]
    import jax.numpy as jnp

    q, k, v = jnp.split(qkv, 3, axis=-1)
    kernel_core_out = fused_attention(q, k, v, ve.NUM_HEADS)
    kernel_final_out = np.array(kernel_core_out) @ np.array(block["out_proj"])

    max_diff = np.abs(real_out - kernel_final_out).max()
    print(f"max abs diff (kernel-based attention vs real model attention): {max_diff}")
    assert max_diff < 1e-3


def test_kernel_precision_fp16():
    """Precision/accuracy tradeoff check -- fp16 should be close but not
    identical to fp32. Report the gap honestly rather than assuming it's fine."""
    rng = np.random.default_rng(1)
    b, n, num_heads, head_dim = 2, 101, ve.NUM_HEADS, 12
    d = num_heads * head_dim

    q = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)
    k = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)
    v = rng.normal(0, 0.3, (b, n, d)).astype(np.float32)

    out_fp32 = np.array(fused_attention(q, k, v, num_heads))
    out_fp16 = np.array(
        fused_attention(q.astype(np.float16), k.astype(np.float16), v.astype(np.float16), num_heads)
    ).astype(np.float32)

    max_diff = np.abs(out_fp32 - out_fp16).max()
    print(f"max abs diff (fp32 vs fp16): {max_diff}")
    # no hard assert here -- this test's job is to report the number for
    # docs/benchmark_results.md, not to silently pass/fail on an arbitrary threshold
