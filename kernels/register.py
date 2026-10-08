"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

HONESTY NOTE (same as custom_call.cpp): JAX's custom-call/FFI API has moved
across versions. This targets `jax.extend.ffi`, the modern documented path
for registering a custom call. It has NOT been run (no CUDA on this
machine) -- this is the first thing to debug on Colab. If `jax.extend.ffi`
doesn't match the installed JAX version's exact API, the fallback is the
older `jax.lib.xla_client.register_custom_call_target` +
`jax.lax.custom_call` /  manual XLA custom-call lowering, which is more
verbose but has been stable for longer. Both custom_call.cpp's exposed
symbol and this file assume the "opaque buffers" ABI described in
custom_call.cpp's comments.

Build step (run on Colab, not here):
    nvcc -shared -Xcompiler -fPIC \
        kernels/fused_attention.cu kernels/custom_call.cpp \
        -o kernels/fused_attention.so
"""

import ctypes
import os
import struct

import jax
import jax.numpy as jnp

SO_PATH = os.path.join(os.path.dirname(__file__), "fused_attention.so")

_registered = False


def _ensure_registered():
    global _registered
    if _registered:
        return
    if not os.path.exists(SO_PATH):
        raise FileNotFoundError(
            f"{SO_PATH} not found -- build it first with the nvcc command in this file's docstring "
            "(requires a CUDA GPU + nvcc, e.g. on Colab)."
        )

    lib = ctypes.CDLL(SO_PATH)
    target_capsule = ctypes.cast(
        getattr(lib, "FusedAttentionCustomCall"), ctypes.c_void_p
    )

    # modern path: jax.extend.ffi.register_ffi_target.
    # if this errors on the installed JAX version, fall back to:
    #   from jax.lib import xla_client
    #   xla_client.register_custom_call_target("fused_attention", target_capsule, platform="gpu")
    jax.extend.ffi.register_ffi_target(
        "fused_attention", target_capsule, platform="gpu"
    )
    _registered = True


def fused_attention(q, k, v, num_heads):
    """q, k, v: [B, N, D] (already projected, NOT yet split into heads).
    Returns: [B, N, D], the attention output (before the final out_proj).

    This replaces vision_encoder._attention's internal reshape + scaled dot
    product + softmax + weighted-sum steps with one fused kernel call. The
    caller still does qkv projection and the final out_proj as ordinary JAX
    matmuls -- only the "core attention" computation is fused, matching the
    profiling finding in docs/phase2_profiling_notes.md and the FlashAttention-
    style motivation described in fused_attention.cu.
    """
    _ensure_registered()

    b, n, d = q.shape
    head_dim = d // num_heads

    # reshape [B, N, D] -> [B, num_heads, N, head_dim], contiguous, matching
    # what the kernel expects (see fused_attention.cu's shape comment)
    def split_heads(x):
        return x.reshape(b, n, num_heads, head_dim).transpose(0, 2, 1, 3)

    q_heads = split_heads(q)
    k_heads = split_heads(k)
    v_heads = split_heads(v)

    opaque = struct.pack("iiii", b, num_heads, n, head_dim)

    out_heads = jax.extend.ffi.ffi_call(
        "fused_attention",
        jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype),
        q_heads,
        k_heads,
        v_heads,
        opaque=opaque,
    )

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
