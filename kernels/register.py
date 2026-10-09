"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

REVERTED after a long real debugging arc on jaxlib 0.11.1 (Colab's
default version): that build's CUDA plugin doesn't support the legacy
ABI this kernel uses, so a modern typed-FFI rewrite was tried -- it
compiled and registered without any error, through two different
registration functions, but was never actually reachable at execution
time (`NOT_FOUND: No FFI handler registered for fused_attention`),
looking like a PJRT-plugin registry isolation issue specific to that
very recent build, not a glue-code mistake.

Pinning jax/jaxlib==0.4.34 sidesteps the whole problem: confirmed on
that version that `jax.ffi` doesn't exist (it's `jax.extend.ffi`), and
the FFI C++ header bundling needed for the typed-FFI approach isn't
available there either -- so the simple, long-established legacy ABI
(custom_call.cpp, reverted alongside this file) is the right one here.

HONEST UNCERTAINTY: whether `jax.extend.ffi.ffi_call` at this version
takes operands directly (single call) or returns a builder (two-stage,
like the newer `jax.ffi.ffi_call` turned out to) was not independently
verified before writing this -- both patterns are tried below, in order,
so this doesn't cost another round-trip if the first guess is wrong.

Build step (run on Colab, not here):
    nvcc -shared -Xcompiler -fPIC -arch=sm_75 \
        kernels/fused_attention.cu kernels/custom_call.cpp \
        -o kernels/fused_attention.so
"""

import ctypes
import os

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

    jax.extend.ffi.register_ffi_target(
        "fused_attention", target_capsule, platform="gpu", api_version=1
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

    opaque = f"{b} {num_heads} {n} {head_dim}"
    result_type = jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype)

    try:
        # guess 1: single-stage call, operands passed directly (matches the
        # ORIGINAL pre-Colab assumption for this older namespace)
        out_heads = jax.extend.ffi.ffi_call(
            "fused_attention",
            result_type,
            q_heads,
            k_heads,
            v_heads,
            custom_call_api_version=1,
            legacy_backend_config=opaque,
        )
    except TypeError:
        # guess 2: two-stage call (builder then call), matching the pattern
        # the newer jax.ffi.ffi_call turned out to use
        call = jax.extend.ffi.ffi_call(
            "fused_attention",
            result_type,
            custom_call_api_version=1,
            legacy_backend_config=opaque,
        )
        out_heads = call(q_heads, k_heads, v_heads)

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
