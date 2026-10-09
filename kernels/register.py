"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

UPDATE (first real Colab run, JAX 0.11.1): `jax.extend.ffi` no longer
exists -- confirmed via `hasattr(jax, "extend")` -> False on the actual
Colab environment. The FFI API was promoted out of the experimental
`jax.extend` namespace to a stable top-level `jax.ffi` module.

UPDATE 2 (same session): `jax.ffi.ffi_call`'s real signature (confirmed
via `inspect.signature` on Colab) is a two-stage call --
`jax.ffi.ffi_call(target_name, result_shape_dtypes, **options)` returns a
callable, which is then called with the actual operands -- not a single
call taking operands directly. There's no `opaque=` kwarg; the legacy
config string is `legacy_backend_config`, passed at the OUTER call
(ffi_call itself), not the inner one. `custom_call_api_version` defaults
to 4 (the modern typed-FFI calling convention, which expects a C++
handler built against `xla/ffi/api/ffi.h`'s CallFrame-based API) -- our
custom_call.cpp implements the OLDER, simpler "buffers + opaque bytes"
ABI (api_version 1, XLA's "ORIGINAL" custom-call convention), so
`custom_call_api_version=1` must be set explicitly.
`jax.ffi.register_ffi_target`'s signature (also confirmed on Colab)
already defaults to `api_version=1`, which matches -- no change needed
there, but `platform="CUDA"` (not "gpu") is required.

Switched the opaque payload from packed struct bytes to a plain text
string ("B NUM_HEADS N HEAD_DIM") since `legacy_backend_config` is typed
`str`, not `bytes` -- avoids any binary/string encoding risk. See
custom_call.cpp's matching parse-side update.

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

    jax.ffi.register_ffi_target(
        "fused_attention", target_capsule, platform="CUDA", api_version=1
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

    call = jax.ffi.ffi_call(
        "fused_attention",
        jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype),
        custom_call_api_version=1,
        legacy_backend_config=opaque,
    )
    out_heads = call(q_heads, k_heads, v_heads)

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
