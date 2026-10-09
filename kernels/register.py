"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

UPDATE 3 (same Colab session): the api_version=1 "legacy" ABI (previous
version of this file) is NOT supported by this jaxlib/CUDA-plugin build
at all -- confirmed by JAX's OWN internal plugin init hitting the
identical "Unsupported custom call target type for api_version=1" error
independent of our code. custom_call.cpp was rewritten to the modern
typed-FFI handler convention (api_version=4, CallFrame-based, built
against `xla/ffi/api/ffi.h` -- confirmed present at
`<jax.ffi.include_dir()>/xla/ffi/api/ffi.h` on Colab).

That handler self-registers via the `XLA_FFI_REGISTER_HANDLER` C++ macro
when the shared library is loaded (dlopen runs static initializers
regardless of which language triggered the load) -- so this file no
longer calls `jax.ffi.register_ffi_target` manually. Loading the .so via
ctypes is still needed to trigger that static registration in the first
place.

`custom_call_api_version` now uses the default (4), and the `num_heads`
attribute is passed as a keyword argument on the INNER call (the
callable `ffi_call(...)` returns), matching `.Attr<int32_t>("num_heads")`
in custom_call.cpp's binding.

Build step (run on Colab, not here -- needs the FFI header path):
    nvcc -shared -Xcompiler -fPIC -arch=sm_75 -std=c++17 \
        -I$(python3 -c "import jax.ffi; print(jax.ffi.include_dir())") \
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

    # loading the library runs its static initializers, which is what
    # actually registers "fused_attention" with XLA (see custom_call.cpp's
    # XLA_FFI_REGISTER_HANDLER) -- no Python-side registration call needed
    ctypes.CDLL(SO_PATH)
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

    call = jax.ffi.ffi_call(
        "fused_attention",
        jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype),
    )
    out_heads = call(q_heads, k_heads, v_heads, num_heads=num_heads)

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
