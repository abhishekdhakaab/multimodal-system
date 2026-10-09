"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

UPDATE 3: the api_version=1 "legacy" ABI (two versions ago) is NOT
supported by this jaxlib/CUDA-plugin build at all -- confirmed by JAX's
OWN internal plugin init hitting the identical "Unsupported custom call
target type for api_version=1" error independent of our code. Rewrote
custom_call.cpp to the modern typed-FFI handler convention (api_version=4).

UPDATE 4: the first typed-FFI attempt used `XLA_FFI_REGISTER_HANDLER`
for C++-side static self-registration, which failed to LINK (undefined
symbol `xla::ffi::GetXlaFfiApi`) -- that macro needs the full XLA
runtime linked in, which isn't available in pip-distributed jaxlib.
Read xla/ffi/api/api.h directly on Colab (rather than guessing a third
time) and found the right tool already documented there:
`XLA_FFI_DEFINE_HANDLER_SYMBOL`, explicitly "for users who want to
export XLA FFI handler from a shared library as a C function symbol" --
exactly this situation. custom_call.cpp now exports a plain `extern "C"`
function; this file grabs it via ctypes and registers it from Python
with `jax.ffi.register_ffi_target(..., api_version=4)` instead of
relying on static self-registration.

The `num_heads` attribute is passed as a keyword argument on the INNER
call (the callable `ffi_call(...)` returns), matching
`.Attr<int32_t>("num_heads")` in custom_call.cpp's binding.

UPDATE 5: with registration reaching XLA cleanly, the real error became
`No FFI handler registered for fused_attention on a platform CUDA
(canonical cuda)` -- XLA prints both the platform string we passed and
its "canonical" (lowercased) form, implying our registration under
`platform="CUDA"` and the dispatch-time lookup (on the canonicalized key)
never matched. Changed to `platform="cuda"` (lowercase).

KNOWN HARMLESS NOISE: every run also logs an ERROR from
`jax_plugins.xla_cuda13.initialize()` about "Unsupported custom call
target type" / "API version ... not supported" during lazy CUDA backend
init. This is JAX's own internal plugin registering ITS OWN handlers
(not "fused_attention"), unrelated to our registration -- confirmed
because it fires with contradictory complaints across different runs
(first said api_version=1 unsupported, then said api_version=4
unsupported, for internal calls we don't control) and never correlates
with whether our own tests pass or fail. Safe to ignore.

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

    lib = ctypes.CDLL(SO_PATH)
    handler_fn = getattr(lib, "FusedAttentionHandler")
    capsule = jax.ffi.pycapsule(handler_fn)
    jax.ffi.register_ffi_target(
        "fused_attention", capsule, platform="cuda", api_version=4
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

    call = jax.ffi.ffi_call(
        "fused_attention",
        jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype),
    )
    out_heads = call(q_heads, k_heads, v_heads, num_heads=num_heads)

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
