"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

Targets `jax.extend.ffi` on a pinned jax/jaxlib==0.4.34 (see
scripts/colab_sync.md for why -- Colab's default 0.11.1 has a CUDA
plugin registry issue unrelated to this code). `jax.extend.ffi` is a
LAZY submodule here: `import jax` alone doesn't attach it as an
attribute (confirmed: `hasattr(jax, "extend")` is False even though
`import jax.extend.ffi` then works) -- so this file imports it
explicitly rather than accessing `jax.extend.ffi` off a bare `import jax`.

Confirmed via `inspect.signature` AND by reading the installed source
directly (`jax/_src/extend/ffi.py`) on Colab:
- `register_ffi_target(name, fn, platform="cpu", api_version=1, **kwargs)`
- `ffi_call(target_name, result_shape_dtypes, *args, vectorized=False,
  has_side_effect=False, **kwargs)` -- single-stage (operands passed
  directly, no separate builder step), and `**kwargs` always flows
  through to the typed FFI attribute-binding machinery
  (`ffi_lowering` -> CallFrame attributes), regardless of api_version.
  That means `ffi_call` was never the right tool for invoking a plain
  legacy "void** buffers + opaque bytes" handler -- this kernel's C++
  side (custom_call.cpp) targets the typed-FFI convention instead
  (`XLA_FFI_DEFINE_HANDLER_SYMBOL`, api_version=4), the same code that
  compiled and linked cleanly on jaxlib 0.11.1 too.

Build step (run on Colab, not here -- needs the FFI header path, which
exists at this jax/jaxlib version too):
    nvcc -shared -Xcompiler -fPIC -arch=sm_75 -std=c++17 \
        -I$(python3 -c "import jax.extend.ffi; print(jax.extend.ffi.include_dir())") \
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

    # imported lazily: jax.extend.ffi genuinely doesn't exist on every jax
    # version (confirmed absent on the M1's local jax 0.11.2) -- this module
    # must still import cleanly there, since kernels/tests/test_correctness.py
    # imports it at module level and is expected to just skip, not error
    import jax.extend.ffi

    lib = ctypes.CDLL(SO_PATH)
    handler_fn = getattr(lib, "FusedAttentionHandler")
    capsule = jax.extend.ffi.pycapsule(handler_fn)
    jax.extend.ffi.register_ffi_target(
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

    out_heads = jax.extend.ffi.ffi_call(
        "fused_attention",
        jax.ShapeDtypeStruct(q_heads.shape, q_heads.dtype),
        q_heads,
        k_heads,
        v_heads,
        num_heads=num_heads,
    )

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
