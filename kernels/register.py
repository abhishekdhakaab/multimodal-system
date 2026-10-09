"""
JAX-side registration of the custom CUDA kernel as a real JAX primitive,
usable inside jit like any other op.

CONFIRMED, with hard evidence, on pinned jax/jaxlib==0.4.34 (see
scripts/colab_sync.md for the full real debugging history):
- `register_custom_call_target` itself raises an explicit error for
  api_version=4 ("Supported versions are 0 and 1") -- this jaxlib
  build's compiled backend genuinely doesn't implement the typed-FFI
  convention, regardless of what the Python wrapper functions expose.
- `jax.extend.ffi.ffi_call`'s own lowering hardcodes
  `kwargs.setdefault("api_version", 4)` with no public parameter to
  override it (read directly from jax/_src/extend/ffi.py's source) --
  so `ffi_call` structurally cannot invoke an api_version=1 (legacy ABI)
  target at this jax version, not a usage mistake.

This file bypasses `ffi_call`/`ffi_lowering` entirely and defines a
plain JAX primitive with a hand-written MLIR lowering rule via
`jax.interpreters.mlir.custom_call`, which DOES expose `api_version` and
accepts `backend_config` as a raw byte string (the legacy convention,
unlike the newer dict-of-attributes convention `ffi_lowering` always
uses) -- matching custom_call.cpp's simple ABI exactly.

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
_primitive = None


def _ensure_registered():
    global _registered, _primitive
    if _registered:
        return
    if not os.path.exists(SO_PATH):
        raise FileNotFoundError(
            f"{SO_PATH} not found -- build it first with the nvcc command in this file's docstring "
            "(requires a CUDA GPU + nvcc, e.g. on Colab)."
        )

    # imported lazily: these aren't needed (and may not exist) on every
    # jax version/platform -- this module must still import cleanly on
    # the M1, where there's no CUDA at all
    from jax._src import core
    from jax.interpreters import mlir
    from jaxlib import xla_client

    lib = ctypes.CDLL(SO_PATH)
    target_capsule = ctypes.cast(
        getattr(lib, "FusedAttentionCustomCall"), ctypes.c_void_p
    )
    xla_client.register_custom_call_target(
        "fused_attention", target_capsule, platform="gpu", api_version=1
    )

    prim = core.Primitive("fused_attention")

    def _abstract_eval(q, k, v, *, num_heads):
        del v, num_heads
        return core.ShapedArray(q.shape, q.dtype)

    prim.def_abstract_eval(_abstract_eval)

    def _lowering(ctx, q, k, v, *, num_heads):
        b, nh, n, hd = ctx.avals_in[0].shape
        opaque = f"{b} {num_heads} {n} {hd}".encode()
        out = mlir.custom_call(
            "fused_attention",
            result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
            operands=[q, k, v],
            backend_config=opaque,
            api_version=1,
        )
        return out.results

    mlir.register_lowering(prim, _lowering, platform="cuda")

    _primitive = prim
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

    # the primitive only has an abstract_eval + mlir lowering rule, no
    # eager "impl" rule -- binding it outside jax.jit raises
    # NotImplementedError ("Evaluation rule ... not implemented"), confirmed
    # directly from a real run. Wrapping in jit here means every call goes
    # through compilation (which does use the mlir lowering), whether this
    # function itself is called eagerly or from inside a larger jitted model.
    bound = jax.jit(lambda q, k, v: _primitive.bind(q, k, v, num_heads=num_heads))
    out_heads = bound(q_heads, k_heads, v_heads)

    return out_heads.transpose(0, 2, 1, 3).reshape(b, n, d)
