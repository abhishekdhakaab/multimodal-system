# Running Phase 3/4 on RunPod (RTX 3090) — real results

This picks up after `scripts/colab_sync.md`'s 6-round debugging arc ended with
Colab producing `NOT_FOUND: No FFI handler registered`, hypothesized at the
time to be a Colab-sandboxing/plugin-isolation issue. **RunPod (full root,
RTX 3090) ruled that hypothesis out** — the identical symptom reproduced
there too, with hard evidence pointing at a real upstream bug, not our code.

## 1. What was tested, with evidence

On a clean `jax[cuda12]==0.4.34` + `jaxlib==0.4.34` install (confirmed via
`pip list`: `jax-cuda12-pjrt==0.4.34`, `jax-cuda12-plugin==0.4.34`, all
versions matching, no skew):

- **Legacy ABI (`api_version=1`)**: `xla_client.register_custom_call_target`
  succeeds with zero error for every platform string tried (`"gpu"`,
  `"CUDA"`, `"cuda"`, `"GPU"`). At execution time, always:
  `UNIMPLEMENTED: No registered implementation for custom call to
  fused_attention for platform CUDA`.
- **Typed-FFI ABI (`api_version=4`, `XLA_FFI_DEFINE_HANDLER_SYMBOL`)**:
  compiles and links cleanly (confirmed `nm -D` shows the symbol).
  `jax.extend.ffi.register_ffi_target` also succeeds with zero error for
  every platform string tried. Same execution-time error, every time.
- **The real finding**: triggering JAX's lazy CUDA backend discovery
  (first `jax.jit` call) prints `Jax plugin configuration error: Exception
  when calling jax_plugins.xla_cuda12.initialize()`, with a traceback
  showing **JAX's own plugin module** calling
  `xla_client.register_custom_call_handler(...)` for **its own built-in
  targets** — and that call itself raises:
  - `INVALID_ARGUMENT: Unsupported custom call target type for api_version=1`
    (seen when probing with api_version=1 registered first), or
  - `UNIMPLEMENTED: API version 4 is not supported by RegisterCustomCallTarget.
    Supported versions are 0 and 1.` (seen when probing with api_version=4).

  This error is swallowed as a non-fatal warning, leaving the CUDA PJRT
  plugin partially initialized. **Our custom-call target registers
  without error every time; it is simply never reachable, because the
  plugin's own self-registration of its built-in kernels fails first.**
  This is a genuine version-skew bug between the pip `jax`/`jax_plugins`
  Python glue and the compiled `jax-cuda12-plugin` native binary for this
  release, not a mistake in `custom_call.cpp` or `register.py`.

Conclusion: continuing to iterate on JAX-side registration was not going to
succeed on this (or likely any nearby) pip jaxlib release. Confirmed with
direct evidence on real hardware, not assumed.

## 2. Pivot: verify the kernel directly, no JAX involved

`kernels/tests/standalone_cuda_test.cu` calls `launch_fused_attention`
(the same function `custom_call.cpp` calls) directly from a plain C++/CUDA
program — no XLA, no custom-call registration, no JAX at all. It:
- generates the same `[B=2, N=101, num_heads=4, head_dim=12]` random inputs
  used in `kernels/tests/test_correctness.py` (seeded, reproducible)
- computes the reference attention (same algorithm as `kernels/reference.py`)
  on the CPU, in C++
- runs the real CUDA kernel on the GPU, compares outputs
- times 200 iterations with CUDA events for a real latency number

Build and run (RTX 3090 / sm_86; use sm_75 for a T4):
```
nvcc -arch=sm_86 -o standalone_cuda_test \
    kernels/fused_attention.cu kernels/tests/standalone_cuda_test.cu
./standalone_cuda_test
```

## 3. Real results (RunPod RTX 3090, 2026-10-10)

```
max abs diff (kernel vs CPU reference): 2.980232e-08
CORRECTNESS: PASS
avg kernel latency: 65.591682 us (over 200 iters, batch=2 heads=4 N=101 head_dim=12)
```

The diff (3e-8) is at fp32 rounding-noise level — the kernel's math is
correct, not just "close enough." This is the first real, verified
execution of the CUDA kernel on a GPU in this project.

## 4. What this does and doesn't prove

- **Proves**: the hand-written FlashAttention-style kernel
  (`fused_attention.cu`) is numerically correct and runs on real Ampere
  hardware with a real measured latency.
- **Does not prove**: that it's reachable from JAX/XLA as a drop-in op —
  that integration is blocked by the upstream bug above, not by anything
  fixable in this project's glue code. If revisited, the next thing worth
  trying is a jaxlib version further from 0.4.34 (older, e.g. 0.4.13, or
  a from-source build) rather than more Python-side registration changes.

## 5. Honest status for PLAN.md / final_report.md

Phase 3 (kernel execution): the kernel itself is built, executed, and
verified correct on a real GPU — but as a standalone CUDA program, not as
a JAX custom call. Report both facts plainly.
Phase 4 (benchmarking): one real latency number exists (65.6us/launch for
the project's actual shapes), from the standalone harness, not from a
JAX-vs-kernel A/B comparison (that comparison needs the JAX integration,
which is blocked as above).
