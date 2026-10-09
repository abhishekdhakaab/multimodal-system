# Running Phase 3 on Google Colab

Everything in `kernels/` was originally written on the M1 (no CUDA available
there), and the sections below were written blind, before ever running on
real hardware. **Updated after the actual first Colab run** — the real
errors hit, and the real fixes, are recorded here instead of the original
guesses, so this doc now reflects what's actually true rather than what
seemed plausible beforehand.

## 1. Get the code onto Colab

```python
!git clone https://github.com/abhishekdhakaab/multimodal-system.git fenris
%cd fenris
!pip install -r requirements.txt
```

## 2. Confirm you have a GPU runtime

Runtime -> Change runtime type -> T4 GPU (free tier). Then verify:

```python
!nvidia-smi
```

## 3. Build the kernel

```python
!nvcc -shared -Xcompiler -fPIC -arch=sm_75 -std=c++17 \
    -I$(python3 -c "import jax.ffi; print(jax.ffi.include_dir())") \
    kernels/fused_attention.cu kernels/custom_call.cpp \
    -o kernels/fused_attention.so
```

The `-arch=sm_75` flag (T4's compute capability) and the FFI header
include path (`jax.ffi.include_dir()`, confirmed present at
`<that path>/xla/ffi/api/ffi.h` on Colab) were both needed for real —
not hypothetical, both come from the actual first successful build.

## 4. What actually went wrong, in order (for real, not hypothetical)

Three real issues were found and fixed, each discovered only by running
on actual hardware and pasting the real error back:

1. **`jax.extend.ffi` doesn't exist** (JAX 0.11.1 on this Colab build).
   The FFI API was promoted to a stable top-level `jax.ffi` module —
   same function names, shorter path. Fixed in `kernels/register.py`.

2. **`jax.ffi.ffi_call`'s real signature is two-stage**: it returns a
   callable, which you then call with the operands — not a single call
   taking operands directly. There's also no `opaque=` kwarg; found the
   real signature via `inspect.signature(jax.ffi.ffi_call)` on Colab
   rather than guessing again.

3. **This build's CUDA plugin doesn't support the legacy "ORIGINAL"
   custom-call ABI (api_version=1) at all** — confirmed because JAX's
   *own internal* plugin initialization hits the identical
   `"Unsupported custom call target type for api_version=1"` error,
   independent of our kernel. This meant the original simple
   `void** buffers + opaque bytes` C++ handler (`custom_call.cpp`)
   had to be rewritten entirely against the modern typed-FFI convention
   (`xla/ffi/api/ffi.h`, `XLA_FFI_DEFINE_HANDLER`/`XLA_FFI_REGISTER_HANDLER`,
   api_version=4) — not just a Python-side glue fix. That rewrite is
   what's in `custom_call.cpp` now; it was a best-effort rewrite against
   the documented API, itself unverified by compilation until you run
   it. If nvcc throws a real compile error on the macro/template usage,
   that's the next thing to paste back and fix — same pattern as the
   first three issues.

## 5. Run correctness tests

```python
!python -m pytest kernels/tests/test_correctness.py -v -s
```

The `-s` flag shows the printed max-diff numbers even on passing tests —
worth keeping for `docs/benchmark_results.md` later.

## 6. If it all passes

Move on to Phase 4 (benchmarking) — `benchmarks/run_latency.py` is next,
comparing stock JAX/XLA against this kernel. Report real numbers, including
if the kernel *doesn't* win on some configuration — that's useful data too,
not a failure. Also worth getting a real wall-clock latency number for
token pruning and the cascade here (see `docs/token_pruning_notes.md` and
`docs/speculative_cascade_notes.md` — both only have FLOP-based theoretical
estimates so far, deferred because the dev machine was too noisy to trust).

## 7. Bring results back

Whatever happens next (compiles clean, another real error, or real
benchmark numbers), paste it back — `PLAN.md`'s STATUS block and the
relevant `docs/` files get updated with what's actually true, not what
was guessed in advance.
