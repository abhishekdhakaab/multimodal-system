# Running Phase 3 on Google Colab

Everything in `kernels/` was originally written on the M1 (no CUDA available
there). **Updated repeatedly after real Colab runs** — the real errors hit,
and the real fixes, are recorded here instead of the original blind guesses.

## The short version

Colab's **default** jaxlib (0.11.1 as of this writing) ships CUDA support
through a separate PJRT plugin that — after 6 real, confirmed debugging
rounds — turned out to have a custom-call registration/execution mismatch
that couldn't be resolved from the Python/C++ glue side (registration
"succeeds" with zero errors through two different registration functions,
but the handler is never actually reachable at execution: `NOT_FOUND: No
FFI handler registered`). **The fix was pinning an older, well-established
jax/jaxlib version (0.4.34)** rather than continuing to fight the newer
build's internals.

## 1. Get the code onto Colab (fresh runtime)

```python
!git clone https://github.com/abhishekdhakaab/multimodal-system.git fenris
%cd fenris
!pip install -r requirements.txt
!pip install -U "jax[cuda12]==0.4.34" "jaxlib==0.4.34"
```

Then **Runtime -> Restart session** (needed for the pinned native libraries
to actually load), and in a fresh cell:

```python
%cd fenris
import jax
print(jax.__version__)  # should print 0.4.34
```

## 2. Confirm you have a GPU runtime

Runtime -> Change runtime type -> T4 GPU (free tier). Then verify:

```python
!nvidia-smi
```

## 3. Build the kernel

```python
!nvcc -shared -Xcompiler -fPIC -arch=sm_75 \
    kernels/fused_attention.cu kernels/custom_call.cpp \
    -o kernels/fused_attention.so
```

No FFI C++ header or `-std=c++17` needed at this version — `custom_call.cpp`
uses the plain legacy ABI (`cuda_runtime.h` only), confirmed to be the right
choice here since `jax.ffi.include_dir()` (needed for the newer typed-FFI
approach) doesn't even exist at 0.4.34.

## 4. The full debugging history (6 real rounds, for the record)

1. **`jax.extend.ffi` doesn't exist on Colab's default 0.11.1** — promoted
   to a stable top-level `jax.ffi` module there.
2. **`jax.ffi.ffi_call` is two-stage** (returns a callable, call it with
   operands) with no `opaque=` kwarg — found via `inspect.signature`.
3. **Colab's default build's CUDA plugin doesn't support the legacy ABI
   (api_version=1) at all** — confirmed because JAX's own internal plugin
   init hits the identical error registering its own handlers.
4. Rewrote to the modern typed-FFI convention (`XLA_FFI_DEFINE_HANDLER` +
   `XLA_FFI_REGISTER_HANDLER`) — compiled, but **failed to link**:
   `GetXlaFfiApi` isn't available in pip-distributed jaxlib (that macro
   needs the full XLA runtime linked in).
5. Found `XLA_FFI_DEFINE_HANDLER_SYMBOL` by reading `xla/ffi/api/api.h`
   directly on Colab — exports a plain C symbol, exactly for this
   externally-loaded-plugin situation. Compiled AND linked.
6. Registration (via both `jax.ffi.register_ffi_target` and
   `jaxlib.xla_client.register_custom_call_target`) succeeded with zero
   errors, but the handler was never reachable at execution
   (`NOT_FOUND`) — looked like a PJRT-plugin registry isolation issue in
   that specific build, not fixable from our side. **Pinned jax/jaxlib
   0.4.34 instead** (step 1 above) and reverted to the simple legacy ABI,
   which doesn't need any of the typed-FFI machinery that version lacks
   anyway.

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
