# Running Phase 3 on Google Colab

Everything in `kernels/` was written on the M1 (no CUDA available there) and
has **not been run**. This is expected per the project plan — Phase 3 is the
first phase that needs a real GPU. Here's the workflow to actually build and
test it.

## 1. Get the code onto Colab

Easiest path: push this repo to a (private, if you prefer) GitHub repo, then
in a Colab notebook:

```python
!git clone <your-repo-url> fenris
%cd fenris
!pip install -r requirements.txt
```

Alternative if you don't want to push to GitHub yet: zip the `fenris/`
folder and upload it directly in the Colab file browser, then `!unzip`.

## 2. Confirm you have a GPU runtime

Runtime -> Change runtime type -> T4 GPU (free tier). Then verify:

```python
!nvidia-smi
```

## 3. Build the kernel

```python
!nvcc -shared -Xcompiler -fPIC \
    kernels/fused_attention.cu kernels/custom_call.cpp \
    -o kernels/fused_attention.so
```

**Expect this to fail on the first try.** Likely issues, in rough order of
likelihood:
- CUDA architecture flag needed for the T4 (compute capability 7.5): add
  `-arch=sm_75` to the nvcc command above if it complains about missing
  kernel launch support.
- `cuda_runtime.h` not found: make sure the Colab runtime actually has the
  CUDA toolkit (`!which nvcc` should print a path; if not, `!apt install
  nvidia-cuda-toolkit` or use a Colab image that already has it).

## 4. Fix the JAX FFI registration

`kernels/register.py` targets `jax.extend.ffi`, written against the current
JAX FFI docs but not tested against Colab's installed JAX version. If
`register_ffi_target` or `ffi_call` errors with an API mismatch:

```python
import jax
print(jax.__version__)
help(jax.extend.ffi.ffi_call)  # check the actual signature on this install
```

The fallback path (older, more stable API) is noted in both `custom_call.cpp`
and `register.py`'s docstrings: `jax.lib.xla_client.register_custom_call_target`
+ a manual `jax.lax.custom_call` wrapper. If you hit this, ask Claude to
rewrite `register.py` against that API instead — the CUDA kernel itself
(`fused_attention.cu`) doesn't need to change, only the Python/C++ glue.

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
not a failure.

## 7. Bring results back

Whatever you learn on Colab (what had to change in `register.py`, the actual
benchmark numbers, anything that didn't work and why), paste it back here —
I'll update `PLAN.md`'s STATUS block and `docs/` with the real findings
rather than what I guessed while writing this blind.
