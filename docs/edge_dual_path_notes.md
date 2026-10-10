# Edge Dual-Path Runtime

## What this is

The trained model (`model/checkpoints/model.pkl`) can run inference two
ways:

- **ARM path** (`edge/arm_path/infer.py`): plain JAX, CPU backend. This is
  the *real* edge path for this project — the M1 laptop itself is the ARM
  device, not a simulation of one.
- **CUDA path** (`edge/cuda_path/infer.py`): the vision encoder's core
  self-attention runs through the hand-written CUDA kernel
  (`kernels/fused_attention.cu`) instead of plain JAX ops. Everything else
  (patch embedding, MLP, lidar encoder, fusion) still runs as ordinary JAX.

Both paths share the exact same model code (`model/vision_encoder.py`'s
`forward(..., backend=...)` parameter selects which attention
implementation runs) and the exact same trained checkpoint — this isn't two
separate models, it's one model with a swappable inference backend for one
specific op.

## What's actually measured right now (on the M1)

```
[ARM path / M1 CPU] true=bed  predicted=bed  latency=0.451 ms/inference
[CUDA path] fused_attention.so not found -- requires a CUDA GPU (see scripts/colab_sync.md)
```

The ARM path is real and measured: 0.451ms for a single inference on the
M1's CPU.

## CUDA path status: kernel verified, JAX integration blocked (confirmed upstream bug, not a guess)

`edge/cuda_path/infer.py` calls the kernel through `kernels/register.py`,
which routes through JAX's XLA custom-call mechanism. That integration path
is confirmed blocked: real debugging across two environments (Colab T4,
then a full-root RunPod RTX 3090 — see `scripts/colab_sync.md` and
`scripts/runpod_sync.md`) traced the failure to JAX's own pip-distributed
CUDA PJRT plugin (`jax[cuda12]==0.4.34`) failing to fully initialize its
custom-call registry — our handler registers with zero error every time,
it's simply never reachable at execution. This is an upstream packaging
bug, not something fixable in this project's glue code, and not something
that will resolve by changing `register.py`/`custom_call.cpp` further.

So `python -m edge.compare_paths` cannot produce a real side-by-side number
through JAX right now — it would just repeat the same blocked call.

**What IS real**: `kernels/tests/standalone_cuda_test.cu` calls the exact
same compiled kernel directly, bypassing JAX/XLA entirely, and on the RunPod
RTX 3090 measured:
```
max abs diff (kernel vs CPU reference): 2.98e-08   (correct, fp32-noise level)
avg kernel latency: 65.59us  (batch=2, heads=4, N=101, head_dim=12, 200 iters)
```
That 65.6us is the real, measured cost of the fused-attention kernel itself
on real hardware — just not yet embeddable in the full model's JAX forward
pass until the upstream bug is worked around (e.g. by trying a jaxlib
version well outside the two already ruled out).

## Real full-model GPU vs. CPU latency (plain JAX, no custom kernel — unaffected by the blocker above)

This comparison doesn't touch the custom CUDA kernel or the broken
custom-call path at all — it's the exact same `model/full_model.forward`
plain-JAX code, run on two real backends, via
`benchmarks/gpu_latency_pruning_cascade.py`'s no-pruning baseline, batch
size 64, min-of-7-trials:

| Backend | Latency/batch (64) |
|---|---|
| M1 CPU (this laptop) | 6.6876 ms |
| RunPod RTX 3090 | 0.3771 ms |
| **Real GPU speedup** | **~17.7x** |

This is the real heterogeneous-hardware number this section originally
asked for — just via plain JAX ops on a real GPU, not the custom kernel,
since that integration path is the one confirmed blocked.

## Why this split is realistic, not just a workaround

In a real heterogeneous robot fleet, this is exactly the situation: cheaper
edge devices (ARM, ~Jetson-Nano-class, no discrete GPU) need a fallback
inference path, while beefier edge boxes or a nearby edge server with an
actual GPU can run the faster kernel-accelerated path. A fleet rollout
system needs to know which path to deploy to which node type — which is
exactly what Phase 6's Kubernetes control plane does next.
