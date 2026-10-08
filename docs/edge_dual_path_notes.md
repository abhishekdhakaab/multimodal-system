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
M1's CPU. The CUDA path correctly reports that it can't run here instead of
pretending to have a number — that's the honest state until the kernel is
built and tested on Colab (Phase 3/4).

## What to do once Colab results come back

Re-run `python -m edge.compare_paths` on Colab (after building
`fused_attention.so` per `scripts/colab_sync.md`) to get the real head-to-
head: both paths will run, and the script reports the speedup ratio and
whether both paths agree on the prediction. Paste those real numbers back
into this file, replacing the "not available" line above — do not estimate
or guess what the CUDA path's latency "should" be.

## Why this split is realistic, not just a workaround

In a real heterogeneous robot fleet, this is exactly the situation: cheaper
edge devices (ARM, ~Jetson-Nano-class, no discrete GPU) need a fallback
inference path, while beefier edge boxes or a nearby edge server with an
actual GPU can run the faster kernel-accelerated path. A fleet rollout
system needs to know which path to deploy to which node type — which is
exactly what Phase 6's Kubernetes control plane does next.
