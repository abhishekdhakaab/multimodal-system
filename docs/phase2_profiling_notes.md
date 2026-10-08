# Phase 2 Profiling Notes

## Setup
Batch size 64, CPU (M1), JIT-compiled JAX functions, 50 timed runs after a warmup call (to exclude JIT trace/compile time from the measurement).

## Top-level breakdown (full forward pass = 6.61 ms/batch)

| Component | Time | % of full forward pass |
|---|---|---|
| Vision encoder (ViT) | 5.86 ms | 88.7% |
| Lidar encoder (PointNet-style) | 0.40 ms | 6.0% |
| Fusion (cross-attention + head) | 0.50 ms | 7.5% |

The vision encoder dominates by a wide margin — unsurprising, it's the only component with transformer self-attention and the most layers.

## Drilling into the vision encoder (one block)

| Sub-op | Time |
|---|---|
| LayerNorm | 0.076 ms |
| **Self-attention (QKV proj + reshape-to-heads + scaled dot-product + softmax + output proj)** | **1.80 ms** |
| MLP (2-layer, ReLU) | 0.84 ms |
| Patchify (reshape/transpose, one-time at input) | 0.037 ms |

With 2 transformer blocks, self-attention alone accounts for roughly 3.6 ms of the encoder's 5.86 ms — attention is ~55% of the vision encoder's time and the single most expensive op-group in the entire model.

## Decision: Phase 3's fusion target

**The custom CUDA kernel will fuse the multi-head self-attention block**: QKV projection, head reshape, scaled dot-product + softmax, and the output projection, as one fused op instead of 5+ separate XLA ops. This is the standard, well-known fusion target in transformer inference (the same bottleneck that motivates FlashAttention-style kernels in production LLM serving) — so beyond fixing this specific model's bottleneck, it's a realistic, industry-relevant choice to talk about in an interview.

Note: on CPU, "fusing ops" mostly saves intermediate-tensor memory traffic and kernel-launch/dispatch overhead rather than exploiting GPU-specific shared memory — the real payoff of a hand-written fused kernel shows up on GPU, which is exactly why Phase 3 moves to CUDA.
