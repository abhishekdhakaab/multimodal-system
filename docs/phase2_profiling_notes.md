# Phase 2 Profiling Notes

**Update (after the ModelNet10 rework, see `docs/dataset_rework_notes.md` and `docs/fusion_bug_notes.md`):** numbers below are from the real-data model (40×40 images, 256-point lidar clouds, 10 classes), re-profiled after fixing the `NUM_CLASSES` bug. The original numbers (from the old synthetic 32×32/4-class setup) are kept struck through for history, since the bottleneck *finding* didn't change, only the magnitude.

## Setup
Batch size 64, CPU (M1), JIT-compiled JAX functions, 50 timed runs after a warmup call (to exclude JIT trace/compile time from the measurement).

## Top-level breakdown (full forward pass = 7.44 ms/batch)

| Component | Time | % of full forward pass |
|---|---|---|
| Vision encoder (ViT) | 6.49 ms | 87.3% |
| Lidar encoder (PointNet-style) | 1.38 ms | 18.5% |
| Fusion (cross-attention + head) | 0.75 ms | 10.0% |

(Old synthetic-data numbers, for reference: full pass 6.61ms, vision 5.86ms/88.7%, lidar 0.40ms/6.0%, fusion 0.50ms/7.5%.)

The vision encoder dominates by a wide margin — unsurprising, it's the only component with transformer self-attention and the most layers.

## Drilling into the vision encoder (one block)

| Sub-op | Time |
|---|---|
| LayerNorm | 0.170 ms |
| **Self-attention (QKV proj + reshape-to-heads + scaled dot-product + softmax + output proj)** | **3.157 ms** |
| MLP (2-layer, ReLU) | 0.882 ms |
| Patchify (reshape/transpose, one-time at input) | 0.042 ms |

(Old synthetic-data numbers: layernorm 0.076ms, attention 1.80ms, mlp 0.84ms, patchify 0.037ms, sequence length N=65.)

Sequence length grew from N=65 (old 32×32 images, 8×8=64 patches + 1 CLS) to N=101 (new 40×40 images, 10×10=100 patches + 1 CLS) when the image resolution was bumped during the overfitting fix — self-attention's cost grows roughly quadratically in N, which is exactly why it got proportionally more expensive (1.80ms → 3.16ms, more than the ~1.55x growth in N alone would suggest) and remains the clear single most expensive op-group in the entire model.

## Decision: Phase 3's fusion target

**The custom CUDA kernel will fuse the multi-head self-attention block**: QKV projection, head reshape, scaled dot-product + softmax, and the output projection, as one fused op instead of 5+ separate XLA ops. This is the standard, well-known fusion target in transformer inference (the same bottleneck that motivates FlashAttention-style kernels in production LLM serving) — so beyond fixing this specific model's bottleneck, it's a realistic, industry-relevant choice to talk about in an interview.

Note: on CPU, "fusing ops" mostly saves intermediate-tensor memory traffic and kernel-launch/dispatch overhead rather than exploiting GPU-specific shared memory — the real payoff of a hand-written fused kernel shows up on GPU, which is exactly why Phase 3 moves to CUDA.
