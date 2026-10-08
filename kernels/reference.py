"""
Pure NumPy reference implementation of the exact algorithm fused_attention.cu
implements, written and verified BEFORE any CUDA code, so we know the math is
right before worrying about CUDA-specific details. This can (and should) run
on the M1 with no GPU -- it's what the correctness test on Colab compares the
real CUDA kernel's output against, one level removed from "trust the CUDA".

The algorithm (per batch item, per head): given Q [N, Dh], K [N, Dh], V [N, Dh]
for one head, compute softmax(Q @ K^T / sqrt(Dh)) @ V, but -- matching what
the CUDA kernel does -- without ever materializing the full [N, N] score
matrix for every head at once in memory; instead do it one query row at a
time, which is what the kernel's per-threadblock loop mirrors.
"""

import numpy as np


def attention_core_reference(q, k, v):
    """q, k, v: [N, Dh] for ONE batch item, ONE head. Returns [N, Dh]."""
    n, head_dim = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(head_dim)

    for i in range(n):  # one query row at a time, mirrors the CUDA kernel's per-row loop
        scores = (q[i : i + 1] @ k.T) * scale  # [1, N]
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / weights.sum()
        out[i] = (weights @ v)[0]

    return out


def multi_head_attention_reference(q, k, v, num_heads):
    """q, k, v: [B, N, D]. Returns [B, N, D]. Splits D into num_heads heads,
    runs attention_core_reference per (batch, head), reassembles."""
    b, n, d = q.shape
    head_dim = d // num_heads
    out = np.zeros_like(q)

    for bi in range(b):
        for h in range(num_heads):
            lo, hi = h * head_dim, (h + 1) * head_dim
            out[bi, :, lo:hi] = attention_core_reference(
                q[bi, :, lo:hi], k[bi, :, lo:hi], v[bi, :, lo:hi]
            )
    return out
