// Fused multi-head self-attention kernel.
//
// Implements exactly the algorithm verified in kernels/reference.py against
// the real model (model/vision_encoder.py's _attention): for each
// (batch, head), compute softmax(Q @ K^T / sqrt(head_dim)) @ V.
//
// WHY THIS IS "FUSED": a naive implementation computes the full [N, N] score
// matrix with one kernel, writes it to global memory (HBM), reads it back for
// the softmax, writes the softmax output, reads it again for the final
// weighted sum with V. That's 3 extra round trips to slow global memory for
// a matrix that's thrown away immediately after. This kernel instead loads
// K and V for one (batch, head) into fast on-chip shared memory ONCE, then
// has each thread compute one query row's full attention output (scores,
// softmax, weighted sum) without ever writing the [N, N] score matrix back
// to global memory -- it never leaves shared memory / registers. This is
// the same memory-traffic argument that motivates FlashAttention-style
// kernels in production transformer serving, just at a toy scale (N ~ 100).
//
// Shapes (fixed for this project's model -- see model/vision_encoder.py):
//   N (sequence length) = 101  (100 patches + 1 CLS token, at IMAGE_SIZE=40)
//   num_heads = 4
//   head_dim = 12              (EMBED_DIM=48 / num_heads)
//
// Launch config: one threadblock per (batch, head), one thread per query row.
// This only works because N=101 is small enough that K and V for one head
// (101 * 12 floats each = ~4.8KB) comfortably fit in shared memory (usually
// 48KB+ per block on any CUDA GPU from the last decade), and N <= max threads
// per block (1024), so "one thread per query row" is a valid launch shape
// here without needing to tile over N like a general-purpose kernel would.

#include <cuda_runtime.h>
#include <math.h>

#define MAX_N 128          // compile-time upper bound on sequence length for the shared-memory buffers
#define MAX_HEAD_DIM 16    // compile-time upper bound on head_dim

extern "C" __global__ void fused_attention_kernel(
    const float* __restrict__ q,  // [B, num_heads, N, head_dim], contiguous
    const float* __restrict__ k,  // same shape as q
    const float* __restrict__ v,  // same shape as q
    float* __restrict__ out,      // same shape as q
    int N,
    int head_dim
) {
    int batch_head_idx = blockIdx.x;  // one block per (batch, head) pair
    int row = threadIdx.x;            // one thread per query row

    if (row >= N) return;

    __shared__ float k_shared[MAX_N * MAX_HEAD_DIM];
    __shared__ float v_shared[MAX_N * MAX_HEAD_DIM];

    const float* k_base = k + (size_t)batch_head_idx * N * head_dim;
    const float* v_base = v + (size_t)batch_head_idx * N * head_dim;
    const float* q_base = q + (size_t)batch_head_idx * N * head_dim;
    float* out_base = out + (size_t)batch_head_idx * N * head_dim;

    // cooperatively load K and V for this (batch, head) into shared memory once;
    // every thread in the block reuses this instead of re-reading from global memory
    for (int idx = row; idx < N * head_dim; idx += blockDim.x) {
        k_shared[idx] = k_base[idx];
        v_shared[idx] = v_base[idx];
    }
    __syncthreads();

    float scale = rsqrtf((float)head_dim);

    // this thread's query row, copied into registers
    float q_row[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q_row[d] = q_base[row * head_dim + d];
    }

    // pass 1: compute all N scores for this query row, track the max (for
    // numerically stable softmax), keep scores in a register array -- never
    // written to global memory
    float scores[MAX_N];
    float max_score = -INFINITY;
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_shared[j * head_dim + d];
        }
        float s = dot * scale;
        scores[j] = s;
        max_score = fmaxf(max_score, s);
    }

    // pass 2: softmax (stable, subtract max) and the weighted sum with V,
    // fused into the same loop -- no separate kernel launch for softmax
    float sum_exp = 0.0f;
    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    for (int j = 0; j < N; j++) {
        float w = expf(scores[j] - max_score);
        sum_exp += w;
        for (int d = 0; d < head_dim; d++) {
            acc[d] += w * v_shared[j * head_dim + d];
        }
    }

    for (int d = 0; d < head_dim; d++) {
        out_base[row * head_dim + d] = acc[d] / sum_exp;
    }
}

// Host-side launch wrapper, called from custom_call.cpp.
extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
) {
    int num_blocks = batch_size * num_heads;
    int threads_per_block = N;  // one thread per query row, see comment above
    fused_attention_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        q, k, v, out, N, head_dim
    );
}
