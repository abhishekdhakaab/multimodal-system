// Standalone CUDA/C++ correctness + timing harness for fused_attention.cu,
// bypassing JAX's XLA custom-call layer entirely.
//
// WHY THIS EXISTS: extensive debugging (see scripts/colab_sync.md and
// scripts/runpod_sync.md) confirmed, with direct evidence gathered on a
// RunPod RTX 3090 (not a Colab sandboxing issue), that the pip-distributed
// jax[cuda12]==0.4.34 package's own CUDA PJRT plugin fails to fully
// initialize regardless of which custom-call ABI is used (api_version=1
// or api_version=4) -- the failure happens inside JAX's own
// jax_plugins.xla_cuda12.initialize() code registering ITS OWN built-in
// handlers, not in this project's glue code. That means the kernel itself
// was never actually exercised on a GPU and verified -- this harness does
// that directly, calling launch_fused_attention() (declared in
// custom_call.cpp / fused_attention.cu) with no JAX in the loop at all.
//
// The reference computation mirrors kernels/reference.py's algorithm
// exactly (stable softmax(QK^T/sqrt(head_dim))V, per batch/head), just
// written in C++ instead of NumPy.
//
// Build (on a CUDA machine):
//   nvcc -arch=sm_86 -o standalone_cuda_test \
//       kernels/fused_attention.cu kernels/tests/standalone_cuda_test.cu
// (sm_86 for RTX 3090/A5000 Ampere; use sm_75 for a T4.)
//
// Run:
//   ./standalone_cuda_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
);

// [N, head_dim] in, [N, head_dim] out, one (batch, head) slice.
static void attention_core_reference(const float* q, const float* k, const float* v,
                                      float* out, int n, int head_dim) {
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(n);
    for (int i = 0; i < n; i++) {
        float max_score = -INFINITY;
        for (int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
            max_score = std::fmax(max_score, scores[j]);
        }
        float sum_exp = 0.0f;
        std::vector<float> weights(n);
        for (int j = 0; j < n; j++) {
            weights[j] = std::exp(scores[j] - max_score);
            sum_exp += weights[j];
        }
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) {
                acc += weights[j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = acc / sum_exp;
        }
    }
}

int main() {
    const int batch_size = 2;
    const int num_heads = 4;
    const int N = 101;
    const int head_dim = 12;
    const size_t total = (size_t)batch_size * num_heads * N * head_dim;

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    std::vector<float> h_q(total), h_k(total), h_v(total), h_out(total), h_ref(total);
    for (size_t i = 0; i < total; i++) {
        h_q[i] = dist(rng);
        h_k[i] = dist(rng);
        h_v[i] = dist(rng);
    }

    // CPU reference, per (batch, head) slice
    for (int bh = 0; bh < batch_size * num_heads; bh++) {
        size_t off = (size_t)bh * N * head_dim;
        attention_core_reference(h_q.data() + off, h_k.data() + off, h_v.data() + off,
                                  h_ref.data() + off, N, head_dim);
    }

    float *d_q, *d_k, *d_v, *d_out;
    size_t bytes = total * sizeof(float);
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

    // correctness run
    launch_fused_attention(d_q, d_k, d_v, d_out, batch_size, num_heads, N, head_dim, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (size_t i = 0; i < total; i++) {
        max_diff = std::fmax(max_diff, std::fabs(h_out[i] - h_ref[i]));
    }
    printf("max abs diff (kernel vs CPU reference): %e\n", max_diff);
    printf("CORRECTNESS: %s\n", max_diff < 1e-3f ? "PASS" : "FAIL");

    // timing: average over many launches, GPU-side event timing only
    const int warmup = 10, iters = 200;
    for (int i = 0; i < warmup; i++) {
        launch_fused_attention(d_q, d_k, d_v, d_out, batch_size, num_heads, N, head_dim, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launch_fused_attention(d_q, d_k, d_v, d_out, batch_size, num_heads, N, head_dim, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("avg kernel latency: %f us (over %d iters, batch=%d heads=%d N=%d head_dim=%d)\n",
           (ms * 1000.0f) / iters, iters, batch_size, num_heads, N, head_dim);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return max_diff < 1e-3f ? 0 : 1;
}
