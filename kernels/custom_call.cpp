// C++ glue between JAX's XLA FFI (custom-call mechanism) and the CUDA
// kernel in fused_attention.cu. This is the piece that lets a JAX-jitted
// function call our hand-written kernel as if it were a normal JAX op.
//
// CONFIRMED on first real Colab run (JAX 0.11.1): this implements XLA's
// older, simpler "ORIGINAL" custom-call ABI (void** buffers, opaque
// bytes) -- registered via `jax.ffi.register_ffi_target(..., api_version=1)`
// and called via `jax.ffi.ffi_call(..., custom_call_api_version=1,
// legacy_backend_config=<string>)` on the Python side (kernels/register.py).
// This was the right ABI choice on the first try -- the only things that
// needed fixing were the Python-side call pattern and module paths, not
// this file's fundamental approach.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
);

// buffers[0..2] are Q, K, V (device pointers), buffers[3] is the output
// (device pointer). `opaque` carries "B NUM_HEADS N HEAD_DIM" as a plain
// space-separated text string (built in register.py's fused_attention()) --
// not null-terminated per XLA's convention, so it's copied into a
// std::string using the explicit length before parsing, rather than
// treated as a C string directly.
extern "C" void FusedAttentionCustomCall(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    size_t opaque_len
) {
    std::string config(opaque, opaque_len);
    int batch_size, num_heads, seq_len, head_dim;
    std::sscanf(config.c_str(), "%d %d %d %d", &batch_size, &num_heads, &seq_len, &head_dim);

    const float* q = reinterpret_cast<const float*>(buffers[0]);
    const float* k = reinterpret_cast<const float*>(buffers[1]);
    const float* v = reinterpret_cast<const float*>(buffers[2]);
    float* out = reinterpret_cast<float*>(buffers[3]);

    launch_fused_attention(
        q, k, v, out,
        batch_size, num_heads, seq_len, head_dim,
        stream
    );
}
