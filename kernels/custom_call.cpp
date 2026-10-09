// C++ glue between JAX's XLA FFI (custom-call mechanism) and the CUDA
// kernel in fused_attention.cu. This is the piece that lets a JAX-jitted
// function call our hand-written kernel as if it were a normal JAX op.
//
// REVERTED to the legacy "ORIGINAL" custom-call ABI (void** buffers,
// opaque bytes) after a long real debugging arc on jaxlib 0.11.1 (Colab's
// default): that build's CUDA plugin doesn't support this ABI (confirmed:
// JAX's own internal plugin init hits the identical error registering
// its own handlers), so a typed-FFI rewrite was tried instead -- which
// compiled and "registered" without error, but was never actually
// reachable at execution time, on what looks like a PJRT-plugin registry
// isolation issue specific to that very recent build.
//
// Pinning an older, well-established jax/jaxlib (0.4.34) sidesteps the
// whole problem: confirmed on that version that `jax.ffi` doesn't exist
// at all (it's `jax.extend.ffi`), and the FFI C++ header bundling
// (`jax.ffi.include_dir()`) is itself a feature of the newer jaxlib --
// meaning the typed-FFI approach isn't even available here. This simple,
// long-established ABI is the right one for this version.

#include <cuda_runtime.h>
#include <cstdio>
#include <string>

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
);

// buffers[0..2] are Q, K, V (device pointers), buffers[3] is the output
// (device pointer). `opaque` carries "B NUM_HEADS N HEAD_DIM" as plain
// space-separated text (built in register.py's fused_attention()) --
// not null-terminated per XLA's convention, so it's copied into a
// std::string using the explicit length before parsing.
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
