// C++ glue between JAX's XLA custom-call mechanism and the CUDA kernel
// in fused_attention.cu.
//
// CONFIRMED, with hard evidence (not a guess) on pinned jax/jaxlib==0.4.34:
// `jaxlib.xla_client.register_custom_call_target` itself raised
// "UNIMPLEMENTED: API version 4 is not supported by RegisterCustomCallTarget.
// Supported versions are 0 and 1." when tested directly on the CPU
// platform. This jaxlib build's compiled backend genuinely does not
// implement the typed-FFI convention (api_version=4) -- the Python-level
// `ffi_call`/`register_ffi_target` wrappers expose that parameter, but
// the underlying native code doesn't support it yet at this version.
// api_version=1 (this file's ABI: plain void** buffers + opaque bytes)
// is the one confirmed supported here.
//
// `jax.extend.ffi.ffi_call` itself can't be used to invoke this ABI
// though -- its internal lowering hardcodes api_version=4 unconditionally
// with no public way to override it. kernels/register.py bypasses it
// entirely and builds the legacy custom-call HLO op directly via
// `jax.interpreters.mlir.custom_call`, which does expose api_version=1.

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
// space-separated text (built in register.py) -- not null-terminated
// per XLA's convention, so it's copied into a std::string using the
// explicit length before parsing.
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
