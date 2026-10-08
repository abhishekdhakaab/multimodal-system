// C++ glue between JAX's XLA FFI (custom-call mechanism) and the CUDA
// kernel in fused_attention.cu. This is the piece that lets a JAX-jitted
// function call our hand-written kernel as if it were a normal JAX op.
//
// IMPORTANT HONESTY NOTE (read before debugging on Colab): JAX's custom-call
// / FFI API has changed across versions (xla_client.register_custom_call_target
// in older JAX, jax.extend.ffi in newer JAX). This file targets the newer
// XLA FFI API (the "ffi" convention: an XLA_FFI_Handler-style entrypoint
// reading typed buffers from an XLA_FFI_CallFrame). It was written against
// the JAX/XLA FFI documentation but has NOT been run or compiled, since this
// machine has no CUDA toolchain. Expect to fix real compile errors on first
// build in Colab -- that's normal, not a sign the whole approach is wrong.
// If the exact FFI struct layout has moved on, the fallback is the older,
// more stable "opaque buffer" custom-call ABI (void** buffers, void* opaque,
// size_t opaque_len) registered via XLA_REGISTER_CUSTOM_CALL_TARGET -- also
// sketched below as CPU-independent fallback notes.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
);

// Metadata passed alongside the buffers: batch_size, num_heads, N, head_dim.
// Packed into the custom call's "opaque" bytes by the JAX-side wrapper
// (see register.py), since XLA custom calls don't carry shape info directly
// to simple buffer-based targets.
struct AttentionOpaque {
    int32_t batch_size;
    int32_t num_heads;
    int32_t seq_len;
    int32_t head_dim;
};

// Older/simpler XLA custom-call ABI: buffers[0..2] are Q, K, V (device
// pointers), buffers[3] is the output (device pointer). `opaque` carries the
// AttentionOpaque struct packed as bytes, `opaque_len` its size. This is the
// form registered with XLA_REGISTER_CUSTOM_CALL_TARGET in older/simpler JAX
// custom-call setups and is the more likely one to "just work" with less
// fighting the exact FFI struct version -- try this path first on Colab if
// the newer jax.extend.ffi path (register.py's primary path) hits API
// mismatches.
extern "C" void FusedAttentionCustomCall(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    size_t opaque_len
) {
    AttentionOpaque meta;
    std::memcpy(&meta, opaque, sizeof(AttentionOpaque));

    const float* q = reinterpret_cast<const float*>(buffers[0]);
    const float* k = reinterpret_cast<const float*>(buffers[1]);
    const float* v = reinterpret_cast<const float*>(buffers[2]);
    float* out = reinterpret_cast<float*>(buffers[3]);

    launch_fused_attention(
        q, k, v, out,
        meta.batch_size, meta.num_heads, meta.seq_len, meta.head_dim,
        stream
    );
}
