// Modern typed XLA FFI handler (CallFrame-based, api_version=4), exported
// as a plain C symbol for Python-side registration.
//
// This is the SAME handler that compiled and linked cleanly on Colab's
// default jaxlib 0.11.1, where it turned out to hit a registry isolation
// issue specific to that build's newer PJRT-plugin CUDA architecture
// (registered with zero errors through two different registration
// functions, but never reachable at execution -- see
// scripts/colab_sync.md for the full history). Retried here against a
// pinned, older jax/jaxlib (0.4.34), which predates that plugin split --
// confirmed the FFI header exists at this version too
// (`jax.extend.ffi.include_dir()` -> .../xla/ffi/api/ffi.h), and
// `jax.extend.ffi.ffi_call`'s kwargs always flow through this same
// typed-attribute mechanism (confirmed by reading its lowering source
// directly), so this is the intended way to invoke a custom kernel here
// too, not a step backward from the legacy ABI.

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace ffi = xla::ffi;

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v, float* out,
    int batch_size, int num_heads, int N, int head_dim,
    cudaStream_t stream
);

static ffi::Error FusedAttentionImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F32> q,
    ffi::Buffer<ffi::DataType::F32> k,
    ffi::Buffer<ffi::DataType::F32> v,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    int32_t num_heads
) {
    // q/k/v arrive as [B, num_heads, N, head_dim] (see fused_attention.cu's
    // shape comment -- register.py reshapes to this layout before calling)
    auto dims = q.dimensions();
    int batch_size = static_cast<int>(dims[0]);
    int n = static_cast<int>(dims[2]);
    int head_dim = static_cast<int>(dims[3]);

    launch_fused_attention(
        q.typed_data(), k.typed_data(), v.typed_data(), out->typed_data(),
        batch_size, num_heads, n, head_dim,
        stream
    );
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FusedAttentionHandler, FusedAttentionImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // k
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // v
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out
        .Attr<int32_t>("num_heads")
);
