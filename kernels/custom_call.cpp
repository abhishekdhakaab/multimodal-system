// Modern typed XLA FFI handler (CallFrame-based, api_version=4), exported
// as a plain C symbol for Python-side registration.
//
// CONFIRMED on Colab (JAX 0.11.1), in order:
// 1. The older "ORIGINAL" custom-call ABI (api_version=1) is NOT
//    supported by this build's CUDA plugin at all -- even JAX's OWN
//    internal plugin initialization fails trying to register an
//    api_version=1 handler for "CUDA" ("Unsupported custom call target
//    type for api_version=1"). Environment-level limitation, not fixable
//    from the registration call.
// 2. `XLA_FFI_REGISTER_HANDLER` (static self-registration via
//    xla::ffi::GetXlaFfiApi()) doesn't link: `GetXlaFfiApi` isn't
//    available in the pip-distributed jaxlib headers/libs -- that macro
//    is meant for code compiled INTO XLA's own runtime, not an
//    externally-loaded plugin like this one.
// 3. The header itself documents the right tool for exactly this case:
//    `XLA_FFI_DEFINE_HANDLER_SYMBOL` ("for users who want to export XLA
//    FFI handler from a shared library as a C function symbol") --
//    confirmed by reading xla/ffi/api/api.h directly on Colab rather
//    than guessing a third time. This exports a plain `extern "C"`
//    function; register.py grabs it via ctypes and registers it from
//    Python with `jax.ffi.register_ffi_target(..., api_version=4)`,
//    no static self-registration needed.

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
