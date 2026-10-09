// Modern typed XLA FFI handler (CallFrame-based, api_version=4).
//
// CONFIRMED on Colab (JAX 0.11.1): the older "ORIGINAL" custom-call ABI
// (api_version=1) is NOT supported by this build's CUDA plugin -- even
// JAX's OWN internal plugin initialization fails trying to register an
// api_version=1 handler for the "CUDA" platform ("Unsupported custom
// call target type for api_version=1"). That's an environment-level
// limitation of this jaxlib/CUDA-plugin build, not something fixable
// from the registration call -- confirmed by the fact that JAX's own
// internal code hits the identical error independent of our kernel.
// Switched to the modern typed FFI convention, which this build's CUDA
// plugin does support.
//
// Registration happens via static initialization: XLA_FFI_REGISTER_HANDLER
// below runs when this shared library is loaded (including via Python's
// ctypes.CDLL, since dlopen runs a library's static initializers
// regardless of which language loaded it) -- no explicit Python-side
// registration call needed anymore, see register.py.
//
// HONESTY NOTE: this is a best-effort rewrite against the documented
// xla/ffi/api/ffi.h C++ API (confirmed present at that exact path on
// Colab). The exact macro/template signatures were not verified by
// compiling before this was written -- if nvcc reports a real compile
// error here, that's the next thing to fix, the same way the Python-side
// API mismatches were found and fixed one real error at a time.

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

XLA_FFI_DEFINE_HANDLER(
    kFusedAttentionHandler, FusedAttentionImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // q
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // k
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // v
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // out
        .Attr<int32_t>("num_heads")
);

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::GetXlaFfiApi(), "fused_attention", "CUDA", kFusedAttentionHandler
);
