#include "sycl_cuda_compat.hpp"

namespace dnnl {
namespace impl {
namespace sycl {
namespace compat {

template <>
CUcontext get_native(const ::sycl::device &device) {
    CUdevice nativeDevice
            = ::sycl::get_native<::sycl::backend::ext_oneapi_cuda>(device);
    CUcontext nativeContext;
    if (cuDevicePrimaryCtxRetain(&nativeContext, nativeDevice)
            != CUDA_SUCCESS) {
        throw std::runtime_error("Could not create a native context");
    }
    return nativeContext;
}

} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl
