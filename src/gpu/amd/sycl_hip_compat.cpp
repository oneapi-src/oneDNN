#include "sycl_hip_compat.hpp"

namespace dnnl {
namespace impl {
namespace sycl {
namespace compat {

template <>
HIPcontext get_native(const ::sycl::device &device) {
    HIPdevice nativeDevice = ::sycl::get_native<::sycl::backend::ext_oneapi_hip>(device));
    HIPcontext nativeContext;
    if (hipDevicePrimaryCtxRetain(&nativeContext, nativeDevice) != hipSuccess) {
        throw std::runtime_error("Could not create a native context");
    }
    return nativeContext;
}
} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl
