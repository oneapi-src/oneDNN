/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "sycl/sycl_utils.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_engine_base.hpp"

#include "sycl/level_zero_utils.hpp"

#ifdef DNNL_SYCL_CUDA
// Do not include sycl_cuda_utils.hpp because it's intended for use in
// gpu/nvidia directory only.

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
bool compare_cuda_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);
}
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

#ifdef DNNL_SYCL_HIP
// Do not include sycl_cuda_utils.hpp because it's intended for use in
// gpu/amd directory only.
namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {
bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);
}
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
namespace dnnl {
namespace impl {
namespace sycl {

backend_t get_sycl_gpu_backend() {
    // Create default GPU device and query its backend (assumed as default)
    static backend_t default_backend = []() {
        const backend_t fallback = backend_t::opencl;

        const auto gpu_type = ::sycl::info::device_type::gpu;
        if (::sycl::device::get_devices(gpu_type).empty()) return fallback;

        ::sycl::device dev {compat::gpu_selector_v};
        backend_t backend = get_sycl_backend(dev);

        return backend;
    }();

    return default_backend;
}

bool are_equal(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_be = get_sycl_backend(lhs);
    auto rhs_be = get_sycl_backend(rhs);
    if (lhs_be != rhs_be) return false;

    // Only one host device exists.
    if (lhs_be == backend_t::host) return true;

    if (lhs_be == backend_t::opencl) {
        // Use wrapper objects to avoid memory leak.
        auto lhs_ocl_handle = compat::get_native<cl_device_id>(lhs);
        auto rhs_ocl_handle = compat::get_native<cl_device_id>(rhs);
        return lhs_ocl_handle == rhs_ocl_handle;
    }

    if (lhs_be == backend_t::level0) { return compare_ze_devices(lhs, rhs); }

#ifdef DNNL_SYCL_CUDA
    if (lhs_be == backend_t::nvidia) {
        return gpu::nvidia::compare_cuda_devices(lhs, rhs);
    }
#endif

#ifdef DNNL_SYCL_HIP
    if (lhs_be == backend_t::amd) {
        return gpu::amd::compare_hip_devices(lhs, rhs);
    }
#endif
    assert(!"not expected");
    return false;
}

device_id_t sycl_device_id(const ::sycl::device &dev) {
    if (is_host(dev))
        return std::make_tuple(static_cast<int>(backend_t::host), 0, 0);

    device_id_t device_id
            = device_id_t {static_cast<int>(backend_t::unknown), 0, 0};
    switch (get_sycl_backend(dev)) {
        case backend_t::opencl: {
            auto ocl_device = gpu::ocl::make_ocl_wrapper(
                    compat::get_native<cl_device_id>(dev));
            device_id = std::make_tuple(static_cast<int>(backend_t::opencl),
                    reinterpret_cast<uint64_t>(ocl_device.get()), 0);
            break;
        }
        case backend_t::level0: {
            device_id = std::tuple_cat(
                    std::make_tuple(static_cast<int>(backend_t::level0)),
                    get_device_uuid(dev));
            break;
        }
        case backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
    assert(std::get<0>(device_id) != static_cast<int>(backend_t::unknown));
    return device_id;
}

bool dev_ctx_consistency_check(
        const ::sycl::device &dev, const ::sycl::context &ctx) {
    auto ctx_devs = ctx.get_devices();

    // Try to find the given device in the given context.
    auto it = std::find_if(ctx_devs.begin(), ctx_devs.end(),
            [&](const ::sycl::device &ctx_dev) {
                return are_equal(ctx_dev, dev);
            });
    // If found.
    if (it != ctx_devs.end()) return true;

    // If not found and the given device is not a sub-device.
    if (!is_subdevice(dev)) return false;

    // Try to find a parent device of the given sub-device in the given
    // context.
    while (is_subdevice(dev)) {
        auto parent_dev = get_parent_device(dev);
        it = std::find_if(ctx_devs.begin(), ctx_devs.end(),
                [&](const ::sycl::device &ctx_dev) {
                    return are_equal(ctx_dev, parent_dev);
                });
        // If found.
        if (it != ctx_devs.end()) return true;
    }

    return false;
}

status_t check_device(engine_kind_t eng_kind, const ::sycl::device &dev,
        const ::sycl::context &ctx) {
    // Check device and context consistency.
    if (!dev_ctx_consistency_check(dev, ctx)) return status::invalid_arguments;

    // Check engine kind and device consistency.
    if (eng_kind == engine_kind::cpu && !dev.is_cpu() && !is_host(dev))
        return status::invalid_arguments;
    if (eng_kind == engine_kind::gpu && !dev.is_gpu())
        return status::invalid_arguments;

#if !defined(DNNL_SYCL_CUDA) && !defined(DNNL_SYCL_HIP)
    // Check that platform is an Intel platform.
    if (!is_host(dev) && !is_intel_platform(dev.get_platform()))
        return status::invalid_arguments;
#endif

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
