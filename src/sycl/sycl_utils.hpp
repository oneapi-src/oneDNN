/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include <vector>
#include <CL/sycl.hpp>

#if defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20220000)
#define DNNL_USE_SYCL121_API 1
#else
#define DNNL_USE_SYCL121_API 0
#endif
#elif defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION)
#if (__LIBSYCL_MAJOR_VERSION == 5 && __LIBSYCL_MINOR_VERSION < 4)
#define DNNL_USE_SYCL121_API 1
#else
#define DNNL_USE_SYCL121_API 0
#endif
#else
#error "Unsupported compiler"
#endif

namespace dnnl {
namespace impl {
namespace sycl {

using buffer_u8_t = ::sycl::buffer<uint8_t, 1>;

inline ::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::compute::nd_range_t &range) {
    auto *local_range = range.local_range();
    auto *global_range = range.global_range();

    auto sycl_global_range = ::sycl::range<3>(
            global_range[2], global_range[1], global_range[0]);

    if (!local_range) {
        assert(!"not expected");
        return ::sycl::nd_range<3>(
                sycl_global_range, ::sycl::range<3>(1, 1, 1));
    }

    auto sycl_local_range
            = ::sycl::range<3>(local_range[2], local_range[1], local_range[0]);
    return ::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

enum class backend_t { unknown, host, level0, opencl, nvidia, amd };

inline std::string to_string(backend_t backend) {
    switch (backend) {
        case backend_t::host: return "Host";
        case backend_t::level0: return "Level Zero";
        case backend_t::opencl: return "OpenCL";
        case backend_t::nvidia: return "Nvidia";
        case backend_t::amd: return "AMD";
        default: return "Unknown";
    }
}

backend_t get_sycl_gpu_backend();

inline bool is_host(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::device_type>()
            == ::sycl::info::device_type::host;
}

inline bool is_host(const ::sycl::platform &plat) {
    auto devices = plat.get_devices();
    if (devices.size() != 1) return false;
    return is_host(devices[0]);
}

inline backend_t get_sycl_backend(const ::sycl::device &dev) {
    if (is_host(dev)) return backend_t::host;

    auto plat = dev.get_platform();
    std::string plat_name = plat.get_info<::sycl::info::platform::name>();
    if (plat_name.find("OpenCL") != std::string::npos) return backend_t::opencl;
    if (plat_name.find("NVIDIA") != std::string::npos) return backend_t::nvidia;
    if (plat_name.find("AMD") != std::string::npos) return backend_t::amd;
    if (plat_name.find("Level-Zero") != std::string::npos)
        return backend_t::level0;

    return backend_t::unknown;
}

bool are_equal(const ::sycl::device &lhs, const ::sycl::device &rhs);
device_id_t sycl_device_id(const ::sycl::device &dev);

status_t check_device(engine_kind_t eng_kind, const ::sycl::device &dev,
        const ::sycl::context &ctx);

inline bool is_intel_device(const ::sycl::device &dev) {
    const int intel_vendor_id = 0x8086;
    auto vendor_id = dev.get_info<::sycl::info::device::vendor_id>();
    return vendor_id == intel_vendor_id;
}

inline bool is_intel_platform(const ::sycl::platform &plat) {
    std::string plat_name = plat.get_info<::sycl::info::platform::name>();
    return plat_name.find("Intel") != std::string::npos;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
