/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#if defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20230000)
#define DNNL_USE_SYCL121_API 1
#else
#define DNNL_USE_SYCL121_API 0
#endif
#elif defined(__LIBSYCL_MAJOR_VERSION)
#if (__LIBSYCL_MAJOR_VERSION < 6)
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

bool dev_ctx_consistency_check(
        const ::sycl::device &dev, const ::sycl::context &ctx);

inline bool is_intel_device(const ::sycl::device &dev) {
    const int intel_vendor_id = 0x8086;
    auto vendor_id = dev.get_info<::sycl::info::device::vendor_id>();
    return vendor_id == intel_vendor_id;
}

inline bool is_intel_platform(const ::sycl::platform &plat) {
    std::string plat_name = plat.get_info<::sycl::info::platform::name>();
    return plat_name.find("Intel") != std::string::npos;
}

inline bool is_subdevice(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::partition_type_property>()
            != ::sycl::info::partition_property::no_partition;
}

inline ::sycl::device get_main_parent_device(const ::sycl::device &dev) {
    // Search for the top level device.
    auto parent_device = dev;
    while (is_subdevice(parent_device)) {
        parent_device
                = parent_device.get_info<::sycl::info::device::parent_device>();
    }
    return parent_device;
}

inline ::sycl::device get_parent_device(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::parent_device>();
}

class sycl_engine_base_t;
status_t create_ocl_engine(
        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                *ocl_engine,
        const sycl_engine_base_t *engine);

status_t get_kernel_binary(
        const ::sycl::kernel &kernel, gpu::compute::binary_t &binary);

status_t create_ocl_engine(
        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                *ocl_engine,
        const sycl_engine_base_t *engine);

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
