/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "common/verbose_msg.hpp"

#include "xpu/sycl/compat.hpp"
#include "xpu/sycl/utils.hpp"

// XXX: Include this header for VERROR_ENGINE.
// TODO: Move VERROR_ENGINE and other similar macros to a separate file.
#include "common/engine.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/sycl/l0/utils.hpp"
#endif

// TODO: Refactor build system for NVIDIA and AMD parts to enable them properly
// to be able to include their utility headers here.
#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
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

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
// Do not include sycl_hip_utils.hpp because it's intended for use in
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
namespace xpu {
namespace sycl {

std::string to_string(backend_t backend) {
    switch (backend) {
        case backend_t::host: return "Host";
        case backend_t::level0: return "Level Zero";
        case backend_t::opencl: return "OpenCL";
        case backend_t::nvidia: return "Nvidia";
        case backend_t::amd: return "AMD";
        default: return "Unknown";
    }
}

std::string to_string(::sycl::info::device_type dev_type) {
    using namespace ::sycl::info;
    switch (dev_type) {
        case device_type::cpu: return "cpu";
        case device_type::gpu: return "gpu";
        case device_type::accelerator: return "accelerator";
        case device_type::custom: return "custom";
        case device_type::automatic: return "automatic";
        case device_type::host: return "host";
        case device_type::all: return "all";
        default: return "unknown";
    }
}

backend_t get_gpu_backend() {
    // Create default GPU device and query its backend (assumed as default)
    static backend_t default_backend = []() {
        const backend_t fallback = backend_t::opencl;

        const auto gpu_type = ::sycl::info::device_type::gpu;
        if (::sycl::device::get_devices(gpu_type).empty()) return fallback;

        ::sycl::device dev {compat::gpu_selector_v};
        const auto backend = get_backend(dev);

        return backend;
    }();

    return default_backend;
}

bool is_host(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::device_type>()
            == ::sycl::info::device_type::host;
}

bool is_host(const ::sycl::platform &plat) {
    auto devices = plat.get_devices();
    if (devices.size() != 1) return false;
    return is_host(devices[0]);
}

backend_t get_backend(const ::sycl::device &dev) {
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

bool is_intel_platform(const ::sycl::platform &plat) {
    std::string plat_name = plat.get_info<::sycl::info::platform::name>();
    return plat_name.find("Intel") != std::string::npos;
}

bool is_subdevice(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::partition_type_property>()
            != ::sycl::info::partition_property::no_partition;
}

::sycl::device get_root_device(const ::sycl::device &dev) {
    // Search for the top level device.
    auto parent_device = dev;
    while (is_subdevice(parent_device)) {
        parent_device
                = parent_device.get_info<::sycl::info::device::parent_device>();
    }
    return parent_device;
}

::sycl::device get_parent_device(const ::sycl::device &dev) {
    return dev.get_info<::sycl::info::device::parent_device>();
}

bool are_equal(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_be = get_backend(lhs);
    auto rhs_be = get_backend(rhs);
    if (lhs_be != rhs_be) return false;

    // Only one host device exists.
    if (lhs_be == backend_t::host) return true;

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL \
        || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (lhs_be == backend_t::opencl) {
        // Use wrapper objects to avoid memory leak.
        auto lhs_ocl_handle = compat::get_native<cl_device_id>(lhs);
        auto rhs_ocl_handle = compat::get_native<cl_device_id>(rhs);
        return lhs_ocl_handle == rhs_ocl_handle;
    }
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    if (lhs_be == backend_t::level0) {
        return gpu::intel::sycl::compare_ze_devices(lhs, rhs);
    }
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (lhs_be == backend_t::nvidia) {
        return gpu::nvidia::compare_cuda_devices(lhs, rhs);
    }
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
    if (lhs_be == backend_t::amd) {
        return gpu::amd::compare_hip_devices(lhs, rhs);
    }
#endif
    assert(!"not expected");
    return false;
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
    VERROR_ENGINE(dev_ctx_consistency_check(dev, ctx),
            status::invalid_arguments, VERBOSE_DEVICE_CTX_MISMATCH);

    // Check engine kind and device consistency.
    VERROR_ENGINE(
            !(eng_kind == engine_kind::cpu && !dev.is_cpu() && !is_host(dev)),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);
    VERROR_ENGINE(!(eng_kind == engine_kind::gpu && !dev.is_gpu()),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);

#if !defined(DNNL_SYCL_CUDA) && !defined(DNNL_SYCL_HIP)
    // Check that platform is an Intel platform.
    VERROR_ENGINE(!(!is_host(dev) && !is_intel_platform(dev.get_platform())),
            status::invalid_arguments, VERBOSE_INVALID_PLATFORM, "sycl",
            "intel",
            dev.get_platform()
                    .get_info<::sycl::info::platform::name>()
                    .c_str());
#endif
    return status::success;
}

static bool is_vendor_device(const ::sycl::device &dev, int vendor_id) {
    return (int)dev.get_info<::sycl::info::device::vendor_id>() == vendor_id;
}

bool is_intel_device(const ::sycl::device &dev) {
    const int intel_vendor_id = 0x8086;
    return is_vendor_device(dev, intel_vendor_id);
}

bool is_nvidia_gpu(const ::sycl::device &dev) {
    const int nvidia_vendor_id = 0x10DE;
    return dev.is_gpu() && is_vendor_device(dev, nvidia_vendor_id);
}

bool is_amd_gpu(const ::sycl::device &dev) {
    const int amd_vendor_id = 0x1002;
    return dev.is_gpu() && is_vendor_device(dev, amd_vendor_id);
}

std::vector<::sycl::device> get_devices(
        ::sycl::info::device_type dev_type, backend_t backend) {
    const uint32_t intel_vendor_id = 0x8086;
#ifdef DNNL_SYCL_CUDA
    const uint32_t vendor_id
            = ((dev_type == ::sycl::info::device_type::gpu) ? 0x10DE
                                                            : intel_vendor_id);
#elif defined(DNNL_SYCL_HIP)
    const uint32_t vendor_id
            = ((dev_type == ::sycl::info::device_type::gpu) ? 0x1002
                                                            : intel_vendor_id);
#else
    const uint32_t vendor_id = intel_vendor_id;
#endif
    auto gpu_backend
            = backend == backend_t::unknown ? get_gpu_backend() : backend;

    std::vector<::sycl::device> devices;
    auto platforms = ::sycl::platform::get_platforms();

    for (const auto &p : platforms) {
#if !defined(DNNL_SYCL_CUDA) && !defined(DNNL_SYCL_HIP)
        if (!is_host(p) && !is_intel_platform(p)) continue;
#endif
        auto p_devices = p.get_devices(dev_type);
        devices.insert(devices.end(), p_devices.begin(), p_devices.end());
    }

    devices.erase(std::remove_if(devices.begin(), devices.end(),
                          [=](const ::sycl::device &dev) {
                              auto _vendor_id = dev.get_info<
                                      ::sycl::info::device::vendor_id>();
                              if (_vendor_id != vendor_id) return true;

                              auto _dev_type = dev.get_info<
                                      ::sycl::info::device::device_type>();
                              if (_dev_type != dev_type) return true;

                              if (dev_type == ::sycl::info::device_type::gpu) {
                                  auto _backend = get_backend(dev);
                                  if (_backend == backend_t::unknown
                                          || _backend != gpu_backend)
                                      return true;
                              }

                              return false;
                          }),
            devices.end());
    return devices;
}

status_t get_device_index(size_t *index, const ::sycl::device &dev) {
    auto dev_type = dev.get_info<::sycl::info::device::device_type>();
    auto backend = get_backend(dev);
    auto devices = get_devices(dev_type, backend);

    // Find the top level device in the list
    auto it = std::find(devices.begin(), devices.end(), get_root_device(dev));
    if (it != devices.end()) {
        *index = it - devices.begin();
        return status::success;
    } else {
        *index = SIZE_MAX;
        // TODO: remove this work around once Level-Zero is fixed
        if (backend == backend_t::level0) return status::success;
        VERROR_ENGINE(false, status::invalid_arguments,
                VERBOSE_INVALID_ENGINE_IDX, SIZE_MAX,
                to_string(dev_type).c_str(), devices.size());
    }
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
