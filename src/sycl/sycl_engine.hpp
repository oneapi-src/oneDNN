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

#ifndef SYCL_ENGINE_FACTORY_HPP
#define SYCL_ENGINE_FACTORY_HPP

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <exception>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "gpu/sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_utils.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "sycl/sycl_cpu_engine.hpp"
#endif

namespace dnnl {
namespace impl {

#ifdef DNNL_SYCL_CUDA
// XXX: forward declarations to avoid cuda dependencies on sycl level.
namespace gpu {
namespace nvidia {

bool is_nvidia_gpu(const ::sycl::device &dev);

status_t cuda_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

} // namespace nvidia
} // namespace gpu
#endif

#ifdef DNNL_SYCL_HIP
// XXX: forward declarations to avoid cuda dependencies on sycl level.
namespace gpu {
namespace amd {

bool is_amd_gpu(const ::sycl::device &dev);

status_t hip_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

} // namespace amd
} // namespace gpu
#endif

namespace sycl {

inline std::vector<::sycl::device> get_sycl_devices(
        ::sycl::info::device_type dev_type,
        backend_t backend = backend_t::unknown) {
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
            = backend == backend_t::unknown ? get_sycl_gpu_backend() : backend;

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
                                  auto _backend = get_sycl_backend(dev);
                                  if (_backend == backend_t::unknown
                                          || _backend != gpu_backend)
                                      return true;
                              }

                              return false;
                          }),
            devices.end());
    return devices;
}

inline status_t get_sycl_device_index(
        size_t *index, const ::sycl::device &dev) {
    auto dev_type = dev.get_info<::sycl::info::device::device_type>();
    auto backend = get_sycl_backend(dev);
    auto devices = get_sycl_devices(dev_type, backend);

    // Find the top level device in the list
    auto it = std::find(
            devices.begin(), devices.end(), get_main_parent_device(dev));
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

class sycl_engine_factory_t : public engine_factory_t {
public:
    sycl_engine_factory_t(engine_kind_t engine_kind)
        : engine_kind_(engine_kind) {
        assert(utils::one_of(engine_kind_, engine_kind::cpu, engine_kind::gpu));
    }

    size_t count() const override {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
        if (engine_kind_ == engine_kind::cpu) return 0;
#endif
        auto dev_type = (engine_kind_ == engine_kind::cpu)
                ? ::sycl::info::device_type::cpu
                : ::sycl::info::device_type::gpu;
        return get_sycl_devices(dev_type).size();
    }

    status_t engine_create(engine_t **engine, size_t index) const override;

    status_t engine_create(engine_t **engine, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index) const;

private:
    engine_kind_t engine_kind_;
};

inline std::unique_ptr<sycl_engine_factory_t> get_engine_factory(
        engine_kind_t engine_kind) {
    return std::unique_ptr<sycl_engine_factory_t>(
            new sycl_engine_factory_t(engine_kind));
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
