/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_cpu_engine.hpp"
#include "sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

inline std::vector<cl::sycl::device> get_sycl_devices(
        cl::sycl::info::device_type dev_type) {
#ifdef DNNL_SYCL_CUDA
    const int vendor_id
            = ((dev_type == cl::sycl::info::device_type::gpu) ? 0x10DE
                                                              : 0x8086);
#else
    const int vendor_id = 0x8086;
#endif
    auto devices = cl::sycl::device::get_devices(dev_type);
    auto gpu_backend = get_sycl_gpu_backend();
    devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                    [=](const cl::sycl::device &dev) {
                        auto _vendor_id = dev.get_info<
                                cl::sycl::info::device::vendor_id>();
                        if (_vendor_id != vendor_id) return true;

                        auto _dev_type = dev.get_info<
                                cl::sycl::info::device::device_type>();
                        if (_dev_type != dev_type) return true;

                        if (dev_type == cl::sycl::info::device_type::gpu) {
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

class sycl_engine_factory_t : public engine_factory_t {
public:
    sycl_engine_factory_t(engine_kind_t engine_kind)
        : engine_kind_(engine_kind) {
        assert(utils::one_of(engine_kind_, engine_kind::cpu, engine_kind::gpu));
    }

    virtual size_t count() const override {
        auto dev_type = (engine_kind_ == engine_kind::cpu)
                ? cl::sycl::info::device_type::cpu
                : cl::sycl::info::device_type::gpu;
        return get_sycl_devices(dev_type).size();
    }

    virtual status_t engine_create(
            engine_t **engine, size_t index) const override;

    status_t engine_create(engine_t **engine, const cl::sycl::device &dev,
            const cl::sycl::context &ctx) const;

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
