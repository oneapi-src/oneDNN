/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <cstdio>
#include <exception>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_cpu_engine.hpp"
#include "sycl/sycl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

inline std::vector<cl::sycl::device> get_intel_sycl_devices(
        cl::sycl::info::device_type dev_type) {
    const int intel_vendor_id = 0x8086;
    auto devices = cl::sycl::device::get_devices(dev_type);
    devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                    [=](const cl::sycl::device &dev) {
                        return dev.get_info<cl::sycl::info::device::vendor_id>()
                                != intel_vendor_id;
                    }),
            devices.end());
    return devices;
}

class sycl_engine_factory_t : public engine_factory_t {
public:
    sycl_engine_factory_t(engine_kind_t engine_kind)
        : engine_kind_(engine_kind) {}

    virtual size_t count() const override {
        auto dev_type = (engine_kind_ == engine_kind::cpu)
                ? cl::sycl::info::device_type::cpu
                : cl::sycl::info::device_type::gpu;
        return get_intel_sycl_devices(dev_type).size();
    }

    virtual status_t engine_create(
            engine_t **engine, size_t index) const override {
        assert(index < count());
        auto dev_type = (engine_kind_ == engine_kind::cpu)
                ? cl::sycl::info::device_type::cpu
                : cl::sycl::info::device_type::gpu;
        auto devices = get_intel_sycl_devices(dev_type);
        auto &dev = devices[index];

        auto exception_handler = [](cl::sycl::exception_list eptr_list) {
            for (auto &eptr : eptr_list) {
                if (get_verbose()) {
                    try {
                        std::rethrow_exception(eptr);
                    } catch (const cl::sycl::exception &e) {
                        printf("dnnl_verbose,gpu,sycl_exception,%s\n",
                                e.what());
                    }
                } else {
                    std::rethrow_exception(eptr);
                }
            }
        };

        // XXX: we could use the platform to construct the context to cover
        // more devices. However in this case SYCL runtime may build a SYCL
        // kernel for all devices from the context (e.g. build both CPU and
        // GPU). This doesn't work for the CPU thunk kernel which works on CPU
        // only because it calls a native CPU function.
        cl::sycl::context ctx(dev, exception_handler);
        return engine_create(engine, dev, ctx);
    }

    status_t engine_create(engine_t **engine, const cl::sycl::device &dev,
            const cl::sycl::context &ctx) const {

        status_t status = status::success;

        // CPU and GPU devices work through OpenCL, so we can extract their
        // OpenCL device/context and validate them.
        if (dev.is_cpu() || dev.is_gpu()) {
            auto ocl_dev = ocl::ocl_utils::make_ocl_wrapper(dev.get());
            auto ocl_ctx = ocl::ocl_utils::make_ocl_wrapper(ctx.get());
            status = ocl::ocl_utils::check_device(
                    engine_kind_, ocl_dev, ocl_ctx);
            if (status != status::success) { return status; }
        }

        assert(utils::one_of(engine_kind_, engine_kind::cpu, engine_kind::gpu));

        std::unique_ptr<sycl_engine_base_t> sycl_engine(
                (engine_kind_ == engine_kind::cpu)
                        ? static_cast<sycl_engine_base_t *>(
                                new sycl_cpu_engine_t(dev, ctx))
                        : static_cast<sycl_engine_base_t *>(
                                new sycl_gpu_engine_t(dev, ctx)));
        if (!sycl_engine) return status::out_of_memory;

        status = sycl_engine->init();
        if (status != status::success) return status;

        *engine = sycl_engine.release();
        return status::success;
    }

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
