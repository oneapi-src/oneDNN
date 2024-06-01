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

#ifndef GPU_INTEL_OCL_OCL_ENGINE_HPP
#define GPU_INTEL_OCL_OCL_ENGINE_HPP

#include "gpu/intel/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

class ocl_engine_factory_t : public engine_factory_t {
public:
    ocl_engine_factory_t(engine_kind_t engine_kind) {
        assert(engine_kind == engine_kind::gpu);
        MAYBE_UNUSED(engine_kind);
    }

    size_t count() const override {
        std::vector<cl_device_id> ocl_devices;
        status_t status
                = xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        if (status != status::success) return status;
        return ocl_devices.size();
    }

    status_t engine_create(
            impl::engine_t **engine, size_t index) const override {
#ifdef DNNL_WITH_SYCL
        gpu_error_not_expected() << "This interface is not for use with SYCL";
        return status::runtime_error;
#endif
        status_t status;
        std::vector<cl_device_id> ocl_devices;

        status = xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        VERROR_ENGINE(
                status == status::success, status, "no ocl devices found");

        VERROR_ENGINE(index < ocl_devices.size(), status::invalid_arguments,
                VERBOSE_INVALID_ENGINE_IDX, ocl_devices.size(), "ocl", index);

        auto *ocl_engine
                = new ocl_gpu_engine_t(ocl_devices[index], nullptr, index);
        if (!ocl_engine) return status::out_of_memory;

        status = ocl_engine->init();
        if (status != status::success) {
            ocl_engine->release();
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }

    status_t engine_create(impl::engine_t **engine, cl_device_id device,
            cl_context context, size_t index,
            const std::vector<uint8_t> &cache_blob = {}) {
        auto *ocl_engine = new ocl_gpu_engine_t(device, context, index);
        if (!ocl_engine) return status::out_of_memory;

        status_t status = ocl_engine->init(cache_blob);
        if (status != status::success) {
            ocl_engine->release();
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }
};
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_OCL_OCL_ENGINE_HPP
