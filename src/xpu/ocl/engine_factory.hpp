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

#ifndef XPU_OCL_ENGINE_FACTORY_HPP
#define XPU_OCL_ENGINE_FACTORY_HPP

#include <cassert>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/utils.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/ocl/engine.hpp"
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

class engine_factory_t : public impl::engine_factory_t {
public:
    engine_factory_t(engine_kind_t engine_kind) {
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
        assert(!"This interface is not for use with SYCL");
        return status::runtime_error;
#else
        status_t status;
        std::vector<cl_device_id> ocl_devices;

        status = xpu::ocl::get_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        VERROR_ENGINE(status == status::success, status,
                VERBOSE_INVALID_ENGINE_KIND, "opencl", "gpu");

        VERROR_ENGINE(ocl_devices.size() > 0, status::invalid_arguments,
                "opencl gpu devices queried but not found");

        VERROR_ENGINE(index < ocl_devices.size(), status::invalid_arguments,
                VERBOSE_INVALID_ENGINE_IDX, ocl_devices.size(), "ocl", index);

        return engine_create(engine, ocl_devices[index], nullptr, index);
#endif
    }

    status_t engine_create(impl::engine_t **engine, cl_device_id device,
            cl_context context, size_t index,
            const std::vector<uint8_t> &cache_blob = {}) const {

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
        return gpu::intel::ocl::engine_create(
                engine, engine_kind::gpu, device, context, index, cache_blob);
#else
        return status::runtime_error;
#endif
    }
};
} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
