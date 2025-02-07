/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef XPU_OCL_VERBOSE_HPP
#define XPU_OCL_VERBOSE_HPP

#include <cstdio>

#include "xpu/ocl/engine_factory.hpp"
#include "xpu/ocl/engine_impl.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/compute/device_info.hpp"
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

inline void print_verbose_header() {
    xpu::ocl::engine_factory_t factory(engine_kind::gpu);

    verbose_printf("info,gpu,engine,opencl device count:%zu %s\n",
            factory.count(),
            factory.count() == 0 ? "- no devices available." : "");

    for (size_t i = 0; i < factory.count(); ++i) {
        impl::engine_t *eng_ptr = nullptr;
        status_t status = factory.engine_create(&eng_ptr, i);
        if (status != status::success) {
            VERROR(common, ocl, VERBOSE_INVALID_DEVICE_ENV,
                    dnnl_engine_kind2str(engine_kind::gpu), i);
            continue;
        }

        const auto *engine_impl
                = utils::downcast<const xpu::ocl::engine_impl_t *>(
                        eng_ptr->impl());
        const auto &s_name = engine_impl->name();
        auto s_ver = engine_impl->runtime_version().str();

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
        auto *compute_engine
                = utils::downcast<gpu::intel::compute::compute_engine_t *>(
                        eng_ptr);
        auto *dev_info = compute_engine->device_info();
        verbose_printf(
                "info,gpu,engine,%d,name:%s,driver_version:%s,binary_kernels:%"
                "s\n",
                (int)i, s_name.c_str(), s_ver.c_str(),
                dev_info->mayiuse_ngen_kernels() ? "enabled" : "disabled");
#else
        verbose_printf("info,gpu,engine,%d,name:%s,driver_version:%s\n", (int)i,
                s_name.c_str(), s_ver.c_str());
#endif
        eng_ptr->release();
    }
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
