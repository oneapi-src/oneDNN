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

#ifndef XPU_SYCL_VERBOSE_HPP
#define XPU_SYCL_VERBOSE_HPP

#include <cstdio>

#include "common/engine.hpp"

#include "xpu/sycl/engine_impl.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/engine.hpp"
#endif

#include "xpu/sycl/engine_factory.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

inline void print_verbose_header(engine_kind_t kind) {
    engine_factory_t factory(kind);
    auto s_engine_kind = (kind == engine_kind::cpu ? "cpu" : "gpu");

    verbose_printf("info,%s,engine,sycl %s device count:%zu %s\n",
            s_engine_kind, s_engine_kind, factory.count(),
            factory.count() == 0 ? "- no devices available." : "");

    for (size_t i = 0; i < factory.count(); ++i) {
        try {
            impl::engine_t *eng_ptr = nullptr;
            factory.engine_create(&eng_ptr, i);
            std::unique_ptr<impl::engine_t, engine_deleter_t> eng;
            eng.reset(eng_ptr);

            const xpu::sycl::engine_impl_t *engine_impl = eng
                    ? utils::downcast<const xpu::sycl::engine_impl_t *>(
                              eng->impl())
                    : nullptr;

            const auto &s_backend = engine_impl
                    ? to_string(engine_impl->backend())
                    : "unknown";
            const auto &s_name = engine_impl ? engine_impl->name() : "unknown";
            const auto &s_ver = engine_impl
                    ? engine_impl->runtime_version().str()
                    : "unknown";
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
            if (kind == engine_kind::gpu) {
                auto *dev_info = eng
                        ? utils::downcast<gpu::intel::engine_t *>(eng.get())
                                  ->device_info()
                        : nullptr;
                auto s_binary_kernels
                        = dev_info && dev_info->mayiuse_ngen_kernels()
                        ? "enabled"
                        : "disabled";
                auto s_product = dev_info ? dev_info->product_name()
                                           : std::string("unknown");

                verbose_printf(
                        "info,%s,engine,%zu,backend:%s,name:%s,product:%s,"
                        "hardware:%s,version_architecture:%u,version_release:%"
                        "u,driver_version:%s,binary_kernels:%s\n",
                        s_engine_kind, i, s_backend.c_str(), s_name.c_str(),
                        s_product.c_str(),
                        dev_info
                                ? gpu::intel::compute::to_string(
                                        dev_info->gpu_arch())
                                : "unknown",
                        dev_info ? dev_info->version_architecture() : 0,
                        dev_info ? dev_info->version_release() : 0,
                        s_ver.c_str(), s_binary_kernels);
                continue;
            }
#endif
            verbose_printf(
                    "info,%s,engine,%zu,backend:%s,name:%s,driver_version:%s\n",
                    s_engine_kind, i, s_backend.c_str(), s_name.c_str(),
                    s_ver.c_str());
        } catch (...) {
            VERROR(common, dpcpp, VERBOSE_INVALID_DEVICE_ENV, s_engine_kind, i);
        }
    }
}

inline void print_verbose_header() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(engine_kind::cpu);
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    print_verbose_header(engine_kind::gpu);
#endif
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
