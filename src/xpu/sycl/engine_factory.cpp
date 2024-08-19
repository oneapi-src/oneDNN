/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "xpu/sycl/engine_factory.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

status_t engine_factory_t::engine_create(
        engine_t **engine, size_t index) const {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    VERROR_ENGINE(engine_kind_ != engine_kind::cpu, status::unimplemented,
            VERBOSE_BAD_ENGINE_KIND);
#endif
    assert(index < count());

    auto dev_type = (engine_kind_ == engine_kind::cpu)
            ? ::sycl::info::device_type::cpu
            : ::sycl::info::device_type::gpu;
    auto devices = xpu::sycl::get_devices(dev_type);
    auto &dev = devices[index];

    auto exception_handler = [](const ::sycl::exception_list &eptr_list) {
        for (auto &eptr : eptr_list) {
            if (get_verbose(verbose_t::error)) {
                try {
                    std::rethrow_exception(eptr);
                } catch (const ::sycl::exception &e) {
                    VERROR(common, sycl, "%s", e.what());
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
    ::sycl::context ctx(dev, exception_handler);
    return engine_create(engine, dev, ctx, index);
}

status_t engine_factory_t::engine_create(engine_t **engine,
        const ::sycl::device &dev, const ::sycl::context &ctx,
        size_t index) const {
    // Validate device and context.
    VERROR_ENGINE(xpu::sycl::dev_ctx_consistency_check(dev, ctx),
            status::invalid_arguments, VERBOSE_DEVICE_CTX_MISMATCH);

#if DNNL_GPU_VENDOR == DNNL_VENDOR_GENERIC
    if (dev.is_gpu())
        return gpu::generic::sycl::engine_create(
                engine, engine_kind_, dev, ctx, index);
#endif

#ifdef DNNL_SYCL_CUDA
    if (xpu::sycl::is_nvidia_gpu(dev))
        return gpu::nvidia::engine_create(
                engine, engine_kind_, dev, ctx, index);
#endif

#ifdef DNNL_SYCL_HIP
    if (xpu::sycl::is_amd_gpu(dev))
        return gpu::amd::engine_create(engine, engine_kind_, dev, ctx, index);
#endif
    VERROR_ENGINE(!(engine_kind_ == engine_kind::cpu && !dev.is_cpu()
                          && !xpu::sycl::is_host(dev)),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);
    VERROR_ENGINE(!(engine_kind_ == engine_kind::gpu && !dev.is_gpu()),
            status::invalid_arguments, VERBOSE_BAD_ENGINE_KIND);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (engine_kind_ == engine_kind::cpu) {
        return cpu::sycl::engine_create(engine, dev, ctx, index);
    }

#else
    VERROR_ENGINE(engine_kind_ != engine_kind::cpu, status::unimplemented,
            VERBOSE_BAD_ENGINE_KIND);
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    if (xpu::sycl::is_intel_device(dev))
        return gpu::intel::sycl::engine_create(
                engine, engine_kind_, dev, ctx, index);
#endif
    return status::runtime_error;
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
