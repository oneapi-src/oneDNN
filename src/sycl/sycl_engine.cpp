/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "sycl/sycl_engine.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_engine_factory_t::engine_create(
        engine_t **engine, size_t index) const {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (engine_kind_ == engine_kind::cpu) return status::unimplemented;
#endif
    assert(index < count());

    auto dev_type = (engine_kind_ == engine_kind::cpu)
            ? ::sycl::info::device_type::cpu
            : ::sycl::info::device_type::gpu;
    auto devices = get_sycl_devices(dev_type);
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

status_t sycl_engine_factory_t::engine_create(engine_t **engine,
        const ::sycl::device &dev, const ::sycl::context &ctx,
        size_t index) const {
    // Validate device and context.
    if (!dev_ctx_consistency_check(dev, ctx)) return status::invalid_arguments;

#ifdef DNNL_SYCL_CUDA
    if (gpu::nvidia::is_nvidia_gpu(dev))
        return gpu::nvidia::cuda_engine_create(
                engine, engine_kind_, dev, ctx, index);
#endif

#ifdef DNNL_SYCL_HIP
    if (gpu::amd::is_amd_gpu(dev))
        return gpu::amd::hip_engine_create(
                engine, engine_kind_, dev, ctx, index);
#endif
    if (engine_kind_ == engine_kind::cpu && !dev.is_cpu() && !is_host(dev))
        return status::invalid_arguments;
    if (engine_kind_ == engine_kind::gpu && !dev.is_gpu())
        return status::invalid_arguments;

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    std::unique_ptr<sycl_engine_base_t, engine_deleter_t> sycl_engine(
            (engine_kind_ == engine_kind::cpu)
                    ? static_cast<sycl_engine_base_t *>(
                            new sycl_cpu_engine_t(dev, ctx, index))
                    : static_cast<sycl_engine_base_t *>(
                            new gpu::sycl::sycl_gpu_engine_t(dev, ctx, index)));
#else

    if (engine_kind_ == engine_kind::cpu) return status::unimplemented;

    std::unique_ptr<sycl_engine_base_t, engine_deleter_t> sycl_engine(
            static_cast<sycl_engine_base_t *>(
                    new gpu::sycl::sycl_gpu_engine_t(dev, ctx, index)));

#endif
    if (!sycl_engine) return status::out_of_memory;

    CHECK(sycl_engine->init());
    *engine = sycl_engine.release();

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
