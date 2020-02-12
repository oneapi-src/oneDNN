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

#include "sycl/sycl_engine.hpp"

#ifdef DNNL_SYCL_CUDA
#include "nvidia/sycl_cuda_engine.hpp"
#endif

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_engine_factory_t::engine_create(
        engine_t **engine, size_t index) const {
    assert(index < count());
    auto dev_type = (engine_kind_ == engine_kind::cpu)
            ? cl::sycl::info::device_type::cpu
            : cl::sycl::info::device_type::gpu;
    auto devices = get_sycl_devices(dev_type);
    auto &dev = devices[index];

    auto exception_handler = [](cl::sycl::exception_list eptr_list) {
        for (auto &eptr : eptr_list) {
            if (get_verbose()) {
                try {
                    std::rethrow_exception(eptr);
                } catch (const cl::sycl::exception &e) {
                    printf("dnnl_verbose,gpu,sycl_exception,%s\n", e.what());
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
status_t sycl_engine_factory_t::engine_create(engine_t **engine,
        const cl::sycl::device &dev, const cl::sycl::context &ctx) const {

    status_t status = status::success;
    // Validate device and context.
    auto ctx_devs = ctx.get_devices();
    auto it = std::find_if(ctx_devs.begin(), ctx_devs.end(),
            [&](const cl::sycl::device &ctx_dev) {
                return are_equal(ctx_dev, dev);
            });
    if (it == ctx_devs.end()) return status::invalid_arguments;
    //Nvidia code path
    constexpr int nvidia_vendor_id = 0x10DE;
    if (dev.is_gpu()
            && dev.get_info<cl::sycl::info::device::vendor_id>()
                    == nvidia_vendor_id) {
#ifdef DNNL_SYCL_CUDA
        status = cuda::check_device(engine_kind_);
        if (status != status::success) return status;
        std::unique_ptr<cuda::sycl_cuda_engine_t> cuda_engine(
                (new cuda::sycl_cuda_engine_t(dev, ctx)));
        if (!cuda_engine) return status::out_of_memory;

        status = cuda_engine->init();
        if (status != status::success) return status;

        *engine = cuda_engine.release();
#else
        return status::invalid_arguments;
#endif
    } else {

        if (engine_kind_ == engine_kind::cpu && !dev.is_cpu() && !dev.is_host())
            return status::invalid_arguments;
        if (engine_kind_ == engine_kind::gpu && !dev.is_gpu())
            return status::invalid_arguments;

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
    }

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
