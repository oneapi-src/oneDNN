/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_GPU_PRIMITIVE_HPP
#define GPU_SYCL_SYCL_GPU_PRIMITIVE_HPP

#include "gpu/sycl/sycl_gpu_engine.hpp"
#include "sycl/sycl_stream.hpp"

#include "gpu/intel/compute/kernel.hpp"
#include "gpu/sycl/sycl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct sycl_gpu_primitive_t : public primitive_t {
    using primitive_t::primitive_t;

protected:
    status_t create_kernel(impl::engine_t *engine, ::sycl::kernel_id kid,
            intel::compute::kernel_t *kernel) {
        using namespace impl::sycl;
        auto ctx = utils::downcast<const xpu::sycl::engine_impl_t *>(
                engine->impl())
                           ->context();

        auto input_bundle
                = ::sycl::get_kernel_bundle<::sycl::bundle_state::input>(
                        ctx, {kid});
        auto exe_bundle = ::sycl::build(input_bundle);

        auto kernel_impl
                = std::make_shared<gpu::sycl::sycl_gpu_kernel_t>(exe_bundle);
        (*kernel) = intel::compute::kernel_t(std::move(kernel_impl));
        return status::success;
    }

    status_t parallel_for(const exec_ctx_t &ctx,
            const intel::compute::kernel_t &kernel,
            const std::function<void(::sycl::handler &)> &cgf) const {
        using namespace impl::sycl;

        const auto cvt_void2handler = [=](void *cgh) {
            ::sycl::handler &handler
                    = *(reinterpret_cast<::sycl::handler *>(cgh));
            cgf(handler);
        };

        intel::compute::compute_stream_t *compute_stream
                = utils::downcast<intel::compute::compute_stream_t *>(
                        ctx.stream());
        CHECK(kernel.parallel_for(*compute_stream, cvt_void2handler));
        return status::success;
    }
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
