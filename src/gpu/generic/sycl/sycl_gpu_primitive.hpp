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

#ifndef GPU_GENERIC_SYCL_SYCL_GPU_PRIMITIVE_HPP
#define GPU_GENERIC_SYCL_SYCL_GPU_PRIMITIVE_HPP

#include "common/primitive.hpp"

#include "xpu/sycl/memory_storage.hpp"

#include "gpu/generic/sycl/sycl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct primitive_t : public impl::primitive_t {
    using impl::primitive_t::primitive_t;

protected:
    status_t create_kernel(
            impl::engine_t *engine, ::sycl::kernel_id kid, kernel_t *kernel) {
        auto ctx = utils::downcast<const xpu::sycl::engine_impl_t *>(
                engine->impl())
                           ->context();
        auto input_bundle
                = ::sycl::get_kernel_bundle<::sycl::bundle_state::input>(
                        ctx, {kid});
        auto exe_bundle = ::sycl::build(input_bundle);

        (*kernel) = kernel_t(exe_bundle);
        return status::success;
    }

    status_t parallel_for(const exec_ctx_t &ctx, const kernel_t &kernel,
            const std::function<void(::sycl::handler &)> &cgf) const {
        return kernel.parallel_for(*ctx.stream(), cgf);
    }
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
