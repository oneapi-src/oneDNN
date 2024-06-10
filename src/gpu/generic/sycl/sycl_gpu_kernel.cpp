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

#include "common/stream.hpp"

#include "xpu/sycl/stream_impl.hpp"

#include "gpu/generic/sycl/sycl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t kernel_t::parallel_for(impl::stream_t &stream,
        const std::function<void(::sycl::handler &)> &cgf) const {
    auto *sycl_stream_impl
            = utils::downcast<xpu::sycl::stream_impl_t *>(stream.impl());
    auto &queue = *sycl_stream_impl->queue();
    auto &deps = sycl_stream_impl->sycl_ctx().get_sycl_deps().events;

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(deps);
        cgh.use_kernel_bundle(*kernel_bundle_);
        cgf(cgh);
    });

    deps = {event};
    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
