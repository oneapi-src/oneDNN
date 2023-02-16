/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/sycl/sycl_gpu_kernel.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

status_t sycl_gpu_kernel_t::parallel_for(
        stream_t &stream, const std::function<void(void *)> &cgf) {
    auto *sycl_stream = utils::downcast<impl::sycl::sycl_stream_t *>(&stream);
    auto &queue = sycl_stream->queue();
    auto &deps = sycl_stream->sycl_ctx().get_sycl_deps().events;

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(deps);
        cgh.use_kernel_bundle(*kernel_bundle_);
        cgf(reinterpret_cast<void *>(&cgh));
    });

    deps = {event};
    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
