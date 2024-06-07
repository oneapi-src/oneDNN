/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

cuda_sycl_scoped_context_handler_t::cuda_sycl_scoped_context_handler_t(
        const nvidia::engine_t &engine)
    : need_to_recover_(false) {
    CUDA_EXECUTE_FUNC(cuCtxGetCurrent, &original_);
    auto desired = engine.get_underlying_context();
    currentDevice_ = engine.get_underlying_device();

    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        CUDA_EXECUTE_FUNC(cuCtxSetCurrent, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to
        // the same underlying CUDA primary context are destroyed. This
        // emulates the behaviour of the CUDA runtime api, and avoids costly
        // context switches. No action is required on this side of the if.
        need_to_recover_ = original_ != nullptr;
    }
}

cuda_sycl_scoped_context_handler_t::
        ~cuda_sycl_scoped_context_handler_t() noexcept(false) {
    // we need to release the placed_context_ since we set it from
    // ctx.get() retains the underlying context so we need to remove it
    CUDA_EXECUTE_FUNC(cuDevicePrimaryCtxRelease, currentDevice_);
    if (need_to_recover_) { CUDA_EXECUTE_FUNC(cuCtxSetCurrent, original_); }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
