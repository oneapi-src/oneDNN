/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#include "nvidia/sycl_cuda_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

cuda_sycl_scoped_context_handler_t::cuda_sycl_scoped_context_handler_t(
        const sycl_cuda_engine_t &engine)
    : need_to_recover_(false) {
    try {
        auto desired = engine.get_underlying_context();
        CUDA_EXECUTE_FUNC(cuCtxGetCurrent, &original_);

        if (original_ != desired) {
            // Sets the desired context as the active one for the thread
            CUDA_EXECUTE_FUNC(cuCtxSetCurrent, desired);
            // No context is installed and the suggested context is primary
            // This is the most common case. We can activate the context in the
            // thread and leave it there until all the PI context referring to
            // the same underlying CUDA primary context are destroyed. This
            // emulates the behaviour of the CUDA runtime api, and avoids costly
            // context switches. No action is required on this side of the if.
            need_to_recover_
                    = !(original_ == nullptr && engine.has_primary_context());
        }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
}

cuda_sycl_scoped_context_handler_t::
        ~cuda_sycl_scoped_context_handler_t() noexcept(false) {
    // we need to release the placed_context_ since we set it from
    // ctx.get() retains the underlying context so we need to remove it
    try {
        if (need_to_recover_) { CUDA_EXECUTE_FUNC(cuCtxSetCurrent, original_); }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
}

} // namespace cuda
} // namespace impl
} // namespace dnnl
