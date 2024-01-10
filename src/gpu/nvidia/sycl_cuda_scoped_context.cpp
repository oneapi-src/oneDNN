/*******************************************************************************
* Copyright 2020 Intel Corporation
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
        const sycl_cuda_engine_t &engine)
    : need_to_recover_(false) {
    try {
        CUDA_EXECUTE_FUNC(cuCtxGetCurrent, &original_);
        auto desired
                = engine.get_underlying_context(); // Getting the context also makes it active
        currentDevice = engine.get_underlying_device();

        if (original_ != desired) {
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
        if (need_to_recover_) {
            CUDA_EXECUTE_FUNC(cuDevicePrimaryCtxRelease, currentDevice);
            CUDA_EXECUTE_FUNC(cuCtxSetCurrent, original_);
        }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
