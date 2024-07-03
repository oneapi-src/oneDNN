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

#include "gpu/amd/sycl_hip_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

// HIP context and functions that work with it have been deprecated.
// However, oneAPI DPC++ Compiler still uses HIP context underneath SYCL context
// therefore we have to use it as well.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

hip_sycl_scoped_context_handler_t::hip_sycl_scoped_context_handler_t(
        const amd::engine_t &engine)
    : need_to_recover_(false) {
    HIP_EXECUTE_FUNC(hipCtxGetCurrent, &original_);
    auto desired = engine.get_underlying_context();
    currentDevice_ = engine.get_underlying_device();

    if (original_ != desired) {
        HIP_EXECUTE_FUNC(hipCtxSetCurrent, desired);
        need_to_recover_ = original_ != nullptr;
    }
}

hip_sycl_scoped_context_handler_t::
        ~hip_sycl_scoped_context_handler_t() noexcept(false) {
    HIP_EXECUTE_FUNC(hipDevicePrimaryCtxRelease, currentDevice_);
    if (need_to_recover_) { HIP_EXECUTE_FUNC(hipCtxSetCurrent, original_); }
}

#pragma clang diagnostic pop

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
