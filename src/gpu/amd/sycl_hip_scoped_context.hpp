/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_AMD_SYCL_HIP_SCOPED_CONTEXT_HPP
#define GPU_AMD_SYCL_HIP_SCOPED_CONTEXT_HPP

#include <memory>
#include <thread>

#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

class hip_sycl_scoped_context_handler_t {
    hipCtx_t original_;
    hipDevice_t currentDevice_;
    bool need_to_recover_;

public:
    hip_sycl_scoped_context_handler_t(const sycl_hip_engine_t &);
    // Destruct the scope p_context placed_context_.
    ~hip_sycl_scoped_context_handler_t() noexcept(false);

    template <typename T, typename U>
    inline T memory(const compat::interop_handle &ih, U acc) {
        return compat::get_native_mem<T>(ih, acc);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
