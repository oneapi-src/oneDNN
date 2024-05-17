/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "common/compiler_workarounds.hpp"

#include "gpu/gpu_impl_list.hpp"

#include "gpu/intel/ocl/micro_sdpa.hpp"
#include "gpu/intel/ocl/ref_sdpa.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = {
        GPU_INSTANCE_INTEL(intel::ocl::micro_sdpa_t)
        GPU_INSTANCE_INTEL_DEVMODE(intel::ocl::ref_sdpa_t)
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_sdpa_impl_list(const sdpa_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
