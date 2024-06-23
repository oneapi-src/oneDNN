/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/ocl/ref_shuffle.hpp"
#include "gpu/intel/ocl/shuffle_by_reorder.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/generic/sycl/ref_shuffle.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_SHUFFLE_P({
        GPU_INSTANCE_INTEL(intel::ocl::shuffle_by_reorder_t)
        GPU_INSTANCE_INTEL(intel::ocl::ref_shuffle_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_shuffle_t)
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_shuffle_impl_list(const shuffle_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
