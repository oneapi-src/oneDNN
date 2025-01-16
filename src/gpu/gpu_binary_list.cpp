/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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
#include "gpu/intel/ocl/gen9_binary.hpp"
#include "gpu/intel/ocl/multi_po_reorder_binary.hpp"
#include "gpu/intel/ocl/simple_binary.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_binary.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_binary.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_binary.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_BINARY_P({
        GPU_INSTANCE_INTEL(intel::ocl::multi_po_reorder_binary_t)
        GPU_INSTANCE_INTEL(intel::ocl::gen9_binary_t)
        GPU_INSTANCE_INTEL(intel::ocl::simple_binary_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_binary_t)
        GPU_INSTANCE_AMD(amd::miopen_binary_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_binary_t)
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_binary_impl_list(const binary_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
