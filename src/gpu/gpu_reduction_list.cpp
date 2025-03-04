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
#include "gpu/intel/ocl/reduction/atomic_reduction.hpp"
#include "gpu/intel/ocl/reduction/combined_reduction.hpp"
#include "gpu/intel/ocl/reduction/ref_reduction.hpp"
#include "gpu/intel/ocl/reduction/reusable_ref_reduction.hpp"

#ifdef DNNL_DEV_MODE
#include "gpu/intel/jit/reduction.hpp"
#endif

#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_reduction.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_reduction.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_reduction.hpp"
#include "gpu/generic/sycl/simple_reduction.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_REDUCTION_P({
        GPU_INSTANCE_INTEL_DEVMODE(intel::jit::reduction_t)
        GPU_INSTANCE_INTEL(intel::ocl::atomic_reduction_t)
        GPU_INSTANCE_INTEL(intel::ocl::combined_reduction_t)
        GPU_INSTANCE_INTEL(intel::ocl::ref_reduction_t)
        GPU_INSTANCE_INTEL(intel::ocl::reusable_ref_reduction_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_reduction_t)
        GPU_INSTANCE_AMD(amd::miopen_reduction_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_reduction_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::simple_reduction_t)
        nullptr,
});
// clang-format on

} // namespace

const impl_list_item_t *get_reduction_impl_list(const reduction_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
