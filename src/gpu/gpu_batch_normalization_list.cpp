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
#include "gpu/intel/ocl/bnorm/gen9_batch_normalization.hpp"
#include "gpu/intel/ocl/bnorm/nhwc_batch_normalization.hpp"
#include "gpu/intel/ocl/bnorm/ref_batch_normalization.hpp"
#include "gpu/intel/ocl/bnorm/reusable_bnorm.hpp"
#include "gpu/intel/ocl/bnorm/simple_bnorm.hpp"

#ifdef DNNL_DEV_MODE
#include "gpu/intel/ocl/bnorm/nhwc_reusable.hpp"
#endif

#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/generic/sycl/ref_batch_normalization.hpp"
#include "gpu/nvidia/cudnn_batch_normalization.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_batch_normalization.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>>
        impl_list_map REG_BNORM_P({
    {{forward}, {
        GPU_INSTANCE_INTEL_DEVMODE(intel::ocl::nhwc_reusable_batch_normalization_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::nhwc_batch_normalization_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::gen9_batch_normalization_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::simple_batch_normalization_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::reusable_batch_normalization_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::ref_batch_normalization_fwd_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_batch_normalization_fwd_t)
        GPU_INSTANCE_AMD(amd::miopen_batch_normalization_fwd_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_batch_normalization_fwd_t)
        nullptr,
    }},
    {{backward}, REG_BWD_PK({
        GPU_INSTANCE_INTEL_DEVMODE(intel::ocl::nhwc_reusable_batch_normalization_bwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::nhwc_batch_normalization_bwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::gen9_batch_normalization_bwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::simple_batch_normalization_bwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::reusable_batch_normalization_bwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::ref_batch_normalization_bwd_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_batch_normalization_bwd_t)
        GPU_INSTANCE_AMD(amd::miopen_batch_normalization_bwd_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_batch_normalization_bwd_t)
        nullptr,
    })},
});
// clang-format on
} // namespace

const impl_list_item_t *get_batch_normalization_impl_list(
        const batch_normalization_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    const auto impl_list_it = impl_list_map.find({prop_kind});
    return impl_list_it != impl_list_map.cend() ? impl_list_it->second.data()
                                                : empty_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
