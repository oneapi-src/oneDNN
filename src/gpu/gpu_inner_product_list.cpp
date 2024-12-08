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
#include "gpu/intel/ocl/convolution_inner_product.hpp"
#include "gpu/intel/ocl/gemm_inner_product.hpp"
#include "gpu/intel/ocl/gemm_post_ops_inner_product.hpp"
#include "gpu/intel/ocl/ref_inner_product.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_conv_inner_product.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_gemm_inner_product.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_inner_product.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>>
        impl_list_map REG_IP_P({
    {{forward}, {
        GPU_INSTANCE_INTEL(intel::ocl::gemm_inner_product_fwd_t)
        GPU_INSTANCE_INTEL(intel::ocl::convolution_inner_product_fwd_t)
        GPU_INSTANCE_INTEL_REF(intel::ocl::ref_inner_product_fwd_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_gemm_inner_product_fwd_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_conv_inner_product_fwd_t)
        GPU_INSTANCE_AMD(amd::miopen_gemm_inner_product_fwd_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_inner_product_fwd_t)
        nullptr,
    }},
    {{backward}, REG_BWD_PK({
        GPU_INSTANCE_INTEL(intel::ocl::gemm_inner_product_bwd_data_t)
        GPU_INSTANCE_INTEL(intel::ocl::gemm_inner_product_bwd_weights_t)
        GPU_INSTANCE_INTEL_REF(intel::ocl::ref_inner_product_bwd_data_t)
        GPU_INSTANCE_INTEL_REF(intel::ocl::ref_inner_product_bwd_weights_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_gemm_inner_product_bwd_data_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_gemm_inner_product_bwd_weights_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_conv_inner_product_bwd_data_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_conv_inner_product_bwd_weights_t)
        GPU_INSTANCE_AMD(amd::miopen_gemm_inner_product_bwd_data_t)
        GPU_INSTANCE_AMD(amd::miopen_gemm_inner_product_bwd_weights_t)
        nullptr,
    })},
});
// clang-format on
} // namespace

const impl_list_item_t *get_inner_product_impl_list(
        const inner_product_desc_t *desc) {
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
