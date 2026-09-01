/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/primitive_attr.hpp"

#include "common/c_types_map.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/utils.hpp"

#include "gpu/intel/gemm/jit.hpp"
#include "gpu/intel/matmul/gemm.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::intel;

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_set_kernel_override(
        primitive_attr_t *attr, const char *kernel) {
    if (utils::any_null(attr)) return status::invalid_arguments;

    // Preserve any GRF/DPAS setting already carried by the gpu attribute.
    int grf_per_thread = 0;
    bool use_dpas = false;
    if (attr->gpu_attr_) {
        auto *cur = utils::downcast<gpu_primitive_attr_t *>(
                attr->gpu_attr_.get());
        grf_per_thread = cur->grf_per_thread();
        use_dpas = cur->use_dpas();
    }
    return attr->set_gpu_attr(gpu_primitive_attr_t(
            grf_per_thread, use_dpas, kernel ? kernel : ""));
}

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_override(
        const primitive_attr_t *attr, const char **kernel) {
    if (utils::any_null(attr, kernel)) return status::invalid_arguments;

    *kernel = "";
    if (attr->gpu_attr_) {
        auto *cur = utils::downcast<const gpu_primitive_attr_t *>(
                attr->gpu_attr_.get());
        *kernel = cur->kernel_override().c_str();
    }
    return status::success;
}

// Deployed (post-preflight/modifyStrategy) gemm strategy string for the kernel
// the primitive descriptor selected -- the catalog default when no override is
// set. "" if the chosen impl is not jit:gemm. Points into the pd; valid for its
// lifetime.
extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_gemm_kernel(
        const_dnnl_primitive_desc_t pd, const char **kernel) {
    if (utils::any_null(pd, kernel)) return status::invalid_arguments;
    *kernel = "";

    const primitive_desc_t *impl = pd->impl().get();
    // matmul dispatches to jit:gemm through a nested gemm pd; unwrap it.
    if (auto *mm = dynamic_cast<const matmul::gemm_t::pd_t *>(impl))
        impl = mm->gemm_pd();
    if (auto *g = dynamic_cast<const gemm::gen_t::pd_t *>(impl))
        *kernel = g->gemm_kernel_str().c_str();
    return status::success;
}
