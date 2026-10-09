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
#include "gpu/intel/gemm/with_post_ops.hpp"
#include "gpu/intel/matmul/gemm.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::intel;

// Dev-only jit:gemm kernel override; the symbol is absent from a release
// library. Read back the deployed kernel via ONEDNN_VERBOSE=xe=info.
#ifdef DNNL_DEV_MODE
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

// Number of jit:gemm selection candidates, i.e. valid rank overrides; -1 if
// the selected implementation is not jit:gemm.
extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_count(
        const_dnnl_primitive_desc_t pd, int *count) {
    if (utils::any_null(pd, count)) return status::invalid_arguments;

    *count = -1;
    const primitive_desc_t *impl = pd->impl().get();
    // matmul and gemm with post-ops reach jit:gemm through a nested gemm pd.
    while (impl) {
        if (auto *mm = dynamic_cast<const matmul::gemm_t::pd_t *>(impl))
            impl = mm->gemm_pd();
        else if (auto *po
                = dynamic_cast<const gemm::with_post_ops_t::pd_t *>(impl))
            impl = po->pd_.get();
        else
            break;
    }
    if (auto *g = dynamic_cast<const gemm::gen_t::pd_t *>(impl))
        *count = g->kernel_count();
    return status::success;
}
#endif // DNNL_DEV_MODE
