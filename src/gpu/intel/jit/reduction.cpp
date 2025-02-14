/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

// A small wrapper on the reduction_generator_t, used to test its functionality.
// Only valid in dev mode for now, until performance is improved.
#include "gpu/intel/gpu_primitive_attr.hpp"
#ifdef DNNL_DEV_MODE

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/reduction.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

using namespace gpu_utils;

status_t reduction_t::pd_t::init_conf(impl::engine_t *engine) {
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const int ndims = src_mdw.ndims();
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *dst_dims = dst_mdw.dims();

    // Allow plain formats only
    bool plain_case = true;
    plain_case &= (src_mdw.blocking_desc().inner_nblks == 0);
    plain_case &= (dst_mdw.blocking_desc().inner_nblks == 0);
    if (!plain_case) return status::unimplemented;

    // Allow only 1 reduced dimension, for now
    for (int i = 0; i < ndims; i++) {
        bool is_reduced = (src_dims[i] != dst_dims[i]);
        if (is_reduced && reduction_size) return status::unimplemented;
        if (is_reduced) {
            reduction_size = src_dims[i];
            reduction_stride = src_mdw.blocking_desc().strides[i];
        }
    }
    assert(reduction_size);
    assert(reduction_stride);

    dim_t dst_nelems = dst_mdw.nelems();
    dim_t inner_nelems = reduction_stride;
    int dt_size = into<int>(sizeof(float));

    auto &compute_engine
            = *utils::downcast<compute::compute_engine_t *>(engine);
    const compute::device_info_t &device_info = *compute_engine.device_info();
    int reg_size = device_info.grf_size();
    int elems_per_reg = reg_size / dt_size;
    int default_nregs
            = utils::max_div(into<int>(inner_nelems / elems_per_reg), 16);

    nregs = dev_getenv("jit_reduction_nregs", into<int>(default_nregs));

    // Only allow cases where inner size aligns with register size
    if (inner_nelems % (elems_per_reg * nregs) != 0)
        return status::unimplemented;

    // Grouping threads into threadgroups: ensures better access patterns (we can use barriers)
    // --> Use the largest threadgroup possible, must fit within the inner dimension
    dim_t gws0 = inner_nelems / nregs;
    dim_t nthreads = gws0 / elems_per_reg;
    int tg_size = [this, &device_info, &nthreads]() {
        const compute::gpu_arch_t arch = device_info.gpu_arch();
        auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
                attr()->gpu_attr_.get());
        const int threads_per_eu = gpu_attr
                ? gpu_attr->threads_per_eu()
                : compute::device_info_t::threads_per_eu(arch);
        int tg_size = utils::rnd_down_pow2(
                device_info.max_eus_per_wg() * threads_per_eu);
        while (nthreads % tg_size != 0) {
            tg_size /= 2;
        }
        tg_size = dev_getenv("jit_reduction_tg_size", tg_size);
        return tg_size;
    }();
    gpu_assert(nthreads % tg_size == 0) << "Invalid tg_size";

    // valid case, now compute the nd_range_t
    dim_t outer_nelems = dst_nelems / inner_nelems;
    compute::range_t gws(into<size_t>(gws0), into<size_t>(outer_nelems));
    compute::range_t lws(into<size_t>(tg_size * elems_per_reg), 1);
    nd_range = compute::nd_range_t(gws, lws);

    return status::success;
}

status_t reduction_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    // Set up the reduction arg list
    compute::kernel_arg_list_t reduction_arg_list;

    reduction_arg_list.append(src);
    reduction_arg_list.append(dst);

    CHECK(parallel_for(ctx, pd()->nd_range, kernel_, reduction_arg_list));

    return status::success;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
