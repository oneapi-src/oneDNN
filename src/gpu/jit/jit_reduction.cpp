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

// A small wrapper on the jit_reduction_generator_t, used to test its functionality.
// Only valid in dev mode for now, until performance is improved.
#ifdef DNNL_DEV_MODE

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"

#include "common/utils.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/utils.hpp"
#include "gpu/jit/jit_reduction.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace gpu_utils;

status_t jit_reduction_t::pd_t::init_conf(engine_t *engine) {
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

    // Only allow cases where inner size aligns with register size
    compute::compute_engine_t &compute_engine
            = *utils::downcast<compute::compute_engine_t *>(engine);
    ngen::HW hw = convert_dnnl_arch_to_ngen(
            compute_engine.device_info()->gpu_arch());
    size_t reg_size = gpu_utils::into<size_t>(ngen::GRF::bytes(hw));
    size_t dst_nelems = gpu_utils::into<size_t>(dst_mdw.nelems());
    size_t inner_nelems = gpu_utils::into<size_t>(reduction_stride);
    if (inner_nelems * sizeof(float) % reg_size != 0)
        return status::unimplemented;

    // valid case, now compute the nd_range_t
    size_t outer_nelems = dst_nelems / inner_nelems;
    compute::range_t gws(inner_nelems, outer_nelems);
    compute::range_t lws(reg_size / sizeof(float), 1);
    nd_range = {gws, lws};

    return status::success;
}

status_t jit_reduction_t::execute(const exec_ctx_t &ctx) const {
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
