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

#include "gpu/intel/ocl/ref_sdpa.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t ref_sdpa_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &qry = CTX_IN_STORAGE(DNNL_ARG_QUERIES);
    const auto &key = CTX_IN_STORAGE(DNNL_ARG_KEYS);
    const auto &val = CTX_IN_STORAGE(DNNL_ARG_VALUES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    const auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    const auto &attn_mask = CTX_IN_STORAGE(DNNL_ARG_ATTN_MASK);

    const auto dst_mdw = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const int last = dst_mdw.ndims() - 1;
    const dim_t B1 = dst_mdw.ndims() > 3 ? dst_mdw.dims()[last - 3] : 1;
    const dim_t B0 = dst_mdw.ndims() > 2 ? dst_mdw.dims()[last - 2] : 1;
    const dim_t V = pd()->desc()->values();
    const dim_t D = pd()->desc()->head_size();
    const dim_t Q = pd()->desc()->queries();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, qry);
    arg_list.set(1, key);
    arg_list.set(2, val);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, attn_mask);
    arg_list.set(6, V);
    arg_list.set(7, D);

    compute::range_t gws = {(size_t)Q, (size_t)B0, (size_t)B1};
    auto nd_range = compute::nd_range_t(gws);

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
