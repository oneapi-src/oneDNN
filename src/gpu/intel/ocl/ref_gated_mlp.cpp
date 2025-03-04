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

#include "gpu/intel/ocl/ref_gated_mlp.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t ref_gated_mlp_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &W_gate = CTX_IN_STORAGE(DNNL_ARG_WTS_GATE);
    const auto &W_up = CTX_IN_STORAGE(DNNL_ARG_WTS_UP);
    const auto &W_down = CTX_IN_STORAGE(DNNL_ARG_WTS_DOWN);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    //const auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    //const auto &attn_mask = CTX_IN_STORAGE(DNNL_ARG_ATTN_MASK);

    //const auto dst_mdw = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    //const int last = dst_mdw.ndims() - 1;
    //const dim_t B1 = dst_mdw.ndims() > 3 ? dst_mdw.dims()[last - 3] : 1;
    //const dim_t B0 = dst_mdw.ndims() > 2 ? dst_mdw.dims()[last - 2] : 1;
    const dim_t MB = pd()->desc()->mb_sz();
    const dim_t IC = pd()->desc()->ic_sz();
    const dim_t OC = pd()->desc()->oc_sz();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, W_gate);
    arg_list.set(2, W_up);
    arg_list.set(3, W_down);
    arg_list.set(4, dst);
    arg_list.set(5, MB);
    arg_list.set(6, IC);
    arg_list.set(7, OC);

    compute::range_t gws = { (size_t)utils::div_up(MB, 128) * 128  }; //TODO: determine legit partition, lws size
    compute::range_t lws = { (size_t)128 };
    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
