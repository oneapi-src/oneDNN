/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/ref_softmax.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_softmax_fwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, src_scale);
    arg_list.set(3, dst_scale);
    append_post_ops_to_arg_list(ctx, arg_list, 4, pd()->attr()->post_ops_);

    if (pd()->group_size > 1) {
        auto nd_range = compute::nd_range_t(pd()->gws, pd()->lws);
        return parallel_for(ctx, nd_range, kernel_, arg_list);
    } else {
        auto nd_range = compute::nd_range_t(pd()->gws);
        return parallel_for(ctx, nd_range, kernel_, arg_list);
    }
}

status_t ref_softmax_bwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &dst = CTX_IN_STORAGE(DNNL_ARG_DST);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dst);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);

    auto nd_range = compute::nd_range_t(pd()->gws, pd()->lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
