/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/reusable_softmax.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

compute::kernel_ctx_t reusable_softmax_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.define_int("LOGSOFTMAX", is_logsoftmax);

    kernel_ctx.set_data_type(src_data_type);
    def_data_type(kernel_ctx, src_data_type, "SRC");
    def_data_type(kernel_ctx, dst_data_type, "DST");

    gws_params.def_kernel_macros(kernel_ctx);
    return kernel_ctx;
}

status_t reusable_softmax_fwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.append(src_scale);
    arg_list.append(dst_scale);
    arg_list.append(pd()->rt_conf.softmax_axis_size);
    arg_list.append(pd()->rt_conf.softmax_axis_stride);
    arg_list.append(pd()->rt_conf.gws_params.get());

    return parallel_for(
            ctx, pd()->rt_conf.gws_params.nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
