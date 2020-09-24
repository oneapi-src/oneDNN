/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <math.h>

#include "common/primitive_exec_types.hpp"

#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/ref_reduction.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_reduction_t::pd_t::init_conf(engine_t *engine) {
    const reduction_pd_t *pd = this;

    const memory_desc_wrapper src_mdw(pd->src_md());
    const memory_desc_wrapper dst_mdw(pd->dst_md());

    const int ndims = src_mdw.ndims();
    const auto src_dims = src_mdw.md_->dims;
    const auto dst_dims = dst_mdw.md_->dims;
    const auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    conf.alg = pd->desc()->alg_kind;
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.dst_type = dst_mdw.data_type();
    conf.src_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.power = pd->desc()->p;
    conf.eps = pd->desc()->eps;
    conf.dispatch = compute_engine->create_dispatch(src_mdw.md_);
    conf.div = 1;

    for (int d = 0; d < ndims; d++) {
        conf.reduce_dims[d] = conf.dst_dims[d] = dim_t {1};
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        conf.is_reduction_dim[d] = is_reduction_dim;

        if (is_reduction_dim) {
            conf.reduce_dims[d] = src_dims[d];
            conf.div *= conf.reduce_dims[d];
        } else {
            conf.dst_dims[d] = src_dims[d];
        }
    }

    conf.dispatch.define_dim("IN", 0, conf.dst_dims[0]);
    conf.dispatch.define_dim("IC", 0, ndims >= 2 ? conf.dst_dims[1] : 1);
    conf.dispatch.define_dim(
            "ID", 0, ndims >= 5 ? conf.dst_dims[ndims - 3] : 1);
    conf.dispatch.define_dim(
            "IH", 0, ndims >= 4 ? conf.dst_dims[ndims - 2] : 1);
    conf.dispatch.define_dim(
            "IW", 0, ndims >= 3 ? conf.dst_dims[ndims - 1] : 1);
    conf.dispatch.generate(false);

    set_offsets(src_mdw, conf.off.src_off);
    set_offsets(dst_mdw, conf.off.dst_off);

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const reduction_conf_t &conf) {
    using namespace alg_kind;

    kernel_ctx.set_data_type(conf.src_type);

    kernel_ctx.define_int("IN", conf.dst_dims[0]);
    kernel_ctx.define_int("IC", conf.ndims >= 2 ? conf.dst_dims[1] : 1);
    kernel_ctx.define_int(
            "ID", conf.ndims >= 5 ? conf.dst_dims[conf.ndims - 3] : 1);
    kernel_ctx.define_int(
            "IH", conf.ndims >= 4 ? conf.dst_dims[conf.ndims - 2] : 1);
    kernel_ctx.define_int(
            "IW", conf.ndims >= 3 ? conf.dst_dims[conf.ndims - 1] : 1);

    switch (conf.alg) {
        case reduction_max: kernel_ctx.define_int("IS_MAX", 1); break;
        case reduction_min: kernel_ctx.define_int("IS_MIN", 1); break;
        case reduction_mean: kernel_ctx.define_int("IS_MEAN", 1); break;
        case reduction_sum: kernel_ctx.define_int("IS_SUM", 1); break;
        case reduction_mul: kernel_ctx.define_int("IS_MUL", 1); break;
        case reduction_norm_lp_max:
            kernel_ctx.define_int("IS_LP_MAX", 1);
            break;
        case reduction_norm_lp_sum:
            kernel_ctx.define_int("IS_LP_SUM", 1);
            break;
        case reduction_norm_lp_power_p_max:
            kernel_ctx.define_int("IS_P_MAX", 1);
            break;
        case reduction_norm_lp_power_p_sum:
            kernel_ctx.define_int("IS_P_SUM", 1);
            break;
        default: return status::invalid_arguments;
    }

    def_offsets(conf.off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(conf.off.dst_off, kernel_ctx, "DST", conf.ndims);

    kernel_ctx.define_int("REDUCTION_IN", conf.reduce_dims[0]);
    kernel_ctx.define_int(
            "REDUCTION_IC", conf.ndims >= 2 ? conf.reduce_dims[1] : 1);
    kernel_ctx.define_int("REDUCTION_ID",
            conf.ndims >= 5 ? conf.reduce_dims[conf.ndims - 3] : 1);
    kernel_ctx.define_int("REDUCTION_IH",
            conf.ndims >= 4 ? conf.reduce_dims[conf.ndims - 2] : 1);
    kernel_ctx.define_int("REDUCTION_IW",
            conf.ndims >= 3 ? conf.reduce_dims[conf.ndims - 1] : 1);

    kernel_ctx.define_int("DIV", conf.div);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("POWER", conf.power);
    kernel_ctx.define_float("EPS", conf.eps);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_reduction_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t ref_reduction_t::execute_ref(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t reduction_arg_list;

    reduction_arg_list.set(0, src);
    reduction_arg_list.set(1, dst);

    auto nd_range = conf.dispatch.nd_range();

    return parallel_for(ctx, nd_range, kernel, reduction_arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
