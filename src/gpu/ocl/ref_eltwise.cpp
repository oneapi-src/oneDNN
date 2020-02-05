/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/ocl/ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(
        eltwise_conf_t &conf, offsets_t &off, const eltwise_pd_t *pd) {
    alg_kind_t alg = pd->desc()->alg_kind;
    bool is_forward = utils::one_of(pd->desc()->prop_kind,
            prop_kind::forward_training, prop_kind::forward_inference);
    const memory_desc_wrapper data_d(pd->src_md());
    const memory_desc_wrapper diff_data_d(
            is_forward ? &glob_zero_md : pd->diff_src_md());

    const int ndims = data_d.ndims();
    conf.ndims = ndims;

    conf.data_type = data_d.data_type();
    conf.alg = alg;
    conf.is_forward = is_forward;

    set_offsets(data_d, off.src_off);
    set_offsets(diff_data_d, off.dst_off);

    const auto &dims = data_d.dims();

    conf.with_zero_padding = data_d.nelems(false) != data_d.nelems(true);

    int max_ndims = 6;
    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());
    conf.dispatch = compute_engine->create_dispatch(
            is_forward ? data_d.md_ : diff_data_d.md_);
    for (int i = 0; i < max_ndims; ++i) {
        if (i < ndims)
            conf.dispatch.define_dim(utils::format("D%d", i), i, dims[i]);
        else
            conf.dispatch.define_dim(utils::format("D%d", i), 1);
    }
    conf.dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const eltwise_conf_t &conf, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);
    kernel_ctx.define_int("RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("LINEAR", alg_kind::eltwise_linear);
    kernel_ctx.define_int("BOUNDED_RELU", alg_kind::eltwise_bounded_relu);
    kernel_ctx.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
    kernel_ctx.define_int("LOGISTIC", alg_kind::eltwise_logistic);
    kernel_ctx.define_int("TANH", alg_kind::eltwise_tanh);
    kernel_ctx.define_int("ELU", alg_kind::eltwise_elu);
    kernel_ctx.define_int("SQUARE", alg_kind::eltwise_square);
    kernel_ctx.define_int("SQRT", alg_kind::eltwise_sqrt);
    kernel_ctx.define_int("ABS", alg_kind::eltwise_abs);
    kernel_ctx.define_int("EXP", alg_kind::eltwise_exp);
    kernel_ctx.define_int("GELU_TANH", alg_kind::eltwise_gelu);
    kernel_ctx.define_int("SWISH", alg_kind::eltwise_swish);
    kernel_ctx.define_int("LOG", alg_kind::eltwise_log);
    kernel_ctx.define_int("CLIP", alg_kind::eltwise_clip);
    kernel_ctx.define_int("POW", alg_kind::eltwise_pow);
    kernel_ctx.define_int("GELU_ERF", alg_kind::eltwise_gelu_erf);

    kernel_ctx.define_int("RELU_DST", alg_kind::eltwise_relu_use_dst_for_bwd);
    kernel_ctx.define_int(
            "LOGISTIC_DST", alg_kind::eltwise_logistic_use_dst_for_bwd);
    kernel_ctx.define_int("TANH_DST", alg_kind::eltwise_tanh_use_dst_for_bwd);
    kernel_ctx.define_int("ELU_DST", alg_kind::eltwise_elu_use_dst_for_bwd);
    kernel_ctx.define_int("SQRT_DST", alg_kind::eltwise_sqrt_use_dst_for_bwd);
    kernel_ctx.define_int("EXP_DST", alg_kind::eltwise_exp_use_dst_for_bwd);

    kernel_ctx.define_int("ALG_KIND", conf.alg);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("GWS0", conf.dispatch.nd_range().global_range()[0]);
    kernel_ctx.define_int("GWS1", conf.dispatch.nd_range().global_range()[1]);
    kernel_ctx.define_int("GWS2", conf.dispatch.nd_range().global_range()[2]);

    kernel_ctx.define_int("ZERO_PADDING", conf.with_zero_padding);

    def_offsets(off.src_off, kernel_ctx, "DATA", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DIFF_DATA",
            conf.is_forward ? 0 : conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_eltwise_fwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_eltwise_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_eltwise_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);

    auto nd_range = conf.dispatch.nd_range();
    return compute_stream->parallel_for(nd_range, kernel_, arg_list);
}

status_t ref_eltwise_bwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_eltwise_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = pd()->use_dst() ? CTX_IN_STORAGE(DNNL_ARG_DST)
                                : CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);
    arg_list.set(3, alpha);
    arg_list.set(4, beta);

    auto nd_range = conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
