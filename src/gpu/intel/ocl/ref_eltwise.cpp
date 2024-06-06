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

#include "gpu/intel/ocl/ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t init_conf_common(ref_eltwise_conf_t &conf,
        const eltwise_pd_t *pd, impl::engine_t *engine) {
    alg_kind_t alg = pd->desc()->alg_kind;
    const bool is_forward = pd->is_fwd();
    const auto &src_md = pd->use_dst() ? pd->dst_md() : pd->src_md();
    const memory_desc_wrapper src_d(src_md);
    const memory_desc_wrapper diff_data_d(
            is_forward ? &glob_zero_md : pd->diff_src_md());

    conf.data_md_info = memory_desc_info_t::create(src_d);
    if (!is_forward)
        conf.data_diff_md_info = memory_desc_info_t::create(diff_data_d);

    const int ndims = src_d.ndims();
    conf.ndims = ndims;

    conf.data_type = src_d.data_type();
    conf.alg = alg;
    conf.is_forward = is_forward;
    conf.attr_info = attr_info_t::create(pd->attr());

    const auto &dims = src_d.padded_dims();

    conf.with_zero_padding = src_d.nelems(false) != src_d.nelems(true);

    int max_ndims = 6;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(
            is_forward ? src_d.md_ : diff_data_d.md_);
    for (int i = 0; i < max_ndims; ++i) {
        if (i < ndims)
            conf.dispatch.define_dim(utils::format("D%d", i), i, dims[i]);
        else
            conf.dispatch.define_dim(utils::format("D%d", i), 1);
    }
    conf.dispatch.generate(/*generate_lws=*/false);

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const ref_eltwise_conf_t &conf, const post_ops_t &post_ops,
        const memory_desc_t *dst_md) {
    kernel_ctx.set_data_type(conf.data_type);

    def_eltwise_alg_kinds(kernel_ctx);

    kernel_ctx.define_int("WITH_ELTWISE", 1);
    kernel_ctx.define_int("ELTWISE_ALG", conf.alg);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("GWS0", conf.dispatch.nd_range().global_range()[0]);
    kernel_ctx.define_int("GWS1", conf.dispatch.nd_range().global_range()[1]);
    kernel_ctx.define_int("GWS2", conf.dispatch.nd_range().global_range()[2]);
    kernel_ctx.define_int("USE_CUSTOM_GWS_GET_ID", 1);

    bool with_binary_post_ops
            = post_ops.find(primitive_kind_t::dnnl_binary) != -1;
    kernel_ctx.define_int(
            "USE_GWS_GET", conf.with_zero_padding || with_binary_post_ops);

    def_memory_desc_info(kernel_ctx, conf.data_md_info, "DATA");

    if (!conf.is_forward) {
        def_memory_desc_info(kernel_ctx, conf.data_diff_md_info, "DIFF_DATA");
    } else {
        kernel_ctx.define_int("IS_FWD", 1);
    }

    CHECK(def_attr_info(kernel_ctx, conf.attr_info, post_ops, *dst_md));
    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_eltwise_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_eltwise_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, invariant_dst_md());
}

status_t ref_eltwise_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {

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

    append_post_ops_to_arg_list(ctx, arg_list, 5, pd()->attr()->post_ops_);

    auto nd_range = conf.dispatch.nd_range();
    return large_parallel_for(ctx, nd_range, kernel_, arg_list, 4);
}

status_t ref_eltwise_bwd_t::pd_t::init_conf(impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_eltwise_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, invariant_dst_md());
}

status_t ref_eltwise_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {

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
    return large_parallel_for(ctx, nd_range, kernel_, arg_list, 5);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
