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

#include "gpu/ocl/ref_layer_normalization.hpp"

#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_layer_normalization_init_conf(
        jit_lnorm_conf_t &jln, const layer_normalization_pd_t *pd) {
    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(pd->dst_md());

    int ndims = src_mdw.ndims();

    jln.data_type = src_mdw.data_type();
    jln.ndims = ndims;
    jln.norm_axis = pd->norm_axis();

    jln.src_md_info = jit_memory_desc_info_t::create(src_mdw);
    jln.dst_md_info = jit_memory_desc_info_t::create(dst_mdw);
    jln.stat_md_info = jit_memory_desc_info_t::create(stat_mdw);

    jln.is_fwd = pd->is_fwd();

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());
    jln.dispatch_scaleshift = compute_engine->create_dispatch();
    jln.dispatch = compute_engine->create_dispatch(
            pd->is_fwd() ? dst_mdw.md_ : src_mdw.md_);
    auto &dims = (pd->is_fwd() ? src_mdw : dst_mdw).dims();
    if (pd->is_fwd()) {
        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            jln.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
    } else {
        jln.dispatch_scaleshift.define_dim("C", pd->norm_axis());

        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            jln.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
    }

    if (!pd->is_fwd()) {
        jln.dispatch_scaleshift.set_kernel_attr_suffix("SCALESHIFT");
        jln.dispatch_scaleshift.generate();
    }

    jln.dispatch.generate();

    jln.use_scaleshift = pd->use_scaleshift();
    jln.calculate_stats = !pd->stats_are_src();
    jln.save_stats = pd->is_training();
    jln.eps = pd->desc()->layer_norm_epsilon;

    return status::success;
}

status_t ref_layer_normalization_init_const_def(
        const jit_lnorm_conf_t &jln, compute::kernel_ctx_t &kernel_ctx) {
    kernel_ctx.set_data_type(jln.data_type);

    kernel_ctx.define_int("C", jln.norm_axis);
    kernel_ctx.define_int("NDIMS", jln.ndims);
    kernel_ctx.define_int("USE_SCALESHIFT", jln.use_scaleshift);
    kernel_ctx.define_int("CALCULATE_STATS", jln.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", jln.save_stats);
    kernel_ctx.define_int("IS_FWD", jln.is_fwd);
    kernel_ctx.define_int("IS_BWD", !jln.is_fwd);

    def_memory_desc_info(kernel_ctx, jln.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, jln.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, jln.stat_md_info, "STAT");

    def_dispatch(kernel_ctx, jln.dispatch);
    if (!jln.is_fwd) def_dispatch(kernel_ctx, jln.dispatch_scaleshift);

    return status::success;
}

status_t ref_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &jln = pd()->jln_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, jln.eps);

    auto nd_range_kernel = jln.dispatch.nd_range();
    status_t status
            = compute_stream->parallel_for(nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t ref_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_scaleshift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE_SHIFT);

    const auto &jln = pd()->jln_;

    if (jln.use_scaleshift) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, mean);
        arg_list.set(2, variance);
        arg_list.set(3, diff_dst);
        arg_list.set(4, diff_scaleshift);
        arg_list.set(5, jln.eps);

        auto nd_range = jln.dispatch_scaleshift.nd_range();
        status_t status = compute_stream->parallel_for(
                nd_range, kernel_scaleshift_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, diff_src);
    arg_list.set(6, jln.eps);

    auto nd_range_kernel = jln.dispatch.nd_range();
    status_t status
            = compute_stream->parallel_for(nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
