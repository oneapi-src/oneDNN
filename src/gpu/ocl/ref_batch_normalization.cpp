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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"

#include "gpu/ocl/ref_batch_normalization.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_batch_normalization_init_conf(jit_bnorm_conf_t &jbn,
        const batch_normalization_pd_t *pd, jit_offsets &jit_off) {
    using namespace dnnl::impl::format_tag;

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();

    jbn.data_type = data_mdw.data_type();

    jbn.ndims = ndims;
    jbn.mb = data_mdw.dims()[0];

    jbn.ic = data_mdw.dims()[1];
    jbn.id = (ndims == 5) ? data_mdw.dims()[2] : 1;
    jbn.ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
    jbn.iw = data_mdw.dims()[ndims - 1];

    jbn.is_forward = pd->is_fwd();
    jbn.is_backward = !pd->is_fwd();

    jbn.use_scaleshift = pd->use_scaleshift();
    jbn.save_stats = pd->is_training();
    jbn.is_training = pd->is_training();
    jbn.fuse_norm_relu = pd->fuse_norm_relu();
    jbn.calculate_stats = !pd->stats_is_src();
    jbn.with_relu = pd->with_relu_post_op();
    jbn.eps = bd.batch_norm_epsilon;
    jbn.calculate_diff_stats = !pd->use_global_stats();
    jbn.diff_scaleshift
            = (pd->use_scaleshift() && bd.prop_kind == prop_kind::backward);

    set_offsets(data_mdw, jit_off.src_off);

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());

    const bool has_padding = !data_mdw.is_dense();
    if (!has_padding
            && data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c, NCw16n16c,
                    NChw16n16c, NCdhw16n16c)) {
        jbn.mb_block = data_mdw.matches_one_of_tag(
                               NCw16n16c, NChw16n16c, NCdhw16n16c)
                ? 16
                : 1;

        jbn.use_16mb_unroll = 1;

        const int max_stat_nblocks = 256;
        int stat_mb_nblocks = jbn.mb / jbn.mb_block;
        int stat_sp_nblocks = utils::max_div(jbn.id * jbn.ih * jbn.iw,
                nstl::max(1, max_stat_nblocks / stat_mb_nblocks));
        assert(stat_mb_nblocks * stat_sp_nblocks <= max_stat_nblocks);

        int stat_sp_block = jbn.id * jbn.ih * jbn.iw / stat_sp_nblocks;

        jbn.reduce_stat_nblocks = stat_mb_nblocks * stat_sp_nblocks;

        jbn.dispatch_calc_stat = compute_engine->create_dispatch();
        jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_SP", 2, jbn.id * jbn.ih * jbn.iw, stat_sp_block);
        jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_IC", 1, jbn.ic);
        jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                "STAT_MB", 0, jbn.mb, jbn.mb_block);
        jbn.dispatch_calc_stat.vectorize_dim("STAT_IC", 16);
        jbn.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
        jbn.dispatch_calc_stat.generate();

        jbn.dispatch_reduce_stat = compute_engine->create_dispatch();
        jbn.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", jbn.ic);
        jbn.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
        jbn.dispatch_reduce_stat.generate();

        jbn.dispatch = compute_engine->create_dispatch(data_mdw.md_);
        jbn.dispatch.define_dim("MB", 0, jbn.mb, jbn.mb_block);
        jbn.dispatch.define_dim("IC", 1, jbn.ic);
        jbn.dispatch.define_dim("ID", nstl::max(2, ndims - 3), jbn.id);
        jbn.dispatch.define_dim("IH", nstl::max(2, ndims - 2), jbn.ih);
        jbn.dispatch.define_dim("IW", nstl::max(2, ndims - 1), jbn.iw);
        jbn.dispatch.vectorize_dim("IC", 16);
        jbn.dispatch.generate();
    } else {
        // Reference
        jbn.use_16mb_unroll = 0;
        jbn.mb_block = 1;
        jbn.dispatch = compute_engine->create_dispatch();
        jbn.dispatch.define_dim("IC", jbn.ic);
        jbn.dispatch.generate();
    }

    return status::success;
}

status_t ref_batch_normalization_init_const_def(
        compute::kernel_ctx_t &kernel_ctx, const jit_bnorm_conf_t &jbn,
        const jit_offsets &jit_off) {
    kernel_ctx.set_data_type(jbn.data_type);

    kernel_ctx.define_int("NDIMS", jbn.ndims);
    kernel_ctx.define_int("MB", jbn.mb);
    kernel_ctx.define_int("IC", jbn.ic);
    kernel_ctx.define_int("ID", jbn.id);
    kernel_ctx.define_int("IH", jbn.ih);
    kernel_ctx.define_int("IW", jbn.iw);
    kernel_ctx.define_int("USE_16MB_UNROLL", jbn.use_16mb_unroll);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", jbn.reduce_stat_nblocks);
    kernel_ctx.define_int("MB_BLOCK", jbn.mb_block);

    if (jbn.is_forward)
        kernel_ctx.define_int("IS_FWD", 1);
    else if (jbn.is_backward)
        kernel_ctx.define_int("IS_BWD", 1);

    kernel_ctx.define_int("WITH_RELU", jbn.with_relu);
    kernel_ctx.define_int("SAVE_STATS", jbn.save_stats);
    kernel_ctx.define_int("IS_TRAINING", jbn.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", jbn.fuse_norm_relu);
    kernel_ctx.define_int("CALCULATE_STATS", jbn.calculate_stats);
    kernel_ctx.define_int("USE_SCALESHIFT", jbn.use_scaleshift);
    kernel_ctx.define_int("CALCULATE_DIFF_STATS", jbn.calculate_diff_stats);
    kernel_ctx.define_int("DIFF_SCALESHIFT", jbn.diff_scaleshift);

    def_offsets(jit_off.src_off, kernel_ctx, "SRC", jbn.ndims);

    if (jbn.use_16mb_unroll) {
        def_dispatch(kernel_ctx, jbn.dispatch_calc_stat);
        def_dispatch(kernel_ctx, jbn.dispatch_reduce_stat);
    }
    def_dispatch(kernel_ctx, jbn.dispatch);

    return status::success;
}

void ref_batch_normalization_init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_bnorm_conf_t &jbn) {
    if (jbn.is_forward) {
        if (jbn.use_16mb_unroll && jbn.calculate_stats) {
            size_t size = 2 * jbn.reduce_stat_nblocks * jbn.ic
                    * types::data_type_size(data_type::f32);

            scratchpad.book(memory_tracking::names::key_bnorm_reduction, size);
        }
    }
    if (jbn.is_backward) {
        if (jbn.use_16mb_unroll) {
            size_t size = 2 * jbn.reduce_stat_nblocks * jbn.ic
                    * types::data_type_size(data_type::f32);
            scratchpad.book(memory_tracking::names::key_bnorm_reduction, size);
        }
    }
}

status_t ref_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);

    auto &mean_ = pd()->stats_is_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance_ = pd()->stats_is_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);

    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    const auto &jbn = pd()->jbn_;

    auto *mean_ptr = &mean_;
    auto *variance_ptr = &variance_;

    std::unique_ptr<memory_storage_t> temp_reduce = nullptr;
    if (jbn.use_16mb_unroll && jbn.calculate_stats) {
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);

        if (!jbn.save_stats) {
            mean_ptr = temp_reduce.get();
            variance_ptr = temp_reduce.get();
        }
    }

    auto &mean = *mean_ptr;
    auto &variance = *variance_ptr;

    if (jbn.use_16mb_unroll && jbn.calculate_stats) {
        status_t status;

        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);

        auto nd_range_calc_mean = jbn.dispatch_calc_stat.nd_range();
        status = compute_stream->parallel_for(
                nd_range_calc_mean, calculate_mean_kernel_, calc_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, mean);

        auto nd_range_reduce_mean = jbn.dispatch_reduce_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_reduce_mean,
                reduce_mean_kernel_, reduce_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);

        auto nd_range_calc_var = jbn.dispatch_calc_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_calc_var,
                calculate_variance_kernel_, calc_var_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, variance);

        auto nd_range_reduce_var = jbn.dispatch_reduce_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_reduce_var,
                reduce_variance_kernel_, reduce_var_arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, ws);
    arg_list.set(6, jbn.eps);

    auto nd_range = jbn.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t ref_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_scaleshift_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE_SHIFT);

    const auto &jbn = pd()->jbn_;

    std::unique_ptr<memory_storage_t> temp_reduce;
    if (jbn.use_16mb_unroll) {
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);
    }

    auto &diff_scaleshift = (jbn.use_16mb_unroll && !jbn.diff_scaleshift)
            ? *temp_reduce
            : diff_scaleshift_;

    if (jbn.use_16mb_unroll) {
        status_t status;

        compute::kernel_arg_list_t calc_stats_arg_list;
        calc_stats_arg_list.set(0, src);
        calc_stats_arg_list.set(1, mean);
        calc_stats_arg_list.set(2, diff_dst);
        calc_stats_arg_list.set(3, ws);
        calc_stats_arg_list.set(4, *temp_reduce);

        auto nd_range = jbn.dispatch_calc_stat.nd_range();
        status = compute_stream->parallel_for(
                nd_range, calculate_stats_kernel_, calc_stats_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_stats_arg_list;
        reduce_stats_arg_list.set(0, *temp_reduce);
        reduce_stats_arg_list.set(1, diff_scaleshift);
        reduce_stats_arg_list.set(2, variance);
        reduce_stats_arg_list.set(3, jbn.eps);

        auto nd_range_reduce_stat = jbn.dispatch_reduce_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_reduce_stat,
                reduce_stats_kernel_, reduce_stats_arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, ws);
    arg_list.set(6, diff_src);
    arg_list.set(7, diff_scaleshift);
    arg_list.set(8, jbn.eps);

    auto nd_range = jbn.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
