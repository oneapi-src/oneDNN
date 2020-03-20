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

#include "gpu/ocl/gen9_batch_normalization.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(bnorm_conf_t &conf, offsets_t &off,
        const batch_normalization_pd_t *pd) {
    using namespace dnnl::impl::format_tag;

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();

    conf.data_type = data_mdw.data_type();

    conf.ndims = ndims;
    conf.mb = data_mdw.dims()[0];

    conf.ic = data_mdw.dims()[1];
    conf.id = (ndims == 5) ? data_mdw.dims()[2] : 1;
    conf.ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
    conf.iw = data_mdw.dims()[ndims - 1];

    conf.is_forward = pd->is_fwd();
    conf.is_backward = !pd->is_fwd();

    conf.use_scaleshift = pd->use_scaleshift();
    conf.save_stats = pd->is_training();
    conf.is_training = pd->is_training();
    conf.fuse_norm_relu = pd->fuse_norm_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op();
    conf.eps = bd.batch_norm_epsilon;
    conf.calculate_diff_stats = !pd->use_global_stats();
    conf.diff_scaleshift
            = (pd->use_scaleshift() && bd.prop_kind == prop_kind::backward);

    set_offsets(data_mdw, off.src_off);

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());

    conf.mb_block = 1;
    conf.ic_block = 16;

    const bool has_padding = !data_mdw.is_dense();
    const bool is_blocked_16c
            = data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c);
    const bool is_blocked_16n16c
            = data_mdw.matches_one_of_tag(NCw16n16c, NChw16n16c, NCdhw16n16c);
    const bool is_nhwc = conf.ic % 16 == 0
            && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc);

    conf.use_nhwc = is_nhwc;

    if (has_padding || !(is_blocked_16c || is_blocked_16n16c || is_nhwc))
        return status::unimplemented;

    conf.mb_block = is_blocked_16n16c ? 16 : 1;

    if (is_nhwc) {
        // reshape to xc
        conf.nn = 1;
        conf.sp = conf.mb * conf.id * conf.ih * conf.iw;
    } else {
        // reshape to nCx16c
        conf.nn = conf.mb / conf.mb_block;
        conf.sp = conf.id * conf.ih * conf.iw * conf.mb_block;
    }

    if (conf.nn == 1)
        conf.stat_sp_block = 256;
    else
        conf.stat_sp_block = nstl::min(utils::rnd_up(conf.sp, 16), 256);

    conf.stat_sp_nblocks
            = utils::rnd_up(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;
    conf.stat_sp_tail
            = utils::rnd_dn(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;

    conf.reduce_stat_nblocks = conf.nn * conf.stat_sp_nblocks;

    conf.dispatch_calc_stat = compute_engine->create_dispatch();
    conf.dispatch_calc_stat.define_dim("STAT_MB", 0, conf.nn);
    conf.dispatch_calc_stat.define_dim("STAT_SP", 1, conf.stat_sp_nblocks);
    conf.dispatch_calc_stat.define_dim("STAT_IC", 2, conf.ic);
    conf.dispatch_calc_stat.vectorize_dim("STAT_IC", 16);
    conf.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    conf.dispatch_calc_stat.generate();

    conf.dispatch_reduce_stat = compute_engine->create_dispatch();
    conf.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
    conf.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    conf.dispatch_reduce_stat.generate();

    conf.vect_size = 8;

    const int sp_pad = utils::rnd_up(conf.sp, conf.vect_size);
    conf.sp_tail = utils::rnd_dn(conf.sp, conf.vect_size);

    conf.dispatch = compute_engine->create_dispatch(data_mdw.md_);
    conf.dispatch.define_dim("MB", 0, conf.nn);
    conf.dispatch.define_dim("SP", 1, sp_pad / conf.vect_size);
    conf.dispatch.define_dim("IC", 2, conf.ic);
    conf.dispatch.vectorize_dim("IC", 16);
    conf.dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("USE_NHWC", conf.use_nhwc);
    kernel_ctx.define_int("SP", conf.sp);
    kernel_ctx.define_int("SP_TAIL", conf.sp_tail);
    kernel_ctx.define_int("VECT_SIZE", conf.vect_size);

    kernel_ctx.define_int("STAT_SP_BLOCK", conf.stat_sp_block);
    kernel_ctx.define_int("STAT_SP_NBLOCKS", conf.stat_sp_nblocks);
    kernel_ctx.define_int("STAT_SP_TAIL", conf.stat_sp_tail);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", conf.reduce_stat_nblocks);

    if (conf.is_forward)
        kernel_ctx.define_int("IS_FWD", 1);
    else if (conf.is_backward)
        kernel_ctx.define_int("IS_BWD", 1);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALESHIFT", conf.use_scaleshift);
    kernel_ctx.define_int("CALCULATE_DIFF_STATS", conf.calculate_diff_stats);
    kernel_ctx.define_int("DIFF_SCALESHIFT", conf.diff_scaleshift);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch_calc_stat);
    def_dispatch(kernel_ctx, conf.dispatch_reduce_stat);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_batch_normalization_fwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t gen9_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t gen9_batch_normalization_fwd_t::pd_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (conf.calculate_stats) {
        size_t size = 2 * conf.reduce_stat_nblocks * conf.ic
                * types::data_type_size(data_type::f32);

        scratchpad.book(key_bnorm_reduction, size);
    }

    return status::success;
}

status_t gen9_batch_normalization_fwd_t::execute_forward(
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

    const auto &conf = pd()->conf;

    auto *mean_ptr = &mean_;
    auto *variance_ptr = &variance_;

    std::unique_ptr<memory_storage_t> temp_reduce;
    if (conf.calculate_stats) {
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);

        if (!conf.save_stats) {
            mean_ptr = temp_reduce.get();
            variance_ptr = temp_reduce.get();
        }
    }

    auto &mean = *mean_ptr;
    auto &variance = *variance_ptr;

    if (conf.calculate_stats) {
        status_t status;

        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);

        auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();
        status = compute_stream->parallel_for(
                nd_range_calc_mean, calculate_mean_kernel_, calc_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, mean);

        auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_reduce_mean,
                reduce_mean_kernel_, reduce_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);

        auto nd_range_calc_var = conf.dispatch_calc_stat.nd_range();
        status = compute_stream->parallel_for(nd_range_calc_var,
                calculate_variance_kernel_, calc_var_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, variance);

        auto nd_range_reduce_var = conf.dispatch_reduce_stat.nd_range();
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
    arg_list.set(6, conf.eps);

    auto nd_range = conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
