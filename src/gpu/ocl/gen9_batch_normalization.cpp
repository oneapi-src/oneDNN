/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#include "gpu/ocl/ocl_utils.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

int get_nhwc_ic_block(int ic, int max_vect_size = 8) {
    const int nblocks = ic / (max_vect_size * 16);
    return nblocks < 2 || (ic / nblocks) % 16 ? ic : ic / nblocks;
}
int get_nhwc_vect_size(int ic, int simd = 16) {
    int vect_size = 8;
    while (true) {
        if (ic / (vect_size * simd)) return vect_size;
        vect_size /= 2;
    }
    return 1;
}

int get_nhwc_sp_block_size(
        int sp, int ic_dim, int eu_count, int threads_per_eu, int simd = 16) {

    float efficiency_eus = 0.0f;
    float efficiency_threads = 0.0f;
    int block_size_eus = sp;
    int block_size_threads = sp;
    int curr_block_size = sp;

    while (true) {
        int nblocks = utils::div_up(sp, curr_block_size);
        int nthr = nblocks * (ic_dim / simd);

        float curr_efficiency_eus = (float)nthr / utils::rnd_up(nthr, eu_count);
        float curr_efficiency_threads
                = (float)nthr / utils::rnd_up(nthr, eu_count * threads_per_eu);

        if (curr_efficiency_threads > efficiency_threads) {
            efficiency_threads = curr_efficiency_threads;
            block_size_threads = curr_block_size;
        }
        if (curr_efficiency_eus >= efficiency_eus) {
            efficiency_eus = curr_efficiency_eus;
            block_size_eus = curr_block_size;
        }
        if (nthr >= eu_count * threads_per_eu && efficiency_eus == 1)
            return block_size_eus;
        if (curr_block_size == 1) return block_size_threads;
        curr_block_size--;
    }
}

int get_block_size(bool is_backward, int hw_threads, int nn, int ic,
        int work_size, int simd = 16) {
    int block_size = 256;
    float thread_efficiency = 0;
    int hw_thread_mult = hw_threads;
    const int align_size = is_backward ? 8 : 16;
    while (true) {
        const int nof_blocks
                = nstl::max(utils::rnd_dn(hw_thread_mult * simd, ic) / ic, 1);
        const int min_block_size
                = utils::rnd_up(work_size, nof_blocks) / nof_blocks;
        const int curr_block_size = utils::rnd_up(min_block_size, align_size);
        const int nof_blocks_generated
                = utils::rnd_up(work_size, curr_block_size) / curr_block_size;
        const int threads_generated = nof_blocks_generated * ic / simd;
        const float curr_thread_efficiency = float(threads_generated * nn)
                / float(utils::rnd_up(threads_generated * nn, hw_threads));
        if (curr_thread_efficiency > thread_efficiency) {
            thread_efficiency = curr_thread_efficiency;
            block_size = curr_block_size;
        }
        if (curr_thread_efficiency == 1.0 || curr_block_size <= 256) { break; }
        hw_thread_mult += hw_threads;
    }
    return block_size;
}

static status_t init_conf_common(bnorm_conf_t &conf, offsets_t &off,
        const batch_normalization_pd_t *pd, engine_t *engine) {
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
    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.save_stats = pd->is_training();
    conf.is_training = pd->is_training();
    conf.fuse_norm_relu = pd->fuse_norm_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op();
    conf.eps = bd.batch_norm_epsilon;
    conf.calculate_diff_stats = !pd->use_global_stats();
    conf.diff_scaleshift
            = (pd->use_scaleshift() && bd.prop_kind == prop_kind::backward);
    conf.diff_scale = (pd->use_scale() && bd.prop_kind == prop_kind::backward);
    conf.diff_shift = (pd->use_shift() && bd.prop_kind == prop_kind::backward);

    set_offsets(data_mdw, off.src_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    conf.mb_block = 1;

    const bool has_padding = !data_mdw.is_dense();
    const bool is_blocked_16c
            = data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c);
    const bool is_blocked_16n16c
            = data_mdw.matches_one_of_tag(NCw16n16c, NChw16n16c, NCdhw16n16c);
    const bool is_blocked_32n16c
            = data_mdw.matches_one_of_tag(NCw32n16c, NChw32n16c, NCdhw32n16c);
    const bool is_nhwc
            = conf.ic % 8 == 0 && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc);

    // Due to intel_sub_group_write_uc requires 16-bytes alignment,
    // IC div by 8 tail processing is not applicable to fuse_norm_relu
    // and char data type.
    if (conf.ic % 8 == 0 && conf.ic % 16
            && (conf.fuse_norm_relu || conf.data_type == data_type::s8))
        return status::unimplemented;
    // IC tail processing performnce boost is not obvious on arch < xe_hpc
    if (conf.ic % 8 == 0 && conf.ic % 16
            && gpu_arch < compute::gpu_arch_t::xe_hpc)
        return status::unimplemented;

    conf.use_stats_one_pass = experimental::use_bnorm_stats_one_pass();

    // IC tail processing is not implemented yet for one pass algorithm
    // TODO: implement it, possible perf boost could be ~ 2x
    if (conf.ic % 8 == 0 && conf.ic % 16 && conf.use_stats_one_pass)
        conf.use_stats_one_pass = false;

    conf.use_nhwc = is_nhwc;

    if (has_padding
            || !(is_blocked_16c || is_blocked_16n16c || is_blocked_32n16c
                    || is_nhwc))
        return status::unimplemented;

    // IC tail processing is not implemented yet for NHWC optimized kernels
    // TODO: implement it
    // The use of NHWC optimized kernels
    // is limited by XeHPG+ due to performance reasons
    conf.nhwc_optimized = conf.ic % 16 == 0
            && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
            && gpu_arch >= compute::gpu_arch_t::xe_hpg;

    if (conf.nhwc_optimized) {
        conf.ic_block = get_nhwc_ic_block(utils::rnd_up(conf.ic, 16));
    } else {
        conf.ic_block = 16;
    }

    conf.mb_block = is_blocked_32n16c ? 32 : is_blocked_16n16c ? 16 : 1;

    if (is_nhwc) {
        // reshape to xc
        conf.nn = 1;
        conf.sp = conf.mb * conf.id * conf.ih * conf.iw;
    } else {
        // reshape to nCx16c
        conf.nn = conf.mb / conf.mb_block;
        conf.sp = conf.id * conf.ih * conf.iw * conf.mb_block;
    }

    // The case IC==8 requires spacial dim to be even because of using one
    // block read/write operation for 2 spacial rows at once
    if (is_nhwc && conf.ic == 8 && conf.sp % 2) return status::unimplemented;

    conf.calc_stat_ic = conf.nhwc_optimized
            ? utils::div_up(conf.ic, conf.ic_block) * 16
            : utils::rnd_up(conf.ic, 16);

    auto eu_count = compute_engine->device_info()->eu_count();
    auto threads_per_eu
            = compute::device_info_t::threads_per_eu(gpu_arch, false);
    const int max_sp_block_size = get_block_size(conf.is_backward, eu_count,
            conf.nn, utils::rnd_up(conf.ic, 16), conf.sp);
    const int nhwc_sp_block = get_nhwc_sp_block_size(
            conf.sp, conf.calc_stat_ic, eu_count, threads_per_eu);

    if (conf.nn == 1)
        conf.stat_sp_block
                = conf.nhwc_optimized ? nhwc_sp_block : max_sp_block_size;
    else
        conf.stat_sp_block
                = nstl::min(utils::rnd_up(conf.sp, 16), max_sp_block_size);

    conf.stat_sp_nblocks
            = utils::rnd_up(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;
    conf.stat_sp_tail
            = utils::rnd_dn(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;

    conf.reduce_stat_nblocks = conf.nn * conf.stat_sp_nblocks;

    conf.vect_size
            = conf.nhwc_optimized ? get_nhwc_vect_size(conf.ic_block) : 8;

    conf.dispatch_calc_stat = compute_engine->create_dispatch();
    conf.dispatch_calc_stat.define_dim("STAT_MB", 0, conf.nn);
    conf.dispatch_calc_stat.define_dim("STAT_SP", 1, conf.stat_sp_nblocks);
    conf.dispatch_calc_stat.define_dim("STAT_IC", 2, conf.calc_stat_ic);
    CHECK(conf.dispatch_calc_stat.vectorize_dim("STAT_IC", 16));
    conf.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    conf.dispatch_calc_stat.generate();

    conf.dispatch_reduce_stat = compute_engine->create_dispatch();
    int reduce_sub_group_count = 1;
    while (conf.reduce_stat_nblocks % (2 * reduce_sub_group_count) == 0
            && 2 * reduce_sub_group_count * 16 <= 256) {
        reduce_sub_group_count = reduce_sub_group_count * 2;
    }
    conf.stat_ic = reduce_sub_group_count * 16;
    conf.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", 0, conf.stat_ic);
    conf.dispatch_reduce_stat.define_dim(
            "REDUCE_IC_GROUP", 1, utils::rnd_up(conf.ic, 16) / 16);
    CHECK(conf.dispatch_reduce_stat.vectorize_dim("REDUCE_STAT_IC", 16));
    conf.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    conf.dispatch_reduce_stat.generate();

    const int sp_pad = utils::rnd_up(conf.sp, conf.vect_size);
    conf.sp_tail = utils::rnd_dn(conf.sp, conf.vect_size);

    conf.dispatch = compute_engine->create_dispatch(data_mdw.md_);
    conf.dispatch.define_dim("MB", 0, conf.nn);
    conf.dispatch.define_dim("SP", 1,
            conf.nhwc_optimized ? conf.stat_sp_nblocks
                                : sp_pad / conf.vect_size);
    conf.dispatch.define_dim("IC", 2, conf.calc_stat_ic);

    CHECK(conf.dispatch.vectorize_dim("IC", 16));
    conf.dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("IC16", utils::rnd_up(conf.ic, 16));
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
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_DIFF_STATS", conf.calculate_diff_stats);
    kernel_ctx.define_int("DIFF_SCALESHIFT", conf.diff_scaleshift);
    kernel_ctx.define_int("DIFF_SCALE", conf.diff_scale);
    kernel_ctx.define_int("DIFF_SHIFT", conf.diff_shift);
    kernel_ctx.define_int("REDUCE_IC_SUB_GROUPS", conf.stat_ic / 16);
    kernel_ctx.define_int("USE_STATS_ONE_PASS", conf.use_stats_one_pass);
    kernel_ctx.define_int("NHWC_OPTIMIZED", conf.nhwc_optimized);

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch_calc_stat);
    def_dispatch(kernel_ctx, conf.dispatch_reduce_stat);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_batch_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void gen9_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (conf.calculate_stats) {
        size_t size_coeff = sizeof(double) / sizeof(float);
        size_t size = 2 * size_coeff * conf.reduce_stat_nblocks
                * utils::rnd_up(conf.ic, 16);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(key_bnorm_reduction, size,
                types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
        if (!conf.save_stats) {
            scratchpad.book(key_bnorm_tmp_mean, conf.ic,
                    types::data_type_size(data_type::f32),
                    OCL_BUFFER_ALIGNMENT);
            scratchpad.book(key_bnorm_tmp_var, conf.ic,
                    types::data_type_size(data_type::f32),
                    OCL_BUFFER_ALIGNMENT);
        }
    }
}

status_t gen9_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);

    auto &mean_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_MEAN, status);
    CHECK(status);

    auto &variance_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_VARIANCE, status);
    CHECK(status);

    auto &scaleshift = CTX_IN_STORAGE(
            conf.use_scale ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);

    auto &dst = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DST, status);
    CHECK(status);
    auto &ws = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_WORKSPACE, status);
    CHECK(status);

    std::unique_ptr<memory_storage_t> temp_reduce;
    std::unique_ptr<memory_storage_t> tmp_mean;
    std::unique_ptr<memory_storage_t> tmp_variance;
    if (conf.calculate_stats) {
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);

        if (!conf.save_stats) {
            tmp_mean = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_mean);
            tmp_variance = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_var);
        }
    }

    auto &mean = (conf.calculate_stats && !conf.save_stats) ? *tmp_mean : mean_;
    auto &variance = (conf.calculate_stats && !conf.save_stats) ? *tmp_variance
                                                                : variance_;

    if (conf.calculate_stats && !conf.use_stats_one_pass) {
        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);

        auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_mean, calculate_mean_kernel_,
                calc_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, mean);

        auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();

        status = parallel_for(ctx, nd_range_reduce_mean, reduce_mean_kernel_,
                reduce_mean_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);

        auto nd_range_calc_var = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_var,
                calculate_variance_kernel_, calc_var_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, variance);

        auto nd_range_reduce_var = conf.dispatch_reduce_stat.nd_range();

        status = parallel_for(ctx, nd_range_reduce_var, reduce_variance_kernel_,
                reduce_var_arg_list);
        if (status != status::success) return status;
    }
    if (conf.calculate_stats && conf.use_stats_one_pass) {
        compute::kernel_arg_list_t calc_mean_var_arg_list;
        calc_mean_var_arg_list.set(0, src);
        calc_mean_var_arg_list.set(1, *temp_reduce);

        auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_mean,
                calculate_mean_var_kernel_, calc_mean_var_arg_list);
        if (status != status::success) return status;

        compute::kernel_arg_list_t reduce_mean_var_arg_list;
        reduce_mean_var_arg_list.set(0, *temp_reduce);
        reduce_mean_var_arg_list.set(1, mean);
        reduce_mean_var_arg_list.set(2, variance);

        auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();

        status = parallel_for(ctx, nd_range_reduce_mean,
                reduce_mean_var_kernel_, reduce_mean_var_arg_list);
        if (status != status::success) return status;
    }
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, shift);
    arg_list.set(6, ws);
    arg_list.set(7, conf.eps);

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

status_t gen9_batch_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void gen9_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size
            = 2 * utils::rnd_up(conf.ic, 16) * (1 + conf.reduce_stat_nblocks);
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t gen9_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scaleshift = CTX_IN_STORAGE(
            conf.use_scale ? DNNL_ARG_SCALE : DNNL_ARG_SCALE_SHIFT);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);

    auto &diff_scaleshift_ = CTX_OUT_CLEAN_STORAGE(
            conf.diff_scale ? DNNL_ARG_DIFF_SCALE : DNNL_ARG_DIFF_SCALE_SHIFT,
            status);
    CHECK(status);
    auto &diff_shift_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    std::unique_ptr<memory_storage_t> temp_reduce;
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);

    auto &diff_scaleshift = (!conf.diff_scaleshift && !conf.diff_scale)
            ? *temp_reduce
            : diff_scaleshift_;
    auto &diff_shift = (!conf.diff_scaleshift && !conf.diff_shift)
            ? *temp_reduce
            : diff_shift_;

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);

    auto nd_range = conf.dispatch_calc_stat.nd_range();
    status = parallel_for(
            ctx, nd_range, calculate_stats_kernel_, calc_stats_arg_list);
    if (status != status::success) return status;

    compute::kernel_arg_list_t reduce_stats_arg_list;
    reduce_stats_arg_list.set(0, *temp_reduce);
    reduce_stats_arg_list.set(1, diff_scaleshift);
    reduce_stats_arg_list.set(2, diff_shift);
    reduce_stats_arg_list.set(3, variance);
    reduce_stats_arg_list.set(4, conf.eps);

    auto nd_range_reduce_stat = conf.dispatch_reduce_stat.nd_range();
    status = parallel_for(ctx, nd_range_reduce_stat, reduce_stats_kernel_,
            reduce_stats_arg_list);
    if (status != status::success) return status;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scaleshift);
    arg_list.set(5, ws);
    arg_list.set(6, diff_src);
    arg_list.set(7, diff_scaleshift);
    arg_list.set(8, diff_shift);
    arg_list.set(9, conf.eps);

    nd_range = conf.dispatch.nd_range();
    status = parallel_for(ctx, nd_range, bwd_kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
