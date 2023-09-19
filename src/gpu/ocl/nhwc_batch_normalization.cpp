/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "gpu/ocl/nhwc_batch_normalization.hpp"
#include "gpu/ocl/batch_normalization_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
using namespace bn_lookup_table;
using namespace bn_utils;
using namespace dnnl::impl::utils;

static bool use_fused_atomics_reduction(bnorm_conf_t &conf, engine_t *engine) {
    // Currently the fused atomics reduction is targeting to PVC only.
    // Heuristics experimentally selected, based on PVC perf data
    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    const size_t sp = conf.mb * conf.id * conf.ih * conf.iw;
    return gpu_arch >= compute::gpu_arch_t::xe_hpc
            && conf.ic % conf.sub_group_size == 0 && sp / conf.ic > 40;
}

static size_t get_slm_buff_size(bnorm_conf_t &conf, size_t *lws) {
    // Returns size of SLM buffer of nhwc stat calculation kernels.
    const size_t base_size = div_up(conf.ic_block, conf.sub_group_size) * lws[0]
            * lws[1] * lws[2];
    if (conf.use_stats_one_pass) {
        return 2 * base_size * 2 * sizeof(float);
    } else {
        return conf.is_forward ? base_size * sizeof(float)
                               : 2 * base_size * sizeof(float);
    }
}
// Local group size adjustment.
static void adjust_lws_calc_kernel(
        bnorm_conf_t &conf, engine_t *engine, bool large_grf_mode) {
    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto eu_count = compute_engine->device_info()->eu_count();
    auto max_lws = compute_engine->device_info()->max_wg_size(large_grf_mode);
    auto eus_per_ss = compute_engine->device_info()->max_eus_per_wg();
    const int max_ss = div_up(eu_count, eus_per_ss);

    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    const int max_slm_size = compute::device_info_t::max_slm_size(gpu_arch);
    auto generated_nd = conf.dispatch_calc_stat.nd_range();
    const size_t *base_gws = generated_nd.global_range();
    const size_t *base_lws = generated_nd.local_range();

    size_t tuned_lws[3], curr_lws[3];
    curr_lws[0] = tuned_lws[0] = conf.sub_group_size; // Assuming IC is dim 0
    curr_lws[1] = tuned_lws[1] = base_lws[1];
    curr_lws[2] = tuned_lws[2] = base_lws[2];

    // The search is based on subslice utilization which calculated as the ratio
    // used_subslices / max_available_subslices.

    size_t best_val = 1;
    curr_lws[1] = 1;
    float best_ss_utilization = 0.0f, curr_ss_utilization;
    const int ss_util_limit = 2; // experimentally selected

    while (curr_lws[0] * curr_lws[1] * curr_lws[2] <= (size_t)max_lws
            && curr_lws[1] <= base_gws[1]
            && get_slm_buff_size(conf, curr_lws) <= (size_t)max_slm_size) {
        if (base_gws[1] % curr_lws[1]) {
            curr_lws[1]++;
            continue;
        }
        tuned_lws[1] = curr_lws[1];
        curr_ss_utilization = get_ss_utilization(max_ss, base_gws, tuned_lws);

        if (curr_ss_utilization > best_ss_utilization
                && curr_ss_utilization < (float)ss_util_limit) {
            best_ss_utilization = curr_ss_utilization;
            best_val = curr_lws[1];
        }
        curr_lws[1]++;
    }
    tuned_lws[1] = best_val;

    conf.dispatch_calc_stat.set_lws(tuned_lws);
}

static int get_nhwc_ic_block(int ic, int sg_size, int max_ocl_vect_size = 8) {
    const int nblocks = ic / (max_ocl_vect_size * sg_size);
    return nblocks < 2 || (ic / nblocks) % sg_size ? ic : ic / nblocks;
}

static int get_nhwc_vect_size(int ic, int max_vect_size, int simd = 16) {
    int vect_size = max_vect_size;
    while (true) {
        if (ic / (vect_size * simd)) return vect_size;
        vect_size /= 2;
    }
    return 1;
}

static int get_nhwc_sp_block_size(
        int sp, int ic_dim, int eu_count, int threads_per_eu, int simd = 16) {

    float efficiency_thr = 0.0f;
    float efficiency_peak_eu_thr = 0.0f;
    int block_size_thr = 1;
    int block_size_peak_eu_thr = 1;
    int curr_block_size = sp;
    int nthr_mul = 1;
    const int ic_nsg = ic_dim / simd; // number of subgroups by ic dim

    // The search is based on threads wave efficiency.
    // Higher priority for cases with peak EUs utilization.
    while (nthr_mul <= 32) {
        const int nthr = nthr_mul * eu_count;
        curr_block_size = div_up(sp * ic_nsg, nthr);
        const int nblock = div_up(sp, curr_block_size);
        const int nthr_gen = nblock * ic_nsg;

        const float curr_efficiency_eus
                = (float)nthr_gen / rnd_up(nthr_gen, eu_count);
        const float curr_efficiency_thr
                = (float)nthr_gen / rnd_up(nthr_gen, eu_count * threads_per_eu);

        if (curr_efficiency_thr > efficiency_thr) {
            efficiency_thr = curr_efficiency_thr;
            block_size_thr = curr_block_size;
        }
        if (curr_efficiency_eus == 1
                && curr_efficiency_thr > efficiency_peak_eu_thr) {
            efficiency_peak_eu_thr = curr_efficiency_thr;
            block_size_peak_eu_thr = curr_block_size;
        }
        nthr_mul++;
    }
    if (efficiency_peak_eu_thr > 0.0f) return block_size_peak_eu_thr;
    return block_size_thr;
}

static status_t init_conf_common(bnorm_conf_t &conf, offsets_t &off,
        const batch_normalization_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());

    init_conf_basic(conf, pd);
    set_offsets(data_mdw, off.src_off);

    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    // nhwc-optimized implemntation does not support ic tail processing yet
    // and was tuned for XeHPG+ only
    conf.nhwc_optimized = conf.ic % 16 == 0
            && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
            && gpu_arch >= compute::gpu_arch_t::xe_hpg;
    if (!conf.nhwc_optimized) return status::unimplemented;

    conf.mb_block = 1;

    const bool has_padding = !data_mdw.is_dense();
    if (has_padding) return status::unimplemented;

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

    // Compiler issue workaround
    // TODO: remove it after fixing the issue
    conf.use_workaround = conf.data_type == data_type::f32
            && gpu_arch == compute::gpu_arch_t::xe_hpg;
    conf.sub_group_size = 16;

    init_flags_lookup_table(conf.flags, pd);

    // Default value is equal to OCL supported max vector size
    // but can be overridden due to performance reason.
    conf.max_vect_size = 8;

    // Attempt to get tunable parameters from a lookup table
    // or from environment in tuning mode
    maybe_override_bn_conf_params(conf, engine);

    if (!conf.is_overrided_use_fused_atomics_reduction) {
        conf.use_fused_atomics_reduction
                = use_fused_atomics_reduction(conf, engine);
    }

    if (!conf.is_overrided_ic_block) {
        conf.ic_block = get_nhwc_ic_block(
                rnd_up(conf.ic, conf.sub_group_size), conf.sub_group_size);
    }

    // reshape to xc
    conf.sp = conf.mb * conf.id * conf.ih * conf.iw;

    conf.calc_stat_ic = div_up(conf.ic, conf.ic_block) * conf.sub_group_size;

    auto eu_count = compute_engine->device_info()->eu_count();
    auto threads_per_eu
            = compute::device_info_t::threads_per_eu(gpu_arch, false);

    if (!conf.is_overrided_stat_sp_block) {
        conf.stat_sp_block = get_nhwc_sp_block_size(conf.sp, conf.calc_stat_ic,
                eu_count, threads_per_eu, conf.sub_group_size);
    }

    if (!conf.is_overrided_update_sp_block) {
        conf.update_sp_block = conf.stat_sp_block;
    }

    if (!conf.is_overrided_update_sp_unroll) conf.update_sp_unroll = 1;
    assert(conf.update_sp_block % conf.update_sp_unroll == 0);
    assert((conf.sp % conf.update_sp_block) % conf.update_sp_unroll == 0);

    conf.stat_sp_nblocks
            = rnd_up(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;
    conf.stat_sp_tail
            = rnd_dn(conf.sp, conf.stat_sp_block) / conf.stat_sp_block;

    conf.update_sp_nblocks
            = rnd_up(conf.sp, conf.update_sp_block) / conf.update_sp_block;
    conf.update_sp_tail
            = rnd_dn(conf.sp, conf.update_sp_block) / conf.update_sp_block;

    conf.reduce_stat_nblocks = conf.stat_sp_nblocks;

    conf.vect_size = get_nhwc_vect_size(
            conf.ic_block, conf.max_vect_size, conf.sub_group_size);

    conf.dispatch_calc_stat = compute_engine->create_dispatch();
    conf.dispatch_calc_stat.define_dim("STAT_MB", 0, 1);
    conf.dispatch_calc_stat.define_dim("STAT_SP", 1, conf.stat_sp_nblocks);
    conf.dispatch_calc_stat.define_dim_with_nesting_level(
            "STAT_IC", 1024, conf.calc_stat_ic);
    CHECK(conf.dispatch_calc_stat.vectorize_dim(
            "STAT_IC", conf.sub_group_size));
    conf.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    conf.dispatch_calc_stat.generate();
    if (conf.use_fused_atomics_reduction) {
        auto *gpu_attr
                = downcast<gpu_primitive_attr_t *>(pd->attr()->gpu_attr_.get());
        bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
        adjust_lws_calc_kernel(conf, compute_engine, large_grf_mode);

        if (!compute_engine->mayiuse(
                    compute::device_ext_t::ext_float_atomics)) {
            return status::unimplemented;
        }
    }

    conf.dispatch_reduce_stat = compute_engine->create_dispatch();
    int reduce_sub_group_count = 1;
    while (conf.reduce_stat_nblocks % (2 * reduce_sub_group_count) == 0
            && 2 * reduce_sub_group_count * conf.sub_group_size <= 256) {
        reduce_sub_group_count = reduce_sub_group_count * 2;
    }
    conf.stat_ic = reduce_sub_group_count * conf.sub_group_size;
    conf.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", 0, conf.stat_ic);
    conf.dispatch_reduce_stat.define_dim("REDUCE_IC_GROUP", 1,
            rnd_up(conf.ic, conf.sub_group_size) / conf.sub_group_size);
    CHECK(conf.dispatch_reduce_stat.vectorize_dim(
            "REDUCE_STAT_IC", conf.sub_group_size));
    conf.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    conf.dispatch_reduce_stat.generate();

    conf.sp_tail = rnd_dn(conf.sp, conf.vect_size);

    conf.dispatch = compute_engine->create_dispatch(data_mdw.md_);
    conf.dispatch.define_dim("MB", 0, 1);
    conf.dispatch.define_dim("SP", 1, conf.update_sp_nblocks);
    conf.dispatch.define_dim_with_nesting_level("IC", 1024, conf.calc_stat_ic);
    CHECK(conf.dispatch.vectorize_dim("IC", conf.sub_group_size));
    conf.dispatch.generate();

    conf.dispatch_reduce_aux = compute_engine->create_dispatch(data_mdw.md_);
    conf.dispatch_reduce_aux.define_dim("IC_AUX", 0, conf.ic);
    conf.dispatch_reduce_aux.set_kernel_attr_suffix("AUX");
    conf.dispatch_reduce_aux.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("PADDED_IC", rnd_up(conf.ic, conf.sub_group_size));
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("SP", conf.sp);
    kernel_ctx.define_int("SP_TAIL", conf.sp_tail);
    kernel_ctx.define_int("VECT_SIZE", conf.vect_size);

    kernel_ctx.define_int("STAT_SP_BLOCK", conf.stat_sp_block);
    kernel_ctx.define_int("UPDATE_SP_BLOCK", conf.update_sp_block);
    kernel_ctx.define_int("STAT_SP_NBLOCKS", conf.stat_sp_nblocks);
    kernel_ctx.define_int("STAT_SP_TAIL", conf.stat_sp_tail);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", conf.reduce_stat_nblocks);

    if (conf.is_forward)
        kernel_ctx.define_int("IS_FWD", 1);
    else if (conf.is_backward)
        kernel_ctx.define_int("IS_BWD", 1);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    if (conf.with_relu && conf.relu_negative_slope != 0.f)
        kernel_ctx.define_int("WITH_LEAKY_RELU", 1);

    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("FUSE_BN_ADD_RELU", conf.fuse_norm_add_relu);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_DIFF_STATS", conf.calculate_diff_stats);
    kernel_ctx.define_int("DIFF_SCALE", conf.diff_scale);
    kernel_ctx.define_int("DIFF_SHIFT", conf.diff_shift);
    kernel_ctx.define_int(
            "REDUCE_IC_SUB_GROUPS", conf.stat_ic / conf.sub_group_size);
    kernel_ctx.define_int("USE_STATS_ONE_PASS", conf.use_stats_one_pass);
    kernel_ctx.define_int("NHWC_OPTIMIZED", conf.nhwc_optimized);
    kernel_ctx.define_int("SG_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("UPDATE_SP_UNROLL", conf.update_sp_unroll);
    kernel_ctx.define_int(
            "FUSED_ATOMICS_REDUCTION", conf.use_fused_atomics_reduction);
    kernel_ctx.define_int("USE_WORKAROUND", conf.use_workaround);

    kernel_ctx.add_option("-cl-std=CL2.0");
    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch_calc_stat);
    def_dispatch(kernel_ctx, conf.dispatch_reduce_stat);
    def_dispatch(kernel_ctx, conf.dispatch_reduce_aux);
    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t nhwc_batch_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t nhwc_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void nhwc_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (conf.calculate_stats) {
        size_t size_coeff = sizeof(double) / sizeof(float);
        size_t size = 2 * size_coeff * conf.reduce_stat_nblocks
                * rnd_up(conf.ic, conf.sub_group_size);

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

status_t nhwc_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &src_add = CTX_IN_STORAGE(DNNL_ARG_SRC_1);

    auto &mean_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_MEAN, status);
    CHECK(status);

    auto &variance_ = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_OUT_CLEAN_STORAGE(DNNL_ARG_VARIANCE, status);
    CHECK(status);

    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
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

    if (conf.calculate_stats && conf.use_fused_atomics_reduction) {
        // Atomics-based reduction requires zeroing mean and variance
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, mean);
        arg_list.set(1, variance);

        auto nd_range = conf.dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, reduce_init_kernel_, arg_list);
        if (status != status::success) return status;
    }

    if (conf.calculate_stats && !conf.use_stats_one_pass) {
        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);
        calc_mean_arg_list.set(2, mean);

        auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_mean, calculate_mean_kernel_,
                calc_mean_arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, mean);
            auto nd_range = conf.dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, mean);

            auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();

            status = parallel_for(
                    ctx, nd_range_reduce_mean, reduce_mean_kernel_, arg_list);
            if (status != status::success) return status;
        }

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);
        calc_var_arg_list.set(3, variance);

        auto nd_range_calc_var = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_var,
                calculate_variance_kernel_, calc_var_arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, variance);
            auto nd_range = conf.dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, variance);

            auto nd_range_reduce_var = conf.dispatch_reduce_stat.nd_range();

            status = parallel_for(ctx, nd_range_reduce_var,
                    reduce_variance_kernel_, arg_list);
            if (status != status::success) return status;
        }
    }
    if (conf.calculate_stats && conf.use_stats_one_pass) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, *temp_reduce);
        arg_list.set(2, mean);
        arg_list.set(3, variance);

        auto nd_range_calc_mean = conf.dispatch_calc_stat.nd_range();

        status = parallel_for(
                ctx, nd_range_calc_mean, calculate_mean_var_kernel_, arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, mean);
            arg_list.set(1, variance);
            auto nd_range_reduce_final = conf.dispatch_reduce_aux.nd_range();

            status = parallel_for(
                    ctx, nd_range_reduce_final, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, mean);
            arg_list.set(2, variance);

            auto nd_range_reduce_mean = conf.dispatch_reduce_stat.nd_range();

            status = parallel_for(ctx, nd_range_reduce_mean,
                    reduce_mean_var_kernel_, arg_list);
            if (status != status::success) return status;
        }
    }
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, shift);
    arg_list.set(6, ws);
    arg_list.set(7, conf.eps);
    arg_list.set(8, src_add);
    arg_list.set(9, conf.relu_negative_slope);

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

status_t nhwc_batch_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t nhwc_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

void nhwc_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size = 2 * rnd_up(conf.ic, conf.sub_group_size)
            * (1 + conf.reduce_stat_nblocks);
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t nhwc_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    auto &diff_src_add = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC_1, status);
    CHECK(status);

    auto &diff_scale_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SCALE, status);
    CHECK(status);
    auto &diff_shift_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    std::unique_ptr<memory_storage_t> temp_reduce;
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);

    auto &diff_scale = !conf.diff_scale ? *temp_reduce : diff_scale_;
    auto &diff_shift = !conf.diff_shift ? *temp_reduce : diff_shift_;

    if (conf.use_fused_atomics_reduction) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_scale);
        arg_list.set(1, diff_shift);

        auto nd_range_reduce_init = conf.dispatch_reduce_aux.nd_range();
        status = parallel_for(
                ctx, nd_range_reduce_init, reduce_init_kernel_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);
    calc_stats_arg_list.set(5, diff_scale);
    calc_stats_arg_list.set(6, diff_shift);

    auto nd_range = conf.dispatch_calc_stat.nd_range();
    status = parallel_for(
            ctx, nd_range, calculate_stats_kernel_, calc_stats_arg_list);
    if (status != status::success) return status;

    if (conf.use_fused_atomics_reduction) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_scale);
        arg_list.set(1, variance);
        arg_list.set(2, conf.eps);
        auto nd_range = conf.dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, reduce_final_kernel_, arg_list);
        if (status != status::success) return status;
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, *temp_reduce);
        arg_list.set(1, diff_scale);
        arg_list.set(2, diff_shift);
        arg_list.set(3, variance);
        arg_list.set(4, conf.eps);

        auto nd_range_reduce_stat = conf.dispatch_reduce_stat.nd_range();
        status = parallel_for(
                ctx, nd_range_reduce_stat, reduce_stats_kernel_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scale);
    arg_list.set(5, ws);
    arg_list.set(6, diff_src);
    arg_list.set(7, diff_scale);
    arg_list.set(8, diff_shift);
    arg_list.set(9, conf.eps);
    arg_list.set(10, diff_src_add);

    nd_range = conf.dispatch.nd_range();
    status = parallel_for(ctx, nd_range, bwd_kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
