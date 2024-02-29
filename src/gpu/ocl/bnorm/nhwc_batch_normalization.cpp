/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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
#include "gpu/ocl/bnorm/nhwc_batch_normalization.hpp"
#include "common/experimental.hpp"
#include "common/utils.hpp"
#include "gpu/compute/utils.hpp"
#include "gpu/ocl/bnorm/bnorm_model.hpp"
#include "gpu/ocl/bnorm/bnorm_utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
using namespace bn_lookup_table;
using namespace bn_utils;
using namespace bn_model;
using namespace dnnl::impl::utils;

static size_t get_slm_buff_size(
        int ic_block, nhwc_bnorm_params_t &conf, const compute::range_t &lws) {
    // Returns size of SLM buffer of nhwc stat calculation kernels.
    const size_t base_size
            = div_up(ic_block, conf.sub_group_size) * lws.nelems();
    if (conf.use_stats_one_pass) {
        return 2 * base_size * 2 * sizeof(float);
    } else {
        return conf.is_forward ? base_size * sizeof(float)
                               : 2 * base_size * sizeof(float);
    }
}
// Local group size adjustment for calc_stat kernel
static void adjust_lws_calc_kernel(int ic_block, nhwc_bnorm_params_t &conf,
        compute::dispatch_t &dispatch, engine_t *engine,
        bool large_grf_mode = false) {
    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto eu_count = compute_engine->device_info()->eu_count();
    auto max_lws = compute_engine->device_info()->max_wg_size(large_grf_mode);
    auto eus_per_ss = compute_engine->device_info()->max_eus_per_wg();
    const int max_ss = div_up(eu_count, eus_per_ss);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    const int max_slm_size = compute::device_info_t::max_slm_size(gpu_arch);

    auto generated_nd = dispatch.nd_range();
    const compute::range_t &base_gws = generated_nd.global_range();
    const compute::range_t &base_lws = generated_nd.local_range();
    gpu_assert(base_lws) << "lws is missing";

    compute::range_t tuned_lws = {gpu_utils::into<size_t>(conf.sub_group_size),
            base_lws[1], base_lws[2]};
    compute::range_t curr_lws = tuned_lws;

    // The search is based on subslice utilization which calculated as the ratio
    // used_subslices / max_available_subslices.

    size_t best_val = 1;
    curr_lws[1] = 1;
    float best_ss_utilization = 0.0f, curr_ss_utilization;
    const int ss_util_limit = 2; // experimentally selected

    while (curr_lws[0] * curr_lws[1] * curr_lws[2] <= (size_t)max_lws
            && curr_lws[1] <= base_gws[1]
            && get_slm_buff_size(ic_block, conf, curr_lws)
                    <= (size_t)max_slm_size) {
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

    dispatch.set_lws(tuned_lws);
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

static int get_reduce_sub_group_count(
        const int reduce_stat_nblocks, const int sub_group_size) {
    int reduce_sub_group_count = 1;
    while (reduce_stat_nblocks % (2 * reduce_sub_group_count) == 0
            && 2 * reduce_sub_group_count * sub_group_size <= 256) {
        reduce_sub_group_count = reduce_sub_group_count * 2;
    }
    return reduce_sub_group_count;
}

inline int get_calc_stat_ic(int ic, int ic_block, int sg_size) {
    return div_up(ic, ic_block) * sg_size;
}

status_t nhwc_bnorm_kernel_dispatching(kernel_kind_t kernel,
        nhwc_bnorm_params_t &conf, engine_t *engine,
        compute::dispatch_t &dispatch) {

    conf.stat_sp_nblocks
            = rnd_up(conf.sp, conf.stat_sp_block()) / conf.stat_sp_block();
    conf.stat_sp_tail
            = rnd_dn(conf.sp, conf.stat_sp_block()) / conf.stat_sp_block();

    conf.update_sp_nblocks
            = rnd_up(conf.sp, conf.update_sp_block()) / conf.update_sp_block();
    conf.update_sp_tail
            = rnd_dn(conf.sp, conf.update_sp_block()) / conf.update_sp_block();
    conf.reduce_stat_nblocks = conf.stat_sp_nblocks;

    const int calc_stat_ic
            = get_calc_stat_ic(conf.ic, conf.ic_block(), conf.sub_group_size);

    switch (kernel) {
        case default_fwd_ker:
        case default_bwd_ker: {
            dispatch.define_dim("MB", 0, 1);
            dispatch.define_dim("SP", 1, conf.update_sp_nblocks);
            dispatch.define_dim_with_nesting_level("IC", 1024, calc_stat_ic);
            CHECK(dispatch.vectorize_dim("IC", conf.sub_group_size));
            dispatch.generate();
        } break;
        case calc_mean_ker:
        case calc_var_ker:
        case calc_mean_var_ker:
        case calc_stats_ker: {
            dispatch.define_dim("STAT_MB", 0, 1);
            dispatch.define_dim("STAT_SP", 1, conf.stat_sp_nblocks);
            dispatch.define_dim_with_nesting_level(
                    "STAT_IC", 1024, calc_stat_ic);
            CHECK(dispatch.vectorize_dim("STAT_IC", conf.sub_group_size));
            dispatch.set_kernel_attr_suffix("CALC");
            dispatch.generate();
            if (conf.use_fused_atomics_reduction()) {
                adjust_lws_calc_kernel(conf.ic_block(), conf, dispatch, engine);
            }
        } break;
        case reduce_stats_fwd_ker:
        case reduce_mean_var_ker:
        case reduce_stats_bwd_ker: {
            const int reduce_sub_group_count = get_reduce_sub_group_count(
                    conf.reduce_stat_nblocks, conf.sub_group_size);
            const int stat_ic = reduce_sub_group_count * conf.sub_group_size;
            conf.stat_ic = stat_ic;
            dispatch.define_dim("REDUCE_STAT_IC", 0, stat_ic);
            dispatch.define_dim(
                    "REDUCE_IC_GROUP", 1, div_up(conf.ic, conf.sub_group_size));
            CHECK(dispatch.vectorize_dim(
                    "REDUCE_STAT_IC", conf.sub_group_size));
            dispatch.set_kernel_attr_suffix("REDUCE");
            dispatch.generate();
        } break;
        case reduce_aux_init_ker:
        case reduce_aux_finalize_ker: {
            dispatch.define_dim("IC_AUX", 0, conf.ic);
            dispatch.set_kernel_attr_suffix("AUX");
            dispatch.generate();
        } break;
        default: assert(!"Wrong kernel"); return status::runtime_error;
    }
    return status::success;
}

// Get the best set of bnorm parameters based on performance model
static status_t get_params_by_model(nhwc_bnorm_params_t &conf,
        const batch_normalization_pd_t *pd, hw_params_t &hw_params) {

    // Create set of possible parameters
    std::vector<model_params_t> params;
    model_params_t p;
    p.ic_block = conf.sub_group_size;
    assert(conf.ic % conf.sub_group_size == 0);
    while (p.ic_block <= conf.ic) {
        if (conf.ic % p.ic_block == 0) {
            const int calc_stat_ic = get_calc_stat_ic(
                    conf.ic, p.ic_block, conf.sub_group_size);
            p.stat_sp_block = get_nhwc_sp_block_size(conf.sp, calc_stat_ic,
                    hw_params.eu_count, hw_params.threads_per_eu,
                    conf.sub_group_size);
            p.vect_size = get_nhwc_vect_size(p.ic_block, conf.max_vect_size());
            p.use_fused_atomics_reduction = 0;
            params.push_back(p);
            if (hw_params.gpu_arch >= compute::gpu_arch_t::xe_hpc
                    && !pd->attr()->deterministic_) {
                // atomics-based reduction on PVC+ only, perforformance reasons
                p.use_fused_atomics_reduction = 1;
                params.push_back(p);
            }
        }
        p.ic_block += conf.sub_group_size;
    }
    dump_params(params);

    // find the best set
    float best_expected_time = FLT_MAX;
    model_params_t best_params;
    for (auto &p : params) {

        // initialize kernel descriptors
        init_kernel_descriptors(p, conf, hw_params);
        // make estimations on execution time
        CHECK(make_perf_estimations(p, conf, hw_params));

        float exp_time = 0.0f;
        for (auto &desc : p.kernel_descs) {
            exp_time += desc.ncalls * desc.time_ns;
            exp_time += hw_params.host_overheads_per_kernel * desc.ncalls;

            DPRINT("%s:%s:%d p: %d %d %d : %s: %.1f(%.1f) \n", PRINTHEAD,
                    p.use_fused_atomics_reduction, p.ic_block, p.stat_sp_block,
                    to_string(desc.kernel).c_str(), desc.time_ns,
                    desc.time_ns * desc.ncalls);
        }
        DPRINT("%s:%s:%d p: %d %d %d : total expected ns = %.1f ( %.4f ms)\n",
                PRINTHEAD, p.use_fused_atomics_reduction, p.ic_block,
                p.stat_sp_block, exp_time, exp_time * 1e-6);

        if (exp_time < best_expected_time) {
            best_params = p;
            best_expected_time = exp_time;
        }
    }

#define SAVE_PARAM(name, val) \
    if (!conf.name##_param().is_overridden()) conf.set_##name(val);

    // save best params to conf
    conf.expected_time_ms = best_expected_time * 1e-6;
    // Some parameters can be set by tuning procedure or taken from table.
    // Other parametes to be set by model.
    SAVE_PARAM(use_fused_atomics_reduction,
            best_params.use_fused_atomics_reduction);
    if (!conf.ic_block_param().is_overridden()
            // guard for tuning, to use default value if overrrided one is wrong
            || (conf.ic_block_param().is_overridden()
                    && conf.ic_block() > conf.ic))
        conf.set_ic_block(best_params.ic_block);
    conf.calc_stat_ic
            = get_calc_stat_ic(conf.ic, conf.ic_block(), conf.sub_group_size);
    SAVE_PARAM(stat_sp_block, best_params.stat_sp_block);
    SAVE_PARAM(update_sp_block, conf.stat_sp_block());
    SAVE_PARAM(update_sp_unroll, 1);

#undef SAVE_PARAM

    conf.vect_size = get_nhwc_vect_size(
            conf.ic_block(), conf.max_vect_size(), conf.sub_group_size);
    // Guard for tuning and lookup table -
    // to use the default value if overrrided one is wrong
    const bool bad_update_sp_unroll
            = conf.update_sp_block() % conf.update_sp_unroll()
            || (conf.sp % conf.update_sp_block()) % conf.update_sp_unroll();
    if (conf.update_sp_unroll_param().is_overridden() && bad_update_sp_unroll) {
        conf.set_update_sp_unroll(1);
    } else {
        assert(!bad_update_sp_unroll);
    }
    return status::success;
}

static status_t init_conf_common(nhwc_bnorm_params_t &conf, offsets_t &off,
        compute::dispatch_t &dispatch_calc_stat,
        compute::dispatch_t &dispatch_reduce_stat,
        compute::dispatch_t &dispatch, compute::dispatch_t &dispatch_reduce_aux,
        const batch_normalization_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());

    init_conf_basic(conf, pd);
    set_offsets(data_mdw, off.src_off);

    // TODO: create flags() accessor that returns the correct type
    conf.flags = (normalization_flags_t)pd->desc()->flags;

    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    // nhwc-optimized implemntation does not support ic tail processing yet
    // and was tuned for XeHPG+ only
    bool nhwc_optimized = conf.ic % 16 == 0
            && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
            && gpu_arch >= compute::gpu_arch_t::xe_hpg;
    if (!nhwc_optimized) return status::unimplemented;

    conf.mb_block = 1;
    conf.is_nhwc = true;

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

    // reshape to xc
    conf.sp = conf.mb * conf.id * conf.ih * conf.iw;

    // Default value is equal to OCL supported max vector size
    // but can be overridden due to performance reason.
    if (!conf.max_vect_size_param().is_overridden()) conf.set_max_vect_size(8);

    auto default_conf = conf;
    // Attempt to get tunable parameters from a lookup table
    // or from environment in tuning mode
    maybe_override_bn_conf_params(conf, engine);

    // If lookup table entry uses atomics, use default conf instead.
    if (conf.use_fused_atomics_reduction()
            && conf.use_fused_atomics_reduction_param().is_overridden()
            && pd->attr()->deterministic_)
        conf = default_conf;

    // Get non-overridden parameters, performance modeling way
    hw_params_t hw_params;
    init_hw_params(hw_params, engine);
    CHECK(get_params_by_model(conf, pd, hw_params));

    // For performance debuging and analisys
    std::string prb_str = get_prb_desc_str(pd);
    std::string params_str = to_string(conf);
    DPRINT_PARAMS(
            "prb_desc,%s,params,%s\n", prb_str.c_str(), params_str.c_str());

    conf.sp_tail = rnd_dn(conf.sp, conf.vect_size);

    // Set dispatching
    dispatch_calc_stat = compute_engine->create_dispatch();
    CHECK(nhwc_bnorm_kernel_dispatching(
            calc_mean_ker, conf, engine, dispatch_calc_stat));
    dispatch_reduce_stat = compute_engine->create_dispatch();
    CHECK(nhwc_bnorm_kernel_dispatching(
            reduce_stats_fwd_ker, conf, engine, dispatch_reduce_stat));

    dispatch = compute_engine->create_dispatch(data_mdw.md_);
    CHECK(nhwc_bnorm_kernel_dispatching(
            default_fwd_ker, conf, engine, dispatch));

    dispatch_reduce_aux = compute_engine->create_dispatch(data_mdw.md_);
    CHECK(nhwc_bnorm_kernel_dispatching(
            reduce_aux_init_ker, conf, engine, dispatch_reduce_aux));

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const nhwc_bnorm_params_t &conf,
        const compute::dispatch_t &dispatch_calc_stat,
        const compute::dispatch_t &dispatch_reduce_stat,
        const compute::dispatch_t &dispatch,
        const compute::dispatch_t &dispatch_reduce_aux, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("PADDED_IC", rnd_up(conf.ic, conf.sub_group_size));
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block());

    kernel_ctx.define_int("SP", conf.sp);
    kernel_ctx.define_int("SP_TAIL", conf.sp_tail);
    kernel_ctx.define_int("VECT_SIZE", conf.vect_size);

    kernel_ctx.define_int("STAT_SP_BLOCK", conf.stat_sp_block());
    kernel_ctx.define_int("UPDATE_SP_BLOCK", conf.update_sp_block());
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
    kernel_ctx.define_int("NHWC_OPTIMIZED", true);
    kernel_ctx.define_int("SG_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("UPDATE_SP_UNROLL", conf.update_sp_unroll());
    kernel_ctx.define_int(
            "FUSED_ATOMICS_REDUCTION", conf.use_fused_atomics_reduction());
    kernel_ctx.define_int("USE_WORKAROUND", conf.use_workaround);

    kernel_ctx.add_option("-cl-std=CL2.0");
    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    def_dispatch(kernel_ctx, dispatch_calc_stat);
    def_dispatch(kernel_ctx, dispatch_reduce_stat);
    def_dispatch(kernel_ctx, dispatch_reduce_aux);
    def_dispatch(kernel_ctx, dispatch);

    return status::success;
}

status_t nhwc_batch_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, dispatch_calc_stat, dispatch_reduce_stat,
            dispatch, dispatch_reduce_aux, this, engine);
}

status_t nhwc_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, dispatch_reduce_aux, off);
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

    if (conf.calculate_stats && conf.use_fused_atomics_reduction()) {
        // Atomics-based reduction requires zeroing mean and variance
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, mean);
        arg_list.set(1, variance);

        auto nd_range = pd()->dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, reduce_init_kernel_, arg_list);
        if (status != status::success) return status;
    }

    if (conf.calculate_stats && !conf.use_stats_one_pass) {
        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);
        calc_mean_arg_list.set(2, mean);

        auto nd_range_calc_mean = pd()->dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_mean, calculate_mean_kernel_,
                calc_mean_arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction()) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, mean);
            auto nd_range = pd()->dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, mean);

            auto nd_range_reduce_mean = pd()->dispatch_reduce_stat.nd_range();

            status = parallel_for(
                    ctx, nd_range_reduce_mean, reduce_mean_kernel_, arg_list);
            if (status != status::success) return status;
        }

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);
        calc_var_arg_list.set(3, variance);

        auto nd_range_calc_var = pd()->dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_var,
                calculate_variance_kernel_, calc_var_arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction()) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, variance);
            auto nd_range = pd()->dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, variance);

            auto nd_range_reduce_var = pd()->dispatch_reduce_stat.nd_range();

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

        auto nd_range_calc_mean = pd()->dispatch_calc_stat.nd_range();

        status = parallel_for(
                ctx, nd_range_calc_mean, calculate_mean_var_kernel_, arg_list);
        if (status != status::success) return status;

        if (conf.use_fused_atomics_reduction()) {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, mean);
            arg_list.set(1, variance);
            auto nd_range_reduce_final = pd()->dispatch_reduce_aux.nd_range();

            status = parallel_for(
                    ctx, nd_range_reduce_final, reduce_final_kernel_, arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.set(0, *temp_reduce);
            arg_list.set(1, mean);
            arg_list.set(2, variance);

            auto nd_range_reduce_mean = pd()->dispatch_reduce_stat.nd_range();

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

    auto nd_range = pd()->dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

status_t nhwc_batch_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, dispatch_calc_stat, dispatch_reduce_stat,
            dispatch, dispatch_reduce_aux, this, engine);
}

status_t nhwc_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, dispatch_reduce_aux, off);
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

    if (conf.use_fused_atomics_reduction()) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_scale);
        arg_list.set(1, diff_shift);

        auto nd_range_reduce_init = pd()->dispatch_reduce_aux.nd_range();
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

    auto nd_range = pd()->dispatch_calc_stat.nd_range();
    status = parallel_for(
            ctx, nd_range, calculate_stats_kernel_, calc_stats_arg_list);
    if (status != status::success) return status;

    if (conf.use_fused_atomics_reduction()) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_scale);
        arg_list.set(1, variance);
        arg_list.set(2, conf.eps);
        auto nd_range = pd()->dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, reduce_final_kernel_, arg_list);
        if (status != status::success) return status;
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, *temp_reduce);
        arg_list.set(1, diff_scale);
        arg_list.set(2, diff_shift);
        arg_list.set(3, variance);
        arg_list.set(4, conf.eps);

        auto nd_range_reduce_stat = pd()->dispatch_reduce_stat.nd_range();
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

    nd_range = pd()->dispatch.nd_range();
    status = parallel_for(ctx, nd_range, bwd_kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
