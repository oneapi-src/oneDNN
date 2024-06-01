/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/ocl/bnorm/nhwc_reusable.hpp"
#include "gpu/intel/ocl/bnorm/bnorm_model.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace bn_lookup_table;
using namespace bn_utils;
using namespace bn_model;
using namespace bn_utils::kernel_id;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::gpu::intel::gpu_utils;

static status_t init_reusable_confs_basic(
        nhwc_reusable_bnorm_compile_params_t &cmpl_conf,
        nhwc_reusable_bnorm_runtime_params_t &rt_conf,
        const batch_normalization_pd_t *pd,
        const memory_desc_wrapper &data_mdw) {

    const batch_normalization_desc_t &bd = *pd->desc();
    cmpl_conf = utils::zero<decltype(cmpl_conf)>();

    cmpl_conf.data_type = data_mdw.data_type();

    cmpl_conf.use_scale = pd->use_scale();
    cmpl_conf.use_shift = pd->use_shift();
    cmpl_conf.is_training = pd->is_training();
    cmpl_conf.fuse_norm_relu = pd->fuse_norm_relu() || pd->fuse_norm_add_relu();
    cmpl_conf.fuse_norm_add_relu = pd->fuse_norm_add_relu();
    cmpl_conf.calculate_stats = !pd->stats_is_src();
    cmpl_conf.with_relu = pd->with_relu_post_op(pd->is_training());

    rt_conf.relu_negative_slope = cmpl_conf.with_relu ? pd->alpha() : 0.f;
    rt_conf.eps = bd.batch_norm_epsilon;
    cmpl_conf.with_leaky_relu
            = cmpl_conf.with_relu && rt_conf.relu_negative_slope != 0.f;

    return status::success;
}

static status_t final_set_rt_params(nhwc_bnorm_params_t &bn_conf,
        nhwc_reusable_bnorm_runtime_params_t &rt_conf) {
    rt_conf.ic_size = bn_conf.ic;
    rt_conf.ic_block = bn_conf.ic_block();
    rt_conf.sp_size = bn_conf.sp;
    rt_conf.update_sp_block = bn_conf.update_sp_block();
    rt_conf.update_sp_unroll = bn_conf.update_sp_unroll();
    rt_conf.stat_sp_block = bn_conf.stat_sp_block();
    rt_conf.reduce_stat_nblocks = bn_conf.reduce_stat_nblocks;
    rt_conf.reduce_ic_sub_groups = bn_conf.stat_ic / bn_conf.sub_group_size;
    rt_conf.sg_size = bn_conf.sub_group_size;
    rt_conf.use_fused_atomics_reduction = bn_conf.use_fused_atomics_reduction();
    rt_conf.calc_adj_lws = bn_conf.calc_adj_lws;

    // Switchers between kernels with or without private buffers.
    // Currently used for performance experiments/tuning
    // TODO: make it as part of perf model
    rt_conf.use_buffers_calc = dev_getenv("USE_BUFFERS_CALC", 0);
    rt_conf.use_buffers_norm = dev_getenv("USE_BUFFERS_NORM", 0);

    return status::success;
}

static status_t init_conf_common(nhwc_bnorm_params_t &bn_conf,
        nhwc_reusable_bnorm_compile_params_t &cmpl_conf,
        nhwc_reusable_bnorm_runtime_params_t &rt_conf,
        compute::dispatch_t &dispatch_calc_stat,
        compute::dispatch_t &dispatch_reduce_stat,
        compute::dispatch_t &dispatch, compute::dispatch_t &dispatch_reduce_aux,
        const batch_normalization_pd_t *pd, impl::engine_t *engine) {

    // This implementation is temporarly unavailable by default
    // TODO: remove the guard after performance tuning
    if (!dev_getenv("enable_bn_nhwc_reusable", 0)) return status::unimplemented;

    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());

    // init cmpl_conf, rt_conf
    CHECK(init_reusable_confs_basic(cmpl_conf, rt_conf, pd, data_mdw));
    // basic init bn_conf
    init_conf_basic(bn_conf, pd);

    // TODO: create flags() accessor that returns the correct type
    bn_conf.flags = (normalization_flags_t)pd->desc()->flags;

    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    // TODO: switch between reusable-nhwc and nhwc version, based on perf data

    // nhwc-optimized implemntation does not support ic tail processing yet
    // and was tuned for XeHPG+ only
    bool nhwc_optimized = bn_conf.ic % 16 == 0
            && data_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
            && gpu_arch >= compute::gpu_arch_t::xe_hpg;
    if (!nhwc_optimized) return status::unimplemented;

    const bool has_padding = !data_mdw.is_dense();
    if (has_padding) return status::unimplemented;

    // Due to intel_sub_group_write_uc requires 16-bytes alignment,
    // IC div by 8 tail processing is not applicable to fuse_norm_relu
    // and char data type.
    if (bn_conf.ic % 8 == 0 && bn_conf.ic % 16
            && (bn_conf.fuse_norm_relu || bn_conf.data_type == data_type::s8))
        return status::unimplemented;
    // IC tail processing performance boost is not obvious on arch < xe_hpc
    if (bn_conf.ic % 8 == 0 && bn_conf.ic % 16
            && gpu_arch < compute::gpu_arch_t::xe_hpc)
        return status::unimplemented;

    cmpl_conf.use_stats_one_pass = experimental::use_bnorm_stats_one_pass();
    bn_conf.use_stats_one_pass = cmpl_conf.use_stats_one_pass;

    // IC tail processing is not implemented yet for one pass algorithm
    // TODO: implement it, possible perf boost could be ~ 2x
    if (bn_conf.ic % 8 == 0 && bn_conf.ic % 16 && cmpl_conf.use_stats_one_pass)
        cmpl_conf.use_stats_one_pass = false;

    // Temporary for performance tuning. TODO: consider adding it to a perf model
    bn_conf.sub_group_size = dev_getenv("SG", 16);
    bn_conf.max_ic_block = dev_getenv("MAX_IC_BLOCK", 128);

    // reshape to xc
    bn_conf.sp = bn_conf.mb * bn_conf.id * bn_conf.ih * bn_conf.iw;

    // Default value is equal to OCL supported max vector size
    // but can be overridden due to performance reason.
    if (!bn_conf.max_vect_size_param().is_overridden())
        bn_conf.set_max_vect_size(8);

    auto default_conf = bn_conf;
    // Attempt to get tunable parameters from a lookup table
    // or from environment in tuning mode

    maybe_override_bn_conf_params(bn_conf, engine);

    // If lookup table entry uses atomics, use default conf instead.
    if (bn_conf.use_fused_atomics_reduction()
            && bn_conf.use_fused_atomics_reduction_param().is_overridden()
            && pd->attr()->deterministic_)
        bn_conf = default_conf;

    // Get non-overridden parameters, performance modeling way
    hw_params_t hw_params;
    init_hw_params(hw_params, engine);
    CHECK(get_params_by_model(bn_conf, pd, hw_params, true));

    cmpl_conf.vect_size = bn_conf.vect_size;
    cmpl_conf.sub_group_size = bn_conf.sub_group_size;
    cmpl_conf.max_ic_block = bn_conf.max_ic_block;

    // For performance debuging and analisys
    std::string prb_str = get_prb_desc_str(pd);
    std::string params_str = to_string(bn_conf);
    DPRINT_PARAMS(
            "prb_desc,%s,params,%s\n", prb_str.c_str(), params_str.c_str());

    bn_conf.sp_tail = rnd_dn(bn_conf.sp, bn_conf.vect_size);

    // Set dispatching

    dispatch_calc_stat = compute_engine->create_dispatch();
    CHECK(nhwc_bnorm_kernel_dispatching(
            calc_mean_ker, bn_conf, engine, dispatch_calc_stat));
    dispatch_reduce_stat = compute_engine->create_dispatch();
    CHECK(nhwc_bnorm_kernel_dispatching(reusable_reduce_stats_fwd_ker, bn_conf,
            engine, dispatch_reduce_stat));
    dispatch = compute_engine->create_dispatch(data_mdw.md_);
    CHECK(nhwc_bnorm_kernel_dispatching(
            default_fwd_ker, bn_conf, engine, dispatch));
    dispatch_reduce_aux = compute_engine->create_dispatch(data_mdw.md_);
    CHECK(nhwc_bnorm_kernel_dispatching(
            reduce_aux_init_ker, bn_conf, engine, dispatch_reduce_aux));

    CHECK(final_set_rt_params(bn_conf, rt_conf));

    return status::success;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const nhwc_reusable_bnorm_compile_params_t &cmpl_conf) {
    kernel_ctx.set_data_type(cmpl_conf.data_type);

    kernel_ctx.define_int("WITH_RELU", cmpl_conf.with_relu);
    if (cmpl_conf.with_leaky_relu) kernel_ctx.define_int("WITH_LEAKY_RELU", 1);

    kernel_ctx.define_int("IS_TRAINING", cmpl_conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", cmpl_conf.fuse_norm_relu);
    kernel_ctx.define_int("FUSE_BN_ADD_RELU", cmpl_conf.fuse_norm_add_relu);
    kernel_ctx.define_int("CALCULATE_STATS", cmpl_conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALE", cmpl_conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", cmpl_conf.use_shift);
    kernel_ctx.define_int("VECT_SIZE", cmpl_conf.vect_size);
    kernel_ctx.define_int("SUB_GROUP_SIZE", cmpl_conf.sub_group_size);
    kernel_ctx.add_option("-cl-std=CL2.0");
    if (cmpl_conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.define_int("MAX_IC_BLOCK", cmpl_conf.max_ic_block);
}

status_t nhwc_reusable_batch_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    return init_conf_common(bn_conf, cmpl_conf, rt_conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, dispatch_reduce_aux, this, engine);
}

compute::kernel_ctx_t
nhwc_reusable_bnorm_compile_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    init_kernel_ctx_common(kernel_ctx, *this);
    return kernel_ctx;
}

void nhwc_reusable_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (cmpl_conf.calculate_stats) {
        size_t reduce_size = static_cast<size_t>(2 * rt_conf.reduce_stat_nblocks
                * rnd_up(rt_conf.ic_size, bn_conf.sub_group_size));
        size_t stats_size = static_cast<size_t>(rt_conf.ic_size);
        size_t elsize = types::data_type_size(data_type::f32);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_bnorm_reduction,
                reduce_size, elsize, OCL_BUFFER_ALIGNMENT);

        if (!cmpl_conf.is_training) {
            scratchpad.book(memory_tracking::names::key_bnorm_tmp_mean,
                    stats_size, elsize, OCL_BUFFER_ALIGNMENT);
            scratchpad.book(memory_tracking::names::key_bnorm_tmp_var,
                    stats_size, elsize, OCL_BUFFER_ALIGNMENT);
        }
    }
}

static dim_t get_calc_slm_size(
        const nhwc_reusable_bnorm_compile_params_t &cmpl_conf,
        const nhwc_reusable_bnorm_runtime_params_t &rt_conf) {
    return rt_conf.use_fused_atomics_reduction
            ? (rt_conf.use_buffers_calc ? sizeof(float) * rt_conf.ic_block
                                    * rt_conf.calc_adj_lws[1]
                                        : sizeof(float) * cmpl_conf.vect_size
                                    * rt_conf.sg_size * rt_conf.calc_adj_lws[1])
            : 0;
}

status_t nhwc_reusable_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    const auto &cmpl_conf = pd()->cmpl_conf;
    const auto &rt_conf = pd()->rt_conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &src_add = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    // Set up mean and variance pointers
    std::unique_ptr<memory_storage_t> tmp_mean = nullptr;
    std::unique_ptr<memory_storage_t> tmp_variance = nullptr;
    std::unique_ptr<memory_storage_t> tmp_reduce = nullptr;

    auto &mean_ = cmpl_conf.calculate_stats ? CTX_OUT_STORAGE(DNNL_ARG_MEAN)
                                            : CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance_ = cmpl_conf.calculate_stats
            ? CTX_OUT_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_IN_STORAGE(DNNL_ARG_VARIANCE);

    if (cmpl_conf.calculate_stats) {
        tmp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);

        if (!cmpl_conf.is_training) {
            tmp_mean = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_mean);
            tmp_variance = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_var);
        }
    }
    auto &mean = (cmpl_conf.calculate_stats && !cmpl_conf.is_training)
            ? *tmp_mean
            : mean_;
    auto &variance = (cmpl_conf.calculate_stats && !cmpl_conf.is_training)
            ? *tmp_variance
            : variance_;

    if (cmpl_conf.calculate_stats && rt_conf.use_fused_atomics_reduction) {
        // Atomics-based reduction requires zeroing mean and variance
        // Single kernel runs faster than two compute_stream_t::fill
        compute::kernel_arg_list_t arg_list;
        arg_list.append(mean);
        arg_list.append(variance);
        arg_list.append(0.f);
        arg_list.append(into<dim_t>(0));
        arg_list.append(aux_use_regular);
        arg_list.append(aux_init_stage);
        arg_list.append(aux_fwd);

        auto nd_range = pd()->dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, kernels_[reduce_aux], arg_list);
        if (status != status::success) return status;
    }

    const dim_t calc_slm_size = get_calc_slm_size(cmpl_conf, rt_conf);

    if (cmpl_conf.calculate_stats && !cmpl_conf.use_stats_one_pass) {
        const dim_t local_sum_size = sizeof(float) * rt_conf.sg_size
                * rt_conf.reduce_ic_sub_groups;

        // mean
        compute::kernel_arg_list_t calc_mean_arg_list;

        calc_mean_arg_list.append(src);
        calc_mean_arg_list.append(*tmp_reduce);
        calc_mean_arg_list.append(mean);
        calc_mean_arg_list.append(rt_conf.ic_size);
        calc_mean_arg_list.append(rt_conf.ic_block);
        calc_mean_arg_list.append(rt_conf.sp_size);
        calc_mean_arg_list.append(rt_conf.stat_sp_block);
        calc_mean_arg_list.append(rt_conf.reduce_stat_nblocks);
        calc_mean_arg_list.append(
                into<int>(rt_conf.use_fused_atomics_reduction));
        calc_mean_arg_list.append(calc_slm_size, nullptr);

        auto nd_range_calc_mean = pd()->dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_mean,
                kernels_[rt_conf.use_buffers_calc ? calc_mean_buff : calc_mean],
                calc_mean_arg_list);
        if (status != status::success) return status;

        if (rt_conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.append(mean);
            arg_list.append(memory_storage_t::empty_storage());
            arg_list.append(0.f);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(aux_use_regular);
            arg_list.append(aux_finalize_stage);
            arg_list.append(aux_fwd);

            auto nd_range = pd()->dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_aux], arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.append(*tmp_reduce);
            arg_list.append(into<dim_t>(0));
            arg_list.append(mean);
            arg_list.append(rt_conf.ic_size);
            arg_list.append(rt_conf.reduce_ic_sub_groups);
            arg_list.append(rt_conf.reduce_stat_nblocks);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(local_sum_size, nullptr);
            auto nd_range = pd()->dispatch_reduce_stat.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_fwd_reg], arg_list);
            if (status != status::success) return status;
        }

        // variance
        compute::kernel_arg_list_t calc_var_arg_list;

        calc_var_arg_list.append(src);
        calc_var_arg_list.append(mean);
        calc_var_arg_list.append(*tmp_reduce);
        calc_var_arg_list.append(variance);
        calc_var_arg_list.append(rt_conf.ic_size);
        calc_var_arg_list.append(rt_conf.ic_block);
        calc_var_arg_list.append(rt_conf.sp_size);
        calc_var_arg_list.append(rt_conf.stat_sp_block);
        calc_var_arg_list.append(rt_conf.reduce_stat_nblocks);
        calc_var_arg_list.append(
                into<int>(rt_conf.use_fused_atomics_reduction));
        calc_var_arg_list.append(calc_slm_size, nullptr);

        auto nd_range_calc_var = pd()->dispatch_calc_stat.nd_range();

        status = parallel_for(ctx, nd_range_calc_var,
                kernels_[rt_conf.use_buffers_calc ? calc_var_buff : calc_var],
                calc_var_arg_list);
        if (status != status::success) return status;

        if (rt_conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.append(variance);
            arg_list.append(memory_storage_t::empty_storage());
            arg_list.append(0.f);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(aux_use_regular);
            arg_list.append(aux_finalize_stage);
            arg_list.append(aux_fwd);

            auto nd_range = pd()->dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_aux], arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            arg_list.append(*tmp_reduce);
            arg_list.append(rt_conf.ic_size * rt_conf.reduce_stat_nblocks);
            arg_list.append(variance);
            arg_list.append(rt_conf.ic_size);
            arg_list.append(rt_conf.reduce_ic_sub_groups);
            arg_list.append(rt_conf.reduce_stat_nblocks);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(local_sum_size, nullptr);

            auto nd_range = pd()->dispatch_reduce_stat.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_fwd_reg], arg_list);
            if (status != status::success) return status;
        }
    }

    if (cmpl_conf.calculate_stats && cmpl_conf.use_stats_one_pass) {

        compute::kernel_arg_list_t arg_list;

        arg_list.append(src);
        arg_list.append(*tmp_reduce);
        arg_list.append(mean);
        arg_list.append(variance);
        arg_list.append(rt_conf.ic_size);
        arg_list.append(rt_conf.ic_block);
        arg_list.append(rt_conf.sp_size);
        arg_list.append(rt_conf.stat_sp_block);
        arg_list.append(rt_conf.reduce_stat_nblocks);
        arg_list.append(into<int>(rt_conf.use_fused_atomics_reduction));
        arg_list.append(2 * calc_slm_size, nullptr);
        arg_list.append(2 * calc_slm_size, nullptr);

        auto nd_range = pd()->dispatch_calc_stat.nd_range();
        status = parallel_for(ctx, nd_range,
                kernels_[rt_conf.use_buffers_calc ? calc_mean_var_buff
                                                  : calc_mean_var],
                arg_list);
        if (status != status::success) return status;

        if (rt_conf.use_fused_atomics_reduction) {
            compute::kernel_arg_list_t arg_list;
            arg_list.append(mean);
            arg_list.append(variance);
            arg_list.append(0.f);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(aux_use_one_pass);
            arg_list.append(aux_finalize_stage);
            arg_list.append(aux_fwd);

            auto nd_range = pd()->dispatch_reduce_aux.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_aux], arg_list);
            if (status != status::success) return status;
        } else {
            compute::kernel_arg_list_t arg_list;
            const dim_t local_sum_size = 2 * sizeof(float) * rt_conf.sg_size
                    * rt_conf.reduce_ic_sub_groups;
            arg_list.append(*tmp_reduce);
            arg_list.append(mean);
            arg_list.append(variance);
            arg_list.append(rt_conf.ic_size);
            arg_list.append(rt_conf.reduce_ic_sub_groups);
            arg_list.append(rt_conf.reduce_stat_nblocks);
            arg_list.append(rt_conf.sp_size);
            arg_list.append(local_sum_size, nullptr); // local_sum
            arg_list.append(local_sum_size, nullptr); // local_sum_sq

            auto nd_range = pd()->dispatch_reduce_stat.nd_range();
            status = parallel_for(
                    ctx, nd_range, kernels_[reduce_fwd_1pass], arg_list);
            if (status != status::success) return status;
        }
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(mean);
    arg_list.append(variance);
    arg_list.append(dst);
    arg_list.append(scale);
    arg_list.append(shift);
    arg_list.append(ws);
    arg_list.append(rt_conf.eps);
    arg_list.append(src_add);
    arg_list.append(rt_conf.relu_negative_slope);
    arg_list.append(rt_conf.ic_size);
    arg_list.append(rt_conf.ic_block);
    arg_list.append(rt_conf.sp_size);
    arg_list.append(rt_conf.update_sp_block);

    auto nd_range = pd()->dispatch.nd_range();
    return parallel_for(ctx, nd_range,
            kernels_[rt_conf.use_buffers_norm ? norm_fwd_buff : norm_fwd],
            arg_list);
}

status_t nhwc_reusable_batch_normalization_bwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    return init_conf_common(bn_conf, cmpl_conf, rt_conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, dispatch_reduce_aux, this, engine);
}

void nhwc_reusable_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t elsize = types::data_type_size(data_type::f32);
    size_t size = rnd_up(rt_conf.ic_size, bn_conf.sub_group_size)
            * (1 + rt_conf.reduce_stat_nblocks);
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size, elsize,
            OCL_BUFFER_ALIGNMENT);
    scratchpad.book(memory_tracking::names::key_bnorm_reduction_shift, size,
            elsize, OCL_BUFFER_ALIGNMENT);
}

status_t nhwc_reusable_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const auto &cmpl_conf = pd()->cmpl_conf;
    const auto &rt_conf = pd()->rt_conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_src_add = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_1);
    auto &diff_scale_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE);
    auto &diff_shift_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SHIFT);

    std::unique_ptr<memory_storage_t> temp_reduce;
    std::unique_ptr<memory_storage_t> temp_reduce_shift;
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);
    temp_reduce_shift = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction_shift);

    auto &diff_scale = !cmpl_conf.use_scale ? *temp_reduce : diff_scale_;
    auto &diff_shift = !cmpl_conf.use_shift ? *temp_reduce_shift : diff_shift_;

    if (rt_conf.use_fused_atomics_reduction) {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(diff_scale);
        arg_list.append(diff_shift);
        arg_list.append(0.f);
        arg_list.append(into<dim_t>(0));
        arg_list.append(aux_use_regular);
        arg_list.append(aux_init_stage);
        arg_list.append(aux_bwd);

        auto nd_range = pd()->dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, kernels_[reduce_aux], arg_list);
        if (status != status::success) return status;
    }

    const dim_t calc_slm_size = get_calc_slm_size(cmpl_conf, rt_conf);

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.append(src);
    calc_stats_arg_list.append(mean);
    calc_stats_arg_list.append(diff_dst);
    calc_stats_arg_list.append(ws);
    calc_stats_arg_list.append(*temp_reduce);
    calc_stats_arg_list.append(*temp_reduce_shift);
    calc_stats_arg_list.append(diff_scale);
    calc_stats_arg_list.append(diff_shift);
    calc_stats_arg_list.append(rt_conf.ic_size);
    calc_stats_arg_list.append(rt_conf.ic_block);
    calc_stats_arg_list.append(rt_conf.sp_size);
    calc_stats_arg_list.append(rt_conf.stat_sp_block);
    calc_stats_arg_list.append(rt_conf.reduce_stat_nblocks);
    calc_stats_arg_list.append(into<int>(rt_conf.use_fused_atomics_reduction));
    calc_stats_arg_list.append(2 * calc_slm_size, nullptr);
    calc_stats_arg_list.append(calc_slm_size);

    auto calc_stats_nd_range = pd()->dispatch_calc_stat.nd_range();
    status = parallel_for(ctx, calc_stats_nd_range,
            kernels_[rt_conf.use_buffers_calc ? calc_stat_buff : calc_stat],
            calc_stats_arg_list);
    if (status != status::success) return status;

    if (rt_conf.use_fused_atomics_reduction) {
        compute::kernel_arg_list_t arg_list;
        arg_list.append(diff_scale);
        arg_list.append(variance);
        arg_list.append(rt_conf.eps);
        arg_list.append(into<dim_t>(0));
        arg_list.append(aux_use_regular);
        arg_list.append(aux_finalize_stage);
        arg_list.append(aux_bwd);

        auto nd_range = pd()->dispatch_reduce_aux.nd_range();
        status = parallel_for(ctx, nd_range, kernels_[reduce_aux], arg_list);
        if (status != status::success) return status;
    } else {
        const dim_t local_sum_size = sizeof(float) * rt_conf.sg_size
                * rt_conf.reduce_ic_sub_groups;

        compute::kernel_arg_list_t arg_list;
        arg_list.append(*temp_reduce);
        arg_list.append(*temp_reduce_shift);
        arg_list.append(diff_scale);
        arg_list.append(diff_shift);
        arg_list.append(variance);
        arg_list.append(rt_conf.eps);
        arg_list.append(rt_conf.ic_size);
        arg_list.append(rt_conf.reduce_ic_sub_groups);
        arg_list.append(rt_conf.reduce_stat_nblocks);
        arg_list.append(local_sum_size, nullptr);
        arg_list.append(local_sum_size, nullptr);

        auto nd_range = pd()->dispatch_reduce_stat.nd_range();
        status = parallel_for(ctx, nd_range, kernels_[reduce_stat], arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(mean);
    arg_list.append(variance);
    arg_list.append(diff_dst);
    arg_list.append(scale);
    arg_list.append(ws);
    arg_list.append(diff_src);
    arg_list.append(diff_scale);
    arg_list.append(diff_shift);
    arg_list.append(rt_conf.eps);
    arg_list.append(diff_src_add);
    arg_list.append(rt_conf.ic_size);
    arg_list.append(rt_conf.ic_block);
    arg_list.append(rt_conf.sp_size);
    arg_list.append(rt_conf.update_sp_block);

    auto nd_range = pd()->dispatch.nd_range();
    return parallel_for(ctx, nd_range,
            kernels_[rt_conf.use_buffers_norm ? norm_bwd_buff : norm_bwd],
            arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
