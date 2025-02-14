/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/intel/ocl/bnorm/ref_batch_normalization.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/ocl/utils.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static void init_calculate_stats_conf(bnorm_conf_t &conf,
        compute::dispatch_t &dispatch_calc_stat,
        compute::dispatch_t &dispatch_reduce_stat, impl::engine_t *engine,
        const memory_desc_wrapper &data_mdw) {
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const int ndims = conf.ndims;
    dispatch_calc_stat = compute_engine->create_dispatch(data_mdw.md_);
    dim_t calc_dims[5];
    auto &dims = data_mdw.dims();
    calc_dims[0] = dims[0];
    calc_dims[1] = dims[1];
    calc_dims[2] = (ndims < 5) ? 1 : dims[ndims - 3];
    calc_dims[3] = (ndims < 4) ? 1 : dims[ndims - 2];
    calc_dims[4] = (ndims < 3) ? 1 : dims[ndims - 1];
    int reduce_dim_idx = 0;
    for (int i = 2; i < 5; i++) {
        if (calc_dims[i] > calc_dims[reduce_dim_idx]) { reduce_dim_idx = i; }
    }
    conf.reduce_dim = calc_dims[reduce_dim_idx];
    conf.reduce_dim_idx = reduce_dim_idx;
    const std::string dim_names[5]
            = {"STAT_MB", "STAT_IC", "STAT_ID", "STAT_IH", "STAT_IW"};

    // Whole reduce dimension will be handled by single work item.
    calc_dims[reduce_dim_idx] = 1;

    conf.stat_ic = utils::array_product(calc_dims, 5);
    dispatch_calc_stat.define_dim(dim_names[0], 0, calc_dims[0]);
    dispatch_calc_stat.define_dim(dim_names[1], 1, calc_dims[1]);
    dispatch_calc_stat.define_dim(
            dim_names[2], nstl::max(1, ndims - 3), calc_dims[2]);
    dispatch_calc_stat.define_dim(
            dim_names[3], nstl::max(1, ndims - 2), calc_dims[3]);
    dispatch_calc_stat.define_dim(
            dim_names[4], nstl::max(1, ndims - 1), calc_dims[4]);

    dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    dispatch_calc_stat.generate();

    dispatch_reduce_stat = compute_engine->create_dispatch();
    dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
    dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    dispatch_reduce_stat.generate();
}

static void init_conf_common(bnorm_conf_t &conf, compute::dispatch_t &dispatch,
        offsets_t &off, const batch_normalization_pd_t *pd,
        const memory_desc_wrapper &data_mdw, impl::engine_t *engine) {

    const batch_normalization_desc_t &bd = *pd->desc();
    const int ndims = data_mdw.ndims();

    conf = utils::zero<decltype(conf)>();
    conf.data_type = data_mdw.data_type();

    conf.ndims = ndims;
    conf.mb = data_mdw.dims()[0];
    conf.ic = data_mdw.dims()[1];
    conf.id = (ndims >= 5) ? data_mdw.dims()[ndims - 3] : 1;
    conf.ih = (ndims >= 4) ? data_mdw.dims()[ndims - 2] : 1;
    conf.iw = (ndims >= 3) ? data_mdw.dims()[ndims - 1] : 1;

    conf.is_forward = pd->is_fwd();
    conf.is_backward = !pd->is_fwd();

    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.save_stats = pd->is_training();
    conf.is_training = pd->is_training();
    conf.fuse_norm_relu = pd->fuse_norm_relu() || pd->fuse_norm_add_relu();
    conf.fuse_norm_add_relu = pd->fuse_norm_add_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op(pd->is_training());
    conf.relu_negative_slope = conf.with_relu ? pd->alpha() : 0.f;
    conf.eps = bd.batch_norm_epsilon;

    set_offsets(data_mdw, off.src_off);

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    dispatch = compute_engine->create_dispatch(data_mdw.md_);
    dispatch.define_dim("MB", 0, conf.mb);
    dispatch.define_dim("IC", 1, conf.ic);
    dispatch.define_dim("ID", nstl::max(1, ndims - 3), conf.id);
    dispatch.define_dim("IH", nstl::max(1, ndims - 2), conf.ih);
    dispatch.define_dim("IW", nstl::max(1, ndims - 1), conf.iw);

    dispatch.generate();
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const compute::dispatch_t &dispatch,
        const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", conf.reduce_stat_nblocks);

    kernel_ctx.define_int("REDUCE_DIM_IDX", conf.reduce_dim_idx);
    kernel_ctx.define_int("REDUCE_DIM", conf.reduce_dim);

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

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    def_dispatch(kernel_ctx, dispatch);
}

void ref_batch_normalization_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    const memory_desc_wrapper data_mdw(src_md());
    init_conf_common(conf, dispatch, off, this, data_mdw, engine);

    if (conf.calculate_stats) {
        init_calculate_stats_conf(conf, dispatch_calc_stat,
                dispatch_reduce_stat, engine, data_mdw);
    }
}

void ref_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_FWD", 1);

    if (conf.calculate_stats) {
        def_dispatch(kernel_ctx, dispatch_calc_stat);
        def_dispatch(kernel_ctx, dispatch_reduce_stat);
    }
    init_kernel_ctx_common(kernel_ctx, conf, dispatch, off);
}

void ref_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (conf.calculate_stats) {

        size_t size = 2 * static_cast<size_t>(conf.stat_ic);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
                types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
    }
}

status_t ref_batch_normalization_fwd_t::execute_forward(
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

    auto *mean_ptr = &mean_;
    auto *variance_ptr = &variance_;

    std::unique_ptr<memory_storage_t> temp_reduce = nullptr;
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
        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);

        auto nd_range_calc_mean = pd()->dispatch_calc_stat.nd_range();

        CHECK(parallel_for(ctx, nd_range_calc_mean, calculate_mean_kernel_,
                calc_mean_arg_list));

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, mean);

        auto nd_range_reduce_kernels = pd()->dispatch_reduce_stat.nd_range();

        CHECK(parallel_for(ctx, nd_range_reduce_kernels, reduce_mean_kernel_,
                reduce_mean_arg_list));

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, mean);
        calc_var_arg_list.set(2, *temp_reduce);

        auto nd_range_calc_var = pd()->dispatch_calc_stat.nd_range();

        CHECK(parallel_for(ctx, nd_range_calc_var, calculate_variance_kernel_,
                calc_var_arg_list));

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, variance);

        CHECK(parallel_for(ctx, nd_range_reduce_kernels,
                reduce_variance_kernel_, reduce_var_arg_list));
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

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

void ref_batch_normalization_bwd_t::pd_t::init_conf(impl::engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(diff_src_md());
    init_conf_common(conf, dispatch, off, this, data_mdw, engine);

    init_calculate_stats_conf(
            conf, dispatch_calc_stat, dispatch_reduce_stat, engine, data_mdw);
}

void ref_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {

    def_dispatch(kernel_ctx, dispatch_calc_stat);
    def_dispatch(kernel_ctx, dispatch_reduce_stat);
    kernel_ctx.define_int("IS_BWD", 1);
    init_kernel_ctx_common(kernel_ctx, conf, dispatch, off);
}

void ref_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size = 2 * static_cast<size_t>(conf.stat_ic);

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t ref_batch_normalization_bwd_t::execute_backward(
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

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);

    auto nd_range = pd()->dispatch_calc_stat.nd_range();

    CHECK(parallel_for(
            ctx, nd_range, calculate_stats_kernel_, calc_stats_arg_list));

    auto &diff_scale = !conf.use_scale ? *temp_reduce : diff_scale_;
    auto &diff_shift = !conf.use_shift ? *temp_reduce : diff_shift_;

    compute::kernel_arg_list_t reduce_stats_arg_list;
    reduce_stats_arg_list.set(0, *temp_reduce);
    reduce_stats_arg_list.set(1, diff_scale);
    reduce_stats_arg_list.set(2, diff_shift);
    reduce_stats_arg_list.set(3, variance);
    reduce_stats_arg_list.set(4, conf.eps);

    auto nd_range_reduce_stat = pd()->dispatch_reduce_stat.nd_range();

    CHECK(parallel_for(ctx, nd_range_reduce_stat, reduce_stats_kernel_,
            reduce_stats_arg_list));

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

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
