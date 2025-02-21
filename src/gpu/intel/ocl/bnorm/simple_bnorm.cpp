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

#include "gpu/intel/ocl/bnorm/simple_bnorm.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/ocl/utils.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t init_conf_common(bnorm_conf_t &conf, offsets_t &off,
        const batch_normalization_pd_t *pd) {

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const dim_idx_t ndims = into<dim_idx_t>(data_mdw.ndims());

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

    conf.mb_block = 1;

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const bnorm_conf_t &conf, const compute::dispatch_t &dispatch_calc_stat,
        const compute::dispatch_t &dispatch_reduce_stat,
        const compute::dispatch_t &dispatch, const offsets_t &off) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);

    kernel_ctx.define_int("REDUCE_DIM_IDX", conf.reduce_dim_idx);
    kernel_ctx.define_int("REDUCE_DIM", conf.reduce_dim);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    if (conf.with_relu && conf.relu_negative_slope != 0.f)
        kernel_ctx.define_int("WITH_LEAKY_RELU", 1);

    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("FUSE_BN_ADD_RELU", conf.fuse_norm_add_relu);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_size);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    if (conf.calculate_stats || conf.is_backward) {
        def_dispatch(kernel_ctx, dispatch_calc_stat);
        def_dispatch(kernel_ctx, dispatch_reduce_stat);
    }
    def_dispatch(kernel_ctx, dispatch);
    return status::success;
}

status_t simple_batch_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    CHECK(init_conf_common(conf, off, this));

    // This implementation optimizes the stat calculation - skip it if we're not calculating stats
    if (!conf.calculate_stats) return status::unimplemented;

    const memory_desc_wrapper data_mdw(src_md());
    const dim_idx_t ndims = into<dim_idx_t>(data_mdw.ndims());

    compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);

    dispatch_calc_stat = compute_engine->create_dispatch(data_mdw.md_);
    dim_t calc_dims[5];
    auto &dims = data_mdw.dims();
    calc_dims[0] = dims[0];
    calc_dims[1] = dims[1];
    calc_dims[2] = (ndims < 5) ? 1 : dims[ndims - 3];
    calc_dims[3] = (ndims < 4) ? 1 : dims[ndims - 2];
    calc_dims[4] = (ndims < 3) ? 1 : dims[ndims - 1];
    dim_idx_t reduce_dim_idx = 0;
    for (dim_idx_t i = 2; i < 5; i++) {
        if (calc_dims[i] > calc_dims[reduce_dim_idx]) { reduce_dim_idx = i; }
    }
    conf.stat_ic = utils::array_product(calc_dims, 5);
    conf.reduce_dim = calc_dims[reduce_dim_idx];
    conf.reduce_dim_idx = reduce_dim_idx;
    const std::string dim_names[5]
            = {"STAT_MB", "STAT_IC", "STAT_ID", "STAT_IH", "STAT_IW"};
    const std::string &reduce_dim_name = dim_names[reduce_dim_idx];

    // Translate reduce_dim_idx from being an index in calc_dims to dims array
    const dim_idx_t base_reduce_dim_idx
            = reduce_dim_idx == 0 ? 0 : reduce_dim_idx - (5 - ndims);
    const dim_t reduce_dim_stride
            = data_mdw.blocking_desc().strides[base_reduce_dim_idx];
    const bool can_vectorize_calc_stats
            = conf.reduce_dim % 16 == 0 && reduce_dim_stride == 1;
    conf.skip_reduce_stat = can_vectorize_calc_stats
            && conf.stat_ic == conf.reduce_dim * calc_dims[1];

    // Fall back to reference unless this heuristic is hit to skip reducing stats
    // WARNING: This is doing a one-pass stat calculation, but it's not checking
    // for the experimental flag. This should be updated to follow the API.
    if (!conf.skip_reduce_stat) return status::unimplemented;

    dim_t calc_dims_blocks[5] = {1, 1, 1, 1, 1};

    // Calculations over reduce dimension will be split
    // between work items in the single subgroup.
    // Each item will read vector_size of elements at once.
    conf.sub_group_size = 16;

    int vector_size = 8;
    while (conf.reduce_dim % (conf.sub_group_size * vector_size) != 0) {
        vector_size /= 2;
    }
    conf.vect_size = vector_size;
    calc_dims_blocks[reduce_dim_idx] = conf.reduce_dim / conf.sub_group_size;

    dispatch_calc_stat.define_dim(
            dim_names[0], 0, calc_dims[0], calc_dims_blocks[0]);
    dispatch_calc_stat.define_dim(
            dim_names[1], 1, calc_dims[1], calc_dims_blocks[1]);
    dispatch_calc_stat.define_dim(dim_names[2], nstl::max(1u, ndims - 3u),
            calc_dims[2], calc_dims_blocks[2]);
    dispatch_calc_stat.define_dim(dim_names[3], nstl::max(1u, ndims - 2u),
            calc_dims[3], calc_dims_blocks[3]);
    dispatch_calc_stat.define_dim(dim_names[4], nstl::max(1u, ndims - 1u),
            calc_dims[4], calc_dims_blocks[4]);

    CHECK(dispatch_calc_stat.vectorize_dim(
            reduce_dim_name, conf.sub_group_size));

    dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    dispatch_calc_stat.generate();

    dispatch_reduce_stat = compute_engine->create_dispatch();
    dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
    dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    dispatch_reduce_stat.generate();

    dispatch = compute_engine->create_dispatch(data_mdw.md_);
    dispatch.define_dim("MB", 0, conf.mb);
    dispatch.define_dim("IC", 1, conf.ic);
    dispatch.define_dim("ID", nstl::max(1u, ndims - 3u), conf.id);
    dispatch.define_dim("IH", nstl::max(1u, ndims - 2u), conf.ih);
    dispatch.define_dim("IW", nstl::max(1u, ndims - 1u), conf.iw);
    dispatch.generate();

    return status::success;
}

status_t simple_batch_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    CHECK(init_kernel_ctx_common(kernel_ctx, conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, off));
    kernel_ctx.define_int("IS_FWD", 1);
    return status::success;
}

void simple_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    size_t size = 2 * conf.stat_ic;

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t simple_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &src_add = CTX_IN_STORAGE(DNNL_ARG_SRC_1);

    status_t status = status::success;
    auto &mean_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_MEAN, status);
    CHECK(status);
    auto &variance_ = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_VARIANCE, status);
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

    if (!conf.save_stats) {
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                key_bnorm_reduction);
        mean_ptr = temp_reduce.get();
        variance_ptr = temp_reduce.get();
    }

    auto &mean = *mean_ptr;
    auto &variance = *variance_ptr;

    compute::kernel_arg_list_t calc_var_arg_list;
    calc_var_arg_list.set(0, src);
    calc_var_arg_list.set(1, mean);
    calc_var_arg_list.set(2, variance);

    auto nd_range_calc_var = pd()->dispatch_calc_stat.nd_range();

    CHECK(parallel_for(ctx, nd_range_calc_var, calculate_mean_variance_kernel_,
            calc_var_arg_list));

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

status_t simple_batch_normalization_bwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    CHECK(init_conf_common(conf, off, this));
    compute::compute_engine_t *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine);
    const memory_desc_wrapper data_mdw(diff_src_md());

    const bool has_padding = !data_mdw.is_dense();
    const bool is_supported_single_block
            = data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c);
    const bool is_supported_double_block
            = data_mdw.matches_one_of_tag(NCw16n16c, NChw16n16c, NCdhw16n16c);

    if (has_padding
            || !(is_supported_single_block || is_supported_double_block)) {
        return status::unimplemented;
    }

    conf.mb_block = 1;
    conf.vect_size = 1;
    if (is_supported_double_block) {
        conf.mb_block = 16;
        conf.vect_size = 8;
    }

    const int max_stat_nblocks = 256;
    dim_t stat_mb_nblocks = conf.mb / conf.mb_block;
    dim_t stat_sp_nblocks = utils::max_div(conf.id * conf.ih * conf.iw,
            nstl::max<dim_t>(1, max_stat_nblocks / stat_mb_nblocks));
    assert(stat_mb_nblocks * stat_sp_nblocks <= max_stat_nblocks);

    dim_t stat_sp_block = conf.id * conf.ih * conf.iw / stat_sp_nblocks;

    conf.reduce_stat_nblocks = stat_mb_nblocks * stat_sp_nblocks;

    dispatch_calc_stat = compute_engine->create_dispatch();
    dispatch_calc_stat.define_dim_with_nesting_level(
            "STAT_SP", 2, conf.id * conf.ih * conf.iw, stat_sp_block);
    dispatch_calc_stat.define_dim_with_nesting_level("STAT_IC", 1, conf.ic);
    dispatch_calc_stat.define_dim_with_nesting_level(
            "STAT_MB", 0, conf.mb, conf.mb_block);
    CHECK(dispatch_calc_stat.vectorize_dim("STAT_IC", 16));
    dispatch_calc_stat.set_kernel_attr_suffix("CALC");
    dispatch_calc_stat.generate();

    dispatch_reduce_stat = compute_engine->create_dispatch();
    dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", conf.ic);
    dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
    dispatch_reduce_stat.generate();

    dispatch = compute_engine->create_dispatch(data_mdw.md_);
    dispatch.define_dim("MB", 0, conf.mb, conf.mb_block);
    dispatch.define_dim("IC", 1, conf.ic);
    dispatch.define_dim("ID", nstl::max(1u, conf.ndims - 3u), conf.id);
    dispatch.define_dim("IH", nstl::max(1u, conf.ndims - 2u), conf.ih);
    dispatch.define_dim("IW", nstl::max(1u, conf.ndims - 1u), conf.iw);
    CHECK(dispatch.vectorize_dim("IC", 16));
    dispatch.generate();

    return status::unimplemented;
}

status_t simple_batch_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    CHECK(init_kernel_ctx_common(kernel_ctx, conf, dispatch_calc_stat,
            dispatch_reduce_stat, dispatch, off));
    kernel_ctx.define_int("IC_BLOCK", 16);
    kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", conf.reduce_stat_nblocks);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("IS_BWD", 1);
    return status::success;
}

void simple_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size = 2 * conf.reduce_stat_nblocks * conf.ic;

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t simple_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    status_t status = status::success;
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
