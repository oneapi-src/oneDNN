/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/ocl/bnorm/reusable_bnorm.hpp"
#include <limits>

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/scratchpad.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/block_structure.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/dispatch_reusable.hpp"
#include "gpu/gpu_primitive_attr.hpp"

using namespace dnnl::impl::memory_tracking::names;

#define MAX_DIMS 5

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Remove blocking structure for dim idx. Combines inner blocks with the outer one.
// XXX: Does not update the format tag to reflect the new blocking
memory_desc_t remove_blocking(memory_desc_t md, size_t idx) {
    auto &blk = md.format_desc.blocking;

    // Tally up inner blocks that will be removed
    std::vector<block_t> blocks;
    dim_t stride = 1;
    for (int i = blk.inner_nblks - 1; i >= 0; i--) {
        if (static_cast<size_t>(blk.inner_idxs[i]) == idx)
            blocks.emplace_back(idx, blk.inner_blks[i], stride);
        stride *= blk.inner_blks[i];
    }

    // Remove the inner blocks
    size_t num_remaining_blocks = 0;
    for (size_t i = 0; i < static_cast<size_t>(blk.inner_nblks); i++) {
        if (static_cast<size_t>(blk.inner_idxs[i]) == idx) continue;

        blk.inner_idxs[num_remaining_blocks] = blk.inner_idxs[i];
        blk.inner_blks[num_remaining_blocks++] = blk.inner_blks[i];
    }
    blk.inner_nblks = static_cast<int>(num_remaining_blocks);

    // Update strides
    dim_t outer_stride = blk.strides[idx];
    for (size_t i = 0; i < static_cast<size_t>(md.ndims); i++) {
        if (blk.strides[i] > outer_stride) continue;
        for (const auto &block : blocks) {
            if (blk.strides[i] >= block.stride) blk.strides[i] /= block.block;
        }
    }

    return md;
}

// Returns the memory_desc_t obtained after dim `idx` is reduced
// to 1 element (thereby removing blocking in that dimension as well)
memory_desc_t reduce_dim(memory_desc_t md, size_t idx) {
    size_t ndims = static_cast<size_t>(md.ndims);
    if (idx >= ndims) return md;

    md = remove_blocking(md, idx);

    auto &blk = md.format_desc.blocking;
    block_t block(
            static_cast<dim_t>(idx), md.padded_dims[idx], blk.strides[idx]);

    md.dims[idx] = 1;
    md.padded_dims[idx] = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (blk.strides[i] > block.stride) blk.strides[i] /= block.block;
    }

    return md;
}

static status_t init_calculate_stats_conf(reusable_bnorm_params_t &conf,
        reusable_bnorm_runtime_params_t &rt_conf, engine_t *engine,
        const memory_desc_wrapper &data_mdw,
        const gpu_primitive_attr_t *gpu_attr) {
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const int ndims = data_mdw.ndims();
    dim_t calc_dims[MAX_DIMS];
    auto &dims = data_mdw.dims();
    for (int i = 0; i < MAX_DIMS; i++) {
        calc_dims[i] = (i < ndims) ? dims[i] : 1;
    }
    size_t reduce_dim_idx = 0;
    for (size_t i = 2; i < MAX_DIMS; i++) {
        if (calc_dims[i] > calc_dims[reduce_dim_idx]) reduce_dim_idx = i;
    }
    rt_conf.reduction_nelems = calc_dims[reduce_dim_idx];
    rt_conf.ic = calc_dims[1];
    rt_conf.div = utils::array_product(calc_dims, MAX_DIMS) / calc_dims[1];
    const int max_int = std::numeric_limits<int>::max();
    if (rt_conf.reduction_nelems > max_int || rt_conf.ic > max_int
            || rt_conf.div > max_int)
        conf.use_int32_offset = false;

    memory_desc_t reduced = reduce_dim(*data_mdw.md_, reduce_dim_idx);
    memory_desc_t reordered = remove_blocking(reduced, 1);

    // Move entire ic dimension to the innermost block (guaranteeing stride = 1)
    blocking_desc_t &blk = reordered.format_desc.blocking;
    blk.inner_blks[blk.inner_nblks] = reordered.padded_dims[1];
    blk.inner_idxs[blk.inner_nblks++] = 1;
    dim_t outer_stride = blk.strides[1];
    for (size_t i = 0; i < static_cast<size_t>(reordered.ndims); i++) {
        if (blk.strides[i] <= outer_stride) {
            blk.strides[i] *= reordered.padded_dims[1];
        }
    }

    rt_conf.reduce_dim_stride = 1;
    size_t num_blocks_reduced = 0;
    if (rt_conf.reduce_dim_stride > std::numeric_limits<int>::max())
        conf.use_int32_offset = false;
    block_layout_t layout(data_mdw);
    for (const auto &block : layout) {
        if (static_cast<size_t>(block.dim_idx) == reduce_dim_idx) {
            rt_conf.reduce_dim_stride = block.stride;
            num_blocks_reduced++;
        }
    }
    if (num_blocks_reduced > 1) return status::unimplemented;

    // Whole reduce dimension will be handled by single work item.
    calc_dims[reduce_dim_idx] = 1;

    rt_conf.stat_ic = utils::array_product(calc_dims, MAX_DIMS);
    if (rt_conf.stat_ic > std::numeric_limits<int>::max())
        conf.use_int32_offset = false;

    compute::named_buffer_t src_buf("SRC", data_mdw.md_);
    src_buf.disable_dim(reduce_dim_idx);

    compute::named_buffer_t dst_buf("DST", reordered);

    compute::reusable_dispatch_config_t calc_stat_dispatch_config;
    CHECK(calc_stat_dispatch_config.register_buffer(src_buf));
    CHECK(calc_stat_dispatch_config.register_buffer(dst_buf));
    CHECK(calc_stat_dispatch_config.define_dim_index("IC_DIM", 1));

    compute::reusable_dispatch_t dispatch_calc_stat;
    auto lws_strategy
            = compute::default_lws_strategy_t(compute_engine, gpu_attr);
    CHECK(calc_stat_dispatch_config.generate(dispatch_calc_stat, lws_strategy));
    conf.calc_stat_params = dispatch_calc_stat.get_compile_params();
    rt_conf.calc_stat_params = dispatch_calc_stat.get_runtime_params();

    memory_desc_t final_layout = types::zero_md();
    final_layout.ndims = reordered.ndims;
    final_layout.data_type = data_type::f32;
    final_layout.format_kind = format_kind::blocked;
    blocking_desc_t &final_blk = final_layout.format_desc.blocking;
    const dim_t ic_padded = reordered.dims[1];
    for (size_t i = 0; i < static_cast<size_t>(final_layout.ndims); i++) {
        if (i == 1) {
            final_layout.dims[i] = ic_padded;
            final_layout.padded_dims[i] = ic_padded;
            final_blk.strides[i] = 1;
        } else {
            final_layout.dims[i] = 1;
            final_layout.padded_dims[i] = 1;
            final_blk.strides[i] = ic_padded;
        }
    }

    compute::named_buffer_t reduce_buffer("BUFFER", final_layout);
    compute::reusable_dispatch_config_t reduce_stat_dispatch_config;
    CHECK(reduce_stat_dispatch_config.register_buffer(reduce_buffer));

    compute::reusable_dispatch_t dispatch_reduce_stat;
    CHECK(reduce_stat_dispatch_config.generate(
            dispatch_reduce_stat, lws_strategy));
    conf.reduce_stat_params = dispatch_reduce_stat.get_compile_params();
    rt_conf.reduce_stat_params = dispatch_reduce_stat.get_runtime_params();
    return status::success;
}

static status_t init_conf_common(reusable_bnorm_params_t &conf,
        reusable_bnorm_runtime_params_t &rt_conf,
        const batch_normalization_pd_t *pd, const memory_desc_wrapper &data_mdw,
        engine_t *engine, const gpu_primitive_attr_t *&gpu_attr) {

    const batch_normalization_desc_t &bd = *pd->desc();

    conf = utils::zero<decltype(conf)>();
    conf.data_type = data_mdw.data_type();

    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.is_training = pd->is_training();
    conf.fuse_norm_relu = pd->fuse_norm_relu() || pd->fuse_norm_add_relu();
    conf.fuse_norm_add_relu = pd->fuse_norm_add_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op(pd->is_training());
    rt_conf.relu_negative_slope = conf.with_relu ? pd->alpha() : 0.f;
    rt_conf.eps = bd.batch_norm_epsilon;

    conf.with_leaky_relu = conf.with_relu && rt_conf.relu_negative_slope != 0.f;
    conf.use_int32_offset = true; // updated throughout following steps

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    // Both src and dst have the same format, use one unified indexing scheme
    compute::named_buffer_t buffer("BUFFER", data_mdw.md_);

    compute::reusable_dispatch_config_t dispatch_config;
    CHECK(dispatch_config.register_buffer(buffer));
    CHECK(dispatch_config.define_dim_index("IC_DIM", 1));

    compute::reusable_dispatch_t dispatch;
    CHECK(dispatch_config.generate(dispatch,
            compute::default_lws_strategy_t(compute_engine, gpu_attr)));
    conf.gws_params = dispatch.get_compile_params();
    rt_conf.gws_params = dispatch.get_runtime_params();

    const dim_t nelems = utils::array_product(
            data_mdw.dims(), static_cast<size_t>(data_mdw.ndims()));
    if (nelems > std::numeric_limits<int>::max()) conf.use_int32_offset = false;
    return status::success;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reusable_bnorm_params_t &conf) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    if (conf.with_leaky_relu) kernel_ctx.define_int("WITH_LEAKY_RELU", 1);

    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("FUSE_BN_ADD_RELU", conf.fuse_norm_add_relu);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);

    if (conf.use_int32_offset) { kernel_ctx.add_option("-DUSE_INT32_OFFSET"); }

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    conf.gws_params.def_kernel_macros(kernel_ctx);
    conf.calc_stat_params.def_kernel_macros(kernel_ctx, "CALC");
    conf.reduce_stat_params.def_kernel_macros(kernel_ctx, "REDUCE");
}

status_t reusable_batch_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    const memory_desc_wrapper data_mdw(src_md());
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    CHECK(init_conf_common(conf, rt_conf, this, data_mdw, engine, gpu_attr));
    CHECK(init_calculate_stats_conf(conf, rt_conf, engine, data_mdw, gpu_attr));
    return status::success;
}

compute::kernel_ctx_t reusable_bnorm_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    init_kernel_ctx_common(kernel_ctx, *this);
    return kernel_ctx;
}

void reusable_batch_normalization_fwd_t::pd_t::init_scratchpad() {
    if (conf.calculate_stats) {
        size_t reduce_size = static_cast<size_t>(rt_conf.stat_ic);
        size_t stats_size = static_cast<size_t>(2 * rt_conf.ic);

        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_bnorm_reduction,
                reduce_size, types::data_type_size(data_type::f32),
                OCL_BUFFER_ALIGNMENT);
        if (!conf.is_training)
            scratchpad.book(memory_tracking::names::key_bnorm_tmp_stats,
                    stats_size, types::data_type_size(data_type::f32),
                    OCL_BUFFER_ALIGNMENT);
    }
}

status_t reusable_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;
    const auto &rt_conf = pd()->rt_conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &src_add = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    // Set up mean and variance pointers
    std::unique_ptr<memory_storage_t> temp_stats
            = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_stats);
    std::unique_ptr<memory_storage_t> mean_temp = nullptr;
    std::unique_ptr<memory_storage_t> variance_temp = nullptr;
    memory_storage_t *mean_ptr;
    memory_storage_t *variance_ptr;
    if (conf.calculate_stats) {
        // Mean and variance aren't inputs
        if (conf.is_training) {
            mean_ptr = ctx.output(DNNL_ARG_MEAN)->memory_storage();
            variance_ptr = ctx.output(DNNL_ARG_VARIANCE)->memory_storage();
        } else {
            size_t size = types::data_type_size(data_type::f32)
                    * static_cast<size_t>(rt_conf.ic);
            mean_temp = temp_stats->get_sub_storage(0, size);
            variance_temp = temp_stats->get_sub_storage(size, size);
            mean_ptr = mean_temp.get();
            variance_ptr = variance_temp.get();
        }
    } else {
        mean_ptr = ctx.input(DNNL_ARG_MEAN)->memory_storage();
        variance_ptr = ctx.input(DNNL_ARG_VARIANCE)->memory_storage();
    }

    if (conf.calculate_stats) {
        std::unique_ptr<memory_storage_t> temp_reduce
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        key_bnorm_reduction);

        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);
        calc_mean_arg_list.append(rt_conf.reduce_dim_stride);
        calc_mean_arg_list.append(rt_conf.reduction_nelems);
        calc_mean_arg_list.append(rt_conf.calc_stat_params.get());

        auto &nd_range_calc = rt_conf.calc_stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_calc, calculate_mean_kernel_,
                calc_mean_arg_list));

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, *mean_ptr);
        reduce_mean_arg_list.append(rt_conf.ic);
        reduce_mean_arg_list.append(rt_conf.div / rt_conf.reduction_nelems);
        reduce_mean_arg_list.append(rt_conf.div);
        reduce_mean_arg_list.append(rt_conf.reduce_stat_params.get());

        auto &nd_range_reduce = rt_conf.reduce_stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_reduce, reduce_mean_kernel_,
                reduce_mean_arg_list));

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, *mean_ptr);
        calc_var_arg_list.set(2, *temp_reduce);
        calc_var_arg_list.append(rt_conf.reduce_dim_stride);
        calc_var_arg_list.append(rt_conf.reduction_nelems);
        calc_var_arg_list.append(rt_conf.calc_stat_params.get());

        CHECK(parallel_for(ctx, nd_range_calc, calculate_variance_kernel_,
                calc_var_arg_list));

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, *variance_ptr);
        reduce_var_arg_list.append(rt_conf.ic);
        reduce_var_arg_list.append(rt_conf.div / rt_conf.reduction_nelems);
        reduce_var_arg_list.append(rt_conf.div);
        reduce_var_arg_list.append(rt_conf.reduce_stat_params.get());

        CHECK(parallel_for(ctx, nd_range_reduce, reduce_variance_kernel_,
                reduce_var_arg_list));
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, *mean_ptr);
    arg_list.set(2, *variance_ptr);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, shift);
    arg_list.set(6, ws);
    arg_list.set(7, rt_conf.eps);
    arg_list.set(8, src_add);
    arg_list.set(9, rt_conf.relu_negative_slope);
    arg_list.append(rt_conf.gws_params.get());

    auto &nd_range = rt_conf.gws_params.nd_range;

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

status_t reusable_batch_normalization_bwd_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(diff_src_md());
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    CHECK(init_conf_common(conf, rt_conf, this, data_mdw, engine, gpu_attr));
    CHECK(init_calculate_stats_conf(conf, rt_conf, engine, data_mdw, gpu_attr));
    return status::success;
}

void reusable_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t size = 2 * static_cast<size_t>(rt_conf.stat_ic);

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t reusable_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {

    const auto &conf = pd()->conf;
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
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);
    calc_stats_arg_list.append(rt_conf.reduce_dim_stride);
    calc_stats_arg_list.append(rt_conf.reduction_nelems);
    calc_stats_arg_list.append(
            rt_conf.div * rt_conf.ic / rt_conf.reduction_nelems);
    calc_stats_arg_list.append(rt_conf.calc_stat_params.get());

    auto &nd_range_calc = rt_conf.calc_stat_params.nd_range;

    CHECK(parallel_for(
            ctx, nd_range_calc, calculate_stats_kernel_, calc_stats_arg_list));

    auto &diff_scale = !conf.use_scale ? *temp_reduce : diff_scale_;
    std::unique_ptr<memory_storage_t> shift_temp_mem = nullptr;
    memory_storage_t *diff_shift_ptr;
    if (conf.use_shift) {
        diff_shift_ptr = &diff_shift_;
    } else {
        size_t size = static_cast<size_t>(rt_conf.stat_ic)
                * types::data_type_size(data_type::f32);
        shift_temp_mem = temp_reduce->get_sub_storage(size, size);
        diff_shift_ptr = shift_temp_mem.get();
    }

    compute::kernel_arg_list_t reduce_stats_arg_list;
    reduce_stats_arg_list.set(0, *temp_reduce);
    reduce_stats_arg_list.set(1, diff_scale);
    reduce_stats_arg_list.set(2, *diff_shift_ptr);
    reduce_stats_arg_list.set(3, variance);
    reduce_stats_arg_list.set(4, rt_conf.eps);
    reduce_stats_arg_list.append(rt_conf.ic);
    reduce_stats_arg_list.append(rt_conf.div / rt_conf.reduction_nelems);
    reduce_stats_arg_list.append(rt_conf.reduce_stat_params.get());

    auto &nd_range_reduce_stat = rt_conf.reduce_stat_params.nd_range;

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
    arg_list.set(8, *diff_shift_ptr);
    arg_list.set(9, rt_conf.eps);
    arg_list.set(10, diff_src_add);
    arg_list.append(rt_conf.div);
    arg_list.append(rt_conf.gws_params.get());

    auto &nd_range = rt_conf.gws_params.nd_range;

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
