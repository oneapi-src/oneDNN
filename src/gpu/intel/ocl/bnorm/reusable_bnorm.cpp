/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/ocl/bnorm/reusable_bnorm.hpp"

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"

using namespace dnnl::impl::memory_tracking::names;

#define MAX_DIMS 5

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

namespace bnorm_dims_t {
dim_idx_t mb = 0;
dim_idx_t ic = 1;
dim_idx_t sp0 = 2;
dim_idx_t sp1 = 3;
dim_idx_t sp2 = 4;
}; // namespace bnorm_dims_t

static std::vector<dim_idx_t> get_dims(size_t ndims) {
    std::vector<dim_idx_t> ret(ndims);
    uint8_t idx = 0;
    ret[idx++] = bnorm_dims_t::mb;
    ret[idx++] = bnorm_dims_t::ic;
    if (ndims >= 3) ret[idx++] = bnorm_dims_t::sp0;
    if (ndims >= 4) ret[idx++] = bnorm_dims_t::sp1;
    if (ndims >= 5) ret[idx++] = bnorm_dims_t::sp2;
    return ret;
}

static status_t init_calculate_stats_conf(reusable_bnorm_params_t &conf,
        reusable_bnorm_runtime_params_t &rt_conf, impl::engine_t *engine,
        const memory_desc_wrapper &data_mdw,
        const gpu_primitive_attr_t *gpu_attr) {
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    const size_t ndims = static_cast<size_t>(data_mdw.ndims());
    dim_t calc_dims[MAX_DIMS];
    auto &dims = data_mdw.dims();
    for (size_t i = 0; i < MAX_DIMS; i++) {
        calc_dims[i] = (i < ndims) ? dims[i] : 1;
    }
    size_t reduce_dim_idx = 0;
    for (size_t i = 2; i < MAX_DIMS; i++) {
        if (calc_dims[i] > calc_dims[reduce_dim_idx]) reduce_dim_idx = i;
    }
    rt_conf.reduction_nelems = calc_dims[reduce_dim_idx];
    rt_conf.div = utils::array_product(calc_dims, MAX_DIMS) / calc_dims[1];

    // Set up the reusable_calculate_* kernels:
    // - SRC: Just the input src buffer
    // - DST: SRC, but with one dim reduced and ic moved to the innermost position
    // - [variance only] MEAN: just the ic dim, stores the mean per ic
    std::vector<dim_idx_t> dim_ids = get_dims(ndims);
    compute::named_buffer_t src_buf("SRC", *data_mdw.md_, dim_ids);

    compute::named_buffer_t dst_buf("DST", src_buf);
    dst_buf.remove_dim(dim_ids[reduce_dim_idx]);

    // Move ic interior
    dst_buf.remove_dim(bnorm_dims_t::ic);
    dst_buf.append_block(bnorm_dims_t::ic, dims[1]);

    rt_conf.reduce_dim_stride = 1;
    size_t num_blocks_reduced = 0;
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

    // Dispatches all dims except for reduction dim
    for (size_t i = 0; i < dim_ids.size(); i++) {
        if (dim_ids[i] == dim_idx_t(reduce_dim_idx)) {
            dim_ids.erase(dim_ids.begin() + static_cast<int>(i));
            break;
        }
    }
    compute::reusable_dispatch_config_t calc_stat_dispatch_config(
            compute_engine, dim_ids);
    CHECK(calc_stat_dispatch_config.register_buffer(src_buf));
    CHECK(calc_stat_dispatch_config.register_buffer(dst_buf));
    CHECK(calc_stat_dispatch_config.define_dim_index(
            "IC_DIM", bnorm_dims_t::ic, rt_conf.ic));

    compute::reusable_dispatch_t dispatch_calc_stat;
    auto lws_strategy
            = compute::default_lws_strategy_t(compute_engine, gpu_attr);
    CHECK(calc_stat_dispatch_config.generate(dispatch_calc_stat, lws_strategy));
    conf.calc_stat_params = dispatch_calc_stat.get_compile_params();
    rt_conf.calc_stat_params = dispatch_calc_stat.get_runtime_params();

    compute::named_buffer_t reduce_buffer("BUFFER", dst_buf);
    for (const auto &dim : dim_ids) {
        if (dim != bnorm_dims_t::ic) reduce_buffer.remove_dim(dim);
    }

    // Reduce kernels dispatch to ic dim only
    compute::reusable_dispatch_config_t reduce_stat_dispatch_config(
            compute_engine, {bnorm_dims_t::ic});
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
        impl::engine_t *engine, const gpu_primitive_attr_t *&gpu_attr) {
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
    rt_conf.ic = data_mdw.dims()[1];

    conf.with_leaky_relu = conf.with_relu && rt_conf.relu_negative_slope != 0.f;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    size_t ndims = static_cast<size_t>(data_mdw.ndims());
    std::vector<dim_idx_t> dims = get_dims(ndims);
    compute::named_buffer_t buffer("BUFFER", *data_mdw.md_, dims);

    // Dispatch to all dims
    compute::reusable_dispatch_config_t dispatch_config(compute_engine, dims);
    CHECK(dispatch_config.register_buffer(buffer));
    CHECK(dispatch_config.define_dim_index(
            "IC_DIM", bnorm_dims_t::ic, rt_conf.ic));

    compute::reusable_dispatch_t dispatch;
    CHECK(dispatch_config.generate(dispatch,
            compute::default_lws_strategy_t(compute_engine, gpu_attr)));
    conf.gws_params = dispatch.get_compile_params();
    rt_conf.gws_params = dispatch.get_runtime_params();

    return status::success;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reusable_bnorm_params_t &conf) {
    kernel_ctx.set_data_type(conf.data_type, false);

    kernel_ctx.define_int("WITH_RELU", conf.with_relu);
    if (conf.with_leaky_relu) kernel_ctx.define_int("WITH_LEAKY_RELU", 1);

    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("FUSE_BN_RELU", conf.fuse_norm_relu);
    kernel_ctx.define_int("FUSE_BN_ADD_RELU", conf.fuse_norm_add_relu);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);

    if (conf.data_type == data_type::s8)
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    conf.gws_params.def_kernel_macros(kernel_ctx);
    conf.calc_stat_params.def_kernel_macros(kernel_ctx, "CALC");
    conf.reduce_stat_params.def_kernel_macros(kernel_ctx, "REDUCE");
}

status_t reusable_batch_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
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
        size_t stats_size = static_cast<size_t>(rt_conf.ic);
        size_t elsize = types::data_type_size(data_type::f32);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_bnorm_reduction,
                reduce_size, elsize, OCL_BUFFER_ALIGNMENT);
        if (!conf.is_training) {
            scratchpad.book(memory_tracking::names::key_bnorm_tmp_mean,
                    stats_size, elsize, OCL_BUFFER_ALIGNMENT);
            scratchpad.book(memory_tracking::names::key_bnorm_tmp_var,
                    stats_size, elsize, OCL_BUFFER_ALIGNMENT);
        }
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
            mean_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_mean);
            variance_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                    key_bnorm_tmp_var);
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

        bool use_int32_offset = conf.gws_params.use_int32_offset;
        const auto &append_off
                = [use_int32_offset](
                          compute::kernel_arg_list_t &arg_list, dim_t off) {
                      if (use_int32_offset) {
                          arg_list.append(into<int32_t>(off));
                      } else {
                          arg_list.append(off);
                      }
                  };

        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.set(0, src);
        calc_mean_arg_list.set(1, *temp_reduce);
        append_off(calc_mean_arg_list, rt_conf.reduce_dim_stride);
        append_off(calc_mean_arg_list, rt_conf.reduction_nelems);
        calc_mean_arg_list.append(rt_conf.calc_stat_params.get());

        auto &nd_range_calc = rt_conf.calc_stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_calc, calculate_mean_kernel_,
                calc_mean_arg_list));

        compute::kernel_arg_list_t reduce_mean_arg_list;
        reduce_mean_arg_list.set(0, *temp_reduce);
        reduce_mean_arg_list.set(1, *mean_ptr);
        append_off(reduce_mean_arg_list, rt_conf.ic);
        append_off(
                reduce_mean_arg_list, rt_conf.div / rt_conf.reduction_nelems);
        append_off(reduce_mean_arg_list, rt_conf.div);
        reduce_mean_arg_list.append(rt_conf.reduce_stat_params.get());

        auto &nd_range_reduce = rt_conf.reduce_stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_reduce, reduce_mean_kernel_,
                reduce_mean_arg_list));

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.set(0, src);
        calc_var_arg_list.set(1, *mean_ptr);
        calc_var_arg_list.set(2, *temp_reduce);
        append_off(calc_var_arg_list, rt_conf.reduce_dim_stride);
        append_off(calc_var_arg_list, rt_conf.reduction_nelems);
        calc_var_arg_list.append(rt_conf.calc_stat_params.get());

        CHECK(parallel_for(ctx, nd_range_calc, calculate_variance_kernel_,
                calc_var_arg_list));

        compute::kernel_arg_list_t reduce_var_arg_list;
        reduce_var_arg_list.set(0, *temp_reduce);
        reduce_var_arg_list.set(1, *variance_ptr);
        append_off(reduce_var_arg_list, rt_conf.ic);
        append_off(reduce_var_arg_list, rt_conf.div / rt_conf.reduction_nelems);
        append_off(reduce_var_arg_list, rt_conf.div);
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

status_t reusable_batch_normalization_bwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper data_mdw(diff_src_md());
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());
    CHECK(init_conf_common(conf, rt_conf, this, data_mdw, engine, gpu_attr));
    CHECK(init_calculate_stats_conf(conf, rt_conf, engine, data_mdw, gpu_attr));
    return status::success;
}

void reusable_batch_normalization_bwd_t::pd_t::init_scratchpad() {
    size_t elsize = types::data_type_size(data_type::f32);
    size_t size = static_cast<size_t>(rt_conf.stat_ic);
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_bnorm_reduction, size, elsize,
            OCL_BUFFER_ALIGNMENT);
    scratchpad.book(memory_tracking::names::key_bnorm_reduction_shift, size,
            elsize, OCL_BUFFER_ALIGNMENT);
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
    std::unique_ptr<memory_storage_t> temp_reduce_shift;
    temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction);
    temp_reduce_shift = ctx.get_scratchpad_grantor().get_memory_storage(
            key_bnorm_reduction_shift);

    bool use_int32_offset = conf.gws_params.use_int32_offset;
    const auto &append_off
            = [use_int32_offset](
                      compute::kernel_arg_list_t &arg_list, dim_t off) {
                  if (use_int32_offset) {
                      arg_list.append(into<int32_t>(off));
                  } else {
                      arg_list.append(off);
                  }
              };

    compute::kernel_arg_list_t calc_stats_arg_list;
    calc_stats_arg_list.set(0, src);
    calc_stats_arg_list.set(1, mean);
    calc_stats_arg_list.set(2, diff_dst);
    calc_stats_arg_list.set(3, ws);
    calc_stats_arg_list.set(4, *temp_reduce);
    calc_stats_arg_list.set(5, *temp_reduce_shift);
    append_off(calc_stats_arg_list, rt_conf.reduce_dim_stride);
    append_off(calc_stats_arg_list, rt_conf.reduction_nelems);
    calc_stats_arg_list.append(rt_conf.calc_stat_params.get());

    auto &nd_range_calc = rt_conf.calc_stat_params.nd_range;

    CHECK(parallel_for(
            ctx, nd_range_calc, calculate_stats_kernel_, calc_stats_arg_list));
    auto &diff_scale = !conf.use_scale ? *temp_reduce : diff_scale_;
    auto &diff_shift = !conf.use_shift ? *temp_reduce_shift : diff_shift_;

    compute::kernel_arg_list_t reduce_stats_arg_list;
    reduce_stats_arg_list.set(0, *temp_reduce);
    reduce_stats_arg_list.set(1, *temp_reduce_shift);
    reduce_stats_arg_list.set(2, diff_scale);
    reduce_stats_arg_list.set(3, diff_shift);
    reduce_stats_arg_list.set(4, variance);
    reduce_stats_arg_list.set(5, rt_conf.eps);
    append_off(reduce_stats_arg_list, rt_conf.ic);
    append_off(reduce_stats_arg_list, rt_conf.div / rt_conf.reduction_nelems);
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
    arg_list.set(8, diff_shift);
    arg_list.set(9, rt_conf.eps);
    arg_list.set(10, diff_src_add);
    append_off(arg_list, rt_conf.div);
    arg_list.append(rt_conf.gws_params.get());

    auto &nd_range = rt_conf.gws_params.nd_range;

    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
