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

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/data_type_converter.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/lnorm_utils.hpp"
#include "gpu/intel/ocl/reusable_lnorm.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t init_conf_common(const layer_normalization_pd_t *pd,
        reusable_lnorm_params_t *conf, reusable_lnorm_runtime_params_t *rt_conf,
        const impl::engine_t *engine, const compute::named_buffer_t &src_buf,
        const compute::named_buffer_t &dst_buf,
        const compute::named_buffer_t &stat_buf,
        const compute::named_buffer_t &ss_buf) {
    conf->use_scale = pd->use_scale();
    conf->use_shift = pd->use_shift();
    conf->src_dt = src_buf.data_type;
    conf->dst_dt = dst_buf.data_type;

    auto scales = pd->attr()->scales_;
    conf->with_src_scale = !scales.has_default_values(DNNL_ARG_SRC);
    conf->with_dst_scale = !scales.has_default_values(DNNL_ARG_DST);

    // We require that the lnorm axis is a single dense block, so that it can
    // be represented by a stride + size alone.
    size_t ndims = into<size_t>(src_buf.ndims);
    std::vector<dim_idx_t> dims = get_dims(ndims);
    block_layout_t layout = src_buf.layout();
    const block_t *norm_block = [&layout, &dims]() -> const block_t * {
        const block_t *ret = nullptr;
        for (const block_t &block : layout) {
            if (into<size_t>(block.dim_idx) == dims.back()) {
                if (ret) return nullptr;
                ret = &block;
            }
        }
        gpu_assert(ret) << "Expected to find a norm block";
        return ret;
    }();
    if (!norm_block) return status::unimplemented;
    rt_conf->norm_stride = norm_block->stride;

    const auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
            pd->attr()->gpu_attr_.get());

    const auto *compute_engine
            = utils::downcast<const compute::compute_engine_t *>(engine);

    // Norm dispatch: all dimensions
    auto lws_strategy
            = compute::default_lws_strategy_t(compute_engine, gpu_attr);
    compute::reusable_dispatch_config_t dispatch_config(compute_engine, dims);
    CHECK(dispatch_config.register_buffer(src_buf));
    CHECK(dispatch_config.register_buffer(dst_buf));
    CHECK(dispatch_config.register_buffer(stat_buf));
    CHECK(dispatch_config.register_buffer(ss_buf));
    compute::reusable_dispatch_t dispatch;

    CHECK(dispatch_config.generate(dispatch, lws_strategy));
    conf->gws_params = dispatch.get_compile_params();
    rt_conf->gws_params = dispatch.get_runtime_params();

    // stat calculation dispatch: all stat dimensions
    compute::reusable_dispatch_config_t calc_stat_dispatch_config(
            compute_engine, stat_buf.get_dim_ids());
    CHECK(calc_stat_dispatch_config.register_buffer(src_buf));
    CHECK(calc_stat_dispatch_config.register_buffer(dst_buf));
    CHECK(calc_stat_dispatch_config.register_buffer(stat_buf));

    compute::reusable_dispatch_t dispatch_calc_stat;
    CHECK(calc_stat_dispatch_config.generate(dispatch_calc_stat, lws_strategy));
    conf->stat_params = dispatch_calc_stat.get_compile_params();
    rt_conf->stat_params = dispatch_calc_stat.get_runtime_params();

    // SS dispatcher is only used in BWD propagation, but we have to set it up
    // to compile in either case, since we compile all kernels at once
    if (conf->use_scale || conf->use_shift) {
        // We need to reduce over all but the last dimension as well. Make sure
        // this is a single block by ensuring that the last dimension's block
        // is either the innermost or outermost block.
        bool is_innermost = into<size_t>(layout.front().dim_idx) == dims.back();
        bool is_outermost = into<size_t>(layout.back().dim_idx) == dims.back();
        if (is_innermost) {
            rt_conf->stat_stride = layout.front().block;
        } else if (is_outermost) {
            rt_conf->stat_stride = 1;
        } else {
            return status::unimplemented;
        }
    }

    // scaleshift dispatch: just the norm axis
    compute::reusable_dispatch_config_t ss_dispatch_config(
            compute_engine, ss_buf.get_dim_ids());
    CHECK(ss_dispatch_config.register_buffer(src_buf));
    CHECK(ss_dispatch_config.register_buffer(dst_buf));
    CHECK(ss_dispatch_config.register_buffer(ss_buf));

    compute::reusable_dispatch_t dispatch_ss;
    CHECK(ss_dispatch_config.generate(dispatch_ss, lws_strategy));
    conf->scaleshift_params = dispatch_ss.get_compile_params();
    rt_conf->scaleshift_params = dispatch_ss.get_runtime_params();

    return status::success;
}

void init_scratchpad_common(
        memory_tracking::registrar_t &scratchpad, dim_t nelems) {
    using namespace memory_tracking::names;
    for (auto key : {key_lnorm_tmp_mean, key_lnorm_tmp_var}) {
        scratchpad.book<float>(key, into<size_t>(nelems), OCL_BUFFER_ALIGNMENT);
    }
}

status_t reusable_layer_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    size_t ndims = static_cast<size_t>(src_md()->ndims);
    std::vector<dim_idx_t> dims = get_dims(ndims);
    std::vector<dim_idx_t> stat_dims = get_dims(ndims, true);

    // FWD buffers:
    // - src: all dims
    // - dst: all dims
    // - stat: (mean/variance) all but last dim
    // - SS: (scale/shift) just the last dim
    compute::named_buffer_t src_buffer("SRC", *src_md(), dims);
    compute::named_buffer_t dst_buffer("DST", *dst_md(), dims);
    compute::named_buffer_t stat_buffer("STAT", *stat_md(), stat_dims);
    compute::named_buffer_t ss_buffer
            = get_ss_buffer(weights_md(), dims.back());
    CHECK(init_conf_common(this, &conf, &rt_conf, engine, src_buffer,
            dst_buffer, stat_buffer, ss_buffer));
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto lws_strategy
            = compute::default_lws_strategy_t(compute_engine, gpu_attr);

    return status::success;
}

compute::kernel_ctx_t reusable_lnorm_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;

    data_type_t acc_dt = types::default_accum_data_type(src_dt, data_type::f32);
    data_type_t acc_bwd_dt
            = types::default_accum_data_type(dst_dt, data_type::f32);

    compute::data_type_converter_t converter;
    converter.register_type("ACC", acc_dt);
    converter.register_type("ACC_BWD", acc_bwd_dt);
    converter.def_kernel_macros(kernel_ctx);

    kernel_ctx.define_int("USE_SCALE", use_scale);
    kernel_ctx.define_int("USE_SHIFT", use_shift);

    kernel_ctx.define_int("WITH_SRC_SCALES", with_src_scale);
    kernel_ctx.define_int("WITH_DST_SCALES", with_dst_scale);

    gws_params.def_kernel_macros(kernel_ctx);
    stat_params.def_kernel_macros(kernel_ctx, "STAT");
    scaleshift_params.def_kernel_macros(kernel_ctx, "SS");
    return kernel_ctx;
}

void reusable_layer_normalization_fwd_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad_common(scratchpad, across_axis());
}

status_t reusable_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto &rt_conf = pd()->rt_conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    // Set up mean and variance pointers
    using namespace dnnl::impl::memory_tracking::names;
    std::unique_ptr<memory_storage_t> tmp_mean = nullptr;
    std::unique_ptr<memory_storage_t> tmp_var = nullptr;
    memory_storage_t *mean_ptr;
    memory_storage_t *variance_ptr;
    if (pd()->stats_are_tmp()) {
        tmp_mean = ctx.get_scratchpad_grantor().get_memory_storage(
                key_lnorm_tmp_mean);
        tmp_var = ctx.get_scratchpad_grantor().get_memory_storage(
                key_lnorm_tmp_var);
        mean_ptr = tmp_mean.get();
        variance_ptr = tmp_var.get();
    } else if (pd()->stats_are_src()) {
        mean_ptr = ctx.input(DNNL_ARG_MEAN)->memory_storage();
        variance_ptr = ctx.input(DNNL_ARG_VARIANCE)->memory_storage();
    } else {
        mean_ptr = ctx.output(DNNL_ARG_MEAN)->memory_storage();
        variance_ptr = ctx.output(DNNL_ARG_VARIANCE)->memory_storage();
    }

    if (!pd()->stats_are_src()) {
        // They must be computed, and stored in the tmp/output buffers above
        compute::kernel_arg_list_t calc_mean_arg_list;
        calc_mean_arg_list.append(src);
        calc_mean_arg_list.append(*mean_ptr);
        calc_mean_arg_list.append(pd()->norm_axis());
        calc_mean_arg_list.append(rt_conf.norm_stride);
        calc_mean_arg_list.append(rt_conf.stat_params.get());

        auto &nd_range_calc = rt_conf.stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_calc, calculate_mean_kernel_,
                calc_mean_arg_list));

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.append(src);
        calc_var_arg_list.append(*mean_ptr);
        calc_var_arg_list.append(*variance_ptr);
        calc_var_arg_list.append(pd()->norm_axis());
        calc_var_arg_list.append(rt_conf.norm_stride);
        calc_var_arg_list.append(rt_conf.stat_params.get());

        CHECK(parallel_for(ctx, nd_range_calc, calculate_variance_kernel_,
                calc_var_arg_list));
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(*mean_ptr);
    arg_list.append(*variance_ptr);
    arg_list.append(dst);
    arg_list.append(scale);
    arg_list.append(shift);
    arg_list.append(pd()->desc()->layer_norm_epsilon);
    arg_list.append(src_scale);
    arg_list.append(dst_scale);
    arg_list.append(rt_conf.gws_params.get());

    auto &nd_range = rt_conf.gws_params.nd_range;
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

/********** BWD implementation ***********/

status_t reusable_layer_normalization_bwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    size_t ndims = static_cast<size_t>(diff_dst_md()->ndims);
    std::vector<dim_idx_t> dims = get_dims(ndims);
    std::vector<dim_idx_t> stat_dims = get_dims(ndims, true);

    // BWD buffers:
    // - diff_dst: all dims
    // - diff_src: all dims (matches src)
    // - stat: (mean/variance) all but last dim
    // - SS: (scale/shift) just the last dim
    compute::named_buffer_t diff_dst_buffer("DST", *diff_dst_md(), dims);
    compute::named_buffer_t diff_src_buffer("SRC", *diff_src_md(), dims);
    compute::named_buffer_t stat_buffer("STAT", *stat_md(), stat_dims);
    compute::named_buffer_t ss_buffer
            = get_ss_buffer(diff_weights_md(), dims.back());

    CHECK(init_conf_common(this, &conf, &rt_conf, engine, diff_src_buffer,
            diff_dst_buffer, stat_buffer, ss_buffer));

    // The scaleshift kernel (called when conf.use_scale || conf.use_shift)
    // reduces stat dimensions, which requires elementwise-alignment between
    // src/stats over these dims. Skip cases where this is not the case
    if (conf.use_scale || conf.use_shift) {
        // OK to mutate diff_src_buffer
        diff_src_buffer.remove_dim(dims.back());
        if (diff_src_buffer.layout() != stat_buffer.layout()) {
            return status::unimplemented;
        }
    }

    return status::success;
}

status_t reusable_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto &rt_conf = pd()->rt_conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);

    if (pd()->use_scale() || pd()->use_shift()) {
        auto &diff_scale = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE);
        auto &diff_shift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SHIFT);

        compute::kernel_arg_list_t ss_arg_list;
        ss_arg_list.append(src);
        ss_arg_list.append(mean);
        ss_arg_list.append(variance);
        ss_arg_list.append(diff_dst);
        ss_arg_list.append(diff_scale);
        ss_arg_list.append(diff_shift);
        ss_arg_list.append(pd()->across_axis());
        ss_arg_list.append(rt_conf.stat_stride);
        ss_arg_list.append(pd()->desc()->layer_norm_epsilon);
        ss_arg_list.append(rt_conf.scaleshift_params.get());

        compute::nd_range_t nd_range = rt_conf.scaleshift_params.nd_range;
        CHECK(parallel_for(ctx, nd_range, scaleshift_kernel_, ss_arg_list));
    }

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);

    // We can't pass booleans to opencl kernels. Instead, pass an int8_t as 0/1
    // and inform the compiler that these are the only values it can have.
    int8_t include_stats = !pd()->stats_are_src() ? 1 : 0;

    compute::kernel_arg_list_t stat_arg_list;
    stat_arg_list.append(src);
    stat_arg_list.append(diff_dst);
    stat_arg_list.append(diff_src);
    stat_arg_list.append(scale);
    stat_arg_list.append(mean);
    stat_arg_list.append(variance);
    stat_arg_list.append(pd()->desc()->layer_norm_epsilon);
    stat_arg_list.append(rt_conf.norm_stride);
    stat_arg_list.append(pd()->norm_axis());
    stat_arg_list.append(include_stats);
    stat_arg_list.append(rt_conf.stat_params.get());

    compute::nd_range_t stat_nd_range = rt_conf.stat_params.nd_range;
    return parallel_for(ctx, stat_nd_range, kernel_, stat_arg_list);
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
