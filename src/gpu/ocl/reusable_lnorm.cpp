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

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"
#include "gpu/compute/dispatch_reusable.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/reusable_lnorm.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace lnorm_dims {
compute::dim_id_t mb = 0;
compute::dim_id_t ic = 1;
compute::dim_id_t sp0 = 2;
compute::dim_id_t sp1 = 3;
compute::dim_id_t sp2 = 4;
}; // namespace lnorm_dims

static std::vector<compute::dim_id_t> get_dims(
        size_t ndims, bool for_stats = false) {
    assert(ndims > 1 && ndims < 6);
    // The last logical dimension is not included in lnorm stats
    if (for_stats) ndims--;
    std::vector<compute::dim_id_t> ret(ndims);
    uint8_t idx = 0;
    ret[idx++] = lnorm_dims::mb;
    if (ndims >= 2) ret[idx++] = lnorm_dims::ic;
    if (ndims >= 3) ret[idx++] = lnorm_dims::sp0;
    if (ndims >= 4) ret[idx++] = lnorm_dims::sp1;
    if (ndims >= 5) ret[idx++] = lnorm_dims::sp2;
    return ret;
}

status_t reusable_layer_normalization_fwd_t::pd_t::init_conf(engine_t *engine) {
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());

    conf.use_scale = use_scale();
    conf.use_shift = use_shift();
    conf.src_dt = src_md()->data_type;
    conf.dst_dt = dst_md()->data_type;

    // If scale/shift are not used, weights_md()->data_type is undefined,
    // which causes problems. Set it to float in this case, since it's
    // unused anyway.
    conf.ss_dt = (conf.use_scale || conf.use_shift) ? weights_md()->data_type
                                                    : data_type::f32;

    auto scales = attr()->scales_;
    conf.with_src_scale = !scales.get(DNNL_ARG_SRC).has_default_values();
    conf.with_dst_scale = !scales.get(DNNL_ARG_DST).has_default_values();

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    size_t ndims = static_cast<size_t>(src_md()->ndims);
    std::vector<compute::dim_id_t> dims = get_dims(ndims);

    // SRC: all dimensions
    compute::named_buffer_t src_buffer("SRC", *src_md(), dims);

    // Only allow SRC formats where the last dimension exists in a dense block
    const block_t *reduction_block = nullptr;
    for (const block_t &block : src_buffer.layout()) {
        if (gpu_utils::into<size_t>(block.dim_idx) == dims.back()) {
            if (reduction_block) return status::unimplemented;
            reduction_block = &block;
        }
    }
    assert(reduction_block);
    rt_conf.reduction_stride = reduction_block->stride;

    // DST: all dimensions (may differ from SRC's format)
    compute::named_buffer_t dst_buffer("DST", *dst_md(), dims);

    // STAT: all but the last dimension (used for mean/variance)
    compute::named_buffer_t stat_buffer(
            "STAT", *stat_md(), get_dims(ndims, true));

    // LINEAR: just the last dimension (used for scale/shift)
    compute::named_buffer_t linear_buffer("LINEAR", src_buffer);
    for (size_t i = 0; i < dims.size() - 1; i++) {
        linear_buffer.remove_dim(dims[i]);
    }

    // full lnorm dispatch: all dimensions
    auto lws_strategy
            = compute::default_lws_strategy_t(compute_engine, gpu_attr);
    compute::reusable_dispatch_config_t dispatch_config(compute_engine, dims);
    CHECK(dispatch_config.register_buffer(src_buffer));
    CHECK(dispatch_config.register_buffer(dst_buffer));
    CHECK(dispatch_config.register_buffer(stat_buffer));
    CHECK(dispatch_config.register_buffer(linear_buffer));
    compute::reusable_dispatch_t dispatch;

    CHECK(dispatch_config.generate(dispatch, lws_strategy));
    conf.gws_params = dispatch.get_compile_params();
    rt_conf.gws_params = dispatch.get_runtime_params();

    // stat calculation dispatch: all stat dimensions
    compute::reusable_dispatch_config_t calc_stat_dispatch_config(
            compute_engine, stat_buffer.get_dim_ids());
    CHECK(calc_stat_dispatch_config.register_buffer(src_buffer));
    CHECK(calc_stat_dispatch_config.register_buffer(stat_buffer));

    compute::reusable_dispatch_t dispatch_calc_stat;
    CHECK(calc_stat_dispatch_config.generate(dispatch_calc_stat, lws_strategy));
    conf.calc_stat_params = dispatch_calc_stat.get_compile_params();
    rt_conf.calc_stat_params = dispatch_calc_stat.get_runtime_params();

    return status::success;
}

static void init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const reusable_lnorm_params_t &conf) {
    kernel_ctx.set_data_type(conf.src_dt);
    def_data_type(kernel_ctx, conf.dst_dt, "DST");
    def_data_type(kernel_ctx, conf.ss_dt, "WEI");

    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);

    kernel_ctx.define_int("WITH_SRC_SCALES", conf.with_src_scale);
    kernel_ctx.define_int("WITH_DST_SCALES", conf.with_dst_scale);

    conf.gws_params.def_kernel_macros(kernel_ctx);
    conf.calc_stat_params.def_kernel_macros(kernel_ctx, "CALC");
}

compute::kernel_ctx_t reusable_lnorm_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    init_kernel_ctx_common(kernel_ctx, *this);
    return kernel_ctx;
}

void reusable_layer_normalization_fwd_t::pd_t::init_scratchpad() {
    if (stats_are_tmp()) {
        auto scratchpad = scratchpad_registry().registrar();
        using namespace memory_tracking::names;
        for (auto key : {key_lnorm_tmp_mean, key_lnorm_tmp_var}) {
            scratchpad.book(key, gpu_utils::into<size_t>(across_axis()),
                    types::data_type_size(data_type::f32),
                    OCL_BUFFER_ALIGNMENT);
        }
    }
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
        calc_mean_arg_list.append(rt_conf.reduction_stride);
        calc_mean_arg_list.append(rt_conf.calc_stat_params.get());

        auto &nd_range_calc = rt_conf.calc_stat_params.nd_range;

        CHECK(parallel_for(ctx, nd_range_calc, calculate_mean_kernel_,
                calc_mean_arg_list));

        compute::kernel_arg_list_t calc_var_arg_list;
        calc_var_arg_list.append(src);
        calc_var_arg_list.append(*mean_ptr);
        calc_var_arg_list.append(*variance_ptr);
        calc_var_arg_list.append(pd()->norm_axis());
        calc_var_arg_list.append(rt_conf.reduction_stride);
        calc_var_arg_list.append(rt_conf.calc_stat_params.get());

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

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
