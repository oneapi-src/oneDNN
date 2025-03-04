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

#include "gpu/intel/ocl/reusable_vectorized_lnorm.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/lnorm_utils.hpp"
#include "gpu/intel/ocl/reusable_lnorm.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

#include <vector>

using std::vector;

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::gpu::intel::compute;
struct single_subgroup_lws_strategy_t : public lws_strategy_t {
    size_t desired_sg_size = 32;
    single_subgroup_lws_strategy_t(const compute_engine_t *engine,
            const gpu_primitive_attr_t *gpu_attr, size_t _desired_sg_size)
        : lws_strategy_t(engine, gpu_attr)
        , desired_sg_size(_desired_sg_size) {};

    range_t create_lws(
            range_t &gws, const gws_bin_mapping_t &mapper) const override {
        range_t lws = {desired_sg_size, 1, 1};
        return lws;
    }

    // this strategy doesn't care which blocks are in the lws
    bool is_included(const mapped_block_t &blocks) const override {
        return false;
    }
};

bool is_sg_and_vector_size_compatible(
        const compute_engine_t *engine, int sg_size, int vector_size) {
    // Check if subgroup size is supported
    if (!engine->mayiuse_sub_group(sg_size)) return false;

    // Check if subgroup size is supported for block reads and writes
    if (!engine->mayiuse_block_reads_writes_with_sub_group(sg_size))
        return false;

    return true;
}

bool is_sg_stride_compatible(int norm_axis, int sg_stride) {
    // Check if norm_axis size is less than the number of elements read by the subgroup
    if (norm_axis < sg_stride) return false;

    // Check if norm_axis is a multiple of the subgroup size and vector size
    if (norm_axis % sg_stride != 0) return false;

    return true;
}

static status_t init_conf_common(const layer_normalization_pd_t *pd,
        reusable_vectorized_lnorm_params_t *conf,
        reusable_vectorized_lnorm_runtime_params_t *rt_conf,
        const impl::engine_t *engine, const compute::named_buffer_t &input_buf,
        const compute::named_buffer_t &output_buf,
        const compute::named_buffer_t &stat_buf,
        const compute::named_buffer_t &ss_buf) {
    conf->use_scale = pd->use_scale();
    conf->use_shift = pd->use_shift();
    conf->input_dt = input_buf.data_type;
    conf->output_dt = output_buf.data_type;
    conf->ss_dt = ss_buf.data_type;
    conf->calculate_stats = !pd->stats_are_src();
    conf->save_stats = pd->is_training();

    auto scales = pd->attr()->scales_;

    // We require that the lnorm axis is a single dense block, so that it can
    // be represented by a stride + size alone.
    size_t ndims = into<size_t>(input_buf.ndims);
    vector<dim_idx_t> dims = get_dims(ndims);

    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper dst_mdw(pd->dst_md());
    if (src_mdw.blocking_desc().inner_nblks != 0
            || dst_mdw.blocking_desc().inner_nblks != 0) {
        VDEBUGINFO(15, primitive, lnorm,
                "Reusable Vectorized LNorm not used because source or "
                "destination tensors have blocked memory layouts.");
        return status::unimplemented;
    }

    bool c_is_last_physical = src_mdw.blocking_desc().strides[ndims - 1] == 1;
    if (!(src_mdw.is_dense() && c_is_last_physical)) {
        VDEBUGINFO(15, primitive, lnorm,
                "Reusable Vectorized LNorm not used because the source tensor "
                "is not dense(%s) or the last axis(stride[ndims-1] = %d) "
                "is not continuous.",
                src_mdw.is_dense() ? "true" : "false",
                int(src_mdw.blocking_desc().strides[ndims - 1]));
        return status::unimplemented;
    }

    const auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
            pd->attr()->gpu_attr_.get());

    const auto *compute_engine
            = utils::downcast<const compute::compute_engine_t *>(engine);

    conf->sg_size = 0;
    conf->vector_size = 0;
    bool found_compatible_sg_and_vector_size = false;
    for (int sg_size : {32, 16}) {
        for (int vector_size : {8, 4, 2, 1}) {
            bool sg_and_vector_size_ok = is_sg_and_vector_size_compatible(
                    compute_engine, sg_size, vector_size);
            bool sg_stride_ok = is_sg_stride_compatible(
                    into<dim_idx_t>(pd->norm_axis()), sg_size * vector_size);

            if (sg_and_vector_size_ok && sg_stride_ok) {
                conf->sg_size = sg_size;
                conf->vector_size = vector_size;
                found_compatible_sg_and_vector_size = true;
                break;
            }
        }
        if (found_compatible_sg_and_vector_size) break;
    }

    if (found_compatible_sg_and_vector_size == false) {
        VDEBUGINFO(15, primitive, lnorm,
                "Reusable Vectorized LNorm not used because norm_axis(%ld) "
                "is not a multiple of the vector size and subgroup size.",
                long(pd->norm_axis()));
        return status::unimplemented;
    }

    conf->unroll = std::min<int>(
            4, (int)pd->norm_axis() / (conf->sg_size * conf->vector_size));

    // Norm dispatch: all dimensions
    auto lws_strategy = single_subgroup_lws_strategy_t(
            compute_engine, gpu_attr, conf->sg_size);

    compute::reusable_dispatch_config_t dispatch_config(
            compute_engine, std::move(dims));
    CHECK(dispatch_config.register_buffer(input_buf));
    CHECK(dispatch_config.register_buffer(output_buf));
    CHECK(dispatch_config.register_buffer(stat_buf));
    CHECK(dispatch_config.register_buffer(ss_buf));

    compute::reusable_dispatch_t dispatch;
    CHECK(dispatch_config.generate(dispatch, lws_strategy));
    conf->gws_params = dispatch.get_compile_params();
    rt_conf->gws_params = dispatch.get_runtime_params();

    return status::success;
}

status_t reusable_vectorized_layer_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    size_t ndims = static_cast<size_t>(src_md()->ndims);
    vector<dim_idx_t> dims = get_dims(ndims);
    vector<dim_idx_t> stat_dims = get_dims(ndims, true);

    //init_scratchpad();
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

    return status::success;
}

compute::kernel_ctx_t
reusable_vectorized_lnorm_params_t::get_kernel_ctx() const {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.set_data_type(input_dt);
    def_data_type(kernel_ctx, input_dt, "SRC");
    def_data_type(kernel_ctx, ss_dt, "WEI");
    def_data_type(kernel_ctx, output_dt, "DST");

    kernel_ctx.define_int("USE_SCALE", use_scale);
    kernel_ctx.define_int("USE_SHIFT", use_shift);

    kernel_ctx.define_int("CALCULATE_STATS", calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", save_stats && calculate_stats);

    kernel_ctx.define_int("SG_SIZE", sg_size);
    kernel_ctx.define_int("VECT_DT_N", vector_size);
    kernel_ctx.define_int("SG_STRIDE", sg_size * vector_size);
    kernel_ctx.define_int("N_UNROLL", unroll);

    gws_params.def_kernel_macros(kernel_ctx);

    return kernel_ctx;
}

status_t reusable_vectorized_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto &rt_conf = pd()->rt_conf;
    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    memory_storage_t &mean = pd()->stats_are_src()
            ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
            : CTX_OUT_STORAGE(DNNL_ARG_MEAN);
    memory_storage_t &variance = pd()->stats_are_src()
            ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    compute::kernel_arg_list_t lnorm_arg_list;
    lnorm_arg_list.append(src);
    lnorm_arg_list.append(mean);
    lnorm_arg_list.append(variance);
    lnorm_arg_list.append(pd()->norm_axis());
    lnorm_arg_list.append(dst);
    lnorm_arg_list.append(scale);
    lnorm_arg_list.append(shift);
    lnorm_arg_list.append(pd()->desc()->layer_norm_epsilon);
    lnorm_arg_list.append(src_scale);
    lnorm_arg_list.append(dst_scale);
    lnorm_arg_list.append((int)utils::div_up(
            pd()->norm_axis(), conf.sg_size * conf.vector_size));
    lnorm_arg_list.append(1.f / (pd()->norm_axis()));

    lnorm_arg_list.append(rt_conf.gws_params.get());

    compute::nd_range_t gws_nd_range_calc(
            {static_cast<size_t>(conf.sg_size),
                    rt_conf.gws_params.nd_range.global_range().data()[1],
                    rt_conf.gws_params.nd_range.global_range().data()[2]},
            {static_cast<size_t>(conf.sg_size), 1, 1});

    return parallel_for(
            ctx, gws_nd_range_calc, calculate_lnorm_kernel_, lnorm_arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
