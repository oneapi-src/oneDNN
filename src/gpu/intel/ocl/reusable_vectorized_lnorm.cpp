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
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/ocl/reusable_lnorm.hpp"
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

    range_t create_lws(const range_t &gws,
            const gws_bin_mapping_t &mapper) const override {
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

    int select_work_group_kernel = gpu_utils::dev_getenv("work_group", -1);
    select_work_group_kernel
            = select_work_group_kernel != -1 && select_work_group_kernel;
    int max_unroll = gpu_utils::dev_getenv("unroll", -1);
    if (max_unroll == -1) { max_unroll = 12; }
    int reuse = gpu_utils::dev_getenv("reuse", -1);
    int reuse_private_mem = gpu_utils::dev_getenv("reuse_private_mem", 999);

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
    size_t ndims = gpu_utils::into<size_t>(input_buf.ndims);
    vector<compute::dim_id_t> dims = get_dims(ndims);

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

    const auto *compute_engine
            = utils::downcast<const compute::compute_engine_t *>(engine);
    const auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
            pd->attr()->gpu_attr_.get());

    size_t lnorm_bytes
            = pd->norm_axis() * types::data_type_size(input_buf.data_type);
    size_t tensor_size
            = src_mdw.nelems() * types::data_type_size(input_buf.data_type);
    size_t cache_size = compute_engine->device_info()->l3_cache_size();

    std::unique_ptr<lws_strategy_t> lws_strategy;
    conf->sg_size = 0;
    conf->vector_size = 0;
    bool found_compatible_sg_and_vector_size = false;

    for (int sg_size : {32, 16}) {
        for (int vector_size : {8, 4, 2, 1}) {
            bool sg_and_vector_size_ok = is_sg_and_vector_size_compatible(
                    compute_engine, sg_size, vector_size);
            bool group_stride_ok = is_sg_stride_compatible(
                    pd->norm_axis(), sg_size * vector_size);

            if (sg_and_vector_size_ok && group_stride_ok) {
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

    float threads_launched_for_wg_kernel = (float)pd->norm_axis()
            / conf->vector_size * (src_mdw.nelems() / pd->norm_axis())
            / conf->sg_size;
    float threads_launched_for_sg_kernel = (src_mdw.nelems() / pd->norm_axis());
    auto di = compute_engine->device_info();

    //printf("sg: %f wg: %f ", threads_launched_for_sg_kernel, threads_launched_for_wg_kernel);
    //printf("sg waves: %f wg waves:%f \n", di->eu_count()*8/threads_launched_for_sg_kernel, di->eu_count()*8/threads_launched_for_wg_kernel);

    int large_grf = gpu_utils::dev_getenv("large_grf", 0);
    float sg_waves = threads_launched_for_sg_kernel / di->hw_threads(large_grf);
    float wg_waves = threads_launched_for_wg_kernel / di->hw_threads(large_grf);
    float sg_full_waves, wg_full_waves;
    float sg_partial_waves = modf(sg_waves, &sg_full_waves);
    float wg_partial_waves = modf(wg_waves, &wg_full_waves);

    //printf("hw_threads(): %d\n", di->hw_threads());
    //printf("sg full: %f partial: %f wg full:%f partial: %f| ", sg_full_waves, sg_partial_waves, wg_full_waves, wg_partial_waves);
    int num_sg_per_wg
            = (int)pd->norm_axis() / (conf->sg_size * conf->vector_size);
    if (sg_waves < 0.8) {
        select_work_group_kernel = wg_waves > sg_waves && num_sg_per_wg > 4;
    } else {
        select_work_group_kernel = false;
    }

    if ((pd->norm_axis() / conf->vector_size)
            > compute_engine->device_info()->max_wg_size(false)) {
        select_work_group_kernel = false;
    }


    //printf("wg_waves(%f) > sg_waves(%f) && num_sg_per_wg(%d) > 4 || ((tensor_size(%zu) / cache_size(%zu) >= 0.75) && lnorm_bytes(%zu) > 1024)", wg_waves, sg_waves , num_sg_per_wg, tensor_size , cache_size , lnorm_bytes);
    if ((select_work_group_kernel || (tensor_size / cache_size >= 0.75))
            && lnorm_bytes > 1280 && !(num_sg_per_wg == 1)) {
        //printf("WG: ");
        conf->unroll = 1;
        conf->private_mem_size = 0;
        conf->large_grf = false;
        conf->wg_size = pd->norm_axis() / conf->vector_size;
        lws_strategy.reset(
                new default_lws_strategy_t(compute_engine, gpu_attr));

        conf->reuse = true;
    } else {
        //printf("SG: ");
        conf->unroll = std::min<int>(max_unroll,
                (int)pd->norm_axis() / (conf->sg_size * conf->vector_size));

        conf->wg_size = conf->sg_size;
        conf->large_grf = sg_waves < 0.8 || conf->unroll > 2;

        //printf("GRF: %d ", conf->large_grf);
        conf->private_mem_size
                = pd->norm_axis() / (conf->sg_size * conf->vector_size);
        lws_strategy.reset(new single_subgroup_lws_strategy_t(
                compute_engine, gpu_attr, conf->sg_size));
    }

    compute::reusable_dispatch_config_t dispatch_config(compute_engine, dims);
    CHECK(dispatch_config.register_buffer(input_buf));
    CHECK(dispatch_config.register_buffer(output_buf));
    CHECK(dispatch_config.register_buffer(stat_buf));
    CHECK(dispatch_config.register_buffer(ss_buf));

    compute::reusable_dispatch_t dispatch;
    CHECK(dispatch_config.generate(dispatch, *lws_strategy));
    conf->gws_params = dispatch.get_compile_params();
    rt_conf->gws_params = dispatch.get_runtime_params();

    return status::success;
}

status_t reusable_vectorized_layer_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    size_t ndims = static_cast<size_t>(src_md()->ndims);
    vector<compute::dim_id_t> dims = get_dims(ndims);
    vector<compute::dim_id_t> stat_dims = get_dims(ndims, true);

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
    kernel_ctx.add_option("-cl-std=CL3.0");

    if (large_grf) { kernel_ctx.add_option("-cl-intel-256-GRF-per-thread"); }

    def_data_type(kernel_ctx, input_dt, "SRC");
    def_data_type(kernel_ctx, ss_dt, "WEI");
    def_data_type(kernel_ctx, output_dt, "DST");

    kernel_ctx.define_int("USE_SCALE", use_scale);
    kernel_ctx.define_int("USE_SHIFT", use_shift);

    kernel_ctx.define_int("CALCULATE_STATS", calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", save_stats && calculate_stats);

    kernel_ctx.define_int("SG_SIZE", sg_size);
    kernel_ctx.define_int("VECT_DT_N", vector_size);
    kernel_ctx.define_int("N_UNROLL", unroll);
    kernel_ctx.define_int("REUSE", reuse);
    kernel_ctx.define_int("PVT_MEM_SIZE", private_mem_size);

    kernel_ctx.define_int("WG_SIZE", wg_size);
    if (reuse) {
        kernel_ctx.add_option("-DGROUP_ADD=work_group_reduce_add");
        kernel_ctx.define_int("GROUP_STRIDE", INT32_MAX);
    } else {
        kernel_ctx.add_option("-DGROUP_ADD=sub_group_reduce_add");
        kernel_ctx.define_int("GROUP_STRIDE", sg_size * vector_size);
    }

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
    if (conf.reuse) {
        lnorm_arg_list.append(1);
    } else {
        lnorm_arg_list.append((int)utils::div_up(
                pd()->norm_axis(), conf.sg_size * conf.vector_size));
    }
    lnorm_arg_list.append(1.f / (pd()->norm_axis()));

    lnorm_arg_list.append(rt_conf.gws_params.get());

    //printf("reuse: %d|  ", conf.reuse);
    if (conf.reuse) {
        compute::nd_range_t gws_nd_range_calc(
                {static_cast<size_t>(pd()->norm_axis() / conf.vector_size),
                        rt_conf.gws_params.nd_range.global_range().data()[1],
                        rt_conf.gws_params.nd_range.global_range().data()[2]},
                {static_cast<size_t>(pd()->norm_axis() / conf.vector_size), 1,
                        1});

        //std::cout << "rt_conf.gws_params.nd_range.str():  "
        //<< rt_conf.gws_params.nd_range.str() << "\n";

        //std::cout << "gws_nd_range_calc: " << gws_nd_range_calc.str() << " vector size: " << conf.vector_size <<  "\n";
        return parallel_for(ctx, gws_nd_range_calc, calculate_lnorm_kernel_,
                lnorm_arg_list);
    } else {
        compute::nd_range_t gws_nd_range_calc(
                {static_cast<size_t>(conf.sg_size),
                        //std::max<size_t>((size_t)3587, rt_conf.gws_params.nd_range.global_range().data()[1]),
                        rt_conf.gws_params.nd_range.global_range().data()[1],
                        rt_conf.gws_params.nd_range.global_range().data()[2]},
                {static_cast<size_t>(conf.sg_size), 1, 1});

        //std::cout << "gws_nd_range_calc: " << gws_nd_range_calc.str() << " vector size: " << conf.vector_size <<  "\n";
        return parallel_for(ctx, gws_nd_range_calc, calculate_lnorm_kernel_,
                lnorm_arg_list);
    }
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
