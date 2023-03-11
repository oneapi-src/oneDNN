/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/vectorized_lnorm.hpp"
#include "common/c_types_map.hpp"

#include "common/primitive_exec_types.hpp"
#include "common/scratchpad.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    // TODO: implement BWD pass
    if (!pd->is_fwd()) return status::unimplemented;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    // Limited due to performance reasons
    if (gpu_arch < compute::gpu_arch_t::xe_hpc) return status::unimplemented;

    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    int ndims = src_mdw.ndims();

    conf.data_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.norm_axis = pd->norm_axis();

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.stat_md_info = memory_desc_info_t::create(stat_mdw);

    conf.is_fwd = pd->is_fwd();

    conf.vectorize_calc_stats = false;
    conf.vect_dt_n = 1;
    conf.sub_group_size = 1;

    int c_block = 1;
    bool c_is_last_physical = false;
    if (src_mdw.blocking_desc().inner_nblks > 0) {
        c_block = src_mdw.blocking_desc()
                          .inner_blks[src_mdw.blocking_desc().inner_nblks - 1];
        c_is_last_physical
                = src_mdw.blocking_desc().inner_idxs[ndims - 1] == ndims - 1;
    } else {
        c_is_last_physical = src_mdw.blocking_desc().strides[ndims - 1] == 1;
    }

    if (!(src_mdw.is_dense() && c_is_last_physical && ndims < 4))
        return status::unimplemented;

    conf.dispatch = compute_engine->create_dispatch(
            pd->is_fwd() ? dst_mdw.md_ : src_mdw.md_);
    const auto &dims = pd->is_fwd() ? src_mdw.padded_dims() : dst_mdw.dims();

    auto eu_count = compute_engine->device_info()->eu_count();
    auto threads_per_eu
            = compute::device_info_t::threads_per_eu(gpu_arch, false);

    // Key feature of this version is single reading src data, keeping it in
    // private buffer and reusing this buffer for stats calculation.
    // It works for relatively short shapes.
    // Limit value experimentally selected, based on PVC perf data.
    const int nthr = pd->across_axis();
    const int nthr_on_ss
            = nstl::min(threads_per_eu, nstl::max(1, nthr / eu_count));
    const int src_buff_KB = nthr_on_ss * conf.norm_axis * sizeof(float) / 1024;
    int buff_size_limit = 128;
    if (conf.data_type == data_type::f16 || conf.data_type == data_type::bf16) {
        buff_size_limit *= 2;
    }
    if (src_buff_KB > buff_size_limit) return status::unimplemented;

    const int desired_sg_size = 32;
    auto mayiuse_sg = [=](const int sg_size) {
        return compute_engine->mayiuse_sub_group(sg_size)
                && compute_engine->mayiuse_block_reads_writes_with_sub_group(
                        sg_size);
    };

    if (pd->is_fwd()) {
        const int best_sg_size = [&]() {
            int size = desired_sg_size;
            while (size > 1) {
                const bool fit_to_shape = (conf.norm_axis % size == 0)
                        && (c_block == 1
                                || (c_block % size == 0 && ndims == 2));
                if (mayiuse_sg(size) && fit_to_shape) return size;
                size -= 16;
            }
            return size;
        }();
        if (best_sg_size <= 1) return status::unimplemented;

        conf.vectorize_calc_stats = true;
        conf.sub_group_size = best_sg_size;
        int vector_size = 8;
        while (conf.norm_axis % (conf.sub_group_size * vector_size) != 0) {
            vector_size /= 2;
        }
        while (c_block > 1 && vector_size * conf.sub_group_size > c_block) {
            vector_size /= 2;
        }
        conf.vect_dt_n = vector_size;

        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            if (i == ndims - 1) {
                dim = conf.sub_group_size;
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
                CHECK(conf.dispatch.vectorize_dim(
                        utils::format("X%d", i), conf.sub_group_size));
            } else
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
        }
    }
    conf.dispatch.generate();

    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.calculate_stats = !pd->stats_are_src();
    conf.save_stats = pd->is_training();
    conf.eps = pd->desc()->layer_norm_epsilon;

    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const lnorm_conf_t &conf) {
    kernel_ctx.set_data_type(conf.data_type);

    kernel_ctx.define_int("C", conf.norm_axis);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECTORIZE_CALC_STATS", conf.vectorize_calc_stats);
    kernel_ctx.define_int("VECTORIZE_BWD", conf.vectorize_bwd);
    kernel_ctx.define_int(
            "VECTORIZE_BWD_SCALESHIFT", conf.vectorize_bwd_scaleshift);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_dt_n);
    kernel_ctx.define_int(
            "VECTOR_SIZE_SCALESHIFT", conf.vector_size_scaleshift);
    kernel_ctx.define_int("N_CHUNK_SIZE", conf.n_chunk_size);
    kernel_ctx.define_int("N_CHUNKS", conf.n_chunks);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, conf.stat_md_info, "STAT");

    def_dispatch(kernel_ctx, conf.dispatch);
    if (!conf.is_fwd) {
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift);
        if (conf.vectorize_bwd_scaleshift)
            def_dispatch(kernel_ctx, conf.dispatch_scaleshift_finalize);
    }

    return status::success;
}

status_t vectorized_lnorm_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t vectorized_lnorm_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t vectorized_lnorm_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto &conf = pd()->conf;
    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, shift);
    arg_list.set(6, conf.eps);

    auto nd_range_kernel = conf.dispatch.nd_range();
    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
