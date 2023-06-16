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

using namespace compute;

static status_t init_conf_common(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();

    // Limited due to performance reasons
    // Vectorized implementation can be used for FWD on DG2+, for BWD on PVC+
    if (gpu_arch < (pd->is_fwd() ? gpu_arch_t::xe_hpg : gpu_arch_t::xe_hpc))
        return status::unimplemented;

    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    int ndims = src_mdw.ndims();

    conf.data_type = src_mdw.data_type();
    conf.ndims = ndims;
    conf.norm_axis = pd->norm_axis();
    conf.across_axis = pd->across_axis();
    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.calculate_stats = !pd->stats_are_src();
    conf.save_stats = pd->is_training();
    conf.eps = pd->desc()->layer_norm_epsilon;

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.stat_md_info = memory_desc_info_t::create(stat_mdw);

    conf.is_fwd = pd->is_fwd();

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

    conf.dispatch_scaleshift = compute_engine->create_dispatch();
    conf.dispatch_scaleshift_finalize = compute_engine->create_dispatch();
    conf.dispatch = compute_engine->create_dispatch(
            conf.is_fwd ? dst_mdw.md_ : src_mdw.md_);
    const auto &dims = conf.is_fwd ? src_mdw.padded_dims() : dst_mdw.dims();

    auto eu_count = compute_engine->device_info()->eu_count();
    auto threads_per_eu = device_info_t::threads_per_eu(gpu_arch, false);
    auto max_eus_per_wg = device_info_t::max_eus_per_wg(gpu_arch);
    const int max_ss = utils::div_up(eu_count, max_eus_per_wg);
    const size_t max_wg_size = compute_engine->device_info()->max_wg_size();

    // Vectorized vs Reference heuristics.
    // PVC, FWD:
    // Key feature of this version is single reading src data, keeping it in
    // private buffer and reusing this buffer for stats calculation.
    // It works for relatively short shapes.
    // PVC, BWD:
    // Because of the sync overhead, the vectorized version performs worse
    // on the shapes with short sizes by across dimensions.
    // ATSM, FWD: vectorized version is faster w/o conditions.
    // Limit values experimentally selected, based on perf data.

    if (gpu_arch >= gpu_arch_t::xe_hpc) {
        if (conf.is_fwd) {
            const int nthr = conf.across_axis;
            const int nthr_on_ss
                    = nstl::min(threads_per_eu, nstl::max(1, nthr / eu_count));
            const int src_buff_KB
                    = nthr_on_ss * conf.norm_axis * sizeof(float) / 1024;
            int buff_size_limit = 128;
            if (conf.data_type == data_type::f16
                    || conf.data_type == data_type::bf16) {
                buff_size_limit *= 2;
            }
            if (src_buff_KB > buff_size_limit) return status::unimplemented;
        } else {
            const int N_dim_limit = 64;
            if (conf.across_axis < N_dim_limit) return status::unimplemented;
        }
    }

    // DG2/ATSM optimization specifics - dont use private SRC buffer
    conf.use_src_buffer = gpu_arch >= gpu_arch_t::xe_hpc;

    const int desired_sg_size = 32;
    auto mayiuse_sg = [=](const int sg_size) {
        return compute_engine->mayiuse_sub_group(sg_size)
                && compute_engine->mayiuse_block_reads_writes_with_sub_group(
                        sg_size);
    };

    if (conf.is_fwd) {
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

        conf.sub_group_size = best_sg_size;
        int vector_size = 8;
        while (conf.norm_axis % (conf.sub_group_size * vector_size) != 0) {
            vector_size /= 2;
        }
        while (c_block > 1 && vector_size * conf.sub_group_size > c_block) {
            vector_size /= 2;
        }
        conf.vect_dt_n = vector_size;

        // Splitting normalized dimension into parts increases HW utilization
        // and can improve performance for some kinds of short shapes
        const int num_wgs = conf.across_axis;
        auto get_ss_utilization = [](const int num_wgs, const int max_ss) {
            return (float)num_wgs / max_ss;
        };
        const int max_num_blocks = gpu_arch >= gpu_arch_t::xe_hpc ? 8 : 16;
        const int min_num_blocks = 4;
        const int best_num_blocks = [=]() {
            int num_blocks = 1;
            int best_num = num_blocks;
            float ss_util = get_ss_utilization(num_wgs, max_ss);
            if (ss_util > 1.0f || c_block > 1) { return best_num; }
            while (num_blocks <= max_num_blocks) {
                if (conf.norm_axis
                                % (conf.sub_group_size * conf.vect_dt_n
                                        * num_blocks)
                        == 0) {
                    best_num = num_blocks;
                }
                num_blocks++;
            }
            return best_num < min_num_blocks ? 1 : best_num;
        }();
        conf.num_norm_blocks = best_num_blocks;
        conf.norm_block = conf.norm_axis / best_num_blocks;
        assert(conf.norm_axis % conf.norm_block == 0);
        assert(conf.norm_block % (conf.sub_group_size * conf.vect_dt_n) == 0);

        const size_t norm_gws = conf.sub_group_size * conf.num_norm_blocks;
        assert(norm_gws <= max_wg_size);
        MAYBE_UNUSED(max_wg_size);

        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            size_t dim = (i < ndims - 1) ? dims[i] : 1;
            if (i == ndims - 1) {
                dim = norm_gws;
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
                CHECK(conf.dispatch.vectorize_dim(
                        utils::format("X%d", i), conf.sub_group_size));
            } else
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
        }
        conf.dispatch.generate();
        const size_t tuned_lws[3] = {norm_gws, 1, 1};
        conf.dispatch.set_lws(tuned_lws);

    } else { // bwd

        const int best_sg_size = [&]() {
            int size = desired_sg_size;
            while (size > 1) {
                const bool fit_to_shape = conf.norm_axis % size == 0
                        && (src_mdw.matches_one_of_tag(ab, abc, abcd, abcde)
                                || (ndims == 2 && c_block % size == 0));
                if (mayiuse_sg(size) && fit_to_shape) return size;
                size -= 16;
            }
            return size;
        }();
        if (best_sg_size <= 1) return status::unimplemented;

        conf.sub_group_size = best_sg_size;
        conf.vect_dt_n = 8;
        while (conf.norm_axis % (conf.sub_group_size * conf.vect_dt_n) != 0) {
            conf.vect_dt_n /= 2;
        }
        while (src_mdw.blocking_desc().inner_nblks > 0
                && c_block % (conf.sub_group_size * conf.vect_dt_n) != 0) {
            conf.vect_dt_n /= 2;
        }

        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            int dim = (i < ndims - 1) ? dims[i] : 1;
            if (i == ndims - 1) {
                conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx,
                        conf.sub_group_size);
                CHECK(conf.dispatch.vectorize_dim(
                        utils::format("X%d", i), conf.sub_group_size));
            } else {
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
            }
        }

        conf.n_chunk_size = 1;
        conf.vector_size_scaleshift = 1;
        conf.n_chunks = dims[0] / conf.n_chunk_size;

        // Scaleshift vectorization is supported for tensors
        // with shapes AxB, 1xBxC
        const bool vectorize_bwd_scaleshift = true
                && stat_mdw.matches_one_of_tag(a, ab)
                && ((ndims == 2
                            && (c_block == best_sg_size
                                    || src_mdw.matches_tag(ab)))
                        || (ndims == 3 && src_mdw.matches_tag(abc)
                                && dims[0] == 1));
        if ((conf.use_scale || conf.use_shift) && !vectorize_bwd_scaleshift)
            return status::unimplemented;

        const int first_dim = ndims == 2 ? dims[0] : dims[1];
        int desired_n_chunk_size = 16; // Experimentally selected values
        while (first_dim % desired_n_chunk_size != 0) {
            desired_n_chunk_size /= 2;
        }
        conf.n_chunk_size = desired_n_chunk_size;
        conf.n_chunks = first_dim / conf.n_chunk_size;

        // Scaleshift kernel does partial reduction of N
        conf.dispatch_scaleshift.define_dim("N", conf.n_chunks);
        conf.dispatch_scaleshift.define_dim(
                "C", utils::div_up(conf.norm_axis, conf.vect_dt_n));
        CHECK(conf.dispatch_scaleshift.vectorize_dim("C", conf.sub_group_size));
        conf.dispatch_scaleshift.set_kernel_attr_suffix("SCALESHIFT");
        conf.dispatch_scaleshift.generate();

        // Scaleshift finalize kernel reduces results of scaleshift kernel
        conf.dispatch_scaleshift_finalize.define_dim(
                "C_finalize", conf.norm_axis);
        const int max_n_finalize = 256; // Experimentally selected values
        size_t n_finalize = conf.n_chunks;
        while (n_finalize > max_n_finalize) {
            n_finalize /= 2;
        }
        conf.dispatch_scaleshift_finalize.define_dim_with_nesting_level(
                "N_finalize", 1, n_finalize);
        conf.dispatch_scaleshift_finalize.set_kernel_attr_suffix(
                "SCALESHIFT_FINALIZE");
        conf.dispatch_scaleshift_finalize.generate();
        const size_t tuned_lws[3] = {n_finalize, 1, 1};
        conf.dispatch_scaleshift_finalize.set_lws(tuned_lws);
    }
    conf.dispatch.generate();

    return status::success;
}

static status_t init_kernel_ctx_common(
        kernel_ctx_t &kernel_ctx, const lnorm_conf_t &conf) {
    kernel_ctx.set_data_type(conf.data_type);

    // Since FWD kernel aggressively uses GRF (allocates a private buffer for
    // SRC chunk), large GRF mode decreases number/probability of register
    // spills.
    if (conf.is_fwd) kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    kernel_ctx.define_int("C", conf.norm_axis);
    kernel_ctx.define_int("NORM_BLOCK", conf.norm_block);
    kernel_ctx.define_int("NUM_NORM_BLOCKS", conf.num_norm_blocks);
    kernel_ctx.define_int("N", conf.across_axis);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_dt_n);
    kernel_ctx.define_int(
            "VECTOR_SIZE_SCALESHIFT", conf.vector_size_scaleshift);
    kernel_ctx.define_int("N_CHUNK_SIZE", conf.n_chunk_size);
    kernel_ctx.define_int("N_CHUNKS", conf.n_chunks);
    kernel_ctx.define_int("USE_SRC_BUFFER", conf.use_src_buffer);

    kernel_ctx.add_option("-cl-std=CL2.0");
    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, conf.stat_md_info, "STAT");

    def_dispatch(kernel_ctx, conf.dispatch);
    if (!conf.is_fwd) {
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift);
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift_finalize);
    }

    return status::success;
}

status_t vectorized_lnorm_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t vectorized_lnorm_fwd_t::pd_t::init_kernel_ctx(
        kernel_ctx_t &kernel_ctx) const {
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

    kernel_arg_list_t arg_list;
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

status_t vectorized_lnorm_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t vectorized_lnorm_bwd_t::pd_t::init_kernel_ctx(
        kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

void vectorized_lnorm_bwd_t::pd_t::init_scratchpad() {
    const size_t size = conf.n_chunks * conf.norm_axis * 2;
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_lnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t vectorized_lnorm_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);

    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    auto &diff_scale = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE);
    auto &diff_shift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SHIFT);

    if (conf.use_scale || conf.use_shift) {
        auto temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_lnorm_reduction);

        kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, mean);
        arg_list.set(2, variance);
        arg_list.set(3, diff_dst);
        arg_list.set(4, *temp_reduce);
        arg_list.set(5, *temp_reduce);
        arg_list.set(6, conf.eps);

        auto nd_range = conf.dispatch_scaleshift.nd_range();
        status = parallel_for(ctx, nd_range, kernel_scaleshift_, arg_list);
        if (status != status::success) return status;

        kernel_arg_list_t arg_list_final;
        arg_list_final.set(0, *temp_reduce);
        arg_list_final.set(1, diff_scale);
        arg_list_final.set(2, diff_shift);

        auto nd_range_finalize = conf.dispatch_scaleshift_finalize.nd_range();
        status = parallel_for(ctx, nd_range_finalize,
                kernel_scaleshift_finalize_, arg_list_final);
        if (status != status::success) return status;
    }

    kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scale);
    arg_list.set(5, diff_src);
    arg_list.set(6, conf.eps);

    auto nd_range_kernel = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
