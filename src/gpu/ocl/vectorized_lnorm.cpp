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
using namespace dnnl::impl::format_tag;

bool mayiuse_sg(const int sg_size, engine_t *engine) {
    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);
    return compute_engine->mayiuse_sub_group(sg_size)
            && compute_engine->mayiuse_block_reads_writes_with_sub_group(
                    sg_size);
};

bool is_fused_kernel_applicable(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, engine_t *engine,
        bool large_grf_mode) {
    auto *compute_engine = utils::downcast<compute_engine_t *>(engine);

    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    memory_desc_wrapper src_mdw(pd->src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    auto eu_count = compute_engine->device_info()->eu_count();
    auto max_eus_per_wg = device_info_t::max_eus_per_wg(gpu_arch);
    const int max_ss = utils::div_up(eu_count, max_eus_per_wg);
    const size_t max_wg_size
            = compute_engine->device_info()->max_wg_size(large_grf_mode);
    const size_t max_slm_size = device_info_t::max_slm_size(gpu_arch);

    // Plain layout only
    const bool is_plain = src_mdw.matches_one_of_tag(ab, abc)
            && stat_mdw.matches_one_of_tag(a, ab)
            // kernel does not support M x 1 x N layout
            && IMPLICATION(src_mdw.ndims() == 3, src_mdw.dims()[1] != 1);
    if (!is_plain) return false;

    const int desired_sg_size = 16; // based on PVC performance data
    conf.sub_group_size = mayiuse_sg(desired_sg_size, engine)
                    && conf.norm_axis % desired_sg_size == 0
            ? desired_sg_size
            : 1;
    if (conf.sub_group_size <= 1) return false;

    conf.vect_size_fused = 4; // based on PVC performance data
    while (conf.norm_axis % (conf.sub_group_size * conf.vect_size_fused) != 0) {
        conf.vect_size_fused /= 2;
    }

    // Setup norm axis blocking
    // Limit values experimentally selected, based on PVC perf data.
    auto get_ss_utilization = [](const int num_wgs, const int max_ss) {
        return (float)num_wgs / max_ss;
    };
    int min_num_c_blocks = 2;
    int max_num_c_blocks = 32;
    int max_ss_util = 32;
    const int best_num_blocks = [&]() {
        int num_blocks = 1;
        int best_num = num_blocks;
        float ss_util = get_ss_utilization(conf.across_axis, max_ss);
        if (ss_util > max_ss_util) { return best_num; }
        while (num_blocks <= max_num_c_blocks) {
            if (conf.norm_axis
                            % (conf.sub_group_size * conf.vect_size_fused
                                    * num_blocks)
                    == 0) {
                best_num = num_blocks;
            }
            num_blocks++;
        }
        return best_num < min_num_c_blocks ? 1 : best_num;
    }();
    conf.num_norm_blocks_fused = best_num_blocks;
    assert(conf.norm_axis % conf.num_norm_blocks_fused == 0);
    conf.norm_block_fused = conf.norm_axis / conf.num_norm_blocks_fused;
    if (conf.num_norm_blocks_fused == 1) return false;

    // Setup across axis blocking
    // based on number of blocks, number of threads, slm buffer and wg size
    auto get_wg_size = [&](const int num_n_blocks) {
        return conf.sub_group_size
                * nstl::max(num_n_blocks, conf.num_norm_blocks_fused);
    };
    auto get_num_thrs = [&](const int num_n_blocks) {
        return num_n_blocks * conf.num_norm_blocks_fused;
    };
    auto get_slm_size = [&](const int num_n_blocks) {
        const size_t scale_shift_size
                = 2 * num_n_blocks * conf.norm_block_fused * sizeof(float);
        const size_t dd_gamma_size = conf.calculate_stats
                ? 2 * conf.sub_group_size * conf.num_norm_blocks_fused
                        * sizeof(float)
                : 0;
        return scale_shift_size + dd_gamma_size;
    };
    auto get_num_blocks = [&](const int n_block) {
        return utils::div_up(conf.across_axis, n_block);
    };
    // Limit values experimentally selected, based on PVC perf data.
    int min_n_block = 4;
    int max_n_block = 12;
    int min_num_thrs = 160;
    const int best_n_block = [&]() {
        int block = min_n_block;
        int best_block = block;
        while (block <= max_n_block) {
            const int n_blocks = get_num_blocks(block);
            if ((size_t)get_wg_size(n_blocks) <= max_wg_size
                    && get_slm_size(n_blocks) <= max_slm_size
                    && get_num_thrs(n_blocks) >= min_num_thrs) {
                best_block = block;
            } else {
                break;
            }
            block++;
        }
        return best_block;
    }();
    conf.num_across_blocks = get_num_blocks(best_n_block);
    conf.across_block = best_n_block;

    // Setup dispatching
    // scale/shift reduction requires:
    //      LWS = SG, number of across blocks, 1
    //      GWS = SG * num_norm_blocks, number of across blocks, 1
    // diff_gamma reduction and update part requires:
    //      LWS = SG, number of norm_blocks, 1
    //      GWS = SG, across dim * num_norm_blocks, 1
    // final dispatching as max:
    //      LWS = SG, max(n_chunks,num_norm_blocks), 1
    //      GWS = SG * num_norm_blocks, N * LWS1, 1

    const size_t lws0 = conf.sub_group_size;
    const size_t lws1
            = nstl::max(conf.num_across_blocks, conf.num_norm_blocks_fused);
    const size_t lws2 = 1;
    if (lws0 * lws1 * lws2 > max_wg_size) return false;

    const size_t gws0 = conf.sub_group_size * conf.num_norm_blocks_fused;
    const size_t gws1 = conf.across_axis * lws1;

    conf.dispatch_fused.define_dim("C_fused", gws0);
    CHECK(conf.dispatch_fused.vectorize_dim("C_fused", conf.sub_group_size));
    conf.dispatch_fused.define_dim("N_fused", gws1);
    conf.dispatch_fused.set_kernel_attr_suffix("FUSED");
    conf.dispatch_fused.generate();
    const size_t tuned_lws[3] = {lws0, lws1, lws2};
    conf.dispatch_fused.set_lws(tuned_lws);
    return true;
}

static status_t init_conf_common(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, engine_t *engine) {

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
    conf.dispatch_fused = compute_engine->create_dispatch(
            conf.is_fwd ? dst_mdw.md_ : src_mdw.md_);
    const auto &dims = conf.is_fwd ? src_mdw.padded_dims() : dst_mdw.dims();

    auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
            pd->attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
    auto eu_count = compute_engine->device_info()->eu_count();
    auto threads_per_eu
            = device_info_t::threads_per_eu(gpu_arch, large_grf_mode);
    auto max_eus_per_wg = device_info_t::max_eus_per_wg(gpu_arch);
    const int max_ss = utils::div_up(eu_count, max_eus_per_wg);
    const size_t max_wg_size
            = compute_engine->device_info()->max_wg_size(large_grf_mode);

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
            if (conf.across_axis < 4) return status::unimplemented;
        }
    }

    // DG2/ATSM optimization specifics - dont use private SRC buffer
    conf.use_src_buffer = gpu_arch >= gpu_arch_t::xe_hpc;

    const int desired_sg_size = conf.is_fwd ? 32 : 16;

    if (conf.is_fwd) {
        const int best_sg_size = [&]() {
            int size = desired_sg_size;
            while (size > 1) {
                const bool fit_to_shape = (conf.norm_axis % size == 0)
                        && (c_block == 1
                                || (c_block % size == 0 && ndims == 2));
                if (mayiuse_sg(size, engine) && fit_to_shape) return size;
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
        const int best_num_blocks = [&]() {
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

        // Try first the fused kernel targeted to relatively small shapes
        // and may get better performance because of host overheads reduction.
        // The fused kernel does not support in-place operation.
        // In-place operation can be recognized on execute phase only,
        // so if it's applicable we should build kernels for both approaches.

        conf.use_fused
                = is_fused_kernel_applicable(conf, pd, engine, large_grf_mode);

        const int best_sg_size = [&]() {
            int size = desired_sg_size;
            while (size > 1) {
                const bool fit_to_shape = conf.norm_axis % size == 0
                        && (src_mdw.matches_one_of_tag(ab, abc, abcd, abcde)
                                || (ndims == 2 && c_block % size == 0));
                if (mayiuse_sg(size, engine) && fit_to_shape) return size;
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
        conf.dispatch.generate();

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
        const int max_n_chunk_size = 16; // Experimentally selected values
        const int min_n_chunk_size = 4;
        int best_n_chunk_size = max_n_chunk_size;

        while (first_dim % best_n_chunk_size != 0
                && best_n_chunk_size > min_n_chunk_size) {
            best_n_chunk_size--;
        }
        conf.n_chunk_size = best_n_chunk_size;
        conf.n_chunks = utils::div_up(first_dim, conf.n_chunk_size);

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

        int n_finalize = conf.n_chunks;
        while (n_finalize > max_n_finalize) {
            n_finalize = utils::div_up(n_finalize, 2);
        }
        conf.finalize_n_chunks = n_finalize;
        conf.dispatch_scaleshift_finalize.define_dim_with_nesting_level(
                "N_finalize", 1, conf.finalize_n_chunks);
        conf.dispatch_scaleshift_finalize.set_kernel_attr_suffix(
                "SCALESHIFT_FINALIZE");
        conf.dispatch_scaleshift_finalize.generate();
        const size_t tuned_lws[3] = {(size_t)conf.finalize_n_chunks, 1, 1};
        conf.dispatch_scaleshift_finalize.set_lws(tuned_lws);
    } // bwd

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
    kernel_ctx.define_int("N", conf.across_axis);
    kernel_ctx.define_int("USE_FUSED", conf.use_fused);
    kernel_ctx.define_int("NORM_BLOCK", conf.norm_block);
    kernel_ctx.define_int("NUM_NORM_BLOCKS", conf.num_norm_blocks);
    kernel_ctx.define_int("NORM_BLOCK_FUSED", conf.norm_block_fused);
    kernel_ctx.define_int("NUM_NORM_BLOCKS_FUSED", conf.num_norm_blocks_fused);
    kernel_ctx.define_int("ACROSS_BLOCK", conf.across_block);
    kernel_ctx.define_int("NUM_ACROSS_BLOCKS", conf.num_across_blocks);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_dt_n);
    kernel_ctx.define_int("VECT_SIZE_FUSED", conf.vect_size_fused);
    kernel_ctx.define_int(
            "VECTOR_SIZE_SCALESHIFT", conf.vector_size_scaleshift);
    kernel_ctx.define_int("N_CHUNK_SIZE", conf.n_chunk_size);
    kernel_ctx.define_int("N_CHUNKS", conf.n_chunks);
    kernel_ctx.define_int("FINALIZE_N_CHUNKS", conf.finalize_n_chunks);
    kernel_ctx.define_int("USE_SRC_BUFFER", conf.use_src_buffer);

    kernel_ctx.add_option("-cl-std=CL2.0");
    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, conf.stat_md_info, "STAT");

    if (conf.is_fwd)
        def_dispatch(kernel_ctx, conf.dispatch);
    else {
        if (conf.use_fused) { def_dispatch(kernel_ctx, conf.dispatch_fused); }
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift);
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift_finalize);
        def_dispatch(kernel_ctx, conf.dispatch);
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

    bool in_place = diff_dst.data_handle() == diff_src.data_handle();

    if (conf.use_fused && !in_place) {
        std::unique_ptr<memory_storage_t> temp_reduce;
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, mean);
        arg_list.set(2, variance);
        arg_list.set(3, diff_dst);
        arg_list.set(4, diff_scale);
        arg_list.set(5, diff_shift);
        arg_list.set(6, scale);
        arg_list.set(7, diff_src);
        arg_list.set(8, conf.eps);

        auto nd_range = conf.dispatch_fused.nd_range();
        status = parallel_for(ctx, nd_range, kernel_fused_, arg_list);
        return status;
    }

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
