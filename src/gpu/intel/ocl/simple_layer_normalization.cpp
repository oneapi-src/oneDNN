/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/ocl/simple_layer_normalization.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/scratchpad.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t init_conf_common(lnorm_conf_t &conf,
        const layer_normalization_pd_t *pd, impl::engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    memory_desc_wrapper src_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    dim_idx_t ndims = into<dim_idx_t>(src_mdw.ndims());

    conf.src_dt = src_mdw.data_type();
    conf.dst_dt = dst_mdw.data_type();
    conf.ndims = ndims;
    conf.norm_axis = into<dim_idx_t>(pd->norm_axis());
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.stat_md_info = memory_desc_info_t::create(stat_mdw);
    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.calculate_stats = !pd->stats_are_src();
    conf.save_stats = pd->is_training();
    conf.eps = pd->desc()->layer_norm_epsilon;
    conf.is_fwd = pd->is_fwd();
    conf.vect_dt_n = 1;
    conf.sub_group_size = 1;

    if (conf.use_scale || conf.use_shift) {
        memory_desc_wrapper weights_mdw(
                pd->is_fwd() ? pd->weights_md() : pd->diff_weights_md());
        conf.weights_data_type = weights_mdw.data_type();
    }

    int c_block = 1;
    bool c_is_last_physical = false;

    if (src_mdw.blocking_desc().inner_nblks > 0) {
        c_block = into<int>(
                src_mdw.blocking_desc()
                        .inner_blks[src_mdw.blocking_desc().inner_nblks - 1]);
        c_is_last_physical
                = src_mdw.blocking_desc().inner_idxs[ndims - 1] == ndims - 1;
    } else {
        c_is_last_physical = src_mdw.blocking_desc().strides[ndims - 1] == 1;
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch_scaleshift = compute_engine->create_dispatch();
    conf.dispatch_scaleshift_finalize = compute_engine->create_dispatch();
    conf.dispatch = compute_engine->create_dispatch(
            pd->is_fwd() ? dst_mdw.md_ : src_mdw.md_);
    const auto &dims = pd->is_fwd() ? src_mdw.padded_dims() : dst_mdw.dims();

    const int desired_sg_size = 32;
    auto mayiuse_sg = [=](const int sg_size) {
        return compute_engine->mayiuse_sub_group(sg_size)
                && compute_engine->mayiuse_block_reads_writes_with_sub_group(
                        sg_size);
    };

    if (pd->is_fwd()) {
        const int sg_size = [&]() {
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
        if (src_mdw.is_dense() && c_is_last_physical && ndims < 4 && sg_size > 1
                && utils::one_of(data_type::f64, conf.src_dt, conf.dst_dt)) {
            conf.vectorize_calc_stats = true;
            conf.sub_group_size = sg_size;
            int vector_size = 8;
            while (conf.norm_axis % (conf.sub_group_size * vector_size) != 0) {
                vector_size /= 2;
            }
            while (c_block > 1 && vector_size * conf.sub_group_size > c_block) {
                vector_size /= 2;
            }
            conf.vect_dt_n = vector_size;
        } else {
            return status::unimplemented;
        }
        for (dim_idx_t i = 0; i < 4; i++) {
            dim_idx_t md_hint_idx = nstl::min(i, ndims - 1);
            dim_t dim = (i < ndims - 1) ? dims[i] : 1;
            if (conf.vectorize_calc_stats && (i == ndims - 1)) {
                dim = sg_size;
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
                CHECK(conf.dispatch.vectorize_dim(
                        utils::format("X%d", i), conf.sub_group_size));
            } else {
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
            }
        }
    } else {
        conf.vectorize_bwd = false;
        const int sg_size = [&]() {
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
        if (src_mdw.is_dense() && c_is_last_physical && sg_size > 1
                && utils::one_of(data_type::f64, conf.src_dt, conf.dst_dt)) {
            conf.vectorize_bwd = true;
            conf.sub_group_size = sg_size;
            conf.vect_dt_n = 8;
            while (conf.norm_axis % (conf.sub_group_size * conf.vect_dt_n)
                    != 0) {
                conf.vect_dt_n /= 2;
            }
            while (src_mdw.blocking_desc().inner_nblks > 0
                    && c_block % (conf.sub_group_size * conf.vect_dt_n) != 0) {
                conf.vect_dt_n /= 2;
            }
        }
        for (dim_idx_t i = 0; i < 4; i++) {
            dim_idx_t md_hint_idx = nstl::min(i, ndims - 1);
            dim_t dim = (i < ndims - 1) ? dims[i] : 1;
            if (conf.vectorize_bwd && (i == ndims - 1)) {
                conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx,
                        conf.sub_group_size);
                CHECK(conf.dispatch.vectorize_dim(
                        utils::format("X%d", i), conf.sub_group_size));
            } else {
                conf.dispatch.define_dim(
                        utils::format("X%d", i), md_hint_idx, dim);
            }
        }

        int n_block = 1;
        conf.n_chunk_size = 1;
        conf.vector_size_scaleshift = 1;
        conf.n_chunks = dims[0] / conf.n_chunk_size;
        if (src_mdw.blocking_desc().inner_nblks == 2
                && src_mdw.blocking_desc().inner_idxs[0] == 0) {
            n_block = into<int>(src_mdw.blocking_desc().inner_blks[0]);
        }
        // Scaleshift vectorization is supported for tensors
        // with shapes AxB, 1xBxC
        conf.vectorize_bwd_scaleshift = conf.vectorize_bwd
                && stat_mdw.matches_one_of_tag(a, ab)
                && ((ndims == 2
                            && (c_block == sg_size || src_mdw.matches_tag(ab)))
                        || (ndims == 3 && src_mdw.matches_tag(abc)
                                && dims[0] == 1))
                && utils::one_of(data_type::f64, conf.src_dt, conf.dst_dt);
        if (!conf.vectorize_bwd_scaleshift) { return status::unimplemented; }
        // Use partial reduction in order to increase number of used threads
        conf.vector_size_scaleshift = c_block == sg_size ? 8 : 1;
        const dim_t first_dim = ndims == 2 ? dims[0] : dims[1];
        while (n_block % conf.vector_size_scaleshift != 0
                || first_dim % conf.vector_size_scaleshift != 0) {
            conf.vector_size_scaleshift /= 2;
        }
        // Experimentally selected values
        const int max_first_dim_elems_per_wi = 32;
        int desired_first_dim_block_reads
                = max_first_dim_elems_per_wi / conf.vector_size_scaleshift;
        while (first_dim
                        % (desired_first_dim_block_reads
                                * conf.vector_size_scaleshift)
                != 0) {
            desired_first_dim_block_reads /= 2;
        }
        while (first_dim
                        % (desired_first_dim_block_reads
                                * conf.vector_size_scaleshift)
                != 0) {
            conf.vector_size_scaleshift /= 2;
        }
        conf.n_chunk_size
                = desired_first_dim_block_reads * conf.vector_size_scaleshift;
        conf.n_chunks = first_dim / conf.n_chunk_size;
        // Scaleshift kernel does partial reduction of N
        conf.dispatch_scaleshift.define_dim("N", conf.n_chunks);
        conf.dispatch_scaleshift.define_dim("C", pd->norm_axis());
        CHECK(conf.dispatch_scaleshift.vectorize_dim("C", conf.sub_group_size));
        conf.dispatch_scaleshift.set_kernel_attr_suffix("SCALESHIFT");
        conf.dispatch_scaleshift.generate();
        // Scaleshift finalize kernel reduces results of scaleshift kernel
        conf.dispatch_scaleshift_finalize.define_dim(
                "C_finalize", pd->norm_axis());
        conf.dispatch_scaleshift_finalize.set_kernel_attr_suffix(
                "SCALESHIFT_FINALIZE");
        conf.dispatch_scaleshift_finalize.generate();
    }

    conf.dispatch.generate();
    return status::success;
}

static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const lnorm_conf_t &conf) {
    kernel_ctx.set_data_type(conf.is_fwd ? conf.src_dt : conf.dst_dt);
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");

    kernel_ctx.define_int("C", conf.norm_axis);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
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
        def_dispatch(kernel_ctx, conf.dispatch_scaleshift_finalize);
    }

    return status::success;
}

status_t simple_layer_normalization_fwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t simple_layer_normalization_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t simple_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
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
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, shift);
    arg_list.set(6, conf.eps);
    arg_list.set(7, src_scale);
    arg_list.set(8, dst_scale);

    auto nd_range_kernel = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t simple_layer_normalization_bwd_t::pd_t::init_conf(
        impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t simple_layer_normalization_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

void simple_layer_normalization_bwd_t::pd_t::init_scratchpad() {
    const size_t size = conf.n_chunks * conf.norm_axis * 2;
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_lnorm_reduction, size,
            types::data_type_size(data_type::f32), OCL_BUFFER_ALIGNMENT);
}

status_t simple_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
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
        std::unique_ptr<memory_storage_t> temp_reduce;
        compute::kernel_arg_list_t arg_list;
        temp_reduce = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_lnorm_reduction);
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

        compute::kernel_arg_list_t arg_list_final;
        arg_list_final.set(0, *temp_reduce);
        arg_list_final.set(1, diff_scale);
        arg_list_final.set(2, diff_shift);

        auto nd_range_finalize = conf.dispatch_scaleshift_finalize.nd_range();
        status = parallel_for(ctx, nd_range_finalize,
                kernel_scaleshift_finalize_, arg_list_final);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
