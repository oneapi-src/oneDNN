/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "gpu/ocl/gen9_pooling.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(pool_conf_t &conf, offsets_t &off,
        const pooling_pd_t *pd, engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::alg_kind;

    const memory_desc_wrapper src_mdw(pd->invariant_src_md());
    const memory_desc_wrapper dst_mdw(pd->invariant_dst_md());

    auto is_c_dense = [](const memory_desc_wrapper &mdw) {
        return mdw.blocking_desc().strides[1] == 1;
    };
    auto is_c_blocked_by
            = [](const memory_desc_wrapper &mdw, const int blockSize) {
                  auto &blk = mdw.blocking_desc();
                  if (blk.inner_nblks == 0) return false;
                  return (blk.inner_idxs[blk.inner_nblks - 1] == 1)
                          && (blk.inner_blks[blk.inner_nblks - 1] == blockSize);
              };

    if (!is_c_blocked_by(src_mdw, 16) && !is_c_blocked_by(src_mdw, 32)
            && !is_c_dense(src_mdw))
        return status::unimplemented;

    if (!is_c_blocked_by(dst_mdw, 16) && !is_c_blocked_by(dst_mdw, 32)
            && !is_c_dense(dst_mdw))
        return status::unimplemented;

    int c_block_size = 1, n_block_size = 1;
    auto &src_blk = src_mdw.blocking_desc();
    if (src_blk.inner_nblks > 0) {
        // C is the last blocked dimension as it was checked in is_c_blocked_by
        c_block_size = src_blk.inner_blks[src_blk.inner_nblks - 1];
        // if there is NC blocking (N is the blocked dimension before C) use N blocks as well
        if (src_blk.inner_nblks > 1
                && src_blk.inner_idxs[src_blk.inner_nblks - 2] == 0) {
            n_block_size = src_blk.inner_blks[src_blk.inner_nblks - 2];
        }
    }

    set_default_pool_conf(conf, *pd->desc(), *pd->invariant_src_md(),
            *pd->invariant_dst_md(), *pd->attr());

    set_offsets(src_mdw, off.src_off);
    set_offsets(dst_mdw, off.dst_off);

    conf.sub_group_size = 16;
    conf.use_mb_c_block = false;
    conf.use_only_c_block = false;
    int c_padded = utils::rnd_up(conf.c_padded, conf.sub_group_size);

    if (c_block_size >= 16 && n_block_size >= 16) {
        c_padded = utils::rnd_up(conf.c_padded, c_block_size);
        conf.use_mb_c_block = true;
        conf.vect_dt_n = 8;
        conf.nvect = 2;
        if (!pd->attr()->post_ops_.has_default_values()) { conf.nvect = 1; }
        conf.chunks_per_c_block = c_block_size / conf.sub_group_size;
        conf.chunks_per_mb_block
                = conf.vect_dt_n * conf.nvect / conf.chunks_per_c_block;
    } else if (c_block_size == 16 && n_block_size == 1) {
        conf.use_only_c_block = true;
        conf.vect_dt_n = 1;
        conf.nvect = 2;
        if (!pd->attr()->post_ops_.has_default_values()) { conf.nvect = 1; }
        if ((conf.c_padded / c_block_size) % conf.nvect != 0) {
            conf.nvect = 1;
        }
        conf.chunks_per_c_block = conf.nvect * conf.vect_dt_n;
        conf.chunks_per_mb_block = 1;
        // heuristics: ocl ref kernel is faster for small filters.
        if (((float)conf.kh / (float)conf.ih) < 0.5)
            return status::unimplemented;
    } else {
        conf.use_only_c_block = true;
        const size_t num_c_blocks = c_padded / conf.sub_group_size;
        conf.vect_dt_n = 8;
        while (num_c_blocks % conf.vect_dt_n != 0) {
            conf.vect_dt_n /= 2;
        }
        if (conf.vect_dt_n < 8) {
            if (!conf.is_backward) {
                if (conf.mb_padded % 2 == 0) { conf.unroll_mb_count = 2; }
            } else if (conf.alg != pooling_avg_exclude_padding) {
                if (conf.mb_padded % 4 == 0) { conf.unroll_mb_count = 4; }
            }
        }
        conf.nvect = 1;
        conf.chunks_per_c_block = conf.nvect * conf.vect_dt_n;
        conf.chunks_per_mb_block = 1;
        // fallback to ref_pooling kernel for better perf.
        if (conf.is_backward) {
            if ((conf.vect_dt_n < 4) && (num_c_blocks > 2))
                return status::unimplemented;
        } else { // FWD
            if (conf.vect_dt_n == 1) return status::unimplemented;
        }
    }
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(
            conf.is_backward ? src_mdw.md_ : dst_mdw.md_);

    auto arch = compute_engine->device_info()->gpu_arch();
    bool is_pre_xe_hpc = arch < compute::gpu_arch_t::xe_hpc;
    size_t input_sz_mb
            = src_mdw.nelems() * src_mdw.data_type_size() / 1024 / 1024;
    // heuristics: use batching on ATS-M for certain shapes for better perf.
    if (!conf.is_backward && pd->attr()->post_ops_.has_default_values()
            && !(conf.unroll_mb_count > 1) && is_pre_xe_hpc
            && (2 * input_sz_mb > (size_t)conf.mb)) {
        conf.num_batches = utils::div_up(conf.mb_padded, conf.mb_block_size);
    }
    // for IO bytes less than 256 KB fall back into ocl ref kernel for better performance.
    size_t io_bytes = src_mdw.nelems() * src_mdw.data_type_size()
            + dst_mdw.nelems() * dst_mdw.data_type_size();
    if (io_bytes < 256 * 1024) return status::unimplemented;

    if (conf.num_batches > 1) {
        conf.dispatch.define_dim("MB", 0,
                nstl::min(conf.mb_block_size, conf.mb_padded),
                conf.chunks_per_mb_block);
    } else {
        conf.dispatch.define_dim("MB", 0, conf.mb_padded / conf.unroll_mb_count,
                conf.chunks_per_mb_block);
    }
    conf.dispatch.define_dim("C", 1, c_padded, conf.chunks_per_c_block);

    int ndims = conf.ndims;
    if (!conf.is_backward) {
        conf.dispatch.define_dim("OD", nstl::max(2, ndims - 3), conf.od);
        conf.dispatch.define_dim("OH", nstl::max(2, ndims - 2), conf.oh);
        conf.dispatch.define_dim("OW", nstl::max(2, ndims - 1), conf.ow);
    } else {
        conf.dispatch.define_dim("ID", nstl::max(2, ndims - 3), conf.id);
        conf.dispatch.define_dim("IH", nstl::max(2, ndims - 2), conf.ih);
        conf.dispatch.define_dim("IW", nstl::max(2, ndims - 1), conf.iw);
    }
    CHECK(conf.dispatch.vectorize_dim("C", conf.sub_group_size));
    conf.dispatch.generate();

    return status::success;
};

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off,
        const post_ops_t &post_ops) {
    using namespace dnnl::impl::alg_kind;
    kernel_ctx.set_data_type(conf.src_dt);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    if (conf.num_batches > 1) {
        kernel_ctx.define_int("MB", nstl::min(conf.mb_block_size, conf.mb));
    } else {
        kernel_ctx.define_int("MB", conf.mb);
    }
    kernel_ctx.define_int("MB_BLOCK_SIZE", conf.mb_block_size);
    kernel_ctx.define_int("C_W_PADDING", conf.c_padded);
    kernel_ctx.define_int("C_WO_PADDING", conf.c);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("IS_BWD", conf.is_backward);
    kernel_ctx.define_int("IS_FWD", !conf.is_backward);

    kernel_ctx.define_int("ALG_MAX", (conf.alg == pooling_max));
    kernel_ctx.define_int(
            "ALG_AVG_NP", (conf.alg == pooling_avg_exclude_padding));
    kernel_ctx.define_int(
            "ALG_AVG_P", (conf.alg == pooling_avg_include_padding));

    kernel_ctx.define_int("VECT_DT_N", conf.vect_dt_n);
    kernel_ctx.define_int("NVECT", conf.nvect);
    kernel_ctx.define_int("USE_ONLY_C_BLOCK", conf.use_only_c_block);
    kernel_ctx.define_int("USE_MB_C_BLOCK", conf.use_mb_c_block);
    kernel_ctx.define_int("CHUNKS_PER_C_BLOCK", conf.chunks_per_c_block);
    kernel_ctx.define_int("CHUNKS_PER_MB_BLOCK", conf.chunks_per_mb_block);
    kernel_ctx.define_int("UNROLL_MB_COUNT", conf.unroll_mb_count);

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    CHECK(def_attr_info(
            kernel_ctx, conf.attr_info, post_ops, conf.dst_md_info.dims));

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t gen9_pooling_fwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_pooling_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, attr()->post_ops_);
}

status_t gen9_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, ws);
    arg_list.set(2, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 4, pd()->attr()->post_ops_);

    auto nd_range = pd()->conf.dispatch.nd_range();

    int num_batches = pd()->conf.num_batches;
    for (int batch_iter = 0; batch_iter < num_batches; batch_iter++) {
        arg_list.set(3, batch_iter);
        status = parallel_for(ctx, nd_range, kernel_, arg_list);
        if (status != status::success) return status;
    }
    return status;
}

status_t gen9_pooling_bwd_t::pd_t::init_conf(engine_t *engine) {
    return init_conf_common(conf, off, this, engine);
}

status_t gen9_pooling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off, attr()->post_ops_);
}

status_t gen9_pooling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {

    status_t status = status::success;
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    CHECK(status);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, ws);
    arg_list.set(2, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
