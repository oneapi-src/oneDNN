/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_OCL_JIT_REF_BNORM_COMMON_KERNEL_HPP
#define GPU_OCL_JIT_REF_BNORM_COMMON_KERNEL_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "gpu/ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct jit_ref_bnorm_common_kernel {

    jit_ref_bnorm_common_kernel(const jit_bnorm_conf_t &ajbn) : jbn(ajbn) {}

    ~jit_ref_bnorm_common_kernel() {}

    static status_t init_conf(jit_bnorm_conf_t &jbn,
            const batch_normalization_pd_t *pd, jit_offsets &jit_off) {
        using namespace dnnl::impl::format_tag;

        const batch_normalization_desc_t &bd = *pd->desc();
        const memory_desc_wrapper data_mdw(
                pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
        const int ndims = data_mdw.ndims();

        jbn.data_type = data_mdw.data_type();

        jbn.ndims = ndims;
        jbn.mb = data_mdw.dims()[0];

        jbn.ic = data_mdw.dims()[1];
        jbn.id = (ndims == 5) ? data_mdw.dims()[2] : 1;
        jbn.ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
        jbn.iw = data_mdw.dims()[ndims - 1];

        jbn.is_forward = pd->is_fwd();
        jbn.is_backward = !pd->is_fwd();

        jbn.use_scaleshift = pd->use_scaleshift();
        jbn.save_stats = pd->is_training();
        jbn.is_training = pd->is_training();
        jbn.fuse_norm_relu = pd->fuse_norm_relu();
        jbn.calculate_stats = !pd->stats_is_src();
        jbn.with_relu = pd->with_relu_post_op();
        jbn.eps = bd.batch_norm_epsilon;
        jbn.calculate_diff_stats = !pd->use_global_stats();
        jbn.diff_scaleshift
                = (pd->use_scaleshift() && bd.prop_kind == prop_kind::backward);

        set_offsets(data_mdw, jit_off.src_off);

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(pd->engine());

        const bool has_padding = !data_mdw.is_dense();
        if (!has_padding
                && data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c,
                        NCw16n16c, NChw16n16c, NCdhw16n16c)) {
            jbn.mb_block = data_mdw.matches_one_of_tag(
                                   NCw16n16c, NChw16n16c, NCdhw16n16c)
                    ? 16
                    : 1;

            jbn.use_16mb_unroll = 1;

            const int max_stat_nblocks = 256;
            int stat_mb_nblocks = jbn.mb / jbn.mb_block;
            int stat_sp_nblocks = utils::max_div(jbn.id * jbn.ih * jbn.iw,
                    nstl::max(1, max_stat_nblocks / stat_mb_nblocks));
            assert(stat_mb_nblocks * stat_sp_nblocks <= max_stat_nblocks);

            int stat_sp_block = jbn.id * jbn.ih * jbn.iw / stat_sp_nblocks;

            jbn.reduce_stat_nblocks = stat_mb_nblocks * stat_sp_nblocks;

            jbn.dispatch_calc_stat = compute_engine->create_dispatch();
            jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                    "STAT_SP", 2, jbn.id * jbn.ih * jbn.iw, stat_sp_block);
            jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                    "STAT_IC", 1, jbn.ic);
            jbn.dispatch_calc_stat.define_dim_with_nesting_level(
                    "STAT_MB", 0, jbn.mb, jbn.mb_block);
            jbn.dispatch_calc_stat.vectorize_dim("STAT_IC", 16);
            jbn.dispatch_calc_stat.set_kernel_attr_suffix("CALC");
            jbn.dispatch_calc_stat.generate();

            jbn.dispatch_reduce_stat = compute_engine->create_dispatch();
            jbn.dispatch_reduce_stat.define_dim("REDUCE_STAT_IC", jbn.ic);
            jbn.dispatch_reduce_stat.set_kernel_attr_suffix("REDUCE");
            jbn.dispatch_reduce_stat.generate();

            jbn.dispatch = compute_engine->create_dispatch(data_mdw.md_);
            jbn.dispatch.define_dim("MB", 0, jbn.mb, jbn.mb_block);
            jbn.dispatch.define_dim("IC", 1, jbn.ic);
            jbn.dispatch.define_dim("ID", nstl::max(2, ndims - 3), jbn.id);
            jbn.dispatch.define_dim("IH", nstl::max(2, ndims - 2), jbn.ih);
            jbn.dispatch.define_dim("IW", nstl::max(2, ndims - 1), jbn.iw);
            jbn.dispatch.vectorize_dim("IC", 16);
            jbn.dispatch.generate();
        } else {
            // Reference
            jbn.use_16mb_unroll = 0;
            jbn.mb_block = 1;
            jbn.dispatch = compute_engine->create_dispatch();
            jbn.dispatch.define_dim("IC", jbn.ic);
            jbn.dispatch.generate();
        }

        return status::success;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_bnorm_conf_t &jbn, const jit_offsets &jit_off) {
        kernel_ctx.set_data_type(jbn.data_type);

        kernel_ctx.define_int("NDIMS", jbn.ndims);
        kernel_ctx.define_int("MB", jbn.mb);
        kernel_ctx.define_int("IC", jbn.ic);
        kernel_ctx.define_int("ID", jbn.id);
        kernel_ctx.define_int("IH", jbn.ih);
        kernel_ctx.define_int("IW", jbn.iw);
        kernel_ctx.define_int("USE_16MB_UNROLL", jbn.use_16mb_unroll);
        kernel_ctx.define_int("REDUCE_STAT_NBLOCKS", jbn.reduce_stat_nblocks);
        kernel_ctx.define_int("MB_BLOCK", jbn.mb_block);

        if (jbn.is_forward)
            kernel_ctx.define_int("IS_FWD", 1);
        else if (jbn.is_backward)
            kernel_ctx.define_int("IS_BWD", 1);

        kernel_ctx.define_int("WITH_RELU", jbn.with_relu);
        kernel_ctx.define_int("SAVE_STATS", jbn.save_stats);
        kernel_ctx.define_int("IS_TRAINING", jbn.is_training);
        kernel_ctx.define_int("FUSE_BN_RELU", jbn.fuse_norm_relu);
        kernel_ctx.define_int("CALCULATE_STATS", jbn.calculate_stats);
        kernel_ctx.define_int("USE_SCALESHIFT", jbn.use_scaleshift);
        kernel_ctx.define_int("CALCULATE_DIFF_STATS", jbn.calculate_diff_stats);
        kernel_ctx.define_int("DIFF_SCALESHIFT", jbn.diff_scaleshift);

        def_offsets(jit_off.src_off, kernel_ctx, "SRC", jbn.ndims);

        if (jbn.use_16mb_unroll) {
            def_dispatch(kernel_ctx, jbn.dispatch_calc_stat);
            def_dispatch(kernel_ctx, jbn.dispatch_reduce_stat);
        }
        def_dispatch(kernel_ctx, jbn.dispatch);

        return status::success;
    }

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_bnorm_conf_t &jbn) {
        if (jbn.is_forward) {
            if (jbn.use_16mb_unroll && jbn.calculate_stats) {
                size_t size = 2 * jbn.reduce_stat_nblocks * jbn.ic
                        * types::data_type_size(data_type::f32);

                scratchpad.book(
                        memory_tracking::names::key_bnorm_reduction, size);
            }
        }
        if (jbn.is_backward) {
            if (jbn.use_16mb_unroll) {
                size_t size = 2 * jbn.reduce_stat_nblocks * jbn.ic
                        * types::data_type_size(data_type::f32);
                scratchpad.book(
                        memory_tracking::names::key_bnorm_reduction, size);
            }
        }
    }

    jit_bnorm_conf_t jbn;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
