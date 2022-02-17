/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_OCL_GEN9_SOFTMAX_HPP
#define GPU_OCL_GEN9_SOFTMAX_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_softmax_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_softmax_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_softmax_fwd_t);

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = is_fwd() && axis_size() % 128 == 0
                    && axis() == src_d.ndims() - 1 && src_d.is_plain()
                    && utils::one_of(src_d.data_type(), data_type::f32,
                            data_type::f16, data_type::bf16)
                    && IMPLICATION(src_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && attr()->has_default_values()
                    && set_default_formats() == status::success
                    && dst_d == src_d;
            if (!ok) return status::unimplemented;

            group_size = 16;

            if (!compute_engine->mayiuse_sub_group((int)group_size))
                return status::unimplemented;

            lws[0] = group_size;
            lws[1] = lws[2] = 1;
            gws[0] = utils::array_product(&src_md()->dims[0], ndims() - 1)
                    * group_size;
            gws[1] = gws[2] = 1;

            return status::success;
        }

        size_t gws[3] = {};
        size_t lws[3] = {};
        size_t block[3] = {};
        size_t group_size = 0;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", pd()->axis());
        kernel_ctx.define_int("SOFTMAX_AXIS_SIZE", pd()->axis_size());
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("IS_FWD", 1);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());

        kernel_ctx.set_data_type(pd()->src_md()->data_type);
        set_offsets(kernel_ctx, pd()->dst_md(), "DATA");

        for (int i = 0; i < 3; ++i)
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), pd()->block[i]);

        create_kernel(engine, &kernel_, "gen9_softmax_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_softmax_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_bwd_pd_t {
        using gpu_softmax_bwd_pd_t::gpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_softmax_bwd_t);

        status_t init(engine_t *engine) {
            using namespace dnnl::impl::format_tag;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = !is_fwd() && axis_size() % 128 == 0
                    && axis() == diff_src_d.ndims() - 1
                    && utils::one_of(
                            dst_d.data_type(), data_type::f32, data_type::bf16)
                    && attr()->has_default_values()
                    && set_default_formats() == status::success
                    && diff_src_d == diff_dst_d && diff_src_d == dst_d
                    && compute_engine->mayiuse_sub_group((16));
            if (!ok) return status::unimplemented;

            is_nhwc = (diff_src_d.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != undef);
            is_blk = (diff_src_d.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c)
                    != undef);
            if (is_nhwc || is_blk)
                group_size = 16 * (axis_size() / 128);
            else
                group_size = 16;
            lws[0] = group_size;
            lws[1] = lws[2] = 1;
            gws[0] = utils::array_product(
                             &diff_src_md(0)->padded_dims[0], ndims() - 1)
                    * group_size;
            gws[1] = gws[2] = 1;
            batches = diff_src_md(0)->padded_dims[0]
                    * diff_src_md(0)->padded_dims[2];
            return status::success;
        }

        size_t gws[3] = {};
        size_t lws[3] = {};
        size_t block[3] = {};
        size_t group_size = 0;
        size_t batches = 0;
        bool is_nhwc = false;
        bool is_blk = false;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", pd()->axis());
        kernel_ctx.define_int("SOFTMAX_AXIS_SIZE", pd()->axis_size());
        kernel_ctx.define_int("SOFTMAX_BUF", 128);
        kernel_ctx.define_int("SUB_GROUP_SIZE", 16);
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("IS_BWD", 1);
        kernel_ctx.define_int("IS_16C", pd()->is_blk);
        kernel_ctx.define_int("BATCH", pd()->batches);
        kernel_ctx.define_int("IC_WO_PADDING", pd()->diff_src_md(0)->dims[1]);
        kernel_ctx.define_int(
                "IC_PADDED", pd()->diff_src_md(0)->padded_dims[1]);
        kernel_ctx.define_int(
                "IC", pd()->is_blk ? 16 : pd()->diff_src_md(0)->padded_dims[1]);
        kernel_ctx.define_int("IS_NHWC", pd()->is_nhwc);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());

        kernel_ctx.set_data_type(pd()->diff_src_md()->data_type);
        set_offsets(kernel_ctx, *pd()->diff_src_md(), "DATA");

        for (int i = 0; i < 3; ++i)
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), pd()->block[i]);

        create_kernel(engine, &kernel_, "gen9_softmax_bwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
