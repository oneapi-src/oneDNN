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

#ifndef GPU_OCL_REF_SOFTMAX_HPP
#define GPU_OCL_REF_SOFTMAX_HPP

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

struct ref_softmax_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const auto src_dt = src_d.data_type();
            const auto dst_dt = dst_d.data_type();

            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            bool ok = is_fwd() && utils::one_of(src_dt, f32, f16, bf16, u8, s8)
                    && utils::one_of(dst_dt, f32, f16, bf16, u8, s8)
                    && IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && compute_engine->mayiuse_sub_group(subgroup_size)
                    && !memory_desc_ndims_ok(src_md(), dst_md())
                    && attr()->has_default_values(skip_mask_t::scales_runtime)
                    && attr_scales_ok()
                    && set_default_formats() == status::success;
            if (!ok) return status::unimplemented;

            gws[0] = 1;
            gws[1] = 1;
            gws[2] = 1;

            lws[0] = 1;
            lws[1] = 1;
            lws[2] = 1;

            block[0] = 1;
            block[1] = 1;
            block[2] = 1;

            int nelems = axis_size(true);

            if (nelems < subgroup_size) {
                group_size = subgroup_size = 1;
            } else if (nelems <= 100) {
                group_size = subgroup_size * 1;
            } else if (nelems <= 1000) {
                group_size = subgroup_size * 2;
            } else if (nelems <= 2000) {
                group_size = subgroup_size * 4;
            } else if (nelems <= 5000) {
                group_size = subgroup_size * 8;
            } else {
                group_size = subgroup_size * 16;
            }

            for (int i = 0, j = 0; i < src_md()->ndims; ++i) {
                if (i != desc()->softmax_axis) {
                    auto dim = src_md()->padded_dims[i];
                    gws[j % 3] *= dim;
                    if (j < 3) block[j % 3] = dim;
                    j++;
                }
            }

            if (group_size != 1) {
                lws[0] = group_size;
                gws[0] *= group_size;
            }

            return status::success;
        }

        size_t gws[3] = {};
        size_t lws[3] = {};
        size_t block[3] = {};
        size_t group_size = 0;
        int subgroup_size = 16;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        const auto *desc = pd()->desc();
        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", desc->softmax_axis);
        kernel_ctx.define_int("SOFTMAX_AXIS", pd()->axis_size(true));
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", pd()->subgroup_size);
        kernel_ctx.define_int("IS_FWD", 1);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());
        kernel_ctx.define_int("WITH_SRC_SCALES",
                !pd()->attr()->scales_.get(DNNL_ARG_SRC).has_default_values());
        kernel_ctx.define_int("WITH_DST_SCALES",
                !pd()->attr()->scales_.get(DNNL_ARG_DST).has_default_values());

        const memory_desc_wrapper dst_mdw(pd()->dst_md());
        const memory_desc_wrapper src_mdw(pd()->src_md());
        const auto dst_md_info = memory_desc_info_t::create(dst_mdw);
        const auto src_md_info = memory_desc_info_t::create(src_mdw);
        def_memory_desc_info(kernel_ctx, dst_md_info, "DST");
        def_memory_desc_info(kernel_ctx, src_md_info, "SRC");
        kernel_ctx.set_data_type(dst_mdw.data_type());
        set_offsets(kernel_ctx, pd()->dst_md(), "DATA");

        for (int i = 0; i < 3; i++)
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), pd()->block[i]);

        CHECK(create_kernel(
                engine, &kernel_, "ref_softmax_fwd_generic", kernel_ctx));
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

struct ref_softmax_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_bwd_pd_t {
        using gpu_softmax_bwd_pd_t::gpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper dst_d(dst_md());

            using namespace data_type;
            bool ok = !is_fwd()
                    && utils::one_of(diff_src_d.data_type(), f32, bf16)
                    && utils::one_of(diff_dst_d.data_type(), f32, bf16)
                    && compute_engine->mayiuse_sub_group(16)
                    && !memory_desc_ndims_ok(
                            dst_md(), diff_src_md(), diff_dst_md())
                    && attr()->has_default_values()
                    && set_default_formats() == status::success
                    && diff_dst_d.data_type() == dst_d.data_type();
            if (!ok) return status::unimplemented;

            gws[0] = 1;
            gws[1] = 1;
            gws[2] = 1;

            lws[0] = 1;
            lws[1] = 1;
            lws[2] = 1;

            block[0] = 1;
            block[1] = 1;
            block[2] = 1;

            for (int i = 0, j = 0; i < dst_d.ndims(); ++i) {
                if (i != axis()) {
                    auto dim = dst_d.padded_dims()[i];
                    gws[j % 3] *= dim;
                    if (j < 3) block[j % 3] = dim;
                    j++;
                }
            }

            int nelems = axis_size(true);
            if (nelems <= 100) {
                group_size = 16;
            } else if (nelems <= 1000) {
                group_size = 32;
            } else if (nelems <= 2000) {
                group_size = 64;
            } else if (nelems <= 5000) {
                group_size = 128;
            } else {
                group_size = 256;
            }

            lws[0] = group_size;
            gws[0] *= group_size;

            return status::success;
        }

        size_t lws[3] = {};
        size_t gws[3] = {};
        size_t block[3] = {};
        size_t group_size = 0;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", pd()->axis());
        kernel_ctx.define_int("SOFTMAX_AXIS", pd()->axis_size(true));
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", 16);
        kernel_ctx.define_int("IS_BWD", 1);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());

        const memory_desc_wrapper diff_src_mdw(pd()->diff_src_md());
        const memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        const auto diff_src_md_info = memory_desc_info_t::create(diff_src_mdw);
        const auto diff_dst_md_info = memory_desc_info_t::create(diff_dst_mdw);
        def_memory_desc_info(kernel_ctx, diff_src_md_info, "SRC");
        def_memory_desc_info(kernel_ctx, diff_dst_md_info, "DST");
        kernel_ctx.set_data_type(diff_src_mdw.data_type());
        set_offsets(kernel_ctx, *pd()->diff_src_md(), "DATA");

        for (int i = 0; i < 3; i++)
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), pd()->block[i]);

        CHECK(create_kernel(
                engine, &kernel_, "ref_softmax_bwd_generic", kernel_ctx));
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
