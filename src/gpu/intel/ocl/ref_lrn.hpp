/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_LRN_HPP
#define GPU_INTEL_OCL_REF_LRN_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_lrn_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_lrn_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_lrn_fwd_pd_t {
        using gpu_lrn_fwd_pd_t::gpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_LRN(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(utils::one_of(src_md()->data_type, f32, f16, bf16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(IMPLICATION(src_md()->data_type == f16,
                                  compute_engine->mayiuse(
                                          compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");

            if (desc_.prop_kind == prop_kind::forward_training) {
                ws_md_ = *src_md();
                if (ws_md_.data_type == data_type::bf16
                        || ws_md_.data_type == data_type::f16)
                    ws_md_.data_type = data_type::f32;
            }

            dispatch = compute_engine->create_dispatch(src_md());
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("IC", 1, C());
            dispatch.define_dim("ID", nstl::max(1, src_md()->ndims - 3), D());
            dispatch.define_dim("IH", nstl::max(1, src_md()->ndims - 2), H());
            dispatch.define_dim("IW", nstl::max(1, src_md()->ndims - 1), W());
            dispatch.generate();

            return status::success;
        }

        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(desc->src_desc.data_type, false);

        kernel_ctx.define_int("IS_FWD", 1);

        if (desc->prop_kind == prop_kind::forward_training)
            kernel_ctx.define_int("IS_TRAINING", 1);

        switch (desc->alg_kind) {
            case lrn_across_channels:
                kernel_ctx.define_int("ACROSS_CHANNEL", 1);
                break;
            case lrn_within_channel:
                kernel_ctx.define_int("WITHIN_CHANNEL", 1);
                break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const int ndims = src_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("IC", pd()->C());
        kernel_ctx.define_int("ID", pd()->D());
        kernel_ctx.define_int("IH", pd()->H());
        kernel_ctx.define_int("IW", pd()->W());

        const dim_t round_norm_size = desc->local_size;
        dim_t num_elements = pow(round_norm_size, nstl::max(0, ndims - 2));
        if (desc->alg_kind == lrn_across_channels) {
            num_elements = round_norm_size;
        }
        const float num_element_div = 1.f / (float)num_elements;
        const auto padding = (desc->local_size - 1) / 2;

        kernel_ctx.define_float("NUM_ELEMENTS_DIV", num_element_div);
        kernel_ctx.define_int("PADDING", padding);
        kernel_ctx.define_int(
                "LOCAL_SIZE", desc->local_size - 1 + desc->local_size % 2);
        kernel_ctx.define_float("LRN_ALPHA", desc->lrn_alpha);
        kernel_ctx.define_float("LRN_BETA", desc->lrn_beta);
        kernel_ctx.define_float("LRN_K", desc->lrn_k);

        offsets_t off;
        set_offsets(src_d, off.src_off);
        set_offsets(dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        CHECK(create_kernel(engine, &kernel_, "ref_lrn_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_lrn_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_lrn_bwd_pd_t {
        using gpu_lrn_bwd_pd_t::gpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            VDISPATCH_LRN(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    utils::everyone_is(src_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");

            ws_md_ = *src_md();
            if (utils::one_of(
                        ws_md_.data_type, data_type::bf16, data_type::f16))
                ws_md_.data_type = data_type::f32;

            VDISPATCH_LRN(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);

            dispatch = compute_engine->create_dispatch(diff_src_md());
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("IC", 1, C());
            dispatch.define_dim("ID", nstl::max(1, src_md()->ndims - 3), D());
            dispatch.define_dim("IH", nstl::max(1, src_md()->ndims - 2), H());
            dispatch.define_dim("IW", nstl::max(1, src_md()->ndims - 1), W());
            dispatch.generate();

            return status::success;
        }

        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(desc->src_desc.data_type, false);

        kernel_ctx.define_int("IS_BWD", 1);

        switch (desc->alg_kind) {
            case lrn_across_channels:
                kernel_ctx.define_int("ACROSS_CHANNEL", 1);
                break;
            case lrn_within_channel:
                kernel_ctx.define_int("WITHIN_CHANNEL", 1);
                break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
        const int ndims = src_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("IC", pd()->C());
        kernel_ctx.define_int("ID", pd()->D());
        kernel_ctx.define_int("IH", pd()->H());
        kernel_ctx.define_int("IW", pd()->W());

        const dim_t round_norm_size = desc->local_size;
        dim_t num_elements = pow(round_norm_size, nstl::max(0, ndims - 2));
        if (desc->alg_kind == lrn_across_channels) {
            num_elements = round_norm_size;
        }
        const float num_element_div = 1.f / (float)num_elements;
        const auto padding = (desc->local_size - 1) / 2;

        kernel_ctx.define_float("NUM_ELEMENTS_DIV", num_element_div);
        kernel_ctx.define_int("PADDING", padding);
        kernel_ctx.define_int(
                "LOCAL_SIZE", desc->local_size - 1 + desc->local_size % 2);
        kernel_ctx.define_float("LRN_ALPHA", desc->lrn_alpha);
        kernel_ctx.define_float("LRN_BETA", desc->lrn_beta);
        kernel_ctx.define_float("LRN_K", desc->lrn_k);

        offsets_t off;
        set_offsets(src_d, off.src_off);
        set_offsets(diff_dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        CHECK(create_kernel(engine, &kernel_, "ref_lrn_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
