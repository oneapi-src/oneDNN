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

#ifndef GPU_INTEL_OCL_REF_INNER_PRODUCT_HPP
#define GPU_INTEL_OCL_REF_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_inner_product_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;

            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference),
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_INNER_PRODUCT_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);

            auto src_dt = src_md()->data_type;
            auto dst_dt = dst_md()->data_type;
            auto wei_dt = weights_md(0)->data_type;

            const bool is_f32 = src_dt == f32
                    && utils::one_of(wei_dt, f32, s8, u8)
                    && utils::one_of(dst_dt, f32, f16, bf16);
            const bool is_f16 = src_dt == f16
                    && utils::one_of(wei_dt, f16, s8, u8)
                    && utils::one_of(dst_dt, u8, s8, f16, bf16);
            const bool is_bf16 = src_dt == bf16
                    && utils::one_of(wei_dt, bf16, s8, u8)
                    && utils::one_of(dst_dt, bf16, f32);
            const bool is_int8 = utils::one_of(src_dt, u8, s8)
                    && utils::one_of(wei_dt, u8, s8)
                    && utils::one_of(dst_dt, f32, s8, u8, s32, f16, bf16);
            VDISPATCH_INNER_PRODUCT((is_int8 || is_f32 || is_f16 || is_bf16),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, u8, s8,
                                    bf16, f16, f32)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_INNER_PRODUCT(
                    post_ops_with_binary_ok(attr(), desc()->dst_desc.data_type),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_INNER_PRODUCT_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(!attr()->scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && attr_scales_ok()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_INNER_PRODUCT(
                    IMPLICATION(desc()->src_desc.data_type == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_INNER_PRODUCT_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_inner_product_fwd", kernel_ctx));
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

struct ref_inner_product_bwd_data_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        using gpu_inner_product_bwd_data_pd_t::gpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_data_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);

            VDISPATCH_INNER_PRODUCT(utils::one_of(this->desc()->prop_kind,
                                            backward, backward_data),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT_SC(
                    this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(true,
                            expect_data_types(
                                    f16, f16, data_type::undef, f16, f32),
                            expect_data_types(
                                    f32, f16, data_type::undef, f16, f32),
                            expect_data_types(
                                    bf16, bf16, data_type::undef, bf16, f32),
                            expect_data_types(
                                    f32, bf16, data_type::undef, bf16, f32),
                            expect_data_types(
                                    f32, f32, data_type::undef, f32, f32)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_INNER_PRODUCT_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_inner_product_bwd_data", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_inner_product_bwd_weights_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_inner_product_bwd_weights_pd_t {
        using gpu_inner_product_bwd_weights_pd_t::
                gpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_weights_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);

            VDISPATCH_INNER_PRODUCT(utils::one_of(this->desc()->prop_kind,
                                            backward, backward_weights),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT_SC(
                    this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f32),
                            expect_data_types(f16, f32, f32, f16, f32),
                            expect_data_types(bf16, bf16, bf16, bf16, f32),
                            expect_data_types(bf16, f32, f32, bf16, f32),
                            expect_data_types(f32, f32, f32, f32, f32)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_INNER_PRODUCT_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_inner_product_bwd_weights", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
