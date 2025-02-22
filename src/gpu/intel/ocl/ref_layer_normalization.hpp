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

#ifndef GPU_INTEL_OCL_REF_LAYER_NORMALIZATION_HPP
#define GPU_INTEL_OCL_REF_LAYER_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_layer_normalization_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_layer_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_layer_normalization_fwd_pd_t {
        using gpu_layer_normalization_fwd_pd_t::
                gpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_dt = src_md()->data_type;
            auto dst_dt = dst_md()->data_type;

            using skip_mask_t = primitive_attr_t::skip_mask_t;

            bool uses_f16 = utils::one_of(f16, src_dt, dst_dt);
            bool uses_f64 = utils::one_of(f64, src_dt, dst_dt);

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(IMPLICATION(uses_f16,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16))
                            && IMPLICATION(uses_f64,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(
                    !memory_desc_ndims_ok(src_md(), dst_md(), stat_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst stat");
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(attr()->has_default_values(skip_mask_t::scales),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        lnorm_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        kernel_ctx.define_int("WITH_SRC_SCALES",
                !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC));
        kernel_ctx.define_int("WITH_DST_SCALES",
                !pd()->attr()->scales_.has_default_values(DNNL_ARG_DST));

        CHECK(create_kernel(engine, &kernel_, "ref_lnorm_fwd", kernel_ctx));
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

struct ref_layer_normalization_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_layer_normalization_bwd_pd_t {
        using gpu_layer_normalization_bwd_pd_t::
                gpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("lnorm_ref:any", ref_layer_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_dt = src_md()->data_type;
            auto diff_dst_dt = diff_dst_md()->data_type;
            auto diff_src_dt = diff_src_md()->data_type;

            bool uses_f16
                    = utils::one_of(f16, src_dt, diff_dst_dt, diff_src_dt);
            bool uses_f64
                    = utils::one_of(f64, src_dt, diff_dst_dt, diff_src_dt);

            VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(IMPLICATION(uses_f16,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16))
                            && IMPLICATION(uses_f64,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        lnorm_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(engine, &kernel_, "ref_lnorm_bwd", kernel_ctx));
        if (pd()->conf.use_scale || pd()->conf.use_shift) {
            CHECK(create_kernel(engine, &kernel_scaleshift_,
                    "ref_lnorm_bwd_scaleshift", kernel_ctx));
            if (!kernel_scaleshift_) return status::runtime_error;
        }
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_scaleshift_;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
