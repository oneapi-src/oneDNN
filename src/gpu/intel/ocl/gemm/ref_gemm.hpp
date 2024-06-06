/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEMM_REF_GEMM_HPP
#define GPU_INTEL_OCL_GEMM_REF_GEMM_HPP

#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_gemm_t : public gpu_gemm_t {
    using gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            const auto a_dt = desc()->a_type();
            const auto b_dt = desc()->b_type();
            const auto c_dt = desc()->c_type();
            const auto acc_dt = desc()->acc_type;
            const auto bia_dt = desc()->bias_type();
            const bool wei_decompress = utils::one_of(b_dt, f32, f16, bf16)
                    && utils::one_of(c_dt, f32, f16, bf16)
                    && utils::one_of(a_dt, u8, s8);

            const auto ndims = desc()->c_desc.ndims;
            const auto a_strides = desc()->a_desc.format_desc.blocking.strides;
            const auto b_strides = desc()->b_desc.format_desc.blocking.strides;
            const auto c_strides = desc()->c_desc.format_desc.blocking.strides;

            VDISPATCH_GEMM(
                    IMPLICATION(acc_dt == s32, attr()->zero_points_.common()),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_GEMM(!has_blocks(), VERBOSE_UNSUPPORTED_FEATURE,
                    "blocked format");
            VDISPATCH_GEMM(desc()->c_desc.ndims <= 3, VERBOSE_BAD_NDIMS,
                    "desc()->c_desc.ndims", desc()->c_desc.ndims);
            VDISPATCH_GEMM(
                    (a_strides[ndims - 1] == 1 || a_strides[ndims - 2] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM(
                    (b_strides[ndims - 1] == 1 || b_strides[ndims - 2] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM((c_strides[ndims - 1] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM(
                    IMPLICATION(desc()->is_batched(),
                            desc()->a_desc.dims[0] == desc()->b_desc.dims[0]),
                    VERBOSE_INCONSISTENT_NDIMS, "desc()->a_desc.dims[0]",
                    "desc()->b_desc.dims[0]");
            VDISPATCH_GEMM(IMPLICATION(acc_dt != s32 && !wei_decompress,
                                   attr()->zero_points_.has_default_values()),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_GEMM(
                    attr()->has_default_values(smask_t::zero_points_runtime
                            | smask_t::post_ops | smask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(attr_oscale_ok(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(attr_zp_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_GEMM(attr_post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_GEMM(desc()->sum_ab == sum_ab::sum_none,
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(
                    ((utils::one_of(a_dt, u8, s8) && utils::one_of(b_dt, u8, s8)
                             && utils::one_of(c_dt, f32, s8, u8, s32)
                             && IMPLICATION(with_bias(),
                                     utils::one_of(bia_dt, f32, u8, s8, s32)))
                            || (utils::one_of(a_dt, f8_e5m2, f8_e4m3)
                                    && utils::one_of(b_dt, f8_e5m2, f8_e4m3)
                                    && utils::one_of(c_dt, f32, f16, bf16,
                                            f8_e5m2, f8_e4m3)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, f32, f8_e5m2,
                                                    f8_e4m3)))
                            || (utils::everyone_is(f32, a_dt, b_dt, c_dt)
                                    && IMPLICATION(with_bias(), bia_dt == f32))
                            || (utils::one_of(a_dt, u8, s8)
                                    && (utils::everyone_is(f32, b_dt, c_dt)
                                            || utils::everyone_is(
                                                    f16, b_dt, c_dt))
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(
                                                    bia_dt, f32, u8, s8, s32)))
                            || (utils::everyone_is(f16, a_dt, b_dt)
                                    && utils::one_of(c_dt, u8, s8, f16)
                                    && IMPLICATION(with_bias(), bia_dt == f16))
                            || wei_decompress
                            || (utils::everyone_is(bf16, a_dt, b_dt)
                                    && utils::one_of(c_dt, bf16, f32)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, bf16, f32)))),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            attr_info = attr_info_t::create(attr());

            return status::success;
        }

        bool set_default_formats() {
            return gpu_gemm_pd_t::set_default_formats();
        }

        bool attr_oscale_ok() const {
            const auto &scales = attr()->scales_;
            return scales.get(DNNL_ARG_SRC).mask_ == 0
                    && scales.get(DNNL_ARG_WEIGHTS).mask_ == 0
                    && scales.get(DNNL_ARG_DST).mask_ == 0;
        }

        bool attr_zp_ok() const {
            return this->attr()->zero_points_.common(DNNL_ARG_A)
                    && this->attr()->zero_points_.common(DNNL_ARG_B)
                    && this->attr()->zero_points_.common(DNNL_ARG_C);
        }

        bool attr_post_ops_ok() const {
            using namespace primitive_kind;
            using namespace data_type;
            const auto &p = attr()->post_ops_;
            switch (p.len()) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }

        bool with_bias() const {
            return desc()->bias_type() != data_type::undef;
        }

        attr_info_t attr_info = {};
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int(
                "NON_DEFAULT_ATTRS", !pd()->attr()->has_default_values());

        const auto d = pd()->desc();
        kernel_ctx.set_data_type(d->c_type());
        CHECK(def_attr_info(kernel_ctx, pd()->attr_info,
                pd()->attr()->post_ops_, *pd()->dst_md()));

        const auto bias_type = d->bias_type() != data_type::undef
                ? d->bias_type()
                : data_type::f32;
        def_data_type(kernel_ctx, d->a_type(), "A");
        def_data_type(kernel_ctx, d->b_type(), "B");
        def_data_type(kernel_ctx, d->c_type(), "C");
        def_data_type(kernel_ctx, d->acc_type, "ACC");
        def_data_type(kernel_ctx, bias_type, "BIA");
        CHECK(create_kernel(engine, &kernel_, "ref_gemm", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
