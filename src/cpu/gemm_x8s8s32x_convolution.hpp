/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef CPU_GEMM_X8S8S32X_CONVOLUTION_HPP
#define CPU_GEMM_X8S8S32X_CONVOLUTION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/platform.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/gemm_convolution_utils.hpp"
#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/zero_point_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct gemm_x8s8s32x_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(src_md()->data_type == data_type::u8
                        ? IGEMM_S8U8S32_IMPL_STR
                        : IGEMM_S8S8S32_IMPL_STR,
                gemm_x8s8s32x_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(src_md()->data_type, s8, u8)
                    && weights_md()->data_type == s8
                    && utils::one_of(
                            dst_md()->data_type, f32, bf16, s32, s8, u8)
                    && IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, bf16,
                                    s32, s8, u8))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(skip_mask_t::scales_runtime
                                    | skip_mask_t::zero_points_runtime
                                    | skip_mask_t::post_ops
                                    | skip_mask_t::sum_dt,
                            dst_type)
                    && attr()->post_ops_.check_sum_consistency(dst_type,
                            /* is_int8 */ true)
                    && attr_scales_ok() && zero_points_valid(attr());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            CHECK(jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads()));
            if (!gemm_x8s8s32x_convolution_utils::post_ops_ok(
                        attr()->post_ops_, &dst_md_))
                return status::unimplemented;
            return status::success;
        }

        conv_gemm_conf_t jcp_;
    };

    gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(pp_ker_, pp_ker_t::create(pd(), pd()->jcp_)));
        return (pp_ker_) ? pp_ker_->create_kernel() : status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr(const int ithr, const int nthr,
            const char *src_base, const int8_t *wei_base, const char *bia_base,
            void *dst_base, const float *scales, const float *dst_scales,
            const zero_point_call_params_t &zp,
            const memory_tracking::grantor_t &scratchpad,
            const void *post_ops_binary_rhs_arg_vec,
            const exec_ctx_t &ctx) const;

    using pp_ker_t = gemm_x8s8s32x_convolution_utils::pp_ker_t;
    std::unique_ptr<pp_ker_t> pp_ker_;
};

struct gemm_x8s8s32x_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(diff_dst_md()->data_type == data_type::u8
                        ? IGEMM_S8U8S32_IMPL_STR
                        : IGEMM_S8S8S32_IMPL_STR,
                gemm_x8s8s32x_convolution_bwd_data_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(diff_dst_md()->data_type, s8, u8)
                    && weights_md()->data_type == s8
                    && utils::one_of(
                            diff_src_md()->data_type, f32, bf16, s32, s8, u8)
                    && IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, bf16,
                                    s32, s8, u8))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::scales_runtime)
                    && output_scales_mask_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_, bias_md_,
                    attr_, dnnl_get_max_threads());
        }

        bool support_bias() const override { return true; }

        conv_gemm_conf_t jcp_;

    protected:
        bool output_scales_mask_ok() const {
            const auto &mask = attr()->output_scales_.mask_;
            return mask == 0 || mask == 1 << 1;
        }
    };

    gemm_x8s8s32x_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_thr(const int ithr, const int nthr,
            const char *diff_dst_base, const int8_t *wei_base,
            const char *bia_base, char *diff_src_base,
            const memory_tracking::grantor_t &scratchpad,
            const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
