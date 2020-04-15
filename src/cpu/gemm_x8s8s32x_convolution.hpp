/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_ISA_STR,
                _gemm_x8s8s32x_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(
                            src_type, s8, data_type::undef, dst_type, s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, f32, s32,
                                    s8, u8))
                    && !has_zero_dim_memory()
                    && set_default_formats_common(
                            dat_tag(), format_tag::any, dat_tag())
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops)
                    && output_scales_mask_ok() && post_ops_ok()
                    && memory_desc_matches_tag(*src_md(), dat_tag())
                    && memory_desc_matches_tag(*dst_md(), dat_tag())
                    && set_or_check_wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md(), weights_md(0), dst_md(),
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            int ndims = src_md()->ndims;
            return utils::pick(ndims - 3, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc);
        }

        bool set_or_check_wei_format() {
            using namespace format_tag;
            int ndims = src_md()->ndims;

            const bool is_src_s8 = src_md_.data_type == data_type::s8;

            memory_desc_t want_wei_md = weights_md_;
            memory_desc_init_by_tag(want_wei_md,
                    with_groups() ? utils::pick(ndims - 3, wigo, hwigo, dhwigo)
                                  : utils::pick(ndims - 3, wio, hwio, dhwio));

            if (is_src_s8) {
                want_wei_md.extra.flags = 0
                        | memory_extra_flags::compensation_conv_s8s8
                        | memory_extra_flags::scale_adjust;
                want_wei_md.extra.compensation_mask
                        = (1 << 0) + (with_groups() ? (1 << 1) : 0);
                want_wei_md.extra.scale_adjust = 1.f;
#if DNNL_X64
                if (!x64::mayiuse(x64::avx512_core_vnni))
                    want_wei_md.extra.scale_adjust = 0.5f;
#endif
            }

            if (weights_md_.format_kind == format_kind::any) {
                weights_md_ = want_wei_md;
                return true;
            }

            return weights_md_ == want_wei_md;
        }

        bool output_scales_mask_ok() const {
            const auto &mask = attr()->output_scales_.mask_;
            return mask == 0 || mask == 1 << 1;
        }

        bool post_ops_ok() const {
            using namespace dnnl::impl::primitive_kind;
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            switch (po.len_) {
                case 0: return true;
                case 1: return is_eltwise(0) || po.contain(sum, 0);
                case 2:
                    return (po.contain(sum, 0) && is_eltwise(1))
                            || (po.contain(sum, 1) && is_eltwise(0));
                default: return false;
            }
            return false;
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {
        pp_ker_.reset(pp_ker_t::create(pd(), pd()->jcp_));
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;

    int nthr_ = 0;

    using pp_ker_t = gemm_x8s8s32x_convolution_utils::pp_ker_t;
    std::unique_ptr<pp_ker_t> pp_ker_;
};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_ISA_STR,
                _gemm_u8s8s32x_convolution_bwd_data_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(
                            dst_type, s8, data_type::undef, u8, s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, f32, s32,
                                    s8, u8))
                    && !has_zero_dim_memory()
                    && set_default_formats_common(
                            dat_tag(), wei_tag(), dat_tag())
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale)
                    && output_scales_mask_ok()
                    && memory_desc_matches_tag(*diff_src_md(), dat_tag())
                    && memory_desc_matches_tag(*diff_dst_md(), dat_tag())
                    && memory_desc_matches_tag(*weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md(), weights_md(), diff_dst_md(),
                    dnnl_get_max_threads());
        }

        virtual bool support_bias() const override { return true; }

        conv_gemm_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            int ndims = diff_src_md()->ndims;
            return utils::pick(ndims - 3, format_tag::nwc, format_tag::nhwc,
                    format_tag::ndhwc);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            int ndims = diff_src_md()->ndims;
            return with_groups() ? utils::pick(ndims - 3, wigo, hwigo, dhwigo)
                                 : utils::pick(ndims - 3, wio, hwio, dhwio);
        }
        bool output_scales_mask_ok() const {
            const auto &mask = attr()->output_scales_.mask_;
            return mask == 0 || mask == 1 << 1;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    void execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
