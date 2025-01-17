/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_avx512_core_amx_conv_kernel.hpp"
#include "cpu/x64/jit_avx512_core_scale_precompute.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jcp_.isa, ""),
                jit_avx512_core_amx_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            bool is_bf16_convolution
                    = (src_md(0)->data_type == bf16
                              && weights_md(0)->data_type == bf16
                              && utils::one_of(dst_md(0)->data_type, f32, bf16))
                    && IMPLICATION(with_bias(),
                            utils::one_of(weights_md(1)->data_type, f32, bf16))
                    && attr()->has_default_values(smask_t::post_ops);
            bool is_int8_convolution
                    = utils::one_of(src_md(0)->data_type, s8, u8)
                    && weights_md(0)->data_type == s8
                    && utils::one_of(
                            dst_md(0)->data_type, s8, u8, s32, f32, bf16)
                    && IMPLICATION(with_bias(),
                            utils::one_of(
                                    weights_md(1)->data_type, f32, s32, s8, u8))
                    && attr()->has_default_values(smask_t::scales_runtime
                                    | smask_t::post_ops
                                    | smask_t::zero_points_runtime
                                    | smask_t::sum_dt,
                            dst_md(0)->data_type);

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV((is_bf16_convolution || is_int8_convolution),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(
                                   dst_md(0)->data_type,
                                   /* is_int8 */ is_int8_convolution),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_CONV(zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);

            // TODO: make `init_conf` assign initialized object to `jcp_`
            CHECK(jit_avx512_core_amx_fwd_kernel_t::init_conf(jcp_, *desc(),
                    src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            CHECK(jit_avx512_core_amx_fwd_kernel_t::init_scratchpad(
                    scratchpad, jcp_, *attr()));

            return status::success;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();

    protected:
        bool zero_points_ok() const {
            const auto &zp = attr()->zero_points_;

            if (!zp.has_default_values(DNNL_ARG_SRC)) {
                int mask_src = zp.get_mask(DNNL_ARG_SRC);
                const bool ok = mask_src == 0;
                if (!ok) return false;
            }
            if (!zp.has_default_values(DNNL_ARG_DST)) {
                int mask_dst = zp.get_mask(DNNL_ARG_DST);
                const bool ok = mask_dst == 0;
                if (!ok) return false;
            }

            return zp.has_default_values(DNNL_ARG_WEIGHTS);
        }
    };

    jit_avx512_core_amx_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_amx_fwd_kernel_t(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        CHECK(kernel_->create_kernel());

        // JIT to precompute scales
        const bool is_jit_supported = mayiuse(avx512_core);
        const auto attr = pd()->attr();
        if (is_jit_supported && pd()->OC() > 1 && req_copy_scales(attr)) {
            const auto &attr_scales = attr->scales_;
            int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
            if (wei_scale_mask > 0) {
                CHECK(safe_ptr_assign(jit_scale_precompute_,
                        new jit_avx512_core_scale_precompute_t(attr)));
                CHECK(jit_scale_precompute_->create_kernel());
            }
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const auto &_pd = pd();
        if (_pd->jcp_.is_depthwise)
            return status::unimplemented;
        else if (_pd->jcp_.is_relo)
            return execute_forward_reduced_lowering(ctx);
        return execute_forward(ctx);
    }

private:
    status_t execute_forward_reduced_lowering(const exec_ctx_t &ctx) const;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    void prepare_padded_bias(const char *&bias,
            const memory_tracking::grantor_t &scratchpad) const;

    std::unique_ptr<jit_avx512_core_amx_fwd_kernel_t> kernel_;
    std::unique_ptr<jit_avx512_core_scale_precompute_t> jit_scale_precompute_;
};

struct jit_avx512_core_amx_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jcp_.isa, ""),
                jit_avx512_core_amx_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            const data_type_t wdt = weights_md_.data_type;
            bool is_xf16_convolution
                    = utils::one_of(wdt, data_type::bf16, data_type::f16)
                    && diff_dst_md_.data_type == wdt
                    && utils::one_of(
                            diff_src_md_.data_type, data_type::f32, wdt);

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(is_xf16_convolution, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            // TODO: make `init_conf` assign initialized object to `jcp_`
            CHECK(jit_avx512_core_amx_bwd_data_kernel_t::init_conf(jcp_,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_,
                    nullptr /* no bias */, attr_, dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_amx_bwd_data_kernel_t::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status::success;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
    };

    jit_avx512_core_amx_convolution_bwd_data_t(const pd_t *apd)
        : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_amx_bwd_data_kernel_t(
                        pd()->jcp_, *pd()->attr())));
        CHECK(kernel_->create_kernel());

        // JIT to precompute scales
        const bool is_jit_supported = mayiuse(avx512_core);
        const auto attr = pd()->attr();
        if (is_jit_supported && pd()->OC() > 1 && req_copy_scales(attr)) {
            const auto &attr_scales = attr->scales_;
            int wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
            if (wei_scale_mask > 0) {
                CHECK(safe_ptr_assign(jit_scale_precompute_,
                        new jit_avx512_core_scale_precompute_t(attr)));
                CHECK(jit_scale_precompute_->create_kernel());
            }
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const auto &_pd = pd();
        if (_pd->jcp_.is_depthwise) {
            assert(!"_pd->jcp_.is_depthwise not implemented");
            return status::unimplemented;
        } else
            return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_core_amx_bwd_data_kernel_t> kernel_;
    std::unique_ptr<jit_avx512_core_scale_precompute_t> jit_scale_precompute_;
};

struct jit_avx512_core_amx_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", jcp_.isa, ""),
                jit_avx512_core_amx_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_CONV(is_bwd_w(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(
                    (expect_data_types(bf16, bf16, data_type::undef, bf16,
                             data_type::undef)
                            || expect_data_types(bf16, f32, data_type::undef,
                                    bf16, data_type::undef)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(
                    IMPLICATION(with_bias(),
                            utils::one_of(diff_bias_md_.data_type, f32, bf16)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);

            // TODO: make `init_conf` assign initialized object to `jcp_`
            CHECK(jit_avx512_core_amx_bwd_weights_kernel_t::init_conf(jcp_,
                    *desc(), src_md_, diff_weights_md_, diff_bias_md_,
                    diff_dst_md_, dnnl_get_max_threads()));

            auto scratchpad = scratchpad_registry().registrar();
            CHECK(jit_avx512_core_amx_bwd_weights_kernel_t::init_scratchpad(
                    scratchpad, jcp_, src_md_, diff_weights_md_, diff_dst_md_));

            return status::success;
        }

        jit_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
    };

    jit_avx512_core_amx_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    struct thread_info_t;

    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    void compute_diff_weights_2d(const thread_info_t *) const;
    void compute_diff_weights_3d(const thread_info_t *) const;
    void compute_diff_weights(const thread_info_t *) const;
    void reduce_and_convert_diff_weights_and_bias(const thread_info_t *) const;
    void store_in_vnni_format(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t tr_src_buf_number(const thread_info_t *ti, int g, int ic) const;
    size_t tr_diff_dst_buf_number(const thread_info_t *ti, int g, int oc) const;
    void trans_src_nxc(src_data_t *tr_src, const src_data_t *src_base,
            int spatial_start, dim_t spatial_start_offset, int icb_start,
            dim_t chb_stride, int my_work) const;
    void trans_dst_nxc(diff_dst_data_t *tr_diff_dst,
            const diff_dst_data_t *diff_dst_base, int spatial_start,
            dim_t spatial_start_offset, int ocb_start, dim_t chb_stride,
            int my_work) const;

    int nthr_ = 0, nthr_mb_ = 0, nthr_g_ = 0, nthr_oc_b_ = 0, nthr_ic_b_ = 0;

    std::unique_ptr<jit_avx512_core_amx_bwd_weights_kernel_t> kernel_;

    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;

    std::unique_ptr<jit_diff_wei_trans_to_vnni_t> diff_wei_trans_kernel_;
    std::unique_ptr<jit_trans_src_t> trans_kernel_;
    std::unique_ptr<jit_trans_dst_t> trans_dst_kernel_;

    inline dim_t wei_offset_int(int g, int oc_b, int ic_b, int kX) const {
        const auto &jcp = kernel_->jcp;
        const dim_t const_extra_offset = jcp.kw * jcp.ic_block * jcp.oc_block;
        dim_t extra_offset = (jcp.ndims == 5) ? kX * jcp.kh * const_extra_offset
                                              : kX * const_extra_offset;
        return (dim_t)((g * jcp.nb_oc + oc_b) * jcp.nb_ic + ic_b) * jcp.kd
                * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block
                + extra_offset;
    }
    inline dim_t wei_offset_ext(int g, int oc_b, int ic_b, int kX) const {
        const auto &jcp = kernel_->jcp;
        const int nb_ic = utils::div_up(jcp.ic, 2 * jcp.ic_block);
        const dim_t const_extra_offset
                = static_cast<dim_t>(jcp.kw) * jcp.ic_block * jcp.oc_block * 2;
        dim_t extra_offset = (jcp.ndims == 5) ? kX * jcp.kh * const_extra_offset
                                              : kX * const_extra_offset;
        return (dim_t)((g * jcp.nb_oc + oc_b) * nb_ic + ic_b) * jcp.kd * jcp.kh
                * jcp.kw * jcp.ic_block * jcp.oc_block * 2
                + extra_offset;
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
