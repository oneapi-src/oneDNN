/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX2_X8S8S32X_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX2_X8S8S32X_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/x64/jit_avx2_x8s8s32x_conv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx2_x8s8s32x_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_int8:", avx2, ""),
                jit_avx2_x8s8s32x_convolution_fwd_t);

        status_t init(engine_t *engine) {
            const bool args_ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, data_type::s8,
                            data_type::undef, dst_type, data_type::s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(bias_md_.data_type, data_type::f32,
                                    data_type::s32, data_type::s8,
                                    data_type::u8))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops)
                    && !has_zero_dim_memory();
            if (!args_ok) return status::unimplemented;

            const auto status = jit_avx2_x8s8s32x_fwd_kernel::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, *attr(),
                    dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_x8s8s32x_fwd_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx2_x8s8s32x_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {
        kernel_ = new jit_avx2_x8s8s32x_fwd_kernel(pd()->jcp_, *pd()->attr());
    }

    ~jit_avx2_x8s8s32x_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        const auto &_pd = pd();
        const int ndims = _pd->ndims();
        const bool is_dw = _pd->jcp_.is_depthwise;

        switch (ndims) {
            case 3: execute_forward_1d(ctx); break;
            case 4:
                if (is_dw)
                    execute_forward_2d_dw(ctx);
                else
                    execute_forward_2d(ctx);
                break;
            case 5: execute_forward_3d(ctx); break;
            default: return status::unimplemented;
        }
        return status::success;
    }

private:
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    void execute_forward_2d_dw(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    jit_avx2_x8s8s32x_fwd_kernel *kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
