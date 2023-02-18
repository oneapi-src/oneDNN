/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_SVE_X8S8S32X_CONVOLUTION_HPP
#define CPU_AARCH64_JIT_SVE_X8S8S32X_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/jit_sve_512_x8s8s32x_conv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_sve_512_x8s8s32x_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_int8:", sve_512, ""),
                jit_sve_512_x8s8s32x_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && utils::one_of(src_md(0)->data_type, s8, u8)
                    && weights_md(0)->data_type == s8
                    && IMPLICATION(with_bias(),
                            utils::one_of(
                                    weights_md(1)->data_type, f32, s32, s8, u8))
                    && utils::one_of(
                            dst_md(0)->data_type, f32, s32, s8, u8 /*, bf16*/)
                    && desc()->accum_data_type == s32
                    && attr()->has_default_values(smask_t::scales_runtime
                                    | smask_t::zero_points_runtime
                                    | smask_t::post_ops | smask_t::sum_dt,
                            dst_md(0)->data_type)
                    && attr()->post_ops_.check_sum_consistent_dt(
                            dst_md(0)->data_type)
                    && !has_zero_dim_memory() && zero_points_ok();
            if (!ok) return status::unimplemented;

            status_t status = jit_sve_512_x8s8s32x_fwd_kernel::init_conf(jcp_,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_sve_512_x8s8s32x_fwd_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool zero_points_ok() const {
            // Only common zero points are supported -> mask should only be 0
            int mask_src = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src);
            attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst);
            return attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && mask_src == 0 && mask_dst == 0;
        }
    };

    jit_sve_512_x8s8s32x_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_sve_512_x8s8s32x_fwd_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const auto &_pd = pd();
        if (_pd->ndims() == 3)
            return execute_forward_1d(ctx);
        else if (_pd->ndims() == 4)
            if (_pd->jcp_.is_depthwise)
                return execute_forward_2d_dw(ctx);
            else
                return execute_forward_2d(ctx);
        else if (_pd->ndims() == 5)
            return execute_forward_3d(ctx);
        return status::unimplemented;
    }

private:
    status_t execute_forward_1d(const exec_ctx_t &ctx) const;
    status_t execute_forward_2d(const exec_ctx_t &ctx) const;
    status_t execute_forward_2d_dw(const exec_ctx_t &ctx) const;
    status_t execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    const float *adjust_oscales(const memory_tracking::grantor_t &scratchpad,
            const float *src_scales, const float *wei_scales) const;

    std::unique_ptr<jit_sve_512_x8s8s32x_fwd_kernel> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
