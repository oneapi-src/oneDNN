/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#ifndef GEMM_X8S8S32X_INNER_PRODUCT_HPP
#define GEMM_X8S8S32X_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "gemm/gemm.hpp"
#include "gemm_inner_product_utils.hpp"
#include "jit_generator.hpp"

#include "cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct gemm_x8s8s32x_inner_product_fwd_t : public primitive_impl_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T(src_type == data_type::u8 ? IGEMM_S8U8S32_IMPL_STR
                                                      : IGEMM_S8S8S32_IMPL_STR,
                gemm_x8s8s32x_inner_product_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init() {
            using namespace data_type;

            bool ok = true && is_fwd() && !has_zero_dim_memory()
                    && src_md()->data_type == src_type
                    && dst_md()->data_type == dst_type
                    && weights_md()->data_type == s8
                    && IMPLICATION(with_bias(),
                            utils::one_of(
                                    weights_md(1)->data_type, f32, s32, s8, u8))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops)
                    && post_ops_ok() && set_default_params() == status::success
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md());
            if (!ok) return status::unimplemented;

            bool do_sum = attr()->post_ops_.find(primitive_kind::sum) >= 0;
            dst_is_acc_ = utils::one_of(dst_type, s32, f32) && !do_sum;

            init_scratchpad();

            return status::success;
        }

        bool dst_is_acc_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(false); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(false); };
            switch (po.len_) {
                case 0: return true; // no post_ops
                case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
                case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
                default: return false;
            }
            return false;
        }

    private:
        void init_scratchpad() {
            if (!dst_is_acc_) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                        sizeof(acc_data_t) * MB() * OC());
            }
        }
    };

    gemm_x8s8s32x_inner_product_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {
        pp_kernel_ = new inner_product_utils::pp_kernel_t<data_type::s32,
                dst_type>(apd, false);
    }
    ~gemm_x8s8s32x_inner_product_fwd_t() { delete pp_kernel_; }

    typedef typename prec_traits<dst_type>::type data_t;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    inner_product_utils::pp_kernel_t<data_type::s32, dst_type> *pp_kernel_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
