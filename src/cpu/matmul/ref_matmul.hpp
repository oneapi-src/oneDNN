/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef REF_MATMUL_HPP
#define REF_MATMUL_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_matmul_pd.hpp"

#include "cpu_isa_traits.hpp"
#include "ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t weights_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct ref_matmul_t : public primitive_impl_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_matmul_t);

        status_t init() {
            using namespace data_type;

            bool ok = src_md()->data_type == src_type
                    && weights_md()->data_type == weights_type
                    && desc()->accum_data_type == acc_type
                    && dst_md()->data_type == dst_type
                    && IMPLICATION(
                            acc_type == s32, attr()->zero_points_.common())
                    && IMPLICATION(acc_type != s32,
                            attr()->zero_points_.has_default_values())
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale_runtime
                            | primitive_attr_t::skip_mask_t::zero_points_runtime
                            | primitive_attr_t::skip_mask_t::post_ops)
                    && attr_oscale_ok() && attr_post_ops_ok()
                    && set_default_formats();

            if (with_bias()) {
                auto bia_dt = weights_md(1)->data_type;
                if (acc_type == f32)
                    ok = ok && utils::one_of(bia_dt, f32);
                else if (acc_type == s32)
                    ok = ok && utils::one_of(bia_dt, f32, s32, s8, u8);
            }

            return ok ? status::success : status::unimplemented;
        }

    private:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0 || oscale.mask_ == (1 << (batched() + 1));
        }

        bool attr_post_ops_ok() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            switch (p.len_) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }
    };

    ref_matmul_t(const pd_t *apd) : primitive_impl_t(apd) {
        int e_idx = pd()->attr()->post_ops_.find(primitive_kind::eltwise);
        if (e_idx != -1)
            eltwise_ker_.reset(new ref_eltwise_scalar_fwd_t(
                    pd()->attr()->post_ops_.entry_[e_idx].eltwise));
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;

    std::unique_ptr<ref_eltwise_scalar_fwd_t> eltwise_ker_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
