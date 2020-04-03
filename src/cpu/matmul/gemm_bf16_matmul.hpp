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

#ifndef GEMM_BF16_MATMUL_HPP
#define GEMM_BF16_MATMUL_HPP

#include <assert.h>

#include "bfloat16.hpp"
#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "cpu_matmul_pd.hpp"

#include "gemm_based_common.hpp"

#include "cpu/gemm_inner_product_utils.hpp"
#include "primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

template <impl::data_type_t dst_type>
struct gemm_bf16_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:jit", gemm_bf16_matmul_t);

        status_t init(engine_t *engine);
        const gemm_based::params_t &params() const { return params_; }

    private:
        status_t check_and_configure_attributes();
        gemm_based::params_t params_;
    };

    gemm_bf16_matmul_t(const pd_t *apd) : primitive_t(apd) {
        if (pd()->params().has_pp_kernel_)
            pp_kernel_.reset(new pp_kernel_t(pd()->N(), pd()->M(),
                    &pd()->params().pp_attr_, pd()->desc()->bias_desc.data_type,
                    false));
    }

    static constexpr data_type_t src_type = data_type::bf16;
    static constexpr data_type_t weights_type = data_type::bf16;
    static constexpr data_type_t acc_type = data_type::f32;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;

    using pp_kernel_t = inner_product_utils::pp_kernel_t<acc_type, dst_type>;
    std::unique_ptr<pp_kernel_t> pp_kernel_;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
