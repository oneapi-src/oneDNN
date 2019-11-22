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

#ifndef GEMM_F32_MATMUL_HPP
#define GEMM_F32_MATMUL_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "cpu_matmul_pd.hpp"

#include "cpu/cpu_isa_traits.hpp"
#include "cpu/gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct gemm_f32_matmul_t : public primitive_impl_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:any", gemm_f32_matmul_t);

        status_t init();
    };

    gemm_f32_matmul_t(const pd_t *apd) : primitive_impl_t(apd) {
        pp_kernel_.reset(new pp_kernel_t(pd()->N(), pd()->attr(),
                pd()->desc()->bias_desc.data_type, true));
    }

    static constexpr data_type_t src_type = data_type::f32;
    static constexpr data_type_t weights_type = data_type::f32;
    static constexpr data_type_t dst_type = data_type::f32;
    static constexpr data_type_t acc_type = data_type::f32;

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

    using pp_kernel_t = inner_product_utils::pp_kernel_t<acc_type, dst_type>;
    std::unique_ptr<pp_kernel_t> pp_kernel_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
