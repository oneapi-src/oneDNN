/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#ifndef ACL_MATMUL_HPP
#define ACL_MATMUL_HPP

#include "common/utils.hpp"
#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const cpu_matmul_pd_t *hint_fwd_pd)
            : cpu_matmul_pd_t(adesc, attr, hint_fwd_pd), amp_() {}

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:acl", acl_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        acl_matmul_conf_t amp_;
        acl_post_ops_t acl_post_ops;
        dnnl::impl::format_kind_t weights_format_kind_;

    protected:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0;
        }
    };

    acl_matmul_t(const pd_t *apd)
        : primitive_t(apd), acl_obj_(std::make_unique<acl_matmul_obj_t>()) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->weights_format_kind_ == format_kind::any) {
            return execute_forward<true>(ctx);
        } else {
            return execute_forward<false>(ctx);
        }
    }

private:
    template <bool IsFixedFormat>
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<acl_matmul_obj_t> acl_obj_;
}; // acl_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
