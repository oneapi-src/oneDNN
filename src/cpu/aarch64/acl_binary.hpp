/*******************************************************************************
* Copyright 2022, 2024 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_ACL_BINARY_HPP
#define CPU_AARCH64_ACL_BINARY_HPP

#include "acl_utils.hpp"
#include "cpu/cpu_binary_pd.hpp"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_binary_conf_t {
    arm_compute::TensorInfo src0_info;
    arm_compute::TensorInfo src1_info;
    arm_compute::TensorInfo dst_info;
    alg_kind_t alg;
};

struct acl_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_binary_t);

        status_t init(engine_t *engine);

        acl_binary_conf_t asp_;

    private:
        arm_compute::Status validate(const acl_binary_conf_t &asp);

        friend struct acl_post_ops_t;
    }; // pd_t

    acl_binary_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t init(engine_t *engine) override;

    status_t execute_forward(const exec_ctx_t &ctx) const;
    // Execute forward with arbitrary src0, src1 and dst, used by acl_post_ops_t
    status_t execute_forward(const exec_ctx_t &ctx, const void *src0,
            const void *src1, void *dst) const;

    const pd_t *pd() const;

    friend struct acl_post_ops_t;

    std::unique_ptr<arm_compute::experimental::INEOperator> binary_op_;

}; // acl_binary_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
