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

#ifndef CPU_AARCH64_ACL_SOFTMAX_HPP
#define CPU_AARCH64_ACL_SOFTMAX_HPP

#include "cpu/cpu_softmax_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/experimental/operators/CpuSoftmax.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_softmax_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    float beta;
    int32_t axis;
    bool is_logsoftmax;
};

struct acl_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_softmax_fwd_t);
        status_t init(engine_t *engine);

        acl_softmax_conf_t asp_;
    }; // pd_t

    // constructor
    acl_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const;

    status_t init(engine_t *engine) override;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    std::unique_ptr<arm_compute::experimental::op::CpuSoftmax> softmax_op_;
}; // acl_softmax_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
