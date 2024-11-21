/*******************************************************************************
* Copyright 2023-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_DEPTHWISE_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_DEPTHWISE_CONVOLUTION_HPP

#include "acl_convolution_utils.hpp"
#include "arm_compute/runtime/experimental/operators/CpuDepthwiseConv2d.h"
#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_depthwise_convolution_fwd_t : public primitive_t {

    using Op = arm_compute::experimental::op::CpuDepthwiseConv2d;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("depthwise_convolution:acl",
                acl_depthwise_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        acl_conv_conf_t acp_ = utils::zero<decltype(acp_)>();
        acl_post_ops_t post_ops;
    };

    acl_depthwise_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), acl_obj_(std::make_unique<acl_obj_t<Op>>()) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t init(engine_t *engine) override;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<acl_obj_t<Op>> acl_obj_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_DEPTHWISE_CONVOLUTION_HPP
