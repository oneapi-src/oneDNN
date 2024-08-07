/*******************************************************************************
* Copyright 2020-2024 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP

#include "common/memory_tracking.hpp"
#include "cpu/cpu_convolution_pd.hpp"

#include "acl_convolution_utils.hpp"
#include "acl_post_ops.hpp"
#include "arm_compute/runtime/experimental/operators/CpuGemmConv2d.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <data_type_t src_type, data_type_t wei_type = src_type,
        data_type_t dst_type = src_type, data_type_t bia_type = dst_type>
struct acl_gemm_convolution_fwd_t : public primitive_t {

    using Op = arm_compute::experimental::op::CpuGemmConv2d;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), acp_() {}

        DECLARE_COMMON_PD_T(
                "gemm:acl", acl_gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        acl_conv_conf_t acp_;
        acl_post_ops_t post_ops;

    protected:
        bool output_scales_mask_ok() const;
        bool zero_points_ok() const;

    private:
        status_t init_conf();
    };

    acl_gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), acl_obj_(std::make_unique<acl_obj_t<Op>>()) {}

    status_t init(engine_t *engine) override;

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<bia_type>::type bia_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<acl_obj_t<Op>> acl_obj_;
}; // acl_gemm_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
