/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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

#ifndef ACL_LOWP_MATMUL_SQ_HPP
#define ACL_LOWP_MATMUL_SQ_HPP

#include <random>

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "cpu/aarch64/acl_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_lowp_matmul_sq_obj_t {
    arm_compute::GEMMLowpOutputStageInfo info;
    arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_lowp_matmul_sq_conf_t {
    bool with_bias;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::GEMMInfo gemm_info;
};

struct acl_lowp_matmul_sq_resource_t : public resource_t {
    acl_lowp_matmul_sq_resource_t()
        : acl_obj_(utils::make_unique<acl_lowp_matmul_sq_obj_t>()) {}

    status_t configure(const acl_lowp_matmul_sq_conf_t &almc);

    acl_lowp_matmul_sq_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_lowp_matmul_sq_resource_t);

private:
    std::unique_ptr<acl_lowp_matmul_sq_obj_t> acl_obj_;
};

struct acl_lowp_matmul_sq_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("lowp_gemm_sq:acl", acl_lowp_matmul_sq_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        status_t init_scratchpad(engine_t *engine,
                memory_tracking::registrar_t &scratchpad,
                acl_post_ops_t &post_ops, dnnl::impl::post_ops_t &attr_post_ops,
                arm_compute::ActivationLayerInfo &act_info,
                const dnnl::impl::memory_desc_t &dst_md);

        acl_lowp_matmul_sq_conf_t almc_;
        acl_post_ops_t acl_post_ops;
    };

    acl_lowp_matmul_sq_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(engine_t *engine, resource_mapper_t &mapper) const;

    status_t execute(exec_ctx_t &ctx) const;

private:
    mutable std::mutex mtx_;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_LOWP_MATMUL_HPP