/*******************************************************************************
* Copyright 2024 Arm Ltd. and affiliates
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

#ifndef ACL_LOWP_MATMUL_HPP
#define ACL_LOWP_MATMUL_HPP

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "cpu/aarch64/acl_post_ops.hpp"
#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_lowp_matmul_obj_t {
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor dst_s8_tensor;
    arm_compute::Tensor dst_cast_tensor;
    arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
    arm_compute::NEQuantizationLayer quant;
    arm_compute::NEDequantizationLayer dequant;
};

struct acl_lowp_matmul_conf_t {
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::TensorInfo dst_s8_tensor_info;
    arm_compute::TensorInfo dst_cast_tensor_info;
    arm_compute::GEMMInfo gemm_info;
    bool with_bias {false};
    bool use_dst_acc {false};
    bool dst_is_s8 {false};
    bool use_cast_acc {false};
    bool sum_is_fused {false};
};

struct acl_lowp_matmul_resource_t : public resource_t {
    acl_lowp_matmul_resource_t()
        : acl_obj_(utils::make_unique<acl_lowp_matmul_obj_t>()) {}

    status_t configure(const acl_lowp_matmul_conf_t &almc);

    acl_lowp_matmul_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_lowp_matmul_resource_t);

private:
    std::unique_ptr<acl_lowp_matmul_obj_t> acl_obj_;
};

struct acl_lowp_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                "lowp_gemm:acl", acl_lowp_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);

        status_t init_scratchpad(memory_tracking::registrar_t &scratchpad);

        acl_lowp_matmul_conf_t almc_ = utils::zero<decltype(almc_)>();
        acl_post_ops_t acl_post_ops;
    };

    acl_lowp_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(engine_t *engine, resource_mapper_t &mapper) const;

    status_t execute(const exec_ctx_t &ctx) const;

private:
    mutable std::mutex mtx;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_LOWP_MATMUL_HPP
