/*******************************************************************************
* Copyright 2021-2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_MATMUL_UTILS_HPP
#define CPU_AARCH64_ACL_MATMUL_UTILS_HPP

#include "arm_compute/runtime/experimental/low_level/CpuGemmAssemblyDispatch.h"
#include "arm_compute/runtime/experimental/operators/CpuActivation.h"
#include "arm_compute/runtime/experimental/operators/CpuTranspose.h"

#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
// Keys are anonymous. So deduce the type automagically.
using matmul_key_t = decltype(memory_tracking::names::key_gemm_asm_tmp_buffer);

// Map: [slot , key]
const std::map<int, matmul_key_t> matmul_keys = {
        {0, matmul_key_t::key_gemm_asm_tmp_buffer},
        {1, matmul_key_t::key_gemm_pretransposed_rhs},
        {2, matmul_key_t::key_gemm_pretranspose},
};
} // namespace

struct acl_matmul_obj_t {
    arm_compute::experimental::op::ll::CpuGemmAssemblyDispatch asm_gemm;
    arm_compute::experimental::op::CpuActivation act;
    arm_compute::experimental::op::CpuTranspose transA;
    arm_compute::experimental::op::CpuTranspose transB;
    arm_compute::experimental::op::CpuTranspose transC;
    arm_compute::experimental::MemoryRequirements aux_mem_req;
};

struct acl_matmul_conf_t {
    bool is_transA;
    bool is_transB;
    bool do_transC;
    bool do_act;
    // If this is true, the result of the matmul goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc_for_sum;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;
    arm_compute::TensorInfo src_acc_info;
    arm_compute::TensorInfo wei_acc_info;
    arm_compute::TensorInfo dst_acc_info;
    arm_compute::GEMMInfo gemm_info;
};

namespace acl_matmul_utils {

template <bool IsFixedFormat>
status_t init_conf_matmul(acl_matmul_conf_t &amp, memory_desc_t &src_md,
        memory_desc_t &wei_md, memory_desc_t &dst_md, const matmul_desc_t &md,
        const primitive_attr_t &attr);

status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const acl_matmul_conf_t &amp, const memory_desc_t &src_md,
        const memory_desc_t &weights_md, const memory_desc_t &dst_md,
        const arm_compute::experimental::MemoryRequirements &aux_mem_req);

} // namespace acl_matmul_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_MATMUL_UTILS_HPP
