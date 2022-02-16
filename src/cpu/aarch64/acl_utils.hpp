/*******************************************************************************
* Copyright 2021-2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_UTILS_HPP
#define CPU_AARCH64_ACL_UTILS_HPP

#include <mutex>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_engine.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_common_utils {

arm_compute::DataType get_acl_data_t(const dnnl_data_type_t dt);
arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr);
arm_compute::ActivationLayerInfo get_acl_act(const eltwise_desc_t &ed);
bool acl_act_ok(alg_kind_t eltwise_activation);
void acl_thread_bind();

// Convert a memory desc to an arm_compute::TensorInfo. Note that memory desc
// must be blocking format, plain, dense and have no zero dimensions.
status_t tensor_info(arm_compute::TensorInfo &info, const memory_desc_t &md);
status_t tensor_info(
        arm_compute::TensorInfo &info, const memory_desc_wrapper &md);

// Insert a dimension of size 1 at the index dim_i of TensorInfo
status_t insert_singleton_dimension(arm_compute::TensorInfo &ti, size_t dim_i);

// Copy the memory descs {d0, d1, d2} to {d0_perm, d1_perm, d2_perm}, but with
// the dimensions permuted so that their last logical dimensions are all dense
// (stride of 1). The function finds the highest indexed dimension with a
// stride of 1 for all descs (common). Then it permutes this to be the last
// dimension. Note that the last dimension is the one that is dense in an
// unpermuted tensor, in this case it would copy the descs unchanged. The
// function may fail to find a common dense dimension, and will return
// unimplemented.
status_t permute_common_dense_dimension_to_last(memory_desc_t *d0_permed,
        memory_desc_t *d1_permed, memory_desc_t *d2_permed,
        const memory_desc_t *d0, const memory_desc_t *d1,
        const memory_desc_t *d2);

#define MAYBE_REPORT_ACL_ERROR(msg) \
    do { \
        if (get_verbose()) printf("onednn_verbose,cpu,error,acl,%s\n", (msg)); \
    } while (0)

} // namespace acl_common_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_UTILS_HPP
