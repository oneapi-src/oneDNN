/*******************************************************************************
* Copyright 2023, 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_layer_normalization_fwd_t::execute(const exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

status_t acl_layer_normalization_fwd_t::init(engine_t *engine) {
    auto aep = pd()->anp;
    acl_obj.get()->configure(
            &aep.data_info, &aep.data_info, desc()->layer_norm_epsilon);
    return status::success;
}

status_t acl_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);

    arm_compute::Tensor data_tensor;

    auto const acp = pd()->anp;

    data_tensor.allocator()->init(acp.data_info);

    data_tensor.allocator()->import_memory(const_cast<float *>(src));

    arm_compute::Tensor data_tensor;

    auto const acp = pd()->anp;

    data_tensor.allocator()->init(acp.data_info);

    data_tensor.allocator()->import_memory(const_cast<float *>(src));

    arm_compute::ITensorPack pack;
    pack.add_tensor(arm_compute::TensorType::ACL_SRC_0, &data_tensor);

    acl_obj.get()->msdNorm.run(pack);

    data_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl