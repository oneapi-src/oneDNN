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

status_t acl_layer_normalization_fwd_t::init(engine_t *engine) {
    auto aep = pd()->anp;
    acl_obj.get()->msdNorm.configure(
            &aep.data_info, &aep.data_info, pd()->desc()->layer_norm_epsilon);
    return status::success;
}

status_t acl_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    auto aep = pd()->anp;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(aep.data_info);
    src_tensor.allocator()->import_memory(const_cast<float *>(src));
    dst_tensor.allocator()->init(aep.data_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::ITensorPack act_pack;
    act_pack.add_tensor(arm_compute::TensorType::ACL_SRC, &src_tensor);
    act_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst_tensor);
    acl_obj.get()->msdNorm.run(act_pack);

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
