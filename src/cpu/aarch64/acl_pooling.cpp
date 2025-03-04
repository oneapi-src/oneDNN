/*******************************************************************************
* Copyright 2022-2023, 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_pooling_fwd_t::init(engine_t *engine) {
    auto asp = pd()->asp_;

    auto op = std::make_unique<arm_compute::experimental::op::CpuPooling>();

    pooling_op_ = std::move(op);

    // Configure pooling operation when workspace tensor is used, mem allocation happens
    if (asp.use_ws) {
        pooling_op_->configure(
                &asp.src_info, &asp.dst_info, asp.pool_info, &asp.ws_info);
    }
    // Configure pooling operation when workspace tensor is not used, mem allocation happens
    else {
        pooling_op_->configure(
                &asp.src_info, &asp.dst_info, asp.pool_info, nullptr);
    }

    return status::success;
}

status_t acl_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    void *ws_base;

    auto asp = pd()->asp_;

    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(asp.src_info);
    src_tensor.allocator()->import_memory(const_cast<void *>(src));
    dst_tensor.allocator()->init(asp.dst_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::Tensor scratch_tensor;
    void *scratchpad_base = ctx.get_scratchpad_grantor().get<void>(
            memory_tracking::names::key_pool_reduction);
    scratch_tensor.allocator()->init(arm_compute::TensorInfo(
            asp.dst_info.tensor_shape(), 1, arm_compute::DataType::F32));
    scratch_tensor.allocator()->import_memory(scratchpad_base);

    arm_compute::Tensor ws_tensor;

    if (asp.use_ws) {
        ws_base = CTX_OUT_MEM(void *, DNNL_ARG_WORKSPACE);
        ws_tensor.allocator()->init(asp.ws_info);
        ws_tensor.allocator()->import_memory(ws_base);
    }
    //for scratchpad based tensor
    arm_compute::ITensorPack run_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src_tensor},
            {arm_compute::TensorType::ACL_DST_0, &dst_tensor},
            {arm_compute::TensorType::ACL_INT_0, &scratch_tensor}};

    if (asp.use_ws) {
        run_pack.add_tensor(arm_compute::TensorType::ACL_DST_1, &ws_tensor);
    }
    pooling_op_->run(run_pack);

    return status;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
