/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_binary_t::execute_forward(const exec_ctx_t &ctx, const void *src0,
        const void *src1, void *dst) const {

    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};

    // Retrieve primitive resource and configured Compute Library objects
    acl_binary_obj_t &acl_obj = ctx.get_resource_mapper()
                                        ->get<acl_binary_resource_t>(this)
                                        ->get_acl_obj();

    acl_obj.src0_tensor.allocator()->import_memory(const_cast<void *>(src0));
    acl_obj.src1_tensor.allocator()->import_memory(const_cast<void *>(src1));
    acl_obj.dst_tensor.allocator()->import_memory(dst);

    acl_obj.binary_op->run();

    acl_obj.src0_tensor.allocator()->free();
    acl_obj.src1_tensor.allocator()->free();
    acl_obj.dst_tensor.allocator()->free();

    return status::success;
}

status_t acl_binary_t::execute_forward(const exec_ctx_t &ctx) const {

    auto src0 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_0);
    auto src1 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    return execute_forward(ctx, src0, src1, dst);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
