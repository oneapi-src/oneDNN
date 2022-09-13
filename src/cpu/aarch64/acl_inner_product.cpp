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

#include "cpu/aarch64/acl_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};

    bool with_bias = pd()->aip.with_bias;
    bool use_dst_acc = pd()->aip.use_dst_acc;

    // Retrieve primitive resource and configured Compute Library objects
    acl_ip_obj_t &acl_obj = ctx.get_resource_mapper()
                                    ->get<acl_ip_resource_t>(this)
                                    ->get_acl_obj();

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    acl_obj.src_tensor.allocator()->import_memory(const_cast<void *>(src_base));

    auto wei_base = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    acl_obj.wei_tensor.allocator()->import_memory(const_cast<void *>(wei_base));

    if (use_dst_acc) {
        // Put the result in a new tensor, it will be accumalated to the dst
        // during the post ops
        acl_obj.dst_tensor.allocator()->allocate();
    } else {
        auto dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        acl_obj.dst_tensor.allocator()->import_memory(dst_base);
    }

    if (with_bias) {
        auto bia_base = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        acl_obj.bia_tensor.allocator()->import_memory(
                const_cast<void *>(bia_base));
    }

    acl_obj.fc.run();

    acl_obj.src_tensor.allocator()->free();
    acl_obj.wei_tensor.allocator()->free();
    if (with_bias) { acl_obj.bia_tensor.allocator()->free(); }

    void *dst = acl_obj.dst_tensor.buffer();
    pd()->post_ops.execute(ctx, dst);

    acl_obj.dst_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
