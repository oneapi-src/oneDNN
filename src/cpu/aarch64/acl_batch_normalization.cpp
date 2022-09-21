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

#include "cpu/aarch64/acl_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};

    // Retrieve primitive resource and configured Compute Library objects
    acl_batch_normalization_obj_t &acl_obj
            = ctx.get_resource_mapper()
                      ->get<acl_batch_normalization_resource_t>(this)
                      ->get_acl_obj();

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    acl_obj.src_tensor.allocator()->import_memory(const_cast<float *>(src));

    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    acl_obj.dst_tensor.allocator()->import_memory(dst);

    auto mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
    acl_obj.mean_tensor.allocator()->import_memory(const_cast<float *>(mean));

    auto variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    acl_obj.var_tensor.allocator()->import_memory(
            const_cast<float *>(variance));

    if (pd()->use_scale()) {
        auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
        acl_obj.gamma_tensor.allocator()->import_memory(
                const_cast<float *>(scale));
    }
    if (pd()->use_shift()) {
        auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);
        acl_obj.beta_tensor.allocator()->import_memory(
                const_cast<float *>(shift));
    }

    acl_obj.bnorm.run();

    acl_obj.src_tensor.allocator()->free();
    acl_obj.gamma_tensor.allocator()->free();
    acl_obj.beta_tensor.allocator()->free();
    acl_obj.mean_tensor.allocator()->free();
    acl_obj.var_tensor.allocator()->free();

    pd()->post_ops.execute(ctx, acl_obj.dst_tensor.buffer());

    acl_obj.dst_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
