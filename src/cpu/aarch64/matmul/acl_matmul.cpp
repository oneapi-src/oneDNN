/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/acl_matmul.hpp"
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace data_type;

template <bool IsFixedFormat>
status_t acl_matmul_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);

    bool is_transA = pd()->amp_.is_transA;
    bool is_transB = pd()->amp_.is_transB;
    bool do_transC = pd()->amp_.do_transC;
    bool use_dst_acc_for_sum = pd()->amp_.use_dst_acc_for_sum;

    std::lock_guard<std::mutex> _lock {this->mtx};
    auto *acl_resource = ctx.get_resource_mapper()->get<acl_resource_t>(this);
    acl_matmul_obj_t &acl_obj = acl_resource->get_acl_obj();

    const auto scratchpad = ctx.get_scratchpad_grantor();

    // Run transpose kernel
    if (is_transA && !is_transB) {
        auto transA_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_src_trans);
        acl_obj.src_tensor.allocator()->import_memory(transA_scratch);
        acl_obj.src_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
        acl_obj.transA.run();
        acl_obj.wei_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
    } else if (is_transB && !is_transA) {
        auto transB_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
        acl_obj.wei_tensor.allocator()->import_memory(transB_scratch);
        acl_obj.wei_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
        acl_obj.transB.run();
        acl_obj.src_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
    } else if (is_transA && is_transB && !do_transC) {
        auto transA_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_src_trans);
        auto transB_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
        acl_obj.src_tensor.allocator()->import_memory(transA_scratch);
        acl_obj.src_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
        acl_obj.wei_tensor.allocator()->import_memory(transB_scratch);
        acl_obj.wei_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
        acl_obj.transA.run();
        acl_obj.transB.run();
    } else {
        acl_obj.src_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
        acl_obj.wei_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
        if (do_transC) {
            auto transC_scratch = scratchpad.get<void>(
                    memory_tracking::names::key_matmul_dst_trans);
            acl_obj.dst_acc_tensor.allocator()->import_memory(transC_scratch);
        }
    }

    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = use_dst_acc_for_sum ? scratchpad.get<void>(
                            memory_tracking::names::key_matmul_dst_in_acc_dt)
                                        : CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    acl_obj.dst_tensor.allocator()->import_memory(dst_base);

    acl_obj.gemm.run();

    if (do_transC) { acl_obj.transC.run(); }

    acl_obj.src_tensor.allocator()->free();
    acl_obj.wei_tensor.allocator()->free();
    if (is_transA) acl_obj.src_acc_tensor.allocator()->free();
    if (is_transB) acl_obj.wei_acc_tensor.allocator()->free();

    void *dst = acl_obj.dst_tensor.buffer();
    pd()->acl_post_ops.execute(ctx, dst);

    acl_obj.dst_tensor.allocator()->free();
    if (do_transC) acl_obj.dst_acc_tensor.allocator()->free();

    return status;
}

template status_t acl_matmul_t::execute_forward<true>(
        const exec_ctx_t &ctx) const;
template status_t acl_matmul_t::execute_forward<false>(
        const exec_ctx_t &ctx) const;

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
