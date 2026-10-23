/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_COMMON_ZEN_POST_OPS_FALLBACK_HPP
#define CPU_X64_ZEN64_COMMON_ZEN_POST_OPS_FALLBACK_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_desc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {

// Executes the post-ops a Zen kernel cannot fuse, i.e. entries
// [start_index, len) of the user chain, as standalone oneDNN binary / eltwise
// / prelu primitives over the kernel output (same scheme as the ACL fallback
// in cpu/aarch64/post_ops_fallback).
//
// Without an accumulator, entries run in place on dst, so the chain must not
// read the original destination (no sum or in-place binary). With one
// (uses_acc()), the kernel writes f32 to a separate buffer instead and every
// entry runs in place on that buffer: a sum adds scale * dst and an in-place
// binary multiplies by dst, both reading the untouched destination. A final
// reorder then rounds the result into dst once, as a fused chain would.
// PReLU does not support in-place execution, so it reads a scratchpad copy
// of the buffer and writes the buffer.
//
// The pd owns an instance after init() / init_scratchpad(); the primitive
// keeps a copy and calls init_primitives() once.
struct post_ops_fallback_t {
    // Binary src1/src2 descriptors in post_ops must already be resolved
    // (post_ops_t::set_default_formats). acc_md, if given, is the f32
    // accumulator descriptor and must have the dst_md dims.
    status_t init(const engine_t *engine, const post_ops_t &post_ops,
            const memory_desc_t &dst_md, int start_index,
            const memory_desc_t *acc_md = nullptr);
    status_t init_primitives(engine_t *engine);
    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

    bool empty() const { return pds_.empty(); }
    bool uses_acc() const { return uses_acc_; }

    // kernel_dst is the kernel output: the DNNL_ARG_DST buffer, or the f32
    // accumulator when uses_acc().
    status_t execute(const exec_ctx_t &ctx, void *kernel_dst) const;

private:
    status_t execute_binary(const exec_ctx_t &ctx, int i, const void *src0,
            const void *src1, const void *src2, void *dst,
            const float *src1_scale) const;
    status_t execute_eltwise(const exec_ctx_t &ctx, int i, void *data) const;
    status_t execute_prelu(const exec_ctx_t &ctx, int i, const void *weights,
            void *data) const;
    status_t execute_reorder(
            const exec_ctx_t &ctx, int i, const void *src, void *dst) const;
    status_t execute_nested(
            const exec_ctx_t &ctx, int i, exec_args_t &&args) const;

    post_ops_t post_ops_;
    int start_index_ = 0;
    bool uses_acc_ = false;
    // Bytes of the kernel output buffer from its base pointer through its
    // last addressed element, when the chain has a prelu entry.
    size_t prelu_copy_size_ = 0;
    // One entry per post-op in [start_index, len), then the accumulator
    // reorder when uses_acc().
    std::vector<std::shared_ptr<primitive_desc_t>> pds_;
    std::vector<std::shared_ptr<primitive_t>> primitives_;
};

} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
