/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_PASS_HOIST_HPP
#define GPU_JIT_PASS_HOIST_HPP

#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Moves invariant expressions out of loops.
stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx, int reserved_regs);

// Moves boolean mask computation from send calls to the top of the statement
// group corresponding to `label`. This is done to reduce GRF consumption and
// to reuse masks between calls. A vector boolean mask is stored as u16 type
// and converted to bool type right before the call. Transformation is limited
// to the statement group corresponding to `label`.
// If `split_by_and` is true then any ((A & B) & C) mask is split into A, B, C
// sub-masks which are initialized independently. This allows reusing those
// sub-masks for other masks.
stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and, int reserved_regs);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
