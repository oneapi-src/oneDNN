/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_JIT_PASS_ALLOC_HPP
#define GPU_INTEL_JIT_PASS_ALLOC_HPP

#include "gpu/intel/jit/ir/legacy.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Lifts alloc statements out of loops.
stmt_t lift_alloc(const stmt_t &s, ir_context_t &ir_ctx, bool reuse_headers);

stmt_t optimize_alloc_let(const stmt_t &s, ir_context_t &ir_ctx);

// Returns a new statement with injected buffer allocations from `allocs`.
// - If put_innermost is false, then `stmt` is nested to all allocations
// - If put_innermost is true, then every allocation is injected as innermost
//   as possible
// - If update_existing is true, then all existing alloc statements are removed
//   and reinserted in the innermost position. This flag requires put_innermost
//   to be set to true.
stmt_t inject_alloc_stmts(const stmt_t &stmt, const std::vector<stmt_t> &allocs,
        bool put_innermost = false, bool update_existing = false);

// Similar to the previous function but allocations are taken from the buffer
// manager, allocations are injected at innermost possible scope.
stmt_t inject_alloc_stmts(const stmt_t &stmt, const buffer_manager_t &buf_mgr);

// Returns a new statement with injected let statements, `stmt` is nested to
// all let statements.
stmt_t inject_let_stmts(const stmt_t &stmt, const std::vector<stmt_t> &lets);

// Injects dangling let statements (having empty body) in the innermost
// possible scope. This allows to use declare and use variables on-the-fly, in
// the imperative manner.
//
// Example:
//     let x = 1 {}
//     let y = (x + 1) {}
//     let z = (y + 1) {}
//     store(..., z)
//
// After injection:
//     let x = 1 {
//       let y = (x + 1) }
//         let z = (y + 1) {
//           store(..., z)
//         }
//       }
//     }
stmt_t inject_dangling_let_stmts(const stmt_t &stmt);

// Injects an allocation attribute to document the buffer access patterns; this
// information is then used during nGEN lowering to better utilize the register
// space by splitting the buffer into potentially several chunks which are only
// contiguous where the operations accessing the buffer need them to be.
stmt_t inject_access_map_attribute(const stmt_t &stmt, ir_context_t &ir_ctx);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
