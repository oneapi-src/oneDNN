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

#ifndef GPU_JIT_PASS_PASS_HPP
#define GPU_JIT_PASS_PASS_HPP

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/pass/alloc.hpp"
#include "gpu/jit/pass/bank_conflict.hpp"
#include "gpu/jit/pass/barrier.hpp"
#include "gpu/jit/pass/cse.hpp"
#include "gpu/jit/pass/dp4a.hpp"
#include "gpu/jit/pass/dpas_atomic.hpp"
#include "gpu/jit/pass/dpasw.hpp"
#include "gpu/jit/pass/expr_scalarizer.hpp"
#include "gpu/jit/pass/hoist.hpp"
#include "gpu/jit/pass/overflow.hpp"
#include "gpu/jit/pass/peephole.hpp"
#include "gpu/jit/pass/send.hpp"
#include "gpu/jit/pass/shuffle_splitter.hpp"
#include "gpu/jit/pass/simplify.hpp"
#include "gpu/jit/pass/slm.hpp"
#include "gpu/jit/pass/strength_reduce.hpp"
#include "gpu/jit/pass/unroll.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

stmt_t inject_external_var_let(const stmt_t &_stmt, ir_context_t &ir_ctx);

// Removes redundant u16 casts inside send masks which may appear after
// previous mask hoisting.
stmt_t remove_spurious_send_mask_cast(const stmt_t &s, ir_context_t &ir_ctx);

// Splits wide GRF stores otherwise unsupported in HW.
stmt_t split_wide_stores(const stmt_t &s, ir_context_t &ir_ctx);

// Injects broadcasts for scalar if conditions. Example:
// Before:
//     if (cond) { ... }
// After (for SIMD8):
//     if (bcast8(cond)) { ... }
stmt_t fixup_if_conditions(const stmt_t &s, ir_context_t &ir_ctx);

// Rewrites mixed 64-bit/32-bit expressions to reduce 64-bit arithmetic.
// Potential overflow is ignored and must be checked/fixed by further passes.
stmt_t optimize_int64_exprs(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
