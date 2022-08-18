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

#ifndef GPU_JIT_CONV_IR_PASS_HPP
#define GPU_JIT_CONV_IR_PASS_HPP

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/pass/alloc.hpp"
#include "gpu/jit/conv/pass/bank_conflict.hpp"
#include "gpu/jit/conv/pass/barrier.hpp"
#include "gpu/jit/conv/pass/dp4a.hpp"
#include "gpu/jit/conv/pass/dpas_atomic.hpp"
#include "gpu/jit/conv/pass/dpasw.hpp"
#include "gpu/jit/conv/pass/expr_scalarizer.hpp"
#include "gpu/jit/conv/pass/hoist.hpp"
#include "gpu/jit/conv/pass/overflow.hpp"
#include "gpu/jit/conv/pass/peephole.hpp"
#include "gpu/jit/conv/pass/send.hpp"
#include "gpu/jit/conv/pass/simplify.hpp"
#include "gpu/jit/conv/pass/slm.hpp"
#include "gpu/jit/conv/pass/strength_reduce.hpp"
#include "gpu/jit/conv/pass/unroll.hpp"

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

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
