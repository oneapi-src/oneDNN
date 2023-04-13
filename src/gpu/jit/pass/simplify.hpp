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

#ifndef GPU_JIT_PASS_SIMPLIFY_HPP
#define GPU_JIT_PASS_SIMPLIFY_HPP

#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Determine the maximum constant factor of an expression, returns 0 in the
// special case that the expression evaluates to 0.
int64_t get_max_const_factor(const expr_t &e, const constraint_set_t &cset);

// Simplifies expression using rewriting rules.
expr_t simplify_rewrite(const expr_t &e);

// Simplifies expression or statement. An optional constraint set is used to
// pass known equalities and inequalities which may be used for simplification.
object_t simplify(const object_t &obj, const constraint_set_t &cset = {});

// Searches for expression patterns to reduce them to the equivalent ternary
// operations.
expr_t simplify_rewrite_with_ternary(const expr_t &e, bool recursive = true);

// Moves constants to the right hand side of an expression.
// Example: (c0 + x) op c1 -> x op (c1 - c0)
expr_t simplify_cmp_move_const_to_rhs(const expr_t &e);

// Rewrites addition with mixed 64-bit/32-bit expressions to reduce 64-bit
// arithmetic. Example:
// Before: ((x.s64 + y.s32) + z.s32) [two 64-bit add]
// After:  ((y.s32 + z.s32) + x.s64) [one 32-bit add and one 64-bit add]
expr_t simplify_64_bit_add(const expr_t &e);

// Reduces left and right hand sides of an expression.
// Example: A * x < A * B -> x < B (if A > 0).
expr_t simplify_cmp_reduce_lhs_rhs(const expr_t &e);

// Propagates shuffle down the expression tree for more effective vectorization.
expr_t simplify_propagate_shuffle(const expr_t &e);

stmt_t simplify(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
