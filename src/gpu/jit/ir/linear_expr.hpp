/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_IR_LINEAR_EXPR_HPP
#define GPU_JIT_IR_LINEAR_EXPR_HPP

#include <iostream>
#include <string>
#include <unordered_map>

#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Splits nested associative binary operations into a vector of expressions.
// Example: (a + (b + c)) -> [a, b, c]
std::vector<expr_t> op_split(op_kind_t kind, const expr_t &e);

// Combines a vector of expressions into a nested binary expression.
// Example: [a, b, c] -> (a + b) + c
expr_t op_combine(op_kind_t kind, const std::vector<expr_t> &args);

// Converts an expression to linear_t expression.
expr_t to_linear(const expr_t &_e);

// Returns the max power of two divisor of an expression. The expression must
// be convertable to linear_t.
int linear_max_pow2_divisor(const expr_t &e);

// Divides an expression by a constant. The expression must be convertable to
// linear_t.
expr_t linear_div(const expr_t &e, int factor);

// Simplifies a modulus of an expression by a constant. The expression must be
// convertable to linear_t.
expr_t linear_mod(const expr_t &e, int factor);

// Returns the base and the increments of linear expression `expr` when
// incrementing `idxs[i]` by 1:
//     init = expr(idxs[i] = 0)
//     incs[i] = expr(idxs[i] + 1) - expr(idx[i]).
void split_to_linear(const expr_t &expr, const std::vector<expr_t> &idxs,
        expr_t &init, std::vector<expr_t> &incs);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
