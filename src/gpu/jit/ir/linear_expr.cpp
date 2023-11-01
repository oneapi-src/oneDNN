/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/ir/linear_expr.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

bool should_use_linear_op(const expr_t &a) {
    if (!linear_expr_ctx_t::is_set()) return false;
    if (a.is<var_t>()) return true;
    if (a.is<const_var_t>()) return true;
    if (a.is<linear_t>()) return true;
    return false;
}

bool should_use_linear_op(const expr_t &a, const expr_t &b) {
    return should_use_linear_op(a) || should_use_linear_op(b);
}

thread_local linear_expr_ctx_t *linear_expr_ctx_t::ctx_ = nullptr;

// Updates the base and the increment of linear expression `expr` when
// incrementing `idx` by 1:
//     inc_idx = expr(idx + 1) - expr(idx).
//     inc += inc_idx
//     return expr(idx = 0)
expr_t split_to_linear_impl(
        const expr_t &expr, const expr_t &idx, expr_t &inc) {
    if (expr.is<int_imm_t>() || expr.is<const_var_t>()) {
        inc = expr_t(0);
        return expr;
    }
    if (auto *var = expr.as_ptr<var_t>()) {
        if (var != idx.impl()) {
            inc = expr_t(0);
            return expr;
        }
        inc = expr_t(1);
        return expr_t(0);
    }

    if (auto *linear = expr.as_ptr<linear_t>()) {
        for (int i = 0; i < linear->nargs(); i++) {
            if (linear->v_vec[i].impl() == idx.impl()) {
                auto u_vec = linear->u_vec;
                auto v_vec = linear->v_vec;
                u_vec.erase(u_vec.begin() + i);
                v_vec.erase(v_vec.begin() + i);
                inc = linear->u_vec[i];
                return linear_t::make(linear->c, u_vec, v_vec);
            }
        }
        inc = expr_t(0);
        return expr;
    }

    ir_error_not_expected();
    return expr;
}

void split_to_linear(const expr_t &expr, const std::vector<expr_t> &idxs,
        expr_t &init, std::vector<expr_t> &incs) {
    incs = std::vector<expr_t>(idxs.size());
    init = expr;
    for (size_t i = 0; i < idxs.size(); i++) {
        init = split_to_linear_impl(init, idxs[i], incs[i]);
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
