/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include "viewer.hpp"
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

expr_c ir_viewer_t::visit(binary_c v) {
    view(v);
    return std::move(v);
}

expr_c ir_viewer_t::visit(logic_c v) {
    view(v);
    return std::move(v);
}

expr_c ir_viewer_t::visit(cmp_c v) {
    view(v);
    return std::move(v);
}

void ir_viewer_t::view(binary_c v) {
    ir_visitor_t::visit(std::move(v));
}

void ir_viewer_t::view(logic_c v) {
    ir_visitor_t::visit(std::move(v));
}

void ir_viewer_t::view(cmp_c v) {
    ir_visitor_t::visit(std::move(v));
}

void ir_viewer_t::dispatch_expr_arr(const std::vector<expr> &arr) {
    for (auto &v : arr) {
        auto n = dispatch(v);
    }
}

#define DECL_VIEWER_PROXY(TYPE) \
    expr_c ir_viewer_t::visit(TYPE##_c v) { \
        view(v); \
        return std::move(v); \
    } \
    void ir_viewer_t::view(TYPE##_c v) { ir_visitor_t::visit(std::move(v)); }

#define DECL_VIEWER_PROXY_CALL_PARENT(TYPE, PARENT) \
    expr_c ir_viewer_t::visit(TYPE##_c v) { \
        view(v); \
        return std::move(v); \
    } \
    void ir_viewer_t::view(TYPE##_c v) { view(v.static_as<PARENT##_c>()); }

DECL_VIEWER_PROXY(constant)
DECL_VIEWER_PROXY(var)
DECL_VIEWER_PROXY(tensor)
DECL_VIEWER_PROXY(tensorptr)
DECL_VIEWER_PROXY(cast)
DECL_VIEWER_PROXY(intrin_call)
DECL_VIEWER_PROXY(func_addr)
DECL_VIEWER_PROXY(ssa_phi)
DECL_VIEWER_PROXY(low_level_intrin)

DECL_VIEWER_PROXY_CALL_PARENT(add, binary) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(sub, binary) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(mul, binary) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(div, binary) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(mod, binary) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_eq, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_lt, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_le, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_gt, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_ge, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(cmp_ne, cmp) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(logic_and, logic) // NOLINT
DECL_VIEWER_PROXY_CALL_PARENT(logic_or, logic) // NOLINT
DECL_VIEWER_PROXY(select)
DECL_VIEWER_PROXY(indexing)
DECL_VIEWER_PROXY(call)
DECL_VIEWER_PROXY(logic_not)

#define DECL_VIEWER_PROXY_STMT(TYPE) \
    stmt_c ir_viewer_t::visit(TYPE##_c v) { \
        view(v); \
        return std::move(v); \
    } \
    void ir_viewer_t::view(TYPE##_c v) { ir_visitor_t::visit(std::move(v)); }

DECL_VIEWER_PROXY_STMT(assign)
DECL_VIEWER_PROXY_STMT(stmts)
DECL_VIEWER_PROXY_STMT(if_else)
DECL_VIEWER_PROXY_STMT(for_loop)
DECL_VIEWER_PROXY_STMT(evaluate)
DECL_VIEWER_PROXY_STMT(returns)
DECL_VIEWER_PROXY_STMT(define)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
