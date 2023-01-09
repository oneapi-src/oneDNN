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

#include <utility>
#include <unordered_map>

#include <compiler/ir/builder.hpp>
#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <util/any_map.hpp>

#include "constant_propagation.hpp"

namespace sc {
namespace sc_xbyak {

class constant_propagation_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    std::unordered_map<expr_c, expr> const_map;

    expr_c dispatch(expr_c v) override {
        if (const_map.find(v) == const_map.end()) {
            return xbyak_visitor_t::dispatch(std::move(v));
        } else {
            return const_map[v];
        }
    }

    stmt_c visit(define_c v) override {
        if (v->init_.defined()) {
            if (mark_const_expr(v->var_, v->init_)) { return std::move(v); }
        }
        return xbyak_visitor_t::visit(std::move(v));
    }

    stmt_c visit(assign_c v) override {
        if (mark_const_expr(v->var_, v->value_)) { return std::move(v); }
        return xbyak_visitor_t::visit(std::move(v));
    }

    bool mark_const_expr(const expr &lhs, const expr &rhs) {
        if (const_map.find(lhs) != const_map.end()) { const_map.erase(lhs); }
        if (lhs.isa<var>() && rhs.isa<constant>()) {
            const_map[lhs] = rhs;
            return true;
        }
        return false;
    }
};

func_c constant_propagation_t::operator()(func_c v) {
    constant_propagation_impl_t constant_propagation;

    return constant_propagation.dispatch(std::move(v));
}

} // namespace sc_xbyak
} // namespace sc
