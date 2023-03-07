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
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>

#include "xbyak_visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

//-----------------
// xbyak_visitor_t
//-----------------

stmt_c xbyak_visitor_t::visit(stmts_c v) {
    const stmt_base_t *previous_scope_ = current_scope_;
    current_scope_ = v.get();
    auto vv = ir_visitor_t::visit(std::move(v));
    current_scope_ = previous_scope_;
    return vv;
}

stmt_c xbyak_visitor_t::visit(for_loop_c v) {
    loop_depth_++;
    auto vv = ir_visitor_t::visit(std::move(v));
    loop_depth_--;
    return vv;
}

expr_c xbyak_visitor_t::visit(tensor_c v) {
    // do not dispatch into tensor
    return v;
}

expr_c xbyak_visitor_t::visit(low_level_intrin_c v) {
    switch (v->kind_) {
        case low_level_intrin_kind::x86_general: {
            return ir_visitor_t::visit(std::move(v));
        } break;
        case low_level_intrin_kind::x86_xbyak: {
            auto vv = v.checked_as<xbyak_intrin_c>();
            return visit(std::move(vv));
        } break;
        default: {
            assert(0 && "Invalid intrin kind.");
            return ir_visitor_t::visit(std::move(v));
        }
    }
}

expr_c xbyak_visitor_t::visit(xbyak_intrin_c v) {
    std::vector<expr> new_arr;
    bool changed = dispatch_expr_vector(v->args_, new_arr);
    if (changed) {
        return make_xbyak_intrin(v->dtype_, new_arr,
                static_cast<xbyak_intrin_type>(v->type_), v->isa_,
                v->modifier_);
    } else {
        return std::move(v);
    }
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
