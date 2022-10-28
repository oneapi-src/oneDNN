/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include "sc_stmt.hpp"
#include "builder.hpp"
#include "ir_comparer.hpp"
#include "visitable.hpp"
#include <compiler/ir/pass/printer.hpp>
#include <util/any_map.hpp>
namespace sc {

stmt_base_t::~stmt_base_t() = default;
stmt_base_t::stmt_base_t(sc_stmt_type type) : node_type_(type) {}

ostream &operator<<(ostream &os, const stmt_c &s) {
    return os << s.get();
}

ostream &operator<<(ostream &os, const stmt_base_t *s) {
    s->to_string(os, 0);
    return os;
}

std::ostream &operator<<(std::ostream &os, for_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case sc::for_type::X: os << "for_type::" #X; break;
        HANDLE_CASE(NORMAL)
        HANDLE_CASE(PARALLEL)
#undef HANDLE_CASE
        default: os << "(unrecognized for_type value)"; break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, sc_stmt_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case sc::sc_stmt_type::X: os << "sc_stmt_type::" #X; break;

        HANDLE_CASE(undef)
        HANDLE_CASE(assign)
        HANDLE_CASE(stmts)
        HANDLE_CASE(if_else)
        HANDLE_CASE(evaluate)
        HANDLE_CASE(for_loop)
        HANDLE_CASE(returns)
        HANDLE_CASE(define)
#undef HANDLE_CASE
        default: os << "(unrecognized sc_stmt_type value)"; break;
    }
    return os;
}

stmt assign_node_t::remake() const {
    return copy_attr(*this, make_stmt<assign_node_t>(var_, value_));
}

void stmt_base_t::to_string(ostream &os, int indent) const {
    ir_printer_t p {os};
    p.indents_ = indent;
    p.dispatch(node_ptr_from_this());
}

bool stmt_base_t::equals(stmt_c other) const {
    ir_comparer cmper;
    return this->equals(std::move(other), cmper);
}

#define CAST_OR_RETURN(v) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    stmt_base_t>; \
    if (!(v).isa<self>()) { return false; } \
    auto other = (v).static_as<self>();

bool assign_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return var_->equals(other->var_, ctx) && value_->equals(other->value_, ctx);
}

stmt stmts_node_t::remake() const {
    std::vector<stmt> seq = seq_;
    return copy_attr(*this, make_stmt<stmts_node_t>(std::move(seq)));
}

bool stmts_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    if (seq_.size() != other->seq_.size()) {
        return ctx.set_result(node_ptr_from_this(), v, false);
    }
    for (size_t i = 0; i < seq_.size(); i++) {
        if (!seq_.at(i)->equals(other->seq_.at(i), ctx)) { return false; }
    }
    return true;
}

stmt if_else_node_t::remake() const {
    return copy_attr(*this,
            make_stmt<if_else_node_t>(condition_, then_case_, else_case_));
}

bool if_else_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return condition_->equals(other->condition_, ctx)
            && then_case_->equals(other->then_case_, ctx)
            && ctx.check_equals_may_null(else_case_, other->else_case_);
}

stmt evaluate_node_t::remake() const {
    return copy_attr(*this, make_stmt<evaluate_node_t>(value_));
}

bool evaluate_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return value_->equals(other->value_, ctx);
}

stmt returns_node_t::remake() const {
    return copy_attr(*this, make_stmt<returns_node_t>(value_));
}

bool returns_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return ctx.check_equals_may_null(value_, other->value_);
}

stmt define_node_t::remake() const {
    return copy_attr(*this, make_stmt<define_node_t>(var_, linkage_, init_));
}

bool define_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return var_->equals(other->var_, ctx) && linkage_ == other->linkage_
            && ctx.check_equals_may_null(init_, other->init_);
}

stmt for_loop_node_t::remake() const {
    return copy_attr(*this,
            make_stmt<for_loop_node_t>(var_, iter_begin_, iter_end_, step_,
                    body_, incremental_, kind_, num_threads_));
}

#define CAST_OR_RETURN(v) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    stmt_base_t>; \
    if (!(v).isa<self>()) { return false; } \
    auto other = (v).static_as<self>();

bool for_loop_node_t::equals(stmt_c v, ir_comparer &ctx) const {
    CAST_OR_RETURN(v);
    return ctx.set_result(node_ptr_from_this(), v,
                   incremental_ == other->incremental_ && kind_ == other->kind_
                           && num_threads_ == other->num_threads_)
            && var_->equals(other->var_, ctx)
            && iter_begin_->equals(other->iter_begin_, ctx)
            && iter_end_->equals(other->iter_end_, ctx)
            && step_->equals(other->step_, ctx)
            && body_->equals(other->body_, ctx);
}

} // namespace sc
