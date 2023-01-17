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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/pass/printer.hpp>
#include <compiler/jit/xbyak/ir/transform/call_transform.hpp>
#include <compiler/jit/xbyak/ir/transform/register_allocation.hpp>

#include "xbyak_expr.hpp"
#include "xbyak_printer.hpp"

namespace sc {
namespace sc_xbyak {

ostream &xbyak_printer_t::print_expr_info(ostream &os, const expr &arg) {
    os << "{";
    if (arg.isa<tensor>()) {
        auto a = arg.static_as<tensor>();
        ir_printer_t p {os};
        a->to_string_full(p);
    } else {
        os << arg << ": " << arg->dtype_;
    }
    auto &virt_reg = GET_VIRTUAL_REG(arg);
    os << ": " << virt_reg;
    if (virt_reg.allocated()) {
        os << "~%" << virtual_slots_map_->get_reg_name(virt_reg.index_);
    }
    os << "}";
    return os;
}

ostream &xbyak_printer_t::print_expr_vec(
        ostream &os, const std::vector<expr_c> &args) {
    for (auto &arg : args) {
        print_expr_info(os, arg.remove_const()) << ", ";
    }
    os << "\n";
    return os;
}

xbyak_printer_t::xbyak_printer_t(
        const_ir_module_ptr &ir_mod, x86_64::target_profile_t &profile)
    : profile_(profile) {
    // Slot map for virtual slot and physical reg mapping
    virtual_slots_map_ = std::make_shared<virtual_slots_map_t>(profile_);
    // Print module vars
    for (auto &f : ir_mod->get_module_vars()) {
        ss_ << f << '\n';
    }
    // Print functions
    for (auto &f : ir_mod->get_contents()) {
        dispatch(f);
        ss_ << '\n';
    }
}

func_c xbyak_printer_t::dispatch(func_c e) {
    indent_ = 0;

    if (e->name_.find("_should_inline_") != std::string::npos) { return e; }

    ss_ << "func " << e->name_ << '(';
    if (!e->params_.empty()) {
        for (unsigned i = 0; i < e->params_.size() - 1; i++) {
            print_expr_info(ss_, e->params_.at(i)) << ", ";
        }
        print_expr_info(ss_, e->params_.back());
    }
    ss_ << "): " << e->ret_type_ << '\n';

    func_t func = std::const_pointer_cast<func_base>(e);
    assert(func->attr_ && func->attr_->has_key(attr_keys::global_spilled));
    auto &spilled
            = func->attr_->get<std::vector<expr_c>>(attr_keys::global_spilled);
    print_padding_indents();
    print_expr_vec(ss_ << "--GLOBAL_SPILLED: ", spilled);

    dispatch(e->body_);

    return e;
}

void xbyak_printer_t::view(stmts_c v) {
    print_index_indents(GET_STMT_INIT_INDEX(v));
    ss_ << "{ \n";
    if (TRANSFORMED_CALL(v)) {
        assert(v->attr_);
        // print func call
        print_padding_indents();
        ss_ << "--FUNC_CALL\n";
    }

    indent_++;
    for (auto &s : v->seq_) {
        dispatch(s);
    }
    indent_--;

    print_index_indents(GET_STMT_INDEX(v));
    ss_ << "}\n";
}

void xbyak_printer_t::view(evaluate_c v) {
    print_index_indents(GET_STMT_INDEX(v));
    ss_ << "evaluate{" << v->value_ << "}\n";
}

void xbyak_printer_t::view(returns_c v) {
    print_index_indents(GET_STMT_INDEX(v));
    ss_ << "return ";
    if (v->value_.defined()) { ss_ << v->value_; }
    ss_ << "\n";
}

void xbyak_printer_t::view(assign_c v) {
    print_index_indents(GET_STMT_INDEX(v));
    ss_ << v->var_ << " = " << v->value_ << "\n";
}

void xbyak_printer_t::view(define_c v) {
    if (v->init_.defined()) {
        print_padding_indents();
        print_expr_info(ss_, v->var_) << "\n";
        print_index_indents(GET_STMT_INDEX(v));
        ss_ << v->var_ << " = " << v->init_ << "\n";
    } else {
        print_index_indents(GET_STMT_INDEX(v));
        print_expr_info(ss_, v->var_) << "\n";
    }
}

void xbyak_printer_t::view(if_else_c v) {
    print_index_indents(GET_STMT_INIT_INDEX(v));
    ss_ << "if (" << v->condition_ << ") \n";

    dispatch(v->then_case_);

    if (v->else_case_.defined()) {
        print_padding_indents();
        ss_ << "else \n";

        dispatch(v->else_case_);
    }
    print_index_indents(GET_STMT_INDEX(v));
    ss_ << "END: if (" << v->condition_ << ") \n";
}

void xbyak_printer_t::view(for_loop_c v) {
    print_index_indents(GET_STMT_INIT_INDEX(v));
    ss_ << "for " << v->var_ << " in (" << v->iter_begin_ << ", "
        << v->iter_end_ << ", " << v->step_ << ") -> ";
    print_expr_info(ss_, v->var_);
    ss_ << "\n";

    if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_begin)) {
        auto load_begin = v->attr_->get<stmt>(attr_keys::load_loop_begin);
        print_padding_indents();
        ss_ << "== load_begin: " << load_begin << "  ";
        print_expr_info(ss_, v->iter_begin_) << "\n";
    }
    if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_end)) {
        auto load_end = v->attr_->get<stmt>(attr_keys::load_loop_end);
        print_padding_indents();
        ss_ << "== load_end: " << load_end << "  ";
        print_expr_info(ss_, v->iter_end_) << "\n";
    }
    if (v->attr_ && v->attr_->has_key(attr_keys::load_loop_step)) {
        auto load_step = v->attr_->get<stmt>(attr_keys::load_loop_step);
        print_padding_indents();
        ss_ << "== load_step: " << load_step << "  ";
        print_expr_info(ss_, v->step_) << "\n";
    }

    dispatch(v->body_);

    print_index_indents(GET_STMT_INDEX(v));
    ss_ << "END: for (" << v->var_ << ") \n";
}

void xbyak_printer_t::print_index_indents(int64_t index) {
    ss_ << std::left << std::setw(index_width_) << index << std::string(1, ' ');
    ss_ << std::string(indent_ * 2, ' ');
}

void xbyak_printer_t::print_padding_indents() {
    ss_ << std::string((indent_ * 2) + index_width_ + 1, ' ');
}

std::stringstream &xbyak_printer_t::get_stream() {
    return ss_;
}

} // namespace sc_xbyak
} // namespace sc
