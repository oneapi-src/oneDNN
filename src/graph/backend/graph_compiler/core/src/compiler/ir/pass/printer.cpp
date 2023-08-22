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
#include "printer.hpp"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include "../viewer.hpp"
#include "util/fp16.hpp"
#include <compiler/ir/intrinsics.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>
#include <util/bf16.hpp>
#include <util/pos_track_stream.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void ir_printer_t::view(constant_c v) {
    if (v->is_vector()) { os_ << '('; }
    for (unsigned i = 0; i < v->value_.size(); i++) {
        switch (v->dtype_.type_code_) {
            case sc_data_etype::BF16: {
                os_ << bf16_t(v->value_[i].f32).storage_ << "UL";
            } break;
            case sc_data_etype::F16: {
                os_ << "(_Float16)"
                    << std::min(std::max(v->value_[i].f32, -65504.f), 65504.f);
            } break;
            case sc_data_etype::F32: {
                if (v->value_[i].f32 - static_cast<int>(v->value_[i].f32)
                        == 0) {
                    os_ << v->value_[i].f32 << ".f";
                } else {
                    if (std::isnan(v->value_[i].f32)) {
                        os_ << "NAN";
                    } else if (std::isinf(v->value_[i].f32)) {
                        if (v->value_[i].s64 & 0x80000000) {
                            os_ << "-INFINITY";
                        } else {
                            os_ << "INFINITY";
                        }
                    } else {
                        os_.precision(std::numeric_limits<float>::max_digits10);
                        os_ << v->value_[i].f32;
                    }
                }
                break;
            }
            case sc_data_etype::S8:
            case sc_data_etype::S32: os_ << v->value_[i].s64; break;
            case sc_data_etype::U8:
            case sc_data_etype::U16:
            case sc_data_etype::U32:
            case sc_data_etype::INDEX: os_ << v->value_[i].u64 << "UL"; break;
            case sc_data_etype::BOOLEAN:
                os_ << (v->value_[0].u64 ? "true" : "false");
                break;
            case sc_data_etype::POINTER:
                os_ << "((void*)" << v->value_[i].u64 << ')';
                break;
            default:
                if (v->dtype_.is_pointer()) {
                    os_ << "((" << v->dtype_ << ')' << v->value_[i].u64 << ')';
                } else {
                    os_ << "((unknown)" << v->value_[i].u64 << ')';
                }
        }
        if (i != v->value_.size() - 1) { os_ << ',' << ' '; }
    }
    if (v->is_vector()) { os_ << ')'; }
}

void ir_printer_t::view(cast_c v) {
    os_ << v->dtype_ << '(';
    do_dispatch(v->in_) << ')';
}
void ir_printer_t::view(var_c v) {
    os_ << v->name_;
    if (utils::compiler_configs_t::get().printer_print_address_) {
        os_ << '_' << (void *)v.get();
    }
}

#define GEN_BINARY(CLASS, OP) \
    void ir_printer_t::view(CLASS##_c v) { \
        os_ << '('; \
        do_dispatch(v->l_) << (OP); \
        do_dispatch(v->r_) << ')'; \
    }

GEN_BINARY(add, " + ")
GEN_BINARY(sub, " - ")
GEN_BINARY(mul, " * ")
GEN_BINARY(div, " / ")
GEN_BINARY(mod, " % ")
GEN_BINARY(cmp_eq, " == ")
GEN_BINARY(cmp_lt, " < ")
GEN_BINARY(cmp_le, " <= ")
GEN_BINARY(cmp_gt, " > ")
GEN_BINARY(cmp_ge, " >= ")
GEN_BINARY(cmp_ne, " != ")
GEN_BINARY(logic_and, " && ")
GEN_BINARY(logic_or, " || ")

void ir_printer_t::view(logic_not_c v) {
    os_ << "!(";
    do_dispatch(v->in_) << ')';
}

void ir_printer_t::view(select_c v) {
    os_ << "(";
    do_dispatch(v->cond_) << "?";
    do_dispatch(v->l_) << ":";
    do_dispatch(v->r_) << ")";
}

void ir_printer_t::view(indexing_c v) {
    do_dispatch(v->ptr_) << '[';
    assert(!v->idx_.empty());
    for (size_t i = 0; i < v->idx_.size() - 1; i++) {
        do_dispatch(v->idx_.at(i)) << ", ";
    }
    do_dispatch(v->idx_.back());
    if (v->dtype_.lanes_ > 1) { os_ << " @ " << v->dtype_.lanes_; }
    if (v->mask_.defined()) {
        os_ << " M= ";
        do_dispatch(v->mask_);
    }
    os_ << ']';
}

void ir_printer_t::view(call_c v) {
    if (auto func = std::dynamic_pointer_cast<func_base>(v->func_)) {
        os_ << func->name_;
    } else {
        auto theexpr = std::dynamic_pointer_cast<expr_base>(v->func_);
        assert(theexpr);
        do_dispatch(expr(theexpr));
    }
    os_ << '(';
    if (!v->args_.empty()) {
        for (unsigned i = 0; i < v->args_.size() - 1; i++) {
            do_dispatch(v->args_.at(i)) << ", ";
        }
        do_dispatch(v->args_.back());
    }
    os_ << ')';
    if (!v->para_attr_.empty()) {
        os_ << "@parallel(";
        for (auto &pv : v->para_attr_) {
            os_ << '[';
            do_dispatch(pv.begin_) << ", ";
            do_dispatch(pv.end_) << ", ";
            do_dispatch(pv.step_) << "], ";
        }
        os_ << ')';
    }
}

void ir_printer_t::view(tensor_c v) {
    os_ << v->name_;
    if (utils::compiler_configs_t::get().printer_print_address_) {
        os_ << '_' << (void *)v.get();
    }
}

void ir_printer_t::view(tensorptr_c v) {
    os_ << '&';
    do_dispatch(v->base_);
}

void ir_printer_t::view(intrin_call_c v) {
    auto &h = get_intrinsic_handler(v->type_);
    os_ << h.name_ << '(';
    if (!v->args_.empty()) {
        for (unsigned i = 0; i < v->args_.size() - 1; i++) {
            do_dispatch(v->args_.at(i)) << ", ";
        }
        do_dispatch(v->args_.back());
    }
    os_ << ')';
}

void ir_printer_t::view(func_addr_c v) {
    os_ << '&' << v->func_->name_;
}

void ir_printer_t::view(ssa_phi_c v) {
    os_ << "phi(";
    if (!v->values_.empty()) {
        for (unsigned i = 0; i < v->values_.size() - 1; i++) {
            do_dispatch(v->values_[i]) << ", ";
        }
        do_dispatch(v->values_.back());
    }
    if (v->is_loop_phi_) { os_ << " loop"; }
    os_ << ')';
}

void ir_printer_t::view(low_level_intrin_c v) {
    switch (v->kind_) {
        case low_level_intrin_kind::x86_xbyak: {
            v.checked_as<xbyak::xbyak_intrin_c>()->to_string(os_);
            return;
        } break;
        default: {
            assert(0 && "Not supported.");
        } break;
    }
    os_ << '(';
    if (!v->args_.empty()) {
        for (unsigned i = 0; i < v->args_.size() - 1; i++) {
            do_dispatch(v->args_.at(i)) << ", ";
        }
        do_dispatch(v->args_.back());
    }
    os_ << ')';
}

void ir_printer_t::view(assign_c v) {
    do_dispatch(v->var_) << " = ";
    do_dispatch(v->value_);
}

void ir_printer_t::view(stmts_c v) {
    os_ << "{\n";
    indents_++;
    for (auto &s : v->seq_) {
        print_indents(os_, indents_);
        do_dispatch(s);
        os_ << '\n';
    }
    indents_--;
    print_indents(os_, indents_);
    os_ << "}";
}

void ir_printer_t::view(if_else_c v) {
    os_ << "if (";
    do_dispatch(v->condition_) << ") ";
    do_dispatch(v->then_case_);
    if (v->else_case_.defined()) {
        os_ << " else ";
        do_dispatch(v->else_case_);
    }
}

void ir_printer_t::view(evaluate_c v) {
    os_ << "evaluate{";
    do_dispatch(v->value_) << '}';
}

void ir_printer_t::view(returns_c v) {
    os_ << "return ";
    if (v->value_.defined()) { do_dispatch(v->value_); }
}

void ir_printer_t::view(define_c v) {
    if (v->attr_) {
        if (auto comments
                = v->attr_->get_or_null<std::vector<std::string>>("comments")) {
            for (auto &str : *comments) {
                os_ << "// " << str << "\n";
                print_indents(os_, indents_);
            }
        }
    }
    auto va = v->var_.as<var>();
    switch (v->linkage_) {
        case linkage::local: break;
        case linkage::static_local: os_ << "static "; break;
        case linkage::private_global: os_ << "private "; break;
        case linkage::public_global: os_ << "public "; break;
        default: break;
    }
    if (va.defined()) {
        os_ << "var ";
        do_dispatch(v->var_) << ": " << va->dtype_;
        if (v->init_.defined()) {
            os_ << " = ";
            do_dispatch(v->init_);
        }
        return;
    }
    auto t = v->var_.as<tensor>();
    if (t.defined()) {
        os_ << "tensor ";
        t->to_string_full(*this);
        if (t->init_value_) {
            if (t->init_value_ != tensor_node::get_zero_tensor_initializer()) {
                os_ << " = [addr=" << t->init_value_->data_
                    << ", size=" << t->init_value_->size_ << ']';
            }
        }
        if (v->init_.defined()) {
            os_ << " = ";
            do_dispatch(v->init_);
        }
        return;
    } else {
        os_ << "(Bad var type)";
    }
}

void ir_printer_t::view(for_loop_c v) {
    const char *type;
    if (v->kind_ == for_type::PARALLEL) {
        type = "parallel ";
    } else {
        type = "";
    }
    os_ << "for ";
    do_dispatch(v->var_) << " in (";
    do_dispatch(v->iter_begin_) << ", ";
    do_dispatch(v->iter_end_) << ", ";
    do_dispatch(v->step_) << ") " << type;
    if (v->num_threads_ > 0) { os_ << '(' << v->num_threads_ << ')'; }
    do_dispatch(v->body_);
}

static ostream &print_single_arg(ir_printer_t &p, const expr &arg) {
    if (arg.isa<tensor>()) {
        auto a = arg.static_as<tensor>();
        a->to_string_full(p);
    } else {
        p.os_ << arg << ": " << arg->dtype_;
    }
    return p.os_;
}

void print_func_comments(const func_c &f, std::ostream &os) {
    if (f->attr_) {
        if (auto comments
                = f->attr_->get_or_null<std::vector<std::string>>("comments")) {
            os << "/**\n";
            for (auto &line : *comments) {
                os << " * " << line << '\n';
            }
            os << "*/\n";
        }
    }
}

func_c ir_printer_t::dispatch(func_c f) {
    print_func_comments(f, os_);
    os_ << "func " << f->name_ << '(';
    if (!f->params_.empty()) {
        for (unsigned i = 0; i < f->params_.size() - 1; i++) {
            print_single_arg(*this, f->params_.at(i)) << ", ";
        }
        print_single_arg(*this, f->params_.back());
    }
    os_ << "): " << f->ret_type_ << ' ';
    if (f->body_.defined()) { do_dispatch(f->body_); }
    return f;
}

std::ostream &ir_printer_t::do_dispatch(const ir_module_t &m) {
    for (auto &f : m.get_module_vars()) {
        do_dispatch(f) << '\n';
    }
    for (auto &f : m.get_contents()) {
        do_dispatch(f) << '\n';
    }
    return os_;
}

std::ostream &ir_printer_t::do_dispatch(const func_c &m) {
    dispatch(m);
    return os_;
}

std::ostream &ir_printer_t::do_dispatch(const expr_c &m) {
    dispatch(m);
    return os_;
}

std::ostream &ir_printer_t::do_dispatch(const stmt_c &m) {
    dispatch(m);
    return os_;
}

class ir_track_pos_printer_t : public ir_printer_t {
public:
    track_pos_stream_t theos_;
    using ir_printer_t::dispatch;
    using ir_printer_t::view;

    func_c dispatch(func_c f) override {
        std::const_pointer_cast<func_base>(f)->attr()["source_pos"]
                = source_pos {theos_.buf_.pos_, theos_.buf_.line_};
        return ir_printer_t::dispatch(f);
    }

    stmt_c dispatch(stmt_c f) override {
        f.remove_const()->attr()["source_pos"]
                = source_pos {theos_.buf_.pos_, theos_.buf_.line_};
        return ir_printer_t::dispatch(f);
    }

    expr_c dispatch(expr_c f) override {
        if (!f.isa<var>() && !f.isa<tensor>()) {
            f.remove_const()->attr()["source_pos"]
                    = source_pos {theos_.buf_.pos_, theos_.buf_.line_};
        }
        return ir_printer_t::dispatch(f);
    }

    ir_track_pos_printer_t(std::ostream &os)
        : ir_printer_t(theos_), theos_(os) {}
};

void print_ir_and_annotate_source_pos(const ir_module_t &v, std::ostream &os) {
    ir_track_pos_printer_t p {os};
    p.do_dispatch(v);
}

void print_ir_and_annotate_source_pos(const func_c &v, std::ostream &os) {
    ir_track_pos_printer_t p {os};
    p.do_dispatch(v);
}

void print_ir_and_annotate_position_in_source(
        const func_c &scope, const node_base *v, std::ostream &os) {
    std::stringstream ss;
    ir_track_pos_printer_t p {ss};
    p.do_dispatch(scope);
    auto src = ss.str();
    os << src;
    if (v->attr_) {
        auto pos = v->attr_->get_or_null<source_pos>("source_pos");
        if (pos) {
            os << "\n===================\nat line:col (" << pos->line_ << ':'
               << pos->pos_ << ')' << '\n';
            auto lines = utils::string_split(src, "\n");
            if (pos->line_ - 1UL < lines.size()) {
                os << "  " << lines[pos->line_ - 1UL] << '\n';
                for (int i = 0; i < pos->pos_ + 2; i++) {
                    os << ' ';
                }
                os << "^^^\n";
            }
        }
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
