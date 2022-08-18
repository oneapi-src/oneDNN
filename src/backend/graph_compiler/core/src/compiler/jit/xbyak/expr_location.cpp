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

#include <compiler/jit/xbyak/expr_location.hpp>

#include <util/utils.hpp>

namespace sc {
namespace sc_xbyak {

//============================================================================
// Constructors
//============================================================================

// None
expr_location::expr_location()
    : content_(UINT64_C(0)), type_(expr_location::type::none) {}

// Immediate
expr_location::expr_location(uint64_t imm, x86_64::cpu_data_type data_type)
    : content_(imm), type_(expr_location::type::imm), data_type_(data_type) {}

// Register
expr_location::expr_location(Xbyak::Reg reg, x86_64::cpu_data_type data_type)
    : content_(reg), type_(expr_location::type::reg), data_type_(data_type) {}

// SIMD Constant
expr_location::expr_location(
        Xbyak::Label *label, x86_64::cpu_data_type data_type)
    : content_(label)
    , type_(expr_location::type::simd_constant)
    , data_type_(data_type) {}

// Stack Var/Tensor
expr_location::expr_location(Xbyak::RegExp reg_exp,
        expr_location::type loc_type, x86_64::cpu_data_type data_type)
    : content_(reg_exp), type_(loc_type), data_type_(data_type) {
    COMPILE_ASSERT(loc_type == expr_location::type::stack_var
                    || loc_type == expr_location::type::stack_tensor,
            "Invalid expr_location type init by reg_exp");
}

//============================================================================
//  get member
//============================================================================

expr_location::type expr_location::get_type() const {
    return type_;
}

x86_64::cpu_data_type expr_location::get_data_type() const {
    return data_type_;
}

uint64_t expr_location::get_imm() const {
    COMPILE_ASSERT(type_ == type::imm, "Not a imm: " << (*this));
    return content_.imm_;
}

Xbyak::Reg expr_location::get_reg() const {
    COMPILE_ASSERT(type_ == type::reg, "Not a reg: " << (*this));
    return content_.reg_;
}

Xbyak::RegExp expr_location::get_stack_var() const {
    COMPILE_ASSERT(type_ == type::stack_var, "Not a stack_var: " << (*this));
    return content_.reg_exp_;
}

Xbyak::RegExp expr_location::get_stack_tensor() const {
    COMPILE_ASSERT(
            type_ == type::stack_tensor, "Not a stack_tensor: " << (*this));
    return content_.reg_exp_;
}

Xbyak::Label *expr_location::get_simd_constant() const {
    COMPILE_ASSERT(
            type_ == type::simd_constant, "Not a simd_constant: " << (*this));
    return content_.label_;
}

//============================================================================
//  Factory methods
//============================================================================

expr_location expr_location::make_imm(
        uint64_t imm, x86_64::cpu_data_type data_type) {
    return expr_location(imm, data_type);
}

expr_location expr_location::make_reg(
        Xbyak::Reg reg, x86_64::cpu_data_type data_type) {
    return expr_location(reg, data_type);
}

expr_location expr_location::make_stack_var(
        Xbyak::RegExp reg_exp, x86_64::cpu_data_type data_type) {
    return expr_location(reg_exp, expr_location::type::stack_var, data_type);
}

expr_location expr_location::make_stack_tensor(Xbyak::RegExp reg_exp) {
    return expr_location(reg_exp, expr_location::type::stack_tensor,
            x86_64::cpu_data_type::uint_64);
}

expr_location expr_location::make_simd_constant(
        Xbyak::Label *label, x86_64::cpu_data_type data_type) {
    return expr_location(label, data_type);
}

//============================================================================
//  MISC.
//============================================================================

std::ostream &operator<<(std::ostream &os, const expr_location &v) {
    switch (v.get_type()) {
        case expr_location::type::none: {
            os << "[none]";
        } break;
        case expr_location::type::imm: {
            os << "[imm: " << v.get_imm() << "]";
        } break;
        case expr_location::type::reg: {
            os << "[reg: " << v.get_reg().toString() << "]";
        } break;
        case expr_location::type::stack_var: {
            os << "[stack_var: %rbp" << std::showpos
               << (int64_t)v.get_stack_var().getDisp() << "]";
        } break;
        case expr_location::type::stack_tensor: {
            os << "[stack_tensor: %rbp" << std::showpos
               << (int64_t)v.get_stack_tensor().getDisp() << "]";
        } break;
        case expr_location::type::simd_constant: {
            os << "[simd_constant: %rip+.L" << v.get_simd_constant()->getId()
               << "]";
        } break;
    }
    return os;
}

} // namespace sc_xbyak
} // namespace sc
