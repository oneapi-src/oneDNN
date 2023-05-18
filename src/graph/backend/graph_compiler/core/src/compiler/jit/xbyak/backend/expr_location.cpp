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

#include <compiler/jit/xbyak/backend/operand.hpp>
#include <util/utils.hpp>

#include "expr_location.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

//============================================================================
//  get member
//============================================================================

expr_location::type expr_location::get_type() const {
    return type_;
}

x86_64::cpu_data_type expr_location::get_data_type() const {
    return data_type_;
}

op_ptr_t expr_location::get_op_ptr() const {
    return content_;
}

int64_t expr_location::get_imm() const {
    COMPILE_ASSERT(type_ == type::imm, "Not a imm: " << (*this));
    return content_->as<int64_t>();
}

int64_t expr_location::get_stack_var() const {
    COMPILE_ASSERT(type_ == type::stack_var, "Not a stack_var: " << (*this));
    return content_->as<int64_t>();
}

int64_t expr_location::get_stack_tensor() const {
    COMPILE_ASSERT(
            type_ == type::stack_tensor, "Not a stack_tensor: " << (*this));
    return content_->as<int64_t>();
}

const Xbyak::Reg &expr_location::get_reg() const {
    COMPILE_ASSERT(type_ == type::reg, "Not a reg: " << (*this));
    return content_->as<Xbyak::Reg>();
}

const Xbyak::Address &expr_location::get_simd_constant() const {
    COMPILE_ASSERT(
            type_ == type::simd_constant, "Not a simd_constant: " << (*this));
    return content_->as<Xbyak::Address>();
}

//============================================================================
//  Factory methods
//============================================================================

template <typename RegT>
expr_location expr_location::make_reg(
        RegT reg, x86_64::cpu_data_type cpu_dtype) {
    return expr_location(type::reg, cpu_dtype, std::move(reg));
}

expr_location expr_location::make_imm(
        int64_t imm, x86_64::cpu_data_type cpu_dtype) {
    return expr_location(type::imm, cpu_dtype, imm);
}

expr_location expr_location::make_stack_var(
        int64_t offset, x86_64::cpu_data_type cpu_dtype) {
    return expr_location(type::stack_var, cpu_dtype, offset);
}

expr_location expr_location::make_stack_tensor(int64_t offset) {
    const auto cpu_dtype = x86_64::cpu_data_type::uint_64;
    return expr_location(type::stack_tensor, cpu_dtype, offset);
}

expr_location expr_location::make_simd_constant(
        Xbyak::Address addr, x86_64::cpu_data_type cpu_dtype) {
    return expr_location(type::simd_constant, cpu_dtype, addr);
}

//============================================================================
//  make_reg template methods
//============================================================================

template expr_location expr_location::make_reg<Xbyak::Reg>(
        Xbyak::Reg reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Reg8>(
        Xbyak::Reg8 reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Reg16>(
        Xbyak::Reg16 reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Reg32>(
        Xbyak::Reg32 reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Reg64>(
        Xbyak::Reg64 reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Xmm>(
        Xbyak::Xmm reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Ymm>(
        Xbyak::Ymm reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Zmm>(
        Xbyak::Zmm reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Tmm>(
        Xbyak::Tmm reg, x86_64::cpu_data_type cpu_dtype);
template expr_location expr_location::make_reg<Xbyak::Opmask>(
        Xbyak::Opmask reg, x86_64::cpu_data_type cpu_dtype);

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
            os << "[stack_var: %rbp" << std::showpos << v.get_stack_var()
               << std::noshowpos << "]";
        } break;
        case expr_location::type::stack_tensor: {
            os << "[stack_tensor: %rbp" << std::showpos << v.get_stack_tensor()
               << std::noshowpos << "]";
        } break;
        case expr_location::type::simd_constant: {
            os << "[simd_constant: %rip+.L"
               << v.get_simd_constant().getLabel()->getId() << "]";
        } break;
    }
    return os;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
