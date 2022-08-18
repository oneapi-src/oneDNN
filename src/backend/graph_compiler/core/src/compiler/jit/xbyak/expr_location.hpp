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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_EXPR_LOCATION_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_EXPR_LOCATION_HPP

#include <sstream>
#include <compiler/ir/sc_expr.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

namespace sc {
namespace sc_xbyak {

class expr_location {
public:
    enum class type {
        none,
        imm,
        reg,
        stack_var,
        stack_tensor,
        simd_constant,
    };

    expr_location();
    expr_location(uint64_t imm, x86_64::cpu_data_type data_type);
    expr_location(Xbyak::Reg reg, x86_64::cpu_data_type data_type);
    expr_location(Xbyak::Label *label, x86_64::cpu_data_type data_type);
    expr_location(
            Xbyak::RegExp reg_exp, type t, x86_64::cpu_data_type data_type);

    type get_type() const;
    x86_64::cpu_data_type get_data_type() const;

    uint64_t get_imm() const;
    Xbyak::Reg get_reg() const;
    Xbyak::RegExp get_stack_var() const;
    Xbyak::RegExp get_stack_tensor() const;
    Xbyak::Label *get_simd_constant() const;

    // Factory methods, for convenience.
    static expr_location make_imm(uint64_t imm,
            x86_64::cpu_data_type data_type = x86_64::cpu_data_type::uint_64);
    static expr_location make_reg(Xbyak::Reg reg,
            x86_64::cpu_data_type data_type = x86_64::cpu_data_type::uint_64);
    static expr_location make_stack_var(Xbyak::RegExp reg_exp,
            x86_64::cpu_data_type data_type = x86_64::cpu_data_type::uint_64);
    static expr_location make_stack_tensor(Xbyak::RegExp reg_exp);
    static expr_location make_simd_constant(Xbyak::Label *label,
            x86_64::cpu_data_type data_type = x86_64::cpu_data_type::uint_64);

    friend std::ostream &operator<<(std::ostream &os, const expr_location &v);

private:
    union defined_location {
        uint64_t imm_;
        Xbyak::Reg reg_;
        Xbyak::Label *label_;
        Xbyak::RegExp reg_exp_;
        defined_location(uint64_t imm) { imm_ = imm; }
        defined_location(Xbyak::Reg reg) { reg_ = reg; }
        defined_location(Xbyak::Label *label) { label_ = label; }
        defined_location(Xbyak::RegExp reg_exp) { reg_exp_ = reg_exp; }
    } content_;

    type type_;
    x86_64::cpu_data_type data_type_;
};

std::ostream &operator<<(std::ostream &os, expr_location &v);

} // namespace sc_xbyak
} // namespace sc

#endif
