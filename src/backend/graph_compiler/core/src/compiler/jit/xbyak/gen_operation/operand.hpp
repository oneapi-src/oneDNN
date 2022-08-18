/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_GEN_OPERATION_OPERAND_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_GEN_OPERATION_OPERAND_HPP

#include <iostream>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>

namespace sc {
namespace sc_xbyak {

/**
 * The xbyak codegen operand, contains imm, reg or addr.
 * Check instruction format and use suitable operand.
 **/
class operand {
public:
    // operand type
    enum class type {
        none = 0,
        imm,
        reg,
        addr,
    };

    // constructor
    operand();
    operand(const uint64_t &imm);
    operand(const Xbyak::Reg &reg);
    operand(const Xbyak::Address &addr);

    // get operand type
    operand::type get_type() const;

    // get operand
    uint64_t get_imm() const;
    Xbyak::Reg get_reg_base() const;
    Xbyak::Reg get_reg() const;
    Xbyak::Xmm get_xyz() const;
    Xbyak::Opmask get_mask() const;
    Xbyak::Address get_addr() const;

    // get reg/addr operand
    const Xbyak::Operand &get_operand() const;

    // get reg[8/16/32/64] operand
    Xbyak::Reg64 get_reg64() const;
    Xbyak::Reg32 get_reg32() const;
    Xbyak::Reg16 get_reg16() const;
    Xbyak::Reg8 get_reg8() const;

    // get [xyz]mm operand
    Xbyak::Xmm get_xmm() const;
    Xbyak::Ymm get_ymm() const;
    Xbyak::Zmm get_zmm() const;

    // check certain operand
    bool is_imm() const;
    bool is_reg() const;
    bool is_xyz() const;
    bool is_mask() const;
    bool is_addr() const;

    // check same operand
    bool operator==(const operand &b) const;

    // std iostream
    friend std::ostream &operator<<(std::ostream &os, const operand &op);

private:
    union defined_operand {
        uint64_t imm_;
        Xbyak::Reg reg_;
        Xbyak::Address addr_;
        defined_operand(uint64_t imm) { imm_ = imm; }
        defined_operand(Xbyak::Reg reg) { reg_ = reg; }
        defined_operand(Xbyak::Address addr) { addr_ = addr; }
    } content_;

    operand::type type_ = operand::type::none;
    // only used for mov compare
    int reg_kind_ = 0;
    int reg_indx_ = 0;
};

std::ostream &operator<<(std::ostream &os, const operand &op);

} // namespace sc_xbyak
} // namespace sc

#endif
