/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_OPERAND_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_OPERAND_HPP

#include <iostream>
#include <memory>
#include <utility>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/x86_64/native_types.hpp>
#include <util/variant.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/**
 * Operand content, contains imm, Xbyak reg or addr.
 * xbyak content can be dynamic convert to its base classes.
 **/
using op_content_t = variant<int64_t, Xbyak::Address, Xbyak::Reg, Xbyak::Reg8,
        Xbyak::Reg16, Xbyak::Reg32, Xbyak::Reg64, Xbyak::Xmm, Xbyak::Ymm,
        Xbyak::Zmm, Xbyak::Tmm, Xbyak::Opmask>;

using op_ptr_t = std::shared_ptr<op_content_t>;

template <typename T>
op_ptr_t wrap_op_ptr(T op) {
    return std::make_shared<op_content_t>(std::move(op));
}

/**
 * The xbyak codegen operand, contains imm, reg or addr.
 * Check instruction format and use suitable operand.
 **/
class operand {
public:
    // constant
    constexpr static bool T_z = true;
    // operand type
    enum type {
        none = 0,
        imm,
        reg,
        addr,
    };

    // constructor
    operand();
    operand(operand::type tp, op_ptr_t ptr);
    operand(operand::type tp, op_ptr_t ptr, //
            int reg_kind, int reg_indx, int reg_bits);

    template <typename RegT>
    explicit operand(RegT reg);
    explicit operand(int64_t imm);
    explicit operand(Xbyak::Address addr);

    // get operand type
    operand::type get_type() const;

    // get imm operand
    int64_t get_imm() const;

    // get reg/addr operand
    const Xbyak::Reg &get_reg() const;
    const Xbyak::Address &get_addr() const;
    const Xbyak::Operand &get_operand() const;

    // get reg[8/16/32/64] operand
    Xbyak::Reg64 get_reg64() const;
    Xbyak::Reg32 get_reg32() const;
    Xbyak::Reg16 get_reg16() const;
    Xbyak::Reg8 get_reg8() const;

    // get [xyz]mm operand
    const Xbyak::Xmm &get_xmm() const;
    const Xbyak::Ymm &get_ymm() const;
    const Xbyak::Zmm &get_zmm() const;

    // get tmm operand
    const Xbyak::Tmm &get_tmm() const;

    // get opmask operand
    const Xbyak::Opmask &get_mask() const;

    // check certain operand
    bool is_imm() const;
    bool is_reg() const;
    bool is_xyz() const;
    bool is_tmm() const;
    bool is_mask() const;
    bool is_addr() const;
    bool is_r_m() const;
    bool is_x_m() const;

    // check reg operand size
    bool is_reg(int bit) const;
    bool is_xyz(int bit) const;
    bool is_bit(int bit) const;

    // Set evex for avx512 operands
    operand set_evex(const operand &mask, bool zero = false) const;

    // check same operand
    bool operator==(const operand &b) const;

    // std iostream
    friend std::ostream &operator<<(std::ostream &os, const operand &op);

private:
    operand::type type_ = operand::type::none;
    op_ptr_t content_;

    // only used for mov compare
    int reg_kind_ = 0;
    int reg_indx_ = 0;
    int reg_bits_ = 0;
};

std::ostream &operator<<(std::ostream &os, const operand &op);

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
