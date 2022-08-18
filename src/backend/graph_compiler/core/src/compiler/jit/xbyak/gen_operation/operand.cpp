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

#include "operand.hpp"

namespace sc {
namespace sc_xbyak {

using K = Xbyak::Operand::Kind;
#define KIND_REG (K::REG)
#define KIND_XYZ (K::XMM | K::YMM | K::ZMM)
#define KIND_MSK (K::OPMASK)

// constructor
operand::operand() : content_(0), type_(operand::type::none) {}

operand::operand(const uint64_t &imm)
    : content_(imm), type_(operand::type::imm) {}

operand::operand(const Xbyak::Reg &reg)
    : content_(reg), type_(operand::type::reg) {
    reg_kind_ = reg.is(KIND_XYZ) ? KIND_XYZ : reg.getKind();
    reg_indx_ = reg.getIdx();
}

operand::operand(const Xbyak::Address &addr)
    : content_(addr), type_(operand::type::addr) {}

// get operand type
operand::type operand::get_type() const {
    return type_;
}

// get operand
uint64_t operand::get_imm() const {
    assert(type_ == operand::type::imm);
    return content_.imm_;
}

Xbyak::Reg operand::get_reg_base() const {
    assert(type_ == operand::type::reg);
    return content_.reg_;
}

Xbyak::Reg operand::get_reg() const {
    assert(is_reg());
    return content_.reg_;
}

Xbyak::Xmm operand::get_xyz() const {
    assert(is_xyz());
    return Xbyak::Xmm(content_.reg_.getKind(), content_.reg_.getIdx());
}

Xbyak::Opmask operand::get_mask() const {
    assert(is_mask());
    return Xbyak::Opmask(content_.reg_.getIdx());
}

Xbyak::Address operand::get_addr() const {
    assert(type_ == operand::type::addr);
    return content_.addr_;
}

// get reg/addr operand
const Xbyak::Operand &operand::get_operand() const {
    if (type_ == operand::type::reg) {
        return content_.reg_;
    } else if (type_ == operand::type::addr) {
        return content_.addr_;
    } else {
        assert(0 && "Operand must be reg or addr");
    }
    return content_.reg_; // unreachable
}

// get reg[8/16/32/64] operand
Xbyak::Reg64 operand::get_reg64() const {
    assert(is_reg());
    return Xbyak::Reg64(content_.reg_.getIdx());
}

Xbyak::Reg32 operand::get_reg32() const {
    assert(is_reg());
    return Xbyak::Reg32(content_.reg_.getIdx());
}

Xbyak::Reg16 operand::get_reg16() const {
    assert(is_reg());
    return Xbyak::Reg16(content_.reg_.getIdx());
}

Xbyak::Reg8 operand::get_reg8() const {
    assert(is_reg());
    return Xbyak::Reg8(content_.reg_.getIdx());
}

// get [xyz]mm operand
Xbyak::Xmm operand::get_xmm() const {
    assert(is_xyz());
    return Xbyak::Xmm(content_.reg_.getIdx());
}

Xbyak::Ymm operand::get_ymm() const {
    assert(is_xyz());
    return Xbyak::Ymm(content_.reg_.getIdx());
}

Xbyak::Zmm operand::get_zmm() const {
    assert(is_xyz());
    return Xbyak::Zmm(content_.reg_.getIdx());
}

// check certain operand
bool operand::is_imm() const {
    return type_ == operand::type::imm;
}

bool operand::is_reg() const {
    if (type_ == operand::type::reg) { return content_.reg_.is(KIND_REG); }
    return false;
}

bool operand::is_xyz() const {
    if (type_ == operand::type::reg) { return content_.reg_.is(KIND_XYZ); }
    return false;
}

bool operand::is_mask() const {
    if (type_ == operand::type::reg) { return content_.reg_.is(KIND_MSK); }
    return false;
}

bool operand::is_addr() const {
    return type_ == operand::type::addr;
}

// check same operand
bool operand::operator==(const operand &b) const {
    if (type_ == b.type_) {
        switch (type_) {
            case operand::type::none: {
                return true;
            }
            case operand::type::imm: {
                return content_.imm_ == b.content_.imm_;
            }
            case operand::type::reg: {
                return reg_kind_ == b.reg_kind_ && reg_indx_ == b.reg_indx_;
            }
            case operand::type::addr: {
                return content_.addr_ == b.content_.addr_;
            }
        }
    }
    return false;
}

// std iostream
std::ostream &operator<<(std::ostream &os, const operand &op) {
    switch (op.type_) {
        case operand::type::none: {
            os << "[empty_operand]";
        } break;
        case operand::type::imm: {
            os << "[" << op.get_imm() << "]";
        } break;
        case operand::type::reg: {
            os << "[" << op.content_.reg_.toString() << "]";
        } break;
        case operand::type::addr: {
            os << "[addr_operand]";
        } break;
    }
    return os;
}

} // namespace sc_xbyak
} // namespace sc
