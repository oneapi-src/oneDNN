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

#include "operand.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using K = Xbyak::Operand::Kind;
#define KIND_REG (K::REG)
#define KIND_XYZ (K::XMM | K::YMM | K::ZMM)
#define KIND_MSK (K::OPMASK)
#define KIND_TMM (K::TMM)

// constructor
operand::operand() : type_(operand::type::none) {}

operand::operand(operand::type tp, op_ptr_t ptr, //
        int reg_kind, int reg_indx, int reg_bits)
    : type_(tp) //
    , content_(std::move(ptr))
    , reg_kind_(reg_kind)
    , reg_indx_(reg_indx)
    , reg_bits_(reg_bits) {}

operand::operand(operand::type tp, op_ptr_t ptr)
    : type_(tp) //
    , content_(std::move(ptr)) {
    if (type_ == operand::type::reg) {
        const auto &reg = content_->as<Xbyak::Reg>();
        reg_kind_ = reg.is(KIND_XYZ) ? KIND_XYZ : reg.getKind();
        reg_indx_ = reg.getIdx();
        reg_bits_ = reg.getBit();
    }
}

operand::operand(int64_t imm)
    : type_(operand::type::imm) //
    , content_(wrap_op_ptr(imm)) {}

operand::operand(Xbyak::Address addr)
    : type_(operand::type::addr) //
    , content_(wrap_op_ptr(addr)) {}

template <typename RegT>
operand::operand(RegT reg) : type_(operand::type::reg) {
    reg_kind_ = reg.is(KIND_XYZ) ? KIND_XYZ : reg.getKind();
    reg_indx_ = reg.getIdx();
    reg_bits_ = reg.getBit();
    content_ = wrap_op_ptr(std::move(reg));
}

// reg constructors
template operand::operand(Xbyak::Reg reg);
template operand::operand(Xbyak::Reg8 reg);
template operand::operand(Xbyak::Reg16 reg);
template operand::operand(Xbyak::Reg32 reg);
template operand::operand(Xbyak::Reg64 reg);
template operand::operand(Xbyak::Xmm reg);
template operand::operand(Xbyak::Ymm reg);
template operand::operand(Xbyak::Zmm reg);
template operand::operand(Xbyak::Tmm reg);
template operand::operand(Xbyak::Opmask reg);

// get operand type
operand::type operand::get_type() const {
    return type_;
}

// get imm operand
int64_t operand::get_imm() const {
    assert(type_ == operand::type::imm);
    return content_->as<int64_t>();
}

// get reg/addr operand
const Xbyak::Reg &operand::get_reg() const {
    assert(type_ == operand::type::reg);
    return content_->as<Xbyak::Reg>();
}

const Xbyak::Address &operand::get_addr() const {
    assert(type_ == operand::type::addr);
    return content_->as<Xbyak::Address>();
}

const Xbyak::Operand &operand::get_operand() const {
    assert(type_ == operand::type::reg || type_ == operand::type::addr);
    return content_->as<Xbyak::Operand>();
}

// get reg[8/16/32/64] operand
Xbyak::Reg64 operand::get_reg64() const {
    assert(is_reg());
    return get_reg().cvt64();
}

Xbyak::Reg32 operand::get_reg32() const {
    assert(is_reg());
    return get_reg().cvt32();
}

Xbyak::Reg16 operand::get_reg16() const {
    assert(is_reg());
    return get_reg().cvt16();
}

Xbyak::Reg8 operand::get_reg8() const {
    assert(is_reg());
    return get_reg().cvt8();
}

// get [xyz]mm operand
const Xbyak::Xmm &operand::get_xmm() const {
    assert(is_xyz());
    return content_->as<Xbyak::Xmm>();
}

const Xbyak::Ymm &operand::get_ymm() const {
    assert(is_xyz());
    return content_->as<Xbyak::Ymm>();
}

const Xbyak::Zmm &operand::get_zmm() const {
    assert(is_xyz());
    return content_->as<Xbyak::Zmm>();
}

// get tmm operand
const Xbyak::Tmm &operand::get_tmm() const {
    assert(is_tmm());
    return content_->as<Xbyak::Tmm>();
}

// get opmask operand
const Xbyak::Opmask &operand::get_mask() const {
    assert(is_mask());
    return content_->as<Xbyak::Opmask>();
}

// check certain operand
bool operand::is_imm() const {
    return type_ == operand::type::imm;
}

bool operand::is_reg() const {
    return type_ == operand::type::reg && reg_kind_ == KIND_REG;
}

bool operand::is_xyz() const {
    return type_ == operand::type::reg && reg_kind_ == KIND_XYZ;
}

bool operand::is_tmm() const {
    return type_ == operand::type::reg && reg_kind_ == KIND_TMM;
}

bool operand::is_mask() const {
    return type_ == operand::type::reg && reg_kind_ == KIND_MSK;
}

bool operand::is_addr() const {
    return type_ == operand::type::addr;
}

bool operand::is_r_m() const {
    return is_reg() || is_addr();
}

bool operand::is_x_m() const {
    return is_xyz() || is_addr();
}

// check reg operand size
bool operand::is_reg(int bit) const {
    return is_reg() && is_bit(bit);
}

bool operand::is_xyz(int bit) const {
    return is_xyz() && is_bit(bit);
}

bool operand::is_bit(int bit) const {
    assert(reg_bits_ != 0);
    return (bit & reg_bits_) != 0;
}

operand operand::set_evex(const operand &mask, bool zero) const {
    assert(is_x_m());
    // Copy current content and get Xbyak::Operand
    auto new_content = op_content_t(*content_);
    auto &op = new_content.as<Xbyak::Operand>();
    // Set EVEX flags
    if (mask.is_mask()) {
        // EVEX mask
        op.setOpmaskIdx(mask.get_mask().getIdx());
    }
    if (zero) {
        // EVEX zero
        op.setZero();
    }
    // new operand
    return operand(type_, wrap_op_ptr(std::move(new_content)), //
            reg_kind_, reg_indx_, reg_bits_);
}

// check same operand
bool operand::operator==(const operand &b) const {
    if (type_ == b.type_) {
        switch (type_) {
            case operand::type::none: {
                return true;
            }
            case operand::type::imm: {
                return get_imm() == b.get_imm();
            }
            case operand::type::reg: {
                return reg_kind_ == b.reg_kind_ && reg_indx_ == b.reg_indx_;
            }
            case operand::type::addr: {
                return get_addr() == b.get_addr();
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
            os << "[" << op.get_reg().toString() << "]";
        } break;
        case operand::type::addr: {
            os << "[addr_operand]";
        } break;
    }
    return os;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
