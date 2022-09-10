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

#ifndef GPU_JIT_CODEGEN_OPERAND_HPP
#define GPU_JIT_CODEGEN_OPERAND_HPP

#include "gpu/jit/codegen/ngen_helpers.hpp"
#include "gpu/jit/codegen/reg_buf.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class ngen_operand_kind_t {
    invalid,
    immediate,
    reg_buf_data,
    flag_register
};

// Wrapper to generalize ngen::FlagRegister, ngen::RegData, reg_buf_data_t and
// ngen::Immediate operands.
class ngen_operand_t {
public:
    ngen_operand_t() : kind_(ngen_operand_kind_t::invalid) {}

    ngen_operand_t(const ngen::FlagRegister &flag)
        : kind_(ngen_operand_kind_t::flag_register)
        , ptr_(new ngen::FlagRegister(flag),
                  destroy<ngen_operand_kind_t::flag_register>) {}

    ngen_operand_t(const reg_buf_data_t &reg_buf_data)
        : kind_(ngen_operand_kind_t::reg_buf_data)
        , ptr_(new reg_buf_data_t(reg_buf_data),
                  destroy<ngen_operand_kind_t::reg_buf_data>) {}

    ngen_operand_t(const ngen::Immediate &imm)
        : kind_(ngen_operand_kind_t::immediate)
        , ptr_(new ngen::Immediate(imm),
                  destroy<ngen_operand_kind_t::immediate>) {}

    template <typename T>
    ngen_operand_t(const T &other, const ngen::InstructionModifier &mod)
        : ngen_operand_t(other) {
        mod_ = mod;
    }

    const ngen::Immediate &immediate() const {
        ir_assert(is_immediate());
        return *(const ngen::Immediate *)ptr_.get();
    }

    const reg_buf_data_t &reg_buf_data() const {
        ir_assert(is_reg_buf_data());
        return *(const reg_buf_data_t *)ptr_.get();
    }

    ngen::RegData reg_data() const {
        auto &rd = reg_buf_data().reg_data();
        return is_negated_ ? -rd : rd;
    }

    const ngen::FlagRegister &flag_register() const {
        ir_assert(is_flag_register());
        return *(const ngen::FlagRegister *)ptr_.get();
    }

    ngen::InstructionModifier flag_register_mod() const {
        ngen::InstructionModifier mod;
        mod |= flag_register();
        return !is_negated() ? mod : ~mod;
    }

    const ngen::InstructionModifier &mod() const { return mod_; }

    bool is_invalid() const { return kind_ == ngen_operand_kind_t::invalid; }

    bool is_immediate() const {
        return kind_ == ngen_operand_kind_t::immediate;
    }

    bool is_reg_buf_data() const {
        return kind_ == ngen_operand_kind_t::reg_buf_data;
    }

    bool is_reg_data() const { return is_reg_buf_data(); }

    bool is_flag_register() const {
        return kind_ == ngen_operand_kind_t::flag_register;
    }

    bool is_negated() const { return is_negated_; }

    ngen::DataType type() const {
        if (is_immediate()) return immediate().getType();
        if (is_reg_buf_data()) return reg_buf_data().type();
        ir_error_not_expected();
        return ngen::DataType::invalid;
    }

    ngen_operand_t operator-() const {
        if (is_immediate()) return ngen_operand_t(ngen_negate(immediate()));
        if (is_reg_buf_data() || is_flag_register()) {
            auto ret = *this;
            ret.is_negated_ = !ret.is_negated_;
            return ret;
        }
        ir_error_not_expected();
        return ngen_operand_t();
    }

    ngen_operand_t reinterpret(const type_t &new_type) const {
        ir_assert(new_type.is_scalar());
        return ngen_operand_t(
                reg_buf_data().reinterpret(to_ngen(new_type)), mod_);
    }

    // Creates an operand with the requested register region based on the
    // existing region. off - offset in elements of the region data type.
    ngen_operand_t sub_reg_data(int off, int exec_size) const {
        int off_bytes = off * ngen::getBytes(reg_buf_data().type())
                * reg_buf_data().hs();
        auto rd = reg_buf_data().format(off_bytes, ngen::DataType::invalid,
                exec_size, reg_buf_data().hs());
        return ngen_operand_t(rd, exec_size);
    }

    bool operator==(const ngen_operand_t &other) const {
        if (kind_ != other.kind_) return false;
        if (mod_.getAll() != other.mod_.getAll()) return false;
        switch (kind_) {
            case ngen_operand_kind_t::immediate: {
                auto &this_imm = immediate();
                auto &other_imm = other.immediate();
                return (this_imm.getType() == other_imm.getType())
                        && (uint64_t(this_imm) == uint64_t(other_imm));
            }
            case ngen_operand_kind_t::flag_register:
                return flag_register() == other.flag_register();
            case ngen_operand_kind_t::reg_buf_data:
                return reg_buf_data() == other.reg_buf_data();
            default: ir_error_not_expected();
        }
        return false;
    }

private:
    template <ngen_operand_kind_t kind>
    static void destroy(void *ptr) {
        if (!ptr) return;

        switch (kind) {
            case ngen_operand_kind_t::immediate:
                delete (ngen::Immediate *)ptr;
                break;
            case ngen_operand_kind_t::reg_buf_data:
                delete (reg_buf_data_t *)ptr;
                break;
            case ngen_operand_kind_t::flag_register:
                delete (ngen::FlagRegister *)ptr;
                break;
            default: ir_error_not_expected();
        }
    }

    ngen_operand_kind_t kind_;
    std::shared_ptr<void> ptr_;
    ngen::InstructionModifier mod_;

    // Whether the operand is negated. Applicable to flag registers and
    // register data operands only. Negation of immediate operands is directly
    // supported through nGEN API.
    bool is_negated_ = false;
};

template <typename T>
T to_cpp(ngen::HW hw, const ngen_operand_t &op) {
    ir_assert(op.is_immediate());
    return to_cpp<T>(op.immediate());
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
