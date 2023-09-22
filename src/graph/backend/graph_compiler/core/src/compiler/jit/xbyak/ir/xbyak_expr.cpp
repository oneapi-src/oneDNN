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
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>

#include "xbyak_expr.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

//=========================================================================
// xbyak intrinsic handlers
//=========================================================================

struct xbyak_intrinsic_handler_t {
    std::string name_;
    xbyak_intrin_format intrin_format_;
    virtual void on_initialize(xbyak_intrin_node &node) = 0;
    xbyak_intrinsic_handler_t(const std::string &name,
            xbyak_intrin_format format = xbyak_intrin_format::undefined)
        : name_(name), intrin_format_(format) {}
    virtual ~xbyak_intrinsic_handler_t() = default;
};

//=========================================================================
// register xbyak intrinsic handlers
//=========================================================================

#define _1A_ 0 // one address intrinsic
#define _2A_ 1 // two address intrinsic
#define _3A_ 2 // three address intrinsic
#define _4A_ 3 // four address intrinsic
#define _5A_ 4 // five address intrinsic

#define TO_INDEX(X) (static_cast<size_t>(X))

#define REGISTER_INTRIN(NAME, ISA, INTRIN, FORMAT, INPUT) \
    struct ISA##INTRIN##_handler_t : public xbyak_intrinsic_handler_t { \
        ISA##INTRIN##_handler_t() \
            : xbyak_intrinsic_handler_t(NAME, xbyak_intrin_format::FORMAT) {} \
        void on_initialize(xbyak_intrin_node &node) override { \
            assert(node.args_.size() == (INPUT)); \
            node.format_ = intrin_format_; \
        } \
    }; \
    intrin_handlers[TO_INDEX(xbyak_intrin_isa::ISA)] \
                   [TO_INDEX(xbyak_intrin_type::INTRIN)] \
            = utils::make_unique<ISA##INTRIN##_handler_t>();

using handler_table
        = std::vector<std::vector<std::unique_ptr<xbyak_intrinsic_handler_t>>>;

static handler_table register_handlers() {
    const auto ISA_NUM = TO_INDEX(xbyak_intrin_isa::NUM_ISAS);
    const auto INTRIN_NUM = TO_INDEX(xbyak_intrin_type::NUM_INTRINSICS);
    // Create a 2d table of (isa * intrin)
    handler_table intrin_handlers(ISA_NUM);
    for (size_t i = 0; i < ISA_NUM; i++) {
        intrin_handlers[i].resize(INTRIN_NUM);
    }

    // Special Intrinsic
    REGISTER_INTRIN("CALL_ARG", x86, call_arg, undefined, _2A_);
    REGISTER_INTRIN("SIGN_EXT", x86, sign_ext, undefined, _2A_);
    REGISTER_INTRIN("REINTERPRET", x86, reinterpret, directed_dst_reg, _2A_);
    REGISTER_INTRIN("REINTERPRET", avx, reinterpret, directed_dst_reg, _2A_);
    REGISTER_INTRIN(
            "SATURATED_CAST", x86, saturated_cast, directed_dst_mem, _2A_);
    REGISTER_INTRIN(
            "SATURATED_CAST", avx, saturated_cast, directed_dst_mem, _2A_);
    REGISTER_INTRIN(
            "ROUND_AND_CAST", x86, round_and_cast, directed_dst_reg, _2A_);
    REGISTER_INTRIN(
            "ROUND_AND_CAST", avx, round_and_cast, directed_dst_reg, _2A_);

    //---------------
    // X86 Intrinsic
    //---------------
    REGISTER_INTRIN("X86_TEST", x86, test, undefined, _2A_);

    REGISTER_INTRIN("X86_CMOV", x86, cmov, directed_dst_reg, _2A_);

    REGISTER_INTRIN("X86_ADD", x86, add, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_SUB", x86, sub, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_SHL", x86, shl, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_SHR", x86, shr, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_SAR", x86, sar, compound_dst_mem, _2A_);

    REGISTER_INTRIN("X86_BIT_OR", x86, bit_or, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_BIT_AND", x86, bit_and, compound_dst_mem, _2A_);
    REGISTER_INTRIN("X86_BIT_XOR", x86, bit_xor, compound_dst_mem, _2A_);

    REGISTER_INTRIN("X86_MUL", x86, mul, compound_dst_reg, _2A_);
    REGISTER_INTRIN("X86_MULI", x86, muli, directed_dst_reg, _3A_);
    REGISTER_INTRIN("X86_MULHL", x86, mulhl, directed_dst_reg, _3A_);

    REGISTER_INTRIN("X86_MIN", x86, min, compound_dst_reg, _2A_);
    REGISTER_INTRIN("X86_MAX", x86, max, compound_dst_reg, _2A_);

    // div/idiv as 4-address to include %rax and %rdx for liveness update
    REGISTER_INTRIN("X86_DIV", x86, div, undefined, _4A_);
    REGISTER_INTRIN("X86_MOD", x86, mod, undefined, _4A_);

    REGISTER_INTRIN("X86_NEG", x86, neg, undefined, _1A_);

    REGISTER_INTRIN("X86_CMP_SET", x86, cmp_set, directed_dst_mem, _3A_);
    REGISTER_INTRIN("X86_BMI_PEXT", x86, bmi_pext, directed_all_reg, _3A_);

    //---------------
    // AVX Intrinsic
    //---------------
    // mask_mov: special case xbyak_format in resolve_spill
    REGISTER_INTRIN("AVX_MASK_MOV", avx, mask_mov, undefined, _2A_);
    REGISTER_INTRIN("AVX_MOV_MASK", avx, mov_mask, directed_all_reg, _2A_);

    REGISTER_INTRIN("AVX_CMOV", avx, cmov, directed_dst_mem, _2A_);
    REGISTER_INTRIN("AVX_MOVD", avx, movd, directed_dst_mem, _2A_);

    REGISTER_INTRIN("AVX_ADD", avx, add, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_SUB", avx, sub, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_SHL", avx, shl, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_SHR", avx, shr, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_SAR", avx, sar, directed_end_mem, _3A_);

    REGISTER_INTRIN("AVX_BIT_OR", avx, bit_or, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_BIT_AND", avx, bit_and, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_BIT_XOR", avx, bit_xor, directed_end_mem, _3A_);

    REGISTER_INTRIN("AVX_MIN", avx, min, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_MAX", avx, max, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_ABS", avx, abs, directed_end_mem, _2A_);

    REGISTER_INTRIN("AVX_MUL", avx, mul, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_MULHL", avx, mulhl, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_DIV", avx, div, directed_end_mem, _3A_);

    REGISTER_INTRIN("AVX_CEIL", avx, ceil, directed_end_mem, _2A_);
    REGISTER_INTRIN("AVX_FLOOR", avx, floor, directed_end_mem, _2A_);
    REGISTER_INTRIN("AVX_ROUND", avx, round, directed_end_mem, _2A_);

    REGISTER_INTRIN("AVX_SQRT", avx, sqrt, directed_end_mem, _2A_);
    REGISTER_INTRIN("AVX_RSQRT", avx, rsqrt, directed_end_mem, _2A_);

    REGISTER_INTRIN("AVX_FMADD", avx, fmadd, compound_dst_reg, _3A_);

    REGISTER_INTRIN("AVX_BLEND", avx, blend, directed_end_mem, _3A_);

    REGISTER_INTRIN("AVX_CMP_SET", avx, cmp_set, directed_end_mem, _3A_);

    REGISTER_INTRIN("AVX_PSHUFFLE", avx, pshuffle, directed_end_mem, _3A_);
    REGISTER_INTRIN("AVX_SHUFFLE", avx, shuffle, directed_end_mem, _5A_);
    REGISTER_INTRIN("AVX_PERMUTE", avx, permute, directed_end_mem, _4A_);
    REGISTER_INTRIN("AVX_INSERT", avx, insert, directed_end_mem, _4A_);
    REGISTER_INTRIN("AVX_EXTRACT", avx, extract, directed_end_mem, _4A_);
    REGISTER_INTRIN("AVX_GATHER", avx, gather, directed_all_reg, _3A_);

    REGISTER_INTRIN("AVX_BROADCAST", avx, broadcast, directed_end_mem, _2A_);

    REGISTER_INTRIN("AVX_UNPACK_LOW", avx, unpack_low, directed_end_mem, _4A_);
    REGISTER_INTRIN(
            "AVX_UNPACK_HIGH", avx, unpack_high, directed_end_mem, _4A_);

    REGISTER_INTRIN(
            "AVX_EXTRACT_LOW", avx, extract_low, directed_dst_mem, _2A_);
    REGISTER_INTRIN(
            "AVX_EXTRACT_HIGH", avx, extract_high, directed_all_reg, _2A_);

    REGISTER_INTRIN(
            "AVX_PERMUTEX2VAR", avx, permutex2var, directed_end_mem, _3A_);
    REGISTER_INTRIN(
            "AVX_PERMUTEXVAR", avx, permutexvar, directed_end_mem, _4A_);

    // Finalize table
    return intrin_handlers;
}

//=========================================================================
// list of xbyak intrinsic handlers
//=========================================================================

static handler_table xbyak_handlers = register_handlers();

//=========================================================================
// get_xbyak_intrin_handler
//=========================================================================

xbyak_intrinsic_handler_t &get_xbyak_intrin_handler(
        xbyak_intrin_isa isa, int64_t intrin) {
    auto &handler = xbyak_handlers[TO_INDEX(isa)][TO_INDEX(intrin)];
    COMPILE_ASSERT(handler,
            "Invalid isa-intrin code: " << TO_INDEX(isa) << " - "
                                        << TO_INDEX(intrin));
    return *handler;
}

//=========================================================================
// xbyak_condition
//=========================================================================

xbyak_condition get_xbyak_condition(sc_expr_type t) {
    switch (t) {
        case sc_expr_type::cmp_eq: return xbyak_condition::eq;
        case sc_expr_type::cmp_lt: return xbyak_condition::lt;
        case sc_expr_type::cmp_le: return xbyak_condition::le;
        case sc_expr_type::cmp_ne: return xbyak_condition::ne;
        case sc_expr_type::cmp_ge: return xbyak_condition::ge;
        case sc_expr_type::cmp_gt: return xbyak_condition::gt;
        default: COMPILE_ASSERT(false, "Invalid compare type: " << t);
    }
    return xbyak_condition::none;
}

std::ostream &operator<<(std::ostream &os, const xbyak_condition t) {
    switch (t) {
        case xbyak_condition::none: os << "NONE"; break;
        case xbyak_condition::eq: os << "EQ"; break;
        case xbyak_condition::ne: os << "NE"; break;
        case xbyak_condition::lt: os << "LT"; break;
        case xbyak_condition::le: os << "LE"; break;
        case xbyak_condition::gt: os << "GT"; break;
        case xbyak_condition::ge: os << "GE"; break;
    }
    return os;
}

//=========================================================================
// xbyak_intrin_node
//=========================================================================

expr xbyak_intrin_node::remake() const {
    expr node = make_xbyak_intrin(this->dtype_, this->args_,
            static_cast<xbyak_intrin_type>(this->type_), this->isa_,
            this->modifier_);
    return copy_attr(*this, std::move(node));
}

void xbyak_intrin_node::to_string(ostream &os) const {
    auto &v = get_xbyak_intrin_handler(isa_, type_);
    if (modifier_.enabled_) {
        os << "{";
        if (modifier_.cond_code_ != xbyak_condition::none) {
            os << modifier_.cond_code_;
        }
        if (modifier_.cond_mask_.defined()) {
            os << "|" << modifier_.cond_mask_;
        }
        if (modifier_.zero_mask_) { os << "|Z"; }
        os << "}";
    }
    os << v.name_ << '(';
    if (!args_.empty()) {
        for (unsigned i = 0; i < args_.size() - 1; i++) {
            os << args_.at(i) << ", ";
        }
        os << args_.back();
    }
    os << ')';
}

xbyak_intrin_node::xbyak_intrin_node(const std::vector<expr> &args,
        xbyak_intrin_type intrin, xbyak_intrin_isa isa,
        xbyak_intrin_modifier modifier)
    : low_level_intrin_node(low_level_intrin_kind::x86_xbyak,
            static_cast<int64_t>(intrin), args, {})
    , modifier_(std::move(modifier))
    , isa_(isa) {
    get_xbyak_intrin_handler(isa_, type_).on_initialize(*this);
}

expr make_xbyak_intrin(sc_data_type_t dtype, const std::vector<expr> &values,
        xbyak_intrin_type intrin, xbyak_intrin_isa isa,
        xbyak_intrin_modifier modifier) {
    expr node = make_expr<xbyak_intrin_node>(
            values, intrin, isa, std::move(modifier));
    node->dtype_ = dtype;
    return node;
}

expr make_physical_reg(
        sc_data_type_t dtype, const Xbyak::Reg &reg, const std::string &post) {
    expr node = builder::make_var(
            dtype, std::string("%") + reg.toString() + post);
    node->temp_data() = xbyak_expr_data_t(reg);
    return node;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
