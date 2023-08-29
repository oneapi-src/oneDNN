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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_EXPR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_EXPR_HPP

#include <string>
#include <vector>

#include <compiler/ir/sc_expr.hpp>
#include <compiler/ir/sc_stmt.hpp>
#include <util/any_map.hpp>

#include "reg_allocation/virtual_reg.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

#define GET_STMT_DATA(STMT) (STMT)->temp_data().get<xbyak::xbyak_stmt_data_t>()
#define GET_STMT_INDEX(STMT) GET_STMT_DATA(STMT).index_
#define GET_STMT_INIT_INDEX(STMT) GET_STMT_DATA(STMT).init_index_

#define GET_EXPR_DATA(EXPR) (EXPR)->temp_data().get<xbyak::xbyak_expr_data_t>()
#define GET_PHYSICAL_REG(EXPR) GET_EXPR_DATA(EXPR).physical_reg_
#define GET_VIRTUAL_REG(EXPR) GET_EXPR_DATA(EXPR).virtual_reg_
#define GET_LIVE_RANGE(EXPR) GET_EXPR_DATA(EXPR).virtual_reg_.live_range_

struct xbyak_stmt_data_t {
    bool optimized_out_ = false;
    uint64_t loop_depth_ = 0;
    stmt_index_t index_ = -1;
    stmt_index_t init_index_ = -1;
    void set_index(stmt_index_t index) {
        index_ = index;
        init_index_ = (init_index_ == -1) ? index : init_index_;
    }
    xbyak_stmt_data_t() = default;
    xbyak_stmt_data_t(uint64_t loop_depth) : loop_depth_(loop_depth) {};
};

struct xbyak_expr_data_t {
    const stmt_base_t *def_scope_ = nullptr;
    virtual_reg_t virtual_reg_;
    Xbyak::Reg physical_reg_;
    xbyak_expr_data_t() = default;
    xbyak_expr_data_t(const Xbyak::Reg &reg) : physical_reg_(reg) {};
    xbyak_expr_data_t(const stmt_base_t *def_scope) : def_scope_(def_scope) {};
};

/**
 * Xbyak low-level intrinsic type
 * The begining numbers are reserved for low_level_intrin
 * */
enum class xbyak_intrin_type {
    call_arg = 0, // special intrin to represent call arg location (reg/stack)
    sign_ext, // special intrin to represent CWD/CDQ/CQO/XOR before div/idiv
    mask_mov, // special intrin to represent avx512 zero masked move
    mov_mask, // special intrin for avx move mask 8/32
    test, // special intrin to represent x86 bool logical compare
    cmov, // conditional move
    movd, // reinterpret dword f32 <-> int32
    add,
    sub,
    mul, // x86: represent low result 2-address mul(r, r/m)
    muli, // x86: represent low result 3-address mul(r, r/m, i)
    mulhl, // x86: represent high/low result rdx:rax = mul(r/rm)~rax
    div,
    mod,
    shl,
    shr,
    sar,
    min,
    max,
    abs,
    neg,
    bit_or,
    bit_and,
    bit_xor,
    cmp_set,
    ceil,
    floor,
    round,
    sqrt,
    rsqrt,
    fmadd,
    blend,
    pshuffle,
    shuffle,
    permute,
    gather,
    insert,
    extract,
    bmi_pext,
    broadcast,
    reinterpret,
    unpack_low,
    unpack_high,
    extract_low,
    extract_high,
    permutex2var,
    permutexvar,
    saturated_cast,
    round_and_cast,
    NUM_INTRINSICS,
};

/**
 * Intrinsic format to abstractly represent instruction format
 * */
enum class xbyak_intrin_format {
    // ------------------------------------------------------
    // [directed_assign] operate on src, store to dst
    // [compound_assign] operate on src and dst, store to dst
    // ------------------------------------------------------
    /* allow unlimited mem operands */
    undefined = 0, // no restriction
    /* allow 0 mem operands */
    directed_all_reg, // [directed_assign], all must be reg
    /* allow up to 1 mem operands */
    directed_end_mem, // [directed_assign], only end can be mem
    directed_dst_mem, // [directed_assign], dst can be mem
    directed_dst_reg, // [directed_assign], dst must be reg
    compound_dst_mem, // [compound_assign], dst can be mem
    compound_dst_reg, // [compound_assign], dst must be reg
};

/**
 * Intrinsic ISA to abstractly represent instruction type
 * */
enum class xbyak_intrin_isa {
    // TODO(XXX): sse and amx
    x86 = 0, // base x86 intrinsic
    avx, // avx2/avx512 intrinsic
    NUM_ISAS,
};

/**
 * Conditional move and compare set modifier
 * */
enum class xbyak_condition {
    none = 0,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
};

// Compare sc_expr_type to xbyak_condition
xbyak_condition get_xbyak_condition(sc_expr_type t);
// xbyak_condition to ostream
std::ostream &operator<<(std::ostream &os, const xbyak_condition t);

/**
 * Modifier for xbyak_intrin_node
 * Reserved for cmp_set, cmov, blend, etc.
 * */
struct xbyak_intrin_modifier {
    xbyak_condition cond_code_;
    sc_data_type_t type_hint_;
    expr cond_mask_;
    bool zero_mask_;
    bool enabled_;
    xbyak_intrin_modifier()
        : cond_code_(xbyak_condition::none)
        , type_hint_(datatypes::undef)
        , zero_mask_(false)
        , enabled_(false) {}
    xbyak_intrin_modifier(sc_data_type_t type_hint)
        : cond_code_(xbyak_condition::none)
        , type_hint_(type_hint)
        , zero_mask_(false)
        , enabled_(true) {}
    xbyak_intrin_modifier(
            xbyak_condition cond, sc_data_type_t type_hint = datatypes::undef)
        : cond_code_(cond)
        , type_hint_(type_hint)
        , zero_mask_(false)
        , enabled_(true) {}
    xbyak_intrin_modifier(expr mask, bool zero = false)
        : cond_code_(xbyak_condition::none)
        , type_hint_(datatypes::undef)
        , cond_mask_(mask)
        , zero_mask_(zero)
        , enabled_(true) {}
};

/**
 * The xbyak_intrin_node node
 * @param args the arguments
 * @param intrin the intrinsic type
 * @param isa the intrinsic isa level
 * @param modifier the intrinsic modifier
 **/
class xbyak_intrin_node : public low_level_intrin_node {
public:
    xbyak_intrin_node(const std::vector<expr> &args, xbyak_intrin_type intrin,
            xbyak_intrin_isa isa,
            xbyak_intrin_modifier modifier = xbyak_intrin_modifier());
    expr remake() const override;
    void to_string(ostream &os) const override;
    xbyak_intrin_modifier modifier_;
    xbyak_intrin_format format_;
    xbyak_intrin_isa isa_;
};
SC_DEFINE_EXPR_NODE_PTR(xbyak_intrin)

/**
 * Makes a xbyak intrin node
 * */
expr make_xbyak_intrin(sc_data_type_t dtype, const std::vector<expr> &values,
        xbyak_intrin_type intrin, xbyak_intrin_isa isa = xbyak_intrin_isa::x86,
        xbyak_intrin_modifier modifier = xbyak_intrin_modifier());

/**
 * Makes a xbyak reg node
 * */
expr make_physical_reg(sc_data_type_t dtype, const Xbyak::Reg &reg,
        const std::string &post = "");

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
