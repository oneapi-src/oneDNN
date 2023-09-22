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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/jit/xbyak/ir/transform/constant_optimizer.hpp>
#include <compiler/jit/xbyak/ir/util/invariant_int.hpp>
#include <compiler/jit/xbyak/ir/util/utils.hpp>
#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <compiler/jit/xbyak/x86_64/registers.hpp>
#include <util/any_map.hpp>
#include <util/array_ref.hpp>

#include "util/utils.hpp"
#include "x86_intrinsics_lowering.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class x86_intrinsics_lowering_impl_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    x86_intrinsics_lowering_impl_t(
            const runtime::target_machine_t &target_machine)
        : cpu_flags_(target_machine.cpu_flags_) {
        COMPILE_ASSERT(cpu_flags_.fAVX2, "Support AVX2");
    }

    std::vector<stmt> transform_seq_;

    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt> new_seq;
        for (auto &s : v->seq_) {
            auto ss = xbyak_visitor_t::dispatch(s);
            if (!transform_seq_.empty()) {
                for (auto &&ts : transform_seq_) {
                    new_seq.push_back(std::move(ts));
                }
                transform_seq_.clear();
            } else {
                new_seq.emplace_back(ss.remove_const());
            }
        }
        return copy_attr(*v, make_stmt<stmts_node_t>(std::move(new_seq)));
    }

    stmt_c visit(assign_c v) override {
        x86_intrinsics_transform(v->var_, v->value_);
        return std::move(v);
    }

    stmt_c visit(define_c v) override {
        if (v->var_.isa<var>() && v->init_.defined()) {
            add_defination(v->var_, v->linkage_);
            x86_intrinsics_transform(v->var_, v->init_);
        } else if (v->var_.isa<tensor>()
                && v->init_.cast<intrin_call>()
                           .filter([](const intrin_call &v) {
                               return v->type_ == intrin_type::reinterpret
                                       && v->args_.at(0)->dtype_
                                       == datatypes::index;
                           })
                           .has_value()) {
            // handle tensor A[...] = reintepret(...)
            auto intri = v->init_.static_as<intrin_call>();
            add_defination(v->var_, v->linkage_, intri->args_[0]);
        }
        return std::move(v);
    }

    stmt_c visit(evaluate_c v) override {
        if (v->value_.isa<low_level_intrin>()) {
            x86_intrinsics_transform(expr(), v->value_);
        }
        return std::move(v);
    }

    void x86_intrinsics_transform(const expr &dst, const expr &val) {
        switch (val->node_type_) {
            case sc_expr_type::add: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::add),
                        transform_intrin(xbyak_intrin_type::add));
            } break;
            case sc_expr_type::sub: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::sub),
                        transform_intrin(xbyak_intrin_type::sub));
            } break;
            case sc_expr_type::mul: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mul(),
                        transform_intrin(xbyak_intrin_type::mul));
            } break;
            case sc_expr_type::div: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mod_div(xbyak_intrin_type::div),
                        transform_avx_div());
            } break;
            case sc_expr_type::mod: {
                auto bin = val.static_as<binary>();
                transform(dst, {bin->l_, bin->r_},
                        dst->dtype_, //
                        transform_x86_mod_div(xbyak_intrin_type::mod),
                        transform_disabled("mod"));
            } break;
            case sc_expr_type::logic_and: {
                auto log = val.static_as<logic>();
                transform(dst, {log->l_, log->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_and),
                        transform_disabled("logic_and"));
            } break;
            case sc_expr_type::logic_or: {
                auto log = val.static_as<logic>();
                transform(dst, {log->l_, log->r_},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_or),
                        transform_disabled("logic_or"));
            } break;
            case sc_expr_type::logic_not: {
                auto log = val.static_as<logic_not>();
                transform(dst, {log->in_, builder::make_constant(UINT64_C(1))},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_xor),
                        transform_disabled("logic_not"));
            } break;
            case sc_expr_type::select: {
                auto sel = val.static_as<select>();
                transform_select(dst, sel);
            } break;
            case sc_expr_type::intrin_call: {
                auto intrin = val.static_as<intrin_call>();
                transform_intrin_call(dst, intrin);
            } break;
            case sc_expr_type::low_level_intrin: {
                auto intrin = val.static_as<low_level_intrin>();
                transform_low_level_intrin(dst, intrin);
            } break;
            case sc_expr_type::cast: {
                auto src = val.static_as<cast>();
                transform_cast(dst, src);
            } break;
            case sc_expr_type::cmp_eq:
            case sc_expr_type::cmp_ne:
            case sc_expr_type::cmp_lt:
            case sc_expr_type::cmp_le:
            case sc_expr_type::cmp_gt:
            case sc_expr_type::cmp_ge: {
                auto src = val.static_as<cmp>();
                transform_cmp_set(dst, src, val->node_type_);
            } break;
            case sc_expr_type::var: {
                // If potential mask store, do not zero mask
                transform_assign(dst, val, dst, false);
            } break;
            case sc_expr_type::indexing: {
                // If potential mask load, need zero mask
                transform_assign(dst, val, val, true);
            } break;
            default: {
                add_assignment(dst, val);
            } break;
        }
    }

    void transform_intrin_call(const expr &dst, const intrin_call &intrin) {
        switch (intrin->type_) {
            case intrin_type::min: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_min_max(xbyak_intrin_type::min),
                        transform_intrin(xbyak_intrin_type::min));
            } break;
            case intrin_type::max: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_min_max(xbyak_intrin_type::max),
                        transform_intrin(xbyak_intrin_type::max));
            } break;
            case intrin_type::abs: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_x86_abs(),
                        transform_intrin(xbyak_intrin_type::abs));
            } break;
            case intrin_type::shl: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_shift(xbyak_intrin_type::shl),
                        transform_intrin(xbyak_intrin_type::shl));
            } break;
            case intrin_type::shr: {
                auto is_uint = (CATE_UINT
                        == get_etype_category(dst->dtype_.type_code_));
                auto sft_type = is_uint ? xbyak_intrin_type::shr
                                        : xbyak_intrin_type::sar;
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_x86_shift(sft_type),
                        transform_intrin(sft_type));
            } break;
            case intrin_type::ceil: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("ceil"),
                        transform_intrin(xbyak_intrin_type::ceil));
            } break;
            case intrin_type::floor: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("floor"),
                        transform_intrin(xbyak_intrin_type::floor));
            } break;
            case intrin_type::round: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("round"),
                        transform_intrin(xbyak_intrin_type::round));
            } break;
            case intrin_type::sqrt: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("sqrt"),
                        transform_intrin(xbyak_intrin_type::sqrt));
            } break;
            case intrin_type::rsqrt: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("rsqrt"),
                        transform_intrin(xbyak_intrin_type::rsqrt));
            } break;
            case intrin_type::fmadd: {
                transform(dst,
                        {intrin->args_[0], intrin->args_[1], intrin->args_[2]},
                        dst->dtype_, //
                        transform_disabled("fmadd"),
                        transform_4a_to_3a(xbyak_intrin_type::fmadd));
            } break;
            case intrin_type::broadcast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("broadcast"), //
                        transform_broadcast(intrin->args_[0]->dtype_));
            } break;
            case intrin_type::reduce_add: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_add"),
                        transform_simd_reduce_seq(xbyak_intrin_type::add));
            } break;
            case intrin_type::reduce_mul: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_mul"),
                        transform_simd_reduce_seq(xbyak_intrin_type::mul));
            } break;
            case intrin_type::reduce_min: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_min"),
                        transform_simd_reduce_seq(xbyak_intrin_type::min));
            } break;
            case intrin_type::reduce_max: {
                transform(dst, {intrin->args_[0]},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("reduce_max"),
                        transform_simd_reduce_seq(xbyak_intrin_type::max));
            } break;
            case intrin_type::unpack_low: {
                auto imm = intrin->intrin_attrs_->get<int>("elem_bits");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("unpack_low"),
                        transform_intrin(xbyak_intrin_type::unpack_low));
            } break;
            case intrin_type::unpack_high: {
                auto imm = intrin->intrin_attrs_->get<int>("elem_bits");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("unpack_high"),
                        transform_intrin(xbyak_intrin_type::unpack_high));
            } break;
            case intrin_type::shuffle: {
                auto imm = intrin->intrin_attrs_->get<int>("shuffle_imm");
                auto type_bits = intrin->intrin_attrs_->get<int>("type_bits");

                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm),
                                builder::make_constant(type_bits)},
                        dst->dtype_, //
                        transform_disabled("shuffle"),
                        transform_intrin(xbyak_intrin_type::shuffle));
            } break;
            case intrin_type::permute: {
                auto imm = intrin->intrin_attrs_->get<int>("permute_imm");
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(imm)},
                        dst->dtype_, //
                        transform_disabled("permute"),
                        transform_intrin(xbyak_intrin_type::permute));
            } break;
            case intrin_type::gather: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_disabled("gather"), transform_gather());
            } break;
            case intrin_type::insert: {
                auto imm = intrin->intrin_attrs_->get<int>("insert_imm");
                auto elem_bits
                        = utils::get_sizeof_type(intrin->args_[1]->dtype_) * 8;
                bool need_to_reg32 = elem_bits < 32;
                auto insert_val = builder::make_var(
                        sc_data_type_t::u32(1), "insert_val_");
                if (need_to_reg32) {
                    // need reg32
                    add_defination(insert_val, linkage::local);
                    add_assignment(insert_val,
                            builder::make_cast(
                                    datatypes::u32, intrin->args_[1]));
                }
                transform(dst,
                        {intrin->args_[0],
                                need_to_reg32 ? insert_val : intrin->args_[1],
                                builder::make_constant(imm),
                                builder::make_constant(elem_bits)},
                        dst->dtype_, //
                        transform_disabled("insert"),
                        transform_5a_to_4a(xbyak_intrin_type::insert));
            } break;
            case intrin_type::extract: {
                auto imm = intrin->intrin_attrs_->get<int>("extract_imm");
                auto elem_bits = utils::get_sizeof_type(intrin->dtype_) * 8;
                transform(dst,
                        {intrin->args_[0], builder::make_constant(imm),
                                builder::make_constant(elem_bits)},
                        intrin->args_[0]->dtype_, //
                        transform_disabled("extract"), transform_extract());
            } break;
            case intrin_type::int_and: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_and),
                        transform_intrin(xbyak_intrin_type::bit_and));
            } break;
            case intrin_type::int_or: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_or),
                        transform_intrin(xbyak_intrin_type::bit_or));
            } break;
            case intrin_type::int_xor: {
                transform(dst, {intrin->args_[0], intrin->args_[1]},
                        dst->dtype_, //
                        transform_3a_to_2a(xbyak_intrin_type::bit_xor),
                        transform_intrin(xbyak_intrin_type::bit_xor));
            } break;
            case intrin_type::reinterpret: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_intrin(xbyak_intrin_type::reinterpret),
                        transform_intrin(xbyak_intrin_type::reinterpret));
            } break;
            case intrin_type::permutex2var: {
                transform(dst,
                        {intrin->args_[0], intrin->args_[1], intrin->args_[2]},
                        dst->dtype_, //
                        transform_disabled("permutex2var"),
                        transform_4a_to_3a(xbyak_intrin_type::permutex2var));
            } break;
            case intrin_type::permutexvar: {
                const int elem_bits
                        = intrin->intrin_attrs_->get_or_else("lanes", 1)
                        * utils::get_sizeof_etype(dst->dtype_.type_code_) * 8;
                transform(dst,
                        {intrin->args_[0], intrin->args_[1],
                                builder::make_constant(elem_bits)},
                        dst->dtype_, //
                        transform_disabled("permutexvar"),
                        transform_intrin(xbyak_intrin_type::permutexvar));
            } break;
            case intrin_type::saturated_cast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_intrin(xbyak_intrin_type::saturated_cast),
                        transform_intrin(xbyak_intrin_type::saturated_cast));
            } break;
            case intrin_type::round_and_cast: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_intrin(xbyak_intrin_type::round_and_cast),
                        transform_intrin(xbyak_intrin_type::round_and_cast));
            } break;
            case intrin_type::load_const_mem: {
                transform(dst, {intrin->args_[0]},
                        dst->dtype_, //
                        transform_disabled("load_const_mem"),
                        transform_assign());
            } break;
            default: add_assignment(dst, intrin); break;
        }
    }

    void transform_low_level_intrin(
            const expr &dst, const low_level_intrin &intrin) {
        if (intrin->kind_ == low_level_intrin_kind::x86_xbyak) {
            add_assignment(dst, intrin);
            return;
        }
        // only transform x86 low_level_intrin
        COMPILE_ASSERT(intrin->kind_ == low_level_intrin_kind::x86_general,
                "Must be x86 intrinsic!");
        switch (intrin->type_) {
            case x86_intrin_type::avx_broadcast_idx: {
                auto &lanes = intrin->args_[2];
                assert(lanes.isa<constant>());
                auto arg = builder::make_indexing( //
                        intrin->args_[0], {intrin->args_[1]},
                        get_const_as_int(lanes.static_as<constant>()));
                transform_broadcast(dst, arg, arg->dtype_);
            } break;
            case x86_intrin_type::avx_mask_cast: {
                assert(intrin->args_.size() == 1);
                transform_avx_mask_cast(dst, intrin->args_[0]);
            } break;
            case x86_intrin_type::avx_compare: {
                assert(intrin->args_.size() == 3);
                assert(intrin->args_[2].isa<constant_c>());
                auto c = intrin->args_[2].static_as<constant_c>();
                auto code = static_cast<xbyak_condition>(c->value_[0].u64);
                add_assignment(dst,
                        make_xbyak_intrin(dst->dtype_,
                                {intrin->args_[0], intrin->args_[1]},
                                xbyak_intrin_type::cmp_set,
                                xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(code, dst->dtype_)));
            } break;
            default:
                COMPILE_ASSERT(false, "Unknown low level intrinsic!");
                break;
        }
    }

    void transform_cast(const expr &dst, const cast &src) {
        const sc_data_type_t src_dtype = src->in_->dtype_;
        const sc_data_type_t dst_dtype = src->dtype_;

        if (dst_dtype == sc_data_type_t::bf16(1)
                && src_dtype == sc_data_type_t::f32(1)) {
            auto bias = builder::make_var(sc_data_type_t::u32(1), "bias");
            auto temp = builder::make_var(sc_data_type_t::u32(1), "temp");
            add_defination(bias, linkage::local);
            add_defination(temp, linkage::local);
            // temp = reinterpret_cast<u32>(src);
            transform_intrin(temp, {src->in_}, xbyak_intrin_type::reinterpret,
                    xbyak_intrin_isa::x86);
            // bias = temp;
            add_assignment(bias, temp);
            // bias = bias >> 16
            transform_intrin(bias, {builder::make_constant(UINT64_C(16))},
                    xbyak_intrin_type::shr, xbyak_intrin_isa::x86);
            // bias = bias & 1
            transform_intrin(bias, {builder::make_constant(UINT64_C(1))},
                    xbyak_intrin_type::bit_and, xbyak_intrin_isa::x86);
            // bias = bias + 0x7FFF
            transform_intrin(bias,
                    {builder::make_constant(
                            {UINT64_C(32767)}, sc_data_type_t::u32(1))},
                    xbyak_intrin_type::add, xbyak_intrin_isa::x86);
            // temp = temp + bias
            transform_intrin(temp, {bias}, xbyak_intrin_type::add,
                    xbyak_intrin_isa::x86);
            // temp = temp >> 16
            transform_intrin(temp, {builder::make_constant(UINT64_C(16))},
                    xbyak_intrin_type::shr, xbyak_intrin_isa::x86);
            // dst = u16(temp)
            add_assignment(dst, builder::make_cast(dst_dtype, temp));
        } else {
            add_assignment(dst, src);
        }
    }

    void transform_select(const expr &dst, const select &sel) {
        const sc_data_type_t dst_dtype = dst->dtype_;
        // Transform select
        auto cond = cast_when_mask(sel->cond_, dst_dtype);
        auto zero = is_const_zero(sel->r_);
        auto isa = convert_x86_operation(dst_dtype) ? xbyak_intrin_isa::x86
                                                    : xbyak_intrin_isa::avx;
        if (cond->dtype_ == datatypes::boolean) {
            // test cond, cond
            transform_intrin(cond, {cond}, xbyak_intrin_type::test,
                    xbyak_intrin_isa::x86);
            // TODO(XXX): fix x86 select:
            // x86 conditional move only accepts cmovcc(r, r/m)
            if (isa == xbyak_intrin_isa::x86) {
                // temporary solution for cmov
                auto rax = make_physical_reg(dst_dtype, x86_64::regs::rax);
                add_defination(rax, linkage::local);
                // rax = lhs
                add_assignment(rax, sel->l_);
                // If x86 select rhs is constant, load to var
                auto rhs = load_when_imm(sel->r_, "__sel_tmp_var");
                // if(cond = false) rax = rhs
                add_assignment(rax,
                        make_xbyak_intrin(dst_dtype, {rhs},
                                xbyak_intrin_type::cmov, isa,
                                xbyak_intrin_modifier(xbyak_condition::eq)));
                add_assignment(dst, rax);
            } else {
                // dst = lhs
                add_assignment(dst, sel->l_);
                // if(cond = false) dst = rhs
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->r_},
                                xbyak_intrin_type::cmov, isa,
                                xbyak_intrin_modifier(xbyak_condition::eq)));
            }
        } else {
            if (zero && cpu_flags_.fAVX512F) {
                // avx512 zero mask move
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->l_},
                                xbyak_intrin_type::mask_mov, isa,
                                xbyak_intrin_modifier(cond, zero)));
            } else {
                // blend order reversed: select second operand if cond is true
                add_assignment(dst,
                        make_xbyak_intrin(dst_dtype, {sel->r_, sel->l_},
                                xbyak_intrin_type::blend, isa,
                                xbyak_intrin_modifier(cond, zero)));
            }
        }
    }

    void transform_cmp_set(const expr &dst, const cmp &src, sc_expr_type t) {
        const sc_data_type_t cmp_dtype = src->l_->dtype_;
        auto code = get_xbyak_condition(t);
        auto isa = convert_x86_operation(cmp_dtype) ? xbyak_intrin_isa::x86
                                                    : xbyak_intrin_isa::avx;
        if (cmp_dtype == datatypes::f32) {
            // vcmpss  xmm0, xmm1, xmm2
            // vmovd   eax, xmm0
            // and     eax, 1
            auto xmm0 = make_physical_reg(datatypes::f32, x86_64::regs::xmm0);
            add_defination(xmm0, linkage::local);
            add_assignment(xmm0,
                    make_xbyak_intrin(datatypes::f32, {src->l_, src->r_},
                            xbyak_intrin_type::cmp_set, xbyak_intrin_isa::avx,
                            xbyak_intrin_modifier(code, cmp_dtype)));
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {xmm0},
                            xbyak_intrin_type::movd, xbyak_intrin_isa::avx));
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_,
                            {builder::make_constant(
                                    {UINT64_C(1)}, datatypes::u8)},
                            xbyak_intrin_type::bit_and, xbyak_intrin_isa::x86));
        } else {
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {src->l_, src->r_},
                            xbyak_intrin_type::cmp_set, isa,
                            xbyak_intrin_modifier(code, cmp_dtype)));
        }
    }

    // --------------------------------
    // Transform operations and lambdas
    // --------------------------------

    using transform_func = std::function<void(
            const expr &, array_ref<expr>, sc_data_type_t, xbyak_intrin_isa)>;

    void transform(const expr &dst, array_ref<expr> args, sc_data_type_t dtype,
            const transform_func &scalar_f, const transform_func &vec_f) {
        if (convert_x86_operation(dtype)) {
            scalar_f(dst, args, dtype, xbyak_intrin_isa::x86);
        } else {
            vec_f(dst, args, dtype, xbyak_intrin_isa::avx);
        }
    }

    transform_func transform_disabled(const std::string &str) {
        return [str](const expr &dst, array_ref<expr> src, sc_data_type_t dtype,
                       xbyak_intrin_isa isa) {
            COMPILE_ASSERT(false, "Transform disabled: " << str);
        };
    }

    transform_func transform_assign() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype,
                       xbyak_intrin_isa isa) { add_assignment(dst, src[0]); };
    }

    transform_func transform_gather() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_gather(dst, src[0], src[1]);
        };
    }

    transform_func transform_extract() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_extract(dst, src[0], src[1], src[2]);
        };
    }

    transform_func transform_broadcast(sc_data_type_t src_dtype) {
        return [this, src_dtype](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_broadcast(dst, src[0], src_dtype);
        };
    }

    transform_func transform_intrin(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_intrin(dst, src, intrin, isa);
        };
    }

    transform_func transform_3a_to_2a(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_3a_to_2a(dst, src[0], src[1], intrin, isa);
        };
    }

    transform_func transform_4a_to_3a(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_4a_to_3a(dst, src[0], src[1], src[2], intrin, isa);
        };
    }

    transform_func transform_5a_to_4a(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_5a_to_4a(
                    dst, src[0], src[1], src[2], src[3], intrin, isa);
        };
    }

    transform_func transform_x86_mul() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_mul(dst, src[0], src[1]);
        };
    }

    transform_func transform_x86_abs() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_abs(dst, src[0]);
        };
    }

    transform_func transform_x86_shift(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_shift(dst, src[0], src[1], intrin);
        };
    }

    transform_func transform_x86_min_max(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_min_max(dst, src[0], src[1], intrin);
        };
    }

    transform_func transform_x86_mod_div(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_x86_mod_div(dst, src[0], src[1], dtype, intrin);
        };
    }

    transform_func transform_avx_div() {
        return [this](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_avx_div(dst, src[0], src[1], dtype);
        };
    }

    transform_func transform_simd_reduce_seq(xbyak_intrin_type intrin) {
        return [this, intrin](const expr &dst, array_ref<expr> src,
                       sc_data_type_t dtype, xbyak_intrin_isa isa) {
            transform_simd_reduce_seq(dst, src[0], dtype, intrin, isa);
        };
    }

    // -----------------
    // Transform helpers
    // -----------------

    void transform_intrin(const expr &v, array_ref<expr> args,
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        add_assignment(v,
                make_xbyak_intrin(v->dtype_, args.as_vector(), intrin, isa,
                        xbyak_intrin_modifier(v->dtype_)));
    }

    void transform_intrin_eval(array_ref<expr> args, //
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        add_evaluate(make_xbyak_intrin(
                datatypes::void_t, args.as_vector(), intrin, isa));
    }

    void transform_intrin_eval_attr(const expr &v, array_ref<expr> args, //
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        add_evaluate(copy_attr(*v,
                make_xbyak_intrin(
                        datatypes::void_t, args.as_vector(), intrin, isa)));
    }

    void transform_3a_to_2a(const expr &dst, const expr &lhs, const expr &rhs,
            xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        // x86 operations only support up to 32 bit imm
        auto load_if_imm_64_bit = [&](const expr &v) -> expr {
            if (isa == xbyak_intrin_isa::x86 && const_exceed_32bit(v)) {
                auto imm = builder::make_var(v->dtype_, "__64_bit_imm");
                add_defination(imm, linkage::local);
                add_assignment(imm, v);
                return imm;
            } else {
                return v;
            }
        };
        // dst = lhs
        add_assignment(dst, lhs);
        // dst = 2a(rhs)
        transform_intrin(dst, {load_if_imm_64_bit(rhs)}, intrin, isa);
    }

    void transform_4a_to_3a(const expr &dst, const expr &a, const expr &b,
            const expr &c, xbyak_intrin_type intrin, xbyak_intrin_isa isa) {
        // dst = a
        add_assignment(dst, a);
        // dst = 3a(b, c)
        transform_intrin(dst, {b, c}, intrin, isa);
    }

    void transform_5a_to_4a(const expr &dst, const expr &a, const expr &b,
            const expr &c, const expr &d, xbyak_intrin_type intrin,
            xbyak_intrin_isa isa) {
        // dst = src1
        add_assignment(dst, a);
        // dst = 4a(b, c, d)
        transform_intrin(dst, {b, c, d}, intrin, isa);
    }

    void transform_x86_mul(const expr &dst, const expr &lhs, const expr &rhs) {
        if (rhs.isa<constant>()) {
            // mul constant always rhs
            transform_intrin(dst, {lhs, rhs}, xbyak_intrin_type::muli,
                    xbyak_intrin_isa::x86);
        } else {
            transform_3a_to_2a(dst, lhs, rhs, xbyak_intrin_type::mul,
                    xbyak_intrin_isa::x86);
        }
    }

    void transform_x86_shift(const expr &dst, const expr &lhs, const expr &rhs,
            xbyak_intrin_type intrin) {
        if (rhs.isa<constant>()) {
            transform_3a_to_2a(dst, lhs, rhs, intrin, xbyak_intrin_isa::x86);
        } else {
            // if shift rhs is not a imm, must use reg CL to store
            auto rcx = make_physical_reg(rhs->dtype_, x86_64::regs::rcx);
            add_defination(rcx, linkage::local);
            add_assignment(rcx, rhs);
            transform_3a_to_2a(dst, lhs, rcx, intrin, xbyak_intrin_isa::x86);
        }
    }

    void transform_x86_min_max(const expr &dst, const expr &lhs,
            const expr &rhs, xbyak_intrin_type intrin) {
        // dst = lhs
        add_assignment(dst, lhs);
        // dst = min/max(rhs)
        auto tmp = load_when_imm(rhs, "__imm_tmp_var");
        transform_intrin(dst, {tmp}, intrin, xbyak_intrin_isa::x86);
    }

    void transform_x86_abs(const expr &dst, const expr &src) {
        auto dtype = src->dtype_;
        assert(utils::is_one_of(dtype, datatypes::s8, datatypes::s32));
        // x86 conditional move only accepts cmovcc(r, r/m)
        auto rax = make_physical_reg(dtype, x86_64::regs::rax);
        add_defination(rax, linkage::local);
        // rax = src
        add_assignment(rax, src);
        // neg(rax)
        add_assignment(rax,
                make_xbyak_intrin(dtype, {}, xbyak_intrin_type::neg,
                        xbyak_intrin_isa::x86));
        // if(SF != OF) rax = src
        add_assignment(rax,
                make_xbyak_intrin(dtype, {src}, xbyak_intrin_type::cmov,
                        xbyak_intrin_isa::x86,
                        xbyak_intrin_modifier(xbyak_condition::lt)));
        // dst = rax
        add_assignment(dst, rax);
    }

    void transform_x86_mod_div(const expr &dst, const expr &lhs,
            const expr &rhs, sc_data_type_t dtype, xbyak_intrin_type intrin) {
        // Physical regs used by mod/div
        auto rax = make_physical_reg(dtype, x86_64::regs::rax);
        auto rdx = make_physical_reg(dtype, x86_64::regs::rdx);
        add_defination(rax, linkage::local);
        add_defination(rdx, linkage::local);
        if (rhs.isa<constant>() && rhs->dtype_ == datatypes::index) {
            // Get multiplier
            const auto dtype = dst->dtype_;
            const auto divisor = rhs.static_as<constant>()->value_[0].u64;
            const auto mult = invariant_int::UintDivMultiplier(divisor, 64);
            // gen code for div
            if (mult.compensate_) {
                // %rdx = mulh([magic]~%rax, lhs)
                // %rax = lhs - %rdx
                // %rax >> [1]
                // %rdx = %rdx + %rax
                // %rdx >> [sh_post]
                // dst = %rdx
                add_assignment(rax, builder::make_constant(mult.magic_));
                add_assignment(rdx,
                        make_xbyak_intrin(dtype, {rax, lhs},
                                xbyak_intrin_type::mulhl,
                                xbyak_intrin_isa::x86));
                add_assignment(rax, lhs);
                add_assignment(rax,
                        make_xbyak_intrin(dtype, {rdx}, xbyak_intrin_type::sub,
                                xbyak_intrin_isa::x86));
                assert(mult.sft_pre_ == 1);
                add_assignment(rax,
                        make_xbyak_intrin(dtype,
                                {builder::make_constant(mult.sft_pre_)},
                                xbyak_intrin_type::shr, xbyak_intrin_isa::x86));
                add_assignment(rdx,
                        make_xbyak_intrin(dtype, {rax}, xbyak_intrin_type::add,
                                xbyak_intrin_isa::x86));
                if (mult.sft_post_ > 0) {
                    add_assignment(rdx,
                            make_xbyak_intrin(dtype,
                                    {builder::make_constant(mult.sft_post_)},
                                    xbyak_intrin_type::shr,
                                    xbyak_intrin_isa::x86));
                }
            } else {
                // %rdx = lhs
                // %rdx >> [sh_pre]
                // %rdx = mulh([magic]~%rax, %rdx)
                // %rdx >> [sh_post]
                add_assignment(rdx, lhs);
                if (mult.sft_pre_ > 0) {
                    add_assignment(rdx,
                            make_xbyak_intrin(dtype,
                                    {builder::make_constant(mult.sft_pre_)},
                                    xbyak_intrin_type::shr,
                                    xbyak_intrin_isa::x86));
                }
                add_assignment(rax, builder::make_constant(mult.magic_));
                add_assignment(rdx,
                        make_xbyak_intrin(dtype, {rax, rdx},
                                xbyak_intrin_type::mulhl,
                                xbyak_intrin_isa::x86));
                if (mult.sft_post_ > 0) {
                    add_assignment(rdx,
                            make_xbyak_intrin(dtype,
                                    {builder::make_constant(mult.sft_post_)},
                                    xbyak_intrin_type::shr,
                                    xbyak_intrin_isa::x86));
                }
            }
            // gen addtional code for mod
            if (intrin == xbyak_intrin_type::mod) {
                // %rdx = %rdx * rhs
                // dst = lhs - %rdx
                add_assignment(rdx,
                        make_xbyak_intrin(dtype,
                                {rdx, builder::make_constant(divisor)},
                                xbyak_intrin_type::muli,
                                xbyak_intrin_isa::x86));
                add_assignment(dst, lhs);
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {rdx}, xbyak_intrin_type::sub,
                                xbyak_intrin_isa::x86));
            } else {
                // dst = %rdx
                add_assignment(dst, rdx);
            }
        } else {
            // mov %rax, lhs
            auto mov_rax = make_stmt<assign_node_t>(rax, lhs);
            transform_seq_.emplace_back(mov_rax);
            // unsigned [xor %rdx, %rdx] or signed [CWD/CDQ/CQO]
            auto sign_ext_rdx = make_stmt<assign_node_t>(rdx,
                    make_xbyak_intrin(
                            dst->dtype_, {rax}, xbyak_intrin_type::sign_ext));
            transform_seq_.emplace_back(sign_ext_rdx);
            // div rhs
            // mov dst, %rax(div)/%rdx(mod)
            // Add %rax, %rdx to xbyak_intrin args so liveness updates
            auto tmp = load_when_imm(rhs, "__div_imm_tmp");
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {tmp, rax, rdx}, intrin,
                            xbyak_intrin_isa::x86));
        }
    }

    void transform_avx_div(const expr &dst, const expr &lhs, const expr &rhs,
            sc_data_type_t dtype) {
        // TODO(longsheng): refactor div transfrom before ssa
        if (rhs.isa<constant>() && rhs->dtype_.is_etype(sc_data_etype::S32)) {
            const auto const_val = rhs.static_as<constant>()->value_;
            COMPILE_ASSERT(const_val.size() == 1,
                    "AVX div by variant constant not supported.")
            const auto divisor = const_val[0].s64;
            const auto mult = invariant_int::SintDivMultiplier(divisor, 32);
            // mulsh
            const auto transform_mulh_s32 = [this, dtype](const expr &dst,
                                                    const expr &lhs,
                                                    uint64_t m) {
                // mask = (0, -1, 0, -1, ...)
                // magic = mult.magic_
                int lanes = lhs->dtype_.lanes_;
                auto dtype_64 = sc_data_type_t::index(lanes / 2);
                std::vector<union_val> val(lanes);
                for (int i = 0; i < lanes; i++) {
                    val[i] = union_val(int64_t(-(i % 2)));
                }
                auto mask_c = builder::make_constant(val, dtype);
                auto magic_c = builder::make_constant({m}, datatypes::s32);
                mask_c->attr().set(attr_keys::force_simd_encode, true);
                magic_c->attr().set(attr_keys::force_simd_encode, true);
                //
                auto magic = builder::make_var(dtype, "__magic");
                auto hi1 = builder::make_var(dtype, "__hi1");
                auto hi2 = builder::make_var(dtype, "__hi2");
                add_defination(magic, linkage::local);
                add_defination(hi1, linkage::local);
                add_defination(hi2, linkage::local);
                // magic = broadcast(magic_c)
                // hi1 = _mm512_mul_epi32(lhs, [magic]);
                // hi1 = _mm512_srli_epi64(hi1, [32]);
                // hi2 = _mm512_srli_epi64(lhs, [32]);
                // hi2 = _mm512_mul_epi32(hi2, [magic]);
                // hi2 = _mm512_and_si512(hi2, [mask]);
                // dst = _mm512_or_si512(hi1, hi2);
                add_assignment(magic,
                        make_xbyak_intrin(dtype, {magic_c},
                                xbyak_intrin_type::broadcast,
                                xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(datatypes::s32)));
                add_assignment(hi1,
                        make_xbyak_intrin(dtype, {lhs, magic},
                                xbyak_intrin_type::mulhl,
                                xbyak_intrin_isa::avx));
                add_assignment(hi1,
                        make_xbyak_intrin(dtype,
                                {hi1, builder::make_constant(32)},
                                xbyak_intrin_type::shr, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype_64)));
                add_assignment(hi2,
                        make_xbyak_intrin(dtype,
                                {lhs, builder::make_constant(32)},
                                xbyak_intrin_type::shr, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype_64)));
                add_assignment(hi2,
                        make_xbyak_intrin(dtype, {hi2, magic},
                                xbyak_intrin_type::mulhl,
                                xbyak_intrin_isa::avx));
                add_assignment(hi2,
                        make_xbyak_intrin(dtype, {hi2, mask_c},
                                xbyak_intrin_type::bit_and,
                                xbyak_intrin_isa::avx));
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {hi1, hi2},
                                xbyak_intrin_type::bit_or,
                                xbyak_intrin_isa::avx));
                // TODO(longsheng): srl+and+or can combine to vpermi2d in avx512
            };
            //
            if (mult.power_of_2_) {
                auto sft1 = builder::make_constant(mult.sft_ - 1);
                auto sft2 = builder::make_constant(32 - mult.sft_);
                auto sft3 = builder::make_constant(mult.sft_);
                // dst = sar(lhs, [l - 1])
                // dst = shr(dst, [N - l])
                // dst = dst + lhs
                // dst = sar(dst, [l])
                // dst = d < 0 ? -dst : dst
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {lhs, sft1},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, sft2},
                                xbyak_intrin_type::shr, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, lhs},
                                xbyak_intrin_type::add, xbyak_intrin_isa::avx));
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, sft3},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                if (mult.negative_) {
                    auto simd_zero = builder::make_var(dtype, "__simd_zero");
                    add_defination(simd_zero, linkage::local);
                    add_assignment(simd_zero,
                            make_xbyak_intrin(dtype, {simd_zero, simd_zero},
                                    xbyak_intrin_type::bit_xor,
                                    xbyak_intrin_isa::avx));
                    add_assignment(dst,
                            make_xbyak_intrin(dtype, {simd_zero, dst},
                                    xbyak_intrin_type::sub,
                                    xbyak_intrin_isa::avx));
                }
            } else if (mult.compensate_) {
                auto sft = builder::make_constant(mult.sft_);
                auto bit = builder::make_constant(31);
                auto xsign = builder::make_var(dtype, "__xsign");
                add_defination(xsign, linkage::local);
                // dst = mulsh(lhs, [magic])
                // dst = dst + lhs
                // dst = sar(dst, [sft])
                // xsign = sar(lhs, [N - 1])
                // dst = d < 0 ? (xsign - dst) : (dst - xsign)
                transform_mulh_s32(dst, lhs, mult.magic_);
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, lhs},
                                xbyak_intrin_type::add, xbyak_intrin_isa::avx));
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, sft},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                add_assignment(xsign,
                        make_xbyak_intrin(dtype, {lhs, bit},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                if (mult.negative_) {
                    add_assignment(dst,
                            make_xbyak_intrin(dtype, {xsign, dst},
                                    xbyak_intrin_type::sub,
                                    xbyak_intrin_isa::avx));
                } else {
                    add_assignment(dst,
                            make_xbyak_intrin(dtype, {dst, xsign},
                                    xbyak_intrin_type::sub,
                                    xbyak_intrin_isa::avx));
                }
            } else {
                auto sft = builder::make_constant(mult.sft_);
                auto bit = builder::make_constant(31);
                auto xsign = builder::make_var(dtype, "__xsign");
                add_defination(xsign, linkage::local);
                // dst = mulsh(lhs, [magic])
                // dst = sar(dst, sft)
                // xsign = sar(lhs, [N - 1])
                // dst = d < 0 ? (xsign - dst) : (dst - xsign)
                transform_mulh_s32(dst, lhs, mult.magic_);
                add_assignment(dst,
                        make_xbyak_intrin(dtype, {dst, sft},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                add_assignment(xsign,
                        make_xbyak_intrin(dtype, {lhs, bit},
                                xbyak_intrin_type::sar, xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(dtype)));
                if (mult.negative_) {
                    add_assignment(dst,
                            make_xbyak_intrin(dtype, {xsign, dst},
                                    xbyak_intrin_type::sub,
                                    xbyak_intrin_isa::avx));
                } else {
                    add_assignment(dst,
                            make_xbyak_intrin(dtype, {dst, xsign},
                                    xbyak_intrin_type::sub,
                                    xbyak_intrin_isa::avx));
                }
            }
        } else {
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {lhs, rhs},
                            xbyak_intrin_type::div, xbyak_intrin_isa::avx));
        }
    }

    void transform_simd_reduce_seq(const expr &dst, const expr &src,
            sc_data_type_t dtype, xbyak_intrin_type intrin,
            xbyak_intrin_isa isa) {
        auto lanes = dtype.lanes_;
        auto current_src = src;
        // lanes must be power of 2
        assert((lanes > 1) && ((lanes & (lanes - 1)) == 0));
        // reduce lanes down to 1
        while (lanes > 1) {
            lanes = lanes / 2;
            dtype.lanes_ = lanes;
            // hig, low
            auto extract_hig = builder::make_var(
                    dtype, "extract_hig_" + std::to_string(lanes));
            auto extract_low = builder::make_var(
                    dtype, "extract_low_" + std::to_string(lanes));
            add_defination(extract_hig, linkage::local);
            add_defination(extract_low, linkage::local);
            // hig = ext_h(src)
            add_assignment(extract_hig,
                    make_xbyak_intrin(dtype, {current_src},
                            xbyak_intrin_type::extract_high, isa,
                            xbyak_intrin_modifier(current_src->dtype_)));
            // low = ext_l(src)
            add_assignment(extract_low,
                    make_xbyak_intrin(dtype, {current_src},
                            xbyak_intrin_type::extract_low, isa,
                            xbyak_intrin_modifier(current_src->dtype_)));
            // hig = op(low, hig)
            if (lanes == 1) {
                if (convert_x86_operation(dtype)) {
                    assert(intrin == xbyak_intrin_type::add
                            || intrin == xbyak_intrin_type::mul
                            || intrin == xbyak_intrin_type::min
                            || intrin == xbyak_intrin_type::max);
                    // e.g. if reduce_add for s32x16 reach s32x1
                    // The calculation cannot use simd intrins any more
                    add_assignment(dst, extract_low);
                    add_assignment(dst,
                            make_xbyak_intrin(dst->dtype_, {extract_hig},
                                    intrin, xbyak_intrin_isa::x86));
                } else {
                    add_assignment(dst,
                            make_xbyak_intrin(dst->dtype_,
                                    {extract_low, extract_hig}, intrin, isa));
                }
            } else {
                add_assignment(extract_low,
                        make_xbyak_intrin(extract_low->dtype_,
                                {extract_low, extract_hig}, intrin, isa));
            }
            // src = hig
            current_src = extract_low;
        }
    }

    void transform_assign(const expr &dst, const expr &src,
            const expr &maybe_masked, bool zero_masked) {
        if (maybe_masked.isa<indexing>()) {
            auto mask = maybe_masked.static_as<indexing>()->mask_;
            if (mask.defined()) {
                // Must be simd move
                assert(dst->dtype_.lanes_ > 1);
                auto cond = cast_when_mask(mask, dst->dtype_);
                add_assignment(dst,
                        make_xbyak_intrin(dst->dtype_, {src},
                                xbyak_intrin_type::mask_mov,
                                xbyak_intrin_isa::avx,
                                xbyak_intrin_modifier(cond, zero_masked)));
                // return here when mask_mov, avoid add_assignment twice
                return;
            }
        }
        add_assignment(dst, src);
    }

    void transform_gather(const expr &dst, const expr &src, const expr &idx) {
        auto get_gather_mask = [this](expr mask, sc_data_type_t dst_type) {
            if (cpu_flags_.fAVX512F) {
                return cast_when_mask(std::move(mask), dst_type);
            } else {
                auto xmm0 = make_physical_reg(dst_type, x86_64::regs::xmm0);
                add_defination(xmm0, linkage::local);
                transform_avx_mask_cast(xmm0, mask);
                return xmm0;
            }
        };
        assert(dst->dtype_.lanes_ > 1);
        // get mask with all bits is 1
        uint64_t imm = (UINT64_C(1) << (dst->dtype_.lanes_)) - 1;
        auto mask = builder::make_constant({imm}, datatypes::u32);
        // get avx512 mask or avx2 mask ymm0
        auto cond = get_gather_mask(std::move(mask), dst->dtype_);
        // make sure dst and idx use different xmm reg
        auto xmm1 = make_physical_reg(idx->dtype_, x86_64::regs::xmm1);
        auto xmm2 = make_physical_reg(dst->dtype_, x86_64::regs::xmm2);
        add_defination(xmm1, linkage::local);
        add_defination(xmm2, linkage::local);
        // transform gather intrin
        add_assignment(xmm1, idx);
        add_assignment(xmm2,
                make_xbyak_intrin(dst->dtype_, {src, xmm1},
                        xbyak_intrin_type::gather, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(cond, false)));
        add_assignment(dst, xmm2);
    }

    void transform_extract(const expr &dst, const expr &src, const expr &imm,
            const expr &elem_bits) {
        add_assignment(dst,
                make_xbyak_intrin(dst->dtype_, {src, imm, elem_bits},
                        xbyak_intrin_type::extract, xbyak_intrin_isa::avx));
    }

    void transform_broadcast(
            const expr &dst, const expr &src, const sc_data_type_t &src_type) {
        // transform broadcast with type hint
        add_assignment(dst,
                make_xbyak_intrin(dst->dtype_, {src},
                        xbyak_intrin_type::broadcast, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(src_type)));
    }

    void transform_avx_mask_cast(const expr &dst, const expr &src) {
        assert(!cpu_flags_.fAVX512F && cpu_flags_.fAVX2);
        // avx mask cast can cast scalar to xmm, or xmm to scalar
        const auto is_scalar_mask = [](const expr &v) {
            auto type_lane = v->dtype_.lanes_;
            auto type_code = v->dtype_.type_code_;
            auto type_size = utils::get_sizeof_etype(type_code);
            return (type_lane == 1
                           && utils::is_one_of((int)type_size, 1, 2, 4, 8))
                    || type_code == sc_data_etype::BOOLEAN;
        };
        //
        if (is_scalar_mask(src)) {
            auto dtype = dst->dtype_;
            // TODO(longsheng): optimize imm mask
            auto tmp = load_when_imm(src, "__msk_tmp_var");
            switch (utils::get_sizeof_etype(dtype.type_code_)) {
                case 1: cast_mask8_avx2(dst, tmp, dtype); break;
                case 2: cast_mask16_avx2(dst, tmp, dtype); break;
                case 4: cast_mask32_avx2(dst, tmp, dtype); break;
                default:
                    COMPILE_ASSERT(false, "Not supported base type: " << dtype);
            }
        } else if (is_scalar_mask(dst)) {
            add_assignment(dst,
                    make_xbyak_intrin(dst->dtype_, {src}, //
                            xbyak_intrin_type::mov_mask, //
                            xbyak_intrin_isa::avx, //
                            xbyak_intrin_modifier(src->dtype_)));
            if (src->dtype_.is_etype(sc_data_etype::BF16)
                    || src->dtype_.is_etype(sc_data_etype::U16)) {
                assert(cpu_flags_.fBMI2);
                add_assignment(dst,
                        make_xbyak_intrin(dst->dtype_,
                                {dst, builder::make_constant(0x5555)},
                                xbyak_intrin_type::bmi_pext,
                                xbyak_intrin_isa::x86));
            }
        } else {
            COMPILE_ASSERT(false, "Invalid avx_mask_cast!");
        }
    }

    // --------------
    // Intrin helpers
    // --------------

    expr load_when_imm(const expr &v, const std::string &name) {
        if (v.isa<constant>()) {
            auto tmp = builder::make_var(v->dtype_, name);
            add_defination(tmp, linkage::local);
            add_assignment(tmp, v);
            return tmp;
        } else {
            return v;
        }
    }

    expr load_to_reg(const expr &v, const Xbyak::Reg &reg) {
        auto ret = make_physical_reg(v->dtype_, reg);
        add_defination(ret, linkage::local);
        add_assignment(ret, v);
        return ret;
    }

    // If simd cond is not mask type, cast to mask
    expr cast_when_mask(expr cond, sc_data_type_t dtype) {
        auto lanes = dtype.lanes_;
        if (cpu_flags_.fAVX512F
                && cond->dtype_ != sc_data_type_t::boolean(lanes)) {
            auto tmp = load_when_imm(cond, "__msk_tmp_var");
            auto mask = builder::make_var(
                    sc_data_type_t::boolean(lanes), "__mmask");
            add_defination(mask, linkage::local);
            add_assignment(mask,
                    builder::make_cast(sc_data_type_t::boolean(lanes), tmp));
            return mask;
        }
        return cond;
    }

    void cast_mask8_avx2(
            const expr &mask, const expr &src, const sc_data_type_t &dtype) {
        // vmovd        xmm0, src
        // vpbroadcastd ymm0, xmm0
        // vpshufb      ymm0, ymm0, table1
        // vpor         ymm0, ymm0, table2
        // vpcmpeqb     ymm1, ymm1, ymm1 // set all bit to 1
        // vpcmpeqb     xmm0, ymm0, ymm1
        const int mask_lanes = mask->dtype_.lanes_;
        const int table_len = mask_lanes / 4;
        assert(utils::is_one_of(mask_lanes, 8, 16, 32));
        auto mask_type = sc_data_type_t(sc_data_etype::U8, mask_lanes);
        auto movd_type = sc_data_type_t(sc_data_etype::U8, 16);
        auto tabl_type = sc_data_type_t(sc_data_etype::U32, table_len);
        // shuffle table
        const std::vector<union_val> val1
                = {UINT64_C(0x00000000), UINT64_C(0x00000000),
                        UINT64_C(0x01010101), UINT64_C(0x01010101),
                        UINT64_C(0x02020202), UINT64_C(0x02020202),
                        UINT64_C(0x03030303), UINT64_C(0x03030303)};
        const std::vector<union_val> v1(val1.begin(), val1.begin() + table_len);
        auto table1 = builder::make_constant(v1, tabl_type);
        // bit or value
        const std::vector<union_val> val2
                = {UINT64_C(0xf7fbfdfe), UINT64_C(0x7fbfdfef),
                        UINT64_C(0xf7fbfdfe), UINT64_C(0x7fbfdfef),
                        UINT64_C(0xf7fbfdfe), UINT64_C(0x7fbfdfef),
                        UINT64_C(0xf7fbfdfe), UINT64_C(0x7fbfdfef)};
        const std::vector<union_val> v2(val2.begin(), val2.begin() + table_len);
        auto table2 = builder::make_constant(v2, tabl_type);
        //
        auto xmm0 = make_physical_reg(movd_type, x86_64::regs::xmm0);
        auto ymm0 = make_physical_reg(mask_type, x86_64::regs::xmm0, "_ymm");
        auto ymm1 = make_physical_reg(mask_type, x86_64::regs::xmm1, "_ymm");
        add_defination(xmm0, linkage::local);
        add_defination(ymm0, linkage::local);
        add_defination(ymm1, linkage::local);
        //
        add_assignment(xmm0,
                make_xbyak_intrin(movd_type, {src}, //
                        xbyak_intrin_type::movd, xbyak_intrin_isa::avx));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {xmm0}, //
                        xbyak_intrin_type::broadcast, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(datatypes::u32)));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table1}, //
                        xbyak_intrin_type::pshuffle, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(mask_type)));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table2}, //
                        xbyak_intrin_type::bit_or, xbyak_intrin_isa::avx));
        add_assignment(ymm1,
                make_xbyak_intrin(mask_type, {ymm1, ymm1}, //
                        xbyak_intrin_type::cmp_set, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(xbyak_condition::eq, mask_type)));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, ymm1}, //
                        xbyak_intrin_type::cmp_set, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(xbyak_condition::eq, mask_type)));
        add_assignment(mask, ymm0);
    }

    void cast_mask16_avx2(
            const expr &mask, const expr &src, const sc_data_type_t &dtype) {
        // vmovd        xmm0, src
        // vpbroadcastw ymm0, xmm0
        // vpand        ymm0, ymm0, table1
        // vpcmpeqw     mask, ymm0, table1
        const int mask_lanes = mask->dtype_.lanes_;
        assert(utils::is_one_of(mask_lanes, 8, 16));
        auto mask_type = sc_data_type_t(sc_data_etype::U16, mask_lanes);
        auto movd_type = sc_data_type_t(sc_data_etype::U16, 8);
        // bit and value
        std::vector<union_val> val;
        for (int i = 0; i < mask_lanes; i++) {
            val.emplace_back(UINT64_C(1) << i);
        }
        auto table1 = builder::make_constant(val, mask_type);
        //
        auto xmm0 = make_physical_reg(movd_type, x86_64::regs::xmm0);
        auto ymm0 = make_physical_reg(mask_type, x86_64::regs::xmm0, "_ymm");
        add_defination(xmm0, linkage::local);
        add_defination(ymm0, linkage::local);
        //
        add_assignment(xmm0,
                make_xbyak_intrin(movd_type, {src}, //
                        xbyak_intrin_type::movd, xbyak_intrin_isa::avx));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {xmm0}, //
                        xbyak_intrin_type::broadcast, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(datatypes::u16)));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table1}, //
                        xbyak_intrin_type::bit_and, xbyak_intrin_isa::avx));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table1}, //
                        xbyak_intrin_type::cmp_set, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(xbyak_condition::eq, mask_type)));
        add_assignment(mask, ymm0);
    }

    void cast_mask32_avx2(
            const expr &mask, const expr &src, const sc_data_type_t &dtype) {
        // vmovd        xmm0, src
        // vpbroadcastd ymm0, xmm0
        // vpand        ymm0, ymm0, table1
        // vpcmpeqd     mask, ymm0, table1
        const int mask_lanes = mask->dtype_.lanes_;
        assert(utils::is_one_of(mask_lanes, 4, 8));
        auto mask_type = sc_data_type_t(sc_data_etype::U32, mask_lanes);
        auto movd_type = sc_data_type_t(sc_data_etype::U32, 4);
        // bit and value
        std::vector<union_val> val;
        for (int i = 0; i < mask_lanes; i++) {
            val.emplace_back(UINT64_C(1) << i);
        }
        auto table1 = builder::make_constant(val, mask_type);
        //
        auto xmm0 = make_physical_reg(movd_type, x86_64::regs::xmm0);
        auto ymm0 = make_physical_reg(mask_type, x86_64::regs::xmm0, "_ymm");
        add_defination(xmm0, linkage::local);
        add_defination(ymm0, linkage::local);
        //
        add_assignment(xmm0,
                make_xbyak_intrin(movd_type, {src}, //
                        xbyak_intrin_type::movd, xbyak_intrin_isa::avx));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {xmm0}, //
                        xbyak_intrin_type::broadcast, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(datatypes::u32)));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table1}, //
                        xbyak_intrin_type::bit_and, xbyak_intrin_isa::avx));
        add_assignment(ymm0,
                make_xbyak_intrin(mask_type, {ymm0, table1}, //
                        xbyak_intrin_type::cmp_set, xbyak_intrin_isa::avx,
                        xbyak_intrin_modifier(xbyak_condition::eq, mask_type)));
        add_assignment(mask, ymm0);
    }

    void add_assignment(const expr &var, const expr &value) {
        transform_seq_.emplace_back(make_stmt<assign_node_t>(var, value));
    }

    void add_evaluate(const expr &value) {
        transform_seq_.emplace_back(make_stmt<evaluate_node_t>(value));
    }

    void add_defination(
            const expr &var, linkage link, const expr &init = expr()) {
        transform_seq_.emplace_back(make_stmt<define_node_t>(var, link, init));
    }

    bool convert_x86_operation(const sc_data_type_t &dtype) {
        return !is_x86_simd(dtype);
    }

    bool is_const_zero(const expr &v) {
        if (v.isa<constant_c>()) {
            auto c = v.static_as<constant_c>();
            return (c->value_.size() == 1) && (c->value_[0].u64 == 0);
        }
        return false;
    };

private:
    const runtime::cpu_flags_t &cpu_flags_;
};

func_c x86_intrinsics_lowering_t::operator()(func_c v) {
    x86_intrinsics_lowering_impl_t x86_intrinsics_lowering(target_machine_);
    return x86_intrinsics_lowering.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
