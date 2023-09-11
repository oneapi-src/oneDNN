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

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>

#include "avx2_legalizer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class avx2_legalizer_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    avx2_legalizer_impl_t() = default;

    bool is_u8s8_bf16(const sc_data_type_t &dtype) {
        return dtype.type_code_ == datatypes::bf16.type_code_
                || dtype.type_code_ == datatypes::u8.type_code_
                || dtype.type_code_ == datatypes::s8.type_code_;
    }

    expr_c visit(tensor_c v) override {
        // avoid dispatch into for loop index dependent tensor
        return v;
    }

    expr_c visit(cast_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).dyn_as<cast_c>();
        assert(vv.defined());
        auto src = vv->in_;
        const auto src_dtype = src->dtype_;
        const auto dst_dtype = vv->dtype_;
        auto is_32bit_avx2_cast = [&](const sc_data_etype &cmp_dst_type,
                                          const sc_data_etype &cmp_src_type) {
            return src_dtype.lanes_ > 1
                    && (dst_dtype.type_code_ == cmp_dst_type)
                    && (src_dtype.type_code_ == cmp_src_type);
        };
        auto s32_smid_cast = [&](bool is_u16 = false) {
            expr cast_u16 = builder::make_cast(
                    sc_data_type_t::u16(src_dtype.lanes_ * 2), src);
            expr permutexvar_expr = builder::make_permutexvar(
                    builder::make_constant(UINT64_C(0b11011000)), cast_u16, 4);
            expr extract_u16_expr = builder::make_extract(
                    permutexvar_expr, 0, src_dtype.lanes_);
            if (is_u16) { return extract_u16_expr; }
            expr cast_dst = builder::make_cast(
                    sc_data_type_t(dst_dtype.type_code_, src_dtype.lanes_ * 2),
                    extract_u16_expr);
            // must cast to index, inorder to use reg64 in extract instruction
            expr reinpt_1 = builder::make_reinterpret(
                    cast_dst, sc_data_type_t(sc_data_etype::INDEX, 2));
            expr extract_64bit = builder::make_extract(reinpt_1, 0);
            expr reinpt_2 = builder::make_reinterpret(extract_64bit, dst_dtype);
            return reinpt_2;
        };
        if ((is_32bit_avx2_cast(sc_data_etype::U8, sc_data_etype::S32)
                    || is_32bit_avx2_cast(sc_data_etype::S8, sc_data_etype::S32)
                    || is_32bit_avx2_cast(sc_data_etype::S8, sc_data_etype::U32)
                    || is_32bit_avx2_cast(
                            sc_data_etype::U8, sc_data_etype::U32))) {
            return s32_smid_cast();
        } else if ((is_32bit_avx2_cast(sc_data_etype::U16, sc_data_etype::U32)
                           || is_32bit_avx2_cast(
                                   sc_data_etype::U16, sc_data_etype::S32))) {
            return s32_smid_cast(true);
        }
        return vv;
    }

    expr_c visit(intrin_call_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).static_as<intrin_call_c>();
        auto dst_dtype = vv->dtype_;
        switch (vv->type_) {
            case intrin_type::rsqrt: {
                if (dst_dtype.is_etype(sc_data_etype::F32)) {
                    // AVX2 rsqrt have low precision. Do as llvm
                    // (rcpps(sqrtps)), 1/sqrt.
                    std::vector<union_val> init_val(dst_dtype.lanes_, 1.f);
                    return builder::make_constant(init_val, dst_dtype)
                            / builder::make_sqrt(vv->args_[0]);
                }
            } break;
            case intrin_type::saturated_cast: {
                // Currently our project just need u32s32 to u8s8 or u16.
                assert(utils::is_one_of(dst_dtype.type_code_, sc_data_etype::S8,
                        sc_data_etype::U16, sc_data_etype::U8));
                assert(utils::is_one_of(vv->args_[0]->dtype_.type_code_,
                        sc_data_etype::S32, sc_data_etype::U32));
                // In avx2, cast is saturated cast.
                return builder::make_cast(dst_dtype, vv->args_[0]);
            }; break;
            default: break;
        }
        return vv;
    }

    expr_c visit(cmp_c v) override {
        auto vv = ir_visitor_t::visit(std::move(v)).dyn_as<cmp_c>();
        assert(vv.defined());
        // AVX2 uint have no cmp other than EQ
        // AVX2 sint have no cmp other than EQ/GT
        auto transform_cmp = [this](const cmp_c &v) -> expr {
            const auto src_dtype = v->l_->dtype_;
            if (src_dtype.lanes_ > 1) {
                switch (get_etype_category(src_dtype)) {
                    case type_category::CATE_UINT: {
                        return avx_uint_cmp(v);
                    } break;
                    case type_category::CATE_INT: {
                        return avx_sint_cmp(v);
                    } break;
                    case type_category::CATE_FLOAT: {
                        return avx_float_cmp(v);
                    } break;
                    default: break; // No need to transform.
                }
            }
            return v.remove_const();
        };
        //
        const auto cmp_dtype = vv->dtype_;
        return avx_mask_cast(transform_cmp(vv), cmp_dtype);
    }

    expr_c visit(select_c v) override {
        // AVX2 mask must cast to the data type it is masking
        auto vv = ir_visitor_t::visit(std::move(v)).dyn_as<select_c>();
        assert(vv.defined());
        auto dtype = vv->dtype_;
        if (dtype != vv->cond_->dtype_) {
            return builder::make_select(
                    avx_mask_cast(vv->cond_, dtype), vv->l_, vv->r_);
        } else {
            return vv;
        }
    }

    expr_c visit(indexing_c v) override {
        // AVX2 mask must cast to the data type it is masking
        auto vv = ir_visitor_t::visit(std::move(v)).dyn_as<indexing_c>();
        assert(vv.defined());
        auto dtype = vv->dtype_;
        if (vv->mask_.defined() && dtype != vv->mask_->dtype_) {
            // We process u8s8 and bf16 datatype one by one like llvm.
            if (is_u8s8_bf16(dtype)) {
                return vv;
            } else {
                return builder::make_indexing(vv->ptr_, vv->idx_, dtype.lanes_,
                        avx_mask_cast(vv->mask_, dtype));
            }
        } else {
            return vv;
        }
    }

    void assign_var_stmt(expr &target_var, expr &data, const int imm,
            const int elem_bits, indexing_c &vv,
            std::vector<stmt_c> &then_block_list,
            std::vector<expr> &tmp_offset) const {
        // insert
        if (INTRIN_TYPE == INSERT_INTRIN) {
            then_block_list.emplace_back(builder::make_assign_unattached(
                    target_var, builder::make_insert(target_var, data, imm)));
        } else if (INTRIN_TYPE == EXTRACT_INTRIN) { // extract
            then_block_list.emplace_back(builder::make_assign_unattached(
                    builder::make_indexing(vv->ptr_, tmp_offset),
                    builder::make_extract(data, imm)));
        } else {
            assert(false && "Expect insert or extract intrin type.");
        }
    }

#define PARAM(X) X
#define DEFINE_OFFSET_ITER_VAR() \
    auto offset_var = builder::make_var( \
            datatypes::index, "offset_var" + std::to_string(var_index++)); \
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(offset_var, \
            linkage::local, builder::make_constant({0UL}, datatypes::index))); \
    std::vector<expr> base = vv->idx_; \
    std::vector<expr> tmp_offset(base); \
    tmp_offset[tmp_offset.size() - 1] \
            = tmp_offset[tmp_offset.size() - 1] + offset_var; \
    expr iter_var = builder::make_var( \
            datatypes::s32, "iter_var" + std::to_string(var_index++)); \
    cur_list.emplace_back(builder::make_var_tensor_def_unattached( \
            iter_var, linkage::local, 1)); \
    auto mask = builder::make_var( \
            datatypes::s32, "mask_var" + std::to_string(var_index++)); \
    cur_list.emplace_back(builder::make_var_tensor_def_unattached(mask, \
            linkage::local, builder::make_cast(datatypes::s32, vv->mask_)));

#define INSERT_EXTRACT_DATA_TO_VAR(target_var, data, elembits, imm) \
    assign_var_stmt(target_var, data, imm, elem_bits, vv, then_block_list, \
            tmp_offset); \
    then_block = builder::make_stmts_unattached(then_block_list); \
    cur_list.emplace_back(builder::make_if_else_unattached( \
            mask >= iter_var, then_block, stmt())); \
    then_block_list.clear(); \
    cur_list.emplace_back(builder::make_assign_unattached(offset_var, \
            builder::make_add(offset_var, \
                    builder::make_constant({1UL}, datatypes::index)))); \
    cur_list.emplace_back(builder::make_assign_unattached( \
            iter_var, builder::make_add(builder::make_shl(iter_var, 1), 1)));

#define INSERT_EXTRACT_DATA_bf16(var_name, data) \
    for (size_t di = 0; di < 8; di++) { \
        INSERT_EXTRACT_DATA_TO_VAR(var_name, data, elem_bits, di) \
    }

#define INSERT_EXTRACT_DATA_u8s8(var_name, data) \
    for (size_t di = 0; di < 16; di++) { \
        INSERT_EXTRACT_DATA_TO_VAR(var_name, data, elem_bits, di) \
    }

    stmt_c insert_byte_by_byte(const expr &value, const expr &var) {
        auto dtype = var->dtype_;
        if (value.defined() && (value->node_type_ == sc_expr_type::indexing)) {
            auto vv = value.static_as<indexing_c>();
            if (vv->mask_.defined() && is_u8s8_bf16(dtype)) {
                INTRIN_TYPE = INSERT_INTRIN;
                std::vector<stmt_c> cur_list, body_list;
                std::vector<stmt_c> then_block_list;
                stmt then_block;
                expr insert_var_1, insert_var_2;

                const int lanes = dtype.lanes_;
                bool is_s8 = dtype.is_etype(sc_data_etype::S8);
                bool is_bf16 = dtype.is_etype(sc_data_etype::BF16);
                const int elem_bits = is_bf16 ? 16 : 8;
                const int type_bits = 128;
                bool need_2_xmm = is_bf16 ? lanes > 8 : lanes > 16;
                const int type_lanes = need_2_xmm ? lanes / 2 : lanes;
                sc_data_type_t insert_type = is_bf16
                        ? sc_data_type_t::bf16(type_lanes)
                        : is_s8 ? sc_data_type_t::s8(type_lanes)
                                : sc_data_type_t::u8(type_lanes);

                // original defined var
                cur_list.emplace_back(builder::make_var_tensor_def_unattached(
                        var, linkage::local,
                        builder::make_constant({0UL}, dtype)));

                DEFINE_OFFSET_ITER_VAR();

                auto data = builder::make_indexing(vv->ptr_, tmp_offset);

                auto insert_kernel = [&](expr &insert_var, expr &data) {
                    if (is_bf16) {
                        INSERT_EXTRACT_DATA_bf16(insert_var, data);
                    } else {
                        INSERT_EXTRACT_DATA_u8s8(insert_var, data);
                    }
                };

                auto make_insert_var = [&](expr &var) {
                    var = builder::make_var(insert_type,
                            "inser_var" + std::to_string(var_index++));
                    cur_list.emplace_back(
                            builder::make_var_tensor_def_unattached(var,
                                    linkage::local,
                                    builder::make_constant(
                                            {0UL}, insert_type)));
                };

                make_insert_var(insert_var_1);
                if (need_2_xmm) { make_insert_var(insert_var_2); }
                insert_kernel(insert_var_1, data);
                if (!need_2_xmm) {
                    cur_list.emplace_back(
                            builder::make_assign_unattached(var, insert_var_1));
                } else {
                    cur_list.emplace_back(builder::make_assign_unattached(
                            var, builder::make_insert(var, insert_var_1, 0)));
                    insert_kernel(insert_var_2, data);
                    cur_list.emplace_back(builder::make_assign_unattached(
                            var, builder::make_insert(var, insert_var_2, 1)));
                }

                auto cur_body = builder::make_stmts_unattached(cur_list);
                return ir_visitor_t::dispatch(std::move(cur_body));
            } else {
                return stmt();
            }
        }
        return stmt();
    }

    stmt_c extract_byte_by_byte(const expr &value, const expr &var) {
        auto dtype = value->dtype_;
        if (var.defined() && (var->node_type_ == sc_expr_type::indexing)) {
            auto vv = var.static_as<indexing_c>();
            if (vv->mask_.defined() && is_u8s8_bf16(dtype)) {
                INTRIN_TYPE = EXTRACT_INTRIN;
                const int extract_bits = 128;
                std::vector<stmt_c> cur_list, body_list;
                std::vector<stmt_c> then_block_list;
                stmt then_block;
                const int lanes = dtype.lanes_;
                bool is_s8 = dtype.is_etype(sc_data_etype::S8);
                bool is_bf16 = dtype.is_etype(sc_data_etype::BF16);
                const int elem_bits = is_bf16 ? 16 : 8;
                const int type_bits = 128;
                bool need_2_xmm = is_bf16 ? lanes > 8 : lanes > 16;
                const int type_lanes = need_2_xmm ? lanes / 2 : lanes;
                sc_data_type_t extract_type = is_bf16
                        ? sc_data_type_t::bf16(type_lanes)
                        : is_s8 ? sc_data_type_t::s8(type_lanes)
                                : sc_data_type_t::u8(type_lanes);
                expr extract_var_1, extract_var_2;

                auto make_extract_var = [&](expr &var, const int imm) {
                    var = builder::make_var(extract_type,
                            "extract_var" + std::to_string(var_index++));
                    cur_list.emplace_back(
                            builder::make_var_tensor_def_unattached(var));
                    cur_list.emplace_back(builder::make_assign_unattached(var,
                            builder::make_extract(value, imm, type_lanes)));
                };

                // We need to extract value to xmm register.
                if (need_2_xmm) {
                    make_extract_var(extract_var_1, 0);
                    make_extract_var(extract_var_2, 1);
                } else {
                    extract_var_1 = builder::make_var(extract_type,
                            "extract_var" + std::to_string(var_index++));
                    cur_list.emplace_back(
                            builder::make_var_tensor_def_unattached(
                                    extract_var_1));
                    cur_list.emplace_back(builder::make_assign_unattached(
                            extract_var_1, value));
                }

                DEFINE_OFFSET_ITER_VAR();

                if (is_bf16) {
                    INSERT_EXTRACT_DATA_bf16(extract_var_1, extract_var_1);
                    if (need_2_xmm) {
                        INSERT_EXTRACT_DATA_bf16(extract_var_2, extract_var_2);
                    }
                } else {
                    INSERT_EXTRACT_DATA_u8s8(extract_var_1, extract_var_1);
                    if (need_2_xmm) {
                        INSERT_EXTRACT_DATA_u8s8(extract_var_2, extract_var_2);
                    }
                }

                auto cur_body = builder::make_stmts_unattached(cur_list);
                return ir_visitor_t::dispatch(std::move(cur_body));
            } else {
                return stmt();
            }
        }
        return stmt();
    }

    stmt_c visit(define_c v) override {
        auto defined = ir_visitor_t::visit(std::move(v)).dyn_as<define_c>();
        assert(defined.defined());
        auto var = defined->var_;
        auto value = defined->init_;
        if (value.defined() && (value->node_type_ == sc_expr_type::indexing)) {
            auto res = insert_byte_by_byte(value, var);
            return res.defined() ? ir_visitor_t::dispatch(res) : defined;
        }
        return defined;
    }

    stmt_c visit(assign_c v) override {
        auto assign = ir_visitor_t::visit(std::move(v)).dyn_as<assign_c>();
        assert(assign.defined());
        auto var = assign->var_;
        auto value = assign->value_;
        if (var.defined() && (var->node_type_ == sc_expr_type::indexing)) {
            auto res = extract_byte_by_byte(value, var);
            return res.defined() ? ir_visitor_t::dispatch(res) : assign;
        } else if (value.defined()
                && value->node_type_ == sc_expr_type::indexing) {
            auto res = insert_byte_by_byte(value, var);
            return res.defined() ? ir_visitor_t::dispatch(res) : assign;
        }
        return assign;
    }

protected:
    // repeat var index
    uint32_t var_index = 1;
    const int INSERT_INTRIN = 0;
    const int EXTRACT_INTRIN = 1;
    int INTRIN_TYPE = -1;

    // Cast AVX2 mask
    expr avx_mask_cast(const expr &v, sc_data_type_t dtype) {
        auto nested = v.cast<low_level_intrin_c>().filter(
                [dtype](const low_level_intrin_c &v) {
                    return v->kind_ == low_level_intrin_kind::x86_general
                            && v->type_ == x86_intrin_type::avx_mask_cast
                            && v->args_[0]->dtype_ == dtype;
                });
        if (nested.has_value()) {
            // eliminate unnecessary nested mask cast
            return nested.get()->args_[0];
        } else if (dtype.lanes_ > 1 || v->dtype_.lanes_ > 1) {
            return builder::make_x86_intrin(
                    x86_intrin_type::avx_mask_cast, {v}, {{"dtype", dtype}});
        } else {
            return v;
        }
    }

    // Transform for AVX2 uint compare
    expr avx_uint_cmp(const cmp_c &v) {
        using sc_etype = sc_expr_type;
        const auto &t = v->node_type_;
        const auto &l = v->l_;
        const auto &r = v->r_;
        switch (t) {
            case sc_etype::cmp_eq: return transform_uint_eq(l, r);
            case sc_etype::cmp_lt: return transform_uint_lt(l, r);
            case sc_etype::cmp_le: return transform_uint_le(l, r);
            case sc_etype::cmp_ne: return transform_uint_ne(l, r);
            case sc_etype::cmp_ge: return transform_uint_ge(l, r);
            case sc_etype::cmp_gt: return transform_uint_gt(l, r);
            default: COMPILE_ASSERT(false, "Invalid compare type: " << t);
        }
        return v.remove_const();
    }
    expr transform_uint_eq(const expr &l, const expr &r) {
        auto code = static_cast<uint64_t>(xbyak_condition::eq);
        return builder::make_x86_intrin(x86_intrin_type::avx_compare,
                {l, r, builder::make_constant(code)});
    }
    expr transform_uint_ne(const expr &l, const expr &r) {
        auto ones = builder::make_constant({INT64_C(-1)}, l->dtype_);
        return builder::make_int_xor(transform_uint_eq(l, r), ones);
    }
    expr transform_uint_ge(const expr &l, const expr &r) {
        return transform_uint_eq(builder::make_max(l, r), l);
    }
    expr transform_uint_le(const expr &l, const expr &r) {
        return transform_uint_ge(r, l);
    }
    expr transform_uint_gt(const expr &l, const expr &r) {
        auto ones = builder::make_constant({INT64_C(-1)}, l->dtype_);
        return builder::make_int_xor(transform_uint_le(l, r), ones);
    }
    expr transform_uint_lt(const expr &l, const expr &r) {
        return transform_uint_gt(r, l);
    }

    // Transform for AVX2 sint compare
    expr avx_sint_cmp(const cmp_c &v) {
        using sc_etype = sc_expr_type;
        const auto &t = v->node_type_;
        const auto &l = v->l_;
        const auto &r = v->r_;
        switch (t) {
            case sc_etype::cmp_eq: return transform_sint_eq(l, r);
            case sc_etype::cmp_lt: return transform_sint_lt(l, r);
            case sc_etype::cmp_le: return transform_sint_le(l, r);
            case sc_etype::cmp_ne: return transform_sint_ne(l, r);
            case sc_etype::cmp_ge: return transform_sint_ge(l, r);
            case sc_etype::cmp_gt: return transform_sint_gt(l, r);
            default: COMPILE_ASSERT(false, "Invalid compare type: " << t);
        }
        return v.remove_const();
    }
    expr transform_sint_eq(const expr &l, const expr &r) {
        auto code = static_cast<uint64_t>(xbyak_condition::eq);
        return builder::make_x86_intrin(x86_intrin_type::avx_compare,
                {l, r, builder::make_constant(code)});
    }
    expr transform_sint_ne(const expr &l, const expr &r) {
        auto ones = builder::make_constant({INT64_C(-1)}, l->dtype_);
        return builder::make_int_xor(transform_sint_eq(l, r), ones);
    }
    expr transform_sint_ge(const expr &l, const expr &r) {
        return transform_sint_eq(builder::make_max(l, r), l);
    }
    expr transform_sint_le(const expr &l, const expr &r) {
        return transform_sint_ge(r, l);
    }
    expr transform_sint_gt(const expr &l, const expr &r) {
        auto code = static_cast<uint64_t>(xbyak_condition::gt);
        return builder::make_x86_intrin(x86_intrin_type::avx_compare,
                {l, r, builder::make_constant(code)});
    }
    expr transform_sint_lt(const expr &l, const expr &r) {
        return transform_sint_gt(r, l);
    }

    // Transform for AVX2 float compare
    expr avx_float_cmp(const cmp_c &v) {
        auto code = static_cast<uint64_t>(get_xbyak_condition(v->node_type_));
        return builder::make_x86_intrin(x86_intrin_type::avx_compare,
                {v->l_, v->r_, builder::make_constant(code)});
    }
};

func_c avx2_legalizer_t::operator()(func_c v) {
    // No need for AVX2 legalization when AVX512 is available
    if (target_machine_.cpu_flags_.fAVX512F) { return v; }
    avx2_legalizer_impl_t avx2_legalizer;
    return avx2_legalizer.dispatch(std::move(v));
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
