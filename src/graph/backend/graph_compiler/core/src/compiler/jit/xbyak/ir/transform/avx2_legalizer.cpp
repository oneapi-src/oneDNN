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
            return builder::make_indexing(vv->ptr_, vv->idx_, dtype.lanes_,
                    avx_mask_cast(vv->mask_, dtype));
        } else {
            return vv;
        }
    }

protected:
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
