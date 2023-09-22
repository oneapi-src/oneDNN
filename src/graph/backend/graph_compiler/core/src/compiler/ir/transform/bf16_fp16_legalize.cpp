/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <vector>
#include "../builder.hpp"
#include "bf16_fp16_legalize.hpp"
#include <compiler/ir/pass_dep_util.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(bf16_fp16_legalizer, SC_PASS_DEPENDS_ON(auto_caster),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

SC_DECL_PASS_INFO(bf16_fp16_eliminator,
        SC_PASS_DEPENDS_ON(constant_folder, bf16_fp16_legalizer, index2var),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

sc_data_type_t get_etype(
        const sc_data_type_t &dtype, const uint16_t lanes = 1) {
    return dtype.is_etype(sc_data_etype::BF16) ? sc_data_type_t::bf16(lanes)
                                               : sc_data_type_t::f16(lanes);
}

static bool check_ref_more_than_one(
        std::unordered_map<expr_c, int> &m, const expr &a) {
    auto var_it = m.find(a);
    // if the var is only used for once, e.g. assignment in reorder, we do not
    // promote it.
    return (var_it != m.end() && var_it->second > 1);
}

static bool define_can_promote(const context_ptr &ctx, const define_c &v) {
    if (v->var_.isa<var>()) {
        if (v->var_->dtype_.is_etype(sc_data_etype::BF16)) {
            return v->var_->dtype_.lanes_
                    <= ctx->get_max_vector_lanes(sc_data_etype::F32)
                    && any_map_t::fetch_or_else(
                            v->var_->attr_.get(), "can_promote_to_f32", true);
        } else if (v->var_->dtype_.is_etype(sc_data_etype::F16)) {
            COMPILE_ASSERT(ctx->machine_.cpu_flags_.fAVX512FP16
                            || ctx->machine_.cpu_flags_.fAVX512AMXFP16,
                    "current cpu does not support fp16 data type.");
            // Only amxfp16 needs to do legalization.
            return ctx->machine_.cpu_flags_.fAVX512AMXFP16
                    && v->var_->dtype_.lanes_
                    <= ctx->get_max_vector_lanes(sc_data_etype::F32)
                    && any_map_t::fetch_or_else(
                            v->var_->attr_.get(), "can_promote_to_f32", true);
        }
    }
    return false;
}

std::tuple<expr_c, expr_c> bf16_fp16_promote_impl_t::docast(
        const expr &orig_a, const expr &orig_b, bool *is_low_precision_fp) {
    auto a = dispatch(orig_a);
    auto b = dispatch(orig_b);
    *is_low_precision_fp = false;
    if (utils::is_one_of(a->dtype_.type_code_, sc_data_etype::BF16,
                sc_data_etype::F16)) {
        COMPILE_ASSERT(utils::is_one_of(b->dtype_.type_code_,
                               a->dtype_.type_code_, sc_data_etype::F32),
                "low precision floating point should be calculated with "
                "low_precision_fp/f32");
        *is_low_precision_fp = true;
    } else if (utils::is_one_of(b->dtype_.type_code_, sc_data_etype::BF16,
                       sc_data_etype::F16)) {
        COMPILE_ASSERT(utils::is_one_of(a->dtype_.type_code_,
                               b->dtype_.type_code_, sc_data_etype::F32),
                "low precision floating point should be calculated with low "
                "precision floating point");
        *is_low_precision_fp = true;
    }
    if (*is_low_precision_fp) {
        sc_data_type_t fp32ty = sc_data_type_t::f32(a->dtype_.lanes_);
        a = utils::is_one_of(a->dtype_.type_code_, sc_data_etype::BF16,
                    sc_data_etype::F16)
                ? builder::make_cast(fp32ty, a)
                : a;
        b = utils::is_one_of(b->dtype_.type_code_, sc_data_etype::BF16,
                    sc_data_etype::F16)
                ? builder::make_cast(fp32ty, b)
                : b;
    }
    return std::make_tuple(a, b);
}

expr_c bf16_fp16_promote_impl_t::visit(binary_c v) {
    expr_c a, b;
    bool is_low_precision_fp = false;
    std::tie(a, b) = docast(v->l_, v->r_, &is_low_precision_fp);
    bool changed
            = !a.ptr_same(v->l_) || !b.ptr_same(v->r_) || is_low_precision_fp;
    if (changed) {
        if (is_low_precision_fp) {
            return copy_attr(*v,
                    builder::make_cast(
                            get_etype(v->l_->dtype_, v->l_->dtype_.lanes_),
                            builder::remake_binary(a, b, v)));
        }
        return copy_attr(*v, builder::remake_binary(a, b, v));
    } else {
        return v;
    }
}

expr_c bf16_fp16_promote_impl_t::visit(cmp_c v) {
    expr_c a, b;
    bool is_low_precision_fp = false;
    std::tie(a, b) = docast(v->l_, v->r_, &is_low_precision_fp);
    bool changed
            = !a.ptr_same(v->l_) || !b.ptr_same(v->r_) || is_low_precision_fp;
    if (changed) {
        return copy_attr(*v, builder::remake_binary(a, b, v));
    } else {
        return v;
    }
}

expr_c bf16_fp16_promote_impl_t::visit(select_c v) {
    if (utils::is_one_of(v->l_->dtype_.type_code_, sc_data_etype::BF16,
                sc_data_etype::F16)
            && v->l_->dtype_.lanes_
                    > ctx_->get_max_vector_lanes(sc_data_etype::F32)) {
        return v;
    }
    expr_c a, b;
    bool is_low_precision_fp = false;
    std::tie(a, b) = docast(v->l_, v->r_, &is_low_precision_fp);
    auto cond = dispatch(v->cond_);
    bool changed = !a.ptr_same(v->l_) || !b.ptr_same(v->r_)
            || !cond.ptr_same(v->cond_) || is_low_precision_fp;
    if (changed) {
        return copy_attr(*v,
                builder::make_cast(
                        get_etype(v->l_->dtype_, v->l_->dtype_.lanes_),
                        builder::make_select(cond, a, b)));
    } else {
        return v;
    }
}

expr_c bf16_fp16_promote_impl_t::visit(intrin_call_c v) {
    std::vector<expr> args;
    bool is_low_precision_fp = false;
    bool changed = false;
    assert(v->args_.size() > 0);
    auto raw_dtype = v->args_[0]->dtype_;
    switch (v->type_) {
        case intrin_type::min:
        case intrin_type::max:
        case intrin_type::abs:
        case intrin_type::round:
        case intrin_type::floor:
        case intrin_type::ceil:
        case intrin_type::exp:
        case intrin_type::log:
        case intrin_type::erf:
        case intrin_type::sqrt:
        case intrin_type::rsqrt:
        case intrin_type::reduce_add:
        case intrin_type::reduce_mul:
        case intrin_type::reduce_max:
        case intrin_type::reduce_min:
        case intrin_type::fmadd:
        case intrin_type::isnan:
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto in = dispatch(v->args_[i]);
                changed = changed || !in.ptr_same(v->args_[i]);
                if (utils::is_one_of(in->dtype_.type_code_, sc_data_etype::BF16,
                            sc_data_etype::F16)) {
                    in = builder::make_cast(
                            sc_data_type_t::f32(in->dtype_.lanes_), in);
                    is_low_precision_fp = true;
                }
                args.emplace_back(in.remove_const());
            }
            if (is_low_precision_fp) {
                for (size_t i = 0; i < args.size(); i++) {
                    COMPILE_ASSERT(
                            !utils::is_one_of(args[i]->dtype_.type_code_,
                                    sc_data_etype::BF16, sc_data_etype::F16),
                            "All input args should be f32 from bf16 / f16.");
                }
            }
            break;
        case intrin_type::int_and:
        case intrin_type::int_or:
        case intrin_type::int_xor:
        case intrin_type::round_and_cast:
        case intrin_type::saturated_cast:
        case intrin_type::shl:
        case intrin_type::shr:
        case intrin_type::brgemm:
        case intrin_type::list_brgemm:
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto in = dispatch(v->args_[i]);
                changed = changed || !in.ptr_same(v->args_[i]);
                COMPILE_ASSERT(!in->dtype_.is_etype(sc_data_etype::BF16),
                        "Intrin type " << v->type_
                                       << " does not support bf16 args");
                args.emplace_back(in.remove_const());
            }
            break;
        case intrin_type::gather:
        case intrin_type::unpack_low:
        case intrin_type::unpack_high:
        case intrin_type::shuffle:
        case intrin_type::permute:
        case intrin_type::broadcast:
        case intrin_type::permutex2var:
        case intrin_type::permutexvar:
        case intrin_type::insert:
        case intrin_type::extract:
        case intrin_type::read_struct:
        case intrin_type::write_struct:
        case intrin_type::get_group_id:
        case intrin_type::get_group_thread_id:
        case intrin_type::prefetch:
        case intrin_type::set_thread_idle_func:
        case intrin_type::reinterpret: break;
        default:
            COMPILE_ASSERT(
                    false, "Unsupport BF16 / FP16 intrin type: " << v->type_);
    }

    changed = changed || is_low_precision_fp;
    if (changed) {
        if (is_low_precision_fp) {
            if (utils::is_one_of(v->type_, intrin_type::reduce_add,
                        intrin_type::reduce_mul, intrin_type::reduce_max,
                        intrin_type::reduce_min)) {
                return copy_attr(*v,
                        builder::make_cast(get_etype(raw_dtype),
                                builder::remake_intrin_call(v, args)));
            } else {
                return copy_attr(*v,
                        builder::make_cast(
                                get_etype(raw_dtype, raw_dtype.lanes_),
                                builder::remake_intrin_call(v, args)));
            }
        } else {
            return copy_attr(*v, builder::remake_intrin_call(v, args));
        }
    } else {
        return v;
    }
}

void bf16_fp16_elimination_analyzer_t::view(var_c v) {
    auto var_it = var_use_cnt_.find(v);
    // if the var is used in non-assignment statement, increase its valid count
    // by 1.
    if (var_it != var_use_cnt_.end()) { var_it->second++; }
}
void bf16_fp16_elimination_analyzer_t::view(assign_c v) {
    auto var_it = var_use_cnt_.find(v->var_);
    auto val_it = var_use_cnt_.find(v->value_);
    if (var_it != var_use_cnt_.end()) { var_it->second++; }
    // If value is not the bf16 / fp16 var, dispatch it.
    // If it is, hold its valid usage count.
    if (val_it == var_use_cnt_.end()) { dispatch(v->value_); }
}
void bf16_fp16_elimination_analyzer_t::view(define_c v) {
    if (define_can_promote(ctx_, v)) {
        // initial count is 0
        int count = 0;
        // if the var has initial value, increase the count by 1
        if (v->init_.defined()) { count = 1; }
        var_use_cnt_[v->var_] = count;
    }
}
void bf16_fp16_elimination_analyzer_t::view(intrin_call_c v) {
    switch (v->type_) {
        case intrin_type::unpack_low:
        case intrin_type::unpack_high:
        case intrin_type::shuffle:
        case intrin_type::permute:
        case intrin_type::broadcast:
        case intrin_type::reinterpret:
            // If an arg is not the bf16 / fp16 var, dispatch it.
            // If it is, hold its valid usage count.
            for (size_t i = 0; i < v->args_.size(); i++) {
                auto it = var_use_cnt_.find(v->args_[i]);
                if (it == var_use_cnt_.end()) { dispatch(v->args_[i]); }
            }
            break;
        default: ir_viewer_t::view(v); break;
    }
}

expr_c bf16_fp16_cast_elimination_impl_t::visit(cast_c v) {
    auto in = dispatch(v->in_);
    if (v->dtype_.is_etype(sc_data_etype::F32)) {
        if (in.isa<cast_c>()) {
            auto inin = in.static_as<cast_c>();
            if (utils::is_one_of(inin->dtype_.type_code_, sc_data_etype::BF16,
                        sc_data_etype::F16)
                    && inin->in_->dtype_.is_etype(sc_data_etype::F32)) {
                return inin->in_;
            }
        }
    }
    bool changed = !in.ptr_same(v->in_);
    if (changed) {
        return copy_attr(*v, builder::make_cast(v->dtype_, in));
    } else {
        return v;
    }
}

expr_c bf16_fp16_cast_elimination_impl_t::visit(var_c v) {
    auto it = cvt_map_.find(v);
    // If we find the bf16 / fp16 old var, we should replace it with bf16 /
    // fp16 (newv) for most ir node. If the var occurs singlely in define/assign
    // node, directly use the newv instead(processed in define/assign node).
    if (it != cvt_map_.end()) {
        return builder::make_cast(
                get_etype(v->dtype_, v->dtype_.lanes_), it->second);
    }
    return v;
}

stmt_c bf16_fp16_cast_elimination_impl_t::visit(define_c v) {
    // if the var is only used for once, e.g. assignment in reorder, we do not
    // promote it.
    if (define_can_promote(ctx_, v)
            && check_ref_more_than_one(var_use_cnt_, v->var_)) {
        expr_c newv = v->var_;
        expr_c init = v->init_;
        bool changed = false;
        if (v->linkage_ == linkage::local) {
            newv = copy_attr(*v->var_,
                    builder::make_var(
                            sc_data_type_t::f32(v->var_->dtype_.lanes_),
                            v->var_.static_as<var>()->name_));
            cvt_map_.insert({v->var_, newv});
            changed = true;
        }
        if (v->init_.defined()) {
            if (v->linkage_ == linkage::local) {
                init = builder::make_cast(
                        sc_data_type_t::f32(v->init_->dtype_.lanes_), v->init_);
            }
            init = dispatch(init);
            changed |= init.ptr_same(v->init_);
        }
        if (changed) {
            return copy_attr(*v,
                    builder::make_var_tensor_def_unattached(
                            newv, v->linkage_, init));
        }
        return v;
    }
    return ir_visitor_t::visit(std::move(v));
}

stmt_c bf16_fp16_cast_elimination_impl_t::visit(assign_c v) {
    expr_c var, val;
    auto varit = cvt_map_.find(v->var_);
    // single var directly replace
    if (varit != cvt_map_.end()) {
        var = varit->second;
    } else {
        var = dispatch(v->var_);
    }
    auto valit = cvt_map_.find(v->value_);
    if (valit != cvt_map_.end()) {
        val = valit->second;
    } else {
        val = dispatch(v->value_);
    }
    bool changed = !var.ptr_same(v->var_) || !val.ptr_same(v->value_);
    //  (v,a,b are bf16, v2,a2,b2 are f32) v = bf16(f32(a) + f32(b)) => v2 = a2
    //  + b2
    if (varit != cvt_map_.end()) {
        assert(v->var_->dtype_.is_etype(sc_data_etype::BF16)
                || v->var_->dtype_.is_etype(sc_data_etype::F16)
                || v->var_->dtype_.is_etype(sc_data_etype::U16));
        assert(var.ptr_same(varit->second));
        if (val.isa<cast_c>()) {
            val = val.static_as<cast_c>()->in_;
        } else if (!val->dtype_.is_etype(sc_data_etype::F32)) {
            val = builder::make_cast(
                    sc_data_type_t::f32(val->dtype_.lanes_), val);
        }
        changed = true;
    } else if (valit != cvt_map_.end()) {
        val = builder::make_cast(
                get_etype(v->value_->dtype_, v->value_->dtype_.lanes_), val);
        changed = true;
    }
    if (changed) {
        return copy_attr(*v, builder::make_assign_unattached(var, val));
    }
    return v;
}

stmt_c bf16_fp16_cast_elimination_impl_t::visit(returns_c v) {
    if (v->value_.isa<var>()) {
        COMPILE_ASSERT(cvt_map_.find(v->value_) == cvt_map_.end(),
                "Not support return a bf16 / fp16 local buffer now");
    }
    return ir_visitor_t::visit(v);
}

func_c bf16_fp16_legalizer_t::operator()(func_c f) {
    bf16_fp16_promote_impl_t promote_pass(ctx_);
    f = promote_pass.dispatch(f);
    return f;
}

stmt_c bf16_fp16_legalizer_t::operator()(stmt_c f) {
    bf16_fp16_promote_impl_t promote_pass(ctx_);
    f = promote_pass.dispatch(f);
    return f;
}

expr_c bf16_fp16_legalizer_t::operator()(expr_c f) {
    bf16_fp16_promote_impl_t promote_pass(ctx_);
    f = promote_pass.dispatch(f);
    return f;
}

func_c bf16_fp16_eliminator_t::operator()(func_c f) {
    if (f->attr_ && f->attr_->get_or_else(function_attrs::low_level, false)) {
        return f;
    }
    bf16_fp16_elimination_analyzer_t analyzer(ctx_);
    analyzer.dispatch(f);
    bf16_fp16_cast_elimination_impl_t pass(ctx_, analyzer.var_use_cnt_);
    f = pass.dispatch(f);
    return f;
}

stmt_c bf16_fp16_eliminator_t::operator()(stmt_c f) {
    bf16_fp16_elimination_analyzer_t analyzer(ctx_);
    analyzer.dispatch(f);
    bf16_fp16_cast_elimination_impl_t pass(ctx_, analyzer.var_use_cnt_);
    f = pass.dispatch(f);
    return f;
}

expr_c bf16_fp16_eliminator_t::operator()(expr_c f) {
    bf16_fp16_elimination_analyzer_t analyzer(ctx_);
    analyzer.dispatch(f);
    bf16_fp16_cast_elimination_impl_t pass(ctx_, analyzer.var_use_cnt_);
    f = pass.dispatch(f);
    return f;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
