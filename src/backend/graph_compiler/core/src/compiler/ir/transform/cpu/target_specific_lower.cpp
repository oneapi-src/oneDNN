/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include <string>
#include <utility>
#include <vector>

#include "target_specific_lower.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/intrinsics.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/visitor.hpp>
#include <microkernel/builtin.hpp>
#include <util/any_map.hpp>

namespace sc {

static expr gen_vec_const(uint32_t lanes, float f) {
    return make_expr<constant_node>(f, sc_data_type_t::f32(lanes));
}

static std::vector<expr> make_args_by_intrinsic(const intrin_call_c &node) {
    std::vector<expr> ret;
    int idx = 0;
    for (auto &v : node->args_) {
        ret.emplace_back(builder::make_var(
                v->dtype_, "_intrin_v" + std::to_string(idx)));
        idx += 1;
    }
    return ret;
}

static std::string get_exp_func_name(const intrin_call_c &node) {
    std::stringstream ss;
    ss << "_should_inline_exp_" << node->dtype_;
    return ss.str();
}

static expr_c create_cast_bf16_to_f32(const context_ptr &ctx, const cast_c &v) {
    builder::ir_builder_t builder;
    auto &in = v->in_;
    const auto count = make_expr<constant_node>(
            16UL, sc_data_type_t::u32(in->dtype_.lanes_));

    auto uint32_v = builder::make_cast(sc_data_type_t::u32(in->dtype_.lanes_),
            builder::make_reinterpret(
                    in, sc_data_type_t::u16(in->dtype_.lanes_)));
    auto tmpf32 = copy_attr(*v,
            builder::make_reinterpret(
                    uint32_v << count, sc_data_type_t::f32(v->dtype_.lanes_)));
    if (!v->dtype_.is_etype(sc_data_etype::F32)) {
        tmpf32 = builder::make_cast(v->dtype_, tmpf32);
    }

    return tmpf32;
}

static expr_c create_cast_f32_to_bf16(const context_ptr &ctx, const cast_c &v) {
    auto &in = v->in_;
    COMPILE_ASSERT(in->dtype_.is_etype(sc_data_etype::F32),
            "bf16 should be cast from f32.");
    const auto count = make_expr<constant_node>(
            16UL, sc_data_type_t::u32(in->dtype_.lanes_));

    if (ctx->flags_.bf16_fast_trunc_) {
        auto uint32_v = builder::make_reinterpret(
                in, sc_data_type_t::u32(in->dtype_.lanes_));
        return copy_attr(*v,
                builder::make_reinterpret(
                        builder::make_cast(
                                sc_data_type_t::u16(in->dtype_.lanes_),
                                uint32_v >> count),
                        sc_data_type_t::bf16(in->dtype_.lanes_)));
    }
    // non-fast trunc
    if (in->dtype_.lanes_ > 1) {
        COMPILE_ASSERT(ctx->machine_.device_type_ == target_machine_t::type::cpu
                        && ctx->machine_.cpu_flags_.fAVX512F,
                "bf16 only support in avx512.");
    }
    if (ctx->machine_.device_type_ == target_machine_t::type::cpu
            && ctx->machine_.cpu_flags_.fAVX512BF16) {
        /* todo: add it to instrinsic for special tag */
        return copy_attr(*v, builder::make_cast(v->dtype_, in));
    } else {
        auto uint32_v = builder::make_reinterpret(
                in, sc_data_type_t::u32(in->dtype_.lanes_));

        auto rounding_bias
                = builder::make_int_and((uint32_v >> count),
                          builder::make_constant(
                                  std::vector<union_val>(
                                          in->dtype_.lanes_, (int64_t)1),
                                  sc_data_type_t::u32(in->dtype_.lanes_)))
                + builder::make_constant(
                        std::vector<union_val>(
                                in->dtype_.lanes_, (int64_t)0x7FFF),
                        sc_data_type_t::u32(in->dtype_.lanes_));
        // reinterpret to bf16 to inference wrapper node dtype e.g.
        // select/intrin_call.
        return copy_attr(*v,
                builder::make_reinterpret(
                        builder::make_cast(
                                sc_data_type_t::u16(in->dtype_.lanes_),
                                (uint32_v + rounding_bias) >> count),
                        sc_data_type_t::bf16(in->dtype_.lanes_)));
    }
}

static std::string get_isnan_func_name(const intrin_call_c &node) {
    std::stringstream ss;
    ss << "_should_inline_isnan_" << node->dtype_;
    return ss.str();
}

static func_t create_isnan_func(const intrin_call_c &node) {
    auto type = node->dtype_;
    uint32_t elements = type.lanes_;
    auto ZERO = make_expr<constant_node>(
            INT64_C(0), sc_data_type_t::s32(elements));
    auto exponent_bits = make_expr<constant_node>(
            INT64_C(0x7F800000), sc_data_type_t::s32(elements));
    auto rm_sign_bits = make_expr<constant_node>(
            INT64_C(0x7FFFFFFF), sc_data_type_t::s32(elements));
    auto ty_epi_32 = sc_data_type_t::s32(elements);
    builder::ir_builder_t builder;
    _function_(type, the_sc_isnan_func, make_args_by_intrinsic(node)) {
        _bind_(inval_f32);
        _var_(inval, ty_epi_32);
        inval = builder::make_reinterpret(inval_f32, ty_epi_32);
        expr mask = (inval & exponent_bits) == exponent_bits;
        expr temp = builder::make_select(mask, inval & rm_sign_bits, ZERO);
        expr ret = exponent_bits < temp;
        _return_(ret);
    }
    std::string fixed_name = get_isnan_func_name(node);
    the_sc_isnan_func->name_ = fixed_name;
    the_sc_isnan_func->decl_->name_ = fixed_name;
    return the_sc_isnan_func;
}

static func_t create_exp_func(const intrin_call_c &node) {
    auto type = node->dtype_;
    uint32_t elements = type.lanes_;

    auto ZERO = gen_vec_const(elements, 0.0f);
    auto ln2 = gen_vec_const(elements, 0.693147181f);
    auto one_over_ln2 = gen_vec_const(elements, 1.442695041f);
    auto ONE_f = gen_vec_const(elements, 1.0f);
    auto ONE_i = make_expr<constant_node>(
            INT64_C(1), sc_data_type_t::s32(elements));
    auto ty_epi_32 = sc_data_type_t::s32(elements);

    builder::ir_builder_t builder;
    _function_(type, the_exp_func, make_args_by_intrinsic(node)) {
        assert(node->args_.size() == 1);
        // to avoid underflow
        _bind_(inval);
        expr l_min_mask = inval >= gen_vec_const(elements, -87.33f);
        // to avoid overflow

        _var_(a_, type);
        a_ = builder::make_min(inval, gen_vec_const(elements, 88.60f));
        // TODO(xxx): currenly clip the input if the value is larger than
        // the upper limit to prevent overflow

        // e^x = 2^k_int * e^r
        _var_(k_float, type);
        k_float = builder::make_floor(
                a_ * one_over_ln2); // k_float = floor(x / ln2)
        _var_(k_int, ty_epi_32);
        k_int = builder::make_cast(ty_epi_32, k_float); // k_int = int(k_float)

        _var_(r, type);
        r = a_ - k_float * ln2; // r = x - k_float * ln2

        expr table[7];
        table[6] = gen_vec_const(elements, 0.142857143f);
        table[5] = gen_vec_const(elements, 0.166666667f);
        table[4] = gen_vec_const(elements, 0.2f);
        table[3] = gen_vec_const(elements, 0.25f);
        table[2] = gen_vec_const(elements, 0.333333333f);
        table[1] = gen_vec_const(elements, 0.5f);
        table[0] = ONE_f;
        // Calculate e^r (Tn)

        _var_(Tn, type);
        Tn = ONE_f;
        for (auto loop = 6; loop > 0; loop--) {
            // Tn = Tn * (r / i) + 1
            Tn = builder::make_fmadd(Tn, r * table[loop - 1], ONE_f);
        }

        // 2^k_int, shift to exponent bits position
        auto const_23 = make_expr<constant_node>(
                INT64_C(23), sc_data_type_t::s32(elements));
        auto p = k_int << const_23;

        _var_(result, ty_epi_32);
        result = p + builder::make_reinterpret(Tn, ty_epi_32);
        _return_(builder::make_select(
                l_min_mask, builder::make_reinterpret(result, type), ZERO));
    }
    std::string fixed_name = get_exp_func_name(node);
    the_exp_func->name_ = fixed_name;
    the_exp_func->decl_->name_ = fixed_name;
    return the_exp_func;
}

using intrin_func_creator = func_t (*)(const intrin_call_c &node);
using intrin_func_namer = std::string (*)(const intrin_call_c &node);
using cast_func_creator = func_t (*)(context_ptr ctx, const cast_c &node);

class target_specific_lower_cpu_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    context_ptr ctx_;
    ir_module_ptr mod_;
    int var_cnt_ = 0;
    // use a var instead of complex arg expr
    std::vector<std::vector<std::pair<expr_c, expr_c>>> need_defs_;

    std::vector<expr_c> visit_need_def_args(const std::vector<expr> &args) {
        std::vector<expr_c> new_args;
        new_args.reserve(args.size());
        for (auto &arg : args) {
            if (!arg.isa<var>() && !arg.isa<tensor>() && !arg.isa<indexing>()
                    && !arg.isa<tensorptr>() && !arg.isa<constant>()) {
                expr_c cache_var = builder::make_var(arg->dtype_,
                        "_arg_cache_" + std::to_string(var_cnt_++));
                need_defs_.back().emplace_back(cache_var, arg);
                new_args.emplace_back(cache_var);
            } else {
                new_args.emplace_back(arg);
            }
        }
        return new_args;
    }

    expr_c do_lower_saturated_cast(const intrin_call_c &v) {
        assert(v->args_.size() == 1);
        const auto &inval1 = v->args_[0];
        auto intype = v->args_[0]->dtype_;
        auto outtype = v->dtype_;
        auto ths = this;
        if (mod_->ctx_->machine_.cpu_flags_.fAVX512F) {
            // the fast path for AVX512
            if (v->dtype_ == sc_data_type_t::s8(16)) {
                if (intype == sc_data_type_t::s32(16)) {
                    return v;
                } else if (intype == sc_data_type_t::f32(16)) {
                    auto real_in = builder::make_round_and_cast(
                            inval1, sc_data_type_t::s32(16));
                    return builder::make_saturated_cast(real_in, v->dtype_);
                }
            } else if (v->dtype_ == sc_data_type_t::u8(16)) {
                if (intype == sc_data_type_t::s32(16)) {
                    auto zero = make_expr<constant_node>(
                            UINT64_C(0), sc_data_type_t::s32(16));
                    return builder::make_saturated_cast(
                            builder::make_max(inval1, zero), v->dtype_);
                } else if (intype == sc_data_type_t::f32(16)) {
                    auto zero = gen_vec_const(16, 0.0f);
                    auto real_in = builder::make_max(inval1, zero);
                    real_in = builder::make_round_and_cast(
                            real_in, sc_data_type_t::s32(16));
                    return builder::make_saturated_cast(real_in, v->dtype_);
                }
            }
        }
        auto cast_s32_u8s8 = [intype, outtype](const expr_c &v) {
            int64_t max_val
                    = outtype.type_code_ == sc_data_etype::U8 ? 255 : 127;
            int64_t min_val
                    = outtype.type_code_ == sc_data_etype::U8 ? 0 : -128;
            expr val255 = make_expr<constant_node>(
                    max_val, sc_data_type_t::s32(intype.lanes_));
            expr val0 = make_expr<constant_node>(
                    min_val, sc_data_type_t::s32(intype.lanes_));
            // ret = min(v, 255)
            auto ret = builder::make_min(v, val255);
            // ret = max(ret, 0)
            ret = builder::make_max(ret, val0);
            return builder::make_cast(outtype, ret);
        };

        COMPILE_ASSERT((v->dtype_.type_code_ == sc_data_etype::S8
                               || v->dtype_.type_code_ == sc_data_etype::U8)
                        && (intype.type_code_ == sc_data_etype::S32
                                || intype.type_code_ == sc_data_etype::F32),
                "saturated_cast cannot handle: " << v << '(' << intype << "->"
                                                 << v->dtype_ << ')');
        expr_c real_in = inval1;
        if (intype.type_code_ == sc_data_etype::F32) {
            real_in = builder::make_round_and_cast(
                    inval1, sc_data_type_t::s32(intype.lanes_));
        }
        return cast_s32_u8s8(real_in);
    }

    expr_c visit(intrin_call_c v) override {
        auto ret = ir_visitor_t::visit(v);
        auto new_args
                = visit_need_def_args(ret.checked_as<intrin_call_c>()->args_);
        intrin_func_creator lower_func = nullptr;
        intrin_func_namer namer_func = nullptr;
        switch (v->type_) {
            case intrin_type::exp:
                COMPILE_ASSERT(ret->dtype_.is_etype(sc_data_etype::F32),
                        "Currently sc_exp only supports f32");
                lower_func = &create_exp_func;
                namer_func = &get_exp_func_name;
                break;
            case intrin_type::isnan:
                COMPILE_ASSERT(v->args_[0]->dtype_.is_etype(sc_data_etype::F32),
                        "Currently sc_isnan only supports f32");
                lower_func = &create_isnan_func;
                namer_func = &get_isnan_func_name;
                break;
            case intrin_type::saturated_cast:
                return do_lower_saturated_cast(ret.checked_as<intrin_call_c>());
                break;
            default: break;
        }
        if (lower_func) {
            func_t f = mod_->get_func(namer_func(v));
            if (f) {
                // if the function is found, check the signature
                const char *prompt
                        = "Bad signature found for intrinsic implementation "
                          "function";
                COMPILE_ASSERT(f->ret_type_ == v->dtype_
                                && f->params_.size() == v->args_.size(),
                        prompt << f);
                for (size_t i = 0; i < f->params_.size(); i++) {
                    COMPILE_ASSERT(f->params_[i]->dtype_ == v->args_[i]->dtype_,
                            prompt << f);
                }
            } else {
                f = lower_func(ret.checked_as<intrin_call>());
                // private function, so that hopefully it can be removed
                // after inlined
                f->attr()[function_attrs::private_] = true;
                mod_->add_func({f});
            }
            auto r = copy_attr(*ret, builder::make_call(f, new_args));
            r->attr()["inline_level"] = 2;
            return r;
        }
        return ret;
    }

    expr_c visit(cast_c v) override {
        auto ret = ir_visitor_t::visit(v).checked_as<cast_c>().remove_const();
        if (ret->in_->dtype_.is_etype(sc_data_etype::BF16)) {
            expr new_in = visit_need_def_args({ret->in_})[0].remove_const();
            if (!new_in.ptr_same(ret->in_)) {
                ret = ret->remake().static_as<cast>();
            }
            ret->in_ = new_in;
            return create_cast_bf16_to_f32(ctx_, ret);
        } else if (ret->dtype_.is_etype(sc_data_etype::BF16)) {
            expr new_in = visit_need_def_args({ret->in_})[0].remove_const();
            if (!new_in.ptr_same(ret->in_)) {
                ret = ret->remake().static_as<cast>();
            }
            ret->in_ = new_in;
            return create_cast_f32_to_bf16(ctx_, ret);
        }

        return ret;
    }

    expr_c visit(constant_c v) override {
        if (v->dtype_.is_etype(sc_data_etype::BF16)) {
            union float_caster {
                float f32;
                uint32_t u32;
            };
            std::vector<union_val> new_value;
            new_value.reserve(v->value_.size());
            for (size_t i = 0; i < v->value_.size(); i++) {
                float_caster caster;
                caster.f32 = v->value_[i].f32;
                if (ctx_->flags_.bf16_fast_trunc_) {
                    new_value.emplace_back((uint64_t)(caster.u32 >> 16));
                } else {
                    uint32_t rounding_bias
                            = ((caster.u32 >> 16) & 1) + (uint32_t)0x7FFF;
                    new_value.emplace_back(
                            (uint64_t)((caster.u32 + rounding_bias) >> 16));
                }
            }
            return copy_attr(*v, builder::make_constant(new_value, v->dtype_));
        }
        return v;
    }

    std::vector<stmt_c> insert_seq_before_;
    stmt_c visit(stmts_c v) override {
        bool changed = false;
        need_defs_.emplace_back();
        std::vector<stmt_c> seqs = std::move(insert_seq_before_);
        size_t def_sz = 0;
        for (auto &st : v->seq_) {
            auto new_st = dispatch(st);
            auto &cur_defs = need_defs_.back();
            changed |= !new_st.ptr_same(st);
            if (cur_defs.size() != def_sz) {
                assert(cur_defs.size() > def_sz);
                for (size_t i = def_sz; i < cur_defs.size(); i++) {
                    expr_c &cache_var = cur_defs[i].first;
                    seqs.emplace_back(builder::make_var_tensor_def_unattached(
                            cache_var, linkage::local, cur_defs[i].second));
                }
                def_sz = cur_defs.size();
            }
            seqs.emplace_back(new_st);
        }
        need_defs_.pop_back();
        changed |= seqs.size() != v->seq_.size();
        if (changed) {
            stmt newv = copy_attr(*v, builder::make_stmts_unattached(seqs));
            return std::move(newv);
        }
        return v;
    }

    target_specific_lower_cpu_impl_t(context_ptr ctx, const ir_module_ptr &m)
        : ctx_(std::move(ctx)), mod_(m) {}
}; // namespace sc

const_ir_module_ptr target_specific_lowering_cpu_t::operator()(
        const_ir_module_ptr m) {
    auto ret = m->copy();
    target_specific_lower_cpu_impl_t pass {ctx_, ret};
    auto &contents = ret->get_contents();
    auto sz = contents.size();
    for (size_t i = 0; i < sz; i++) {
        auto f = std::const_pointer_cast<func_base>(pass.dispatch(contents[i]));
        contents[i] = std::move(f);
    }
    return ret;
}

} // namespace sc
