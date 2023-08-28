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

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "target_specific_lower.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/intrinsics.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

extern const uint32_t sc_log_const_int_vals_1[32];
extern const uint32_t sc_log_const_int_vals_2[32];
extern const uint32_t sc_erf_const_int_vals[6][32];
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(target_specific_lowering_cpu,
        SC_PASS_DEPENDS_ON(bf16_fp16_legalizer, bf16_fp16_eliminator,
                func_inliner, dyn_tensor_transformer),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(FUNC_INLINED));

static expr gen_vec_const(uint32_t lanes, float f) {
    return make_expr<constant_node>(f, sc_data_type_t::f32(lanes));
}

static expr gen_vec_const_int(uint32_t lanes, int64_t f) {
    return make_expr<constant_node>(f, sc_data_type_t::s32(lanes));
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

static std::string get_read_struct_func_name(const intrin_call_c &node) {
    auto &name
            = node->intrin_attrs_->get<std::string>(intrin_attr::struct_name);
    auto &field = node->intrin_attrs_->get<int>(intrin_attr::struct_field);
    std::stringstream ss;
    ss << "_should_inline_read_struct_" << name << "_" << field;
    return ss.str();
}
static std::string get_write_struct_func_name(const intrin_call_c &node) {
    auto &name
            = node->intrin_attrs_->get<std::string>(intrin_attr::struct_name);
    auto &field = node->intrin_attrs_->get<int>(intrin_attr::struct_field);
    std::stringstream ss;
    ss << "_should_inline_write_struct_" << name << "_" << field;
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
    if (ctx->machine_.device_type_ == runtime::target_machine_t::type::cpu
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

static std::string get_log_func_name(const intrin_call_c &node) {
    std::stringstream ss;
    ss << "_should_inline_log_" << node->dtype_;
    return ss.str();
}

static std::string get_erf_func_name(const intrin_call_c &node) {
    std::stringstream ss;
    ss << "_should_inline_erf_" << node->dtype_;
    return ss.str();
}

class target_specific_lower_cpu_impl_t;
static func_t create_isnan_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
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

static func_t create_exp_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
    auto type = node->dtype_;
    uint32_t elements = type.lanes_;
    bool overflow_check
            = !node->attr_ || node->attr_->get_or_else("overflow_check", true);

    auto ZERO = gen_vec_const(elements, 0.0f);
    auto ln2 = gen_vec_const(elements, 0.693147181f);
    auto minus_ln2 = gen_vec_const(elements, -0.693147181f);
    auto one_over_ln2 = gen_vec_const(elements, 1.442695041f);
    auto ONE_f = gen_vec_const(elements, 1.0f);
    auto ONE_i = make_expr<constant_node>(
            INT64_C(1), sc_data_type_t::s32(elements));
    auto ty_epi_32 = sc_data_type_t::s32(elements);

    builder::ir_builder_t builder;
    _function_(type, the_exp_func, make_args_by_intrinsic(node)) {
        assert(node->args_.size() == 1);
        _bind_(inval);
        // to avoid overflow

        _var_init_(a_, type, inval);
        if (overflow_check) {
            a_ = builder::make_min(inval, gen_vec_const(elements, 88.60f));
        }
        // TODO(xxx): currenly clip the input if the value is larger than
        // the upper limit to prevent overflow

        // e^x = 2^k_int * e^r
        _var_(k_float, type);
        k_float = builder::make_floor(
                a_ * one_over_ln2); // k_float = floor(x / ln2)
        _var_(k_int, ty_epi_32);
        k_int = builder::make_cast(ty_epi_32, k_float); // k_int = int(k_float)

        _var_(r, type);
        // TODO(x): mabye utilize fnmadd, fmsub, etc. in the future
        // r = a_ - k_float * ln2;
        r = builder::make_fmadd(k_float, minus_ln2, a_);

        expr table[7];
        table[6] = gen_vec_const(elements, 0.142857143f);
        table[5] = gen_vec_const(elements, 0.166666667f);
        table[4] = gen_vec_const(elements, 0.2f);
        table[3] = gen_vec_const(elements, 0.25f);
        table[2] = gen_vec_const(elements, 0.333333333f);
        table[1] = gen_vec_const(elements, 0.5f);
        table[0] = ONE_f;
        // Calculate e^r (Tn)

        const size_t top_idx = 5;
        _var_(Tn, type);
        Tn = builder::make_fmadd(r, table[top_idx], ONE_f);
        for (auto loop = top_idx; loop > 0; loop--) {
            // Tn = Tn * (r / i) + 1
            Tn = builder::make_fmadd(Tn, r * table[loop - 1], ONE_f);
        }

        // 2^k_int, shift to exponent bits position
        auto const_23 = make_expr<constant_node>(
                INT64_C(23), sc_data_type_t::s32(elements));
        auto p = k_int << const_23;

        _var_(result, ty_epi_32);
        result = p + builder::make_reinterpret(Tn, ty_epi_32);

        // to avoid underflow
        expr l_min_mask = inval >= gen_vec_const(elements, -87.33f);
        _return_(builder::make_select(
                l_min_mask, builder::make_reinterpret(result, type), ZERO));
    }
    std::string fixed_name = get_exp_func_name(node);
    the_exp_func->name_ = fixed_name;
    the_exp_func->decl_->name_ = fixed_name;
    return the_exp_func;
}

static func_t create_log_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor);

static func_t create_erf_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor);

static size_t get_field_offset(const std::string &in, int field) {
    if (in == dyn_tsr_struct_t::name) {
        return dyn_tsr_struct_t::offsets[field];
    } else {
        COMPILE_ASSERT(false, "struct " << in << " has not been supported!");
    }
    return 0;
}

static func_t create_read_struct_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
    builder::ir_builder_t builder;
    auto name = node->intrin_attrs_->get<std::string>(intrin_attr::struct_name);
    auto field = node->intrin_attrs_->get<int>(intrin_attr::struct_field);
    auto dtype = get_dtype_from_struct_and_field(name, field);
    _function_(dtype, read_struct, {node->args_[0]}) {
        assert(node->args_.size() == 1);
        expr util_tsr = builder::make_tensor("util_tsr", {1}, dtype);
        util_tsr->attr().set(attr_keys::tsr_dont_buf_sched, true);
        util_tsr->attr().set(attr_keys::no_dead_write, true);
        util_tsr->attr().set(attr_keys::no_tensor2var, true);
        size_t offset = get_field_offset(name, field);
        builder::get_current_builder()->push_var_tensor_def(util_tsr,
                linkage::local, builder::tensor_ptr(node->args_[0], {offset}));
        expr ret = builder::make_indexing(util_tsr, 0);
        _return_(ret);
    }
    auto func_name = get_read_struct_func_name(node);
    read_struct->name_ = func_name;
    read_struct->decl_->name_ = func_name;
    return read_struct;
}

static func_t create_write_struct_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
    builder::ir_builder_t builder;
    auto name = node->intrin_attrs_->get<std::string>(intrin_attr::struct_name);
    auto field = node->intrin_attrs_->get<int>(intrin_attr::struct_field);
    auto dtype = node->args_[1]->dtype_;
    if (dtype.is_pointer()) { dtype = datatypes::pointer; }
    _function_(datatypes::void_t, write_struct, {node->args_[0]},
            {builder::make_var(dtype, "_intrin_v")}) {
        assert(node->args_.size() == 2);
        assert(node->args_[1]->dtype_ == (dtype)
                || node->args_[1]->dtype_.is_pointer());
        _bind_(dyn_tsr, inval);
        expr util_tsr = builder::make_tensor("util_tsr", {1},
                dtype.is_pointer() ? datatypes::pointer : dtype);
        util_tsr->attr().set(attr_keys::tsr_dont_buf_sched, true);
        util_tsr->attr().set(attr_keys::no_dead_write, true);
        util_tsr->attr().set(attr_keys::no_tensor2var, true);
        size_t offset = get_field_offset(name, field);
        builder::get_current_builder()->push_var_tensor_def(util_tsr,
                linkage::local, builder::tensor_ptr(dyn_tsr, {offset}));
        builder::get_current_builder()->push_assign(
                builder::make_indexing(util_tsr, 0), inval);
    }
    auto func_name = get_write_struct_func_name(node);
    write_struct->name_ = func_name;
    write_struct->decl_->name_ = func_name;
    return write_struct;
}

using intrin_func_creator = func_t (*)(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *);
using intrin_func_namer = std::string (*)(const intrin_call_c &node);

class target_specific_lower_cpu_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    context_ptr ctx_;
    ir_module_ptr mod_;
    int var_cnt_ = 0;
    int parallel_depth_ = 0;
    // use a var instead of complex arg expr
    std::vector<std::vector<std::pair<expr_c, expr_c>>> need_defs_;
    const stmts_node_t *cur_scope_ = nullptr;
    std::unordered_map<int, std::pair<expr, expr>> barrier_idle_func_args_;

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
            } else if (v->dtype_ == sc_data_type_t::s32(16)) {
                assert(intype == sc_data_type_t::f32(16));
                return builder::make_round_and_cast(
                        inval1, sc_data_type_t::s32(16));
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

    stmt_c visit(for_loop_c v) override {
        if (v->kind_ == for_type::PARALLEL) { parallel_depth_++; }
        auto ret = ir_visitor_t::visit(v);
        if (v->kind_ == for_type::PARALLEL) { parallel_depth_--; }
        return ret;
    }

    int get_barrier_id(const node_base *v) {
        int barrier_id = -1;
        if (v->attr_) {
            barrier_id = v->attr_->get_or_else("post_barrier_id", -1);
        }
        return barrier_id;
    }

    expr_c visit(call_c v) override {
        auto ret = ir_visitor_t::visit(v);
        if (v->func_ == builtin::get_barrier_arrive_func()) {
            int bar_id = get_barrier_id(v.get());
            auto itr = barrier_idle_func_args_.find(bar_id);
            if (bar_id >= 0 && itr != barrier_idle_func_args_.end()) {
                auto ret2 = ret.static_as<call_c>();
                // pass idle_func and idle_args to barrier func
                return copy_attr(*v,
                        builtin::get_barrier_arrive_func()(ret2->args_.at(0),
                                itr->second.first, itr->second.second));
            }
        }
        return ret;
    }
    expr_c visit(intrin_call_c v) override {
        auto ret = ir_visitor_t::visit(v);
        if (v->type_ == intrin_type::set_thread_idle_func) {
            int barrier_id = -1;
            if (parallel_depth_ > 0) {
                // get the barrier id of the set_thread_idle_func call, and find
                // the barrier call of the same id
                barrier_id = get_barrier_id(v.get());
                bool found_consumer = false;
                for (auto &seq : cur_scope_->seq_) {
                    if (seq.isa<evaluate>()) {
                        auto &val = seq.static_as<evaluate>()->value_;
                        if (val.isa<call>()) {
                            auto call_node = val.static_as<call>();
                            if (call_node->func_
                                    == builtin::get_barrier_arrive_func()) {
                                auto call_id = get_barrier_id(call_node.get());
                                if (call_id == barrier_id) {
                                    found_consumer = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (!found_consumer) {
                    // remove the intrinsic call
                    return 0;
                }
            }
            std::vector<stmt_c> insert_before;
            expr args_pack;

            if (v->args_.size() == 2UL) {
                args_pack = v->args_[1];
            } else {
                args_pack = builder::make_tensor(
                        std::string("__idle_args") + std::to_string(var_cnt_++),
                        {v->args_.size() - 1}, datatypes::generic);
                auto def = builder::make_var_tensor_def_unattached(args_pack);
                insert_before.emplace_back(def);
                def->attr()[attr_keys::tsr_dont_buf_sched] = true;

                for (size_t i = 1; i < v->args_.size(); i++) {
                    insert_before.emplace_back(
                            builder::make_assign_unattached(args_pack[i - 1],
                                    builder::make_cast(
                                            datatypes::generic, v->args_[i])));
                }
                insert_seq_before_ = std::move(insert_before);
            }
            if (parallel_depth_ > 0) {
                // inform the barrier call to set the idle
                // functions
                barrier_idle_func_args_[barrier_id] = {v->args_[0], args_pack};
                // remove the intrinsic call. The idle function will be passed
                // in barrier call
                return 0;
            }
            if (!runtime_config_t::get().managed_thread_pool_) { return 0; }
            return builtin::get_set_idle_func_managed_func()(
                    v->args_[0], args_pack);
        } else if (v->type_ == intrin_type::get_group_thread_id) {
            int64_t level_id
                    = get_const_as_int(v->args_[0].checked_as<constant_c>());
            COMPILE_ASSERT(level_id < 0,
                    "get_group_thread_id is used outside of nested "
                    "parallel. It must has group level -1.");
            return builtin::get_thread_id_func()();
        } else if (v->type_ == intrin_type::get_group_id) {
            throw std::runtime_error(
                    "get_group_id cannot be used outside of nested parallel.");
            return expr();
        }
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
            case intrin_type::log:
                COMPILE_ASSERT(v->args_[0]->dtype_.is_etype(sc_data_etype::F32),
                        "Currently sc_log only supports f32");
                lower_func = &create_log_func;
                namer_func = &get_log_func_name;
                break;
            case intrin_type::erf:
                COMPILE_ASSERT(v->args_[0]->dtype_.is_etype(sc_data_etype::F32),
                        "Currently erf only supports f32");
                lower_func = &create_erf_func;
                namer_func = &get_erf_func_name;
                break;
            case intrin_type::saturated_cast:
                return do_lower_saturated_cast(ret.checked_as<intrin_call_c>());
                break;
            case intrin_type::read_struct:
                COMPILE_ASSERT(v->args_[0]->dtype_.get_pointer_element()
                                == datatypes::u8,
                        "User defined struct tensor should be a u8 tensor in "
                        "IR.");
                lower_func = &create_read_struct_func;
                namer_func = &get_read_struct_func_name;
                break;
            case intrin_type::write_struct:
                COMPILE_ASSERT(v->args_[0]->dtype_.get_pointer_element()
                                == datatypes::u8,
                        "User defined struct tensor should be a u8 tensor in "
                        "IR.");
                lower_func = &create_write_struct_func;
                namer_func = &get_write_struct_func_name;
                break;

            default: break;
        }
        if (lower_func) {
            auto new_args = visit_need_def_args(
                    ret.checked_as<intrin_call_c>()->args_);
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
                    COMPILE_ASSERT(f->params_[i]->dtype_ == v->args_[i]->dtype_
                                    || (f->params_[i]->dtype_
                                                    == datatypes::pointer
                                            && v->args_[i]
                                                       ->dtype_.is_pointer()),
                            prompt << f);
                }
            } else {
                f = lower_func(mod_, ret.checked_as<intrin_call>(), this);
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

    std::vector<stmt_c> insert_seq_before_;
    stmt_c visit(stmts_c v) override {
        auto old_scope = cur_scope_;
        cur_scope_ = v.get();
        bool changed = false;
        need_defs_.emplace_back();
        std::vector<stmt_c> seqs;
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
            for (auto &insert : insert_seq_before_) {
                seqs.emplace_back(std::move(insert));
            }
            insert_seq_before_.clear();
            seqs.emplace_back(new_st);
        }
        need_defs_.pop_back();
        changed |= seqs.size() != v->seq_.size();
        if (changed) {
            stmt newv = copy_attr(*v, builder::make_stmts_unattached(seqs));
            cur_scope_ = old_scope;
            return std::move(newv);
        }
        cur_scope_ = old_scope;
        return v;
    }

    target_specific_lower_cpu_impl_t(context_ptr ctx, const ir_module_ptr &m)
        : ctx_(std::move(ctx)), mod_(m) {}
};

static func_t create_log_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
    auto type = node->dtype_;
    uint32_t elements = type.lanes_;

    auto ZERO = gen_vec_const(elements, 0.0f);
    auto ln2 = gen_vec_const(elements, 0.693147181f);
    auto ONE_f = gen_vec_const(elements, 1.0f);
    auto ONE_i = gen_vec_const_int(elements, 1);
    auto ty_epi_32 = sc_data_type_t::s32(elements);
    auto neg_inf_f
            = gen_vec_const(elements, -std::numeric_limits<float>::infinity());
    auto inf_f
            = gen_vec_const(elements, std::numeric_limits<float>::infinity());
    auto qnan_f
            = gen_vec_const(elements, std::numeric_limits<float>::quiet_NaN());

    builder::ir_builder_t builder;
    std::string log_const_table_name_1 = "log_const_table_1",
                log_const_table_name_2 = "log_const_table_2";
    auto global_log_consts_1
            = mod->get_var_def_from_symbol(log_const_table_name_1);
    expr log_table_tsr_1, log_table_tsr_2;
    if (!global_log_consts_1.defined()) {
        log_table_tsr_1 = builder::make_tensor(log_const_table_name_1, {32},
                datatypes::f32, address_space::automatic,
                std::make_shared<static_data_t>(sc_log_const_int_vals_1,
                        sizeof(sc_log_const_int_vals_1)));
        log_table_tsr_2 = builder::make_tensor(log_const_table_name_2, {32},
                datatypes::f32, address_space::automatic,
                std::make_shared<static_data_t>(sc_log_const_int_vals_2,
                        sizeof(sc_log_const_int_vals_2)));
        log_table_tsr_1->attr().set(
                attr_keys::static_global, (void *)sc_log_const_int_vals_1);
        log_table_tsr_2->attr().set(
                attr_keys::static_global, (void *)sc_log_const_int_vals_2);
        mod->add_global_var(builder::make_var_tensor_def_unattached(
                log_table_tsr_1, linkage::public_global)
                                    .checked_as<define>());
        mod->add_global_var(builder::make_var_tensor_def_unattached(
                log_table_tsr_2, linkage::public_global)
                                    .checked_as<define>());
    } else {
        log_table_tsr_1 = global_log_consts_1->var_;
        log_table_tsr_2
                = mod->get_var_def_from_symbol(log_const_table_name_2)->var_;
    }
    const int approx_bits = 5;
    const int mantissa_bits = 23;
    _function_(type, the_log_func, make_args_by_intrinsic(node)) {
        assert(node->args_.size() == 1);
        // to avoid underflow
        _bind_(inval);
        _var_init_(inval_int, ty_epi_32,
                builder::make_reinterpret(inval, ty_epi_32));
        _var_(aux1_int, ty_epi_32);
        aux1_int = ((inval_int >> gen_vec_const_int(
                             elements, mantissa_bits - approx_bits))
                & gen_vec_const_int(elements, 0x0000001f));
        _var_init_(aux2_int, ty_epi_32,
                aux1_int >> gen_vec_const_int(elements, approx_bits - 1));
        _var_init_(aux3_int, ty_epi_32,
                (inval_int >> gen_vec_const_int(elements, mantissa_bits))
                        + aux2_int);
        _var_init_(aux3_f, type, builder::make_cast(type, aux3_int));
        aux2_int = builder::make_int_xor(
                           aux2_int, gen_vec_const_int(elements, 0x0000007f))
                << gen_vec_const_int(elements, 23);
        inval_int = (inval_int & gen_vec_const_int(elements, 0x007fffff))
                | aux2_int;
        _var_init_(
                aux2_f, type, builder::make_gather(log_table_tsr_1, aux1_int));
        aux2_f = aux2_f * builder::make_reinterpret(inval_int, type) - ONE_f;
        _var_init_(poly_f, type, gen_vec_const(elements, 0.199984118f));
        poly_f = builder::make_fmadd(
                poly_f, aux2_f, gen_vec_const(elements, -0.250035613f));
        poly_f = builder::make_fmadd(
                poly_f, aux2_f, gen_vec_const(elements, 0.333333343f));
        poly_f = builder::make_fmadd(
                poly_f, aux2_f, gen_vec_const(elements, -0.5f));
        poly_f = builder::make_fmadd(poly_f, aux2_f, ONE_f) * aux2_f;
        aux2_f = builder::make_gather(log_table_tsr_2, aux1_int);
        aux2_f = builder::make_fmadd(aux3_f, ln2, aux2_f);
        // two sum algorithm
        _var_init_(res_hi, type, poly_f + aux2_f);
        _var_init_(res_lo, type, res_hi - aux2_f);
        res_lo = res_lo - poly_f;
        res_hi = res_hi + res_lo;
        res_hi = builder::make_select(inval == ZERO, neg_inf_f, res_hi);
        res_hi = builder::make_select(inval < ZERO, qnan_f, res_hi);
        res_hi = builder::make_select(inval == inf_f, inf_f, res_hi);
        auto isnan = visitor->dispatch(builder::make_isnan(inval));
        res_hi = builder::make_select(isnan, qnan_f, res_hi);
        res_hi = builder::make_select(inval == ONE_f, ZERO, res_hi);
        _return_(res_hi);
    }
    std::string fixed_name = get_log_func_name(node);
    the_log_func->name_ = fixed_name;
    the_log_func->decl_->name_ = fixed_name;
    return the_log_func;
}

const_ir_module_ptr target_specific_lowering_cpu_t::operator()(
        const_ir_module_ptr m) {
    auto ret = m->copy();
    target_specific_lower_cpu_impl_t pass {ctx_, ret};
    auto &contents = ret->get_contents();
    auto sz = contents.size();
    for (size_t i = 0; i < sz; i++) {
        auto f = std::const_pointer_cast<func_base>(pass.dispatch(contents[i]));
        contents[i] = std::move(f);
        pass.barrier_idle_func_args_.clear();
    }
    return ret;
}

static func_t create_erf_func(const ir_module_ptr &mod,
        const intrin_call_c &node, target_specific_lower_cpu_impl_t *visitor) {
    auto type = node->dtype_;
    uint32_t elements = type.lanes_;

    auto ONE_i = gen_vec_const_int(elements, 1);
    auto ONE_f = gen_vec_const(elements, 1.0f);
    expr ZERO_f = make_expr<constant_node>(0.0f, sc_data_type_t::f32(elements));
    auto half_f = gen_vec_const(elements, 0.5f);
    auto INT_23 = gen_vec_const_int(elements, 23);
    auto INT_24 = gen_vec_const_int(elements, 24);
    auto positive_mask = gen_vec_const_int(elements, 0x7fffffff);
    auto sign_mask = gen_vec_const_int(elements, 0x80000000);
    auto index_bias = gen_vec_const_int(elements, 0xc21fffff);
    union {
        float v;
        int v2;
    } caster;
    caster.v2 = 0x40b15cee;
    auto rbound = gen_vec_const(elements, caster.v);
    caster.v2 = 0x3FB504F3;
    auto sqrt_2 = gen_vec_const(elements, caster.v);
    auto ty_epi_32 = sc_data_type_t::s32(elements);

    builder::ir_builder_t builder;
    if (mod->ctx_->machine_.cpu_flags_.fAVX512F && elements >= 4) {
        std::vector<std::string> erf_const_table_names = {"erf_const_table_0",
                "erf_const_table_1", "erf_const_table_2", "erf_const_table_3",
                "erf_const_table_4", "erf_const_table_5"};
        auto global_erf_const_0
                = mod->get_var_def_from_symbol(erf_const_table_names[0]);
        std::vector<expr> erf_table_tsr_list(6);
        if (!global_erf_const_0.defined()) {
            for (int i = 0; i < 6; i++) {
                erf_table_tsr_list[i]
                        = builder::make_tensor(erf_const_table_names[i], {32},
                                datatypes::f32, address_space::automatic,
                                std::make_shared<static_data_t>(
                                        sc_erf_const_int_vals[i],
                                        sizeof(uint32_t) * 32));
                erf_table_tsr_list[i]->attr().set(attr_keys::static_global,
                        (void *)sc_erf_const_int_vals[i]);
                mod->add_global_var(builder::make_var_tensor_def_unattached(
                        erf_table_tsr_list[i], linkage::public_global)
                                            .checked_as<define>());
            }
        } else {
            erf_table_tsr_list[0] = global_erf_const_0->var_;
            for (int i = 1; i < 6; i++) {
                erf_table_tsr_list[i]
                        = mod->get_var_def_from_symbol(erf_const_table_names[i])
                                  ->var_;
            }
        }
        _function_(type, the_erf_func, make_args_by_intrinsic(node)) {
            assert(node->args_.size() == 1);
            // to avoid underflow
            _bind_(inval);
            inval = inval * sqrt_2;
            auto in_s32 = builder::make_reinterpret(inval, ty_epi_32);
            _var_init_(src_pos_s32, ty_epi_32, in_s32 & positive_mask);
            _var_init_(src_pos, inval->dtype_,
                    builder::make_reinterpret(src_pos_s32, inval->dtype_));
            _var_init_(indices, ty_epi_32,
                    builder::make_min(
                            builder::make_max((src_pos_s32 + index_bias)
                                            >> gen_vec_const_int(elements, 21),
                                    ONE_i),
                            INT_24));
            indices = builder::make_select(src_pos > rbound, INT_23, indices);
            _var_init_(pol, inval->dtype_,
                    builder::make_indexing(
                            erf_table_tsr_list[5], {0}, elements));
            pol = builder::make_permutex2var(pol, indices,
                    builder::make_indexing(
                            erf_table_tsr_list[5], {16}, elements));
            _var_(tmp, inval->dtype_);
            for (int i = 4; i >= 0; i--) {
                tmp = builder::make_indexing(
                        erf_table_tsr_list[i], {0}, elements);
                tmp = builder::make_permutex2var(tmp, indices,
                        builder::make_indexing(
                                erf_table_tsr_list[i], {16}, elements));
                pol = builder::make_fmadd(pol, src_pos, tmp);
            }
            auto tmp2 = in_s32 & sign_mask;
            pol = builder::make_reinterpret(
                    builder::make_reinterpret(pol, ty_epi_32) ^ tmp2,
                    inval->dtype_);
            _return_(pol);
        }
        std::string fixed_name = get_erf_func_name(node);
        the_erf_func->name_ = fixed_name;
        the_erf_func->decl_->name_ = fixed_name;
        return the_erf_func;
    } else {
        expr const_a1 = make_expr<constant_node>(
                0.254829592f, sc_data_type_t::f32(elements));
        expr const_a2 = make_expr<constant_node>(
                -0.284496736f, sc_data_type_t::f32(elements));
        expr const_a3 = make_expr<constant_node>(
                1.421413741f, sc_data_type_t::f32(elements));
        expr const_a4 = make_expr<constant_node>(
                -1.453152027f, sc_data_type_t::f32(elements));
        expr const_a5 = make_expr<constant_node>(
                1.061405429f, sc_data_type_t::f32(elements));
        expr const_p = make_expr<constant_node>(
                0.3275911f, sc_data_type_t::f32(elements));
        _function_(type, the_erf_func, make_args_by_intrinsic(node)) {
            assert(node->args_.size() == 1);
            // to avoid underflow
            _bind_(inval);
            _var_init_(temp, sc_data_type_t::f32(elements),
                    builder::make_abs(inval));
            _var_init_(neg_square, sc_data_type_t::f32(elements),
                    ZERO_f - inval * inval);
            _var_init_(Q, sc_data_type_t::f32(elements),
                    ZERO_f - visitor->dispatch(builder::make_exp(neg_square)));
            _var_init_(t, sc_data_type_t::f32(elements),
                    builder::make_div(
                            ONE_f, builder::make_fmadd(const_p, temp, ONE_f)));
            _var_init_(result, sc_data_type_t::f32(elements), const_a5);
            _var_init_(sign, sc_data_type_t::s32(elements),
                    builder::make_int_and(
                            builder::make_reinterpret(
                                    inval, sc_data_type_t::s32(elements)),
                            sign_mask));
            temp = builder::make_mul(Q, t);
            result = builder::make_fmadd(result, t, const_a4);
            result = builder::make_fmadd(result, t, const_a3);
            result = builder::make_fmadd(result, t, const_a2);
            result = builder::make_fmadd(result, t, const_a1);
            result = builder::make_fmadd(result, temp, ONE_f);
            result = builder::make_reinterpret(
                    builder::make_int_xor(sign,
                            builder::make_reinterpret(
                                    result, sc_data_type_t::s32(elements))),
                    sc_data_type_t::f32(elements));
            _return_(result);
        }
        std::string fixed_name = get_erf_func_name(node);
        the_erf_func->name_ = fixed_name;
        the_erf_func->decl_->name_ = fixed_name;
        return the_erf_func;
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
