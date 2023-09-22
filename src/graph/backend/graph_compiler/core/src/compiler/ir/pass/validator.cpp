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
#include "validator.hpp"
#include <string>
#include <utility>
#include <vector>
#include "../intrinsics.hpp"
#include "../viewer.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/tir_pos_trace.hpp>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(validator,
        SC_PASS_DEPENDS_ON(dyn_tensor_transformer, interface_generalizer,
                tensor_shrinker, index_flattener, auto_caster),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

class validate_impl_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    const func_base *cur_func_ = nullptr;
    using var_scope = std::unordered_set<const expr_base *>;
    std::vector<var_scope> defined_vars_;
    bool allow_tensor_view_ = false;
    int for_loop_levels_ = 0;
    tir_pos_tracer pass_error_tracer_;

    bool is_var_defined(const expr_base *v) {
        for (auto &scope : defined_vars_) {
            if (scope.find(v) != scope.end()) { return true; }
        }
        return false;
    }

    void add_def(const expr_base *v, const stmt_c &s) {
        assert(!defined_vars_.empty());
        auto &cur_scope = defined_vars_.back();
        COMPILE_ASSERT_POS(!is_var_defined(v),
                "The variable/tensor "
                        << v
                        << " is already defined. The second definition is: "
                        << s);
        defined_vars_.back().insert(v);
    }

    expr_c dispatch(expr_c v) override {
        TIR_ERROR_TRACE(v);
        ir_viewer_t::dispatch(v);
        return v;
    }

    stmt_c dispatch(stmt_c v) override {
        TIR_ERROR_TRACE(v);
        ir_viewer_t::dispatch(v);
        return v;
    }

    func_c dispatch(func_c v) override;
    void view(binary_c v) override;
    void view(cmp_c v) override;
    void view(logic_c v) override;
    void view(logic_not_c v) override;
    void view(select_c v) override;
    void view(indexing_c v) override;
    void view(call_c v) override;
    void view(tensor_c v) override;
    void view(tensorptr_c v) override;
    void view(intrin_call_c v) override;
    void view(var_c v) override;

    void view(assign_c v) override;
    void view(if_else_c v) override;
    void view(for_loop_c v) override;
    void view(returns_c v) override;
    void view(define_c v) override;
    void view(stmts_c v) override;

    void validate_type(sc_data_type_t dtype, const expr_c &e) const;
    void validate_type(const expr_c &e) const;
    void validate_binary_op(
            const expr_c &v, const expr_c &l, const expr_c &r) const;
    void check_indexing(const std::vector<expr> &idxvec, const expr_c &v,
            unsigned expected_dims) const;
    void check_var_tensor_def(
            const define_c &v, bool allow_tensor_view, bool is_global) const;
};

func_c validate_impl_t::dispatch(func_c v) {
    TIR_ERROR_TRACE(v);
    cur_func_ = v.get();
    defined_vars_.emplace_back(var_scope());
    allow_tensor_view_ = cur_func_->attr_
            && (cur_func_->attr_->get_or_else("allow_tensor_view", false)
                    || cur_func_->attr_->get_or_else(
                            function_attrs::low_level, false));
    for (auto &p : v->params_) {
        defined_vars_.back().insert(p.get());
    }
    auto ret = ir_viewer_t::dispatch(std::move(v));
    allow_tensor_view_ = false;
    cur_func_ = nullptr;
    defined_vars_.pop_back();
    return ret;
}

void validate_impl_t::validate_type(
        sc_data_type_t dtype, const expr_c &e) const {
    COMPILE_ASSERT_POS(dtype != datatypes::undef && dtype != datatypes::void_t
                    && dtype.lanes_ > 0,
            "Invalid type: met undef/void/zero-length vector: " << e);
}

void validate_impl_t::validate_type(const expr_c &e) const {
    validate_type(e->dtype_, e);
}

void validate_impl_t::validate_binary_op(
        const expr_c &v, const expr_c &l, const expr_c &r) const {
    validate_type(v);
    COMPILE_ASSERT_POS(l->dtype_ == r->dtype_,
            "The types of LHS and RHS should be the same: "
                    << l->dtype_ << " v.s. " << r->dtype_ << ", expr = " << v);
    COMPILE_ASSERT_POS(r->dtype_ != datatypes::pointer,
            "Do not support binary op on pointers " << v);
}

static bool validate_brgemm_dtype(sc_data_type_t dtype_A,
        sc_data_type_t dtype_B, sc_data_type_t dtype_C, bool has_postop) {
    if ((dtype_A == datatypes::f32 && dtype_B == datatypes::f32
                && dtype_C == datatypes::f32)
            || (utils::is_one_of(dtype_A, datatypes::u8, datatypes::s8)
                    && utils::is_one_of(dtype_B, datatypes::u8, datatypes::s8)
                    && (has_postop ? utils::is_one_of(dtype_C, datatypes::s32,
                                datatypes::f32, datatypes::u8, datatypes::s8)
                                   : dtype_C == datatypes::s32))
            || (dtype_A == datatypes::bf16 && dtype_B == datatypes::bf16
                    && (has_postop ? utils::is_one_of(
                                dtype_C, datatypes::f32, datatypes::bf16)
                                   : dtype_C == datatypes::f32))
            || (dtype_A == datatypes::f16 && dtype_B == datatypes::f16
                    && (has_postop ? utils::is_one_of(
                                dtype_C, datatypes::f32, datatypes::f16)
                                   : dtype_C == datatypes::f32))) {
        return true;
    }
    return false;
}

void validate_impl_t::view(binary_c v) {
    dispatch(v->l_);
    dispatch(v->r_);
    validate_binary_op(v, v->l_, v->r_);
    if (v->node_type_ == sc_expr_type::mod) {
        auto dtype = v->l_->dtype_;
        COMPILE_ASSERT_POS(dtype == datatypes::index || dtype == datatypes::s32
                        || dtype == datatypes::u8 || dtype == datatypes::s8,
                "%% operator cannot be applied on this type: " << v);
    }
}
void validate_impl_t::view(cmp_c v) {
    dispatch(v->l_);
    dispatch(v->r_);
    COMPILE_ASSERT_POS(v->dtype_.is_etype(sc_data_etype::BOOLEAN),
            "The type of cmp should be boolean, got: "
                    << v->dtype_ << ". The expr is " << v);
    COMPILE_ASSERT_POS(v->l_->dtype_ == v->r_->dtype_,
            "The type of LHS and RHS should be the same: "
                    << v->l_->dtype_ << " v.s. " << v->r_->dtype_
                    << ". expr = " << v);
    COMPILE_ASSERT_POS(get_etype_category_nothrow(v->l_->dtype_) != CATE_OTHER,
            "comparison expressions should have valid type, got type: "
                    << v->l_->dtype_ << ", expr = " << v);
}
void validate_impl_t::view(logic_c v) {
    dispatch(v->l_);
    dispatch(v->r_);
    COMPILE_ASSERT_POS(v->dtype_ == datatypes::boolean,
            "The type of logic should be boolean, got: "
                    << v->dtype_ << ". The expr is " << v);
    COMPILE_ASSERT_POS(v->l_->dtype_ == datatypes::boolean,
            "The type of LHS should be a boolean expr: " << v);
    COMPILE_ASSERT_POS(v->r_->dtype_ == datatypes::boolean,
            "The type of RHS should be a boolean expr: " << v);
}
void validate_impl_t::view(logic_not_c v) {
    dispatch(v->in_);
    COMPILE_ASSERT_POS(v->dtype_ == datatypes::boolean,
            "The type of logic not should be boolean, got: "
                    << v->dtype_ << ". The expr is " << v);
    COMPILE_ASSERT_POS(v->in_->dtype_ == datatypes::boolean,
            "The type of in_ should be a boolean expr: " << v);
}

void validate_impl_t::view(select_c v) {
    dispatch(v->cond_);
    dispatch(v->l_);
    dispatch(v->r_);
    COMPILE_ASSERT_POS(v->l_->dtype_ == v->r_->dtype_,
            "The two candidates in select should have same dtype, got: "
                    << v->l_->dtype_ << " v.s. " << v->r_->dtype_);
    auto &cond_dtype = v->cond_->dtype_;
    if (cond_dtype.lanes_ == 1
            && !cond_dtype.is_etype(sc_data_etype::BOOLEAN)) {
        uint64_t candidate_lanes = static_cast<uint64_t>(v->l_->dtype_.lanes_);
        uint64_t cond_bits = utils::get_sizeof_type(cond_dtype) * 8;
        COMPILE_ASSERT_POS(candidate_lanes == cond_bits,
                "When condition is bit mask, its number of bit should equal to "
                "number of left/right hand vector, got: "
                        << candidate_lanes << " v.s. " << cond_bits);
    }
}

void validate_impl_t::check_indexing(const std::vector<expr> &idxvec,
        const expr_c &v, unsigned expected_dims) const {
    COMPILE_ASSERT_POS(idxvec.size() == expected_dims,
            "Indexing node should have the same dimemsion of the tensor "
            "element, expecting "
                    << expected_dims << ", got " << idxvec.size()
                    << ". expr = " << v);
    int cnt = 0;
    sc_data_type_t idxtype = idxvec.front()->dtype_;
    for (auto &idx : idxvec) {
        cnt += 1;
        auto cate = get_type_category_nothrow(idx->dtype_);
        COMPILE_ASSERT_POS(idx->dtype_ != datatypes::boolean
                        && (cate == CATE_INT || cate == CATE_UINT),
                "The " << cnt << "-th index of the indexing expr has type "
                       << idx->dtype_ << ". Expecting an integer: " << v);
        COMPILE_ASSERT_POS(idx->dtype_ == idxtype,
                "Expecting all the indices within the indexing expression "
                "having the same dtype. Current dimemsion: "
                        << cnt << ". expr = " << v);
    }
}

void validate_impl_t::view(indexing_c v) {
    validate_type(v);
    dispatch(v->ptr_);
    for (auto &idx : v->idx_) {
        dispatch(idx);
    }
    sc_data_etype elem_type;
    if (v->ptr_.isa<tensorptr>()) {
        auto ptr = v->ptr_.static_as<tensorptr_c>();
        COMPILE_ASSERT_POS(!ptr->shape_.empty(),
                "Tensorptr expects a shape when used in indexing: " << v);
        check_indexing(v->idx_, v, ptr->shape_.size());
    } else {
        COMPILE_ASSERT_POS(v->ptr_.isa<tensor>(),
                "Indexing node is expecting a tensor/tensorptr as the ptr: "
                        << v);
        tensor tnode = v->ptr_.static_as<tensor>();
        COMPILE_ASSERT_POS(
                v->dtype_.type_code_ == tnode->elem_dtype_.type_code_,
                "Indexing node should have the same type of the tensor "
                "element, "
                "got " << v->dtype_
                       << " and " << tnode->elem_dtype_ << ". expr = " << v);
        check_indexing(v->idx_, v, tnode->dims_.size());
    }
}
void validate_impl_t::view(call_c v) {
    for (auto &arg : v->args_) {
        dispatch(arg);
    }
    func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
    func_t proto_func;
    if (!the_func) {
        auto the_expr = std::dynamic_pointer_cast<expr_base>(v->func_);
        COMPILE_ASSERT_POS(the_expr, "Expecting expr or func in call node");
        proto_func = the_expr->attr().get_or_else("prototype", func_t());
        COMPILE_ASSERT_POS(proto_func,
                "Expecting attr prototype in the expr of call node");
        COMPILE_ASSERT_POS(the_expr->dtype_ == datatypes::pointer,
                "Expecting the callee to be a pointer typed value");
    } else {
        proto_func = the_func;
    }
    COMPILE_ASSERT_POS(v->dtype_ != datatypes::undef, "Met undef. " << v);
    COMPILE_ASSERT_POS(v->args_.size() == proto_func->params_.size(),
            "Wrong number of parameters, given "
                    << v->args_.size() << ", expecting "
                    << proto_func->params_.size() << ". Expr = " << v);
    for (size_t i = 0; i < v->args_.size(); i++) {
        sc_data_type_t ty1 = v->args_.at(i)->dtype_;
        sc_data_type_t ty2 = proto_func->params_.at(i)->dtype_;
        COMPILE_ASSERT_POS(ty1 == ty2,
                "Unmatched types for parameter " << i + 1 << " : given " << ty1
                                                 << ", expecting " << ty2
                                                 << ". Expr = " << v);
    }
    COMPILE_ASSERT_POS(v->dtype_ == proto_func->ret_type_,
            "Unmatched types of call node and the func_t: " << v);
}

void validate_impl_t::view(intrin_call_c v) {
    for (auto &arg : v->args_) {
        dispatch(arg);
    }
    switch (v->type_) {
        case intrin_type::max:
        case intrin_type::min:
        case intrin_type::unpack_low:
        case intrin_type::unpack_high:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "Binary intrinsics take two parameters. Got " << v);
            validate_binary_op(v, v->args_[0], v->args_[1]);
            break;
        case intrin_type::int_and:
        case intrin_type::int_or:
        case intrin_type::int_xor:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "Binary intrinsics take two parameters. Got " << v);
            validate_binary_op(v, v->args_[0], v->args_[1]);
            type_category cate;
            cate = get_etype_category_nothrow(v->args_[0]->dtype_);
            COMPILE_ASSERT_POS(cate == CATE_INT || cate == CATE_UINT,
                    "int_and and int_or only supports ints, got " << v);
            break;
        case intrin_type::abs:
        case intrin_type::round:
        case intrin_type::floor:
        case intrin_type::ceil:
        case intrin_type::exp:
        case intrin_type::log:
        case intrin_type::erf:
        case intrin_type::sqrt:
        case intrin_type::rsqrt:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "abs/ round/ floor/ ceil/ exp/ sqrt/ rsqrt expects 1 "
                    "argument, "
                    "got " << v);
            COMPILE_ASSERT_POS(v->args_[0]->dtype_ == v->dtype_,
                    "abs/ round/ floor/ ceil/ exp/ sqrt/ rsqrt node should "
                    "have the "
                    "same type "
                    "of its argument");
            break;
        case intrin_type::reduce_add:
        case intrin_type::reduce_mul:
        case intrin_type::reduce_max:
        case intrin_type::reduce_min:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "reduce expects 1 argument, "
                    "got " << v);
            break;
        case intrin_type::fmadd:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 3,
                    "Trinary intrinsics take three parameters. Got " << v);
            COMPILE_ASSERT_POS(v->args_[0]->dtype_ == v->args_[1]->dtype_
                            && v->args_[0]->dtype_ == v->args_[2]->dtype_,
                    "The types of three args should be the same: "
                            << v->args_[0]->dtype_ << " v.s. "
                            << v->args_[1]->dtype_ << " v.s. "
                            << v->args_[2]->dtype_ << ", expr = " << v);
            break;
        case intrin_type::shuffle:
        case intrin_type::permute:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "Shuffle/permute intrinsics take two parameters. Got "
                            << v);
            COMPILE_ASSERT_POS(v->args_[0]->dtype_ == v->args_[1]->dtype_
                            && (v->intrin_attrs_->get_or_else("shuffle_imm", -1)
                                            != -1
                                    || v->intrin_attrs_->get_or_else(
                                               "permute_imm", -1)
                                            != -1),
                    "The types of the first two args should be the same and "
                    "the third arg should be valid imm: "
                            << v->args_[0]->dtype_ << " v.s. "
                            << v->args_[1]->dtype_ << ", expr = " << v);
            break;
        case intrin_type::permutex2var:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 3,
                    "Trinary intrinsics take three parameters. Got " << v);
            COMPILE_ASSERT_POS(v->args_[0]->dtype_ == v->args_[2]->dtype_,
                    "The types of the first and last args should be the same"
                            << v->args_[0]->dtype_ << " v.s. "
                            << v->args_[2]->dtype_);

            break;
        case intrin_type::permutexvar:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "Trinary intrinsics take two parameters. Got " << v);
            break;
        case intrin_type::insert:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "The intrinsics take two parameters. Got " << v);
            COMPILE_ASSERT_POS(
                    v->intrin_attrs_->get_or_null<int>("insert_imm") != nullptr,
                    "Must specify intrin_attrs_.");
            break;
        case intrin_type::extract:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "The intrinsics take one parameter. Got " << v);
            COMPILE_ASSERT_POS(v->intrin_attrs_->get_or_null<int>("extract_imm")
                            != nullptr,
                    "Must specify intrin_attrs_.");
            break;
        case intrin_type::gather:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "Gather intrinsics take two parameters. Got " << v);
            COMPILE_ASSERT_POS(v->args_[0]->dtype_.is_pointer()
                            && v->args_[0]
                                       ->dtype_.get_pointer_element()
                                       .is_etype(sc_data_etype::F32),
                    "The first type of gather should be a pointer to f32. Got "
                            << v);
            COMPILE_ASSERT_POS(v->args_[1]->dtype_.is_etype(sc_data_etype::S32),
                    "The second type of gather should be s32. Got " << v);
            break;
        case intrin_type::broadcast:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "broadcast expects 1 argument, got " << v);
            break;
        case intrin_type::reinterpret:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "reinterpret expects 1 argument, got " << v);
            break;
        case intrin_type::isnan:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "isnan expects 1 argument, got " << v);
            COMPILE_ASSERT_POS(v->dtype_.is_etype(sc_data_etype::F32)
                            || v->dtype_.is_etype(sc_data_etype::U16),
                    "isnan node should receive float or bfloat16 type.");
            COMPILE_ASSERT_POS(utils::get_sizeof_type(v->args_[0]->dtype_)
                            == utils::get_sizeof_type(v->dtype_),
                    "isnan node should have types of same size, got " << v);
            break;
        case intrin_type::saturated_cast:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 1,
                    "isnan expects 1 argument, got " << v);
            break;
        case intrin_type::shl:
        case intrin_type::shr:
            validate_type(v);
            COMPILE_ASSERT_POS(v->args_.size() == 2,
                    "shl shr expects 2 argument, got " << v);
            COMPILE_ASSERT_POS(
                    (v->args_[0]->dtype_.lanes_ == v->args_[1]->dtype_.lanes_)
                            || v->args_[1]->dtype_.lanes_ == 1,
                    "shl shr does not support A << B or A >> B where A is a "
                    "scalar and B is a vector ");
            type_category cate1, cate2;
            cate1 = get_etype_category_nothrow(v->args_[0]->dtype_);
            cate2 = get_etype_category_nothrow(v->args_[1]->dtype_);
            COMPILE_ASSERT_POS((cate1 == CATE_INT || cate1 == CATE_UINT)
                            && (cate2 == CATE_INT || cate2 == CATE_UINT),
                    "operands of shl and shr should be ints, got " << v);
            break;
        case intrin_type::brgemm: {
            COMPILE_ASSERT_POS(v->dtype_ == datatypes::void_t,
                    "brgemm should return void");
            COMPILE_ASSERT_POS(
                    v->check_brgemm_arg_size(brgemm_args::NUM_FULL_ARGS_STRIDE),
                    "Wrong number of arguments for brgemm");
            auto &extras = v->intrin_attrs_->get<brgemm_args::extra_args_t>(
                    intrin_attr::brgemm_extras);
            if (extras.is_cpu_) {
                COMPILE_ASSERT_POS(validate_brgemm_dtype(extras.dtype_A_,
                                           extras.dtype_B_, extras.dtype_C_,
                                           !extras.postops_setting_.empty()),
                        "BRGEMM currently only support f32, got: " << v);
            } else {
                COMPILE_ASSERT_POS(extras.dtype_A_ == datatypes::bf16
                                && extras.dtype_B_ == datatypes::bf16
                                && extras.dtype_C_ == datatypes::bf16,
                        "BRGEMM currently only support bf16, got: " << v);
            }
            auto tsr_dtype_A = extras.dtype_A_.get_pointerof();
            auto tsr_dtype_B = extras.dtype_B_.get_pointerof();
            auto tsr_dtype_C = extras.dtype_C_.get_pointerof();
            COMPILE_ASSERT_POS(v->args_[0]->dtype_ == tsr_dtype_A,
                    "The " << 0
                           << "-th argument of brgemm does not match. "
                              "Expecting"
                           << tsr_dtype_A << " Got" << v->args_[0]->dtype_);
            COMPILE_ASSERT_POS(v->args_[1]->dtype_ == tsr_dtype_B,
                    "The " << 1
                           << "-th argument of brgemm does not match. "
                              "Expecting"
                           << tsr_dtype_B << " Got" << v->args_[1]->dtype_);
            COMPILE_ASSERT_POS(v->args_[2]->dtype_ == tsr_dtype_C,
                    "The " << 2
                           << "-th argument of brgemm does not match. "
                              "Expecting"
                           << tsr_dtype_C << " Got" << v->args_[2]->dtype_);
            for (unsigned i = brgemm_args::C + 1;
                    i < brgemm_args::NUM_FULL_ARGS_STRIDE; i++) {
                COMPILE_ASSERT_POS(
                        v->args_[i]->dtype_ == brgemm_args::arg_types[i],
                        "The " << i
                               << "-th argument of brgemm does not match. "
                                  "Expecting"
                               << brgemm_args::arg_types[i] << " Got "
                               << v->args_[i]->dtype_);
            }
            break;
        }
        case intrin_type::list_brgemm: {
            COMPILE_ASSERT_POS(v->dtype_ == datatypes::void_t,
                    "list_brgemm should return void");
            COMPILE_ASSERT_POS(
                    v->check_brgemm_arg_size(brgemm_args::NUM_FULL_ARGS_LIST),
                    "Wrong number of arguments for list_brgemm");

            auto &extras = v->intrin_attrs_->get<brgemm_args::extra_args_t>(
                    intrin_attr::brgemm_extras);
            if (extras.is_cpu_) {
                COMPILE_ASSERT_POS(validate_brgemm_dtype(extras.dtype_A_,
                                           extras.dtype_B_, extras.dtype_C_,
                                           !extras.postops_setting_.empty()),
                        "list_brgemm currently only support "
                        "u8s8s32/s8s8s32/bf16bf16f32/f32f32f32, got: "
                                << v);
            } else {
                COMPILE_ASSERT_POS(extras.dtype_A_ == datatypes::f32
                                && extras.dtype_B_ == datatypes::f32
                                && extras.dtype_C_ == datatypes::f32,
                        "list_BRGEMM currently only support fp32, got: " << v);
            }

            for (unsigned i = 0; i < brgemm_args::NUM_FULL_ARGS_LIST; i++) {
                COMPILE_ASSERT_POS(
                        v->args_[i]->dtype_ == brgemm_args::list_arg_types[i],
                        "The " << i
                               << "-th argument of list_brgemm does not match. "
                                  "Expecting"
                               << brgemm_args::list_arg_types[i] << " Got"
                               << v->args_[i]->dtype_);
            }
            break;
        }
        default: break;
    }
}

void validate_impl_t::view(tensor_c v) {
    if (cur_func_) {
        COMPILE_ASSERT_POS(is_var_defined(v.get()), "Use before define: " << v);
    }
    COMPILE_ASSERT_POS(v->dtype_.is_etype_pointer(),
            "Tensor should have tensor type, got: " << v->dtype_);
    validate_type(v->elem_dtype_, v);
    COMPILE_ASSERT_POS(v->elem_dtype_.lanes_ == 1,
            "tensor cannot contain vector types: " << v);
    COMPILE_ASSERT_POS(!v->dims_.empty(), "Expecting the dimension > 0: " << v);
    for (auto &dim : v->dims_) {
        dispatch(dim);
    }
    COMPILE_ASSERT_POS(v->dims_.size() == v->strides_.size(),
            "Expecting dims and strides having same length, but got dims "
            "length: "
                    << v->dims_.size()
                    << " and strides length: " << v->strides_.size());
    int cnt = 0;
    sc_data_type_t idxtype = v->dims_.front()->dtype_;
    for (auto &idx : v->dims_) {
        cnt += 1;
        COMPILE_ASSERT_POS(idx->dtype_ == datatypes::index
                        || idx->dtype_ == datatypes::s32,
                "The " << cnt << "-th index of the tensor has type "
                       << idx->dtype_ << ". Expecting an integer: " << v);
        COMPILE_ASSERT_POS(idx->dtype_ == idxtype,
                "Expecting the all dimemsions within the tensor definition "
                "having the same dtype. Current dimemsion: "
                        << cnt << ". expr = " << v);
    }
}

void validate_impl_t::view(tensorptr_c v) {
    COMPILE_ASSERT_POS(v->dtype_.is_etype_pointer(),
            "Tensor should have tensor type, got: " << v->dtype_);
    dispatch(v->base_);
}

void validate_impl_t::view(var_c v) {
    if (cur_func_) {
        COMPILE_ASSERT_POS(is_var_defined(v.get()), "Use before define: " << v);
    }
    validate_type(v);
}

void validate_impl_t::view(assign_c v) {
    COMPILE_ASSERT_POS(v->var_.isa<var>() || v->var_.isa<indexing>(),
            "Assignment only supports tensor or var, got: " << v);
    dispatch(v->var_);
    dispatch(v->value_);
    COMPILE_ASSERT_POS(v->var_->dtype_ == v->value_->dtype_,
            "Assignment expects the LHS and RHS of the same type, but got "
                    << v->var_->dtype_ << " and " << v->value_->dtype_
                    << " expr = " << v);
}

void validate_impl_t::view(returns_c v) {
    ir_viewer_t::view(v);
    if (cur_func_) {
        if (v->value_.defined()) {
            COMPILE_ASSERT_POS(v->value_->dtype_ == cur_func_->ret_type_,
                    "The current function should return "
                            << cur_func_->ret_type_ << ", but got "
                            << v->value_->dtype_ << ": " << v);
        } else {
            COMPILE_ASSERT_POS(cur_func_->ret_type_ == datatypes::void_t,
                    "Returning void in a non-void function: " << v);
        }
    }
    COMPILE_ASSERT_POS(for_loop_levels_ == 0
                    || (cur_func_ && cur_func_->attr_
                            && cur_func_->attr_->get_or_else(
                                    function_attrs::low_level, false)),
            "Cannot return in a for-loop: " << v);
}

void validate_impl_t::check_var_tensor_def(
        const define_c &v, bool allow_tensor_view, bool is_global) const {
    COMPILE_ASSERT_POS(v->var_.isa<var>() || v->var_.isa<tensor>(),
            "Expecting var/tensor: " << v);
    if (v->init_.defined()) {
        if (!allow_tensor_view) {
            if (v->var_.isa<var>()) {
                COMPILE_ASSERT_POS(v->init_->dtype_ == v->var_->dtype_,
                        "The init val has different type from the var "
                        "definition "
                                << v);
            } else {
                COMPILE_ASSERT_POS(v->init_.isa<intrin_call>()
                                && utils::is_one_of(
                                        v->init_.static_as<intrin_call>()
                                                ->type_,
                                        intrin_type::read_struct),
                        "The init val of tensor should come from dynamic "
                        "extract intrin call "
                                << v);
            }
        }
    }
    if (!is_global && v->var_.isa<tensor>()) {
        auto &init = v->var_.static_as<tensor>()->init_value_;
        COMPILE_ASSERT_POS(init == nullptr
                        || init == tensor_node::get_zero_tensor_initializer()
                        || init->size_ == sizeof(union_val)
                        || (v->var_->attr_
                                && v->var_->attr_->has_key(
                                        attr_keys::shared_const)),
                "The tensor defined in function cannot have init value: " << v);
    }
}

void validate_impl_t::view(stmts_c v) {
    if (cur_func_) { defined_vars_.emplace_back(var_scope()); }
    ir_viewer_t::view(v);
    if (cur_func_) { defined_vars_.pop_back(); }
}

void validate_impl_t::view(define_c v) {
    if (cur_func_) { add_def(v->var_.get(), v); }
    ir_viewer_t::view(v);
    COMPILE_ASSERT_POS(v->linkage_ != linkage::private_global
                    && v->linkage_ != linkage::public_global,
            "The variable defined in function cannot be global: " << v);
    check_var_tensor_def(v, allow_tensor_view_, false);
}

void validate_impl_t::view(if_else_c v) {
    dispatch(v->condition_);
    dispatch(v->then_case_);
    if (v->else_case_.defined()) { dispatch(v->else_case_); }
    COMPILE_ASSERT_POS(v->condition_->dtype_ == datatypes::boolean,
            "If-else node expects an boolean expr as the condition, got "
                    << v->condition_->dtype_ << " expr = " << v);
}
void validate_impl_t::view(for_loop_c v) {
    for_loop_levels_++;
    COMPILE_ASSERT_POS(v->isvalid(), "met an invalid for-loop");
    if (cur_func_) { add_def(v->var_.get(), v); }
    dispatch(v->var_);
    dispatch(v->iter_begin_);
    dispatch(v->iter_end_);
    dispatch(v->step_);
    dispatch(v->body_);

    if (cur_func_) { defined_vars_.back().erase(v->var_.get()); }
    COMPILE_ASSERT_POS(v->var_->dtype_ == datatypes::index
                    || v->var_->dtype_ == datatypes::s32,
            "for_loop node expects an index or s32 itervar, got "
                    << v->var_->dtype_ << " expr = " << v);
    COMPILE_ASSERT_POS(v->var_->dtype_ == v->iter_begin_->dtype_,
            "iter_begin of for_loop node expects an "
                    << v->var_->dtype_ << " as the itervar, got "
                    << v->iter_begin_->dtype_ << " expr = " << v);
    COMPILE_ASSERT_POS(v->var_->dtype_ == v->iter_end_->dtype_,
            "iter_end of for_loop node expects an "
                    << v->var_->dtype_ << " as the itervar, got "
                    << v->iter_end_->dtype_ << " expr = " << v);
    COMPILE_ASSERT_POS(v->var_->dtype_ == v->step_->dtype_,
            "step of for_loop node expects an "
                    << v->var_->dtype_ << " as the itervar, got "
                    << v->step_->dtype_ << " expr = " << v);
    if (v->kind_ == for_type::NORMAL) {
        COMPILE_ASSERT_POS(v->num_threads_ == 0,
                "Expecting non-parallel for-loop's num threads = 0");
    }
    for_loop_levels_--;
}

func_c validator_t::operator()(func_c f) {
    validate_impl_t v;
    return v.dispatch(std::move(f));
}

stmt_c validator_t::operator()(stmt_c f) {
    validate_impl_t v;
    return v.dispatch(std::move(f));
}

expr_c validator_t::operator()(expr_c f) {
    validate_impl_t v;
    return v.dispatch(std::move(f));
}

const_ir_module_ptr validator_t::operator()(const_ir_module_ptr f) {
    validate_impl_t vali;
    vali.defined_vars_.emplace_back(validate_impl_t::var_scope());
    for (auto &v : f->get_module_vars()) {
        vali.add_def(v->var_.get(), v);
        vali.ir_viewer_t::view(v);
        COMPILE_ASSERT(v->linkage_ == linkage::private_global
                        || v->linkage_ == linkage::public_global,
                "The variable defined in module should be global: " << v);
        vali.check_var_tensor_def(v, false, true);
    }
    for (auto &funct : f->get_contents()) {
        vali.dispatch(funct);
    }
    return f;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
