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
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/xbyak/ir/xbyak_expr.hpp>

#include "call_transform.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
using namespace xbyak::x86_64;

// Cached ABI infomation inside func node
abi_function_interface::ptr cached_func_abi_interface(const func_t &v) {
    assert(v->attr_ && v->attr_->has_key(attr_keys::abi_interface));
    auto func_iface = v->attr_->get<abi_function_interface::ptr>(
            attr_keys::abi_interface);
    return func_iface;
}

// Cached ABI infomation inside call node
abi_function_interface::ptr cached_call_abi_interface(const call_c &v) {
    return cached_func_abi_interface(v->get_prototype());
}

/* *
 * Extract call node as root expr node, so it is easier to transform individual
 * call into a call scope stmts.
 * */
class call_extract_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    call_extract_impl_t() = default;

    size_t index_ = 0;
    size_t expr_depth_ = 0;

    std::vector<stmt> stmt_seq_;

    expr_c dispatch(expr_c v) override {
        expr_depth_++;
        auto ret = ir_visitor_t::dispatch(std::move(v));
        expr_depth_--;
        if (ret.isa<call_c>() && expr_depth_ > 0) {
            return extract_call(ret.remove_const());
        }
        return ret;
    }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt> new_seq;
        for (auto &s : v->seq_) {
            auto ss = ir_visitor_t::dispatch(s);
            for (auto &&ts : stmt_seq_) {
                new_seq.push_back(std::move(ts));
            }
            new_seq.emplace_back(ss.remove_const());
            stmt_seq_.clear();
        }
        return copy_attr(*v, make_stmt<stmts_node_t>(std::move(new_seq)));
    }

    expr extract_call(expr v) {
        auto call_var = builder::make_var(
                v->dtype_, "tmp_call_var_" + std::to_string(index_++));
        stmt_seq_.emplace_back(make_stmt<define_node_t>(
                call_var, linkage::local, std::move(v)));
        return call_var;
    }
};

/* *
 * Transform each call node to dedicated stmts node, mark the node as function
 * call scope, add reverse order args inside scope. And make arg passed to %rdx
 * reg to be the last arg assigned (%rdx is a special reg, may used by certain
 * operations inside call scope)
 * */
class call_transform_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    call_transform_impl_t(const x86_64::target_profile_t &profile)
        : profile_(profile) {}

    const x86_64::target_profile_t &profile_;

    abi_function_interface::ptr func_iface_;

    std::vector<stmt> new_seq_;
    std::vector<stmt> new_rdx_;

    size_t index_ = 0;

    func_c dispatch(func_c v) override {
        func_t func = std::const_pointer_cast<func_base>(v);
        cache_abi_interface(func);
        return ir_visitor_t::dispatch(std::move(v));
    }

    // Reverse call args order
    expr_c dispatch(expr_c v) override {
        if (v.isa<call>()) {
            auto vv = v.static_as<call_c>();
            func_t callee = vv->get_prototype();
            func_iface_ = cache_abi_interface(callee);
            auto &old_args = vv->args_;
            std::vector<expr> new_args;
            for (size_t i = 0; i < old_args.size(); i++) {
                bool is_reg_rdx = false;
                auto &arg = old_args[i];
                auto &initial_loc = func_iface_->param_locs_[i];
                // Create new arg placeholder to be replace args in call node
                auto arg_placeholder = builder::make_var(arg->dtype_,
                        "call_arg_" + std::to_string(index_) + "_"
                                + std::to_string(i + 1));
                // Create wrapper for actural arg value
                auto arg_wrapper = make_xbyak_intrin(
                        arg->dtype_, {arg}, xbyak_intrin_type::call_arg);
                // Get currrent arg location, %rdx is spcical arg
                if (initial_loc.get_type()
                        == abi_value_location::tag_type::REGISTER) {
                    Xbyak::Reg arg_reg = initial_loc.get_register();
                    if (arg_reg.isREG()
                            && arg_reg.getIdx() == Xbyak::Operand::RDX) {
                        is_reg_rdx = true;
                    }
                }
                // Arg passed to %rdx reg need to be the last assigned
                if (is_reg_rdx) {
                    new_rdx_.emplace_back(make_stmt<define_node_t>(
                            arg_placeholder, linkage::local, arg_wrapper));
                } else {
                    new_seq_.emplace_back(make_stmt<define_node_t>(
                            arg_placeholder, linkage::local, arg_wrapper));
                }
                // Add to new call node args
                new_args.emplace_back(std::move(arg_placeholder));
            }
            index_++;
            return copy_attr(
                    *vv, make_expr<call_node>(vv->func_, std::move(new_args)));
        } else {
            // After extraction, call node only exists in root expr node.
            return v;
        }
    }

    stmt_c transform_call_stmt(bool is_call, stmt_c v) {
        if (is_call) {
            std::reverse(new_seq_.begin(), new_seq_.end());
            if (!new_rdx_.empty()) {
                new_seq_.emplace_back(std::move(new_rdx_.back()));
            }
            new_seq_.emplace_back(v.remove_const());
            auto ret = make_stmt<stmts_node_t>(std::move(new_seq_));
            ret->attr().set(attr_keys::abi_interface, func_iface_);
            new_seq_.clear();
            new_rdx_.clear();
            return ret;
        } else {
            return v;
        }
    }

    stmt_c visit(define_c v) override {
        bool is_call = v->init_.defined() && v->init_.isa<call>();
        auto vv = ir_visitor_t::visit(std::move(v));
        return transform_call_stmt(is_call, std::move(vv));
    }

    stmt_c visit(assign_c v) override {
        bool is_call = v->value_.defined() && v->value_.isa<call>();
        auto vv = ir_visitor_t::visit(std::move(v));
        return transform_call_stmt(is_call, std::move(vv));
    }

    stmt_c visit(evaluate_c v) override {
        bool is_call = v->value_.defined() && v->value_.isa<call>();
        auto vv = ir_visitor_t::visit(std::move(v));
        return transform_call_stmt(is_call, std::move(vv));
    }

    stmt_c visit(returns_c v) override {
        bool is_call = v->value_.defined() && v->value_.isa<call>();
        auto vv = ir_visitor_t::visit(std::move(v));
        return transform_call_stmt(is_call, std::move(vv));
    }

    abi_function_interface::ptr cache_abi_interface(func_t &v) {
        if (v->attr_ && v->attr_->has_key(attr_keys::abi_interface)) {
            auto func_iface = v->attr_->get<abi_function_interface::ptr>(
                    attr_keys::abi_interface);
            return func_iface;
        } else {
            auto func_iface = abi_function_interface::make_interface(
                    profile_, v->params_, v->ret_type_);
            v->attr().set(attr_keys::abi_interface, func_iface);
            return func_iface;
        }
    }
};

func_c call_transform_t::operator()(func_c v) {
    if (v->name_.find("_should_inline_") != std::string::npos) { return v; }
    call_extract_impl_t call_extract;
    call_transform_impl_t call_transform(profile_);
    return call_transform.dispatch(call_extract.dispatch(std::move(v)));
}

call_transform_t::call_transform_t(const x86_64::target_profile_t &profile)
    : profile_(profile) {}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
