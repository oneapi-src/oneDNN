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
#include <algorithm>
#include <atomic>
#include <utility>
#include "tensor_shrink.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass/ir_copy_internal.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(func_inliner, SC_PASS_DEPENDS_ON(validator, trace_inserter),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(FUNC_INLINED), SC_PASS_UNSET_STATE());

// if v is a stmts_node_t, return v
// else, return a stmts_node_t containing v
static stmts_c promote_stmt_to_stmts(stmt_c v) {
    if (v.isa<stmts>()) { return v.static_as<stmts>(); }
    return builder::make_stmts_unattached({std::move(v)}).static_as<stmts_c>();
}

// The function body copier
class func_body_copier_t : public ir_copier_impl_t {
public:
    using ir_copier_impl_t::copy;
    using ir_copier_impl_t::dispatch;
    using ir_copier_impl_t::view;
    bool already_returned_ = false;
    expr ret_var_;

    func_body_copier_t(
            std::unordered_map<expr_c, expr> &replace_map, expr ret_var)
        : ir_copier_impl_t(replace_map), ret_var_(std::move(ret_var)) {}

    expr copy(const expr_c &v) override {
        auto ret = ir_copier_impl_t::copy(v);
        if (ret.isa<tensor>() || ret.isa<tensorptr>()) {
            if (ret->attr_
                    && ret->attr_->has_key(
                            tensor_shrinker_attrs::should_shrink)) {
                auto &shrink_info
                        = ret->attr_->get<tensor_shrinker_t::shrink_info_t>(
                                tensor_shrinker_attrs::should_shrink);
                std::vector<expr> new_info;
                new_info.reserve(shrink_info.base_.size());
                for (auto &old : shrink_info.base_) {
                    new_info.emplace_back(copy(old));
                }
                shrink_info.base_ = std::move(new_info);

                new_info.reserve(shrink_info.shape_.size());
                for (auto &old : shrink_info.shape_) {
                    new_info.emplace_back(copy(old));
                }
                shrink_info.shape_ = std::move(new_info);
            }
        }
        return ret;
    }

    void view(returns_c v) override {
        already_returned_ = true;
        if (v->value_.defined()) {
            COMPILE_ASSERT(ret_var_.defined(),
                    "The function to inline returns a value, but ret_var_ is "
                    "not set");
            expr newval = copy(v->value_);
            returned_stmt_
                    = make_stmt<assign_node_t>(ret_var_, std::move(newval));
        } else {
            returned_stmt_ = make_stmt<stmts_node_t>(std::vector<stmt>());
        }
    }

    stmt_c dispatch(stmt_c v) override {
        COMPILE_ASSERT(!already_returned_,
                "return_node should be the last statement in the IR, got "
                        << v);
        return ir_copier_impl_t::dispatch(std::move(v));
    }
};

// the count of the inlined function calls
static std::atomic<int> count(0);

// TODO(xxx): Recursively inline the target function (with max recursion depth)
class func_inliner_impl_t : public ir_visitor_t {
public:
    struct insertion_point_t {
        std::vector<stmt> &base;
        size_t index;
    };

    // recursion inline depth
    int recursions = 0;
    // the current insertion point within a stmts_node_t
    insertion_point_t *ins_point = nullptr;
    const_ir_module_ptr modu_ = nullptr;
    bool need_index_flatten_;
    func_inliner_impl_t(
            bool need_index_flatten, const const_ir_module_ptr &modu = nullptr)
        : modu_(modu), need_index_flatten_(need_index_flatten) {}

    bool is_func_marked_inline(node_base *callee) {
        if (auto f = dynamic_cast<func_base *>(callee)) {
            if (modu_) {
                if (auto real_f = modu_->get_func(f->name_)) {
                    return any_map_t::fetch_or_else(
                                   real_f->attr_.get(), "inline_level", -1)
                            == 2;
                }
            }
        }
        return false;
    }
    expr_c visit(call_c v) override {
        bool is_parallel_call = bool(!v->para_attr_.empty());
        auto callee = v->func_;
        auto ret = ir_visitor_t::visit(std::move(v));
        if (!is_parallel_call
                && (any_map_t::fetch_or_else(ret->attr_.get(), "inline_level",
                            -1) == 2
                        || is_func_marked_inline(callee.get()))) {
            if (modu_) {
                return inline_at(ret.checked_as<call>(), modu_);
            } else {
                return inline_at(ret.checked_as<call>());
            }
        }
        return ret;
    }

    stmt_c visit(if_else_c v) override {
        auto cond = dispatch(v->condition_);
        auto thencase = dispatch(promote_stmt_to_stmts(v->then_case_));
        stmt_c elsecase;
        if (v->else_case_.defined())
            elsecase = dispatch(promote_stmt_to_stmts(v->else_case_));
        bool changed = !cond.ptr_same(v->condition_)
                || !elsecase.ptr_same(v->else_case_)
                || !thencase.ptr_same(v->then_case_);
        if (changed) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(cond, thencase, elsecase));
        }
        return v;
    }

    stmt_c visit(for_loop_c v) override {
        auto var = dispatch(v->var_);
        auto begin = dispatch(v->iter_begin_);
        auto end = dispatch(v->iter_end_);
        auto step = dispatch(v->step_);
        auto body = dispatch(promote_stmt_to_stmts(v->body_));

        bool changed = !((var.ptr_same(v->var_)
                && begin.ptr_same(v->iter_begin_) && end.ptr_same(v->iter_end_)
                && step.ptr_same(v->step_) && body.ptr_same(v->body_)));
        if (changed) {
            return copy_attr(*v,
                    builder::make_for_loop_unattached(var, begin, end, step,
                            body, v->incremental_, v->kind_, v->num_threads_));
        }
        return v;
    }

    stmt_c visit(evaluate_c v) override {
        bool is_call = v->value_.isa<call>();
        auto newv = dispatch(v->value_);
        if (!newv.ptr_same(v->value_)) {
            // if we have inlined a function returning void
            if (!newv.defined()) {
                return make_stmt<stmts_node_t>(std::vector<stmt>());
            }
            // if we have inlined a function call and the function call returns
            // a value, do not make it a variable definition
            if (is_call && (newv.isa<var>() || newv.isa<tensor>())) {
                return make_stmt<stmts_node_t>(std::vector<stmt>());
            }
            return builder::make_evaluate_unattached(newv);
        }
        return v;
    }

    stmt_c visit(stmts_c v) override {
        bool changed = false;
        // a shadow array of stmt
        std::vector<stmt> bb;
        bb.reserve(v->seq_.size());
        insertion_point_t insp {bb, 0};
        auto old_insp = ins_point;
        ins_point = &insp;
        for (unsigned i = 0; i < v->seq_.size(); i++) {
            // always append to the tail of the BB
            insp.index = bb.size();
            auto newv = dispatch(v->seq_.at(i));
            changed |= !newv.ptr_same(v->seq_.at(i));
            bb.emplace_back(std::move(newv).remove_const());
        }
        ins_point = old_insp;
        if (changed) {
            return copy_attr(*v, make_stmt<stmts_node_t>(std::move(bb)));
        }
        return v;
    }

    expr_c inline_at(call_c site, const const_ir_module_ptr &modu = nullptr) {
        auto the_func = std::dynamic_pointer_cast<func_base>(site->func_);
        // if the callee is a declaration, find its function body in modu.
        if (!the_func) { return site; }
        if (modu) {
            if (auto real_func = modu->get_func(the_func->name_)) {
                the_func = real_func;
            }
        }
        if (!the_func->body_.defined()) { return site; }
        recursions++;
        COMPILE_ASSERT(recursions < 20, "Reached max inline recursion depth");

        // fill in the replace map
        std::unordered_map<expr_c, expr> rmap;
        bool has_tptr_arg = false;
        for (size_t i = 0; i < site->args_.size(); i++) {
            rmap.insert(
                    std::make_pair(the_func->params_.at(i), site->args_.at(i)));
            // if the arg has tensor ptr, we need to call index flatten in the
            // new body to make sure nested tptr like "&&a[0][0]" is flattened
            if (site->args_[i].isa<tensorptr>()) { has_tptr_arg = true; }
        }
        if (modu) {
            auto module_vars = modu->get_module_vars();
            for (size_t i = 0; i < module_vars.size(); i++) {
                rmap.insert(std::make_pair(
                        module_vars[i]->var_, module_vars[i]->var_));
            }
        }

        // a "simple" function is a function with only one statement: return ...
        // we can extract and inline the returned expression
        stmt the_only_stmt;
        if (!the_func->body_.isa<stmts>()) {
            the_only_stmt = the_func->body_;
        } else {
            auto &thestmts = the_func->body_.static_as<stmts>()->seq_;
            if (thestmts.size() == 1) { the_only_stmt = thestmts.front(); }
        }
        // if the function has only one stmt and it is a return, then it is
        // "simple"
        if (the_only_stmt.defined() && the_only_stmt.isa<returns>()) {
            // if it is a "simple" function:
            expr the_expr = the_only_stmt.static_as<returns>()->value_;
            assert(the_expr.defined());
            // first copy the function body
            func_body_copier_t impl(rmap, expr());
            expr_c newexpr = impl.copy(the_expr);
            // then recursively inline the "call_node"s in the body
            newexpr = dispatch(std::move(newexpr));
            recursions--;
            return newexpr;
        } else {
            expr outvar;
            std::vector<stmt>::iterator itr
                    = ins_point->base.begin() + ins_point->index;
            if (site->dtype_ != datatypes::void_t) {
                // declare the return value as a var
                outvar = builder::make_var(
                        site->dtype_, "_retval" + std::to_string(count++));
                itr = ins_point->base.insert(itr,
                              builder::make_var_tensor_def_unattached(outvar))
                        + 1;
            }
            func_body_copier_t impl(rmap, outvar);
            stmt_c new_body = impl.copy(promote_stmt_to_stmts(the_func->body_));
            // then recursively inline the "call_node"s in the body
            new_body = dispatch(std::move(new_body));
            if (has_tptr_arg && need_index_flatten_) {
                // need to flatten the nested tensorptrs
                new_body = index_flattener_t()(new_body);
            }
            // push the new stmt in the insertion point
            ins_point->base.insert(itr, std::move(new_body).remove_const());

            recursions--;
            return outvar;
        }
    }
};

expr_c func_inliner_t::inline_at(call_c c, std::vector<stmt> &seq,
        size_t insert_idx, const const_ir_module_ptr &modu) {
    func_inliner_impl_t impl {false};
    func_inliner_impl_t::insertion_point_t insp {seq, insert_idx};
    impl.ins_point = &insp;
    return impl.inline_at(std::move(c), modu);
}

static func_t get_callee(const stmt &s) {
    return s.cast<evaluate>()
            .flat_map([](const evaluate &v) { return v->value_.cast<call>(); })
            .map([](const call &v) {
                return std::dynamic_pointer_cast<func_base>(v->func_);
            })
            .get_or_else(nullptr);
}

// check if the function is simple enough to inline. Currently, we consider a
// function which only have eval-call as a simply function.
static bool is_simple_func(const func_t &f) {
    if (!f->body_.defined()) { return false; }
    for (auto &s : f->body_.checked_as<stmts>()->seq_) {
        if (!get_callee(s)) { return false; }
    }
    return true;
}

const_ir_module_ptr func_inliner_t::operator()(const_ir_module_ptr f) {
    func_inliner_impl_t impl(needs_index_flatten_, f);
    auto mainf = f->get_entry_func();
    for (auto &fu : f->get_contents()) {
        if (fu != mainf && is_simple_func(fu)) {
            fu->attr()["inline_level"] = 2;
            fu->decl_->attr()["inline_level"] = 2;
        }
    }

    return dispatch_module_on_visitor(&impl, f);
}

func_c func_inliner_t::operator()(func_c f) const {
    func_inliner_impl_t impl {needs_index_flatten_};
    return impl.dispatch(std::move(f));
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
