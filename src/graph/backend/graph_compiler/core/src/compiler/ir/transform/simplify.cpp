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
#include "simplify.hpp"
#include <string>
#include <utility>
#include <vector>
#include "../visitor.hpp"
#include "constant_fold.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(ir_simplifier, SC_PASS_DEPENDS_ON(validator, constant_folder),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(IR_SIMPLIFIED), SC_PASS_UNSET_STATE());

/** dead write elimination implementation
 *  this impl is fast and may be called many times.
 * */
class simplify_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    bool skip_rename_;
    simplify_impl_t(bool skip_rename) : skip_rename_(skip_rename) {}
    // the current/ancestor stmts
    std::vector<stmt_c> cur;
    // the defined var/tensor in stmts
    std::unordered_set<std::string> defs;
    // old var to new var map
    std::unordered_map<expr_c, expr_c> rmap;
    // repeat var index
    int var_index = 1;
    // the parent stmts nodes currently met stmt sequences
    std::vector<stmt_c> *pnewseq = nullptr;

    expr_c dispatch(expr_c v) override {
        if (skip_rename_) { return v; }
        return ir_visitor_t::dispatch(v);
    }

    stmt_c dispatch(stmt_c v) override {
        if (cur.empty()) { defs.clear(); }
        cur.emplace_back(v);
        auto ret = ir_visitor_t::dispatch(v);
        cur.pop_back();
        if (cur.empty()) { defs.clear(); }
        return ret;
    }

    // only interested in var/tensor
    expr_c visit(var_c v) override {
        if (rmap.find(v) != rmap.end()) { return rmap[v]; }
        return v;
    }
    expr_c visit(tensor_c v) override {
        if (rmap.find(v) != rmap.end()) { return rmap[v]; }
        return v;
    }

    stmt_c visit(define_c v) override {
        if (skip_rename_) { return v; }
        auto ret = ir_visitor_t::visit(v).static_as<define_c>();
        bool changed = !ret.ptr_same(v);
        auto var0 = ret->var_;
        expr new_var = var0;
        // in stmts
        if (!cur.empty()) {
            if (var0.isa<var>()) {
                auto &name = var0.static_as<var>()->name_;
                if (defs.find(name) == defs.end()) {
                    defs.insert(name);
                } else {
                    new_var = var0->remake();
                    new_var.static_as<var>()->name_
                            = name + "_" + std::to_string(var_index++);
                    rmap[var0] = new_var;
                    changed = true;
                }
            } else {
                assert(var0.isa<tensor>());
                auto &name = var0.static_as<tensor>()->name_;
                if (defs.find(name) == defs.end()) {
                    defs.insert(name);
                } else {
                    new_var = var0->remake();
                    new_var.static_as<tensor>()->name_
                            = name + "_" + std::to_string(var_index++);
                    rmap[var0] = new_var;
                    changed = true;
                }
            }
        }
        if (changed) {
            return copy_attr(*ret,
                    builder::make_var_tensor_def_unattached(
                            new_var, ret->linkage_, ret->init_));
        }
        return v;
    }

    stmt_c visit(evaluate_c v) override {
        if (v->value_.isa<call_c>() || v->value_.isa<intrin_call_c>()
                || v->value_.isa<low_level_intrin_c>()) {
            return ir_visitor_t::visit(std::move(v));
        }
        return stmt_c();
    }

    // Rename for_loop var if same
    // Will case potential error especially in nested loops
    stmt_c visit(for_loop_c v) override {
        if (skip_rename_) { return ir_visitor_t::visit(v); }
        bool changed = false;
        auto new_var = v->var_;
        auto &name = v->var_.static_as<var>()->name_;
        // in stmts
        if (!cur.empty()) {
            // Create new var if exist same name
            if (defs.find(name) == defs.end()) {
                defs.insert(name);
            } else {
                new_var = v->var_->remake();
                new_var.static_as<var>()->name_
                        = name + "_" + std::to_string(var_index++);
                rmap[v->var_] = new_var;
                changed = true;
            }
            // traverse for_loop
            auto begin = dispatch(v->iter_begin_);
            auto end = dispatch(v->iter_end_);
            auto step = dispatch(v->step_);
            auto body = dispatch(v->body_);
            changed |= !begin.ptr_same(v->iter_begin_);
            changed |= !end.ptr_same(v->iter_end_);
            changed |= !step.ptr_same(v->step_);
            changed |= !body.ptr_same(v->body_);
            // make new for_loop if changed
            if (changed) {
                return copy_attr(*v,
                        builder::make_for_loop_unattached(new_var, begin, end,
                                step, body, v->incremental_, v->kind_,
                                v->num_threads_));
            }
            return v;
        }
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(stmts_c v) override {
        bool parent_is_stmts
                = (cur.size() > 1 && cur[cur.size() - 2].isa<stmts>());
        // if the current seq is empty and parent is stmt, return null
        if (v->seq_.empty() && parent_is_stmts) { return stmt_c(); }
        auto parent_seq = pnewseq;
        std::vector<stmt_c> newseq;
        pnewseq = &newseq;

        bool changed = false;
        for (auto &s : v->seq_) {
            auto ret = dispatch(s);
            if (ret.defined()) { newseq.emplace_back(ret); }
            changed |= !ret.ptr_same(s);
        }
        pnewseq = parent_seq;
        if (parent_is_stmts) {
            // if we have no definitions in the current scope and direct parent
            // is a stmts, promote to parent seq
            parent_seq->insert(parent_seq->end(), newseq.begin(), newseq.end());
            return stmts_c();
        }
        if (changed) {
            return copy_attr(*v, builder::make_stmts_unattached(newseq));
        }
        return v;
    }
};

/**
 * This impl will simplify and eliminate stmts including:
 * 1. for_loop
 * 2. if_else
 * @note: this impl can be treated as extended simplifier for ir_simplifier_t
 * above, which has more quick visit for ir.
 * */
class if_loop_simplify_impl_t : public ir_consistent_visitor_t {
public:
    using ir_consistent_visitor_t::dispatch;
    bool should_replace_expr_ = false;
    expr_c dispatch(expr_c v) override {
        if (!should_replace_expr_) {
            return v;
        } else {
            return ir_consistent_visitor_t::dispatch(v);
        }
    }

    // eliminate if(){} else(){}
    stmt_c visit(if_else_c v) override {
        stmt_c then_case, else_case;
        expr_c condition = constant_folder_t()(dispatch(v->condition_));
        bool else_is_empty_stmts = false;
        then_case = dispatch(v->then_case_);
        if (v->else_case_.defined()) {
            else_case = dispatch(v->else_case_);
            else_is_empty_stmts = else_case.isa<stmts>()
                    && else_case.checked_as<stmts>()->seq_.empty();
        }

        /** simplify always true
         * if(True){
         *    // if block
         * }
         * else{
         *    // else block
         * }
         * RETURN:
         * if block
         * */
        if (condition.isa<constant>() && get_expr_as_int(condition) > 0) {
            // similar to always FALSE
            return then_case;
        }
        /** simplify always false
         * iif(FALSE){
         *    // if block
         * }
         * else{
         *    // else block
         * }
         * RETURN:
         * else block
         * */
        if (condition.isa<constant>() && get_expr_as_int(condition) == 0) {
            // similar to always FALSE
            return else_case.defined()
                    ? else_case
                    : make_stmt<stmts_node_t>(std::vector<stmt> {});
        }

        /** simplify empty if
         * if(conditon){
         *    // empty
         * }
         * else{
         *    // else block
         * }
         * RETURN:
         * NULL or
         * if(!conditon){
         *    // else block
         * }
         * */
        if (then_case.isa<stmts>()
                && then_case.static_as<stmts>()->seq_.empty()) {
            if (!else_case.defined() || else_is_empty_stmts) {
                return copy_attr(
                        *v, make_stmt<stmts_node_t>(std::vector<stmt> {}));
            } else {
                return copy_attr(*v,
                        builder::make_if_else_unattached(
                                constant_folder_t()(!condition), else_case,
                                stmt()));
            }
        }
        /** simplify empty else
         * if(conditon){
         *    // if block
         * }
         * else{
         *    // empty
         * }
         * RETURN:
         * if(){
         *    // if block
         * }
         * */
        if (else_is_empty_stmts) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(
                            condition, then_case, stmt()));
        }

        bool changed_ = !(condition.ptr_same(v->condition_)
                && then_case.ptr_same(v->then_case_)
                && else_case.ptr_same(v->else_case_));
        if (changed_) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(
                            condition, then_case, else_case));
        }
        return std::move(v);
    }

    stmt_c dispatch_loop_body_replace_loop_var(
            const expr_c &var, const expr_c &begin, const stmt &body) {
        auto old_should_replace_expr = should_replace_expr_;
        should_replace_expr_ = true;
        replace_map_[var] = begin;
        auto outbody = dispatch(body);
        // the replace map should only affect in current scope
        replace_map_.erase(var);
        should_replace_expr_ = old_should_replace_expr;
        return outbody;
    }

    // eliminate loop
    bool is_loop_merge = false;
    stmt_c visit(for_loop_c v) override {
        // eliminate for(...){}
        if (v->body_.isa<stmts>()
                && v->body_.static_as<stmts>()->seq_.empty()) {
            return copy_attr(*v, make_stmt<stmts_node_t>(std::vector<stmt> {}));
        }
        bool cached_loop_merge = is_loop_merge;
        auto var = dispatch(v->var_);
        auto begin = dispatch(v->iter_begin_);
        auto end = dispatch(v->iter_end_);
        auto step = dispatch(v->step_);

        // check if the constant folder has attached loop_len_hint
        if (v->attr_) {
            int64_t loop_len
                    = v->attr_->get_or_else("loop_len_hint", INT64_C(-1));
            if (loop_len >= 0) {
                if (loop_len == 0) {
                    return copy_attr(
                            *v, make_stmt<stmts_node_t>(std::vector<stmt> {}));
                } else if (loop_len == 1) {
                    return dispatch_loop_body_replace_loop_var(
                            var, begin, v->body_);
                }
            }
        }

        is_loop_merge |= (v->attr_
                && v->attr_->get_or_else(stmt_attr_key::merge_loop, false));

        if (begin.isa<constant>() && end.isa<constant>()
                && step.isa<constant>()) {
            // begin > end
            if (get_expr_as_int(begin) >= get_expr_as_int(end)) {
                is_loop_merge = cached_loop_merge;
                return copy_attr(
                        *v, make_stmt<stmts_node_t>(std::vector<stmt> {}));
            }
            // (begin + step) >= end
            if ((get_expr_as_int(begin) + get_expr_as_int(step))
                    >= get_expr_as_int(end)) {
                auto body = dispatch_loop_body_replace_loop_var(
                        var, begin, v->body_);
                is_loop_merge = cached_loop_merge;
                return body;
            }
        }
        auto body = dispatch(v->body_);
        bool changed = !(var.ptr_same(v->var_) && begin.ptr_same(v->iter_begin_)
                && end.ptr_same(v->iter_end_) && step.ptr_same(v->step_)
                && body.ptr_same(v->body_));

        if (changed
                || (is_loop_merge
                        && (!v->attr_
                                || is_loop_merge
                                        != v->attr_->get_or_else(
                                                stmt_attr_key::merge_loop,
                                                false)))) {
            auto loop = copy_attr(*v,
                    make_stmt<for_loop_node_t>(var.remove_const(),
                            begin.remove_const(), end.remove_const(),
                            step.remove_const(), body.remove_const(),
                            v->incremental_, v->kind_, v->num_threads_))
                                .checked_as<for_loop>();
            if (is_loop_merge) loop->attr()[stmt_attr_key::merge_loop] = true;
            is_loop_merge = cached_loop_merge;
            return loop;
        }

        is_loop_merge = cached_loop_merge;
        return std::move(v);
    }
};

func_c ir_simplifier_t::operator()(func_c f) {
    simplify_impl_t simpl {skip_rename_};
    auto ret = simpl.dispatch(f);
    if (!skip_if_loop_) {
        if_loop_simplify_impl_t ilimpl;
        ret = simpl.dispatch(ilimpl.dispatch(ret));
    }
    return ret;
}
stmt_c ir_simplifier_t::operator()(stmt_c f) const {
    simplify_impl_t simpl {skip_rename_};
    auto ret = simpl.dispatch(std::move(f));
    if (!skip_if_loop_) {
        if_loop_simplify_impl_t ilimpl;
        ret = simpl.dispatch(ilimpl.dispatch(ret));
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
