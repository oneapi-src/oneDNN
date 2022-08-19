/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <utility>
#include <vector>
#include "loop_invariant_code_motion.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>

namespace sc {

struct licm_analysis_data_t {
    const stmt_base_t *parent_;
    // if the var is global or volatile
    bool volatile_ = false;
    // The loop vars stmt depends on
    std::unordered_set<expr_c> dep_vars_;

    licm_analysis_data_t(const stmt_base_t *parent) : parent_(parent) {}
};

// Currently tensor/indexing/tptr/call/intrin_call can not be hoisted.
static bool expr_can_hoist(const expr_c &s) {
    return s.isa<var_c>() || s.isa<cast_c>() || (s.instanceof <binary_c>())
            || (s.instanceof <cmp_c>()) || (s.instanceof <logic_c>())
            || s.isa<select_c>() || s.isa<constant_c>() || s.isa<ssa_phi_c>();
}

static bool stmt_can_hoist(const stmt_c &s) {
    return s.isa<define_c>() || s.isa<if_else_c>() || s.isa<for_loop_c>()
            || s.isa<stmts_c>();
}

struct licm_analysis_viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;
    const stmt_base_t *current_ = nullptr;
    // currently we treat if_scope as a whole stmt to judge if it is an
    // invariant.
    int if_scope_depth_ = 0;
    std::vector<expr_c> cur_loop_vars_;
    std::unordered_set<expr_c> vars_in_if_scope_;
    // output map: loop var => vector of invariants, the invariants will
    // be inserted just before their correspond loop vars.
    std::unordered_map<expr_c, std::vector<stmt_c>> loop_invariant_map_;
    std::unordered_set<stmt_c> stmt_to_remove_set_;
    void register_loop_invariant_stmt() {
        assert(current_ != nullptr);
        // if the stmt is not in loop, return.
        if (cur_loop_vars_.empty()) { return; }
        // if the stmt is in if_else scope, do not hoist
        if (if_scope_depth_ > 0) { return; }
        auto &st_data = current_->temp_data().get<licm_analysis_data_t>();
        if (st_data.volatile_) { return; }
        stmt_c s = current_->node_ptr_from_this();
        // hoist out of outmost loop
        if (st_data.dep_vars_.empty()) {
            loop_invariant_map_[cur_loop_vars_[0]].emplace_back(s);
            stmt_to_remove_set_.insert(s);
            return;
        }
        for (auto it = cur_loop_vars_.rbegin(); it != cur_loop_vars_.rend();
                it++) {
            if (std::find_if(st_data.dep_vars_.begin(), st_data.dep_vars_.end(),
                        [&](const expr_c &v) { return v.ptr_same(*it); })
                    != st_data.dep_vars_.end()) {
                if (it != cur_loop_vars_.rbegin()) {
                    loop_invariant_map_[*(it - 1)].emplace_back(s);
                    stmt_to_remove_set_.insert(s);
                }
                return;
            }
        }
        COMPILE_ASSERT(false,
                "Stmt " << current_ << " has unknown loop var "
                        << *st_data.dep_vars_.begin());
    }
    stmt_c dispatch(stmt_c s) override {
        if (!s->get_temp_data().isa<licm_analysis_data_t>()) {
            s->temp_data() = licm_analysis_data_t(current_);
        } else {
            // if the vardef is used before define, update the parent
            auto &licm_data = s->temp_data().get<licm_analysis_data_t>();
            assert(licm_data.parent_ == nullptr);
            licm_data.parent_ = current_;
        }
        if (!stmt_can_hoist(s)) {
            s->temp_data().get<licm_analysis_data_t>().volatile_ = true;
            return s;
        }
        auto old = current_;
        current_ = s.get();
        auto ret = ssa_viewer_t::dispatch(std::move(s));
        register_loop_invariant_stmt();
        current_ = old;
        return ret;
    }
    expr_c dispatch(expr_c s) override {
        if (!expr_can_hoist(s) && current_ != nullptr
                && !cur_loop_vars_.empty()) {
            current_->temp_data().get<licm_analysis_data_t>().volatile_ = true;
            return s;
        }
        if (!s->get_temp_data().isa<licm_analysis_data_t>()) {
            s->temp_data() = licm_analysis_data_t(current_);
        }
        return ssa_viewer_t::dispatch(std::move(s));
    }
    void view(for_loop_c v) override {
        cur_loop_vars_.emplace_back(v->var_);
        ssa_viewer_t::view(v);
        cur_loop_vars_.pop_back();
        auto &v_data = v->temp_data().get<licm_analysis_data_t>();
        auto &body_data = v->body_->temp_data().get<licm_analysis_data_t>();
        v_data.volatile_ |= body_data.volatile_;
        v_data.dep_vars_.insert(
                body_data.dep_vars_.begin(), body_data.dep_vars_.end());
    }
    void view(stmts_c v) override {
        ssa_viewer_t::view(v);
        auto &v_data = v->temp_data().get<licm_analysis_data_t>();
        for (auto &st : v->seq_) {
            // if st is garbage
            if (!st->get_temp_data().isa<licm_analysis_data_t>()) { continue; }
            auto &st_data = st->temp_data().get<licm_analysis_data_t>();
            v_data.dep_vars_.insert(
                    st_data.dep_vars_.begin(), st_data.dep_vars_.end());
            if (st_data.volatile_) {
                v_data.volatile_ = true;
                break;
            }
        }
    }
    void view(define_c v) override {
        auto &st_data = v->temp_data().get<licm_analysis_data_t>();
        dispatch(v->var_);
        if (v->init_.defined()) { dispatch(v->init_); }
        // synchronize volatile attribute between define node and its var node;
        if (v->var_.isa<var_c>()) {
            auto &var_data = v->var_->temp_data().get<licm_analysis_data_t>();
            var_data = st_data;
            if (if_scope_depth_ > 0) { vars_in_if_scope_.insert(v->var_); }
        }
    }
    void view(if_else_c v) override {
        if (if_scope_depth_ == 0) { assert(vars_in_if_scope_.empty()); }
        if_scope_depth_++;
        ssa_viewer_t::view(v);
        auto &st_data = v->temp_data().get<licm_analysis_data_t>();
        auto &then_data
                = v->then_case_->temp_data().get<licm_analysis_data_t>();
        st_data.volatile_ |= then_data.volatile_;
        st_data.dep_vars_.insert(
                then_data.dep_vars_.begin(), then_data.dep_vars_.end());
        if (v->else_case_.defined()) {
            auto &else_data
                    = v->else_case_->temp_data().get<licm_analysis_data_t>();
            st_data.volatile_ |= else_data.volatile_;
            st_data.dep_vars_.insert(
                    else_data.dep_vars_.begin(), else_data.dep_vars_.end());
        }
        if_scope_depth_--;
        if (if_scope_depth_ == 0) {
            for (auto &var : vars_in_if_scope_) {
                auto &var_data = var->temp_data().get<licm_analysis_data_t>();
                var_data.volatile_ |= st_data.volatile_;
                var_data.dep_vars_.insert(
                        st_data.dep_vars_.begin(), st_data.dep_vars_.end());
            }
            vars_in_if_scope_.clear();
        }
    }

    void view(var_c v) override {
        auto owner = v->ssa_data_->get_owner();
        auto &is_param = v->ssa_data_->is_param_;
        if (current_ == nullptr) {
            assert(is_param);
            return;
        }
        auto &st_data = current_->temp_data().get<licm_analysis_data_t>();
        auto &var_data = v->temp_data().get<licm_analysis_data_t>();
        if (var_data.volatile_) {
            st_data.volatile_ = true;
            return;
        }
        st_data.dep_vars_.insert(
                var_data.dep_vars_.begin(), var_data.dep_vars_.end());
        // param can be considered as invariant.
        // if a var is used before defined, treat it as volatile.
        if (owner.defined()
                && owner->temp_data().get_or_null<licm_analysis_data_t>()) {
            if (owner.isa<for_loop_c>()) { st_data.dep_vars_.insert(v); }
        } else if (!is_param) {
            var_data.volatile_ = true;
            st_data.volatile_ = true;
        }
    }
    void view(ssa_phi_c v) override {
        ssa_viewer_t::view(v);
        if (v->is_loop_phi_) {
            for (auto &val : v->values_) {
                auto owner = val->ssa_data_->get_owner();
                if (owner.defined()
                        && owner->temp_data()
                                   .get_or_null<licm_analysis_data_t>()) {
                    owner->temp_data().get<licm_analysis_data_t>().volatile_
                            = true;
                }
            }
        }
    }
};

// Second filter map and set to remove stmt by volatile == true
static void filter_stmt_by_volatile(
        std::unordered_map<expr_c, std::vector<stmt_c>> &hoist_map,
        std::unordered_set<stmt_c> &stmt_to_remove) {
    for (auto &kv : hoist_map) {
        for (auto it = kv.second.begin(); it != kv.second.end();) {
            if ((*it)->temp_data().get<licm_analysis_data_t>().volatile_) {
                it = kv.second.erase(it);
            } else {
                it++;
            }
        }
    }
    for (auto it = stmt_to_remove.begin(); it != stmt_to_remove.end();) {
        if ((*it)->temp_data().get<licm_analysis_data_t>().volatile_) {
            it = stmt_to_remove.erase(it);
        } else {
            it++;
        }
    }
}
struct licm_hoister_t : public ssa_visitor_t {
    using ssa_visitor_t::dispatch;
    using ssa_visitor_t::visit;
    std::unordered_map<expr_c, std::vector<stmt_c>> &hoist_map_;
    std::unordered_set<stmt_c> &stmt_to_remove_;
    licm_hoister_t(std::unordered_map<expr_c, std::vector<stmt_c>> &hoist_map,
            std::unordered_set<stmt_c> &stmt_to_remove)
        : hoist_map_(hoist_map), stmt_to_remove_(stmt_to_remove) {}
    // we are not interested in exprs. Simply return and don't dispatch down
    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt_c> ret_vec;
        bool changed = false;

        for (auto &s : v->seq_) {
            // if s is garbage
            if (!s->get_temp_data().isa<licm_analysis_data_t>()) {
                changed = true;
                continue;
            }
            auto sz_before = ret_vec.size();
            auto ret = dispatch(s);

            // if an stmt is inserted, the IR is changed
            changed |= sz_before != ret_vec.size();
            changed |= !ret.ptr_same(s);
            if (stmt_to_remove_.find(ret) != stmt_to_remove_.end()) {
                assert(ret->temp_data().get<licm_analysis_data_t>().volatile_
                        == false);
                changed = true;
                continue;
            }
            if (ret.isa<for_loop_c>()) {
                auto it = hoist_map_.find(ret.static_as<for_loop_c>()->var_);
                if (it != hoist_map_.end()) {
                    std::vector<stmt_c> *cur_hoist_stmts = &(it->second);
                    ret_vec.insert(ret_vec.end(), cur_hoist_stmts->begin(),
                            cur_hoist_stmts->end());
                    changed = true;
                }
            }
            ret_vec.emplace_back(ret);
        }
        if (!changed) {
            return v;
        } else {
            return copy_attr(*v, builder::make_stmts_unattached(ret_vec));
        }
    }
};

func_c loop_invariant_code_motion_t::operator()(func_c f) {
    licm_analysis_viewer_t analyzer;
    analyzer.dispatch(f);
    filter_stmt_by_volatile(
            analyzer.loop_invariant_map_, analyzer.stmt_to_remove_set_);
    licm_hoister_t hoister(
            analyzer.loop_invariant_map_, analyzer.stmt_to_remove_set_);
    return hoister.top_level_dispatch(f);
}

} // namespace sc
