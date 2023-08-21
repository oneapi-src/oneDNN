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
#include <utility>
#include <vector>
#include "loop_invariant_code_motion.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/passlet/volatility_analysis.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(loop_invariant_code_motion, SC_PASS_DEPENDS_ON(ssa_transform),
        SC_PASS_REQUIRE_STATE(SSA_STAGE), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

struct licm_analysis_data_t {
    const stmt_base_t *parent_;
    // if the var is global or volatile
    bool volatile_ = false;
    // The loop vars stmt depending on
    std::unordered_set<expr_c> dep_vars_;
    std::unordered_set<expr_c> dep_tensors_;

    licm_analysis_data_t(const stmt_base_t *parent) : parent_(parent) {}
};

static bool unordered_intersects(const std::unordered_set<expr_c> &a,
        const std::unordered_set<expr_c> &b) {
    for (const auto &elem : a) {
        if (b.end() != std::find_if(b.begin(), b.end(), [&](const expr_c &v) {
                return v.ptr_same(elem);
            })) {
            return true;
        }
    }
    return false;
}

// Promotion depends on call, call nodes must be hoisted before this pass.
// Currently tensorptr can not be hoisted.
static bool expr_can_hoist(const expr_c &s) {
    return passlet::non_volatile_expr(s.get()) || s.isa<tensor>()
            || s.isa<indexing>();
}

static bool stmt_can_hoist(const stmt_c &s) {
    return s.isa<define_c>() || s.isa<if_else_c>() || s.isa<for_loop_c>()
            || s.isa<stmts_c>();
}

struct tensor_analysis_data_t {
    // The base tensor vars depends on
    std::unordered_set<expr_c> base_tensors_;
};
#define BASE_TENSORS(V) \
    (V)->temp_data().get<tensor_analysis_data_t>().base_tensors_

// Analyze call and tensor volatility in loops for licm promotion assessment
struct loop_analysis_viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;
    const stmt_base_t *current_ = nullptr;
    std::vector<expr_c> cur_loop_vars_;
    std::vector<std::unordered_set<expr_c>> cur_loop_ptrs_;
    std::unordered_map<expr_c, bool> call_volatile_map_;
    std::unordered_map<expr_c, std::unordered_set<expr_c>> base_tensor_map_;
    std::unordered_map<expr_c, std::unordered_set<expr_c>> tensor_volatile_map_;
    std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
            alias_map_;

    expr_c dispatch(expr_c s) override {
        if (!s->get_temp_data().isa<tensor_analysis_data_t>()) {
            s->temp_data() = tensor_analysis_data_t();
        }
        return ssa_viewer_t::dispatch(std::move(s));
    }

    stmt_c dispatch(stmt_c s) override {
        if (!s->get_temp_data().isa<tensor_analysis_data_t>()) {
            s->temp_data() = tensor_analysis_data_t();
        }
        auto old = current_;
        current_ = s.get();
        auto ret = ssa_viewer_t::dispatch(std::move(s));
        current_ = old;
        return ret;
    }

    void view(for_loop_c v) override {
        cur_loop_vars_.emplace_back(v->var_);
        cur_loop_ptrs_.emplace_back(std::unordered_set<expr_c>());
        call_volatile_map_[v->var_] = false;
        tensor_volatile_map_[v->var_] = std::unordered_set<expr_c>();
        ssa_viewer_t::view(v);
        // If inner loop call volatile, mark outer nested loops as call volatile
        if (call_volatile_map_[v->var_]) {
            for (const auto &loop_var : cur_loop_vars_) {
                call_volatile_map_[loop_var] = true;
            }
        }
        // If current loop contain volatile tensor, mark all alias as volatile
        auto &cur_tensor_volatile_map = tensor_volatile_map_[v->var_];
        for (const auto &tsr : cur_tensor_volatile_map) {
            auto alias = alias_info::get_alias_info(*tsr);
            if (!alias || alias->has_no_alias()) { continue; }
            for (auto &cliq : alias->alias_cliques_) {
                for (auto aid : cliq->set_) {
                    auto other_alias_id = aid.lock();
                    // if the tensor has been removed, skip
                    if (!other_alias_id) { continue; }
                    auto itr = alias_map_.find(other_alias_id.get());
                    if (itr != alias_map_.end()) {
                        cur_tensor_volatile_map.insert(itr->second);
                    }
                }
            }
        }
        // for each volatile base tensor, filter loop volatile pointers
        // mark all volatile pointers
        std::unordered_set<expr_c> volatile_ptrs;
        const auto &cur_loop_ptrs = cur_loop_ptrs_.back();
        for (const auto &tsr : cur_tensor_volatile_map) {
            auto iter = base_tensor_map_.find(tsr);
            if (iter != base_tensor_map_.end()) {
                auto &alias = iter->second;
                // only pointer used inside loop will be considered
                // avoiding marking other loop's volatile pointers
                for (const auto &a : alias) {
                    if (cur_loop_ptrs.find(a) != cur_loop_ptrs.end()) {
                        volatile_ptrs.insert(a);
                    }
                }
            }
        }
        cur_tensor_volatile_map.insert(
                volatile_ptrs.begin(), volatile_ptrs.end());
        // end of loop scope
        cur_loop_vars_.pop_back();
        cur_loop_ptrs_.pop_back();
    }

    void view(var_c v) override {
        // Prepare for volatile pointer filter
        if (v->dtype_.is_pointer() && !cur_loop_ptrs_.empty()) {
            for (auto &loop_ptrs : cur_loop_ptrs_) {
                loop_ptrs.insert(v);
            }
        }
    }

    void view(tensor_c v) override {
        // Prepare for tensor alias analysis
        auto alias = alias_info::get_alias_info(*v);
        if (alias) { alias_map_[alias] = v; }
        // Prepare for base tensor analysis
        BASE_TENSORS(v).insert(v);
    }

    void view(cast_c v) override {
        ssa_viewer_t::view(v);
        // Prepare for base tensor analysis
        auto &st_base = BASE_TENSORS(current_);
        auto &vi_base = BASE_TENSORS(v->in_);
        st_base.insert(vi_base.begin(), vi_base.end());
    }

    void view(binary_c v) override {
        ssa_viewer_t::view(v);
        // Prepare for base tensor analysis
        auto &st_base = BASE_TENSORS(current_);
        auto &vl_base = BASE_TENSORS(v->l_);
        auto &vr_base = BASE_TENSORS(v->r_);
        st_base.insert(vl_base.begin(), vl_base.end());
        st_base.insert(vr_base.begin(), vr_base.end());
    }

    void view(define_c v) override {
        ssa_viewer_t::view(v);
        if (v->var_.isa<var>()) {
            auto &st_base = BASE_TENSORS(current_);
            auto &va_base = BASE_TENSORS(v->var_);
            va_base.insert(st_base.begin(), st_base.end());
            if (v->var_->dtype_.is_pointer()) {
                for (const auto &tsr : va_base) {
                    base_tensor_map_[tsr].insert(v->var_);
                }
            }
        } else if (v->var_.isa<tensor>()) {
            for (const auto &loop_var : cur_loop_vars_) {
                auto &volatile_map = tensor_volatile_map_[loop_var];
                volatile_map.insert(v->var_);
            }
        }
    }

    void view(call_c v) override {
        ssa_viewer_t::view(v);
        // If loop contain call, mark this loop as call volatile
        if (!cur_loop_vars_.empty()) {
            call_volatile_map_[cur_loop_vars_.back()] = true;
        }
        // If loop contain volatile tensor, mark this loop and
        // outer nested loops with this volatile tensor
        // In call(&A[i]), A considered volatile
        if (!cur_loop_vars_.empty()) {
            for (const auto &arg : v->args_) {
                auto &base = BASE_TENSORS(arg);
                for (const auto &loop_var : cur_loop_vars_) {
                    auto &volatile_map = tensor_volatile_map_[loop_var];
                    volatile_map.insert(base.begin(), base.end());
                }
            }
        }
    }

    void view(assign_c v) override {
        ssa_viewer_t::view(v);
        // If loop contain volatile tensor, mark this loop and
        // outer nested loops with this volatile tensor
        // In A[i] = x, A considered volatile
        if (v->var_.isa<indexing>() && !cur_loop_vars_.empty()) {
            auto &base = BASE_TENSORS(v->var_.static_as<indexing>()->ptr_);
            for (const auto &loop_var : cur_loop_vars_) {
                auto &volatile_map = tensor_volatile_map_[loop_var];
                volatile_map.insert(base.begin(), base.end());
            }
        }
    }
};

static std::vector<stmt_c> *find_stmt_in_map(
        std::unordered_map<expr_c, std::vector<stmt_c>> &m, const stmt_c &st) {
    for (auto &it : m) {
        if (std::find_if(it.second.begin(), it.second.end(),
                    [&st](const stmt_c &in) { return in.ptr_same(st); })
                != it.second.end()) {
            return &it.second;
        }
    }
    return nullptr;
}

static stmt get_same_scope_owner(stmt owner, const stmt_base_t *scope) {
    while (owner.defined()
            && owner->temp_data().get_or_null<licm_analysis_data_t>()) {
        auto owner_scope
                = owner->temp_data().get<licm_analysis_data_t>().parent_;
        if (!owner_scope || owner_scope == scope) { break; }
        owner = owner_scope->node_ptr_from_this().remove_const();
    }
    return owner;
}

static void process_with_non_loop_phi(
        std::unordered_map<expr_c, std::vector<stmt_c>> &m,
        const std::vector<expr> &phi_values, const ssa_phi_c &phi) {
    size_t index = 0;
    bool dont_hoist = false;
    std::vector<stmt_c> *hoist_scope = nullptr;
    auto cur_scope = phi->temp_data().get<licm_analysis_data_t>().parent_;
    cur_scope = cur_scope->temp_data().get<licm_analysis_data_t>().parent_;
    while (index < phi_values.size()) {
        auto owner = phi_values[index]->ssa_data_->get_owner();
        owner = get_same_scope_owner(owner, cur_scope);
        if (owner.defined()) {
            hoist_scope = find_stmt_in_map(m, owner);
            break;
        }
        index++;
    }

    for (size_t i = index + 1; i < phi_values.size(); i++) {
        auto owner = phi_values[i]->ssa_data_->get_owner();
        owner = get_same_scope_owner(owner, cur_scope);
        if (owner.defined()) {
            if (hoist_scope) {
                // need all hoist in the same scope
                if (std::find_if(hoist_scope->begin(), hoist_scope->end(),
                            [&owner](const stmt_c &in) {
                                return in.ptr_same(owner);
                            })
                        == hoist_scope->end()) {
                    dont_hoist = true;
                    break;
                }
            } else {
                if (find_stmt_in_map(m, owner)) {
                    dont_hoist = true;
                    break;
                }
            }
        }
    }
    if (dont_hoist) {
        for (size_t i = 0; i < phi_values.size(); i++) {
            auto owner = phi_values[i]->ssa_data_->get_owner();
            owner = get_same_scope_owner(owner, cur_scope);
            if (owner.defined()
                    && owner->temp_data().get_or_null<licm_analysis_data_t>()) {
                owner->temp_data().get<licm_analysis_data_t>().volatile_ = true;
            }
        }
        auto owner = phi->ssa_data_->get_owner();
        if (owner.defined()
                && owner->temp_data().get_or_null<licm_analysis_data_t>()) {
            owner->temp_data().get<licm_analysis_data_t>().volatile_ = true;
        }
    }
}

struct licm_analysis_viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;
    const stmt_base_t *current_ = nullptr;
    //
    std::vector<stmt_c> cur_if_scopes_;
    std::vector<expr_c> cur_loop_vars_;
    // currently we treat if_scope as a special stmt to judge
    // how loop invariants interact with it
    std::unordered_map<expr_c, stmt_c> if_scope_loop_map_;
    std::unordered_map<stmt_c, std::unordered_set<expr_c>> if_scope_var_map_;
    // output map: loop var => vector of invariants, the invariants will
    // be inserted just before their correspond loop vars.
    std::unordered_map<expr_c, std::vector<stmt_c>> loop_invariant_map_;
    std::unordered_set<stmt_c> stmt_to_remove_set_;
    // input map: call and tensor info
    std::unordered_map<expr_c, bool> &call_volatile_map_;
    std::unordered_map<expr_c, std::unordered_set<expr_c>>
            &tensor_volatile_map_;
    //
    licm_analysis_viewer_t(std::unordered_map<expr_c, bool> &call_volatile_map,
            std::unordered_map<expr_c, std::unordered_set<expr_c>>
                    &tensor_volatile_map)
        : call_volatile_map_(call_volatile_map)
        , tensor_volatile_map_(tensor_volatile_map) {
        // Create a dummy if_scope
        cur_if_scopes_.emplace_back(
                builder::make_if_else_unattached(false, {}, {}));
    }
    void register_loop_invariant_stmt() {
        assert(current_ != nullptr);
        // if stmts (need follow parent if or for), return
        if (current_->node_type_ == sc_stmt_type::stmts) { return; }
        // if the stmt is not in loop, return.
        if (cur_loop_vars_.empty()) { return; }
        auto &st_data = current_->temp_data().get<licm_analysis_data_t>();
        if (st_data.volatile_) { return; }
        // Find the loop to promote, from inner-most to out-most
        auto it = cur_loop_vars_.rbegin();
        for (; it != cur_loop_vars_.rend(); it++) {
            // If any dep_tensors_ is defined or to be stored in the loop
            // cannot promote
            bool volatile_by_tensor = unordered_intersects(
                    st_data.dep_tensors_, tensor_volatile_map_[*it]);
            // If any dep_vars_ is the loop var, cannot promote
            bool volatile_by_var = //
                    st_data.dep_vars_.end()
                    != std::find_if(st_data.dep_vars_.begin(),
                            st_data.dep_vars_.end(),
                            [&](const expr_c &v) { return v.ptr_same(*it); });
            // If loop contains call node, cannot promote
            bool volatile_by_call = call_volatile_map_[*it];
            // If loop outside current if scope, cannot promote
            bool outside_if_scope
                    = !if_scope_loop_map_[*it].ptr_same(cur_if_scopes_.back());
            // If depends not volatile, continue find the loop to promote
            if (volatile_by_tensor || volatile_by_var || volatile_by_call
                    || outside_if_scope) {
                break;
            }
        }
        if (it != cur_loop_vars_.rbegin()) {
            stmt_c s = current_->node_ptr_from_this();
            loop_invariant_map_[*(it - 1)].emplace_back(s);
            stmt_to_remove_set_.insert(s);
            return;
        } else {
            // current_ is volatile in its loop, cannot promote
            return;
        }
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
        if_scope_loop_map_[v->var_] = cur_if_scopes_.back();
        cur_loop_vars_.emplace_back(v->var_);
        ssa_viewer_t::view(v);
        cur_loop_vars_.pop_back();
        auto &v_data = v->temp_data().get<licm_analysis_data_t>();
        auto &body_data = v->body_->temp_data().get<licm_analysis_data_t>();
        v_data.volatile_ |= body_data.volatile_;
        v_data.dep_vars_.insert(
                body_data.dep_vars_.begin(), body_data.dep_vars_.end());
        v_data.dep_tensors_.insert(
                body_data.dep_tensors_.begin(), body_data.dep_tensors_.end());
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
            v_data.dep_tensors_.insert(
                    st_data.dep_tensors_.begin(), st_data.dep_tensors_.end());
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
            if_scope_var_map_[cur_if_scopes_.back()].insert(v->var_);
        } else if (v->var_.isa<tensor>() && !cur_loop_vars_.empty()) {
            auto &var_data = v->var_->temp_data().get<licm_analysis_data_t>();
            var_data.dep_vars_.insert(cur_loop_vars_.back());
            st_data.volatile_ = true;
        }
    }
    void view(if_else_c v) override {
        cur_if_scopes_.emplace_back(v);
        ssa_viewer_t::view(v);
        auto &st_data = v->temp_data().get<licm_analysis_data_t>();
        auto &then_data
                = v->then_case_->temp_data().get<licm_analysis_data_t>();
        st_data.volatile_ |= then_data.volatile_;
        st_data.dep_vars_.insert(
                then_data.dep_vars_.begin(), then_data.dep_vars_.end());
        st_data.dep_tensors_.insert(
                then_data.dep_tensors_.begin(), then_data.dep_tensors_.end());
        if (v->else_case_.defined()) {
            auto &else_data
                    = v->else_case_->temp_data().get<licm_analysis_data_t>();
            st_data.volatile_ |= else_data.volatile_;
            st_data.dep_vars_.insert(
                    else_data.dep_vars_.begin(), else_data.dep_vars_.end());
            st_data.dep_tensors_.insert(else_data.dep_tensors_.begin(),
                    else_data.dep_tensors_.end());
        }
        // Pass down all dependencies to vars defined in this if_else_c
        auto &vars_in_if_scope_ = if_scope_var_map_[v];
        for (auto &var : vars_in_if_scope_) {
            auto &var_data = var->temp_data().get<licm_analysis_data_t>();
            var_data.volatile_ |= st_data.volatile_;
            var_data.dep_vars_.insert(
                    st_data.dep_vars_.begin(), st_data.dep_vars_.end());
            var_data.dep_tensors_.insert(
                    st_data.dep_tensors_.begin(), st_data.dep_tensors_.end());
        }
        vars_in_if_scope_.clear();
        // End of this if_scopes
        cur_if_scopes_.pop_back();
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
        st_data.dep_tensors_.insert(
                var_data.dep_tensors_.begin(), var_data.dep_tensors_.end());
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
    void view(tensor_c v) override {
        auto &is_param = v->ssa_data_->is_param_;
        if (current_ == nullptr) {
            assert(is_param);
            return;
        }
        auto &st_data = current_->temp_data().get<licm_analysis_data_t>();
        auto &var_data = v->temp_data().get<licm_analysis_data_t>();
        st_data.dep_vars_.insert(
                var_data.dep_vars_.begin(), var_data.dep_vars_.end());
        if (var_data.volatile_) {
            st_data.volatile_ = true;
            return;
        }
    }
    void view(indexing_c v) override {
        ssa_viewer_t::view(v);
        auto &st_data = current_->temp_data().get<licm_analysis_data_t>();
        st_data.dep_tensors_.insert(v->ptr_);
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
        } else if (!cur_loop_vars_.empty() && v->values_.size() > 1) {
            process_with_non_loop_phi(loop_invariant_map_, v->values_, v);
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
    // if_else and for_loop may changed after dispatch
    std::unordered_map<stmt_c, stmt_c> stmt_replace_map_;
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
            if (stmt_to_remove_.find(s) != stmt_to_remove_.end()) {
                assert(s->temp_data().get<licm_analysis_data_t>().volatile_
                        == false);
                changed = true;
                continue;
            }
            if (ret.isa<for_loop_c>()) {
                auto it = hoist_map_.find(ret.static_as<for_loop_c>()->var_);
                if (it != hoist_map_.end()) {
                    std::vector<stmt_c> *cur_hoist_stmts = &(it->second);
                    // ret_vec insert cur_hoist_stmts with replacement
                    for (auto &s : *cur_hoist_stmts) {
                        auto rep = stmt_replace_map_.find(s);
                        if (rep != stmt_replace_map_.end()) {
                            ret_vec.push_back(rep->second);
                        } else {
                            ret_vec.push_back(s);
                        }
                    }
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

    stmt_c visit(for_loop_c v) override {
        auto vv = ssa_visitor_t::visit(v);
        if (!vv.ptr_same(v)) { stmt_replace_map_[v] = vv; }
        return vv;
    }

    stmt_c visit(if_else_c v) override {
        auto vv = ssa_visitor_t::visit(v);
        if (!vv.ptr_same(v)) { stmt_replace_map_[v] = vv; }
        return vv;
    }
};

func_c loop_invariant_code_motion_t::operator()(func_c f) {
    // Analyze loop first
    loop_analysis_viewer_t loop_analyzer;
    loop_analyzer.dispatch(f);
    // Mark loop-invariant code
    licm_analysis_viewer_t analyzer(loop_analyzer.call_volatile_map_,
            loop_analyzer.tensor_volatile_map_);
    analyzer.dispatch(f);
    // Move code
    filter_stmt_by_volatile(
            analyzer.loop_invariant_map_, analyzer.stmt_to_remove_set_);
    licm_hoister_t hoister(
            analyzer.loop_invariant_map_, analyzer.stmt_to_remove_set_);
    return hoister.top_level_dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
