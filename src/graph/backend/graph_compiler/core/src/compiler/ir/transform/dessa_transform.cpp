/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "dessa_transform.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <util/any_map.hpp>
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(dessa_transform, SC_PASS_DEPENDS_ON(ssa_transform),
        SC_PASS_REQUIRE_STATE(SSA_STAGE), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(SSA_STAGE));

struct dessa_analysis_data_t {
    std::vector<const stmt_base_t *> uses_;
    const stmt_base_t *parent_;

    // if this var-define node should be removed
    bool should_remove_ = false;
    // the stmt to be inserted at the begining of the current stmts. If the
    // current stmt is not stmts, it should be empty
    std::vector<stmt> inserted_within_;
    // the stmt to be inserted after the current stmts.
    std::vector<stmt> inserted_after_;
    // the copy for the var for loop-phi to solve lost-copy and swap problem
    expr shadow_phi_var_;

    dessa_analysis_data_t(const stmt_base_t *parent) : parent_(parent) {}
};

// Loop phi coalesce is achieved by checking the usage of a, b, c
// in a = phi(b, c loop) is not overlapped and there is no interference.
// ssa_analysis_viewer_t will set to check b, c usage after a is defined
// and check a, b usage after c is defined, so if there is interference
// between any vars in any sub loops, all vars will not be coalesced.
// This will coalesce all loop phi vars in reduce situations
struct loop_phi_coalesce_data_t {
    // check for interference after this var is defined
    std::weak_ptr<expr_base> check_after_define_;
    // the final coalesced var
    std::weak_ptr<expr_base> coalesced_var_;
    // flag for var defined inside loop
    bool from_loop_ = false;
    // flag for interfered phi vars
    bool interfered_ = false;
    // flag for checking interference
    // when check_ is true and the var is used
    // interfered_ will be set to true
    bool check_ = false;
    // flag for preserved var, a copy has been made
    // original var can be used in other node
    bool preserved_ = false;

    loop_phi_coalesce_data_t(const expr &v = expr())
        : check_after_define_(v.impl) {}
};

// get loop_phi_coalesce_data_t or null
static inline loop_phi_coalesce_data_t *get_coalesce_data(const expr_c &v) {
    if (!v->get_temp_data().isa<loop_phi_coalesce_data_t>()) { //
        return nullptr;
    }
    return &(v->temp_data().get<loop_phi_coalesce_data_t>());
}
// get loop_phi_coalesce_data_t or create new
static inline loop_phi_coalesce_data_t *get_or_create_coalesce_data(
        const expr_c &v, const expr &c = expr()) {
    if (!v->get_temp_data().isa<loop_phi_coalesce_data_t>()) {
        v->temp_data() = loop_phi_coalesce_data_t(c);
    }
    return &(v->temp_data().get<loop_phi_coalesce_data_t>());
}

// Merge related loop phi var interfered data,
// if one interfered, mark all interfered
static inline bool merge_interfered_data(
        const expr_c &var, const ssa_phi_c &phi) {
    if (!get_coalesce_data(var)) { return true; }
    bool interfered = get_coalesce_data(var)->interfered_;
    for (auto &val : phi->values_) {
        if (!get_coalesce_data(val)) { return true; }
        interfered |= get_coalesce_data(val)->interfered_;
    }
    get_coalesce_data(var)->interfered_ = interfered;
    for (auto &val : phi->values_) {
        get_coalesce_data(val)->interfered_ = interfered;
    }
    return interfered;
}

// Get final coalesced var
static inline expr get_coalesced_from_var(const expr_c &v) {
    if (get_coalesce_data(v)) {
        auto ptr = get_coalesce_data(v)->coalesced_var_;
        if (!utils::is_uninitialized_weakptr(ptr)) {
            assert(ptr.lock());
            return ptr.lock()->node_ptr_from_this();
        }
    }
    return expr();
}
// Get if the coalesced var is preserved
static inline bool get_perservd_from_var(const expr_c &v) {
    return get_coalesce_data(v) ? get_coalesce_data(v)->preserved_ : false;
}
// Get final coalesced var from phi node, if incoming coalesced ptr_same
static inline expr get_coalesced_from_phi(const ssa_phi_c &phi) {
    assert(phi->values_.size() == 2);
    auto c0 = get_coalesced_from_var(phi->values_[0]);
    auto c1 = get_coalesced_from_var(phi->values_[1]);
    return c0.defined() && c1.defined() && c0.ptr_same(c1) ? c0 : expr();
}
// Merge final coalesced var, set to the first incoming phi var
static inline void merge_coalesced_var(
        const expr_c &thevar, const ssa_phi_c &phi, const stmt &toplevel) {
    assert(phi->values_.size() == 2);
    auto &val_0 = phi->values_[0];
    auto &val_1 = phi->values_[1];

    expr coalesced = get_coalesced_from_var(val_0);

    if (!coalesced.defined()) {
        // perserve the outer loop defined var
        get_coalesce_data(val_0)->preserved_ = true;
        // make a copy of outer loop var
        coalesced = thevar->remake();
        coalesced.static_as<var>()->name_ += "_coalesced";
        // insert define after original var
        auto coalesced_def = builder::make_var_tensor_def_unattached(
                coalesced, linkage::local, val_0);
        auto valowner = val_0->ssa_data_->get_owner();
        if (valowner.defined()) {
            auto &tempdata = valowner->temp_data().get<dessa_analysis_data_t>();
            tempdata.inserted_after_.emplace_back(coalesced_def);
        } else {
            auto &tempdata = toplevel->temp_data().get<dessa_analysis_data_t>();
            tempdata.inserted_within_.emplace_back(coalesced_def);
        }
    }

    get_coalesce_data(thevar)->coalesced_var_ = coalesced.impl;
    get_coalesce_data(val_0)->coalesced_var_ = coalesced.impl;
    get_coalesce_data(val_1)->coalesced_var_ = coalesced.impl;
}

struct ssa_analysis_viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;
    const stmt_base_t *current_ = nullptr;

    std::vector<define_c> defines_;
    // all the phi node defined in the current loop
    std::vector<std::vector<std::pair<expr, ssa_phi>>> curr_loop_phi_;
    // restore parent loop's phi vars check state after curr loop
    std::vector<std::vector<std::pair<expr, bool>>> curr_check_state_;

    void process_define(const expr &var, const expr &init = expr()) {
        // process var data for interference check
        if (get_coalesce_data(var)) {
            auto data = get_coalesce_data(var);
            auto ptr = data->check_after_define_;
            if (!utils::is_uninitialized_weakptr(ptr)) {
                assert(ptr.lock());
                auto node = ptr.lock()->node_ptr_from_this();
                // set to check interference for merged node
                get_coalesce_data(node)->check_ = true;
                get_coalesce_data(var)->check_ = false;
            }
        } else {
            var->temp_data() = loop_phi_coalesce_data_t();
        }
        // If the var is defined not inside any loop, a copy will
        // be made when coalescing, and orignal var is preserved
        bool from_loop = !curr_loop_phi_.empty();
        get_coalesce_data(var)->from_loop_ = from_loop;
        // process loop phi define
        if (init.defined() && init.isa<ssa_phi>()) {
            auto phi = init.static_as<ssa_phi>();
            if (phi->is_loop_phi_) {
                assert(phi->values_.size() == 2);
                auto &phi_val_0 = phi->values_[0];
                auto &phi_val_1 = phi->values_[1];
                if (get_coalesce_data(phi_val_0)
                        && get_coalesce_data(phi_val_1)) {
                    // critical loop var defined before current phi var
                    // exist interference (potentially swap problem)
                    // may need parallel copy implementation to resolve
                    get_coalesce_data(var)->interfered_ = true;
                    get_coalesce_data(phi_val_0)->interfered_ = true;
                    get_coalesce_data(phi_val_1)->interfered_ = true;
                } else if (phi_val_0.isa<constant>()) {
                    // TODO(longsheng): deal with uninitialized vars
                    get_or_create_coalesce_data(var)->interfered_ = true;
                    get_or_create_coalesce_data(phi_val_0)->interfered_ = true;
                    get_or_create_coalesce_data(phi_val_1)->interfered_ = true;
                } else {
                    // assume phi_val_0 is the incoming var
                    // assume phi_val_1 is the critical var
                    // save incoming phi check state
                    COMPILE_ASSERT(get_coalesce_data(phi_val_0),
                            "incoming phi var should be defined already.");
                    auto &curr_state = curr_check_state_.back();
                    bool check = get_coalesce_data(phi_val_0)->check_;
                    curr_state.emplace_back(std::make_pair(phi_val_0, check));
                    // set to check interference for incoming phi vars
                    // and critical phi vars
                    bool v0_check = get_coalesce_data(phi_val_0)->from_loop_;
                    get_coalesce_data(phi_val_0)->check_ = v0_check;
                    phi_val_1->temp_data() = loop_phi_coalesce_data_t(var);
                    get_coalesce_data(phi_val_1)->check_ = true;
                }
                // record loop phi in current loop
                curr_loop_phi_.back().emplace_back(std::make_pair(var, phi));
            } else {
                // if phi merged loop phi vars inherent the interferece state
                merge_interfered_data(var, phi);
                if (phi->values_.size() == 2) {
                    auto v0_data = get_coalesce_data(phi->values_[0]);
                    auto v1_data = get_coalesce_data(phi->values_[1]);
                    bool v0_interfered = v0_data ? v0_data->interfered_ : true;
                    bool v1_interfered = v1_data ? v1_data->interfered_ : true;
                    bool interfered = v0_interfered || v1_interfered;
                    bool v0_check = v0_data && v0_data->from_loop_;
                    if (v0_data && !interfered) { v0_data->check_ = v0_check; }
                    if (v1_data && !interfered) { v1_data->check_ = true; }
                }
            }
        }
    }

    func_c dispatch(func_c v) override {
        for (auto &p : v->params_) {
            process_define(p);
        }
        return ssa_viewer_t::dispatch(std::move(v));
    }

    stmt_c dispatch(stmt_c s) override {
        if (!s->get_temp_data().isa<dessa_analysis_data_t>()) {
            s->temp_data() = dessa_analysis_data_t(current_);
        } else {
            // if the vardef is used before define, update the parent
            auto &dessa_data = s->temp_data().get<dessa_analysis_data_t>();
            assert(dessa_data.parent_ == nullptr);
            dessa_data.parent_ = current_;
        }
        auto old = current_;
        current_ = s.get();
        auto ret = ssa_viewer_t::dispatch(std::move(s));
        current_ = old;
        return ret;
    }

    void view(for_loop_c v) override {
        curr_loop_phi_.emplace_back(std::vector<std::pair<expr, ssa_phi>>());
        curr_check_state_.emplace_back(std::vector<std::pair<expr, bool>>());
        ssa_viewer_t::view(v);
        // merge current loop phi var interference
        for (auto &kv : curr_loop_phi_.back()) {
            merge_interfered_data(kv.first, kv.second);
        }
        curr_loop_phi_.pop_back();
        // recover incoming phi check state
        for (auto &kv : curr_check_state_.back()) {
            get_coalesce_data(kv.first)->check_ = kv.second;
        }
        curr_check_state_.pop_back();
    }

    void view(define_c v) override {
        if (v->init_.defined()) { dispatch(v->init_); }
        process_define(v->var_, v->init_);
        defines_.emplace_back(std::move(v));
    }

    void view(var_c v) override {
        auto owner = v->ssa_data_->get_owner();
        if (owner.defined()) {
            if (auto dessa_data
                    = owner->temp_data().get_or_null<dessa_analysis_data_t>()) {
                dessa_data->uses_.emplace_back(current_);
            } else {
                // if it is a use-before-define, this usually means that this is
                // in a phi node in for-loop. We insert a placeholder for the
                // var-def
                owner->temp_data() = dessa_analysis_data_t(nullptr);
                owner->temp_data()
                        .get<dessa_analysis_data_t>()
                        .uses_.emplace_back(current_);
            }
        }
        if (get_coalesce_data(v) && get_coalesce_data(v)->check_) {
            // check for loop phi var interference
            get_coalesce_data(v)->interfered_ = true;
        }
    }
};

static var_node *get_var_if_is_define(const stmt_c &s) {
    if (s.isa<define>()) {
        auto def = s.static_as<define>();
        if (def->var_.isa<var>() && def->var_->ssa_data_) {
            // assert(def->var_->ssa_data_);
            return def->var_.static_as<var>().get();
        }
    }
    return nullptr;
}

struct ssa_mutator_t : public ir_visitor_t {
    using ir_visitor_t::dispatch;
    std::vector<stmt> inserted_after;
    stmt_c dispatch(stmt_c v) override {
        auto &dessa_data = v->get_temp_data().get<dessa_analysis_data_t>();
        if (v.isa<stmts>()) {
            assert(dessa_data.inserted_after_.empty());
        } else {
            assert(dessa_data.inserted_within_.empty());
        }
        if (dessa_data.should_remove_) { return stmt_c(); }
        return ir_visitor_t::dispatch(std::move(v));
    }

    expr_c visit(var_c v) override {
        auto coalesced = get_coalesced_from_var(v);
        auto preserved = get_perservd_from_var(v);
        if (coalesced.defined() && !preserved) { return coalesced; }
        return v;
    }

    stmt_c visit(define_c v) override {
        if (v->var_.isa<var>()
                && v->var_.static_as<var>()->dtype_ == datatypes::void_t) {
            assert(v->init_.defined());
            return builder::make_evaluate_unattached(
                    ir_visitor_t::dispatch(v->init_));
        }
        return ir_visitor_t::visit(std::move(v));
    }

    stmt_c visit(evaluate_c v) override {
        if (v->value_.isa<var>()
                && v->value_.static_as<var>()->dtype_ == datatypes::void_t) {
            return stmt_c();
        }
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt_c> ret_vec;
        bool changed = false;
        // if the current stmts has "inserted_within_"
        auto &dessa_data = v->get_temp_data().get<dessa_analysis_data_t>();
        ret_vec.insert(ret_vec.end(), dessa_data.inserted_within_.begin(),
                dessa_data.inserted_within_.end());
        for (auto &s : v->seq_) {
            if (auto var_p = get_var_if_is_define(s)) {
                if (var_p->ssa_data_->is_garbage()) {
                    changed = true;
                    continue;
                }
            }
            auto sz_before = ret_vec.size();
            auto ret = dispatch(s);

            // if an stmt is inserted, the IR is changed
            changed |= sz_before != ret_vec.size();
            changed |= !ret.ptr_same(s);
            // we allow the return value to be empty, which means we need to
            // remove the stmt from parent
            if (!ret.defined()) {
                changed = true;
            } else {
                ret_vec.emplace_back(std::move(ret));
            } // insert "inserted_after_"
            auto &s_dessa_data
                    = s->get_temp_data().get<dessa_analysis_data_t>();
            if (!s_dessa_data.inserted_after_.empty()) {
                std::transform(s_dessa_data.inserted_after_.begin(),
                        s_dessa_data.inserted_after_.end(),
                        std::back_inserter(ret_vec), [this](stmt in) {
                            return ir_visitor_t::dispatch(std::move(in))
                                    .remove_const();
                        });
                changed = true;
            }
        }
        if (!changed) {
            return v;
        } else {
            return copy_attr(*v, builder::make_stmts_unattached(ret_vec));
        }
    }
};

static const stmt_base_t *get_parent(const stmt_base_t *cur) {
    return cur->get_temp_data().get<dessa_analysis_data_t>().parent_;
}

static bool is_parent_of(const stmt_base_t *parent, const stmt_base_t *other) {
    const stmt_base_t *cur = other;
    while (cur) {
        if (cur == parent) { return true; }
        cur = get_parent(cur);
    }
    return false;
}

static const stmt_base_t *find_parent_stmts(const stmt_base_t *other) {
    const stmt_base_t *cur = other;
    while (cur) {
        if (cur->node_type_ == sc_stmt_type::stmts) { return cur; }
        cur = get_parent(cur);
    }
    assert(0 && "bad dessa_analysis_data_t");
    return nullptr;
}

static const stmt_base_t *find_parent_for(const stmt_base_t *other) {
    const stmt_base_t *cur = other;
    while (cur) {
        if (cur->node_type_ == sc_stmt_type::for_loop) { return cur; }
        cur = get_parent(cur);
    }
    return nullptr;
}

static const stmt_base_t *find_shared_parent(
        const stmt_base_t *lhs, const stmt_base_t *other) {
    auto cur = lhs;
    while (cur) {
        if (is_parent_of(cur, other)) { return cur; }
        cur = get_parent(cur);
    }
    assert(0 && "bad dessa_analysis_data_t");
    return nullptr;
}

static const stmt_base_t *find_shared_parent_for_uses(const stmt_base_t *cur) {
    const dessa_analysis_data_t &data
            = cur->get_temp_data().get<dessa_analysis_data_t>();
    for (auto use : data.uses_) {
        cur = find_shared_parent(cur, use);
    }
    return find_parent_stmts(cur);
}

// If the loop phi is coalesced, use the coalesced var to replace the usage of
// all merged loop phi vars and romove the define.
// For non-loop phis, if the 2 merged vars are both coalesced to same var,
// use the coalesced var to replace the phi node
static void coalesce_var_for_phi(const expr &coalesced, const expr &var_for_phi,
        const stmt_base_t *cur) {
    dessa_analysis_data_t &data = cur->temp_data().get<dessa_analysis_data_t>();
    auto def = cur->node_ptr_from_this().checked_as<define>();
    data.should_remove_ = true; // remove the phi node

    auto coalesced_loop_var = get_coalesced_from_var(var_for_phi);
    if (!coalesced_loop_var.defined()) {
        // this is a non loop phi merge, use coalesced var as init to define var
        auto newnode = def->remake();
        newnode.static_as<define>()->init_ = coalesced;
        data.inserted_after_.insert(data.inserted_after_.begin(), newnode);
    } else if (!coalesced_loop_var.ptr_same(coalesced)) {
        // this is a non loop phi merge, but src and dst are both coalesced,
        // use coalesced var as init to assign to other coalesced var
        data.inserted_after_.insert(data.inserted_after_.begin(),
                builder::make_assign_unattached(coalesced_loop_var, coalesced));
    }
}

// in SSA form, a var may be defined in a child scope (e.g. within if-else), but
// is used in parent scope. This function inserts a var def in parent scope and
// replaces the var def in child scope with an assignment to make the IR valid.
// returns true if the var def needs to be moved
static bool promote_scope_for_non_phi_var(
        const define_c &def, const expr &value) {
    dessa_analysis_data_t &curdata
            = def->temp_data().get<dessa_analysis_data_t>();
    auto coalesced = get_coalesced_from_var(def->var_);
    if (coalesced.defined()) {
        // The var is coalesced
        curdata.should_remove_ = true;
        curdata.inserted_after_.emplace_back(
                builder::make_assign_unattached(coalesced, value));
        return false;
    }
    auto shared_parent = find_shared_parent_for_uses(def.get());
    if (shared_parent != get_parent(def.get())) {
        curdata.should_remove_ = true;
        if (value->dtype_ == datatypes::void_t) {
            if (!value.isa<var>()) {
                curdata.inserted_after_.emplace_back(
                        builder::make_evaluate_unattached(value));
            }
        } else {
            shared_parent->temp_data()
                    .get<dessa_analysis_data_t>()
                    .inserted_within_.emplace_back(
                            builder::make_var_tensor_def_unattached(
                                    def->var_, def->linkage_));
            curdata.inserted_after_.insert(curdata.inserted_after_.begin(),
                    builder::make_assign_unattached(def->var_, value));
        }
        return true;
    }
    return false;
}

// insert copying to phi vars after an input of the phi node is computed. If a
// phi input is defined in IR function parameters, put the copy in the top-level
// stmts
static void insert_value_copy_for_phi(const ssa_phi &phi,
        const expr &var_for_phi, const stmt_base_t *cur, const stmt &toplevel) {
    dessa_analysis_data_t &data = cur->temp_data().get<dessa_analysis_data_t>();
    const stmt_base_t *shared_parent = cur;
    const expr &the_target_var = data.shadow_phi_var_.defined()
            ? data.shadow_phi_var_
            : var_for_phi;
    // if the phi has an input from parameter
    expr init_val_for_param;
    for (auto &val : phi->values_) {
        auto valowner = val->ssa_data_->get_owner();
        if (!valowner.defined()) {
            COMPILE_ASSERT(!init_val_for_param.defined(),
                    "PHI node can only accept only one parameter input");
            init_val_for_param = val;
            valowner = toplevel;
            shared_parent = toplevel.get();
        } else {
            auto &tempdata = valowner->temp_data().get<dessa_analysis_data_t>();
            shared_parent = find_shared_parent(shared_parent, valowner.get());
            tempdata.inserted_after_.emplace_back(
                    builder::make_assign_unattached(the_target_var, val));
        }
    }
    shared_parent = find_shared_parent(
            shared_parent, find_shared_parent_for_uses(cur));
    shared_parent = find_parent_stmts(shared_parent);
    auto &parent_data = shared_parent->temp_data().get<dessa_analysis_data_t>();
    parent_data.inserted_within_.emplace_back(
            builder::make_var_tensor_def_unattached(
                    var_for_phi, linkage::local, init_val_for_param));
    if (data.shadow_phi_var_.defined()) {
        parent_data.inserted_within_.emplace_back(
                builder::make_var_tensor_def_unattached(data.shadow_phi_var_,
                        linkage::local, init_val_for_param));
    }
    data.should_remove_ = true; // remove the phi node
}

static void process_phi(
        const ssa_phi &phi, const stmt_base_t *cur, const stmt &toplevel) {
    dessa_analysis_data_t &data = cur->temp_data().get<dessa_analysis_data_t>();
    auto def = cur->node_ptr_from_this().checked_as<define>();
    auto thevar = def->var_.checked_as<var>();
    if (phi->values_.size() > 1) {
        // first check if this phi depends on a value in for-loop
        const stmt_base_t *cur_for = find_parent_for(cur);
        bool is_loop_phi = phi->is_loop_phi_;

        if (is_loop_phi) {
            COMPILE_ASSERT(cur_for, "Cannot find parent for-loop for loop phi");
            if (!merge_interfered_data(thevar, phi)) {
                // No interference between loop_phi vars
                merge_coalesced_var(thevar, phi, toplevel);
            } else {
                // the phi is a loop-phi
                data.shadow_phi_var_ = builder::make_var(
                        thevar->dtype_, thevar->name_ + "_shadow");
                auto cur_for_body
                        = static_cast<const for_loop_node_t *>(cur_for)->body_;
                // copy the value in shadow to old var at the begining of the
                // for-loop
                cur_for_body->temp_data()
                        .get<dessa_analysis_data_t>()
                        .inserted_within_.emplace_back(
                                builder::make_assign_unattached(
                                        thevar, data.shadow_phi_var_));
            }
        }
        expr coalesced = get_coalesced_from_phi(phi);
        if (coalesced.defined()) {
            coalesce_var_for_phi(coalesced, thevar, cur);
        } else {
            insert_value_copy_for_phi(phi, thevar, cur, toplevel);
        }
    } else {
        // not-phi
        if (!promote_scope_for_non_phi_var(def, phi->values_.front())) {
            // if the current scope is OK for the node
            data.should_remove_ = true;
            auto newnode = def->remake();
            newnode.static_as<define>()->init_ = phi->values_.front();
            data.inserted_after_.emplace_back(std::move(newnode));
        }
    }
}

func_c dessa_transform_t::operator()(func_c f) {
    ssa_analysis_viewer_t analyzer;
    analyzer.dispatch(f);
    for (auto &def : analyzer.defines_) {
        if (def->var_.isa<var>() && def->init_.defined()) {
            if (def->init_.isa<ssa_phi>()) {
                auto phi = def->init_.static_as<ssa_phi>();
                process_phi(phi, def.get(), f->body_);
            } else {
                assert(def->init_.defined());
                promote_scope_for_non_phi_var(def, def->init_);
            }
        }
    }
    ssa_mutator_t mutator;
    return mutator.dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
