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
#include "ssa_transform.hpp"
#include <map>
#include <utility>
#include <vector>
#include "module_globals_resolve.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(ssa_transform,
        SC_PASS_DEPENDS_ON(module_globals_resolver, local_tensor_lowering_cpu,
                closurizer_cpu, buffer_scheduler),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED, IR_SIMPLIFIED, FUNC_INLINED),
        SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(SSA_STAGE),
        SC_PASS_UNSET_STATE());

struct ssa_var_status_t {
    expr current_value;
    size_t defined_scope_idx;
    // the phi node for the variable that is referenced in the current
    // for-loop, which is defined outside of the loop
    std::vector<expr> for_loop_phi;
};

struct var_cmper_t {
    bool operator()(const expr_c &l, const expr_c &r) const {
        if (l->node_type_ != r->node_type_) {
            return static_cast<int>(l->node_type_)
                    < static_cast<int>(r->node_type_);
        }
        if (l->node_type_ == sc_expr_type::var) {
            const auto &l_name = l.static_as<var>()->name_;
            const auto &r_name = r.static_as<var>()->name_;
            return (l_name == r_name) ? l.get() < r.get() : l_name < r_name;
        } else {
            assert(l->node_type_ == sc_expr_type::tensor);
            const auto &l_name = l.static_as<tensor>()->name_;
            const auto &r_name = r.static_as<tensor>()->name_;
            return (l_name == r_name) ? l.get() < r.get() : l_name < r_name;
        }
    }
};

struct ssa_scope_t {
    // old var => ssa_var_status_t. Using ordered map to make unit tests happy
    std::map<expr_c, ssa_var_status_t, var_cmper_t> vars_;
    std::vector<stmt> inserted_;
    enum class kind {
        normal,
        for_loop,
        if_then,
        if_else,
    };
    kind kind_;
    int for_depth_;

    ssa_scope_t(int for_depth, kind kind)
        : kind_(kind), for_depth_(for_depth) {}
};

class ssa_transform_impl_t : public ssa_visitor_t {
public:
    using ssa_visitor_t::dispatch;
    using ssa_visitor_t::visit;
    std::vector<ssa_scope_t> scopes_;
    // if the current expr needs to be flatten in dispatch()
    bool need_flatten_ = true;
    expr add_ssa_def(const expr_c &ret) {
        // if is global variable, add a "load instance"
        auto newret = add_def(ret);
        // copy the function pointer prototype
        if (newret->dtype_ == datatypes::pointer && ret->attr_) {
            if (auto proto = ret->attr_->get_or_else("prototype", func_t())) {
                newret->attr()["prototype"] = proto;
            }
        }
        return newret;
    }

    define add_ssa_def_to_parent_scope(const expr_c &ret, ssa_scope_t *scope) {
        // if is global variable, add a "load instance"
        auto newret = make_def_and_process(ret);
        // copy the function pointer prototype
        if (ret->dtype_ == datatypes::pointer && ret->attr_) {
            if (auto proto = ret->attr_->get_or_else("prototype", func_t())) {
                newret->var_->attr()["prototype"] = proto;
            }
        }
        scope->inserted_.emplace_back(newret);
        return newret;
    }

    ssa_scope_t &push_scope(ssa_scope_t::kind k) {
        int for_depth;
        if (scopes_.empty()) {
            for_depth = 0;
        } else {
            for_depth = scopes_.back().for_depth_;
        }
        if (k == ssa_scope_t::kind::for_loop) { for_depth++; }
        scopes_.emplace_back(for_depth, k);
        return scopes_.back();
    }

    ssa_scope_t pop_scope() {
        auto ret = std::move(scopes_.back());
        scopes_.pop_back();
        return ret;
    }

    // add an old var definition to scopes, returns the new var
    ssa_var_status_t *insert_local_var(
            const expr_c &old_var, const expr &new_val, size_t scope_idx) {
        auto itr = scopes_[scope_idx].vars_.insert(std::make_pair(old_var,
                ssa_var_status_t {new_val, scope_idx, std::vector<expr>()}));
        return &itr.first->second;
    }

    // add an old var definition to scopes, returns the new var
    ssa_var_status_t *insert_local_var(
            const expr_c &old_var, const expr &new_val) {
        return insert_local_var(old_var, new_val, scopes_.size() - 1);
    }

    ssa_var_status_t *get_local_var_nothrow(const expr_c &old_var) {
        for (auto itr = scopes_.rbegin(); itr != scopes_.rend(); ++itr) {
            auto varitr = (*itr).vars_.find(old_var);
            if (varitr != (*itr).vars_.end()) { return &varitr->second; }
        }
        return nullptr;
    }

    // find the scope id where the local var is first defined
    int64_t get_local_var_top_level(const expr_c &old_var) {
        for (auto itr = scopes_.begin(); itr != scopes_.end(); ++itr) {
            auto varitr = (*itr).vars_.find(old_var);
            if (varitr != (*itr).vars_.end()) { return itr - scopes_.begin(); }
        }
        throw std::runtime_error("Undefined var in SSA transform");
    }

    ssa_var_status_t *get_local_var(const expr_c &old_var) {
        auto ret = get_local_var_nothrow(old_var);
        COMPILE_ASSERT(ret, "Undefined var:" << old_var);
        return ret;
    }

    ssa_var_status_t *get_local_var_for_update(const expr_c &old_var) {
        if (is_old_var_global(old_var.get())) { return get_local_var(old_var); }
        auto &back = scopes_.back();
        auto varitr = back.vars_.find(old_var);
        if (varitr != back.vars_.end()) { return &varitr->second; }
        return insert_local_var(old_var, expr());
    }

    bool is_old_var_global(const expr_base *old_var) {
        return old_var->node_type_ == sc_expr_type::var && old_var->attr_
                && old_var->attr_->has_key(attr_keys::module_global_offset);
    }

    void set_var_as_global(const expr_c &old_var) {
        if (!old_var->ssa_data_) {
            old_var.remove_const()->ssa_data_
                    = utils::make_unique<ssa_data_t>();
        }
        old_var->ssa_data_->is_global_ = true;
    }

    ssa_data_t *init_ssa_data(expr_base *ex) {
        assert(!ex->ssa_data_);
        ex->ssa_data_ = utils::make_unique<ssa_data_t>();
        return ex->ssa_data_.get();
    }

    expr_c dispatch(expr_c f) override {
        bool old_need_flatten = need_flatten_;
        need_flatten_ = true;
        auto ret = ssa_visitor_t::dispatch(std::move(f));
        if (old_need_flatten && !ret.isa<var>() && !ret.isa<tensor>()) {
            return add_ssa_def(ret);
        }
        return ret;
    }

    func_c dispatch(func_c f) override {
        push_scope(ssa_scope_t::kind::normal);
        std::vector<expr> new_params;
        for (auto &p : f->params_) {
            auto newp = p->remake();
            init_ssa_data(newp.get())->is_param_ = true;
            insert_local_var(p, newp);
            new_params.emplace_back(std::move(newp));
        }
        auto body = dispatch(f->body_);
        pop_scope();
        return copy_attr(*f,
                builder::make_func(f->name_, new_params, body.remove_const(),
                        f->ret_type_));
    }

    expr_c visit(tensor_c v) override {
        auto ret = get_local_var(v);
        return ret->current_value;
    }
    expr_c visit(var_c v) override {
        auto ret = get_local_var(v);
        auto &cur_scope = scopes_.back();
        if (!ret->current_value->ssa_data_->is_global_) {
            auto parent_scope_id = ret->defined_scope_idx;
            auto cur_val = ret->current_value;
            if (cur_scope.for_depth_ > scopes_[parent_scope_id].for_depth_) {
                // if the variable depends on a value created outside the
                // current for loop
                // we now need to create a phi node. The source of the phi is
                // the value from parent for-loop scope
                if (cur_scope.for_depth_
                        != scopes_[parent_scope_id].for_depth_ + 1) {
                    // if there is other for-loop scopes between current scope
                    // and the parent loop scope, we need to insert a PHI for
                    // each for-loop scopes between them
                    for (size_t i = parent_scope_id + 1; i < scopes_.size() - 1;
                            i++) {
                        auto &the_scope = scopes_[i];
                        if (the_scope.kind_ != ssa_scope_t::kind::for_loop) {
                            continue;
                        }
                        if (the_scope.for_depth_ >= cur_scope.for_depth_) {
                            break;
                        }
                        auto phi = make_expr<ssa_phi_node>(
                                std::vector<expr> {cur_val}, false);
                        auto new_ssa_def
                                = add_ssa_def_to_parent_scope(phi, &the_scope);
                        cur_val = new_ssa_def->var_;
                        rename_temp_var_with_version(
                                cur_val.checked_as<var>(), v);
                        insert_local_var(v, cur_val, i)
                                ->for_loop_phi.emplace_back(cur_val);
                    }
                }
                ssa_scope_t *insert_scope = nullptr;
                size_t scope_idx = 0;
                for (int64_t itr = scopes_.size() - 1; itr >= 0; --itr) {
                    if (scopes_[itr].kind_ == ssa_scope_t::kind::for_loop) {
                        insert_scope = &scopes_[itr];
                        scope_idx = itr;
                        break;
                    }
                }
                assert(scope_idx != 0);
                expr phi;
                auto phi_expr = make_expr<ssa_phi_node>(
                        std::vector<expr> {cur_val}, false);
                if (insert_scope == &scopes_.back()) {
                    phi = add_ssa_def(phi_expr);
                } else {
                    phi = add_ssa_def_to_parent_scope(phi_expr, insert_scope)
                                  ->var_;
                }
                assert(cur_val.isa<var>() || cur_val.isa<tensor>()
                        || cur_val.isa<constant>());
                rename_temp_var_with_version(phi.checked_as<var>(), v);
                // update the local var mapping to the phi node
                insert_local_var(v, phi, scope_idx)
                        ->for_loop_phi.emplace_back(phi);
                // remember that we need to update this phi node after for-loop
                // exits
                return phi;
            }
        } else {
            return add_ssa_def(ret->current_value);
        }
        return ret->current_value;
    }

    stmt_c visit(define_c v) override {
        expr_c lhs;
        assert(v->linkage_ == linkage::local);

        auto info = insert_local_var(v->var_, expr());
        enum { LOCAL_VAR, GLOBAL_VAR, TENSOR } type;
        if (v->var_.isa<var>()) {
            if (is_old_var_global(v->var_.get())) {
                type = GLOBAL_VAR;
            } else {
                type = LOCAL_VAR;
            }
        } else {
            assert(v->var_.isa<tensor>());
            type = TENSOR;
        }
        if (type == LOCAL_VAR && !v->init_.defined()) {
            // pure local var-def without init value, simply remove it
            info->current_value
                    = make_expr<constant_node>(INT64_C(0), v->var_->dtype_);
            init_ssa_data(info->current_value.get());
            return stmt_c();
        }
        auto newvar = v->var_->remake();
        init_ssa_data(newvar.get());
        info->current_value = newvar;
        if (type == GLOBAL_VAR) { newvar->ssa_data_->is_global_ = true; }
        expr_c init;
        if (v->init_.defined()) {
            need_flatten_ = false;
            init = dispatch(v->init_);
        }
        return copy_attr(*v,
                builder::make_var_tensor_def_unattached(
                        newvar, v->linkage_, init));
    }

    uint64_t var_version_idx = 0;

    void rename_temp_var_with_version(const var &newv, const var_c &old_var) {
        if (newv->ssa_data_->is_local()) {
            newv->name_
                    = old_var->name_ + "_" + std::to_string(var_version_idx++);
        }
    }

    stmt_c visit(assign_c v) override {
        if (v->var_.isa<var>()) {
            auto rhs = dispatch(v->value_);
            auto var_info = get_local_var_for_update(v->var_);
            if (!var_info->current_value.defined()
                    || !var_info->current_value->ssa_data_->is_global_) {
                if (!v->value_.isa<var>()
                        || get_local_var(v->value_)->defined_scope_idx
                                == scopes_.size() - 1) {
                    // if we are sure that RHS will be transformed to a temp
                    // var in the current scope, we can avoid making a copy of
                    // the var
                    var_info->current_value = rhs.remove_const();
                    assert(var_info->current_value.isa<var>()
                            || var_info->current_value.isa<constant>());
                    auto cur_value = var_info->current_value.static_as<var>();
                    rename_temp_var_with_version(
                            cur_value, v->var_.static_as<var>());
                    return stmt_c();
                }
                // make a copy for RHS
                auto ret = make_def(rhs);
                auto newvar = ret->var_;
                rename_temp_var_with_version(
                        newvar.checked_as<var>(), v->var_.static_as<var>());
                // update the local var mapping
                var_info->current_value = newvar;
                return copy_attr(*v, ret);
            } else {
                // if is global var
                return copy_attr(*v,
                        builder::make_assign_unattached(
                                var_info->current_value, rhs));
            }
        } else {
            assert(v->var_.isa<indexing>());
            need_flatten_ = false;
            auto lhs = dispatch(v->var_);
            return copy_attr(*v,
                    builder::make_assign_unattached(lhs, dispatch(v->value_)));
        }
    }

    expr resolve_single_phi(const expr &val, expr *out_last_level_var) {
        if (val.isa<var>()) {
            if (out_last_level_var) { *out_last_level_var = val; }
            if (val->ssa_data_->is_global_
                    || !val->ssa_data_->get_owner().defined()) {
                return val;
            }
            return resolve_single_phi(
                    val->ssa_data_->get_value_of_var(), out_last_level_var);
        }
        if (val.isa<ssa_phi>()) {
            auto val_phi = val.static_as<ssa_phi>();
            if (val_phi->values_.size() == 1) {
                return resolve_single_phi(
                        val_phi->values_[0], out_last_level_var);
            }
        }
        return val;
    }

    bool is_same_with_parent_var(
            const expr &val, const expr &parent_val, expr *out_same) {
        expr last_level_var;
        auto v1 = resolve_single_phi(val, nullptr);
        auto v2 = resolve_single_phi(
                parent_val, out_same ? &last_level_var : nullptr);
        if (v1.ptr_same(v2)) {
            // if the variable is unchanged in loop
            if (out_same) {
                if (!v1.isa<var>() && !v1.isa<tensor>()
                        && !v1.isa<constant>()) {
                    *out_same = last_level_var;
                } else {
                    *out_same = v1;
                }
            }
            return true;
        }
        return false;
    }

    stmt_c visit(for_loop_c v) override {
        auto begin = dispatch(v->iter_begin_);
        auto end = dispatch(v->iter_end_);
        auto step = dispatch(v->step_);

        push_scope(ssa_scope_t::kind::for_loop);
        auto thevar = v->var_->remake();
        insert_local_var(v->var_, thevar);
        init_ssa_data(thevar.get());
        auto body = dispatch(v->body_);
        ssa_scope_t scope = pop_scope();
        if (!scope.inserted_.empty()) {
            auto &bodyseq = body.checked_as<stmts>()->seq_;
            bodyseq.insert(bodyseq.begin(), scope.inserted_.begin(),
                    scope.inserted_.end());
        }
        for (auto &kv : scope.vars_) {
            auto parent_var = get_local_var_nothrow(kv.first);
            if (parent_var) {
                // if the variable is a for-loop-phi
                for (auto &phi : kv.second.for_loop_phi) {
                    if (is_same_with_parent_var(
                                phi, kv.second.current_value, nullptr)) {
                        continue;
                    }
                    // if the variable is changed in loop, we need to update the
                    // phi node input
                    auto thephi = phi->ssa_data_->get_value_of_var()
                                          .checked_as<ssa_phi>();
                    thephi->values_.emplace_back(kv.second.current_value);
                    thephi->is_loop_phi_ = true;
                }
                if (is_same_with_parent_var(kv.second.current_value,
                            parent_var->current_value, nullptr)) {
                    continue;
                }
                auto new_var = make_expr<ssa_phi_node>(
                        std::vector<expr> {parent_var->current_value,
                                kv.second.current_value},
                        false);
                auto cur_v = add_def_after_current_stmt(new_var);
                get_local_var_for_update(kv.first)->current_value = cur_v;
                rename_temp_var_with_version(
                        cur_v.checked_as<var>(), kv.first.checked_as<var>());
            }
        }
        return copy_attr(*v,
                builder::make_for_loop_unattached(thevar, begin, end, step,
                        body, v->incremental_, v->kind_, v->num_threads_));
    }

    stmt_c visit(if_else_c v) override {
        auto cond = dispatch(v->condition_);
        push_scope(ssa_scope_t::kind::if_then);
        auto then_block = dispatch(v->then_case_);
        ssa_scope_t then_scope = pop_scope();

        stmt_c else_block;
        if (v->else_case_.defined()) {
            push_scope(ssa_scope_t::kind::if_else);
            else_block = dispatch(v->else_case_);
            ssa_scope_t else_scope = pop_scope();
            // merge ths diverged variables with phi
            std::map<expr_c, std::vector<expr>, var_cmper_t> updated_vars;
            for (auto &kv : then_scope.vars_) {
                updated_vars[kv.first].emplace_back(kv.second.current_value);
            }
            for (auto &kv : else_scope.vars_) {
                updated_vars[kv.first].emplace_back(kv.second.current_value);
            }
            for (auto &kv : updated_vars) {
                if (kv.first.isa<tensor>()) {
                    // tensors/pointers are immutable, don't need phi
                    continue;
                }
                auto parent_var = get_local_var_nothrow(kv.first);
                if (!parent_var) {
                    // if it is a var defined in child scope
                    continue;
                }

                if (kv.second.size() == 1) {
                    auto current_value = parent_var->current_value;
                    if (scopes_[parent_var->defined_scope_idx].for_depth_
                            != scopes_.back().for_depth_) {
                        // if parent var value is computed outside of the
                        // current for-loop, we need to build PHI for parent
                        // value
                        current_value = visit(kv.first.static_as<var_c>())
                                                .remove_const();
                    }
                    kv.second.emplace_back(current_value);
                }
                // if it is a var defined in parent scope
                // let parent for-loop to remember to reset the phi inputs
                auto &ph = get_local_var_for_update(kv.first)->for_loop_phi;
                auto itr = then_scope.vars_.find(kv.first);
                if (itr != then_scope.vars_.end()) {
                    ph.insert(ph.end(), itr->second.for_loop_phi.begin(),
                            itr->second.for_loop_phi.end());
                }
                itr = else_scope.vars_.find(kv.first);
                if (itr != else_scope.vars_.end()) {
                    ph.insert(ph.end(), itr->second.for_loop_phi.begin(),
                            itr->second.for_loop_phi.end());
                }

                if (kv.second.size() == 2) {
                    // if both phi branches has the same value
                    expr same_var;
                    if (is_same_with_parent_var(
                                kv.second[0], kv.second[1], &same_var)) {
                        get_local_var_for_update(kv.first)->current_value
                                = same_var;
                        assert(same_var.isa<var>() || same_var.isa<tensor>()
                                || same_var.isa<constant>());
                        continue;
                    }
                }
                if (kv.second.size() == 1) {
                    get_local_var_for_update(kv.first)->current_value
                            = kv.second[0];
                    continue;
                }
                auto new_phi = make_expr<ssa_phi_node>(kv.second, false);
                auto new_var = add_def_after_current_stmt(new_phi);
                get_local_var_for_update(kv.first)->current_value = new_var;
                rename_temp_var_with_version(
                        new_var.checked_as<var>(), kv.first.checked_as<var>());
            }
        } else {
            for (auto &kv : then_scope.vars_) {
                auto parent_var = get_local_var_nothrow(kv.first);
                if (parent_var) {
                    if (kv.first.isa<tensor>()) {
                        // tensors/pointers are immutable, don't need phi
                        continue;
                    }
                    if (is_same_with_parent_var(kv.second.current_value,
                                parent_var->current_value, nullptr)) {
                        continue;
                    }
                    expr current_value = parent_var->current_value;
                    if (scopes_[parent_var->defined_scope_idx].for_depth_
                            != scopes_.back().for_depth_) {
                        // if parent var value is computed outside of the
                        // current for-loop, we need to build PHI for parent
                        // value
                        current_value = visit(kv.first.static_as<var_c>())
                                                .remove_const();
                    }
                    auto status = get_local_var_for_update(kv.first);
                    auto &ph = status->for_loop_phi;
                    ph.insert(ph.end(), kv.second.for_loop_phi.begin(),
                            kv.second.for_loop_phi.end());

                    auto new_phi = make_expr<ssa_phi_node>(
                            std::vector<expr> {
                                    current_value, kv.second.current_value},
                            false);
                    auto new_var = add_def_after_current_stmt(new_phi);
                    status->current_value = new_var;
                    rename_temp_var_with_version(new_var.checked_as<var>(),
                            kv.first.checked_as<var>());
                }
            }
        }
        return copy_attr(*v,
                builder::make_if_else_unattached(cond, then_block, else_block));
    }
};

func_c ssa_transform_t::operator()(func_c f) {
    ssa_transform_impl_t impl;
    return impl.top_level_dispatch(std::move(f));
}

stmt_c ssa_transform_t::operator()(stmt_c f) {
    ssa_transform_impl_t impl;
    return impl.top_level_dispatch(std::move(f));
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
