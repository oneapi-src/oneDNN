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
#include "value_numbering.hpp"
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/passlet/ssa_simplify.hpp>
#include <compiler/ir/passlet/ssa_value_hash.hpp>
#include <compiler/ir/passlet/structural_analysis.hpp>
#include <compiler/ir/passlet/volatility_analysis.hpp>
#include <compiler/ir/ssa_data.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(value_numbering, SC_PASS_DEPENDS_ON(ssa_transform),
        SC_PASS_REQUIRE_STATE(SSA_STAGE), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

using namespace passlet;
struct vn_result_t {
    structural_result_t parent_info_;
    size_t hash_ = 0;
    volatility_result_t vresult_;
    // the parent stmts for define node. It will only be set if the stmts has
    // been visited
    stmts_node_t *finalized_parent_ = nullptr;
    // the stmt to replace current stmt after visit()
    const stmt_base_t *new_object_ = nullptr;
    bool ref_by_phi_ = false;
};

static vn_result_t &get_vn_result(const stmt_base_t *v) {
    return v->temp_data().get<vn_result_t>();
}

static vn_result_t &get_vn_result(const stmt_c &v) {
    return get_vn_result(v.get());
}

class value_numbering_analysis_t : public ssa_viewer_t {
    volatility_analysis_t v_ana_;
    ssa_value_hash_t hasher_;
    structural_analysis_t s_ana_;
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;

public:
    value_numbering_analysis_t()
        : v_ana_ {false, sc_make_temp_data_addresser(&vn_result_t::vresult_)}
        , hasher_ {sc_make_temp_data_addresser(&vn_result_t::hash_)}
        , s_ana_ {sc_make_temp_data_addresser(&vn_result_t::parent_info_)} {}

    expr_c dispatch(expr_c f) override { return f; }

    void view(define_c v) override {
        v_ana_.view(v, pass_phase::PRE_VISIT);
        hasher_.view(v, pass_phase::PRE_VISIT);

        v_ana_.view(v, pass_phase::POST_VISIT);
        hasher_.view(v, pass_phase::POST_VISIT);

        // we need to mark PHI node's dependencies
        if (v->init_.defined()) {
            auto &f = v->init_;
            if (f.isa<ssa_phi_c>()) {
                for (auto &v : f.static_as<ssa_phi_c>()->values_) {
                    if (v.isa<var>()) {
                        if (!v->ssa_data_->has_owner()) { continue; }
                        auto owner = v->ssa_data_->get_owner();

                        if (!owner->temp_data().isa<vn_result_t>()) {
                            owner->temp_data() = vn_result_t {};
                        }
                        get_vn_result(owner.get()).ref_by_phi_ = true;
                    }
                }
            }
        }
    }

    stmt_c dispatch(stmt_c f) override {
        s_ana_.view(f, pass_phase::PRE_VISIT);
        ssa_viewer_t::dispatch(f);
        s_ana_.view(f, pass_phase::POST_VISIT);
        return f;
    }

    func_c dispatch(func_c f) override {
        v_ana_.view(f, pass_phase::PRE_VISIT);
        auto ret = ssa_viewer_t::dispatch(f);
        v_ana_.view(f, pass_phase::POST_VISIT);
        return ret;
    }
};

struct ssa_hasher_t {
    size_t operator()(const define_c &v) const {
        return v->temp_data().get<vn_result_t>().hash_;
    }
};

struct ssa_cmper_t {
    bool operator()(const define_c &v, const define_c &v2) const {
        // an IR cmper which compares the var/tensor pointers, instead of create
        // a mapping of them. Also checkes IR considering commutative law
        if (v->var_->dtype_ != v2->var_->dtype_) { return false; }
        assert(v->init_.defined() && v2->init_.defined());
        ir_comparer cmper {false, false, true, false, true};
        return cmper.compare(v->init_, v2->init_);
    }
};

class value_numbering_mutator_t : public ssa_visitor_t {
    using ssa_value_set
            = std::unordered_set<define_c, ssa_hasher_t, ssa_cmper_t>;
    using replace_map = std::unordered_map<expr_c, expr_c>;
    struct scope_t {
        std::unordered_set<stmt_c> alive_vars_;
        replace_map repleace_map_;
        const stmts_node_t *old_stmts_;
        // inserting stmt in this seq will affect the resulting stmts for
        // visiting old_stmts_
        std::vector<stmt_c> *cur_seq_to_insert_;
        scope_t(const stmts_node_t *s, std::vector<stmt_c> *seq)
            : old_stmts_(s), cur_seq_to_insert_(seq) {}
    };
    // the set to help to find the same values of ssa. It is a stack for nested
    // scopes
    std::vector<scope_t> scopes_;
    ssa_value_set var_set_;
    ssa_simplify_t simplifier_;
    bool should_update_scopes_ = false;
    using ssa_visitor_t::dispatch;
    using ssa_visitor_t::visit;

    structural_result_t::typed_addresser_t addresser_
            = sc_make_temp_data_addresser(&vn_result_t::parent_info_);

    stmt_c dispatch(stmt_c f) override {
        if (should_update_scopes_) {
            should_update_scopes_ = false;
            assert(scopes_.back().cur_seq_to_insert_ == nullptr);
            scopes_.back().cur_seq_to_insert_ = get_current_scope();
        }
        auto ret = ssa_visitor_t::dispatch(f);
        get_vn_result(f).new_object_ = ret.get();
        return ret;
    }

    expr_c visit(var_c v) override {
        if (simplifier_.is_in_phi_) { return v; }
        auto ret = simplifier_.visit(v);
        if (ret.isa<var>()) {
            for (auto &m : scopes_) {
                auto itr = m.repleace_map_.find(ret);
                if (itr != m.repleace_map_.end()) { return itr->second; }
            }
        }
        return ret;
    }
    expr_c visit(ssa_phi_c v) override {
        simplifier_.enter_phi();
        auto ret = ssa_visitor_t::visit(v);
        if (ret.isa<ssa_phi_c>()) {
            ret = simplifier_.visit(ret.static_as<ssa_phi_c>());
        }
        simplifier_.leave_phi();
        return ret;
    }

    stmt_c visit(stmts_c v) override {
        scopes_.emplace_back(scope_t {v.get(), nullptr});
        should_update_scopes_ = true;
        auto ret = ssa_visitor_t::visit(v);
        should_update_scopes_ = false;
        // make sure we make a new stmts node, so that remove_const is safe
        if (ret.ptr_same(v)) { ret = ret->remake(); }
        // update all var's finalized_parent_
        for (auto &alive : scopes_.back().alive_vars_) {
            get_vn_result(alive).finalized_parent_
                    = ret.static_as<stmts_c>().remove_const().get();
        }
        scopes_.pop_back();
        return ret;
    }

    stmt_c visit(define_c v) override {
        stmt_c new_def = v;
        if (v->init_.defined()) {
            auto new_init = dispatch(v->init_);
            if (!new_init.ptr_same(v->init_)) {
                new_def = builder::make_var_tensor_def_unattached(
                        v->var_, v->linkage_, new_init);
            }
        }
        if (!v->var_.isa<var>()) { return new_def; }
        auto &vn_result = get_vn_result(v);
        if (vn_result.vresult_.is_volatile_ == volatility_result_t::YES) {
            return new_def;
        }
        if (!new_def.ptr_same(v)) { new_def->temp_data() = v->temp_data(); }
        bool replaced = false;
        auto itr = var_set_.find(new_def.checked_as<define_c>());
        if (itr != var_set_.end()) {
            for (auto &s : scopes_) {
                auto &def = s.alive_vars_;
                if (def.count(*itr) != 0UL) {
                    replaced = true;
                    break;
                }
            }
            auto &old_def = *itr;
            auto &old_vn_data = get_vn_result(old_def);
            if (!replaced && !old_vn_data.ref_by_phi_) {
                // the var is met before, but old var is not alive
                // try to move the old var to a parent scope, so that the old
                // var can be shared by old and new uses

                // if the old var is referenced by phi, we cannot move it
                auto &old_info = old_vn_data.parent_info_;
                // find the nearest shared parent of old var def and new var def
                // it will not come across the for-loop boundary
                auto shared_parent = old_info.find_shared_parent(
                        get_vn_result(new_def).parent_info_, addresser_, false,
                        true);
                if (shared_parent) {
                    if (shared_parent->node_type_ == sc_stmt_type::if_else) {
                        shared_parent = get_vn_result(shared_parent)
                                                .parent_info_.get_raw_parent();
                    }
                    // remove old var def from its current parent
                    assert(shared_parent->node_type_ == sc_stmt_type::stmts);
                    assert(old_vn_data.finalized_parent_);
                    auto &seq_remove = old_vn_data.finalized_parent_->seq_;
                    auto olditr = std::find_if(seq_remove.begin(),
                            seq_remove.end(), [&old_def](const stmt_c &v) {
                                return v.get() == old_def.get();
                            });

                    assert(olditr != seq_remove.end());
                    seq_remove.erase(olditr);
                    // update parent_info_ of old_info
                    // insert to shared_parent, update current parent

                    // firstly, find the shared_parent's scope
                    auto scope_itr = std::find_if(scopes_.begin(),
                            scopes_.end(), [shared_parent](scope_t &scope) {
                                return scope.old_stmts_ == shared_parent;
                            });
                    assert(scope_itr != scopes_.end());
                    auto &scope = *scope_itr;
                    {
                        // else, need to insert the old var def before it is
                        // used. Find the second level parent
                        // for example, for IR like
                        // if (...) {
                        //   g=a+1
                        //   A[0]=g
                        // }
                        // b=a+1
                        //
                        // We found the duplication at the point of "b=a+1", and
                        // we need to move "g=a+1" before the "if". We need to
                        // first find the "if" node in the shared_parent
                        const structural_result_t *second_level = nullptr;
                        get_vn_result(shared_parent)
                                .parent_info_.is_parent_of(old_info, addresser_,
                                        false, true, &second_level);
                        COMPILE_ASSERT(second_level,
                                "expecting second_level being not null");
                        auto &the_seq = *scope.cur_seq_to_insert_;
                        // get the point in shared parent to insert the def
                        // before
                        auto insert_before = get_vn_result(
                                second_level->get_raw_cur_node())
                                                     .new_object_;
                        if (!insert_before) {
                            // if old def has not been attached to ancestor stmt
                            // seq
                            the_seq.emplace_back(old_def);
                        } else {
                            auto insertion_point = std::find_if(the_seq.begin(),
                                    the_seq.end(),
                                    [insert_before](const stmt_c &v) {
                                        return v.get() == insert_before;
                                    });
                            assert(insertion_point != the_seq.end());
                            the_seq.insert(insertion_point, old_def);
                        }
                    }
                    // insert to alive vars of shared_parent
                    scope.alive_vars_.insert(old_def);

                    old_info.parent_ = shared_parent->shared_from_this();
                    old_vn_data.finalized_parent_ = nullptr;

                    replaced = true;
                }
            }

            if (replaced) {
                scopes_.back().repleace_map_.insert(
                        std::make_pair(v->var_, (*itr)->var_));
            } else {
                var_set_.erase(itr);
            }
        }
        if (!replaced) {
            scopes_.back().alive_vars_.insert(new_def);
            var_set_.insert(new_def.static_as<define_c>());
        }
        return new_def;
    }
};

func_c value_numbering_t::operator()(func_c f) {
    value_numbering_analysis_t().dispatch(f);
    return value_numbering_mutator_t().top_level_dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
