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
#include "dependency_analyzer.hpp"
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "../ir_comparer.hpp"
#include "../viewer.hpp"
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/utils.hpp>

// fixme: the tensor/tensorptr passed to functions not marked in dep graph
// fixme: if-else merge issue (see merge() below)
// todo: add IR hash so that we don't need to compare the IR one by one

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using namespace dependency_analysis;
namespace dependency_analysis {
dependency_t &get_dep_info(const node_base *s) {
    return s->temp_data().get<dependency_t>();
}
} // namespace dependency_analysis

static stmt_base_t *get_indexing_owner(const expr_c &access) {
    auto ret = get_dep_info(access.get()).indexing_owner_.lock();
    assert(ret);
    return ret.get();
}

class dep_analyzer_impl_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    stmt_c cur_stmt;
    dependency_t *cur_dep = nullptr;
    using stmt_set = stmt_weak_set;
    // the states for var and tensor, may be nested for if-else
    struct nested_state_t {
        // expr(var) => the statements that may be the last update to it.
        std::unordered_map<expr_c, stmt_set> var_last_update;

        // merges diverged branches' state of if-else into this state
        void merge(nested_state_t &then_block, nested_state_t &else_block) {
            // todo: if a var is updated in both then and else, can remove the
            // last update in this state (the mainstream state)
            for (auto &kv : then_block.var_last_update) {
                var_last_update[kv.first].merge(kv.second);
            }
            for (auto &kv : else_block.var_last_update) {
                var_last_update[kv.first].merge(kv.second);
            }
        }
    };
    std::vector<nested_state_t> state_stack;
    // tensor => the loop depth where it is defined
    std::unordered_map<expr_c, int> tsr_defined_loop_depth;
    // tensor => the set of indexing on the tensor in the loops
    std::unordered_map<expr_c, std::unordered_set<expr_c>> tsr_accesses_in_loop;
    // tensor => the set of indexing on the tensor outside of any loops
    std::unordered_map<expr_c, std::unordered_set<expr_c>> tsr_accesses;
    for_loop_c cur_loop_;
    int loop_depth = 0;
    int if_depth = 0;

    const stmt_set *get_last_update_for_var(const var_c &v) {
        // checks nested_states. Check stack top first, if not found, look down
        for (auto itr = state_stack.rbegin(); itr != state_stack.rend();
                ++itr) {
            auto varitr = itr->var_last_update.find(v);
            if (varitr != itr->var_last_update.end()) {
                return &varitr->second;
            }
        }
        return nullptr;
    }

    void update_var(const expr_c &v, const stmt_c &st) {
        state_stack.back().var_last_update[v] = {st.impl};
    }

    void add_to_dependency(stmt_base_t *depending, stmt_base_t *dependee) {
        get_dep_info(depending).depends_on_.insert(
                dependee->shared_from_this());
        get_dep_info(dependee).depended_by_.insert(
                depending->shared_from_this());
    }

    // checks if two indexing are the same: same ptr, same indices and their
    // dependency are the same (the vars in the indices are unchanged)
    bool indexing_same(const indexing_c &v1, const indexing_c &v2) {
        if (!v1->ptr_.ptr_same(v2->ptr_)) { return false; }
        assert(v1->idx_.size() == v2->idx_.size());

        ir_comparer cmper {false, false, true};
        for (unsigned i = 0; i < v1->idx_.size(); i++) {
            if (!cmper.compare(v1->idx_[i], v2->idx_[i])) { return false; }
        }
        return get_dep_info(v1.get()).depends_on_
                == get_dep_info(v2.get()).depends_on_;
    }

    // returns true when two indices can be proven not the same
    // currently only check the constant indices
    bool indexing_definitely_not_same(
            const indexing_c &v1, const indexing_c &v2) {
        assert(v1->idx_.size() == v2->idx_.size());
        for (unsigned i = 0; i < v1->idx_.size(); i++) {
            if (!v1->idx_[i].isa<constant>()) { return false; }
            if (!v2->idx_[i].isa<constant>()) { return false; }
            if (v1->idx_[i]->dtype_ != v2->idx_[i]->dtype_) { return false; }
            if (v1->idx_[i].static_as<constant>()->value_
                    == v2->idx_[i].static_as<constant>()->value_) {
                return false;
            }
        }
        return true;
    }

    void add_tensor_depends_on_top_level_access(
            const tensor_c &tsr, const indexing_c &v) {
        auto itr = tsr_accesses.find(tsr);
        if (itr != tsr_accesses.end()) {
            for (auto &access : itr->second) {
                auto other_owner = get_indexing_owner(access);
                if (v.defined()) {
                    if (indexing_definitely_not_same(
                                access.checked_as<indexing_c>(), v)) {
                        // if the access and v are definitely not the same, no
                        // need to add dependency
                    } else {
                        add_to_dependency(
                                cur_stmt.remove_const().get(), other_owner);
                    }
                }
            }
        }
    }

    void add_tensor_depends_on_loop_access(
            const tensor_c &tsr, const indexing_c &v) {
        auto itr = tsr_accesses_in_loop.find(tsr);
        if (itr != tsr_accesses_in_loop.end()) {
            for (auto &access : itr->second) {
                auto other_owner = get_indexing_owner(access);
                if (v.defined()) {
                    if (indexing_definitely_not_same(
                                access.checked_as<indexing_c>(), v)) {
                        // if the access's index is proven not be the same of v,
                        // no need to add dependency
                    } else if (indexing_same(
                                       access.checked_as<indexing_c>(), v)) {
                        // if the access's index is the same expr of v, no need
                        // to add reverse dependency, since v happens after
                        // access
                        add_to_dependency(
                                cur_stmt.remove_const().get(), other_owner);
                        auto def_loop_depth = tsr_defined_loop_depth[tsr];
                        /* if the tensor is defined in an outer loop, like
                        tensor A[100]
                        for(i,...) {
                            for(j,...) {
                                t=A[i]
                                A[i]=t+1
                            }
                        }

                        We cannot remove the write to A[i]

                        Another case is that the indexing does not depend on the
                        loop:
                        tensor A[100]
                        for(j,...) {
                            t=A[0]
                            A[0]=t+1
                        }
                        */
                        if (loop_depth > 0
                                && (def_loop_depth < loop_depth - 1
                                        || !get_dep_info(v.get())
                                                    .depends_on_.has(
                                                            cur_loop_.impl))) {
                            add_to_dependency(
                                    other_owner, cur_stmt.remove_const().get());
                        }
                    } else {
                        // if the access's index may be the same of v,
                        // conservatively add reverse dependency
                        add_to_dependency(
                                cur_stmt.remove_const().get(), other_owner);
                        add_to_dependency(
                                other_owner, cur_stmt.remove_const().get());
                    }
                }
            }
        }
    }

    void view(define_c v) override {
        if (!v->var_.isa<tensor>()) {
            tsr_defined_loop_depth[v->var_] = loop_depth;
            dispatch(v->var_);
        }
        if (v->init_.defined()) { dispatch(v->init_); }
    }

    // indexing's base tensor will not go here
    void view(tensor_c v) override {
        ir_viewer_t::view(v);
        const auto &tsr = v;
        add_tensor_depends_on_loop_access(tsr, indexing_c());
        add_tensor_depends_on_top_level_access(tsr, indexing_c());
        v.remove_const()->attr()[attr_directly_accessed] = true;
    }

    void view(tensorptr_c v) override {
        ir_viewer_t::view(v);
        v->base_->ptr_.checked_as<tensor_c>()
                .remove_const()
                ->attr()[attr_directly_accessed]
                = true;
    }

    void view(indexing_c v) override {
        auto old_cur_dep = cur_dep;
        any_t &dep_val = v->temp_data();
        dep_val = dependency_t();
        cur_dep = &dep_val.get<dependency_t>();
        cur_dep->indexing_owner_ = cur_stmt.impl;

        // ir_viewer_t::view(v);
        // dispatch sub nodes without dispatching on the base tensor
        for (auto &idx : v->idx_) {
            dispatch(idx);
        }
        if (v->mask_.defined()) { dispatch(v->mask_); }

        old_cur_dep->depended_by_.merge(cur_dep->depended_by_);
        old_cur_dep->depends_on_.merge(cur_dep->depends_on_);
        cur_dep = old_cur_dep;

        auto tsr = v->ptr_.checked_as<tensor>();
        add_tensor_depends_on_loop_access(tsr, v);
        add_tensor_depends_on_top_level_access(tsr, v);
        if (loop_depth > 0) { tsr_accesses_in_loop[tsr].insert(v); }
        tsr_accesses[tsr].insert(v);
    }

    void view(var_c v) override {
        // visiting on a var or tensor, adds its latest updating stmt to the
        // dependency
        if (cur_dep) {
            auto last_update = get_last_update_for_var(v);
            if (last_update) {
                cur_dep->depends_on_.merge(*last_update);
                for (auto upd_stmt : *last_update) {
                    auto ptr = upd_stmt.lock();
                    assert(ptr);
                    assert(cur_stmt.defined());
                    get_dep_info(ptr.get()).depended_by_.insert(cur_stmt.impl);
                }
            }
        }
    }

    void view(assign_c v) override {
        ir_viewer_t::view(v);
        if (v->var_.isa<var_c>()) { update_var(v->var_, cur_stmt); }
    }
    void view(if_else_c v) override {
        dispatch(v->condition_);
        state_stack.emplace_back(nested_state_t());
        dispatch(v->then_case_);
        nested_state_t top_state1 = std::move(state_stack.back());
        state_stack.pop_back();

        nested_state_t top_state2;
        if (v->else_case_.defined()) {
            state_stack.emplace_back(nested_state_t());
            dispatch(v->else_case_);
            top_state2 = std::move(state_stack.back());
            state_stack.pop_back();
        }
        state_stack.back().merge(top_state1, top_state2);
    }
    void view(for_loop_c v) override {
        loop_depth++;
        for_loop_c old_loop = std::move(cur_loop_);
        cur_loop_ = v;
        update_var(v->var_, v);
        ir_viewer_t::view(v);
        state_stack.back().var_last_update.erase(v->var_);
        cur_loop_ = std::move(old_loop);
        loop_depth--;
        if (loop_depth == 0) { tsr_accesses_in_loop.clear(); }
    }

    stmt_c dispatch(stmt_c v) override {
        auto old_cur_stmt = cur_stmt;
        auto old_cur_dep = cur_dep;
        cur_stmt = v;
        any_t &dep_val = v->temp_data();
        dep_val = dependency_t();
        cur_dep = &dep_val.get<dependency_t>();

        ir_viewer_t::dispatch(v);

        cur_stmt = std::move(old_cur_stmt);
        cur_dep = old_cur_dep;
        return v;
    }

    func_c dispatch(func_c v) override {
        state_stack = {nested_state_t()};
        return ir_viewer_t::dispatch(v);
    }
};

func_c dependency_analyzer_t::operator()(func_c f) {
    dep_analyzer_impl_t v;
    return v.dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
