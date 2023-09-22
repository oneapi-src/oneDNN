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
#include "index2var.hpp"
#include <memory>
#include <utility>
#include <vector>
#include "pointer_alias_info.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/array_ref.hpp>
#include <util/utils.hpp>
#include <util/weakptr_utils.hpp>

SC_MODULE(pass.index2var)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(index2var,
        SC_PASS_DEPENDS_ON(constant_folder, ir_simplifier, validator,
                index_flattener, parallel_workload_dispatcher, tensor_inplace),
        SC_PASS_REQUIRE_STATE(FUNC_INLINED), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE(IR_SIMPLIFIED));
// it requires tensor_inplace for pointer alias info

// the visitor to find the mutable dependencies in the indices of indexing
// nodes. e.g., for A[i+j], it will find i and j as dependencies. Note that if
// there is indexing_node/call_node in the indices, this pass will set
// `is_valid_` = false, meaning that the indices are untraceable - we are unable
// to tell if the indices are changed after some statements.
class var_dependency_finder_t : public ir_viewer_t {
    utils::weakptr_hashset_t<expr_base> *vars_;
    // the output flag to mark if the input expr is good for index2var_t
    bool is_valid_ = true;
    var_dependency_finder_t(utils::weakptr_hashset_t<expr_base> *vars)
        : vars_(vars) {}

    void view(call_c v) override {
        bool is_pure_no_args
                = some_opt(dynamic_cast<func_base *>(v->func_.get()))
                          .filter([](func_base *f) {
                              return f->params_.empty();
                          })
                          .map([](func_base *f) { return f->attr_.get(); })
                          .filter([](any_map_t *m) {
                              return m->get_or_else(
                                      function_attrs::pure, false);
                          })
                          .has_value();
        if (is_pure_no_args) {
            ir_viewer_t::view(v);
            return;
        }
        is_valid_ = false;
        SC_MODULE_INFO << "Found call node in index: " << v;
    }
    void view(indexing_c v) override {
        is_valid_ = false;
        SC_MODULE_INFO << "Found indexing node in index: " << v;
    }
    void view(var_c v) override { vars_->insert(v.impl); }

public:
    static bool find(
            utils::weakptr_hashset_t<expr_base> &vars, array_ref<expr> idx) {
        var_dependency_finder_t f(&vars);
        for (auto &v : idx) {
            f.dispatch(v);
        }
        return f.is_valid_;
    }
};

struct written_tensor_analysis_result_t {
    std::unordered_set<expr_c> written_;
};

struct tensor_usage_analysis_result_t {
    bool used_in_broadcast_;
    // the cached result of alias_info::get_alias_info
    alias_info::tensor_alias_identity_t *alias_id_ = nullptr;

    tensor_usage_analysis_result_t(bool used_in_broadcast)
        : used_in_broadcast_(used_in_broadcast) {}
    void for_each_alias_tensor(
            const std::unordered_map<alias_info::tensor_alias_identity_t *,
                    expr_c> &alias_map,
            const std::function<void(const expr_c &)> &func) const {
        for (auto &cliq : alias_id_->alias_cliques_) {
            for (auto aid : cliq->set_) {
                auto other_alias_id = aid.lock();
                // if the tensor has been removed, skip
                if (!other_alias_id) { continue; }
                auto itr = alias_map.find(other_alias_id.get());
                if (itr != alias_map.end()) { func(itr->second); }
            }
        }
    }
};

struct var_state_analysis_result_t {
    utils::weakptr_hashset_t<expr_base> depending_on_;
    int assignments_ = 0;
    bool is_complex_ = false;
};

// the visitor to find all tensor written in all stmts. Also finds if the tensor
// is read in "broadcast" intrinsic.
class index2var_analysis_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    std::unordered_set<expr_c> written_;
    std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
            alias_map_;

    stmt_c dispatch(stmt_c v) override {
        written_ = std::unordered_set<expr_c> {};
        ir_viewer_t::dispatch(v);
        v->temp_data() = written_tensor_analysis_result_t {std::move(written_)};
        written_ = std::unordered_set<expr_c> {};
        return v;
    }

    expr get_tensor_from_indexing(const expr &v) {
        if (v.isa<indexing>()) {
            auto idx = v.static_as<indexing>();
            COMPILE_ASSERT(idx->ptr_.isa<tensor>(),
                    "The indexing should be based on a tensor. " << v);
            return idx->ptr_;
        }
        return expr();
    }

    var_state_analysis_result_t *get_or_create_var_state(const expr &v) {
        if (auto ret
                = v->temp_data().get_or_null<var_state_analysis_result_t>()) {
            return ret;
        }
        v->temp_data() = var_state_analysis_result_t {};
        return v->temp_data().get_or_null<var_state_analysis_result_t>();
    }

    void view(define_c v) override {
        if (v->var_.isa<var>() && v->init_.defined()) {
            auto var_state = get_or_create_var_state(v->var_);
            var_state->assignments_++;
            var_state->is_complex_ = !var_dependency_finder_t::find(
                    var_state->depending_on_, {v->init_});
        }
    }
    void view(assign_c v) override {
        ir_viewer_t::view(v);
        auto tsr = get_tensor_from_indexing(v->var_);
        if (tsr.defined()) {
            written_.insert(tsr);
        } else {
            auto var_state = get_or_create_var_state(v->var_);
            var_state->assignments_++;
            if (var_state->assignments_ == 1) {
                var_state->is_complex_ = !var_dependency_finder_t::find(
                        var_state->depending_on_, {v->value_});
            }
        }
    }

    void view(tensor_c v) override {
        ir_viewer_t::view(v);
        auto alias = alias_info::get_alias_info(*v);
        if (!alias || alias->has_no_alias()) { return; }
        tensor_usage_analysis_result_t *result
                = v->temp_data().get_or_null<tensor_usage_analysis_result_t>();
        if (!result) {
            alias_map_[alias] = v;
            v->temp_data() = tensor_usage_analysis_result_t {false};
            result = &v->temp_data_->get<tensor_usage_analysis_result_t>();
        }
        result->alias_id_ = alias;
    }

    void view(intrin_call_c v) override {
        ir_viewer_t::view(v);
        if (v->type_ == intrin_type::broadcast) {
            auto &arg = v->args_.at(0);
            auto tsr = get_tensor_from_indexing(arg);
            if (tsr.defined()) {
                if (auto result
                        = tsr->temp_data()
                                  .get_or_null<
                                          tensor_usage_analysis_result_t>()) {
                    result->used_in_broadcast_ = true;
                } else {
                    tsr->temp_data() = tensor_usage_analysis_result_t {true};
                }
            }
        }
    }

    void view(for_loop_c v) override {
        ir_viewer_t::view(v);
        written_ = v->body_->get_temp_data()
                           .get<written_tensor_analysis_result_t>()
                           .written_;
    }

    void view(if_else_c v) override {
        ir_viewer_t::view(v);

        written_ = v->then_case_->get_temp_data()
                           .get<written_tensor_analysis_result_t>()
                           .written_;
        if (v->else_case_.defined()) {
            auto &else_result = v->else_case_->get_temp_data()
                                        .get<written_tensor_analysis_result_t>()
                                        .written_;
            written_.insert(else_result.begin(), else_result.end());
        }
    }
    void view(stmts_c v) override {
        for (auto &s : v->seq_) {
            dispatch(s);
        }

        for (auto &s : v->seq_) {
            auto &result = s->get_temp_data()
                                   .get<written_tensor_analysis_result_t>()
                                   .written_;
            written_.insert(result.begin(), result.end());
        }
    }
};

class indexing2var_impl_t : public ir_visitor_t {
    struct scope_info_t;
    // the "cache" for an element of a tensor
    // currently, one tensor has only one "cache"
    struct tensor_cache_t {
        tensor_c tsr_;
        std::vector<expr_c> idx_;
        // the last write of the cached value. Null if the value is not yet
        // written in the original IR
        stmts last_write_;
        // the var for the cache
        var_c var_;
        // the vector size
        unsigned lanes_;
        expr_c mask_;
        // the tensor cache var definition may be lifted to a parent for-loop
        scope_info_t *may_lift_to_ = nullptr;
        utils::weakptr_hashset_t<expr_base> dependencies_;
        tensor_cache_t(tensor_c tsr, std::vector<expr_c> &&idx, var_c var,
                int lanes, expr_c mask = expr())
            : tsr_(std::move(tsr))
            , idx_(std::move(idx))
            , var_(std::move(var))
            , lanes_(lanes)
            , mask_(std::move(mask)) {}
        // returns true if `v` is exactly the same of the cached indexing
        bool is_match(const indexing_c &v) const {
            if (!v->ptr_.ptr_same(tsr_.static_as<expr>())) return false;
            assert(idx_.size() == v->idx_.size());
            ir_comparer cmp(false, false, true);
            if (v->dtype_.lanes_ == lanes_) {
                for (unsigned i = 0; i < v->idx_.size(); i++) {
                    if (!cmp.compare(v->idx_[i], idx_[i])) { return false; }
                }
                if (v->mask_.defined()) {
                    if (!mask_.defined()) { return false; }
                    if (!cmp.compare(v->mask_, mask_)) { return false; }
                } else {
                    if (mask_.defined()) { return false; }
                }
                return true;
            }
            return false;
        }
        bool is_valid() const { return tsr_.defined(); }
    };

    using tensor_cache_ptr = std::shared_ptr<tensor_cache_t>;

    struct scope_info_t {
        const for_loop_node_t *loop_;
        std::vector<stmt_c> *insert_point_;
        const std::unordered_set<expr_c> &written_tensors_;
        std::unordered_set<tensor_cache_ptr> outstanding_cache_;
        // the number of times we flush cached of an tensor
        std::unordered_map<expr_c, int> num_flushes_;
        // the statements to be inserted after current call of visit() of a
        // sub-node in stmts node
        std::vector<stmt> insert_after_;
        bool is_cache_defined_here(const tensor_cache_ptr &v) {
            return outstanding_cache_.find(v) != outstanding_cache_.end();
        }

        bool tensor_not_written_here(const expr_c &v) {
            return written_tensors_.find(v) == written_tensors_.end();
        }
    };

    // the tensor -> tensor_cache, it stores all "cached" indexing
    std::unordered_map<expr_c, tensor_cache_ptr> cached_index_;
    // the var -> tensor_cache map. The var is the variable used in indexing,
    // NOT the caching var. e.g. if there is A[i,j] cached, there will be
    // {i->cache of A} and {j-> cache of A} in dependency_map_. If
    std::unordered_multimap<expr_c, tensor_cache_ptr> dependency_map_;
    // the number of vars for caching
    int var_cnt_ = 0;
    // the insertion point for the current stmts. We may insert var definition
    // and var initialization here
    std::vector<stmt_c> *insertion_point_ = nullptr;
    // the cached tensors which are created in the current scope (stmts). The
    // first dimension is a stack of scopes. At the end of a scope, we need to
    // evict all cache items in scope_info_.back(), then call
    // scope_info_.pop()
    std::vector<scope_info_t> scope_info_;
    int for_depth_ = 0;
    std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
            &alias_map_;
    std::unordered_set<expr_c> loop_vars_;
    const for_loop_node_t *cur_for_loop_ = nullptr;

    // flushes the cache
    void invalidate(tensor_cache_ptr c, // NOLINT
            std::vector<stmt> *writeback_point = nullptr) { // NOLINT
        if (c->is_valid()) {
            // if the cache is dirty, write back after the last write
            if (c->last_write_.defined()) {
                if (!writeback_point) {
                    writeback_point = &c->last_write_->seq_;
                }
                writeback_point->emplace_back(builder::make_assign_unattached(
                        builder::make_indexing(
                                c->tsr_, c->idx_, c->lanes_, c->mask_),
                        c->var_));
            }
            // mark the cache invalid
            cached_index_.erase(c->tsr_);
            scope_info_.back().num_flushes_[c->tsr_]++;
            c->tsr_ = tensor_c();
        }
    }

    // invalidates a tensor in the cache, returns true if is in cache
    bool invalidate_if_exist(const expr_c &arg) {
        auto tsr = arg.static_as<tensor>();
        auto itr = cached_index_.find(tsr);
        if (itr != cached_index_.end()) {
            invalidate(itr->second);
            return true;
        }
        return false;
    }

    bool is_var_dependent_only_on_loop_var(const expr_c &itr, int recur_count) {
        if (recur_count > 7) {
            // the dependency is too complex, or there is a loop in dependency
            return false;
        }
        if (loop_vars_.count(itr) == 0) {
            // if it depends on a non-loop var
            if (auto var_state
                    = itr->get_temp_data()
                              .get_or_null<var_state_analysis_result_t>()) {
                if (var_state->assignments_ == 1 && !var_state->is_complex_) {
                    for (auto v : var_state->depending_on_) {
                        auto ptr = v.lock();
                        if (ptr.get() == itr.get()) {
                            // self dependent
                            return false;
                        }
                        if (!is_var_dependent_only_on_loop_var(
                                    expr_c(ptr), recur_count + 1)) {
                            return false;
                        }
                    }
                }
            } else {
                return false;
            }
        }
        return true;
    }

    void collect_var_dependency(const std::shared_ptr<expr_base> &itr,
            utils::weakptr_hashset_t<expr_base> &out) {
        if (out.has(itr)) { return; }
        out.insert(itr);
        if (auto var_state
                = itr->get_temp_data()
                          .get_or_null<var_state_analysis_result_t>()) {
            if (var_state->assignments_ == 1 && !var_state->is_complex_) {
                for (auto v : var_state->depending_on_) {
                    auto ptr = v.lock();
                    collect_var_dependency(ptr, out);
                }
            }
        }
    }

    // inserts an indexing to cache
    // if `is_read`, sets the cached var to the indexing value
    // if the indexing node is not cache-able, returns `v` and leaves
    // `out_cache` = nullptr
    expr_c make_cache(indexing_c v, bool is_read, tensor_cache_ptr &out_cache) {
        SC_MODULE_INFO << "Make cache: " << v;
        // the vars that the indices of `v` depends on
        utils::weakptr_hashset_t<expr_base> vars;
        // if we can trace the changes of the indices
        bool is_valid = var_dependency_finder_t::find(vars, v->idx_);
        if (v->mask_.defined()) {
            is_valid = is_valid
                    && var_dependency_finder_t::find(vars, &(v->mask_));
        }
        if (!is_valid) {
            out_cache = nullptr;
            return std::move(v);
        }
        tensor tsr = v->ptr_.as<tensor>();
        assert(tsr.defined());
        if (auto ana = tsr->get_temp_data()
                               .get_or_null<tensor_usage_analysis_result_t>()) {
            // if the tensor is used in broadcast and it is currently loaded as
            // scalar
            if (ana->used_in_broadcast_ && v->dtype_.lanes_ == 1) {
                out_cache = nullptr;
                return std::move(v);
            }
        }
        var vcache = builder::make_var(
                v->dtype_, "__cached_" + std::to_string(var_cnt_++))
                             .static_as<var>();
        assert(insertion_point_);
        // declare a var, insert before the current stmt
        // if read, set the var cache to the value in memory
        insertion_point_->emplace_back(builder::make_var_tensor_def_unattached(
                vcache, linkage::local, is_read ? v.remove_const() : expr()));
        out_cache = std::make_shared<tensor_cache_t>(tsr,
                std::vector<expr_c>(v->idx_.begin(), v->idx_.end()), vcache,
                v->dtype_.lanes_, v->mask_);
        scope_info_.back().outstanding_cache_.insert(out_cache);
        // remember the dependency
        // and check if the index only depends on loop vars
        bool only_loop_dependent = true;
        for (auto itr : vars) {
            auto ptr = itr.lock();
            dependency_map_.insert(std::make_pair(expr(ptr), out_cache));
            only_loop_dependent = only_loop_dependent
                    && is_var_dependent_only_on_loop_var(expr_c(ptr), 0);
        }
        if (only_loop_dependent) {
            utils::weakptr_hashset_t<expr_base> dep_vars;
            for (auto dvar : vars) {
                collect_var_dependency(dvar.lock(), dep_vars);
            }
            // traverse the parent for-loops to find if it is possible to move
            // the definition up in a later moment
            scope_info_t *lifted_scope = nullptr;
            for (auto itr = scope_info_.rbegin(); itr != scope_info_.rend();
                    ++itr) {
                auto &sinfo = *itr;
                if (!sinfo.loop_) { break; }
                for_loop_node_t *nested_loop
                        = sinfo.loop_->body_.cast<stmts>()
                                  .filter([](const stmts &v) {
                                      return v->seq_.size() == 1UL;
                                  })
                                  .map([](const stmts &v) {
                                      return v->seq_[0].as<for_loop>().get();
                                  })
                                  .get_or_else(nullptr);

                lifted_scope = &sinfo;
                if (sinfo.loop_->kind_ != for_type::NORMAL) { break; }
                if (dep_vars.has(sinfo.loop_->var_.impl) != 0) {
                    // if the index depends on the current loop, break
                    break;
                }
                if (itr != scope_info_.rbegin() && !nested_loop) { break; }
            }
            if (lifted_scope && lifted_scope != &scope_info_.back()) {
                out_cache->may_lift_to_ = lifted_scope;
                out_cache->dependencies_ = std::move(dep_vars);
            }
        }
        // put into the cache
        cached_index_.insert(std::make_pair(tsr, out_cache));
        return std::move(vcache);
    }

    expr_c visit(cast_c v) override {
        if (v->in_.isa<tensor>()) {
            if (invalidate_alias_group(v->in_, true)) {
                SC_MODULE_INFO << "Evict due to cast: " << v;
            }
            scope_info_.back().num_flushes_[v->in_]++;
        }
        return ir_visitor_t::visit(std::move(v));
    }

    expr_c visit(call_c v) override {
        auto ret = ir_visitor_t::visit(std::move(v));
        for (auto &arg : ret.as<call_c>()->args_) {
            if (arg.isa<tensor>()) {
                if (invalidate_alias_group(arg, true)) {
                    SC_MODULE_INFO << "Evict due to function call: " << ret;
                }
                scope_info_.back().num_flushes_[arg]++;
            }
        }
        return ret;
    }

    expr_c visit(tensorptr_c v) override {
        // dispatch the fields inside of indexing, without calling
        // index2var_t::visit on indexing. We don't need to create cache slot
        // for tensorptr
        auto ret_base = ir_visitor_t::visit(v->base_);
        auto ret_idx = ret_base.as<indexing_c>();
        auto tsr = ret_idx->ptr_.as<tensor>();
        if (invalidate_alias_group(tsr, true)) {
            SC_MODULE_INFO << "Evict due to tensorptr: " << v;
        }
        scope_info_.back().num_flushes_[tsr]++;
        if (ret_base.ptr_same(v->base_)) {
            return std::move(v);
        } else {
            return builder::tensor_ptr(tsr, ret_idx->idx_);
        }
    }

    bool invalidate_alias_group(const expr_c &tsr, bool invalidate_self) {
        auto analysis_result
                = tsr->get_temp_data()
                          .get_or_null<tensor_usage_analysis_result_t>();
        bool ret = false;
        if (analysis_result && analysis_result->alias_id_) {
            auto ths = this;
            // if the tensor has alias
            analysis_result->for_each_alias_tensor(
                    alias_map_, [&tsr, ths, &ret](const expr_c &v) {
                        if (!v.ptr_same(tsr)) {
                            ret |= ths->invalidate_if_exist(v);
                        }
                    });
        }
        if (invalidate_self) { ret |= invalidate_if_exist(tsr); }
        return ret;
    }

    expr_c visit_indexing(
            indexing_c v, bool is_read, tensor_cache_ptr &out_cache) {
        auto ret = ir_visitor_t::visit(std::move(v)).as<indexing_c>();
        auto tsr = ret->ptr_.as<tensor>();
        if (tsr->attr_
                && tsr->attr_->get_or_else(attr_keys::must_tensor2var, false)) {
            // if the tensor is marked to be transformed to var, no need to
            // optimize
            return ret;
        }
        if (tsr->attr_
                && tsr->attr_->get_or_else(attr_keys::no_index2var, false)) {
            return ret;
        }
        if (!is_read) {
            // if it is not read, need to evict all other tensors in the alias
            // group. no need to invalidate tsr itself
            if (invalidate_alias_group(tsr, false)) {
                SC_MODULE_INFO << "Alias group invalidated for " << tsr;
            }
        }
        auto itr = cached_index_.find(tsr);
        if (itr != cached_index_.end()) {
            // if the tensor is cached
            if (itr->second->is_match(ret)) {
                // the cached index is a match, return the var
                out_cache = itr->second;
                // If the indexing is cached, we can use it if:
                // 1. it is read and we are not in for_loop (if it is in a
                // for-loop, we currently do not know if there will be writes on
                // the cache, which will invalidate the parent scope cache)
                // 2. or the cache is created in the same scope of the access
                if ((is_read && for_depth_ == 0)
                        || scope_info_.back().is_cache_defined_here(out_cache)
                        || scope_info_.back().tensor_not_written_here(tsr)) {
                    return itr->second->var_;
                }
                SC_MODULE_INFO << "Evict parent scope cache in child scope: "
                               << ret;
                // if we need to write to a cached var which is defined in
                // parent scopes (not in current scope), we need to evict it
                // If we don't, consider this case
                // A[0] = 1 // cached here in parent scope
                // if (...) {
                //   A[0] = 1 // no write back here
                // } else {
                //   A[0] = 1 // write back due to the use of A[1]
                //   A[1] = 2
                // }
                // in the above case, the else-block will evict A[0] because
                // the use of A[1], but the last use of A[0] is still in the
                // else block, so the write to A[0] in then-block will be
                // lost
            } else {
                SC_MODULE_INFO << "Evict old for unmatched index: " << ret;
            }
            // the tensor is cached, but with different index, evict it
            invalidate(itr->second);
        }
        return make_cache(std::move(ret), is_read, out_cache);
    }

    expr_c visit(indexing_c v) override {
        if (v->attr_ && v->attr_->get_or_else(attr_keys::no_index2var, false)) {
            return v;
        }
        tensor_cache_ptr out_cache;
        return visit_indexing(std::move(v), true, out_cache);
    }

    stmt_c visit(stmts_c v) override {
        auto old_insert = insertion_point_;
        std::vector<stmt_c> seq;
        seq.reserve(v->seq_.size());
        insertion_point_ = &seq;
        const for_loop_node_t *parent_loop = cur_for_loop_;
        cur_for_loop_ = nullptr;
        scope_info_.emplace_back(scope_info_t {parent_loop, insertion_point_,
                v->get_temp_data()
                        .get<written_tensor_analysis_result_t>()
                        .written_,
                {}, {}, {}});

        bool changed = false;
        for (auto &s : v->seq_) {
            auto newstmt = dispatch(s);
            changed |= !newstmt.ptr_same(s);
            seq.emplace_back(std::move(newstmt));
            auto &insert_after = scope_info_.back().insert_after_;
            if (!insert_after.empty()) {
                changed = true;
                seq.insert(seq.end(), insert_after.begin(), insert_after.end());
                insert_after.clear();
            }
        }
        changed |= v->seq_.size() != seq.size();

        // evict all cache items that will die at the end of this stmts
        for (auto &v : scope_info_.back().outstanding_cache_) {
            // where to writeback the cache: the parent scope? or after the last
            // write?
            if (v->is_valid()) {
                SC_MODULE_INFO << "Evict at the end of scope: " << v->tsr_;
                auto check_is_tensor_defined_in_current_scope = [&v, &seq]() {
                    for (auto &s : seq) {
                        if (s.cast<define>()
                                        .filter([&v](const define &def) {
                                            return def->var_.get()
                                                    == v->tsr_.get();
                                        })
                                        .has_value()) {
                            return true;
                        }
                    }
                    return false;
                };
                std::vector<stmt> *writeback_point = nullptr;
                // check if tensor cache can be lifted: a) check may_lift_to_ b)
                // check if the cache has been flushed before c) if the tensor
                // is defined in current scope, we cannot lift it
                if (v->may_lift_to_
                        && scope_info_.back().num_flushes_[v->tsr_] == 0
                        && !check_is_tensor_defined_in_current_scope()) {
                    utils::weakptr_hashset_t<expr_base> &dep_vars
                            = v->dependencies_;
                    // if the tensor cache var can be lifted, move the
                    // definition
                    for (auto itr = seq.begin(); itr != seq.end();) {
                        // if the assign node assigns to dependency var
                        bool assign_dep
                                = (*itr).cast<assign>()
                                          .map([](const assign &def) {
                                              return def->var_.as<var>().get();
                                          })
                                          .filter([&dep_vars](var_node *var) {
                                              return dep_vars.has(
                                                      var->shared_from_this());
                                          })
                                          .get_or_else(nullptr);
                        // if the var definition is for the cache var or
                        // dependency var
                        // clang-format off
                        bool def_var = (*itr).cast<define>()
                                .map([](const define &def) {
                                    return def->var_.as<var>().get();
                                })
                                .filter([&v, &dep_vars](
                                                var_node *var) {
                                    return var == v->var_.get()
                                            || dep_vars.has(
                                                    var->shared_from_this());
                                })
                                .get_or_else(nullptr);
                        // clang-format on
                        if (assign_dep || def_var) {
                            v->may_lift_to_->insert_point_->emplace_back(
                                    std::move(*itr));
                            itr = seq.erase(itr);
                            continue;
                        }

                        ++itr;
                    }
                    writeback_point = &v->may_lift_to_->insert_after_;
                }
                invalidate(v, writeback_point);
            }
        }
        if (scope_info_.size() > 1) {
            auto itrprev = scope_info_.rbegin();
            auto &prev = *++itrprev;
            for (auto &kv : scope_info_.back().num_flushes_) {
                prev.num_flushes_[kv.first] += kv.second;
            }
        }
        scope_info_.pop_back();
        insertion_point_ = old_insert;

        if (changed) { return builder::make_stmts_unattached(seq); }
        return std::move(v);
    }
    stmt_c visit(for_loop_c v) override {
        for_depth_++;
        cur_for_loop_ = v.get();
        loop_vars_.insert(v->var_);
        auto ret = ir_visitor_t::visit(v);
        loop_vars_.erase(v->var_);
        for_depth_--;
        return ret;
    }
    stmt_c visit(assign_c v) override {
        if (v->var_.isa<indexing_c>()) {
            auto rhs = dispatch(v->value_);
            tensor_cache_ptr out_cache;
            auto lhs = visit_indexing(
                    v->var_.static_as<indexing_c>(), false, out_cache);
            // cache creation may fail due to there is a call/indexing in
            // the indices
            if (out_cache) {
                auto mask = v->var_.static_as<indexing_c>()->mask_;
                if (mask.defined()) {
                    if (mask.isa<constant>()) {
                        rhs = get_expr_as_int(mask) > 0
                                ? rhs
                                : builder::make_constant(
                                        {UINT64_C(0)}, rhs->dtype_);
                    } else {
                        rhs = builder::make_select(mask, rhs,
                                builder::make_constant(
                                        {UINT64_C(0)}, rhs->dtype_));
                    }
                }
                // if successfully created a cache for the indexing
                auto ret = builder::make_stmts_unattached(
                        {builder::make_assign_unattached(lhs, rhs)});
                out_cache->last_write_ = ret.static_as<stmts>();
                return ret;
            } else if (!rhs.ptr_same(v->value_) || !lhs.ptr_same(v->var_)) {
                return builder::make_assign_unattached(lhs, rhs);
            } else {
                return std::move(v);
            }
        } else {
            assert(v->var_.isa<var>());
            // if a var is changed, all indexing nodes depends on this var
            // should be evicted
            auto its = dependency_map_.equal_range(v->var_);
            if (its.first != its.second) {
                for (auto it = its.first; it != its.second; ++it) {
                    if (it->second->is_valid()) {
                        SC_MODULE_INFO
                                << "Evict due to change of index = " << v->var_
                                << ", tensor = " << it->second->tsr_;
                        invalidate(it->second);
                    }
                }
                dependency_map_.erase(v->var_);
            }
            return ir_visitor_t::visit(std::move(v));
        }
    }

public:
    using ir_visitor_t::dispatch;
    indexing2var_impl_t(
            std::unordered_map<alias_info::tensor_alias_identity_t *, expr_c>
                    &alias_map)
        : alias_map_(alias_map) {}
};

func_c index2var_t::operator()(func_c f) {
    if (f->attr_ && f->attr_->get_or_else(function_attrs::low_level, false)) {
        return f;
    }
    index2var_analysis_t pass;
    pass.dispatch(f);
    indexing2var_impl_t impl {pass.alias_map_};
    return impl.dispatch(f);
}

stmt_c index2var_t::operator()(const stmts_c &f) {
    index2var_analysis_t pass;
    pass.dispatch(f);
    indexing2var_impl_t impl {pass.alias_map_};
    return impl.dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
