/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/utils.hpp>

SC_MODULE(pass.index2var)

namespace sc {

// the visitor to find the mutable dependencies in the indices of indexing
// nodes. e.g., for A[i+j], it will find i and j as dependencies. Note that if
// there is indexing_node/call_node in the indices, this pass will set
// `is_valid_` = false, meaning that the indices are untraceable - we are unable
// to tell if the indices are changed after some statements.
class var_dependency_finder_t : public ir_viewer_t {
    std::unordered_set<expr_c> *vars_;
    // the output flag to mark if the input expr is good for index2var_t
    bool is_valid_ = true;
    var_dependency_finder_t(std::unordered_set<expr_c> *vars) : vars_(vars) {}

    void view(call_c v) override {
        is_valid_ = false;
        SC_MODULE_INFO << "Found call node in index: " << v;
    }
    void view(indexing_c v) override {
        is_valid_ = false;
        SC_MODULE_INFO << "Found indexing node in index: " << v;
    }
    void view(var_c v) override { vars_->insert(v); }

public:
    static bool find(
            std::unordered_set<expr_c> &vars, const std::vector<expr> &idx) {
        var_dependency_finder_t f(&vars);
        for (auto &v : idx) {
            f.dispatch(v);
        }
        return f.is_valid_;
    }
};

class indexing2var_impl_t : public ir_visitor_t {
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
                    if (!cmp.compare(v->mask_, mask_)) { return false; }
                }
                return true;
            }
            return false;
        }
        bool is_valid() const { return tsr_.defined(); }
    };
    using tensor_cache_ptr = std::shared_ptr<tensor_cache_t>;
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
    // evict all cache items in outstanding_cache_.back(), then call
    // outstanding_cache_.pop()
    std::vector<std::unordered_set<tensor_cache_ptr>> outstanding_cache_;
    int for_depth_ = 0;

    // flushes the cache
    void invalidate(tensor_cache_ptr c) { // NOLINT
        if (c->is_valid()) {
            // if the cache is dirty, write back after the last write
            if (c->last_write_.defined()) {
                c->last_write_->seq_.emplace_back(
                        builder::make_assign_unattached(
                                builder::make_indexing(
                                        c->tsr_, c->idx_, c->lanes_, c->mask_),
                                c->var_));
            }
            // mark the cache invalid
            cached_index_.erase(c->tsr_);
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

    // inserts an indexing to cache
    // if `is_read`, sets the cached var to the indexing value
    // if the indexing node is not cache-able, returns `v` and leaves
    // `out_cache` = nullptr
    expr_c make_cache(indexing_c v, bool is_read, tensor_cache_ptr &out_cache) {
        SC_MODULE_INFO << "Make cache: " << v;
        // the vars that the indices of `v` depends on
        std::unordered_set<expr_c> vars;
        // if we can trace the changes of the indices
        bool is_valid = var_dependency_finder_t::find(vars, v->idx_);
        if (!is_valid) {
            out_cache = nullptr;
            return std::move(v);
        }
        tensor tsr = v->ptr_.as<tensor>();
        assert(tsr.defined());
        var vcache = builder::make_var(
                v->dtype_, "__cached_" + std::to_string(var_cnt_++))
                             .static_as<var>();
        assert(insertion_point_);
        // declare a var, insert before the current stmt
        insertion_point_->emplace_back(
                builder::make_var_tensor_def_unattached(vcache));
        if (is_read) {
            // if read, set the var cache to the value in memory
            insertion_point_->emplace_back(
                    builder::make_assign_unattached(vcache, v));
        }
        out_cache = std::make_shared<tensor_cache_t>(tsr,
                std::vector<expr_c>(v->idx_.begin(), v->idx_.end()), vcache,
                v->dtype_.lanes_, v->mask_);
        outstanding_cache_.back().insert(out_cache);
        // remember the dependency
        for (auto &itr : vars) {
            dependency_map_.insert(std::make_pair(itr, out_cache));
        }
        // put into the cache
        cached_index_.insert(std::make_pair(tsr, out_cache));
        return std::move(vcache);
    }

    expr_c visit(call_c v) override {
        auto ret = ir_visitor_t::visit(std::move(v));
        for (auto &arg : ret.as<call_c>()->args_) {
            if (arg.isa<tensor>()) {
                if (invalidate_if_exist(arg)) {
                    SC_MODULE_INFO << "Evict due to function call: " << ret;
                }
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
        if (invalidate_if_exist(tsr)) {
            SC_MODULE_INFO << "Evict due to tensorptr: " << v;
        }
        if (ret_base.ptr_same(v->base_)) {
            return std::move(v);
        } else {
            return builder::tensor_ptr(tsr, ret_idx->idx_);
        }
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
                        || outstanding_cache_.back().find(out_cache)
                                != outstanding_cache_.back().end()) {
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
        outstanding_cache_.emplace_back(std::unordered_set<tensor_cache_ptr>());

        bool changed = false;
        for (auto &s : v->seq_) {
            auto newstmt = dispatch(s);
            changed |= !newstmt.ptr_same(s);
            seq.emplace_back(std::move(newstmt));
        }
        changed |= v->seq_.size() != seq.size();

        // evict all cache items that will die at the end of this stmts
        for (auto &v : outstanding_cache_.back()) {
            if (v->is_valid()) {
                SC_MODULE_INFO << "Evict at the end of scope: " << v->tsr_;
                invalidate(v);
            }
        }
        outstanding_cache_.pop_back();
        insertion_point_ = old_insert;

        if (changed) { return builder::make_stmts_unattached(seq); }
        return std::move(v);
    }
    stmt_c visit(for_loop_c v) override {
        for_depth_++;
        auto ret = ir_visitor_t::visit(std::move(v));
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
};

func_c index2var_t::operator()(func_c f) {
    indexing2var_impl_t impl;
    return impl.dispatch(f);
}

stmt_c index2var_t::operator()(const stmts_c &f) {
    indexing2var_impl_t impl;
    return impl.dispatch(f);
}

} // namespace sc
