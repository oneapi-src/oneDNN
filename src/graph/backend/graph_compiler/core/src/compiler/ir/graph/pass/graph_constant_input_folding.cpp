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

#include <atomic>
#include <list>
#include <mutex>
#include "../fused_op.hpp"
#include "../fusible_op.hpp"
#include "../visitor.hpp"
#include "graph_constant_cache.hpp"
#include "pass.hpp"
#include <compiler/ir/statics_table.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <unordered_map>

SC_MODULE(graph.pass.const_input_fold);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

size_t graph_weak_ptr_hasher::operator()(
        const std::weak_ptr<sc_graph_t> &v) const {
    auto g = v.lock();
    if (!g) { return 0; }
    return g->hash_contents();
}
bool graph_weak_ptr_cmper::operator()(const std::weak_ptr<sc_graph_t> &v1,
        const std::weak_ptr<sc_graph_t> &v2) const {
    auto g1 = v1.lock();
    if (!g1) { return false; }
    auto g2 = v2.lock();
    if (!g2) { return false; }
    if (g1 == g2) { return true; }
    return compare_graph(*g1, *g2);
}

struct shared_global_data_allocator_t {
    std::mutex lock_;
    size_t needed_allocation_size(
            const std::vector<std::shared_ptr<cached_const_graph_tensor>>
                    &cache,
            const std::vector<void *> &existing_data, size_t &out_real_const) {
        size_t ret_lazy = 0, ret_compile_time_const = 0;
        for (size_t i = 0; i < cache.size(); i++) {
            auto &v = cache[i];
            size_t &val = existing_data[i] ? ret_compile_time_const : ret_lazy;
            if (!v->buf_base_) {
                val += utils::divide_and_ceil(v->size_, 64) * 64;
            }
        }
        out_real_const = ret_compile_time_const;
        return ret_lazy;
    }
    void
    alloc(const std::vector<std::shared_ptr<cached_const_graph_tensor>> &cache,
            const std::vector<void *> &existing_data,
            runtime::engine_t *engine) {
        size_t ret_lazy = 0, ret_compile_time_const = 0;
        ret_lazy = needed_allocation_size(
                cache, existing_data, ret_compile_time_const);
        if (ret_lazy + ret_compile_time_const) {
            std::lock_guard<std::mutex> guard {lock_};
            // do double-check locking
            ret_lazy = needed_allocation_size(
                    cache, existing_data, ret_compile_time_const);
            if (ret_lazy) {
                // need to register into const cache manager for
                // lazy-initialized buffers
                auto base = runtime::create_and_register_const_cache(
                        engine, ret_lazy);
                size_t offset = 0;
                for (size_t idx = 0; idx < cache.size(); idx++) {
                    auto &v = cache[idx];
                    if (existing_data[idx]) continue;
                    // the update on buf_ is protected by lock_
                    if (!v->buf_base_) {
                        v->buf_base_ = base;
                        v->offset_ = offset;
                        offset += utils::divide_and_ceil(v->size_, 64) * 64;
                        SC_MODULE_INFO << "Alloc buffer for " << v
                                       << ", offset=" << v->offset_;
                    }
                }
            }
            if (ret_compile_time_const) {
                // no need to register into const cache manager for
                // compile-time constants because they are never evicted
                std::shared_ptr<void> baseptr = std::shared_ptr<void> {
                        engine->vtable_->persistent_alloc(
                                engine, ret_compile_time_const),
                        [engine](void *p) {
                            engine->vtable_->persistent_dealloc(engine, p);
                        }};
                auto base = std::make_shared<runtime::const_cache_proxy>(
                        baseptr, baseptr.get(), ret_compile_time_const,
                        /*lazy*/ false);
                size_t offset = 0;
                for (size_t idx = 0; idx < cache.size(); idx++) {
                    auto &v = cache[idx];
                    if (!existing_data[idx]) continue;
                    // the update on buf_ is protected by lock_
                    if (!v->buf_base_) {
                        v->buf_base_ = base;
                        v->offset_ = offset;
                        memcpy((char *)baseptr.get() + offset,
                                existing_data[idx], v->size_);
                        offset += utils::divide_and_ceil(v->size_, 64) * 64;
                        SC_MODULE_INFO << "Alloc buffer for " << v
                                       << ", offset=" << v->offset_;
                    }
                }
            }
        }
    }
};

static std::shared_ptr<const_graph_tensor_cache> get_cache();
struct const_graph_tensor_cache {
    std::mutex lock_;
    tensor_id_map from_tensor_id_;
    graph_weak_ptr_map from_dep_graph_;
    shared_global_data_allocator_t alloca_;

    std::shared_ptr<cached_const_graph_tensor> add_tensor(
            const std::shared_ptr<sc_graph_t> &dep_graph, size_t buf_size,
            runtime::engine_t *engine) {
        std::lock_guard<std::mutex> guard {lock_};
        auto itr = from_dep_graph_.find(dep_graph);
        if (itr != from_dep_graph_.end()) {
            auto ret = itr->second.v_.lock();
            if (ret) { return ret; }
            // the weakptr expired. mark the key as deleted.
            *itr->second.deletion_flag_ = true;
            from_dep_graph_.erase(itr);
        }
        auto ret = std::make_shared<cached_const_graph_tensor>(
                dep_graph, buf_size, get_cache());
        // insert into graph map
        auto graph_iter = from_dep_graph_.insert(std::make_pair(dep_graph,
                flaged_cached_const_graph_tensor_t {ret, ret->deletion_flag_}));
        assert(graph_iter.second);
        // allocate a unique ID for the tensor
        auto hash = dep_graph->hash_contents();
        bool found = false;
        for (int retries = 0; retries < 3000; retries++) {
            auto id_iter = from_tensor_id_.find(hash);
            if (id_iter == from_tensor_id_.end()) {
                found = true;
                break;
            }
            hash++;
        }
        COMPILE_ASSERT(found, "Cannot insert unique cached tensor id");
        auto id_iter = from_tensor_id_.insert(std::make_pair(hash, ret));
        ret->graph_iter_ = graph_iter.first;
        ret->id_iter_ = id_iter.first;
        return ret;
    }

    void remove(cached_const_graph_tensor &v) {
        std::lock_guard<std::mutex> guard {lock_};
        /* need to check if the key in the map is already deleted. Consider the
         * following case if we don't use the deletion_flag_.
         * 1. Thread 1: The last reference to a cached_const_graph_tensor is
         * destroyed, calling the destructor of cached_const_graph_tensor
         * 2. Thread 2: Before thread 1 enters
         * const_graph_tensor_cache::remove(), thread2 enters add_tensor. The
         * graph_key is the same as but a different instance of the graph_key of
         * thread1's. Thread2 will find that the graph is already in
         * from_dep_graph_ as the key, but the value (weakptr to
         * cached_const_graph_tensor) already expired. So thread2 cannot reuse
         * the key-value already in from_dep_graph_. It removes the key from
         * from_dep_graph_ and adds a new key to it.
         * 3. Thread 1 enters const_graph_tensor_cache::remove() and calls
         * from_dep_graph_.erase(v.graph_iter_) - however, the iterator is
         * already invalidated because in step 2, thread2 removed it.
         *
         */
        bool already_deleted = *v.deletion_flag_;
        if (already_deleted) { return; }
        if (v.graph_iter_ != from_dep_graph_.end())
            from_dep_graph_.erase(v.graph_iter_);
        if (v.id_iter_ != from_tensor_id_.end())
            from_tensor_id_.erase(v.id_iter_);
    }
};

static std::shared_ptr<const_graph_tensor_cache> get_cache() {
    static std::shared_ptr<const_graph_tensor_cache> c
            = std::make_shared<const_graph_tensor_cache>();
    return c;
}

cached_const_graph_tensor::~cached_const_graph_tensor() {
    // if dependency_, the cached tensor should be created by unit tests
    if (dependency_) cache_owner_->remove(*this);
}

static std::atomic<size_t> internal_tensor_id = {0xfffff000};

SC_INTERNAL_API void graph_constant_input_folding_impl(
        sc_graph_t &mgr, const context_ptr &ctx, bool share_constants) {
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(mgr.ops_.size());
    std::vector<sc_op *> edge_ops;
    vis.visit_graph(mgr, [&edge_ops](op_visitor_t *vis, const sc_op_ptr &node) {
        if (node->isa<input_op>()) {
            if (!node->attrs_.get_or_null<size_t>("temp.tensor_id")) {
                size_t tsr_id = internal_tensor_id++;
                node->attrs_["temp.tensor_id"] = tsr_id;
            }
        }
        auto cur_const_state
                = node->attrs_.get_or_else("constant", const_kind::not_const);
        if (node->isa<constant_op_t>() || cur_const_state) {
            if (node->isa<constant_op_t>()) {
                edge_ops.emplace_back(node.get());
                node->attrs_.set("constant", const_kind::local_const);
            }
            bool all_constant_inputs = true;
            for (const auto &input : node->get_inputs()) {
                auto parent_node = input->producer_owner_;
                if (parent_node->attrs_.get_or_else(
                            "constant", const_kind::not_const)
                                == const_kind::not_const
                        && !parent_node->isa<constant_op_t>()) {
                    all_constant_inputs = false;
                    break;
                }
            }
            if (!all_constant_inputs) {
                for (const auto &input : node->get_inputs()) {
                    auto parent_node = input->producer_owner_;
                    if (parent_node->attrs_.get_or_else(
                                "constant", const_kind::not_const)
                            != const_kind::not_const) {
                        parent_node->attrs_.set(
                                "constant", const_kind::global_const);
                        edge_ops.emplace_back(parent_node);
                        if (parent_node->isa<tensor_view_op_t>()) {
                            auto tv_in = parent_node;
                            while (tv_in->isa<tensor_view_op_t>()) {
                                tv_in = tv_in->get_inputs()[0]->producer_owner_;
                            }
                            tv_in->attrs_.set(
                                    "constant", const_kind::global_const);
                            edge_ops.emplace_back(tv_in);
                        }
                    }
                }
                node->attrs_.set("constant", const_kind::not_const);
            } else {
                if (!node->isa<output_op>()) {
                    // Setting attrs here is intermediary status.
                    // Current node is constant node, its uses may
                    // also be constant, so we set `local_const`
                    // here temporarily meaning `may_constant`.
                    // Later when visiting its uses, we check their
                    // all inputs and decide whether we reserve this
                    // attr.
                    for (auto &out : node->get_outputs()) {
                        for (auto &cld_node : out->uses_) {
                            cld_node.second->attrs_.set(
                                    "constant", const_kind::local_const);
                        }
                    }
                }
            }
        }
    });
    if (share_constants && !edge_ops.empty() && ctx->flags_.const_share_) {
        op_dep_matrix_t dependency {mgr};
        std::vector<size_t> hash_cache(mgr.ops_.size());
        std::vector<void *> existing_data_vec;
        // the list of cached tensors in the whole graph
        std::vector<std::shared_ptr<cached_const_graph_tensor>> caches;
        for (auto op : edge_ops) {
            if (op->attrs_.has_key(op_attr_key::const_input_cache)) {
                continue;
            }
            if (op->isa<input_op>()) { continue; }
            // the list of cached tensors in this op
            std::vector<std::shared_ptr<cached_const_graph_tensor>> results;
            // the op-ids of inputs and constants that current op depends on
            std::list<sc_op_ptr> depending_inputs;
            // if all of the dependent ops are tensor_view, skip
            bool is_all_tensor_view = true;
            std::vector<bool> op_mask(mgr.ops_.size());
            for (size_t i = 0; i < mgr.ops_.size(); i++) {
                if (i != (size_t)op->logical_op_id_
                        && dependency.lookup(i, op->logical_op_id_) == 1) {
                    // if is input/constant
                    if (mgr.ops_[i]->get_inputs().empty()) {
                        depending_inputs.emplace_back(mgr.ops_[i]);
                    }
                    op_mask[i] = true;
                    if (!mgr.ops_[i]->isa<tensor_view_op_t>()
                            && !mgr.ops_[i]->isa<input_op>()
                            && !mgr.ops_[i]->isa<constant_op_t>()) {
                        is_all_tensor_view = false;
                    }
                }
            }
            if (is_all_tensor_view && op->isa<tensor_view_op_t>()) { continue; }
            // the selector for graph visitor to visit sub graph for the
            // current op. Try to normalize the visiting order by hash of
            // the ops
            auto selector
                    = [&op_mask, &hash_cache](op_visitor_t *v) -> sc_op_ptr {
                std::list<sc_op_ptr>::iterator theitr;
                size_t max_hash = 0;
                bool found = false;
                for (auto itr = v->to_visit_.begin();
                        itr != v->to_visit_.end();) {
                    auto &cur = *itr;
                    // if the cur is not depended by op, skip
                    if (v->has_visited(cur->logical_op_id_)
                            || !op_mask[cur->logical_op_id_]) {
                        itr = v->to_visit_.erase(itr);
                        continue;
                    }
                    auto &cached_hash = hash_cache[cur->logical_op_id_];
                    size_t cur_hash = 0;
                    if (!cached_hash) {
                        cur_hash = cur->hash_contents();
                        cached_hash = cur_hash;
                    } else {
                        cur_hash = cached_hash;
                    }
                    // find the op with max hash value
                    if (cur_hash >= max_hash) {
                        max_hash = cur_hash;
                        found = true;
                        theitr = itr;
                    }
                    ++itr;
                }
                if (found) {
                    auto ret = *theitr;
                    v->to_visit_.erase(theitr);
                    return ret;
                }
                return nullptr;
            };
            op_visitor_t visitor {selector,
                    op_visitor_t::create_DAG_updater(mgr.ops_.size()), false};
            std::vector<size_t> depending_ops;
            visitor.to_visit_ = std::move(depending_inputs);
            visitor.visit(
                    [&depending_ops](op_visitor_t *v, const sc_op_ptr &p) {
                        depending_ops.emplace_back(p->logical_op_id_);
                    });
            depending_ops.emplace_back(op->logical_op_id_);
            for (size_t i = 0; i < op->get_outputs().size(); i++) {
                void *existing_data = nullptr;
                if (auto c_op = op->dyn_cast<constant_op_t>()) {
                    auto &src = *(c_op->get_constant_values());
                    existing_data = src.data_;
                }
                existing_data_vec.emplace_back(existing_data);
                auto buf_size = op->get_outputs()[i]
                                        ->details_.get_blocking_byte_size();
                if (buf_size <= 8UL) {
                    // skip scalars, don't register it in the cache manager
                    auto cached_tsr
                            = std::make_shared<cached_const_graph_tensor>(
                                    nullptr, buf_size, get_cache());
                    // mark as deleted from the cache manager
                    *cached_tsr->deletion_flag_ = true;
                    results.emplace_back(cached_tsr);
                    caches.emplace_back(cached_tsr);
                    continue;
                }
                // for each output tensor of the edge op, rebuild the
                // dependency graph
                auto g = std::make_shared<sc_graph_t>();
                std::vector<sc_op_ptr> newops;
                newops.resize(mgr.ops_.size());
                for (auto &depop_id : depending_ops) {
                    const auto &theop = mgr.ops_[depop_id];
                    if (theop->isa<input_op>()) {
                        newops[depop_id] = g->make_input(
                                copy_logical_tsr(theop->get_outputs()),
                                theop->attrs_);
                        newops[depop_id]->attrs_["constant_input_id"]
                                = theop->attrs_["temp.tensor_id"];
                    } else {
                        auto copyable
                                = theop->dyn_cast<op_traits::copyable_t>();
                        COMPILE_ASSERT(copyable,
                                "The const cache ops must be copyable");
                        std::vector<graph_tensor_ptr> newins;
                        auto &oldins = theop->get_inputs();
                        for (size_t in_idx = 0; in_idx < oldins.size();
                                in_idx++) {
                            auto oldowner = oldins[in_idx]->producer_owner_;
                            auto &mapped_op = newops[oldowner->logical_op_id_];
                            auto idx
                                    = std::find(oldowner->get_outputs().begin(),
                                              oldowner->get_outputs().end(),
                                              oldins[in_idx])
                                    - oldowner->get_outputs().begin();
                            newins.emplace_back(mapped_op->get_outputs()[idx]);
                        }
                        newops[depop_id] = copyable->copy(newins,
                                copy_logical_tsr(theop->get_outputs()), *g);
                    }
                }
                auto &lastop = newops[op->logical_op_id_];
                g->make_output({lastop->get_outputs()[i]});
                results.emplace_back(get_cache()->add_tensor(g,
                        op->get_outputs()[i]->details_.get_blocking_byte_size(),
                        ctx->engine_));
                caches.emplace_back(results.back());
                if (auto sc_stream_temp = ::dnnl::impl::graph::gc::runtime::
                                get_info_logging_stream(__sc_module_name)) {
                    (*sc_stream_temp.stream_)
                            << "Putting into shared cache: " << results.back()
                            << ", uid="
                            << (void *)results.back()->id_iter_->first
                            << ", size=" << results.back()->size_
                            << ", dep =\n";
                    print_graph(*g, *sc_stream_temp.stream_, true, true);
                }
            }
            // mark the op with const_input_cache attr. We need to skip the
            // tensorview ops, because we don't generate tensor buffer for these
            // ops
            auto cached_op = op;
            while (cached_op->isa<tensor_view_op_t>()) {
                cached_op = cached_op->get_inputs()[0]->producer_owner_;
            }
            cached_op->attrs_[op_attr_key::const_input_cache]
                    = std::move(results);
        }
        get_cache()->alloca_.alloc(caches, existing_data_vec, ctx->engine_);
    }
}

SC_INTERNAL_API void graph_constant_input_folding(
        sc_graph_t &mgr, const context_ptr &ctx) {
    graph_constant_input_folding_impl(mgr, ctx, false);
}
SC_INTERNAL_API void graph_constant_input_folding_and_share_constants(
        sc_graph_t &mgr, const context_ptr &ctx) {
    graph_constant_input_folding_impl(mgr, ctx, true);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
