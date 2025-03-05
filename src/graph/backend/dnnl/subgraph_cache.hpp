/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef BACKEND_DNNL_SUBGRAPH_CACHE_HPP
#define BACKEND_DNNL_SUBGRAPH_CACHE_HPP

#include <memory>
#include <mutex>
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/interface/partition_hashing.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

class subgraph_cache_t {
public:
    // Get the singleton instance of the cache
    static subgraph_cache_t &instance() {
        static subgraph_cache_t instance;
        return instance;
    }

    // Get a subgraph from the cache
    std::shared_ptr<subgraph_t> get(const size_t &key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) { return it->second; }
        return nullptr;
    }

    // Add a subgraph to the cache
    void put(const size_t &key, const std::shared_ptr<subgraph_t> &subgraph) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = subgraph;
    }

private:
    subgraph_cache_t() = default;
    ~subgraph_cache_t() = default;

    // Disable copy and assignment
    subgraph_cache_t(const subgraph_cache_t &) = delete;
    subgraph_cache_t &operator=(const subgraph_cache_t &) = delete;

    std::unordered_map<size_t, std::shared_ptr<subgraph_t>> cache_;
    std::mutex mutex_;
};

size_t subgraph_cache_key(const dnnl_partition_impl_t *part,
        const engine_t *g_engine, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    size_t seed = 0;
    // Combine hash for op_kinds & attributes with the computed hash
    seed = hash_combine(seed, inputs.size());
    seed = hash_combine(seed, outputs.size());
    seed = partition_hashing::get_array_hash(seed, part->get_ops());
    return seed;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // BACKEND_DNNL_SUBGRAPH_CACHE_HPP