/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include <chrono>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/partition.hpp"
#include "graph/interface/partition_cache.hpp"

#include "common/rw_mutex.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {

compiled_partition_cache_t &compiled_partition_cache() {
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    static const int capacity
            = getenv_int("DNNL_GRAPH_COMPILED_PARTITION_CACHE_CAPACITY", 1024);
#else
    static const int capacity = 0;
#endif
    static compiled_partition_cache_t cache(capacity);
    return cache;
}

const partition_t *compiled_partition_cache_t::get_partition(const key_t &key) {
    result_t result = cache_.get(key);
    return result.value != nullptr ? &(result.value->src_partition()) : nullptr;
}

} // namespace graph
} // namespace impl
} // namespace dnnl

// API
dnnl::impl::graph::status_t dnnl_graph_get_compiled_partition_cache_capacity(
        int *capacity) {
    if (capacity == nullptr)
        return dnnl::impl::graph::status::invalid_arguments;
    *capacity = 0;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    *capacity = dnnl::impl::graph::compiled_partition_cache().get_capacity();
#endif
    return dnnl::impl::graph::status::success;
}

dnnl::impl::graph::status_t dnnl_graph_set_compiled_partition_cache_capacity(
        int capacity) {
    if (capacity < 0) return dnnl::impl::graph::status::invalid_arguments;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    return dnnl::impl::graph::compiled_partition_cache().set_capacity(capacity);
#endif
    return dnnl::impl::graph::status::success;
}
