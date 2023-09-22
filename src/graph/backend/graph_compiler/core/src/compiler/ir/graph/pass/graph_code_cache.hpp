/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_GRAPH_CODE_CACHE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_GRAPH_CODE_CACHE_HPP

#include <memory>
#include <compiler/config/context.hpp>
#include <compiler/ir/statics_table.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * The graph code cache uses prehashed_graph_for_code_share_t as the key and
 * graph_code_cache_handle as the value. It contains a weakptr to the
 * jit_module_code. The jit_module_code should hold a sharedptr to this handle.
 * When a jit_module_code is destructed, it will destroy its
 * graph_code_cache_handle as well. This will automatically unregister the graph
 * in graph code cache
 */
struct graph_code_cache_handle;

/**
 * The key in graph code cache
 */
struct prehashed_graph_for_code_share_t;
class sc_graph_t;
struct jit_module;

/**
 * Create and prepare the JIT module data of a cached graph.
 */
statics_table_t prepare_static_table_for_cached_code(
        graph_code_cache_handle &v, const sc_graph_t &orig_graph);

/**
 * Register the compilation result and the query key (graph) into the graph code
 * cache.
 *
 * @returns the pointer to the cache item handle. This can be null if the key
 * already exists in the cache
 */
std::shared_ptr<graph_code_cache_handle> register_code_in_graph_cache(
        const jit_module &m,
        std::shared_ptr<prehashed_graph_for_code_share_t> &&key);

// get the number of cache code for a given context
size_t query_cached_code_of_context(const context_ptr &ctx);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
