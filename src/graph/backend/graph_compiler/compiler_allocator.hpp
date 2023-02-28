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

#ifndef BACKEND_GRAPH_COMPILER_COMPILER_ALLOCATOR_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_ALLOCATOR_HPP

#include <memory>
#include <unordered_set>

#include "common/engine.hpp"
#include "graph/interface/allocator.hpp"

#include "runtime/context.hpp"
#include "runtime/parallel.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

struct compiler_graph_engine_t : public gc::runtime::engine_t {
    allocator_t *allocator_;
    compiler_graph_engine_t(
            gc::runtime::engine_vtable_t *vtable, allocator_t *allocator)
        : gc::runtime::engine_t {vtable}, allocator_ {allocator} {}
};

struct compiler_graph_stream_t : public gc::runtime::stream_t {
    compiler_graph_stream_t(compiler_graph_engine_t *eng);
};

extern gc::runtime::engine_vtable_t graph_engine_vtable;
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
