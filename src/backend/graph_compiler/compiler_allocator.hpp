/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include "interface/allocator.hpp"
#include "interface/engine.hpp"

#include "runtime/context.hpp"
#include "runtime/parallel.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

struct compiler_graph_engine_t : public sc::runtime::engine_t {
    impl::allocator_t *allocator_ = nullptr;
    compiler_graph_engine_t(sc::runtime::engine_vtable_t *vtable)
        : sc::runtime::engine_t {vtable} {}
};

struct compiler_graph_stream_t : public sc::runtime::stream_t {
    impl::allocator_t *allocator_ = nullptr;
    impl::engine_t engine_; // used to ensure allocator's existence
    compiler_graph_stream_t(sc::runtime::stream_vtable_t *vtable)
        : sc::runtime::stream_t {vtable} {}
};

extern sc::runtime::engine_vtable_t graph_engine_vtable;
extern sc::runtime::stream_vtable_t graph_stream_vtable;
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
