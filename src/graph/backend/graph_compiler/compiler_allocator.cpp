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
#include <memory>

#include "compiler_allocator.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

using namespace gc::runtime;

#define ALLOCATOR_ALIGNMENT 64

using sc_engine_t = gc::runtime::engine_t;

static void *compiler_graph_global_alloc(sc_engine_t *eng, size_t sz) {
    return static_cast<compiler_graph_engine_t *>(eng)->allocator_->allocate(
            sz, {allocator_t::mem_type_t::persistent, ALLOCATOR_ALIGNMENT});
}

static void compiler_graph_global_free(sc_engine_t *eng, void *p) {
    static_cast<compiler_graph_engine_t *>(eng)->allocator_->deallocate(p);
}

#if 0
static void *compiler_graph_temp_alloc(sc_engine_t *eng, size_t sz) {
    return static_cast<compiler_graph_engine_t *>(eng)->allocator_->allocate(sz,
            {allocator_t::mem_type_t::temp, ALLOCATOR_ALIGNMENT});
}

static void compiler_graph_temp_free(sc_engine_t *eng, void *p) {
    static_cast<compiler_graph_engine_t *>(eng)->allocator_->deallocate(p);
}
#endif

engine_vtable_t graph_engine_vtable {compiler_graph_global_alloc,
        compiler_graph_global_free, compiler_graph_global_alloc,
        compiler_graph_global_free};

compiler_graph_stream_t::compiler_graph_stream_t(compiler_graph_engine_t *eng)
    : gc::runtime::stream_t {{sc_parallel_call_cpu_with_env_impl}, eng} {}

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
