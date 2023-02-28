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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_RUNTIME_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_RUNTIME_HPP
#include <stddef.h>
#include <stdint.h>
#include <util/def.hpp>

extern "C" {

SC_API void print_float(float f);
SC_API void print_index(uint64_t f);
SC_API void print_int(int f);
SC_API void print_str(char *f);
SC_API void *sc_global_aligned_alloc(size_t sz, size_t align);
SC_API void sc_global_aligned_free(void *ptr, size_t align);
SC_API void sc_make_trace(int id, int in_or_out, int arg);
SC_API void sc_make_trace_kernel(int id, int in_or_out, int arg);
// dynamic
SC_API void *sc_extract_dyn_base(void *tsr);
SC_API void *sc_extract_dyn_shape(void *tsr);
SC_API void sc_initialize_dyn_tsr(
        void *dyn_tsr, void *tsr, void *shapes, uint8_t dyn_mask, int ndims);
};

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct engine_t;
}
SC_API void release_runtime_memory(runtime::engine_t *engine);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
