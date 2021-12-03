/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include "context.hpp"
#include <assert.h>
#include "memorypool.hpp"
#include "parallel.hpp"
#include "runtime.hpp"

namespace sc {

namespace runtime {
static void *global_alloc(runtime::engine *eng, size_t sz) {
    return sc_global_aligned_alloc(sz, 64);
}

static void global_free(runtime::engine *eng, void *p) {
    return sc_global_aligned_free(p, 64);
}

static stream_vtable_t vtable {global_alloc, global_free,
        sc::memory_pool::alloc_by_mmap, sc::memory_pool::dealloc_by_mmap,
        sc_parallel_call_cpu_with_env_impl};

stream_t *get_default_stream() {
    static std::shared_ptr<stream_t> default_stream
            = std::make_shared<stream_t>(&vtable);
    return default_stream.get();
}

} // namespace runtime
} // namespace sc
