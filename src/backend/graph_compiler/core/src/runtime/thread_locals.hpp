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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_THREAD_LOCALS_HPP
#include "memorypool.hpp"

namespace sc {
namespace runtime {
struct engine;
struct stream;

struct amx_buffer_t {
    void *ptr_ = nullptr;
    stream_t *stream_ = nullptr;
    ~amx_buffer_t();
    void reset(stream_t *stream);
    void release();
};

struct thread_local_buffer {
    amx_buffer_t amx_buffer;
    // if the current thread is the "main" thread, use this pool
    memory_pool::filo_memory_pool_t main_memory_pool {
            memory_pool::main_chunk_size};
    // if the current thread is a worker thread, use this pool
    memory_pool::filo_memory_pool_t thread_memory_pool {
            memory_pool::threadlocal_chunk_size};
};

extern thread_local thread_local_buffer tls_buffer;

} // namespace runtime
} // namespace sc

#endif
