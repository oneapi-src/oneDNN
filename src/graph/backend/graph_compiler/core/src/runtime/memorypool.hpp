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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MEMORYPOOL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_MEMORYPOOL_HPP

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct stream_t;
struct engine_t;
} // namespace runtime

namespace memory_pool {

// 4MB
constexpr size_t threadlocal_chunk_size = 4 * 1024 * 1024;
// 16MB
constexpr size_t main_chunk_size = 16 * 1024 * 1024;

// The chunk of memory that is allocated to the user
struct memory_chunk_t {
    static constexpr uint64_t magic_check_num_ = 0xc0ffeebeef0102ff;
    // `canary` should be set as a magic value to check the existence of
    // overflow
    uint64_t canary_;
    // the size of the memory_chunk_t allocated, incluing this `memory_chunk_t`
    size_t size_;
    // the memory for the user
    char buffer_[0];
    // initalizes the memory chunk, given the address of data
    static memory_chunk_t *init(intptr_t pdata, size_t sz);
};

// the control block for pre-allocated memory block - created by page-wise
// allocation system calls (mmap). We can divide the memory block into memory
// chunks for user memory allocation in memory starting from `buffer_`
struct memory_block_t {
    // size of the memory block, starting from `this`
    size_t size_;
    // size of allocated bytes, including this struct
    size_t allocated_;
    memory_block_t *prev_;
    memory_block_t *next_;
    runtime::engine_t *engine_;
    // here starts the allocatable memory
    char buffer_[0];

    /**
     * Calculates the next pointer to allocate with alignment = 512-bits (64
     * bytes). The  (returned pointer - sizeof(memory_chunk_t)) should be the
     * address of memory_chunk_t
     * */
    intptr_t calc_alloc_ptr();

    static memory_block_t *make(runtime::stream_t *stream, size_t sz,
            memory_block_t *prev, memory_block_t *next);
};

// The FILO memory pool. The memory allocation and deallocation should be in
// first-in-last-out fashion
struct filo_memory_pool_t {
    size_t block_size_;
    // the linked list of all allocated memory blocks
    memory_block_t *buffers_ = nullptr;
    memory_block_t *current_ = nullptr;
    size_t get_block_size(size_t sz) const;
    void *alloc(runtime::stream_t *stream, size_t sz);
    void dealloc(void *ptr);
    filo_memory_pool_t(size_t block_size) : block_size_(block_size) {}
    ~filo_memory_pool_t();
    // release the memory to os/underlying memory allocator
    void release();
    // reset the memory pool, but keep the allocated memory in the pool
    void clear();
};
void dealloc_by_mmap(runtime::engine_t *eng, void *b);
void *alloc_by_mmap(runtime::engine_t *eng, size_t sz);
} // namespace memory_pool
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

extern "C" SC_API void *sc_aligned_malloc(
        dnnl::impl::graph::gc::runtime::stream_t *stream, size_t sz) noexcept;
extern "C" SC_API void sc_aligned_free(
        dnnl::impl::graph::gc::runtime::stream_t *stream, void *p) noexcept;
extern "C" SC_API void *sc_thread_aligned_malloc(
        dnnl::impl::graph::gc::runtime::stream_t *stream, size_t sz) noexcept;
extern "C" SC_API void sc_thread_aligned_free(
        dnnl::impl::graph::gc::runtime::stream_t *stream, void *p) noexcept;

#endif
