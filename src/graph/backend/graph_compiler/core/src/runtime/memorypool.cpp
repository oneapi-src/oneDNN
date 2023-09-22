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

#include <memory.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include "context.hpp"
#include "memorypool.hpp"
#include "thread_locals.hpp"
#include <runtime/os.hpp>
#include <util/simple_math.hpp>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#endif

#ifdef _MSC_VER
#define __builtin_expect(EXP_, C) (EXP_)
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace memory_pool {
using utils::divide_and_ceil;
static constexpr size_t default_alignment = 64;

memory_chunk_t *memory_chunk_t::init(intptr_t pdata, size_t sz) {
    memory_chunk_t *ths = reinterpret_cast<memory_chunk_t *>(
            pdata - sizeof(memory_chunk_t));
    ths->canary_ = magic_check_num_;
    ths->size_ = sz;
    return ths;
}

intptr_t memory_block_t::calc_alloc_ptr() {
    intptr_t start_addr = reinterpret_cast<intptr_t>(this) + allocated_
            + sizeof(memory_chunk_t);
    return divide_and_ceil(start_addr, default_alignment) * default_alignment;
}

void *alloc_by_mmap(runtime::engine_t *eng, size_t sz) {
#ifdef _MSC_VER
    auto ret = VirtualAlloc(
            nullptr, sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    auto ret = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif
    assert(ret);
    return ret;
}

memory_block_t *memory_block_t::make(runtime::stream_t *stream, size_t sz,
        memory_block_t *prev, memory_block_t *next) {
    auto ret = stream->engine_->vtable_->temp_alloc(stream->engine_, sz);
    if (!ret) { throw std::runtime_error("Out of Memory."); }
    memory_block_t *blk = reinterpret_cast<memory_block_t *>(ret);
    blk->size_ = sz;
    blk->allocated_ = sizeof(memory_block_t);
    blk->engine_ = stream->engine_;
    static_assert(sizeof(memory_block_t) == offsetof(memory_block_t, buffer_),
            "sizeof(memory_block_t) == offsetof(memory_block_t, buffer_)");
    blk->prev_ = prev;
    blk->next_ = next;
    return blk;
}

void dealloc_by_mmap(runtime::engine_t *eng, void *b) {
#ifdef _MSC_VER
    auto ret = VirtualFree(b, 0, MEM_RELEASE);
    SC_UNUSED(ret);
    assert(ret);
#else
    munmap(b, reinterpret_cast<memory_block_t *>(b)->size_);
#endif
}

static void free_memory_block_list(memory_block_t *b) {
    while (b) {
        memory_block_t *next = b->next_;
        auto engine = b->engine_;
        engine->vtable_->temp_dealloc(engine, b);
        b = next;
    }
}

size_t filo_memory_pool_t::get_block_size(size_t sz) const {
    // calculate the aligned size of management blocks in the header
    constexpr size_t header_size
            = divide_and_ceil(sizeof(memory_block_t) + sizeof(memory_chunk_t),
                      default_alignment)
            * default_alignment;
    // the allocated size should include the aligned header size
    sz = sz + header_size;
    if (sz > block_size_) {
        return divide_and_ceil(sz, runtime::get_os_page_size())
                * runtime::get_os_page_size();
    } else {
        return block_size_;
    }
}

void *filo_memory_pool_t::alloc(runtime::stream_t *stream, size_t sz) {
    if (unlikely(!buffers_)) {
        buffers_ = memory_block_t::make(
                stream, get_block_size(sz), nullptr, nullptr);
        current_ = buffers_;
    }
    do {
        intptr_t newptr = current_->calc_alloc_ptr();
        size_t newallocated
                = newptr + sz - reinterpret_cast<intptr_t>(current_);
        if (likely(newallocated <= current_->size_)) {
            // if the current block is not full
            size_t alloc_size = newallocated - current_->allocated_;
            current_->allocated_ = newallocated;
            memory_chunk_t *chunk = memory_chunk_t::init(newptr, alloc_size);
            return reinterpret_cast<void *>(newptr);
        }
        // if the block is full, check the next block
        // if there is no next block left, allocate a new one
        if (!current_->next_) {
            current_->next_ = memory_block_t::make(
                    stream, get_block_size(sz), current_, nullptr);
        }
        current_ = current_->next_;
    } while (true);
}

void filo_memory_pool_t::dealloc(void *ptr) {
    auto intptr = reinterpret_cast<intptr_t>(ptr);
    auto intcur = reinterpret_cast<intptr_t>(current_);
    // Optional: check if the pointer is valid in the current block

    assert(intptr > intcur
            && intptr - intcur < static_cast<ptrdiff_t>(current_->size_));
    auto chunk = reinterpret_cast<memory_chunk_t *>(
            intptr - sizeof(memory_chunk_t));
    // Optional: check if the stack is ok
    assert(chunk->canary_ == memory_chunk_t::magic_check_num_
            && "Corrupt stack detected");
    assert(current_->allocated_ > chunk->size_);
    current_->allocated_ -= chunk->size_;

    // skip the empty blocks
    while (unlikely(current_->allocated_ == sizeof(memory_block_t))) {
        if (current_->prev_) {
            current_ = current_->prev_;
        } else {
            break;
        }
    }
}

void filo_memory_pool_t::release() {
    free_memory_block_list(buffers_);
    buffers_ = nullptr;
    current_ = nullptr;
}

filo_memory_pool_t::~filo_memory_pool_t() {
    release();
}

} // namespace memory_pool
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

using stream_t = dnnl::impl::graph::gc::runtime::stream_t;
namespace runtime = dnnl::impl::graph::gc::runtime;
extern "C" SC_API void *sc_aligned_malloc(
        stream_t *pstream, size_t sz) noexcept {
    if (sz == 0) { return nullptr; }
    return runtime::get_tls(pstream).main_memory_pool_.alloc(pstream, sz);
}

extern "C" SC_API void sc_aligned_free(stream_t *pstream, void *p) noexcept {
    runtime::get_tls(pstream).main_memory_pool_.dealloc(p);
}

extern "C" SC_API void *sc_thread_aligned_malloc(
        stream_t *pstream, size_t sz) noexcept {
    return runtime::get_tls(pstream).thread_memory_pool_.alloc(pstream, sz);
}

extern "C" SC_API void sc_thread_aligned_free(
        stream_t *pstream, void *p) noexcept {
    runtime::get_tls(pstream).thread_memory_pool_.dealloc(p);
}
