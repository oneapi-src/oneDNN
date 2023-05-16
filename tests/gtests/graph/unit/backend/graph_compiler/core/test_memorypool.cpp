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

#include <iostream>
#include <thread>
#include <runtime/config.hpp>
#include <runtime/context.hpp>
#include <runtime/memorypool.hpp>
#include <runtime/os.hpp>
#include <runtime/parallel.hpp>
#include <runtime/runtime.hpp>
#include <runtime/thread_locals.hpp>
#include <util/utils.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc::memory_pool;

TEST(GCCore_CPU_test_memorypool, TestMemoryPool) {
    // push 10; push large; push 100; push 200; pop 200; pop 100; pop large;
    // push page-200; push 200; pop 200; pop page-200; push 10
    unsigned pagesize = dnnl::impl::graph::gc::runtime::get_os_page_size();
    auto *rctx = dnnl::impl::graph::gc::runtime::get_default_stream();
    filo_memory_pool_t pool(pagesize);
    auto ptr1 = pool.alloc(rctx, 10);
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr1) % 64, 0);
    auto block1 = pool.current_;
    ASSERT_EQ(block1->allocated_, 64u + 10);
    ASSERT_EQ(block1->size_, pagesize);

    auto ptr2 = pool.alloc(rctx,
            2 * pagesize
                    - 64); // very large buffer, but can be allocated in 2 pages
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr2) % 64, 0);
    auto block2 = pool.current_;
    ASSERT_NE(block1, block2);
    ASSERT_EQ(block2->allocated_, 2 * pagesize);
    ASSERT_EQ(block2->size_, 2 * pagesize);

    // small object in new block
    auto ptr3 = pool.alloc(rctx, 100);
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr3) % 64, 0);
    auto block3 = pool.current_;
    ASSERT_NE(block3, block2);
    ASSERT_EQ(block3->allocated_, 64u + 100);
    ASSERT_EQ(block3->size_, pagesize);

    // small object in the current block
    auto ptr4 = pool.alloc(rctx, 200);
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr4) % 64, 0);
    ASSERT_EQ(block3, pool.current_);
    ASSERT_EQ(block3->allocated_, 64u * 3 + 200);

    // pop 200
    pool.dealloc(ptr4);
    ASSERT_EQ(block3, pool.current_);
    ASSERT_EQ(block3->allocated_, 64u + 100);

    // pop 100
    pool.dealloc(ptr3);
    ASSERT_EQ(block2, pool.current_);
    ASSERT_EQ(block2->allocated_, 2u * pagesize);

    // pop ptr2
    pool.dealloc(ptr2);
    ASSERT_EQ(block1, pool.current_);
    ASSERT_EQ(block1->allocated_, 64u + 10);

    // alloc pagesize -200, should not switch block
    auto ptr5 = pool.alloc(rctx, pagesize - 200);
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr5) % 64, 0);
    ASSERT_EQ(block1, pool.current_);
    ASSERT_EQ(block1->allocated_, 128u + pagesize - 200);

    // alloc 200, should reuse block2
    auto ptr6 = pool.alloc(rctx, 200);
    ASSERT_EQ(reinterpret_cast<intptr_t>(ptr6) % 64, 0);
    ASSERT_EQ(block2, pool.current_);
    ASSERT_EQ(block2->allocated_, 64u + 200);

    // pop ptr6
    pool.dealloc(ptr6);
    ASSERT_EQ(block1, pool.current_);
    ASSERT_EQ(block1->allocated_, 128u + pagesize - 200);

    // pop ptr5
    pool.dealloc(ptr5);
    ASSERT_EQ(block1, pool.current_);
    ASSERT_EQ(block1->allocated_, 64u + 10);

    // pop ptr1
    pool.dealloc(ptr1);
    ASSERT_EQ(block1, pool.current_);
    ASSERT_EQ(block1->allocated_, sizeof(memory_block_t));
}

using namespace dnnl::impl::graph::gc;

static void thread_workload(void *v1, void *v2, int64_t i, generic_val *args) {
    auto s = dnnl::impl::graph::gc::runtime::get_default_stream();
    void *a = sc_aligned_malloc(s, 64);
    void *b = sc_thread_aligned_malloc(s, 64);
    sc_aligned_free(s, a);
    sc_thread_aligned_free(s, b);
}

static void run_alloc_and_free() {
    sc_parallel_call_cpu_with_env_impl(thread_workload, 0, nullptr, nullptr, 0,
            runtime_config_t::get().get_num_threads(), 1, nullptr);
};

TEST(GCCore_CPU_test_memorypool, TestMemoryPoolRelease) {
    run_alloc_and_free();
    auto stream = runtime::get_default_stream();
    filo_memory_pool_t *thread_p
            = &runtime::get_tls(stream).thread_memory_pool_,
            *main_p = &runtime::get_tls(stream).main_memory_pool_;
    ASSERT_NE(thread_p->buffers_, nullptr);
    ASSERT_NE(main_p->buffers_, nullptr);
    dnnl::impl::graph::gc::release_runtime_memory(stream->engine_);
    ASSERT_EQ(thread_p->buffers_, nullptr);
    ASSERT_EQ(main_p->buffers_, nullptr);
    // make sure that after resetting, the memory pool can still work
    run_alloc_and_free();

    {
        // make sure when a thread is destroyed before exit(), everything is ok
        std::thread th {thread_workload, nullptr, nullptr, 0, nullptr};
        th.join();
    }
}
