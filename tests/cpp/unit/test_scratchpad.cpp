/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/scratchpad.hpp"

#include <dnnl.hpp>

TEST(scratchpad, buffer_can_reuse) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::thread_local_scratchpad_t;

    // FIXME(qun) When the alloc, engine and scratchpad are in the same thread,
    // then the alloc and engine may be freed before scratchpad's thread local
    // buffer. This will cause segment fault when existing thread because the
    // buffer free rely on the allocator instance.

    allocator_t alloc;
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    auto func = [&]() {
        std::vector<size_t> buf_sizes = {2048, 1024, 2048, 512};
        char *orig_address = nullptr;
        for (size_t i = 0; i < buf_sizes.size(); i++) {
            thread_local_scratchpad_t scratchpad(buf_sizes[i], eng, alloc);
            if (!orig_address) {
                orig_address = scratchpad.get_buffer();
                ASSERT_EQ(buf_sizes[i], scratchpad.size());
            } else {
                // if buffer size is smaller than existing buffer size,
                // then we will reuse the existing one
                ASSERT_EQ(orig_address, scratchpad.get_buffer());
                // the scratchpad size will not be changed
                ASSERT_EQ(buf_sizes[0], scratchpad.size());
                ASSERT_TRUE(buf_sizes[i] <= scratchpad.size());
            }
        }
    };

    std::thread t1(func);
    t1.join();
}

TEST(scratchpad, buffer_cannot_reuse) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::thread_local_scratchpad_t;

    // FIXME(qun) When the alloc, engine and scratchpad are in the same thread,
    // then the alloc and engine may be freed before scratchpad's thread local
    // buffer. This will cause segment fault when existing thread because the
    // buffer free rely on the allocator instance.

    allocator_t alloc;
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    auto func = [&]() {
        std::vector<size_t> buf_sizes = {512, 1024, 2048, 4096};
        char *orig_address = nullptr;
        for (size_t i = 0; i < buf_sizes.size(); i++) {
            thread_local_scratchpad_t scratchpad(buf_sizes[i], eng, alloc);
            // the scratchpad size will be changed
            ASSERT_EQ(buf_sizes[i], scratchpad.size());
        }
    };

    std::thread t1(func);
    t1.join();
}

TEST(scratchpad, temporary_scratchpad) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::temporary_scratchpad_t;

    allocator_t alloc;
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    // No seg fault here because the memory will be deleted when
    // temporary_scratchpad_t is destroyed
    std::vector<size_t> buf_sizes = {512, 1024, 2048, 4096};
    for (size_t i = 0; i < buf_sizes.size(); i++) {
        temporary_scratchpad_t scratchpad(buf_sizes[i], eng, alloc);
        // the scratchpad size will be changed
        ASSERT_EQ(buf_sizes[i], scratchpad.size());
    }

    buf_sizes = {4096, 2048, 1024, 512};
    for (size_t i = 0; i < buf_sizes.size(); i++) {
        temporary_scratchpad_t scratchpad(buf_sizes[i], eng, alloc);
        // the scratchpad size will be changed
        ASSERT_EQ(buf_sizes[i], scratchpad.size());
    }
}

TEST(scratchpad, multithreading) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::thread_local_scratchpad_t;

    allocator_t alloc;
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    char *t1_scratchpad_address = nullptr;
    char *t2_scratchpad_address = nullptr;

    auto func = [&](size_t id) {
        size_t buf_size = 1024;
        thread_local_scratchpad_t scratchpad(buf_size, eng, alloc);
        if (id == 1)
            t1_scratchpad_address = scratchpad.get_buffer();
        else
            t2_scratchpad_address = scratchpad.get_buffer();

        // make sure that a thread is still runing when another thread allocate
        // buffer
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    };

    std::thread t1(func, 1);
    std::thread t2(func, 2);
    t1.join();
    t2.join();

    ASSERT_NE(t1_scratchpad_address, t2_scratchpad_address);
}

TEST(scratchpad, registry) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::grantor_t;
    using dnnl::graph::impl::dnnl_impl::registrar_t;
    using dnnl::graph::impl::dnnl_impl::registry_t;

    size_t alignment = 64;
    std::vector<registry_t::key_t> keys = {0, 1, 2, 3, 4};
    std::vector<size_t> sizes = {12, 58, 99, 500, 1024};

    registry_t registry;
    registrar_t registrar = registry.registrar();

    for (size_t i = 0; i < keys.size(); i++) {
        registrar.book(keys[i], sizes[i], alignment);
    }

    // the base pointer is aligned
    char *aligned_base_ptr = (char *)4096;
    grantor_t aligned_grantor = registry.grantor(aligned_base_ptr);

    ASSERT_EQ(aligned_grantor.get(keys[0]), aligned_base_ptr);

    for (size_t i = 0; i < keys.size(); i++) {
        char *address = aligned_grantor.get(keys[i]);
        ASSERT_EQ((size_t)address % alignment, 0);
    }

    char *piece_end = aligned_grantor.get(keys.back()) + sizes.back();
    char *total_end = aligned_base_ptr + registry.size();
    ASSERT_TRUE(piece_end <= total_end); // make sure no overflow

    // the base pointer is not aligned
    char *unaligned_base_ptr = (char *)4631;
    grantor_t unaligned_grantor = registry.grantor(unaligned_base_ptr);

    ASSERT_TRUE(unaligned_grantor.get(keys[0]) > unaligned_base_ptr);
    ASSERT_TRUE((unaligned_grantor.get(keys[0]) - unaligned_base_ptr)
            < registry.lcm_alignment());

    for (size_t i = 0; i < keys.size(); i++) {
        char *address = unaligned_grantor.get(keys[i]);
        ASSERT_EQ((size_t)address % alignment, 0);
    }

    piece_end = unaligned_grantor.get(keys.back()) + sizes.back();
    total_end = unaligned_base_ptr + registry.size();
    ASSERT_TRUE(piece_end <= total_end); // make sure no overflow
}

TEST(scratchpad, registry_multithreading) {
    using dnnl::graph::impl::allocator_t;
    using dnnl::graph::impl::dnnl_impl::grantor_t;
    using dnnl::graph::impl::dnnl_impl::registrar_t;
    using dnnl::graph::impl::dnnl_impl::registry_t;

    size_t alignment = 64;
    std::vector<registry_t::key_t> keys = {0, 1, 2, 3, 4};
    std::vector<size_t> sizes = {12, 58, 99, 500, 1024};

    registry_t registry;
    registrar_t registrar = registry.registrar();

    for (size_t i = 0; i < keys.size(); i++) {
        registrar.book(keys[i], sizes[i], alignment);
    }

    // make sure the registry is thread safe when access it through grantor in
    // multithreads
    auto func = [&](size_t id) {
        // the base pointer is aligned
        char *aligned_base_ptr = (char *)(id * 1024);
        grantor_t grantor = registry.grantor(aligned_base_ptr);

        for (size_t i = 0; i < keys.size(); i++) {
            char *address = grantor.get(keys[i]);
            ASSERT_EQ((size_t)address % alignment, 0);
        }

        char *piece_end = grantor.get(keys.back()) + sizes.back();
        char *total_end = aligned_base_ptr + registry.size();
        ASSERT_TRUE(piece_end <= total_end); // make sure no overflow
    };

    std::thread t1(func, 1);
    std::thread t2(func, 2);
    t1.join();
    t2.join();
}
