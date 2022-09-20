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

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/scratchpad.hpp"

#include "graph/unit/unit_test_common.hpp"

#include <dnnl.hpp>

namespace graph = dnnl::impl::graph;

TEST(Scratchpad, TemporaryScratchpad) {
    using dnnl::impl::graph::allocator_t;
    using dnnl::impl::graph::dnnl_impl::temporary_scratchpad_t;

    graph::engine_t *g_eng = get_engine();
    allocator_t *alloc = static_cast<allocator_t *>(g_eng->get_allocator());
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    // No seg fault here because the memory will be deleted when
    // temporary_scratchpad_t is destroyed
    std::vector<size_t> buf_sizes = {512, 1024, 2048, 4096};
    for (size_t i = 0; i < buf_sizes.size(); i++) {
        temporary_scratchpad_t scratchpad(buf_sizes[i], p_eng, *alloc);
        // the scratchpad size will be changed
        ASSERT_EQ(buf_sizes[i], scratchpad.size());
    }

    buf_sizes = {4096, 2048, 1024, 512};
    for (size_t i = 0; i < buf_sizes.size(); i++) {
        temporary_scratchpad_t scratchpad(buf_sizes[i], p_eng, *alloc);
        // the scratchpad size will be changed
        ASSERT_EQ(buf_sizes[i], scratchpad.size());
    }
}

TEST(Scratchpad, Registry) {
    using dnnl::impl::graph::allocator_t;
    using dnnl::impl::graph::dnnl_impl::grantor_t;
    using dnnl::impl::graph::dnnl_impl::registrar_t;
    using dnnl::impl::graph::dnnl_impl::registry_t;

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
        ASSERT_EQ((size_t)address % alignment, 0U);
    }

    char *piece_end = aligned_grantor.get(keys.back()) + sizes.back();
    char *total_end = aligned_base_ptr + registry.size();
    ASSERT_TRUE(piece_end <= total_end); // make sure no overflow

    // the base pointer is not aligned
    char *unaligned_base_ptr = (char *)4631;
    grantor_t unaligned_grantor = registry.grantor(unaligned_base_ptr);

    ASSERT_TRUE(unaligned_grantor.get(keys[0]) > unaligned_base_ptr);
    ASSERT_TRUE((size_t)(unaligned_grantor.get(keys[0]) - unaligned_base_ptr)
            < registry.lcm_alignment());

    for (size_t i = 0; i < keys.size(); i++) {
        char *address = unaligned_grantor.get(keys[i]);
        ASSERT_EQ((size_t)address % alignment, 0U);
    }

    piece_end = unaligned_grantor.get(keys.back()) + sizes.back();
    total_end = unaligned_base_ptr + registry.size();
    ASSERT_TRUE(piece_end <= total_end); // make sure no overflow
}

TEST(Scratchpad, RegistryMultithreading) {
    using dnnl::impl::graph::allocator_t;
    using dnnl::impl::graph::dnnl_impl::grantor_t;
    using dnnl::impl::graph::dnnl_impl::registrar_t;
    using dnnl::impl::graph::dnnl_impl::registry_t;

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
            ASSERT_EQ((size_t)address % alignment, 0U);
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
