/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#include "gtest/gtest.h"

#include "interface/constant_tensor_cache.hpp"

#include "backend/dnnl/dnnl_constant_tensor_cache.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "utils/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace dnnl_impl = graph::dnnl_impl;

TEST(ConstantCache, SetGetCapacity) {
    graph::constant_tensor_cache_t cache(0);
    ASSERT_EQ(cache.set_capacity(11), graph::status::success);
    ASSERT_EQ(cache.get_capacity(), 11U);
}

TEST(ConstantCache, GetOrAddEmpty) {
    using key_t = graph::constant_tensor_cache_t::key_t;
    using value_t = graph::constant_tensor_cache_t::value_t;

    graph::constant_tensor_cache_t cache(100);
    ASSERT_EQ(cache.set_capacity(0), graph::status::success);
    ASSERT_FALSE(cache.get_or_add(key_t(), key_t(), 1024, value_t()).valid());
}

TEST(ConstantCache, CombineKey) {
    using key_t = graph::constant_tensor_cache_t::key_t;

    key_t backend_id = 0;
    key_t backend_specific_key = 10;
    key_t key1 = graph::constant_tensor_cache_t::combine_key(
            backend_id, backend_specific_key);

    backend_specific_key = 20;
    key_t key2 = graph::constant_tensor_cache_t::combine_key(
            backend_id, backend_specific_key);

    ASSERT_NE(key1, key2);

    backend_id = 1;
    key_t key3 = graph::constant_tensor_cache_t::combine_key(
            backend_id, backend_specific_key);
    ASSERT_NE(key2, key3);
}

TEST(ConstantCache, NoEvictWhenCacheFull) {
    graph::engine_t &engine = *get_engine();
    auto p_engine_ = dnnl_impl::make_dnnl_engine(engine);
    auto g_alloc_ = static_cast<graph::allocator_t *>(engine.get_allocator());

    graph::constant_tensor_cache_t cache(0);
    ASSERT_EQ(cache.set_capacity(0), graph::status::success);
    ASSERT_EQ(cache.set_capacity(5), graph::status::success);

    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise1;
    ASSERT_NO_THROW(cache.get_or_add(0, 1, 1, c_promise1.get_future()));
    graph::constant_tensor_cache_t::cached_t c_buffer1
            = std::make_shared<dnnl_impl::dnnl_constant_buffer_t>(
                    1, p_engine_, g_alloc_);
    c_promise1.set_value(c_buffer1);

    // should cache hit
    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise1_2;
    ASSERT_TRUE(cache.get_or_add(0, 1, 1, c_promise1_2.get_future()).valid());

    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise2;
    ASSERT_NO_THROW(cache.get_or_add(0, 2, 2, c_promise2.get_future()));
    graph::constant_tensor_cache_t::cached_t c_buffer2
            = std::make_shared<dnnl_impl::dnnl_constant_buffer_t>(
                    2, p_engine_, g_alloc_);
    c_promise2.set_value(c_buffer2);
    ASSERT_EQ(cache.get_size(), 3U); // c_buffer1 + c_buffer2

    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise3;
    // remained capacity is 2 MB, not enough to cache c_buffer3, it will be
    // ignore since we use no_evict policy
    ASSERT_NO_THROW(cache.get_or_add(0, 3, 3, c_promise3.get_future()));
    graph::constant_tensor_cache_t::cached_t c_buffer3
            = std::make_shared<dnnl_impl::dnnl_constant_buffer_t>(
                    3, p_engine_, g_alloc_);
    c_promise3.set_value(c_buffer3);
    ASSERT_EQ(cache.get_size(), 3U); // c_buffer1 + c_buffer2

    // should cache hit
    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise1_3;
    ASSERT_TRUE(cache.get_or_add(0, 1, 1, c_promise1_3.get_future()).valid());

    // should cache miss
    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise3_2;
    // remained capacity is 2 MB, not enough to cache c_buffer3, it will be
    // ignore since we use no_evict policy
    ASSERT_FALSE(cache.get_or_add(0, 3, 3, c_promise3_2.get_future()).valid());
}

TEST(ConstantCache, MultiDevices) {
    graph::engine_t &engine = *get_engine();
    auto alloc = static_cast<graph::allocator_t *>(engine.get_allocator());

    // each device should have dedicated cache
    dnnl::impl::engine_kind_t eng_kind = engine.kind();
    auto cache0 = graph::get_constant_tensor_cache(eng_kind, 0);
    auto cache1 = graph::get_constant_tensor_cache(eng_kind, 1);
    ASSERT_NE(cache0, cache1);

    // Each cache can set capacity separately
    ASSERT_EQ(cache0->set_capacity(5), graph::status::success);
    ASSERT_EQ(cache1->set_capacity(10), graph::status::success);
    ASSERT_EQ(cache0->get_capacity(), 5U);
    ASSERT_EQ(cache1->get_capacity(), 10U);

    auto p_engine = dnnl_impl::make_dnnl_engine(engine);

    // test cache0
    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise0;
    ASSERT_NO_THROW(cache0->get_or_add(0, 1, 1, c_promise0.get_future()));
    graph::constant_tensor_cache_t::cached_t c_buffer0
            = std::make_shared<dnnl_impl::dnnl_constant_buffer_t>(
                    5, p_engine, alloc);
    c_promise0.set_value(c_buffer0);

    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise0_query;
    ASSERT_TRUE(
            cache0->get_or_add(0, 1, 1, c_promise0_query.get_future()).valid());

    // test cache1
    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise1;
    ASSERT_NO_THROW(cache1->get_or_add(0, 1, 1, c_promise1.get_future()));
    graph::constant_tensor_cache_t::cached_t c_buffer1
            = std::make_shared<dnnl_impl::dnnl_constant_buffer_t>(
                    5, p_engine, alloc);
    c_promise1.set_value(c_buffer1);

    std::promise<graph::constant_tensor_cache_t::cached_t> c_promise1_query;
    ASSERT_TRUE(
            cache1->get_or_add(0, 1, 1, c_promise1_query.get_future()).valid());
}
