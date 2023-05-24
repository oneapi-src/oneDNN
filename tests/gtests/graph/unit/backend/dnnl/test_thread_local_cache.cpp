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
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/thread_local_cache.hpp"

#include <dnnl.hpp>

template <typename T>
using thread_local_cache_t
        = dnnl::impl::graph::dnnl_impl::thread_local_cache_t<T>;

struct test_resource_t {
    test_resource_t(size_t data) : data_(data) {}
    size_t data_;
};

TEST(ThreadLocalCache, SingleThread) {
    thread_local_cache_t<test_resource_t> cache;
    cache.clear();

    size_t key1 = 1U;
    test_resource_t *resource_ptr1 = cache.get_or_add(
            key1, []() { return std::make_shared<test_resource_t>(10); });

    size_t key2 = 2U;
    test_resource_t *resource_ptr2 = cache.get_or_add(
            key2, []() { return std::make_shared<test_resource_t>(20); });

    ASSERT_TRUE(cache.has_resource(key1));
    ASSERT_TRUE(cache.has_resource(key2));
    ASSERT_EQ(resource_ptr1->data_, 10U);
    ASSERT_EQ(resource_ptr2->data_, 20U);

    // the given creator will not take effect since the key1 is already in the
    // mapper
    resource_ptr1 = cache.get_or_add(
            key1, []() { return std::make_shared<test_resource_t>(100); });
    ASSERT_EQ(resource_ptr1->data_, 10U);
    ASSERT_EQ(cache.size(), 2U);

    cache.remove_if_exist(key1);
    cache.remove_if_exist(key2);
}

TEST(ThreadLocalCache, Multithreading) {
    auto func = []() {
        thread_local_cache_t<test_resource_t> cache;
        cache.clear();

        ASSERT_EQ(cache.size(), 0U);

        size_t key1 = 1U;
        test_resource_t *resource_ptr1 = cache.get_or_add(
                key1, []() { return std::make_shared<test_resource_t>(10); });

        size_t key2 = 2U;
        test_resource_t *resource_ptr2 = cache.get_or_add(
                key2, []() { return std::make_shared<test_resource_t>(20); });

        ASSERT_TRUE(cache.has_resource(key1));
        ASSERT_TRUE(cache.has_resource(key2));
        ASSERT_EQ(resource_ptr1->data_, 10U);
        ASSERT_EQ(resource_ptr2->data_, 20U);

        resource_ptr1->data_ = 30;
        ASSERT_EQ(resource_ptr1->data_, 30U);

        size_t key3 = 3U;
        cache.get_or_add(
                key3, []() { return std::make_shared<test_resource_t>(100); });

        ASSERT_EQ(cache.size(), 3U);
    };

    std::thread t1(func);
    std::thread t2(func);
    std::thread t3(func);
    t1.join();
    t2.join();
    t3.join();
}

TEST(ThreadLocalCache, Clear) {
    thread_local_cache_t<test_resource_t> cache;
    size_t key1 = (size_t)1;
    cache.get_or_add(
            key1, []() { return std::make_shared<test_resource_t>(10); });
    size_t key2 = (size_t)2;
    cache.get_or_add(
            key2, []() { return std::make_shared<test_resource_t>(20); });
    ASSERT_NO_THROW(cache.clear());
}

TEST(ThreadLocalCache, RetainAndRelease) {
    auto func = []() {
        thread_local_cache_t<test_resource_t> cache;
        cache.retain();
        cache.clear();

        ASSERT_EQ(cache.size(), 0U);

        size_t key1 = 1U;
        test_resource_t *resource_ptr1 = cache.get_or_add(
                key1, []() { return std::make_shared<test_resource_t>(10); });

        ASSERT_TRUE(cache.has_resource(key1));
        ASSERT_EQ(resource_ptr1->data_, 10U);
        ASSERT_EQ(cache.size(), 1U);
        cache.release();
    };

    std::thread t1(func);
    func();
    t1.join();
}
