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

#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/resource.hpp"

#include <dnnl.hpp>

using dnnl::graph::impl::dnnl_impl::resource_cache_t;
using dnnl::graph::impl::dnnl_impl::resource_t;

struct test_resource_t : public resource_t {
    test_resource_t(size_t data) : data_(data) {}
    size_t data_;
};

TEST(resource, resource_cache) {
    resource_cache_t resource_cache;
    resource_cache.clear();

    resource_cache_t::key_t key1 = (resource_cache_t::key_t)1;
    std::unique_ptr<resource_t> resource1(new test_resource_t(10));
    resource_cache.add<test_resource_t>(key1, std::move(resource1));

    resource_cache_t::key_t key2 = (resource_cache_t::key_t)2;
    std::unique_ptr<resource_t> resource2(new test_resource_t(20));
    resource_cache.add<test_resource_t>(key2, std::move(resource2));

    ASSERT_TRUE(resource_cache.has_resource(key1));
    ASSERT_TRUE(resource_cache.has_resource(key2));

    test_resource_t *resource_ptr1 = resource_cache.get<test_resource_t>(key1);
    test_resource_t *resource_ptr2 = resource_cache.get<test_resource_t>(key2);

    ASSERT_EQ(resource_ptr1->data_, 10);
    ASSERT_EQ(resource_ptr2->data_, 20);

    resource_ptr1->data_ = 30;
    resource_ptr1 = resource_cache.get<test_resource_t>(key1);
    ASSERT_EQ(resource_ptr1->data_, 30);

    // the given creator will not take effect since the key1 is already in the
    // cache
    resource_ptr1 = resource_cache.get<test_resource_t>(key1, []() {
        return std::unique_ptr<resource_t>(new test_resource_t(100));
    });
    ASSERT_EQ(resource_ptr1->data_, 30);
    ASSERT_EQ(resource_cache.size(), 2);

    // the given creator will take effect since the key3 is not in the cache
    resource_cache_t::key_t key3 = (resource_cache_t::key_t)3;
    test_resource_t *resource_ptr3
            = resource_cache.get<test_resource_t>(key3, []() {
                  return std::unique_ptr<resource_t>(new test_resource_t(100));
              });
    ASSERT_EQ(resource_ptr3->data_, 100);
    ASSERT_EQ(resource_cache.size(), 3);
}

TEST(resource, resource_cache_multithreading) {
    auto func = []() {
        resource_cache_t resource_cache;
        resource_cache.clear();

        ASSERT_EQ(resource_cache.size(), 0);

        resource_cache_t::key_t key1 = (resource_cache_t::key_t)1;
        std::unique_ptr<resource_t> resource1(new test_resource_t(10));
        resource_cache.add<test_resource_t>(key1, std::move(resource1));

        resource_cache_t::key_t key2 = (resource_cache_t::key_t)2;
        std::unique_ptr<resource_t> resource2(new test_resource_t(20));
        resource_cache.add<test_resource_t>(key2, std::move(resource2));

        ASSERT_TRUE(resource_cache.has_resource(key1));
        ASSERT_TRUE(resource_cache.has_resource(key2));

        test_resource_t *resource_ptr1
                = resource_cache.get<test_resource_t>(key1);
        test_resource_t *resource_ptr2
                = resource_cache.get<test_resource_t>(key2);

        ASSERT_EQ(resource_ptr1->data_, 10);
        ASSERT_EQ(resource_ptr2->data_, 20);

        resource_ptr1->data_ = 30;
        resource_ptr1 = resource_cache.get<test_resource_t>(key1);
        ASSERT_EQ(resource_ptr1->data_, 30);

        resource_cache_t::key_t key3 = (resource_cache_t::key_t)3;
        test_resource_t *resource_ptr3
                = resource_cache.get<test_resource_t>(key3, []() {
                      return std::unique_ptr<resource_t>(
                              new test_resource_t(100));
                  });

        ASSERT_EQ(resource_cache.size(), 3);
    };

    std::thread t1(func);
    std::thread t2(func);
    t1.join();
    t2.join();
}
