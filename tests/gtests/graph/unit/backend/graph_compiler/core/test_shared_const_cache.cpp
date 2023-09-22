/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include <runtime/const_cache_wrapper.hpp>
#include <runtime/context.hpp>
#include <util/utils.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc::runtime;

struct gc_simulated_const_cache_item {
    int buffer_[32];
    std::shared_ptr<const_cache_proxy> proxy_;
};

TEST(GCCore_CPU_shared_const_cache, TestFunctional) {
    auto item = std::make_shared<gc_simulated_const_cache_item>();
    item->proxy_ = std::make_shared<const_cache_proxy>(
            item, item->buffer_, sizeof(int) * 32, false);

    auto proxy = item->proxy_;
    int32_t inited = 1;
    auto retbuf = sc_acquire_const_cache(
            get_default_stream(), proxy.get(), 32 * sizeof(int), &inited);
    ASSERT_EQ(inited, 0);
    ASSERT_EQ(retbuf, item->buffer_);
    sc_release_const_cache(get_default_stream(), proxy.get(), retbuf);

    // second run of the kernel
    inited = 1;
    retbuf = sc_acquire_const_cache(
            get_default_stream(), proxy.get(), 32 * sizeof(int), &inited);
    ASSERT_EQ(inited, 1);
    ASSERT_EQ(retbuf, item->buffer_);
    sc_release_const_cache(get_default_stream(), proxy.get(), retbuf);

    // third run of the kernel
    inited = 1;
    retbuf = sc_acquire_const_cache(
            get_default_stream(), proxy.get(), 32 * sizeof(int), &inited);
    // simulate the eviction of the cache while the kernel is running
    proxy->deref();
    ASSERT_TRUE(proxy->is_alive());
    ASSERT_EQ(inited, 1);
    ASSERT_EQ(retbuf, item->buffer_);
    sc_release_const_cache(get_default_stream(), proxy.get(), retbuf);

    ASSERT_FALSE(proxy->is_alive());
    // fourth run of the kernel, the buffer is dead
    inited = 1;
    retbuf = sc_acquire_const_cache(
            get_default_stream(), proxy.get(), 32 * sizeof(int), &inited);
    ASSERT_EQ(inited, 0);
    ASSERT_NE(retbuf, item->buffer_);
    sc_release_const_cache(get_default_stream(), proxy.get(), retbuf);

    proxy->is_lazy_ = false;
}
