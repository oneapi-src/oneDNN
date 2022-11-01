/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "backend/dnnl/constant_cache.hpp"

#include "interface/value.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "utils/utils.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;

TEST(ConstantCache, SetGetCapacity) {
    impl::dnnl_impl::constant_cache_t cache;
    ASSERT_EQ(cache.set_capacity(11), impl::status::success);
    ASSERT_EQ(cache.get_capacity(), 11U);
}

TEST(ConstantCache, GetOrAddEmpty) {
    using key_t = impl::dnnl_impl::constant_cache_t::key_t;
    using value_t = impl::dnnl_impl::constant_cache_t::value_t;

    impl::dnnl_impl::constant_cache_t cache;
    ASSERT_EQ(cache.set_capacity(0), impl::status::success);
    ASSERT_FALSE(cache.get_or_add(key_t(), value_t()).valid());
}

TEST(ConstantCache, Evict) {
    impl::engine_t &engine = get_engine();
    auto p_engine_ = dnnl_impl::make_dnnl_engine(engine);
    auto g_alloc_ = engine.get_allocator();

    impl::dnnl_impl::constant_cache_t cache;
    ASSERT_EQ(cache.set_capacity(0), impl::status::success);
    ASSERT_EQ(cache.set_capacity(5), impl::status::success);

    std::promise<dnnl_impl::constant_cache_t::cached_t> c_promise1;
    dnnl_impl::constant_cache_t::cached_t c_buffer1
            = std::make_shared<dnnl_impl::constant_buffer_t>(
                    1, p_engine_, g_alloc_);
    c_promise1.set_value(c_buffer1);
    ASSERT_NO_THROW(cache.get_or_add(1, c_promise1.get_future()));

    std::promise<dnnl_impl::constant_cache_t::cached_t> c_promise2;
    dnnl_impl::constant_cache_t::cached_t c_buffer2
            = std::make_shared<dnnl_impl::constant_buffer_t>(
                    2, p_engine_, g_alloc_);
    c_promise2.set_value(c_buffer2);
    ASSERT_NO_THROW(cache.get_or_add(2, c_promise2.get_future()));

    std::promise<dnnl_impl::constant_cache_t::cached_t> c_promise3;
    dnnl_impl::constant_cache_t::cached_t c_buffer3
            = std::make_shared<dnnl_impl::constant_buffer_t>(
                    3, p_engine_, g_alloc_);
    c_promise3.set_value(c_buffer3);
    ASSERT_NO_THROW(cache.get_or_add(3, c_promise3.get_future()));

    ASSERT_EQ(cache.set_capacity(3), impl::status::success);
    ASSERT_EQ(cache.set_capacity(0), impl::status::success);
}
