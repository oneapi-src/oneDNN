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

#include "gtest/gtest.h"

#include "backend/dnnl/constant_cache.hpp"

#include "interface/value.hpp"

#include "utils/utils.hpp"

namespace impl = dnnl::graph::impl;

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
