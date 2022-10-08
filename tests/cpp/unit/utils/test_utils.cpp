/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#include <sstream>
#include <string>
#include <thread>

#include "gtest/gtest.h"

#include "utils/compatible.hpp"
#include "utils/utils.hpp"

#ifndef DNNL_GRAPH_SUPPORT_CXX17
TEST(Utils, optional) {
    //init, copy constructor
    dnnl::graph::impl::utils::optional_impl_t<int64_t> o1, o2 = 2, o3 = o2;
    ASSERT_EQ(*o2, 2);
    ASSERT_EQ(*o3, 2);
    dnnl::graph::impl::utils::optional_impl_t<float> o4(
            dnnl::graph::impl::utils::nullopt),
            o5 = dnnl::graph::impl::utils::nullopt;
    ASSERT_EQ(o4.has_value(), false);
    ASSERT_EQ(o4, o5);
    EXPECT_THROW(o4.value(), std::logic_error);
    EXPECT_THROW(o1.value(), std::logic_error);
}
#endif

TEST(Utils, any) {
    dnnl::graph::impl::utils::any_t a = 1;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<int>(a), 1);
    int *i = dnnl::graph::impl::utils::any_cast<int>(&a);
    ASSERT_NE(i, nullptr);
    ASSERT_EQ(*i, 1);
    a = 3.14;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<double>(a), 3.14);
    a = true;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<bool>(a), true);
    dnnl::graph::impl::utils::any_t b;
    ASSERT_EQ(b.empty(), true);
    EXPECT_THROW(dnnl::graph::impl::utils::any_cast<int>(b),
            dnnl::graph::impl::utils::bad_any_cast_t);
}

TEST(Utils, sizeoftype) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::impl::utils;
    EXPECT_EQ(sizeof(float), size_of(data_type::f32));
    EXPECT_EQ(sizeof(int32_t), size_of(data_type::s32));
    EXPECT_EQ(sizeof(uint8_t), size_of(data_type::u8));
    EXPECT_EQ(sizeof(int8_t), size_of(data_type::s8));
    EXPECT_EQ(sizeof(int16_t), size_of(data_type::f16));
    EXPECT_EQ(sizeof(int16_t), size_of(data_type::bf16));
    EXPECT_EQ(0U, size_of(static_cast<data_type_t>(data_type::bf16 + 10)));
}

TEST(Utils, iffy_getenv) {
    namespace utils = dnnl::graph::impl::utils;
    char buffer[10] = {'\0'};
    EXPECT_EQ(INT_MIN, utils::getenv(nullptr, buffer, 10));
    EXPECT_EQ(INT_MIN, utils::getenv("foo", nullptr, 10));
    EXPECT_EQ(INT_MIN, utils::getenv("foo", buffer, -1));
}

TEST(Utils, Float2int) {
    namespace utils = dnnl::graph::impl::utils;
    float a = 3.0f;
    int *b = (int *)(&a);
    ASSERT_EQ(utils::float2int(a), *b);
    b = nullptr;
}

TEST(Utils, ThreadIdToStr) {
    namespace utils = dnnl::graph::impl::utils;
    std::stringstream ss;
    auto id = std::this_thread::get_id();
    ss << id;
    ASSERT_EQ(utils::thread_id_to_str(id), ss.str());
}
