/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include <string>
#include <gtest/gtest.h>

#include "utils/any.hpp"
#include "utils/utils.hpp"

TEST(utils_test, any) {
    using namespace dnnl::impl::graph::utils;
    any_t a = 1;
    ASSERT_EQ(any_cast<int>(a), 1);
    int *i = any_cast<int>(&a);
    ASSERT_NE(i, nullptr);
    ASSERT_EQ(*i, 1);
    a = 3.14;
    ASSERT_EQ(any_cast<double>(a), 3.14);
    a = true;
    ASSERT_EQ(any_cast<bool>(a), true);
    any_t b;
    ASSERT_EQ(b.empty(), true);
    EXPECT_THROW(any_cast<int>(b), bad_any_cast_t);
}

TEST(BadAnyCast, What) {
    using namespace dnnl::impl::graph::utils;

    bad_any_cast_t bad_any;
    ASSERT_EQ(std::string(bad_any.what()), std::string("bad any_cast"));
}

TEST(utils_test, iffy_getenv) {
    char buffer[10] = {'\0'};
    EXPECT_EQ(INT_MIN, dnnl::impl::getenv(nullptr, buffer, 10));
    EXPECT_EQ(INT_MIN, dnnl::impl::getenv("foo", nullptr, 10));
    EXPECT_EQ(INT_MIN, dnnl::impl::getenv("foo", buffer, -1));
}
