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

#include <gtest/gtest.h>

#include "interface/attribute_value.hpp"

#include <vector>

TEST(attribute_value_test, int64_value) {
    using namespace dnnl::graph::impl;

    const int64_t i = 1234;
    attribute_value v1 {i};
    ASSERT_EQ(v1.get_kind(), attribute_kind::i);
    ASSERT_EQ(v1.get<int64_t>(), i);
}

TEST(attribute_value_test, int64_vector_value) {
    using namespace dnnl::graph::impl;

    const std::vector<int64_t> is {1234, 5678};
    attribute_value v1 {is};
    ASSERT_EQ(v1.get_kind(), attribute_kind::is);
    ASSERT_EQ(v1.get<std::vector<int64_t>>(), is);
}

TEST(attribute_value_test, float_value) {
    using namespace dnnl::graph::impl;

    const float f = 0.5;
    attribute_value v1 {f};
    ASSERT_EQ(v1.get_kind(), attribute_kind::f);
    ASSERT_EQ(v1.get<float>(), f);
}

TEST(attribute_value_test, float_vector_value) {
    using namespace dnnl::graph::impl;

    const std::vector<float> fs {0.5, 0.25};
    attribute_value v1 {fs};
    ASSERT_EQ(v1.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v1.get<std::vector<float>>(), fs);
}

TEST(attribute_value_test, bool_value) {
    using namespace dnnl::graph::impl;

    const bool b = true;
    attribute_value v1 {b};
    ASSERT_EQ(v1.get_kind(), attribute_kind::b);
    ASSERT_EQ(v1.get<bool>(), b);
}

TEST(attribute_value_test, string_value) {
    using namespace dnnl::graph::impl;

    const std::string s = "string attribute";
    attribute_value v1 {s};
    ASSERT_EQ(v1.get_kind(), attribute_kind::s);
    ASSERT_EQ(v1.get<std::string>(), s);
}

TEST(attribute_value_test, copy) {
    using namespace dnnl::graph::impl;

    const std::vector<float> fs {0.5, 0.25};
    attribute_value v1 {fs};
    ASSERT_EQ(v1.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v1.get<std::vector<float>>(), fs);

    attribute_value v2 = v1;
    ASSERT_EQ(v1.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v1.get<std::vector<float>>(), fs);
}

TEST(attribute_value_test, equal) {
    using namespace dnnl::graph::impl;

    const std::vector<float> fs1 {0.5, 0.25};
    attribute_value v1 {fs1};
    attribute_value v2 {fs1};
    ASSERT_EQ(v1, v2);

    const std::vector<float> fs2 {0.5, 0.5};
    attribute_value v3 {fs2};
    ASSERT_NE(v1, v3);
}
