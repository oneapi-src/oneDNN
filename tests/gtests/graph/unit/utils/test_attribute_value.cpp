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

#include <gtest/gtest.h>

#include "utils/attribute_value.hpp"

#include <vector>

TEST(AttributeValue, Int64) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const int64_t i = 1234;
    attribute_value_t v1 {i};
    ASSERT_EQ(v1.get_kind(), attribute_kind::i);
    ASSERT_EQ(v1.get<int64_t>(), i);
}

TEST(AttributeValue, Int64Vector) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const std::vector<int64_t> is {1234, 5678};
    attribute_value_t v1 {is};
    ASSERT_EQ(v1.get_kind(), attribute_kind::is);
    ASSERT_EQ(v1.get<std::vector<int64_t>>(), is);
}

TEST(AttributeValue, Float) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const float f = 0.5;
    attribute_value_t v1 {f};
    ASSERT_EQ(v1.get_kind(), attribute_kind::f);
    ASSERT_EQ(v1.get<float>(), f);
}

TEST(AttributeValue, FloatVector) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const std::vector<float> fs {0.5, 0.25};
    attribute_value_t v1 {fs};
    ASSERT_EQ(v1.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v1.get<std::vector<float>>(), fs);
}

TEST(AttributeValue, BoolValue) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const bool b = true;
    attribute_value_t v1 {b};
    ASSERT_EQ(v1.get_kind(), attribute_kind::b);
    ASSERT_EQ(v1.get<bool>(), b);
}

TEST(AttributeValue, StringValue) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const std::string s = "string attribute";
    attribute_value_t v1 {s};
    ASSERT_EQ(v1.get_kind(), attribute_kind::s);
    ASSERT_EQ(v1.get<std::string>(), s);
}

TEST(AttributeValue, Copy) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const std::vector<float> fs {0.5, 0.25};
    attribute_value_t v1 {fs};
    ASSERT_EQ(v1.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v1.get<std::vector<float>>(), fs);

    attribute_value_t v2 = v1; // NOLINT
    ASSERT_EQ(v2.get_kind(), attribute_kind::fs);
    ASSERT_EQ(v2.get<std::vector<float>>(), fs);
}

TEST(AttributeValue, Equal) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const std::vector<float> fs1 {0.5, 0.25};
    attribute_value_t v1 {fs1};
    attribute_value_t v2 {fs1};
    ASSERT_EQ(v1, v2);

    const std::vector<float> fs2 {0.5, 0.5};
    attribute_value_t v3 {fs2};
    ASSERT_NE(v1, v3);
}

TEST(AttributeValue, AttributeValueAssignOperator) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::utils;

    const attribute_value_t v1 {int64_t(3)};
    attribute_value_t v2 {int64_t(1)};
    v2 = v1;
    ASSERT_EQ(v1, v2);
}
