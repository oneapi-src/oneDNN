/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

#include <gtest/gtest.h>

#include <vector>

TEST(api_tensor, create_with_shape) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    using tensor = dnnl::graph::tensor;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;

    // 0D
    logical_tensor lt_0 {id, data_type::f32, logical_tensor::dims_t {},
            layout_type::strided};
    int n0 = 0;
    void *handle0 = &n0;
    tensor t_0 {lt_0, handle0};
    ASSERT_EQ(t_0.get_data_handle<float>(), handle0);
    ASSERT_EQ(t_0.get_element_num(), 0);

    // 1D
    logical_tensor lt_1 {id, data_type::f32, logical_tensor::dims_t {3},
            layout_type::strided};
    std::vector<float> n1 {0, 1, 2};
    void *handle1 = n1.data();
    tensor t_1 {lt_1, handle1};
    ASSERT_EQ(t_1.get_data_handle<float>(), handle1);
    ASSERT_EQ(t_1.get_element_num(), 3);

    // 2D
    logical_tensor lt_2 {id, data_type::f32, logical_tensor::dims_t {3, 4},
            layout_type::strided};
    std::vector<float> n2;
    n2.resize(3 * 4);
    void *handle2 = n2.data();
    tensor t_2 {lt_2, handle2};
    ASSERT_EQ(t_2.get_data_handle<float>(), handle2);
    ASSERT_EQ(t_2.get_element_num(), 3 * 4);

    // 3D
    logical_tensor lt_3 {id, data_type::f32, logical_tensor::dims_t {3, 4, 5},
            layout_type::strided};
    std::vector<float> n3;
    n3.resize(3 * 4 * 5);
    void *handle3 = n3.data();
    tensor t_3 {lt_3, handle3};
    ASSERT_EQ(t_3.get_data_handle<float>(), handle3);
    ASSERT_EQ(t_3.get_element_num(), 3 * 4 * 5);

    // 4D
    logical_tensor lt_4 {id, data_type::f32,
            logical_tensor::dims_t {3, 4, 5, 6}, layout_type::strided};
    std::vector<float> n4;
    n4.resize(3 * 4 * 5 * 6);
    void *handle4 = n4.data();
    tensor t_4 {lt_4, handle4};
    ASSERT_EQ(t_4.get_data_handle<float>(), handle4);
    ASSERT_EQ(t_4.get_element_num(), 3 * 4 * 5 * 6);

    std::vector<float> n5 {0};
    void *handle5 = n5.data();
    tensor t_5 {lt_0, nullptr};
    t_5.set_data_handle(handle5);
    ASSERT_EQ(t_5.get_data_handle<float>(), handle5);
}

TEST(api_tensor, shallow_copy) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using tensor = dnnl::graph::tensor;
    using layout_type = logical_tensor::layout_type;

    const size_t id = 123;
    logical_tensor lt_1 {
            id, logical_tensor::data_type::f32, layout_type::strided};

    int n = 0;
    void *handle = &n;

    tensor t_1 {lt_1, handle};
    tensor t_2(t_1);

    ASSERT_EQ(t_2.get_data_handle<float>(), handle);
    ASSERT_EQ(t_1.get_element_num(), t_2.get_element_num());
}

template <typename T>
class tensor_test : public ::testing::Test {};

// TODO(Rong): int64_t, uint64_t, double>;
using tested_types = ::testing::Types<int8_t, float>;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
TYPED_TEST_SUITE(tensor_test, tested_types);
#ifdef __clang__
#pragma clang diagnostic pop
#endif

TYPED_TEST(tensor_test, create_with_logical_tensor) {
    using namespace dnnl::graph;
    logical_tensor lt {0, logical_tensor::data_type::f32,
            logical_tensor::layout_type::any};
    tensor t {lt, nullptr};

    ASSERT_EQ(t.get_element_num(), -1);
}
