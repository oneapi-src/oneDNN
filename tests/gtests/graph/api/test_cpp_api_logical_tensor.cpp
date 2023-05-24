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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

#include <gtest/gtest.h>

#include <vector>

TEST(APILogicalTensor, SimpleCreate) {
    using logical_tensor = dnnl::graph::logical_tensor;
    const size_t id = 123;
    logical_tensor lt {id, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    ASSERT_EQ(lt.get_id(), id);
}

TEST(APILogicalTensor, CreateWithShape) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;

    // 0D
    logical_tensor lt_0 {
            id, data_type::f32, logical_tensor::dims {}, layout_type::strided};
    ASSERT_EQ(lt_0.get_id(), id);
    ASSERT_EQ(lt_0.get_dims().size(), 0U);

    // 1D
    logical_tensor lt_1 {
            id, data_type::f32, logical_tensor::dims {3}, layout_type::strided};
    ASSERT_EQ(lt_1.get_id(), id);
    ASSERT_EQ(lt_1.get_dims().size(), 1U);

    // 2D
    logical_tensor lt_2 {id, data_type::f32, {3, 4}, layout_type::strided};
    ASSERT_EQ(lt_2.get_id(), id);
    ASSERT_EQ(lt_2.get_dims().size(), 2U);

    // 3D
    logical_tensor lt_3 {id, data_type::f32, {3, 4, 5}, layout_type::strided};
    ASSERT_EQ(lt_3.get_id(), id);
    ASSERT_EQ(lt_3.get_dims().size(), 3U);

    // 4D
    logical_tensor lt_4 {
            id, data_type::f32, {3, 4, 5, 6}, layout_type::strided};
    ASSERT_EQ(lt_4.get_id(), id);
    ASSERT_EQ(lt_4.get_dims().size(), 4U);
}

TEST(APILogicalTensor, CreateWithStrides) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    const size_t id = 123;

    // 0D
    logical_tensor lt_0 {id, data_type::f32, logical_tensor::dims {},
            logical_tensor::dims {}};
    ASSERT_EQ(lt_0.get_id(), id);
    ASSERT_EQ(lt_0.get_dims().size(), 0U);
    ASSERT_EQ(lt_0.get_strides().size(), 0U);

    // 1D
    logical_tensor lt_1 {id, data_type::f32, {3}, logical_tensor::dims {1}};
    ASSERT_EQ(lt_1.get_id(), id);
    ASSERT_EQ(lt_1.get_dims().size(), 1U);
    ASSERT_EQ(lt_1.get_dims()[0], 3);
    ASSERT_EQ(lt_1.get_strides()[0], 1);

    // 2D
    logical_tensor lt_2 {id, data_type::f32, {3, 4}, {4, 1}};
    ASSERT_EQ(lt_2.get_id(), id);
    ASSERT_EQ(lt_2.get_dims().size(), 2U);
    ASSERT_EQ(lt_2.get_dims()[0], 3);
    ASSERT_EQ(lt_2.get_dims()[1], 4);
    ASSERT_EQ(lt_2.get_strides()[0], 4);
    ASSERT_EQ(lt_2.get_strides()[1], 1);

    // 3D
    logical_tensor lt_3 {id, data_type::f32, {3, 4, 5}, {20, 5, 1}};
    ASSERT_EQ(lt_3.get_id(), id);
    ASSERT_EQ(lt_3.get_dims().size(), 3U);
    ASSERT_EQ(lt_3.get_dims()[0], 3);
    ASSERT_EQ(lt_3.get_dims()[1], 4);
    ASSERT_EQ(lt_3.get_dims()[2], 5);
    ASSERT_EQ(lt_3.get_strides()[0], 20);
    ASSERT_EQ(lt_3.get_strides()[1], 5);
    ASSERT_EQ(lt_3.get_strides()[2], 1);

    // 4D
    logical_tensor lt_4 {id, data_type::f32, {3, 4, 5, 6}, {120, 30, 6, 1}};
    ASSERT_EQ(lt_4.get_id(), id);
    ASSERT_EQ(lt_4.get_dims().size(), 4U);
    ASSERT_EQ(lt_4.get_dims()[0], 3);
    ASSERT_EQ(lt_4.get_dims()[1], 4);
    ASSERT_EQ(lt_4.get_dims()[2], 5);
    ASSERT_EQ(lt_4.get_dims()[3], 6);
    ASSERT_EQ(lt_4.get_strides()[0], 120);
    ASSERT_EQ(lt_4.get_strides()[1], 30);
    ASSERT_EQ(lt_4.get_strides()[2], 6);
    ASSERT_EQ(lt_4.get_strides()[3], 1);
}

TEST(APILogicalTensor, CreateWithDataType) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;

    // f32
    logical_tensor lt_f32 {id, data_type::f32, {3, 4}, layout_type::strided};
    ASSERT_EQ(lt_f32.get_id(), id);
    ASSERT_EQ(lt_f32.get_data_type(), data_type::f32);

    // f16
    logical_tensor lt_f16 {id, data_type::f16, {3, 4}, layout_type::strided};
    ASSERT_EQ(lt_f16.get_id(), id);
    ASSERT_EQ(lt_f16.get_data_type(), data_type::f16);

    // s8
    logical_tensor lt_s8 {id, data_type::s8, {3, 4}, layout_type::strided};
    ASSERT_EQ(lt_s8.get_id(), id);
    ASSERT_EQ(lt_s8.get_data_type(), data_type::s8);

    // bool
    logical_tensor lt_boolean {
            id, data_type::boolean, {3, 4}, layout_type::strided};
    ASSERT_EQ(lt_boolean.get_id(), id);
    ASSERT_EQ(lt_boolean.get_data_type(), data_type::boolean);
}

TEST(APILogicalTensor, ShallowCopy) {
    using logical_tensor = dnnl::graph::logical_tensor;
    const size_t id = 123;
    logical_tensor lt_1 {id, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    logical_tensor lt_2(lt_1);

    ASSERT_EQ(lt_1.get_id(), lt_2.get_id());
}

TEST(APILogicalTensor, DifferentLogicalTensorWithSameID) {
    using namespace dnnl::graph;

    graph g(dnnl::engine::kind::cpu);
    // x1 and x2 are two different tensors (with different layout_type)
    // but with same id
    logical_tensor x1 {0, logical_tensor::data_type::f32, {},
            logical_tensor::layout_type::undef};
    logical_tensor x2 {0, logical_tensor::data_type::f32, {},
            logical_tensor::layout_type::strided};
    logical_tensor y {1, logical_tensor::data_type::f32, {},
            logical_tensor::layout_type::undef};
    op add(0, op::kind::Add, "add");
    add.add_inputs({x1, x2});
    add.add_output(y);
    g.add_op(add);
    EXPECT_THROW(g.finalize(), dnnl::error);
}

TEST(APILogicalTensor, CompareLayoutAndDataType) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    logical_tensor lt0 {0, data_type::f32, logical_tensor::dims {1, 2, 3},
            layout_type::strided};
    logical_tensor lt1 {0, data_type::f32, logical_tensor::dims {1, 2, 3},
            layout_type::strided};
    ASSERT_EQ(lt0.is_equal(lt1), true);

    logical_tensor lt4 {1, data_type::f32, logical_tensor::dims {1, 2, 3},
            layout_type::strided};
    logical_tensor lt5 {1, data_type::bf16, logical_tensor::dims {1, 2, 3},
            layout_type::strided};
    ASSERT_EQ(lt4.is_equal(lt5), false);

    logical_tensor lt6 {2, data_type::f32, logical_tensor::dims {1, 2, 3},
            layout_type::strided};
    logical_tensor lt7 {2, data_type::f32, logical_tensor::dims {3, 2, 1},
            layout_type::strided};
    ASSERT_EQ(lt6.is_equal(lt7), false);
}

TEST(APILogicalTensor, TestProperty) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    using property_type = logical_tensor::property_type;

    logical_tensor lt0 {0, data_type::f32, {1, 2, 3}, layout_type::strided};
    ASSERT_EQ(lt0.get_property_type(), property_type::undef);

    logical_tensor lt1 {1, data_type::f32, {1, 2, 3}, layout_type::strided,
            property_type::constant};
    ASSERT_EQ(lt1.get_property_type(), property_type::constant);
}

TEST(APILogicalTensor, CreateWith0Dims) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;
    ASSERT_NO_THROW(
            { logical_tensor(id, data_type::f32, {}, layout_type::strided); });
}

TEST(APILogicalTensor, GetDimsWithError) {
    using logical_tensor = dnnl::graph::logical_tensor;
    dnnl_graph_logical_tensor_t c_lt;
    memset((char *)&c_lt, 0, sizeof(c_lt));

    c_lt.id = 0;
    c_lt.ndims = -1;
    logical_tensor lt(c_lt);
    ASSERT_THROW(lt.get_dims(), dnnl::error);
}

TEST(APILogicalTensor, GetLayoutIdWithError) {
    using logical_tensor = dnnl::graph::logical_tensor;
    dnnl_graph_logical_tensor_t c_lt;
    memset((char *)&c_lt, 0, sizeof(c_lt));

    c_lt.id = 0;
    c_lt.ndims = -1;
    c_lt.data_type = dnnl_f16;
    c_lt.property = dnnl_graph_tensor_property_undef;
    c_lt.layout_type = dnnl_graph_layout_type_strided;
    c_lt.layout.layout_id = 1;
    logical_tensor lt(c_lt);
    ASSERT_THROW(lt.get_layout_id(), dnnl::error);
}

TEST(APILogicalTensor, GetStridesWithError) {
    using logical_tensor = dnnl::graph::logical_tensor;
    {
        dnnl_graph_logical_tensor_t c_lt;
        memset((char *)&c_lt, 0, sizeof(c_lt));

        c_lt.id = 0;
        c_lt.ndims = -1;
        c_lt.data_type = dnnl_f16;
        c_lt.property = dnnl_graph_tensor_property_undef;
        c_lt.layout_type = dnnl_graph_layout_type_opaque;
        c_lt.layout.layout_id = 1;

        logical_tensor lt(c_lt);
        ASSERT_THROW(lt.get_strides(), dnnl::error);
    }
    {
        dnnl_graph_logical_tensor_t c_lt;
        memset((char *)&c_lt, 0, sizeof(c_lt));

        c_lt.id = 0;
        c_lt.ndims = -1;
        c_lt.data_type = dnnl_f16;
        c_lt.property = dnnl_graph_tensor_property_undef;
        c_lt.layout_type = dnnl_graph_layout_type_strided;
        c_lt.layout.layout_id = 1;
        logical_tensor lt(c_lt);
        ASSERT_THROW(lt.get_strides(), dnnl::error);
    }
}

TEST(APILogicalTensor, LogicalTensorSize) {
    using logical_tensor = dnnl::graph::logical_tensor;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    const size_t id = 123;
    const logical_tensor::dims shape = {3, 4};
    const size_t num_elem = 3 * 4;

    logical_tensor lt_1 {id, data_type::boolean, shape, layout_type::strided};
    ASSERT_EQ(lt_1.get_id(), id);
    ASSERT_EQ(lt_1.get_data_type(), data_type::boolean);
    ASSERT_EQ(lt_1.get_mem_size(), num_elem * sizeof(bool));

    logical_tensor lt_2 {id, data_type::f32, shape, layout_type::strided};
    ASSERT_EQ(lt_2.get_id(), id);
    ASSERT_EQ(lt_2.get_data_type(), data_type::f32);
    ASSERT_EQ(lt_2.get_mem_size(), num_elem * sizeof(float));

    logical_tensor lt_3 {id, data_type::s8, shape, layout_type::strided};
    ASSERT_EQ(lt_3.get_id(), id);
    ASSERT_EQ(lt_3.get_data_type(), data_type::s8);
    ASSERT_EQ(lt_3.get_mem_size(), num_elem * sizeof(int8_t));
}
