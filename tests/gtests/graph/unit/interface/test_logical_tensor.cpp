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

#include <vector>
#include <gtest/gtest.h>

#include "graph/unit/utils.hpp"
#include "interface/backend.hpp"
#include "interface/logical_tensor.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(LogicalTensor, CreateDefault) {
    const size_t id = 123;
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(id, graph::data_type::f32);

    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, graph::data_type::f32);
}

TEST(LogicalTensor, CreateWithShape) {
    const size_t id = 123;

    graph::logical_tensor_t lt_0
            = utils::logical_tensor_init(id, {}, graph::data_type::f32);
    ASSERT_EQ(lt_0.id, id);
    ASSERT_EQ(lt_0.ndims, 0);
    ASSERT_EQ(lt_0.data_type, graph::data_type::f32);

    graph::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3}, graph::data_type::f32);
    ASSERT_EQ(lt_1.id, id);
    ASSERT_EQ(lt_1.ndims, 1);
    ASSERT_EQ(lt_1.data_type, graph::data_type::f32);

    graph::logical_tensor_t lt_2
            = utils::logical_tensor_init(id, {3, 4}, graph::data_type::f32);
    ASSERT_EQ(lt_2.id, id);
    ASSERT_EQ(lt_2.ndims, 2);
    ASSERT_EQ(lt_2.data_type, graph::data_type::f32);

    graph::logical_tensor_t lt_3
            = utils::logical_tensor_init(id, {3, 4, 5}, graph::data_type::f32);
    ASSERT_EQ(lt_3.id, id);
    ASSERT_EQ(lt_3.ndims, 3);
    ASSERT_EQ(lt_3.data_type, graph::data_type::f32);

    graph::logical_tensor_t lt_4 = utils::logical_tensor_init(
            id, {3, 4, 5, 6}, graph::data_type::f32);
    ASSERT_EQ(lt_4.id, id);
    ASSERT_EQ(lt_4.ndims, 4);
    ASSERT_EQ(lt_4.data_type, graph::data_type::f32);

    graph::logical_tensor_t lt_5
            = utils::logical_tensor_init(id, {4, 5, 0}, graph::data_type::f32);
    ASSERT_EQ(lt_5.id, id);
    ASSERT_EQ(lt_5.ndims, 3);
    ASSERT_EQ(lt_5.data_type, graph::data_type::f32);
    ASSERT_EQ(lt_5.layout_type, graph::layout_type::strided);
    ASSERT_EQ(lt_5.layout.strides[0], 5);
    ASSERT_EQ(lt_5.layout.strides[1], 1);
    ASSERT_EQ(lt_5.layout.strides[2], 1);
}

TEST(LogicalTensor, Copy) {
    const size_t id = 123;

    graph::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_2(lt_1);

    ASSERT_EQ(lt_1.id, lt_2.id);
    ASSERT_EQ(lt_1.ndims, lt_2.ndims);
    ASSERT_EQ(lt_1.data_type, lt_2.data_type);
}

TEST(LogicalTensor, Assign) {
    const size_t id = 123;

    graph::logical_tensor_t lt_1
            = utils::logical_tensor_init(id, {3, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_2 = lt_1;

    ASSERT_EQ(lt_1.id, lt_2.id);
    ASSERT_EQ(lt_1.ndims, lt_2.ndims);
    ASSERT_EQ(lt_1.data_type, lt_2.data_type);
}

TEST(LogicalTensor, PushToVector) {
    size_t num_inputs = 3;
    std::vector<graph::dim_t> dims {1};
    std::vector<graph::logical_tensor_t> lt_vec;
    lt_vec.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        lt_vec.emplace_back(
                utils::logical_tensor_init(i, dims, graph::data_type::f32));
    }

    for (size_t i = 0; i < num_inputs; ++i) {
        ASSERT_EQ((size_t)lt_vec[i].ndims, dims.size());
    }
}

TEST(LogicalTensor, IdenticalSimilar) {
    using ltw = graph::logical_tensor_wrapper_t;

    // unknown dims and strides
    graph::logical_tensor_t lt1 = utils::logical_tensor_init(
            0, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t lt2 = utils::logical_tensor_init(
            0, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t lt3 = utils::logical_tensor_init(
            1, graph::data_type::f32, graph::layout_type::strided);
    ASSERT_EQ(ltw(lt1).is_identical(ltw(lt2)), true);
    ASSERT_EQ(ltw(lt1).is_identical(ltw(lt3)), false);

    // given dims and strides
    graph::logical_tensor_t lt4 = utils::logical_tensor_init(
            1, {1, 2, 3}, graph::data_type::f32, graph::layout_type::strided);
    // implicit strides
    graph::logical_tensor_t lt5 = utils::logical_tensor_init(
            1, {1, 2, 3}, graph::data_type::f32, graph::layout_type::strided);
    // explicit strides
    graph::logical_tensor_t lt6 = utils::logical_tensor_init(
            1, {1, 2, 3}, {6, 3, 1}, graph::data_type::f32);
    ASSERT_EQ(ltw(lt4).is_identical(ltw(lt5)), true);
    ASSERT_EQ(ltw(lt4).is_identical(ltw(lt6)), true);

    // same id + same shape/strides
    graph::logical_tensor_t lt7 = utils::logical_tensor_init(
            1, {1, 2, 3}, graph::data_type::f32, graph::layout_type::strided);
    // same id + different shape/strides
    graph::logical_tensor_t lt8 = utils::logical_tensor_init(
            1, {1, 2, 1}, graph::data_type::f32, graph::layout_type::strided);
    ASSERT_TRUE(ltw(lt4) == ltw(lt7));
    ASSERT_TRUE(ltw(lt4) != ltw(lt8));

    // different id + same shape/strides
    graph::logical_tensor_t lt9 = utils::logical_tensor_init(
            2, {1, 2, 3}, graph::data_type::f32, graph::layout_type::strided);
    // different id + different shape/strides
    graph::logical_tensor_t lt10 = utils::logical_tensor_init(
            2, {1, 2, 1}, graph::data_type::f32, graph::layout_type::strided);
    ASSERT_EQ(ltw(lt4).is_similar(ltw(lt9)), true);
    ASSERT_EQ(ltw(lt4).is_similar(ltw(lt10)), false);
}

TEST(LogicalTensor, GetWeightSpatialDims) {
    using ltw = graph::logical_tensor_wrapper_t;
    graph::logical_tensor_t lt = utils::logical_tensor_init(0, {1, 2, 2, 1},
            graph::data_type::f32, graph::layout_type::strided);

    auto wrap = ltw(lt);
    ASSERT_EQ(wrap.get_weight_spatial_dims("XXX").empty(), true);
    ASSERT_EQ(wrap.get_weight_spatial_dims("XIO")[0], 1);
    ASSERT_EQ(wrap.get_weight_spatial_dims("XIO")[1], 2);
    ASSERT_EQ(wrap.get_weight_spatial_dims("OIX")[0], 2);
    ASSERT_EQ(wrap.get_weight_spatial_dims("OIX")[1], 1);
}

TEST(LogicalTensor, GetSrcSpatialDims) {
    using ltw = graph::logical_tensor_wrapper_t;
    graph::logical_tensor_t lt = utils::logical_tensor_init(0, {1, 2, 2, 1},
            graph::data_type::f32, graph::layout_type::strided);

    auto wrap = ltw(lt);
    ASSERT_EQ(wrap.get_src_spatial_dims("XXX").size(), 0U);
    ASSERT_EQ(wrap.get_src_spatial_dims("NCX")[0], 2);
    ASSERT_EQ(wrap.get_src_spatial_dims("NCX")[1], 1);
    ASSERT_EQ(wrap.get_src_spatial_dims("NXC")[0], 2);
    ASSERT_EQ(wrap.get_src_spatial_dims("NXC")[1], 2);
}

TEST(LogicalTensor, GetWeightOrSrcIO) {
    using ltw = graph::logical_tensor_wrapper_t;
    graph::logical_tensor_t lt = utils::logical_tensor_init(0, {1, 2, 2, 1},
            graph::data_type::f32, graph::layout_type::strided);

    auto wrap = ltw(lt);
    ASSERT_EQ(wrap.get_weight_i("XXX"), DNNL_GRAPH_UNKNOWN_DIM);
    ASSERT_EQ(wrap.get_weight_i("OIX"), 2);
    ASSERT_EQ(wrap.get_weight_i("XIO"), 2);

    ASSERT_EQ(wrap.get_weight_o("XXX"), DNNL_GRAPH_UNKNOWN_DIM);
    ASSERT_EQ(wrap.get_weight_o("OIX"), 1);
    ASSERT_EQ(wrap.get_weight_o("XIO"), 1);

    ASSERT_EQ(wrap.get_src_c("XXX"), DNNL_GRAPH_UNKNOWN_DIM);
    ASSERT_EQ(wrap.get_src_c("NCX"), 2);
    ASSERT_EQ(wrap.get_src_c("NXC"), 1);
}

TEST(LogicalTensor, IsIdentical) {
    using ltw = graph::logical_tensor_wrapper_t;
    graph::logical_tensor_t lt1 = utils::logical_tensor_init(
            0, {1, 2, 2, 1}, graph::data_type::f32, graph::layout_type::undef);
    graph::logical_tensor_t lt2 = utils::logical_tensor_init(
            0, {1, 2, 2, 1}, graph::data_type::f32, graph::layout_type::undef);
    graph::logical_tensor_t lt3 = utils::logical_tensor_init(
            1, {1, 2, 2, 1}, graph::data_type::f32, graph::layout_type::undef);
    graph::logical_tensor_t lt4 = utils::logical_tensor_init(
            1, {1, 2, 2, 1}, graph::data_type::f32, graph::layout_type::opaque);
    graph::logical_tensor_t lt5 = utils::logical_tensor_init(
            1, {1, 2, 2, 1}, graph::data_type::f32, graph::layout_type::opaque);
    ASSERT_EQ(ltw(lt1).is_identical(lt2), true);
    ASSERT_EQ(ltw(lt1).is_identical(lt3), false);
    ASSERT_EQ(ltw(lt3).is_identical(lt4), false);
    ASSERT_EQ(ltw(lt4).is_identical(lt5), true);
}
