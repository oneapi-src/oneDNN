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
#include "oneapi/dnnl/dnnl_graph.h"

#include "interface/c_types_map.hpp"
#include "interface/tensor.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "gtest/gtest.h"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Tensor, SetDataHandle) {
    test::vector<int> data;
    auto tensor = dnnl_graph_tensor();
    tensor.set_data_handle(data.data());
    ASSERT_EQ(tensor.get_data_handle(), data.data());
}

TEST(Tensor, GetEngine) {
    graph::engine_t &engine = *get_engine();
    graph::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, graph::data_type::f32, graph::layout_type::strided);
    test::vector<float> data;
    auto tensor = dnnl_graph_tensor(lt, &engine, data.data());
    ASSERT_EQ(tensor.get_engine(), &engine);
}

#define DESTROY_TENSOR(t) \
    do { \
        dnnl_graph_tensor_destroy(t); \
        (t) = nullptr; \
    } while (0);

TEST(Tensor, DnnlGraphTensorCreate) {
    graph::tensor_t *tensor;
    graph::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, graph::data_type::f32, graph::layout_type::strided);
    graph::engine_t &engine = *get_engine();
    void *handle = nullptr;

    ASSERT_EQ_SAFE(dnnl_graph_tensor_create(&tensor, &lt, &engine, handle),
            graph::status::success, DESTROY_TENSOR(tensor));
    ASSERT_EQ_SAFE(dnnl_graph_tensor_destroy(tensor), graph::status::success,
            DESTROY_TENSOR(tensor));

    ASSERT_EQ_SAFE(dnnl_graph_tensor_create(&tensor, nullptr, &engine, handle),
            graph::status::invalid_arguments, DESTROY_TENSOR(tensor));
}

TEST(Tensor, DnnlGraphTensorSetDataHandle) {
    auto tensor = dnnl_graph_tensor();
    void *handle = nullptr;

    ASSERT_EQ(dnnl_graph_tensor_set_data_handle(nullptr, handle),
            graph::status::invalid_arguments);

    ASSERT_EQ(dnnl_graph_tensor_set_data_handle(&tensor, handle),
            graph::status::success);
}

TEST(Tensor, DnnlGraphTensorGetEngine) {
    graph::tensor_t *tensor;
    graph::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, graph::data_type::f32, graph::layout_type::strided);
    graph::engine_t &engine = *get_engine();
    void *handle = nullptr;
    ASSERT_EQ_SAFE(dnnl_graph_tensor_create(&tensor, &lt, &engine, handle),
            graph::status::success, DESTROY_TENSOR(tensor));
    graph::engine_t *ref_engine = nullptr;
    ASSERT_EQ_SAFE(dnnl_graph_tensor_get_engine(tensor, nullptr),
            graph::status::invalid_arguments, DESTROY_TENSOR(tensor));
    ASSERT_EQ_SAFE(dnnl_graph_tensor_get_engine(tensor, &ref_engine),
            graph::status::success, DESTROY_TENSOR(tensor));
    ASSERT_EQ_SAFE(ref_engine, &engine, DESTROY_TENSOR(tensor));
    ASSERT_EQ_SAFE(dnnl_graph_tensor_destroy(tensor), graph::status::success,
            DESTROY_TENSOR(tensor));
}
