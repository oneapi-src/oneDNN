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

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Tensor, SetDataHandle) {
    test::vector<int> data;
    auto tensor = dnnl_graph_tensor();
    tensor.set_data_handle(data.data());
    ASSERT_EQ(tensor.get_data_handle(), data.data());
}

TEST(Tensor, GetEngine) {
    impl::engine_t &engine = get_engine();
    impl::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, impl::data_type::f32, impl::layout_type::strided);
    test::vector<float> data;
    auto tensor = dnnl_graph_tensor(lt, &engine, data.data());
    ASSERT_EQ(tensor.get_engine(), &engine);
}

TEST(Tensor, DnnlGraphTensorCreate) {
    impl::tensor_t *tensor = nullptr;
    impl::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, impl::data_type::f32, impl::layout_type::strided);
    impl::engine_t &engine = get_engine();
    void *handle = nullptr;
    impl::status_t ret = impl::status::success;

    ret = dnnl_graph_tensor_create(&tensor, &lt, &engine, handle);
    if (ret != impl::status::success) {
        dnnl_graph_tensor_destroy(tensor);
        ASSERT_EQ(ret, impl::status::success);
    }

    if (tensor != nullptr) {
        dnnl_graph_tensor_destroy(tensor);
        tensor = nullptr;
    }

    ret = dnnl_graph_tensor_create(&tensor, nullptr, &engine, handle);
    if (ret != impl::status::invalid_arguments) {
        dnnl_graph_tensor_destroy(tensor);
        ASSERT_EQ(ret, impl::status::success);
    }

    if (tensor != nullptr) {
        dnnl_graph_tensor_destroy(tensor);
        tensor = nullptr;
    }
}

TEST(Tensor, DnnlGraphTensorSetDataHandle) {
    auto tensor = dnnl_graph_tensor();
    void *handle = nullptr;

    ASSERT_EQ(dnnl_graph_tensor_set_data_handle(nullptr, handle),
            impl::status::invalid_arguments);

    ASSERT_EQ(dnnl_graph_tensor_set_data_handle(&tensor, handle),
            impl::status::success);
}

TEST(Tensor, DnnlGraphTensorGetEngine) {
    impl::tensor_t *tensor = nullptr;
    impl::logical_tensor_t lt = utils::logical_tensor_init(
            0, {1, 2}, impl::data_type::f32, impl::layout_type::strided);
    impl::engine_t &engine = get_engine();
    void *handle = nullptr;

    impl::status_t ret = impl::status::success;

    ret = dnnl_graph_tensor_create(&tensor, &lt, &engine, handle);
    if (ret != impl::status::success) {
        dnnl_graph_tensor_destroy(tensor);
        FAIL();
    }

    impl::engine_t *ref_engine = nullptr;
    ret = dnnl_graph_tensor_get_engine(tensor, nullptr);
    if (ret != impl::status::invalid_arguments) {
        dnnl_graph_tensor_destroy(tensor);
        FAIL();
    }

    ret = dnnl_graph_tensor_get_engine(tensor, &ref_engine);
    if (ret != impl::status::success) {
        dnnl_graph_tensor_destroy(tensor);
        FAIL();
    }

    if (ref_engine != &engine) {
        dnnl_graph_tensor_destroy(tensor);
        FAIL();
    }

    ASSERT_EQ(dnnl_graph_tensor_destroy(tensor), impl::status::success);
}
