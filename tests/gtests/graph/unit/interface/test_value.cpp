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

#include "interface/c_types_map.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/value.hpp"

#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Value, Create) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {matmul, 0, lt};

    ASSERT_EQ(val.is_internal(), false);
}

TEST(Value, CreateInternal) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {matmul, 0, lt, true};

    ASSERT_EQ(val.is_internal(), true);
}

TEST(Value, GetLogicalTensor) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {matmul, 0, lt};

    auto lt1 = val.get_logical_tensor();
    ASSERT_TRUE(graph::logical_tensor_wrapper_t(lt)
            == graph::logical_tensor_wrapper_t(lt1));
}
TEST(Value, GetProducer) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {matmul, 0, lt};

    ASSERT_TRUE(val.has_producer());

    auto &prod = val.get_producer();
    ASSERT_EQ(&prod, &matmul);
}

TEST(Value, SetProducer) {
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {lt, false};
    ASSERT_FALSE(val.has_producer());

    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    val.set_producer(matmul);
    ASSERT_TRUE(val.has_producer());

    auto &prod = val.get_producer();
    ASSERT_EQ(&prod, &matmul);
}

TEST(Value, Equal) {
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val1 {lt, false};
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    val1.set_producer(matmul);
    val1.set_offset(0);

    graph::value_t val2 {matmul, 0, lt};
    ASSERT_EQ(val1, val2);

    graph::value_t val3 {matmul, 1, lt};
    ASSERT_NE(val1, val3);

    graph::logical_tensor_t lt1
            = utils::logical_tensor_init(123, graph::data_type::f32);

    graph::value_t val4 {matmul, 0, lt1};
    ASSERT_NE(val1, val4);
}

TEST(Value, Offset) {
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::value_t val {matmul, 1, lt};
    size_t offset = val.get_offset();
    ASSERT_EQ(offset, 1U);

    val.set_offset(2);
    offset = val.get_offset();
    ASSERT_EQ(offset, 2U);
}

TEST(Value, DefaultOffset) {
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::value_t val {lt};
    size_t offset = val.get_offset();
    size_t max = std::numeric_limits<size_t>::max();
    ASSERT_EQ(offset, max);

    val.set_offset(2);
    offset = val.get_offset();
    ASSERT_EQ(offset, 2U);
}

TEST(Value, AddConsumer) {
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(1, graph::data_type::f32);
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::value_t val {matmul, 0, lt};

    std::vector<graph::value_t::consumer_t> consumers = val.get_consumers();
    ASSERT_EQ(consumers.size(), 0U);

    graph::op_t relu {2, graph::op_kind::ReLU, std::string("relu")};
    val.add_consumer(relu, 0);
    consumers = val.get_consumers();
    ASSERT_EQ(consumers.size(), 1U);
}

TEST(Value, FindConsumer) {
    size_t id = 0;
    graph::logical_tensor_t lt
            = utils::logical_tensor_init(++id, graph::data_type::f32);
    graph::op_t matmul {++id, graph::op_kind::MatMul, std::string("matmul")};
    graph::value_t val {matmul, 0, lt};

    graph::op_t relu_op {id++, graph::op_kind::ReLU, std::string("relu")};
    val.add_consumer(relu_op, 0);

    graph::op_t abs_op {id++, graph::op_kind::Abs, std::string("abs")};
    val.add_consumer(abs_op, 1);

    auto ret1 = val.find_consumer(1, graph::op_kind::ReLU, 0);
    ASSERT_EQ(ret1.has_value(), false);

    auto ret2 = val.find_consumer(0, graph::op_kind::Abs, 0);
    ASSERT_EQ(ret2.has_value(), false);

    auto ret3 = val.find_consumer(0, graph::op_kind::Abs, 1);
    ASSERT_EQ(ret3.has_value(), true);

    auto ret4 = val.find_consumer(0, graph::op_kind::Abs, 1, true);
    ASSERT_EQ(ret4.has_value(), true);
}
