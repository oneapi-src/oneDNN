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

#include "gtest/gtest.h"

#include <limits>

#include "interface/c_types_map.hpp"
#include "interface/internal_ops.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op_schema.hpp"
#include "interface/op_v2.hpp"
#include "interface/partition.hpp"
#include "interface/partition_impl.hpp"

#include "utils.hpp"

TEST(op_test, create) {
    using namespace dnnl::graph::impl;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.get_id(), 0);
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_FALSE(matmul.is_internal());
    ASSERT_NE(matmul.get_schema(), nullptr);
    ASSERT_EQ(matmul.get_schema()->get_name(), std::string("MatMul"));
}

TEST(op_test, create_internal) {
    using namespace dnnl::graph::impl;

    op_t matmul {0, op_kind::MatMul, std::string("matmul"), true};
    ASSERT_EQ(matmul.get_id(), 0);
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_TRUE(matmul.is_internal());
}

TEST(op_test, create_without_id) {
    using namespace dnnl::graph::impl;

    op_t matmul {op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.get_id(), std::numeric_limits<size_t>::max());
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_TRUE(matmul.is_internal());
}

TEST(op_test, add_input) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.num_inputs(), 0);
    ASSERT_EQ(matmul.num_outputs(), 0);

    logical_tensor_t lt0 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    ASSERT_EQ(matmul.num_inputs(), 2);
    ASSERT_EQ(matmul.num_outputs(), 0);
}

TEST(op_test, get_input) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt0 = logical_tensor_init(1, {2, 3}, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, {3, 4}, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    auto value0 = matmul.get_input_value(0);
    ASSERT_EQ(logical_tensor_wrapper(value0->get_logical_tensor()),
            logical_tensor_wrapper(lt0));

    auto value1 = matmul.get_input_value(1);
    ASSERT_EQ(logical_tensor_wrapper(value1->get_logical_tensor()),
            logical_tensor_wrapper(lt1));
}

TEST(op_test, get_input_values) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt0 = logical_tensor_init(1, {2, 3}, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, {3, 4}, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    auto values = matmul.get_input_values();
    ASSERT_EQ(values.size(), 2);
    ASSERT_EQ(logical_tensor_wrapper(values[0]->get_logical_tensor()),
            logical_tensor_wrapper(lt0));
    ASSERT_EQ(logical_tensor_wrapper(values[1]->get_logical_tensor()),
            logical_tensor_wrapper(lt1));
}

TEST(op_test, set_input_op) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t relu {0, op_kind::ReLU, std::string("relu")};
    logical_tensor_t lt0 = logical_tensor_init(1, {2, 3}, data_type::f32);
    relu.add_output(lt0);

    op_t matmul {2, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt1 = logical_tensor_init(3, {3, 4}, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    // setup the connection
    matmul.connect_input(0, relu, 0);

    auto op0 = matmul.get_input_op(0);
    ASSERT_EQ(op0->get_kind(), op_kind::ReLU);
    ASSERT_EQ(op0->get_id(), 0);
    ASSERT_EQ(op0->get_name(), std::string("relu"));

    ASSERT_EQ(relu.num_output_consumers(0), 1);
}

TEST(op_test, add_output) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.num_inputs(), 0);
    ASSERT_EQ(matmul.num_outputs(), 0);

    logical_tensor_t lt = logical_tensor_init(1, data_type::f32);
    matmul.add_output(lt);

    ASSERT_EQ(matmul.num_inputs(), 0);
    ASSERT_EQ(matmul.num_outputs(), 1);
}

TEST(op_test, get_output) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt = logical_tensor_init(1, {2, 3}, data_type::f32);
    matmul.add_output(lt);

    auto value = matmul.get_output_value(0);
    ASSERT_EQ(logical_tensor_wrapper(value->get_logical_tensor()),
            logical_tensor_wrapper(lt));
}

TEST(op_test, get_output_values) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt = logical_tensor_init(1, {2, 3}, data_type::f32);
    matmul.add_output(lt);

    auto values = matmul.get_output_values();
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(logical_tensor_wrapper(values[0]->get_logical_tensor()),
            logical_tensor_wrapper(lt));
}

TEST(op_test, set_attribute_b) {
    using namespace dnnl::graph::impl;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    matmul.set_attr<bool>("transpose_a", true);
    matmul.set_attr<bool>("transpose_b", false);

    ASSERT_EQ(matmul.num_attributes(), 2);
    ASSERT_TRUE(matmul.has_attr("transpose_a"));
    ASSERT_TRUE(matmul.has_attr("transpose_b"));
    ASSERT_FALSE(matmul.has_attr("hidden"));
    ASSERT_TRUE(matmul.get_attr<bool>("transpose_a"));
    ASSERT_FALSE(matmul.get_attr<bool>("transpose_b"));
}

TEST(op_test, set_attribute_f_s) {
    using namespace dnnl::graph::impl;

    op_t bn {0, op_kind::BatchNormInference, std::string("batch_norm")};
    bn.set_attr<float>("epsilon", 0.5);
    ASSERT_EQ(bn.get_attr<float>("epsilon"), 0.5);

    bn.set_attr<std::string>("data_format", "NCX");
    ASSERT_EQ(bn.get_attr<std::string>("data_format"), std::string("NCX"));
}

TEST(op_test, set_attribute_is) {
    using namespace dnnl::graph::impl;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<int64_t>("groups", 2);
    ASSERT_EQ(conv.get_attr<int64_t>("groups"), 2);

    conv.set_attr<std::vector<int64_t>>("pads_begin", {1, 1});
    auto ret = conv.get_attr<std::vector<int64_t>>("pads_begin");
    ASSERT_EQ(ret.size(), 2);
    ASSERT_EQ(ret[0], 1);
    ASSERT_EQ(ret[1], 1);
}

TEST(op_test, merge_attributes) {
    using namespace dnnl::graph::impl;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<int64_t>("groups", 2);
    conv.set_attr<std::string>("data_format", "NCX");

    std::unordered_map<std::string, op_t::attribute_value_t> other {};
    op_t::attribute_value_t fmt {std::string("OIX")};
    other.insert({"filter_format", fmt});
    std::vector<int64_t> pad {2, 2};
    other.insert({"pads_begein", {pad}});

    conv.merge_attributes(other);
    ASSERT_EQ(conv.num_attributes(), 4);
}

TEST(op_test, same_attributes) {
    using namespace dnnl::graph::impl;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<std::vector<int64_t>>("pads_begin", {2, 2});
    conv.set_attr<std::string>("data_format", "NCX");

    op_t pool {1, op_kind::MaxPool, std::string("max_pool")};
    pool.set_attr<std::vector<int64_t>>("pads_begin", {2, 2});
    pool.set_attr<std::string>("data_format", "NCX");

    ASSERT_TRUE(conv.is_same_attr_value(pool, "pads_begin"));
    ASSERT_TRUE(conv.is_same_attr_value(pool, "data_format"));
    ASSERT_TRUE(conv.has_same_attr_values(pool));

    op_t bn {2, op_kind::BatchNormInference, std::string("batch_norm")};
    bn.set_attr<std::string>("data_format", "NCX");
    ASSERT_TRUE(conv.is_same_attr_value(bn, "data_format"));
    ASSERT_FALSE(conv.has_same_attr_values(bn));
}

TEST(op_test, assigned_partition) {
    using namespace dnnl::graph::impl;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};

    ASSERT_FALSE(conv.is_assigned_to_partition());

    partition_impl_t *part {nullptr}; // use an empty partition for test purpose
    conv.set_partition(part);
    ASSERT_EQ(conv.get_partition(), part);
}

TEST(op_test, fused_op) {
    using namespace dnnl::graph::impl;
    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    op_t relu {1, op_kind::ReLU, std::string("relu")};
    ASSERT_FALSE(conv.is_fused());
    ASSERT_FALSE(relu.is_fused());

    op_t conv_relu {2, op_kind::conv_relu, std::string("conv_relu"), true};
    conv_relu.add_op_ids({0, 1});
    ASSERT_TRUE(conv_relu.is_fused());

    std::vector<size_t> ret = conv_relu.get_op_ids();
    ASSERT_EQ(ret.size(), 2);
    ASSERT_EQ(ret[0], 0);
    ASSERT_EQ(ret[1], 1);
}
