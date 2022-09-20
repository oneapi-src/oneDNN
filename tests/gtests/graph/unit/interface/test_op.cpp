/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <array>
#include <limits>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/op_schema.hpp"
#include "interface/partition.hpp"
#include "interface/partition_impl.hpp"

#include "graph/unit/utils.hpp"

TEST(Op, ValidateMatmul) {
    using namespace dnnl::impl::graph;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.get_id(), 0U);
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_FALSE(matmul.is_internal());
}

TEST(Op, CreateInternal) {
    using namespace dnnl::impl::graph;

    op_t matmul {0, op_kind::MatMul, std::string("matmul"), true};
    ASSERT_EQ(matmul.get_id(), 0U);
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_TRUE(matmul.is_internal());
}

TEST(Op, CreateWithoutId) {
    using namespace dnnl::impl::graph;

    op_t matmul {op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.get_id(), std::numeric_limits<size_t>::max());
    ASSERT_EQ(matmul.get_kind(), op_kind::MatMul);
    ASSERT_EQ(matmul.get_name(), std::string("matmul"));
    ASSERT_TRUE(matmul.is_internal());
}

TEST(Op, AddInput) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.num_inputs(), 0U);
    ASSERT_EQ(matmul.num_outputs(), 0U);

    logical_tensor_t lt0 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    ASSERT_EQ(matmul.num_inputs(), 2U);
    ASSERT_EQ(matmul.num_outputs(), 0U);
}

TEST(Op, GetInput) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt0 = logical_tensor_init(1, {2, 3}, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, {3, 4}, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    auto value0 = matmul.get_input_value(0);
    ASSERT_EQ(logical_tensor_wrapper_t(value0->get_logical_tensor()),
            logical_tensor_wrapper_t(lt0));

    auto value1 = matmul.get_input_value(1);
    ASSERT_EQ(logical_tensor_wrapper_t(value1->get_logical_tensor()),
            logical_tensor_wrapper_t(lt1));
}

TEST(Op, GetInputValues) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt0 = logical_tensor_init(1, {2, 3}, data_type::f32);
    logical_tensor_t lt1 = logical_tensor_init(2, {3, 4}, data_type::f32);
    matmul.add_input(lt0);
    matmul.add_input(lt1);

    auto values = matmul.get_input_values();
    ASSERT_EQ(values.size(), 2U);
    ASSERT_EQ(logical_tensor_wrapper_t(values[0]->get_logical_tensor()),
            logical_tensor_wrapper_t(lt0));
    ASSERT_EQ(logical_tensor_wrapper_t(values[1]->get_logical_tensor()),
            logical_tensor_wrapper_t(lt1));
}

TEST(Op, SetInputOp) {
    using namespace dnnl::impl::graph;
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
    ASSERT_EQ(op0->get_id(), 0U);
    ASSERT_EQ(op0->get_name(), std::string("relu"));

    ASSERT_EQ(relu.num_output_consumers(0), 1U);
}

TEST(Op, AddOutput) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    ASSERT_EQ(matmul.num_inputs(), 0U);
    ASSERT_EQ(matmul.num_outputs(), 0U);

    logical_tensor_t lt = logical_tensor_init(1, data_type::f32);
    matmul.add_output(lt);

    ASSERT_EQ(matmul.num_inputs(), 0U);
    ASSERT_EQ(matmul.num_outputs(), 1U);
}

TEST(Op, GetOutput) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt = logical_tensor_init(1, {2, 3}, data_type::f32);
    matmul.add_output(lt);

    auto value = matmul.get_output_value(0);
    ASSERT_EQ(logical_tensor_wrapper_t(value->get_logical_tensor()),
            logical_tensor_wrapper_t(lt));
}

TEST(Op, GetOutputValues) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt = logical_tensor_init(1, {2, 3}, data_type::f32);
    matmul.add_output(lt);

    auto values = matmul.get_output_values();
    ASSERT_EQ(values.size(), 1U);
    ASSERT_EQ(logical_tensor_wrapper_t(values[0]->get_logical_tensor()),
            logical_tensor_wrapper_t(lt));
}

TEST(Op, SetAttributeBool) {
    using namespace dnnl::impl::graph;

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    matmul.set_attr<bool>(op_attr::transpose_a, true);
    matmul.set_attr<bool>(op_attr::transpose_b, false);

    ASSERT_EQ(matmul.num_attributes(), 2U);
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_a));
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_b));
    ASSERT_FALSE(matmul.has_attr(op_attr::undef));
    ASSERT_TRUE(matmul.get_attr<bool>(op_attr::transpose_a));
    ASSERT_FALSE(matmul.get_attr<bool>(op_attr::transpose_b));
}

TEST(Op, SetAttributeFloatString) {
    using namespace dnnl::impl::graph;

    op_t bn {0, op_kind::BatchNormInference, std::string("batch_norm")};
    bn.set_attr<float>(op_attr::epsilon, 0.5);
    ASSERT_EQ(bn.get_attr<float>(op_attr::epsilon), 0.5);

    bn.set_attr<std::string>(op_attr::data_format, "NCX");
    ASSERT_EQ(
            bn.get_attr<std::string>(op_attr::data_format), std::string("NCX"));
}

TEST(Op, SetAttributeInt64Vector) {
    using namespace dnnl::impl::graph;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<int64_t>(op_attr::groups, 2);
    ASSERT_EQ(conv.get_attr<int64_t>(op_attr::groups), 2);

    conv.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {1, 1});
    auto ret = conv.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    ASSERT_EQ(ret.size(), 2U);
    ASSERT_EQ(ret[0], 1);
    ASSERT_EQ(ret[1], 1);
}

TEST(Op, MergeAttributes) {
    using namespace dnnl::impl::graph;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<int64_t>(op_attr::groups, 2);
    conv.set_attr<std::string>(op_attr::data_format, "NCX");

    std::unordered_map<op_attr_t, op_t::attribute_value_t> other {};
    op_t::attribute_value_t fmt {std::string("OIX")};
    other.insert({op_attr::weights_format, fmt});
    std::vector<int64_t> pad {2, 2};
    other.insert({op_attr::pads_begin, {pad}});

    conv.merge_attributes(other);
    ASSERT_EQ(conv.num_attributes(), 4U);
}

TEST(Op, ValidateSameAttributes) {
    using namespace dnnl::impl::graph;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    conv.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {2, 2});
    conv.set_attr<std::string>(op_attr::data_format, "NCX");

    op_t pool {1, op_kind::MaxPool, std::string("max_pool")};
    pool.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {2, 2});
    pool.set_attr<std::string>(op_attr::data_format, "NCX");

    ASSERT_TRUE(conv.is_same_attr_value(pool, op_attr::pads_begin));
    ASSERT_TRUE(conv.is_same_attr_value(pool, op_attr::data_format));
    ASSERT_TRUE(conv.has_same_attr_values(pool));

    op_t bn {2, op_kind::BatchNormInference, std::string("batch_norm")};
    bn.set_attr<std::string>(op_attr::data_format, "NCX");
    ASSERT_TRUE(conv.is_same_attr_value(bn, op_attr::data_format));
    ASSERT_FALSE(conv.has_same_attr_values(bn));
}

TEST(Op, OverwriteAttributes) {
    using namespace dnnl::impl::graph;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    std::vector<int64_t> pad = {2, 2};
    conv.set_attr<std::vector<int64_t>>(op_attr::pads_begin, pad);
    conv.set_attr<std::string>(op_attr::data_format, "NCX");
    ASSERT_TRUE(conv.has_attr(op_attr::pads_begin));
    ASSERT_TRUE(conv.has_attr(op_attr::data_format));
    ASSERT_EQ(conv.get_attr<std::vector<int64_t>>(op_attr::pads_begin), pad);
    ASSERT_EQ(conv.get_attr<std::string>(op_attr::data_format), "NCX");

    // reset vector and string attributes
    pad = {1, 1};
    conv.set_attr<std::vector<int64_t>>(op_attr::pads_begin, pad);
    conv.set_attr<std::string>(op_attr::data_format, "NXC");
    ASSERT_TRUE(conv.has_attr(op_attr::pads_begin));
    ASSERT_TRUE(conv.has_attr(op_attr::data_format));
    ASSERT_EQ(conv.get_attr<std::vector<int64_t>>(op_attr::pads_begin), pad);
    ASSERT_EQ(conv.get_attr<std::string>(op_attr::data_format), "NXC");

    op_t matmul {0, op_kind::MatMul, std::string("matmul")};
    matmul.set_attr<bool>(op_attr::transpose_a, true);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    ASSERT_EQ(matmul.num_attributes(), 2U);
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_a));
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_b));
    ASSERT_TRUE(matmul.get_attr<bool>(op_attr::transpose_a));
    ASSERT_FALSE(matmul.get_attr<bool>(op_attr::transpose_b));

    // reset boolean attributes
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, true);
    ASSERT_EQ(matmul.num_attributes(), 2U);
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_a));
    ASSERT_TRUE(matmul.has_attr(op_attr::transpose_b));
    ASSERT_FALSE(matmul.get_attr<bool>(op_attr::transpose_a));
    ASSERT_TRUE(matmul.get_attr<bool>(op_attr::transpose_b));
}

TEST(Op, KindString) {
    using namespace dnnl::impl::graph;

    ASSERT_EQ(op_t::kind2str(op_kind::Abs), "Abs");
    ASSERT_EQ(op_t::kind2str(op_kind::Convolution), "Convolution");
    ASSERT_EQ(op_t::kind2str(op_kind::MatMul), "MatMul");
    ASSERT_EQ(op_t::kind2str(op_kind::LastSymbol), "LastSymbol");
}
