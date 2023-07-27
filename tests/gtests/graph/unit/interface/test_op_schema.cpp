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

#include <gtest/gtest.h>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op_def.hpp"
#include "interface/op_schema.hpp"

#include "graph/unit/utils.hpp"

using namespace dnnl::impl::graph;
using namespace dnnl::graph::tests::unit::utils;

#ifndef NDEBUG

TEST(OpSchemaDeathTest, DuplicateAttribute) {
    EXPECT_DEATH(op_schema_t()
                         .set_attr(op_attr::kernel, true, attribute_kind::b)
                         .set_attr(op_attr::kernel, true, attribute_kind::b),
            "provided attribute has already been set");
}

TEST(OpSchemaDeathTest, DuplicatedInput) {
    EXPECT_DEATH(op_schema_t().set_num_inputs(5).set_input(3, "mean").set_input(
                         3, "mean"),
            "provided `in_offset` has already been set");
}

TEST(OpSchemaDeathTest, DuplicatedOutput) {
    EXPECT_DEATH(op_schema_t()
                         .set_num_outputs(1)
                         .set_output(0, "output")
                         .set_output(0, "output"),
            "provided `out_offset` has already been set");
}

TEST(OpSchemaDeathTest, SetInputBeforeSetNumInputs) {
    EXPECT_DEATH(op_schema_t().set_input(0, "a").set_num_inputs(2),
            "input set before setting num_inputs_");
}

TEST(OpSchemaDeathTest, SetOutputBeforeSetNumOutputs) {
    EXPECT_DEATH(op_schema_t().set_output(0, "output").set_num_outputs(1),
            "output set before setting num_outputs_");
}

TEST(OpSchemaDeathTest, ExceededNumInputs) {
    EXPECT_DEATH(
            op_schema_t().set_num_inputs(1).set_input(0, "a").set_input(1, "b"),
            "input offset exceeds declared num of inputs");
}

TEST(OpSchemaDeathTest, ExceededNumOutputs) {
    EXPECT_DEATH(op_schema_t().set_num_outputs(1).set_output(0, "a").set_output(
                         1, "b"),
            "output offset exceeds declared num of outputs");
}

#endif

TEST(OpSchemaDeathTest, OpschemaMethod) {
    auto op_schema = op_schema_t(op_kind::Add, 1);

    ASSERT_NO_THROW({ op_schema.set_num_inputs(2); });
    ASSERT_NO_THROW({ op_schema.set_num_outputs(1); });
    ASSERT_NO_THROW({ op_schema.set_input(0, "a", "T"); });
    ASSERT_NO_THROW({ op_schema.set_input(1, "b", "T"); });
    ASSERT_NO_THROW({ op_schema.get_inputs(); });
    ASSERT_NO_THROW({ op_schema.set_output(0, "output", "T"); });

    ASSERT_NO_THROW({
        op_schema.set_type_constraints(
                "T", {data_type::f32, data_type::bf16, data_type::f16});
    });
    ASSERT_NO_THROW({ op_schema.get_outputs(); });
}

TEST(OpSchema, Convolution) {
    const op_kind_t op_kind_ = op_kind::Convolution;
    const std::set<size_t> expected_in_sizes = {2, 3};
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::groups, false}, {op_attr::data_format, false},
            {op_attr::weights_format, false}};
    for (auto expected_in_size : expected_in_sizes) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferConvolutionAutoPadShape) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Convolution);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::Convolution, op_t::kind2str(op_kind::Convolution)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t groups = 1;
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);

        auto lt_data = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        auto lt_weight
                = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &inferred_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &inferred_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {2, 2};
            std::vector<int64_t> expected_pads_end {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 1, 3, 3};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvolutionAutoPadNegtivePaddingSize) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Convolution);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::Convolution, op_t::kind2str(op_kind::Convolution)};
    std::vector<int64_t> strides = {3, 3};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {"SAME_UPPER", "SAME_LOWER"};
    int64_t groups = 1;
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);

        auto lt_data = logical_tensor_init(id++, {1, 1, 5, 5}, data_type::f32);
        auto lt_weight
                = logical_tensor_init(id++, {1, 1, 1, 1}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &inferred_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &inferred_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
        const std::vector<int64_t> expected_out_shape {1, 1, 2, 2};
        ASSERT_EQ(inferred_out_shape, expected_out_shape);

        std::vector<int64_t> expected_pads_begin {0, 0};
        std::vector<int64_t> expected_pads_end {0, 0};
        EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
        EXPECT_EQ(inferred_pads_end, expected_pads_end);
    }
}

TEST(OpSchema, InferConvolutionNXCAutoPadShapeWithDilations) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Convolution);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::Convolution, op_t::kind2str(op_kind::Convolution)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NXC";
    std::string filter_format = "XIO";
    const std::vector<std::string> auto_pads_vec {
            "None", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t groups = 1;
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);

        auto lt_data
                = logical_tensor_init(id++, {1, 100, 100, 160}, data_type::f32);
        auto lt_weight
                = logical_tensor_init(id++, {7, 1, 160, 192}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &infered_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &infered_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto infered_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "None" || auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 88, 100, 192};
            ASSERT_EQ(infered_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        } else {
            const std::vector<int64_t> expected_out_shape {1, 100, 100, 192};
            ASSERT_EQ(infered_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {6, 0};
            std::vector<int64_t> expected_pads_end {6, 0};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvtransposeAutoPadNegtivePaddingSize) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::ConvTranspose);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTranspose, op_t::kind2str(op_kind::ConvTranspose)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {"SAME_UPPER", "SAME_LOWER"};

    int64_t groups = 1;
    int64_t id = 0;
    // according to the convtranspose semantic, output_padding is bigger
    // than 0 only if stride is greater than 1.
    std::vector<int64_t> output_padding = {0, 0};
    for (const auto &auto_pad : auto_pads_vec) {
        set_convtranspose_common_attr(a_op, strides, pads_begin, pads_end,
                dilations, auto_pad, data_format, filter_format, groups,
                output_padding);

        auto lt_data = logical_tensor_init(0, {1, 1, 3, 3}, data_type::f32);
        auto lt_weight = logical_tensor_init(1, {1, 1, 1, 1}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &infered_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &infered_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto infered_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        const std::vector<int64_t> expected_out_shape {1, 1, 5, 5};
        ASSERT_EQ(infered_out_shape, expected_out_shape);

        std::vector<int64_t> expected_pads_begin {0, 0};
        std::vector<int64_t> expected_pads_end {0, 0};
        EXPECT_EQ(infered_pads_begin, expected_pads_begin);
        EXPECT_EQ(infered_pads_end, expected_pads_end);
    }
}

TEST(OpSchema, InferConvtransposeAutoPadShapeWithDilations) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::ConvTranspose);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTranspose, op_t::kind2str(op_kind::ConvTranspose)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};

    int64_t groups = 1;
    int64_t id = 0;
    // according to the convtranspose semantic, output_padding is bigger
    // than 0 only if stride is greater than 1.
    std::vector<int64_t> output_padding = {0, 0};
    for (const auto &auto_pad : auto_pads_vec) {
        set_convtranspose_common_attr(a_op, strides, pads_begin, pads_end,
                dilations, auto_pad, data_format, filter_format, groups,
                output_padding);

        auto lt_data = logical_tensor_init(0, {1, 1, 10, 10}, data_type::f32);
        auto lt_weight = logical_tensor_init(1, {1, 1, 4, 4}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &infered_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &infered_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto infered_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape {1, 1, 13, 13};
            ASSERT_EQ(infered_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 10, 10};
            ASSERT_EQ(infered_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {3, 3};
            std::vector<int64_t> expected_pads_end {3, 3};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 1, 16, 16};
            ASSERT_EQ(infered_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvolutionDilationsShape) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Convolution);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::Convolution, op_t::kind2str(op_kind::Convolution)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;
    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    auto lt_data = logical_tensor_init(id++, {1, 1, 16, 16}, data_type::f32);
    auto lt_weight = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 10, 10};

    auto lt_o = logical_tensor_init(id++, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    ASSERT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, ConvTranspose) {
    const op_kind_t op_kind_ = op_kind::ConvTranspose;
    const std::set<size_t> expected_in_sizes = {2, 3};
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::groups, false}, {op_attr::data_format, false},
            {op_attr::weights_format, false}, {op_attr::output_padding, false}};
    for (auto expected_in_size : expected_in_sizes) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferConvtransposeAutoPadShape) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::ConvTranspose);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTranspose, op_t::kind2str(op_kind::ConvTranspose)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};

    int64_t groups = 1;
    int64_t id = 0;
    // according to the convtranspose semantic, output_padding is bigger
    // than 0 only if stride is greater than 1.
    std::vector<int64_t> output_padding = {0, 0};
    for (const auto &auto_pad : auto_pads_vec) {
        set_convtranspose_common_attr(a_op, strides, pads_begin, pads_end,
                dilations, auto_pad, data_format, filter_format, groups,
                output_padding);

        auto lt_data = logical_tensor_init(0, {1, 1, 6, 6}, data_type::f32);
        auto lt_weight = logical_tensor_init(1, {1, 1, 4, 4}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &inferred_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &inferred_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {2, 2};
            std::vector<int64_t> expected_pads_end {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 1, 9, 9};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvtransposeDilationsShape) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::ConvTranspose);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTranspose, op_t::kind2str(op_kind::ConvTranspose)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;
    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    auto lt_data = logical_tensor_init(id++, {1, 1, 10, 10}, data_type::f32);
    auto lt_weight = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 16, 16};

    auto lt_o = logical_tensor_init(id++, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    ASSERT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvtransposeBiasOutputShape) {
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::ConvTranspose);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTranspose, op_t::kind2str(op_kind::ConvTranspose)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {}; // empty pads_begin
    std::vector<int64_t> pads_end = {}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;
    std::vector<int64_t> output_padding = {0, 0};

    set_convtranspose_common_attr(a_op, strides, pads_begin, pads_end,
            dilations, "SAME_UPPER", data_format, filter_format, groups,
            output_padding);

    auto lt_data = logical_tensor_init(0, {1, 1, 5, 5}, data_type::f32);
    auto lt_weight = logical_tensor_init(1, {1, 1, 3, 3}, data_type::f32);
    auto lt_o = logical_tensor_init(
            2, {1, 1, 5, 5}, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);

    pads_begin = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    pads_end = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    std::vector<int64_t> expected_pads_begin = {1, 1};
    std::vector<int64_t> expected_pads_end = {1, 1};
    EXPECT_EQ(pads_begin, expected_pads_begin);
    EXPECT_EQ(pads_end, expected_pads_end);
}

TEST(OpSchema, ConvTransposeBackwardData) {
    const op_kind_t op_kind_ = op_kind::ConvTransposeBackwardData;
    const std::set<size_t> expected_in_sizes = {2};
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::groups, false}, {op_attr::data_format, false},
            {op_attr::weights_format, false}};
    for (auto expected_in_size : expected_in_sizes) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferConvTransposeBackwardDataDilationsShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            op_kind::ConvTransposeBackwardData);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTransposeBackwardData,
            op_t::kind2str(op_kind::ConvTransposeBackwardData)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;
    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    auto lt_data = logical_tensor_init(id++, {1, 1, 16, 16}, data_type::f32);
    auto lt_weight = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 10, 10};

    auto lt_o = logical_tensor_init(id++, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    ASSERT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvTransposeBackwardDataAutoPadShape) {
    const op_kind_t op_kind_ = op_kind::ConvTransposeBackwardData;
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t a_op {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};

    int64_t groups = 1;
    int64_t id = 0;
    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);

        auto lt_data = logical_tensor_init(0, {1, 1, 6, 6}, data_type::f32);
        auto lt_weight = logical_tensor_init(1, {1, 1, 4, 4}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &inferred_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &inferred_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {2, 2};
            std::vector<int64_t> expected_pads_end {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 1, 3, 3};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvTransposeBackwardDataShapeWithNxcXio) {
    const auto kind = op_kind::ConvTransposeBackwardData;
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};
    const int64_t groups {1};
    const int64_t stride {4};
    const int64_t pad {0};
    const int64_t dilation {1};
    const std::vector<int64_t> diff_dst_shape {256, 3, 227, 227};
    const std::vector<int64_t> wei_shape {3, 96, 11, 11};
    const std::vector<int64_t> expected_diff_src_shape {256, 96, 55, 55};

    verify_shape_infer_for_convtranspose_bprop_data(kind, data_format,
            filter_format, groups, stride, pad, dilation, diff_dst_shape,
            wei_shape, expected_diff_src_shape);
}

TEST(OpSchema, InferConvTransposeBackwardDataShapeWithNcxOix) {
    const auto kind = op_kind::ConvTransposeBackwardData;
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};
    const int64_t groups {1};
    const int64_t stride {4};
    const int64_t pad {0};
    const int64_t dilation {1};
    const std::vector<int64_t> diff_dst_shape {256, 3, 227, 227};
    const std::vector<int64_t> wei_shape {3, 96, 11, 11};
    const std::vector<int64_t> expected_diff_src_shape {256, 96, 55, 55};

    verify_shape_infer_for_convtranspose_bprop_data(kind, data_format,
            filter_format, groups, stride, pad, dilation, diff_dst_shape,
            wei_shape, expected_diff_src_shape);
}

TEST(OpSchema, InferConvTransposeBackwardDataShapeWithNxcXioAndGroups) {
    const auto kind = op_kind::ConvTransposeBackwardData;
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};
    const int64_t groups {2};
    const int64_t stride {1};
    const int64_t pad {1};
    const int64_t dilation {1};
    const std::vector<int64_t> diff_dst_shape {256, 384, 13, 13};
    const std::vector<int64_t> wei_shape {192, 256, 3, 3};
    const std::vector<int64_t> expected_diff_src_shape {256, 256, 13, 13};

    verify_shape_infer_for_convtranspose_bprop_data(kind, data_format,
            filter_format, groups, stride, pad, dilation, diff_dst_shape,
            wei_shape, expected_diff_src_shape);
}

TEST(OpSchema, InferConvTransposeBackwardDataShapeWithNcxOixAndGroups) {
    const auto kind = op_kind::ConvTransposeBackwardData;
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};
    const int64_t groups {2};
    const int64_t stride {1};
    const int64_t pad {1};
    const int64_t dilation {1};
    const std::vector<int64_t> diff_dst_shape {256, 384, 13, 13};
    const std::vector<int64_t> wei_shape {192, 256, 3, 3};
    const std::vector<int64_t> expected_diff_src_shape {256, 256, 13, 13};

    verify_shape_infer_for_convtranspose_bprop_data(kind, data_format,
            filter_format, groups, stride, pad, dilation, diff_dst_shape,
            wei_shape, expected_diff_src_shape);
}

TEST(OpSchema, ConvTransposeBackwardWeights) {
    const op_kind_t op_kind_ = op_kind::ConvTransposeBackwardWeights;
    const std::set<size_t> expected_in_sizes = {2, 3};
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::groups, false}, {op_attr::data_format, false},
            {op_attr::weights_format, false}, {op_attr::weights_shape, false}};
    for (auto expected_in_size : expected_in_sizes) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferConvTransposeBackwardWeightsShapeFromAttribute) {
    const auto kind = op_kind::ConvTransposeBackwardWeights;
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(kind);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {kind, op_t::kind2str(kind)};
    std::vector<int64_t> strides = {4, 4};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    const std::vector<int64_t> expected_diff_wei_shape {3, 96, 11, 11};
    a_op.set_attr<std::vector<int64_t>>(
            op_attr::weights_shape, expected_diff_wei_shape);

    auto src_lt = logical_tensor_init(0, {256, 96, 55, 55}, data_type::f32);
    auto diff_dst_lt
            = logical_tensor_init(1, {256, 3, 227, 227}, data_type::f32);
    auto diff_wei_lt = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> in_lts {&src_lt, &diff_dst_lt};
    std::vector<logical_tensor_t *> out_lts {&diff_wei_lt};
    a_op_schema->shape_infer(&a_op, in_lts, out_lts);

    const std::vector<int64_t> inferred_diff_wei_shape
            = logical_tensor_wrapper_t(diff_wei_lt).vdims();
    EXPECT_EQ(inferred_diff_wei_shape, expected_diff_wei_shape);
}

TEST(OpSchema, InferConvTransposeBackwardWeightsDilationsShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            op_kind::ConvTransposeBackwardWeights);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvTransposeBackwardWeights,
            op_t::kind2str(op_kind::ConvTransposeBackwardWeights)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;
    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    // data shape {N, IC, H, W}
    const std::vector<int64_t> &in_data = {1, 1, 10, 10};
    const std::vector<int64_t> &expected_out_shape = {4, 4, 1, 1};

    a_op.set_attr(op_attr::weights_shape, expected_out_shape);
    auto lt_data = logical_tensor_init(id++, in_data, data_type::f32);

    logical_tensor_t lt_output_delta
            = logical_tensor_init(id++, {1, 1, 16, 16}, data_type::f32);

    std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
    logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    a_op_schema->shape_infer(&a_op, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvTransposeBackwardWeightsAutoPadShape) {
    const op_kind_t op_kind_ = op_kind::ConvTransposeBackwardWeights;
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t a_op {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};

    int64_t groups = 1;
    int64_t id = 0;
    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);

        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 1, 6, 6};
        const std::vector<int64_t> &expected_out_shape = {4, 4, 1, 1};

        a_op.set_attr(op_attr::weights_shape, expected_out_shape);
        auto lt_data = logical_tensor_init(id++, in_data, data_type::f32);

        logical_tensor_t lt_output_delta
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        if (auto_pad == "VALID") {
            lt_output_delta
                    = logical_tensor_init(id++, {1, 1, 9, 9}, data_type::f32);
        }

        std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
        logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        a_op_schema->shape_infer(&a_op, in, out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, GenerateDefaultAttribute) {
    const op_schema_t *matmul_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MatMul);
    op_t matmul_op {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt_data_a = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_b = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);

    matmul_op.add_input(lt_data_a);
    matmul_op.add_input(lt_data_b);
    matmul_op.add_output(lt_out);
    EXPECT_TRUE(matmul_op_schema->verify(&matmul_op));

    logical_tensor_t lt_bias = logical_tensor_init(3, data_type::f32);
    matmul_op.add_input(lt_bias);
    EXPECT_TRUE(matmul_op_schema->verify(&matmul_op));

    matmul_op.set_attr(op_attr::transpose_a, true);
    const bool *flag;
    const bool **ret_flag = &flag;
    matmul_op.get_attr<bool>(op_attr::transpose_a, ret_flag);
    EXPECT_TRUE(ret_flag);
    EXPECT_EQ(matmul_op.get_attr<bool>(op_attr::transpose_b, ret_flag),
            status::invalid_arguments);

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(&matmul_op), status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 1U);

    const auto &graph_matmul_op = agraph.get_ops()[0];
    EXPECT_TRUE(graph_matmul_op->get_attr<bool>(op_attr::transpose_a));
    EXPECT_FALSE(graph_matmul_op->get_attr<bool>(op_attr::transpose_b));
}

TEST(OpSchema, TestVerifyFunction) {
    const op_schema_t *conv_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Convolution);

    op_t conv_op {0, op_kind::Convolution, std::string("convolution")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);

    conv_op.add_input(lt_data);
    conv_op.add_input(lt_weight);
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    conv_op.add_output(lt_out);
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    conv_op.set_attr(op_attr::strides, strides);
    conv_op.set_attr(op_attr::pads_begin, pads_begin);
    conv_op.set_attr(op_attr::pads_end, pads_end);
    conv_op.set_attr(op_attr::dilations, dilations);
    conv_op.set_attr(op_attr::data_format, data_format);
    conv_op.set_attr(op_attr::weights_format, filter_format);

    EXPECT_TRUE(conv_op_schema->verify(&conv_op));

    conv_op.set_attr(op_attr::auto_pad, false);
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    std::string auto_pad = "VALID";
    conv_op.set_attr(op_attr::auto_pad, auto_pad);

    EXPECT_TRUE(conv_op_schema->verify(&conv_op));

    float arbitrary_value = 123.0;
    conv_op.set_attr(op_attr::undef, arbitrary_value);

    // not allow undefined attribute
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    const op_schema_t *select_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Select);

    logical_tensor_t lt_cond_invalid_dt
            = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_cond_valid_dt
            = logical_tensor_init(0, data_type::boolean);
    logical_tensor_t lt_src_0 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_src_1 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_dst = logical_tensor_init(3, data_type::f32);

    op_t select_op_invalid_dt {0, op_kind::Select, std::string("select")};
    select_op_invalid_dt.add_input(lt_cond_invalid_dt);
    select_op_invalid_dt.add_input(lt_src_0);
    select_op_invalid_dt.add_input(lt_src_1);
    select_op_invalid_dt.add_output(lt_dst);
    EXPECT_FALSE(select_op_schema->verify(&select_op_invalid_dt));

    op_t select_op_valid_dt {0, op_kind::Select, std::string("select")};
    select_op_valid_dt.add_input(lt_cond_valid_dt);
    select_op_valid_dt.add_input(lt_src_0);
    select_op_valid_dt.add_input(lt_src_1);
    select_op_valid_dt.add_output(lt_dst);
    EXPECT_TRUE(select_op_schema->verify(&select_op_valid_dt));
}

TEST(OpSchema, InferConvOutputShape) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NCX";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 111, 111};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvtransposeOutputShape) {
    const op_kind_t op_kind_ = op_kind::ConvTranspose;

    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::vector<int64_t> groups_vec {1, 2, 4};

    // data shape {N, IC, H, W}
    const std::vector<int64_t> &in_data = {1, 16, 111, 111};
    const std::vector<int64_t> &expected_out_shape = {1, 32, 224, 224};
    for (auto groups : groups_vec) {
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {32 / groups, 16, 3, 3};

        verify_shape_infer_for_convtranspose(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, expected_out_shape);
    }
}

TEST(OpSchema, InferConv3dOutputShape) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NCX";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, D, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224, 224};
        // weight shape {OC, IC, KD, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3, 3};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 111, 111, 111};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvtransposeOutputShapeWithNxcFormat) {
    const op_kind_t op_kind_ = op_kind::ConvTranspose;

    std::string data_format = "NXC";
    std::string filter_format = "OIX";
    std::vector<int64_t> groups_vec {1, 2, 4};

    // data shape {N, H, W, IC}
    const std::vector<int64_t> &in_data = {1, 111, 111, 16};
    const std::vector<int64_t> &expected_out_shape = {1, 224, 224, 32};
    for (auto groups : groups_vec) {
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {32 / groups, 16, 3, 3};

        verify_shape_infer_for_convtranspose(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, expected_out_shape);
    }
}

TEST(OpSchema, InferConvOutputShapeWithNxcFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NXC";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 32};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3};
        const std::vector<int64_t> &expected_out_shape = {1, 111, 111, 16};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConv3dOutputShapeWithNxcFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NXC";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, D, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 224, 32};
        // weight shape {OC, IC, KD, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3, 3};
        const std::vector<int64_t> &expected_out_shape = {1, 111, 111, 111, 16};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvOutputShapeWithNxcXioFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NXC";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 32};
        // weight shape {KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 32 / groups, 16};
        const std::vector<int64_t> &expected_out_shape = {1, 111, 111, 16};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvtransposeOutputShapeWithNxcXioFormat) {
    const op_kind_t op_kind_ = op_kind::ConvTranspose;

    std::string data_format = "NXC";
    std::string filter_format = "XIO";
    std::vector<int64_t> groups_vec {1, 2, 4};

    // data shape {N, H, W, IC}
    const std::vector<int64_t> &in_data = {1, 111, 111, 16};
    const std::vector<int64_t> &expected_out_shape = {1, 224, 224, 32};
    for (auto groups : groups_vec) {
        // weight shape {KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 16, 32 / groups};

        verify_shape_infer_for_convtranspose(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, expected_out_shape);
    }
}

TEST(OpSchema, InferConv3dOutputShapeWithNxcXioFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NXC";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, D, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 224, 32};
        // weight shape {KD, KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 3, 32 / groups, 16};
        const std::vector<int64_t> &expected_out_shape = {1, 111, 111, 111, 16};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvOutputShapeWithXioFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NCX";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224};
        // weight shape {KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 32 / groups, 16};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 111, 111};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, InferConvtransposeOutputShapeWithXioFormat) {
    const op_kind_t op_kind_ = op_kind::ConvTranspose;

    std::string data_format = "NCX";
    std::string filter_format = "XIO";
    std::vector<int64_t> groups_vec {1, 2, 4};

    // data shape {N, IC, H, W}
    const std::vector<int64_t> &in_data = {1, 16, 111, 111};
    const std::vector<int64_t> &expected_out_shape = {1, 32, 224, 224};
    for (auto groups : groups_vec) {
        // weight shape {KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 16, 32 / groups};

        verify_shape_infer_for_convtranspose(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, expected_out_shape);
    }
}

TEST(OpSchema, InferConv3dOutputShapeWithXioFormat) {
    const op_kind_t op_kind_ = op_kind::Convolution;

    std::string data_format = "NCX";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, D, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224, 224};
        // weight shape {KD, KH, KW, IC, OC}
        const std::vector<int64_t> &in_weight = {3, 3, 3, 32 / groups, 16};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 111, 111, 111};

        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, expected_out_shape);

        const std::vector<int64_t> &in_bias = {16};
        verify_shape_infer_for_conv(op_kind_, data_format, filter_format,
                groups, in_data, in_weight, in_bias, expected_out_shape);
    }
}

TEST(OpSchema, MaxPool) {
    const op_kind_t op_kind_ = op_kind::MaxPool;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::kernel, true}, {op_attr::pads_begin, true},
            {op_attr::pads_end, true}, {op_attr::dilations, false},
            {op_attr::data_format, false}, {op_attr::auto_pad, false},
            {op_attr::rounding_type, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMaxpoolOutputShape) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPool);

    op_t pool_op {op_kind::MaxPool, op_t::kind2str(op_kind::MaxPool)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> kernel = {3, 3};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "SAME_UPPER";
    std::string data_format = "NCX";

    pool_op.set_attr(op_attr::strides, strides);
    pool_op.set_attr(op_attr::pads_begin, pads_begin);
    pool_op.set_attr(op_attr::pads_end, pads_end);
    pool_op.set_attr(op_attr::kernel, kernel);
    pool_op.set_attr(op_attr::dilations, dilations);
    pool_op.set_attr(op_attr::auto_pad, auto_pad);
    pool_op.set_attr(op_attr::data_format, data_format);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_data};
    logical_tensor_t lt_o
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out {&lt_o};

    // if output shape is unknown, infer output shape
    pool_op_schema->shape_infer(&pool_op, lt_in, lt_out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    const std::vector<int64_t> inferred_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(inferred_out_strides, expected_out_strides);

    // if output shape is known, infer auto pad
    pool_op_schema->shape_infer(&pool_op, lt_in, lt_out);
    auto inferred_pads_begin
            = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto inferred_pads_end
            = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    const std::vector<int64_t> expected_pads_begin = {0, 0};
    const std::vector<int64_t> expected_pads_end = {1, 1};
    EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
    EXPECT_EQ(inferred_pads_end, expected_pads_end);
}

TEST(OpSchema, InferMaxpoolAutoPadShape) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPool);
    op_t pool_op {op_kind::MaxPool, op_t::kind2str(op_kind::MaxPool)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";

    pool_op.set_attr(op_attr::strides, strides);
    pool_op.set_attr(op_attr::pads_begin, pads_begin);
    pool_op.set_attr(op_attr::pads_end, pads_end);
    pool_op.set_attr(op_attr::kernel, kernel);
    pool_op.set_attr(op_attr::dilations, dilations);

    pool_op.set_attr(op_attr::data_format, data_format);
    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        pool_op.set_attr(op_attr::auto_pad, auto_pad);
        logical_tensor_t lt_data
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        std::vector<logical_tensor_t *> lt_in {&lt_data};
        logical_tensor_t lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        // if output shape is unknown, infer output shape
        pool_op_schema->shape_infer(&pool_op, lt_in, lt_out);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();

        auto inferred_pads_begin
                = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        auto inferred_pads_end
                = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);

        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {2, 2};
            const std::vector<int64_t> expected_pads_end = {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);

        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 3, 3};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {0, 0};
            const std::vector<int64_t> expected_pads_end = {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferMaxpoolDilationsShape) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPool);
    op_t pool_op {op_kind::MaxPool, op_t::kind2str(op_kind::MaxPool)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    const std::vector<int64_t> expected_out_shape = {1, 1, 10, 10};
    int64_t id = 0;

    pool_op.set_attr(op_attr::strides, strides);
    pool_op.set_attr(op_attr::pads_begin, pads_begin);
    pool_op.set_attr(op_attr::pads_end, pads_end);
    pool_op.set_attr(op_attr::kernel, kernel);
    pool_op.set_attr(op_attr::dilations, dilations);

    pool_op.set_attr(op_attr::data_format, data_format);
    pool_op.set_attr<std::string>(op_attr::auto_pad, "VALID");

    logical_tensor_t lt_data
            = logical_tensor_init(id++, {1, 1, 16, 16}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_data};
    logical_tensor_t lt_o
            = logical_tensor_init(id++, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    // if output shape is unknown, infer output shape
    pool_op_schema->shape_infer(&pool_op, lt_in, lt_out);

    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferMaxpoolOutputShapeWithRoundingType) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPool);

    std::set<std::string> rounding_types = {"ceil", "floor"};
    for (auto &rounding_type : rounding_types) {
        op_t pool_op {op_kind::MaxPool, op_t::kind2str(op_kind::MaxPool)};
        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> kernel = {3, 3};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> dilations = {1, 1};
        std::string data_format = "NCX";
        pool_op.set_attr(op_attr::strides, strides);
        pool_op.set_attr(op_attr::pads_begin, pads_begin);
        pool_op.set_attr(op_attr::pads_end, pads_end);
        pool_op.set_attr(op_attr::kernel, kernel);
        pool_op.set_attr(op_attr::dilations, dilations);
        pool_op.set_attr(op_attr::data_format, data_format);
        pool_op.set_attr(op_attr::rounding_type, rounding_type);

        logical_tensor_t lt_data
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        std::vector<logical_tensor_t *> lt_in {&lt_data};
        logical_tensor_t lt_o
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_out {&lt_o};

        // if output shape is unknown, infer output shape
        pool_op_schema->shape_infer(&pool_op, lt_in, lt_out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();
        const std::vector<int64_t> inferred_out_strides
                = logical_tensor_wrapper_t(lt_o).vstrides();
        if (rounding_type == "ceil") {
            const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(inferred_out_strides, expected_out_strides);
        } else { // rounding_type = floor
            const std::vector<int64_t> expected_out_shape = {1, 3, 111, 111};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(inferred_out_strides, expected_out_strides);
        }
    }
}

TEST(OpSchema, InferMatmulOutputShape) {
    const op_schema_t *matmul_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MatMul);

    op_t matmul_op {op_kind::MatMul, op_t::kind2str(op_kind::MatMul)};
    bool transpose_a = true;
    matmul_op.set_attr(op_attr::transpose_a, transpose_a);

    // test 2 dims matmul
    logical_tensor_t lt_in1
            = logical_tensor_init(0, {1024, 64}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(1, {1024, 1000}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1, &lt_in2};
    logical_tensor_t lt_o1
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out1);

    const std::vector<int64_t> inferred_out_shape1
            = logical_tensor_wrapper_t(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(inferred_out_shape1, expected_out_shape1);

    const std::vector<int64_t> inferred_out_strides1
            = logical_tensor_wrapper_t(lt_o1).vstrides();
    const std::vector<int64_t> expected_out_strides1
            = compute_dense_strides(expected_out_shape1);
    EXPECT_EQ(inferred_out_strides1, expected_out_strides1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    logical_tensor_t lt_o2
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out2);

    auto &inferred_out_shape2 = lt_o2.dims;
    EXPECT_EQ(lt_o2.ndims, 1);
    EXPECT_EQ(inferred_out_shape2[0], 1000);
    auto &inferred_out_strides2 = lt_o2.layout.strides;
    EXPECT_EQ(inferred_out_strides2[0], 1);

    // test >2 dims matmul
    lt_in1 = logical_tensor_init(0, {3, 1, 10, 1024, 64}, data_type::f32);
    lt_in2 = logical_tensor_init(1, {5, 1, 1024, 1000}, data_type::f32);
    logical_tensor_t lt_o3
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out3);

    const std::vector<int64_t> inferred_out_shape3
            = logical_tensor_wrapper_t(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(inferred_out_shape3, expected_out_shape3);

    const std::vector<int64_t> inferred_out_strides3
            = logical_tensor_wrapper_t(lt_o3).vstrides();
    const std::vector<int64_t> expected_out_strides3
            = compute_dense_strides(expected_out_shape3);
    EXPECT_EQ(inferred_out_strides3, expected_out_strides3);
}

TEST(OpSchema, Abs) {
    const op_kind_t op_kind_ = op_kind::Abs;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, AbsBackward) {
    const op_kind_t op_kind_ = op_kind::AbsBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, Add) {
    const op_kind_t op_kind_ = op_kind::Add;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferAddOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Add;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferAddOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Add;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, BatchNormForwardTraining) {
    const op_kind_t op_kind_ = op_kind::BatchNormForwardTraining;
    const size_t expected_in_size = 5;
    const size_t expected_out_size = 5;
    const size_t expected_attr_size = 3;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::epsilon, true},
            {op_attr::momentum, false}, {op_attr::data_format, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferBatchNormForwardTrainingOutputShape) {
    const op_kind_t op_kind_ = op_kind::BatchNormForwardTraining;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(1, {224}, data_type::f32);
    logical_tensor_t lt_variance
            = logical_tensor_init(2, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_mean, &lt_variance};
    logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_r_mean = logical_tensor_init(4, data_type::f32);
    logical_tensor_t lt_r_var = logical_tensor_init(5, data_type::f32);
    logical_tensor_t lt_b_mean = logical_tensor_init(6, data_type::f32);
    logical_tensor_t lt_b_var = logical_tensor_init(7, data_type::f32);
    std::vector<logical_tensor_t *> out {
            &lt_out, &lt_r_mean, &lt_r_var, &lt_b_mean, &lt_b_var};

    op_.set_attr<float>(op_attr::epsilon, 0.01f);
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape0
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape0 = {1, 3, 224, 224};
    EXPECT_EQ(inferred_out_shape0, expected_out_shape0);

    const std::vector<int64_t> inferred_out_shape1
            = logical_tensor_wrapper_t(lt_r_mean).vdims();
    const std::vector<int64_t> inferred_out_shape2
            = logical_tensor_wrapper_t(lt_r_var).vdims();
    const std::vector<int64_t> inferred_out_shape3
            = logical_tensor_wrapper_t(lt_b_mean).vdims();
    const std::vector<int64_t> inferred_out_shape4
            = logical_tensor_wrapper_t(lt_b_var).vdims();
    const std::vector<int64_t> expected_out_shape_1D = {224};
    EXPECT_EQ(inferred_out_shape1, expected_out_shape_1D);
    EXPECT_EQ(inferred_out_shape2, expected_out_shape_1D);
    EXPECT_EQ(inferred_out_shape3, expected_out_shape_1D);
    EXPECT_EQ(inferred_out_shape4, expected_out_shape_1D);

    logical_tensor_t lt_out_not_filled = logical_tensor_init(8, data_type::f32);
    std::vector<logical_tensor_t *> out_partially_not_filled {
            &lt_out_not_filled, &lt_r_mean, &lt_r_var, &lt_b_mean, &lt_b_var};
    const std::string data_f = "NCX";
    op_.set_attr(op_attr::data_format, data_f);
    auto result = op_schema_->shape_infer(&op_, in, out_partially_not_filled);
    EXPECT_EQ(result, status::invalid_shape);
}

TEST(OpSchema, BatchNormTrainingBackward) {
    const op_kind_t op_kind_ = op_kind::BatchNormTrainingBackward;
    const size_t expected_in_size = 5;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 2;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::epsilon, true}, {op_attr::data_format, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferBatchNormTrainingBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::BatchNormTrainingBackward;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_output_delta
            = logical_tensor_init(1, {1, 2, 224, 224}, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(2, {224}, data_type::f32);
    logical_tensor_t lt_variance
            = logical_tensor_init(3, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in {
            &lt_in, &lt_output_delta, &lt_mean, &lt_variance};
    logical_tensor_t lt_input_delta = logical_tensor_init(4, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_input_delta};

    op_.set_attr<float>(op_attr::epsilon, 0.01f);
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape0
            = logical_tensor_wrapper_t(lt_input_delta).vdims();
    const std::vector<int64_t> expected_out_shape0 = {1, 3, 224, 224};
    EXPECT_EQ(inferred_out_shape0, expected_out_shape0);

    logical_tensor_t lt_gamma = logical_tensor_init(5, {224}, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(6, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in_with_options {&lt_in, &lt_output_delta,
            &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
    logical_tensor_t lt_gamma_delta = logical_tensor_init(7, data_type::f32);
    logical_tensor_t lt_beta_delta = logical_tensor_init(8, data_type::f32);
    std::vector<logical_tensor_t *> out_with_options {
            &lt_input_delta, &lt_gamma_delta, &lt_beta_delta};

    op_schema_->shape_infer(&op_, in_with_options, out_with_options);
    const std::vector<int64_t> expected_out_shape_1D = {224};
    const std::vector<int64_t> inferred_out_shape1
            = logical_tensor_wrapper_t(lt_gamma_delta).vdims();
    const std::vector<int64_t> inferred_out_shape2
            = logical_tensor_wrapper_t(lt_beta_delta).vdims();
    EXPECT_EQ(inferred_out_shape1, expected_out_shape_1D);
    EXPECT_EQ(inferred_out_shape2, expected_out_shape_1D);
}

TEST(OpSchema, BiasAdd) {
    const op_kind_t op_kind_ = op_kind::BiasAdd;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::data_format, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferBiasAddOutputShape) {
    const op_kind_t op_kind_ = op_kind::BiasAdd;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(1, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_bias};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 224, 224};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    logical_tensor_t lt_out_not_filled = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out_not_filled {&lt_out_not_filled};
    const std::string data_f = "NCX";
    op_.set_attr(op_attr::data_format, data_f);
    auto result = op_schema_->shape_infer(&op_, in, out_not_filled);
    EXPECT_EQ(result, status::invalid_shape);
}

TEST(OpSchema, BiasAddBackward) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackward;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::data_format, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferBiasAddBackwardOutputShapeWithNxcFormat) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackward;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    const int64_t channels = 16;
    logical_tensor_t lt_in
            = logical_tensor_init(0, {3, 64, 64, channels}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // no need to set attribute, NXC value should be default
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {channels};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    // explicitly setting data_format to NXC
    const std::string default_data_f = "NXC";
    op_.set_attr(op_attr::data_format, default_data_f);
    logical_tensor_t lt_out_expl = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out_expl {&lt_out_expl};

    op_schema_->shape_infer(&op_, in, out_expl);
    const std::vector<int64_t> inferred_out_shape_expl
            = logical_tensor_wrapper_t(lt_out_expl).vdims();
    EXPECT_EQ(inferred_out_shape_expl, expected_out_shape);
}

TEST(OpSchema, InferBiasAddBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackward;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    const int64_t channels = 16;
    logical_tensor_t lt_in
            = logical_tensor_init(0, {3, channels, 64, 64}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};
    const std::string data_f = "NCX";
    op_.set_attr(op_attr::data_format, data_f);

    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {channels};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, Clamp) {
    const op_kind_t op_kind_ = op_kind::Clamp;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::min, true}, {op_attr::max, true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferClampOutputShape) {
    const op_kind_t op_kind_ = op_kind::Clamp;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, ClampBackward) {
    const op_kind_t op_kind_ = op_kind::ClampBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 3;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::min, true}, {op_attr::max, true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferClampBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::ClampBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, ConvolutionBackwardData) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 10;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::output_padding, false}, {op_attr::groups, false},
            {op_attr::data_format, false}, {op_attr::weights_format, false},
            {op_attr::dst_shape, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferConvolutionBackwardDataOutputShape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;

    std::string data_format = "NCX";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};
    // data shape {N, IC, H, W}
    std::vector<int64_t> in_data {1, 16, 224, 224};
    std::vector<int64_t> in_output_shape {};
    std::vector<int64_t> expected_out_shape {1, 32, 452, 452};

    for (auto groups : groups_vec) {
        // weight shape {OC, IC, KH, KW}
        std::vector<int64_t> in_weight = {16, 32 / groups, 3, 3};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(OpSchema, InferConvolutionBackwardDataAutoPadShape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t a_op {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t groups = 1;
    int64_t id = 0;
    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);
        if (auto_pad == "NONE") {
            a_op.set_attr<std::vector<int64_t>>(
                    op_attr::dst_shape, {1, 1, 6, 6});
        } else if (auto_pad == "SAME_UPPER") {
            a_op.set_attr<std::vector<int64_t>>(
                    op_attr::dst_shape, {1, 1, 6, 6});
        } else if (auto_pad == "SAME_LOWER") {
            a_op.set_attr<std::vector<int64_t>>(
                    op_attr::dst_shape, {1, 1, 6, 6});
        } else if (auto_pad == "VALID") {
            a_op.set_attr<std::vector<int64_t>>(
                    op_attr::dst_shape, {1, 1, 9, 9});
        }
        auto lt_data = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        auto lt_weight
                = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
        auto lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);

        const auto &inferred_pads_begin
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        const auto &inferred_pads_end
                = a_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {1, 1};
            std::vector<int64_t> expected_pads_end {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape {1, 1, 6, 6};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {2, 2};
            std::vector<int64_t> expected_pads_end {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape {1, 1, 9, 9};
            ASSERT_EQ(inferred_out_shape, expected_out_shape);

            std::vector<int64_t> expected_pads_begin {0, 0};
            std::vector<int64_t> expected_pads_end {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferConvolutionBackwardDataDilationsShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            op_kind::ConvolutionBackwardData);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvolutionBackwardData,
            op_t::kind2str(op_kind::ConvolutionBackwardData)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;
    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);
    a_op.set_attr<std::vector<int64_t>>(op_attr::dst_shape, {1, 1, 16, 16});

    auto lt_data = logical_tensor_init(id++, {1, 1, 10, 10}, data_type::f32);
    auto lt_weight = logical_tensor_init(id++, {1, 1, 4, 4}, data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 16, 16};

    auto lt_o = logical_tensor_init(id++, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    const auto inferred_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    ASSERT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvolutionBackwardDataOutputShapeWithNxcFormat) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;

    std::string data_format = "NXC";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};
    // data shape {N, H, W, IC}
    std::vector<int64_t> in_data {1, 224, 224, 16};
    std::vector<int64_t> in_output_shape {};
    std::vector<int64_t> expected_out_shape {1, 452, 452, 32};

    for (auto groups : groups_vec) {
        // weight shape {OC, IC, KH, KW}
        std::vector<int64_t> in_weight = {16, 32 / groups, 3, 3};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(OpSchema, InferConvolutionBackwardDataOutputShapeWithNxcXioFormat) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;

    std::string data_format = "NXC";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};
    // data shape {N, H, W, IC}
    std::vector<int64_t> in_data {1, 224, 224, 16};
    std::vector<int64_t> in_output_shape {};
    std::vector<int64_t> expected_out_shape {1, 452, 452, 32};

    for (auto groups : groups_vec) {
        // weight shape {KH, KW, IC, OC}
        std::vector<int64_t> in_weight = {3, 3, 32 / groups, 16};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(OpSchema, InferConvolutionBackwardDataOutputShapeWithXioFormat) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardData;

    std::string data_format = "NCX";
    std::string filter_format = "XIO";

    // data shape {N, IC, H, W}
    std::vector<int64_t> in_data {1, 16, 224, 224};
    std::vector<int64_t> in_output_shape {};
    std::vector<int64_t> expected_out_shape {1, 32, 452, 452};

    std::vector<int64_t> groups_vec {1, 2, 4};
    for (auto groups : groups_vec) {
        // weight shape {KH, KW, IC, OC}
        std::vector<int64_t> in_weight {3, 3, 32 / groups, 16};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(OpSchema, ConvolutionBackwardWeights) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardWeights;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::dilations, true}, {op_attr::auto_pad, false},
            {op_attr::groups, false}, {op_attr::data_format, false},
            {op_attr::weights_format, false}, {op_attr::weights_shape, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferConvolutionBackwardWeightsDilationsShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            op_kind::ConvolutionBackwardWeights);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {op_kind::ConvolutionBackwardWeights,
            op_t::kind2str(op_kind::ConvolutionBackwardWeights)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";

    int64_t groups = 1;
    int64_t id = 0;

    const std::vector<int64_t> &in_data = {1, 1, 16, 16};
    const std::vector<int64_t> &expected_out_shape = {4, 4, 1, 1};
    const std::vector<int64_t> &in_output_delta = {1, 1, 10, 10};

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);
    a_op.set_attr(op_attr::weights_shape, expected_out_shape);

    auto lt_data = logical_tensor_init(id++, in_data, data_type::f32);
    logical_tensor_t lt_output_delta
            = logical_tensor_init(id++, in_output_delta, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
    logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    a_op_schema->shape_infer(&a_op, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvolutionBackwardWeightsOutputShape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardWeights;

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";
    std::string data_format = "NCX";
    std::string filter_format = "XIO";
    int64_t groups = 1;

    // data shape {N, IC, H, W}
    const std::vector<int64_t> &in_data = {1, 3, 224, 224};
    const std::vector<int64_t> &expected_out_shape = {4, 4, 3, 16};
    const std::vector<int64_t> &in_output_delta = {1, 16, 111, 111};

    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    set_conv_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);
    op_.set_attr(op_attr::weights_shape, expected_out_shape);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_output_delta
            = logical_tensor_init(1, in_output_delta, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvolutionBackwardWeightsAutoPadShape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackwardWeights;
    const op_schema_t *a_op_schema
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t a_op {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {1, 1}; // empty pads_begin
    std::vector<int64_t> pads_end = {2, 2}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> filters {1, 1, 4, 4};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    const std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t groups = 1;
    int64_t id = 0;
    for (const auto &auto_pad : auto_pads_vec) {
        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                auto_pad, data_format, filter_format, groups);
        a_op.set_attr(op_attr::weights_shape, filters);

        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 1, 6, 6};
        const std::vector<int64_t> &expected_out_shape = {4, 4, 1, 1};

        a_op.set_attr(op_attr::weights_shape, expected_out_shape);
        auto lt_data = logical_tensor_init(id++, in_data, data_type::f32);

        logical_tensor_t lt_output_delta
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        if (auto_pad == "VALID") {
            lt_output_delta
                    = logical_tensor_init(id++, {1, 1, 3, 3}, data_type::f32);
        }

        std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
        logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        a_op_schema->shape_infer(&a_op, in, out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, AvgPool) {
    const op_kind_t op_kind_ = op_kind::AvgPool;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::kernel, true}, {op_attr::pads_begin, true},
            {op_attr::pads_end, true}, {op_attr::data_format, false},
            {op_attr::auto_pad, false}, {op_attr::rounding_type, false},
            {op_attr::exclude_pad, true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferAvgpoolOutputShape) {
    const op_schema_t *avg_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::AvgPool);

    op_t avg_op {op_kind::AvgPool, op_t::kind2str(op_kind::AvgPool)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> kernel = {3, 3};
    bool exclude_pad = false;
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::string auto_pad = "SAME_UPPER";
    std::string data_format = "NCX";

    avg_op.set_attr(op_attr::strides, strides);
    avg_op.set_attr(op_attr::pads_begin, pads_begin);
    avg_op.set_attr(op_attr::pads_end, pads_end);
    avg_op.set_attr(op_attr::exclude_pad, exclude_pad);
    avg_op.set_attr(op_attr::kernel, kernel);
    avg_op.set_attr(op_attr::auto_pad, auto_pad);
    avg_op.set_attr(op_attr::data_format, data_format);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_data};
    logical_tensor_t lt_o
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out {&lt_o};

    // if output shape is unknown, infer output shape
    avg_op_schema->shape_infer(&avg_op, lt_in, lt_out);
    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    const std::vector<int64_t> inferred_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(inferred_out_strides, expected_out_strides);

    // if output shape is known, infer auto pad
    avg_op_schema->shape_infer(&avg_op, lt_in, lt_out);
    auto inferred_pads_begin
            = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto inferred_pads_end
            = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    const std::vector<int64_t> expected_pads_begin = {0, 0};
    const std::vector<int64_t> expected_pads_end = {1, 1};
    EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
    EXPECT_EQ(inferred_pads_end, expected_pads_end);
}

TEST(OpSchema, InferAvgpoolAutoPadShape) {
    const op_schema_t *avg_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::AvgPool);
    op_t avg_op {op_kind::AvgPool, op_t::kind2str(op_kind::AvgPool)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    bool exclude_pad = false;
    std::string data_format = "NCX";

    avg_op.set_attr(op_attr::strides, strides);
    avg_op.set_attr(op_attr::pads_begin, pads_begin);
    avg_op.set_attr(op_attr::pads_end, pads_end);
    avg_op.set_attr(op_attr::exclude_pad, exclude_pad);
    avg_op.set_attr(op_attr::kernel, kernel);
    avg_op.set_attr(op_attr::data_format, data_format);

    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        avg_op.set_attr(op_attr::auto_pad, auto_pad);
        logical_tensor_t lt_data
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        std::vector<logical_tensor_t *> lt_in {&lt_data};
        logical_tensor_t lt_o = logical_tensor_init(
                id++, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        // if output shape is unknown, infer output shape
        avg_op_schema->shape_infer(&avg_op, lt_in, lt_out);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();

        auto inferred_pads_begin
                = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        auto inferred_pads_end
                = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);

        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {2, 2};
            const std::vector<int64_t> expected_pads_end = {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);

        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_out_shape = {1, 1, 3, 3};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> expected_pads_begin = {0, 0};
            const std::vector<int64_t> expected_pads_end = {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferAvgpoolOutputShapeWithRoundingType) {
    const op_schema_t *avg_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::AvgPool);

    std::set<std::string> rounding_types = {"ceil", "floor"};
    for (auto &rounding_type : rounding_types) {
        op_t avg_op {op_kind::AvgPool, op_t::kind2str(op_kind::AvgPool)};
        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> kernel = {3, 3};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        bool exclude_pad = false;
        std::string data_format = "NCX";
        avg_op.set_attr(op_attr::strides, strides);
        avg_op.set_attr(op_attr::pads_begin, pads_begin);
        avg_op.set_attr(op_attr::pads_end, pads_end);
        avg_op.set_attr(op_attr::kernel, kernel);
        avg_op.set_attr(op_attr::exclude_pad, exclude_pad);
        avg_op.set_attr(op_attr::data_format, data_format);
        avg_op.set_attr(op_attr::rounding_type, rounding_type);

        logical_tensor_t lt_data
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        std::vector<logical_tensor_t *> lt_in {&lt_data};
        logical_tensor_t lt_o
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_out {&lt_o};

        // if output shape is unknown, infer output shape
        avg_op_schema->shape_infer(&avg_op, lt_in, lt_out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();
        const std::vector<int64_t> inferred_out_strides
                = logical_tensor_wrapper_t(lt_o).vstrides();
        if (rounding_type == "ceil") {
            const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(inferred_out_strides, expected_out_strides);
        } else { // rounding_type = floor
            const std::vector<int64_t> expected_out_shape = {1, 3, 111, 111};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(inferred_out_strides, expected_out_strides);
        }
    }
}

TEST(OpSchema, AvgPoolBackward) {
    const op_kind_t op_kind_ = op_kind::AvgPoolBackward;

    const std::set<size_t> expected_in_sizes = {2};
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::strides, true}, {op_attr::pads_begin, true},
                    {op_attr::pads_end, true}, {op_attr::kernel, true},
                    {op_attr::auto_pad, false}, {op_attr::exclude_pad, true},
                    {op_attr::data_format, false}, {op_attr::src_shape, false}};
    for (auto expected_in_size : expected_in_sizes) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, AvgPoolBackwardAutoPadShape) {
    const op_schema_t *avg_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::AvgPoolBackward);
    op_t avg_op {
            op_kind::AvgPoolBackward, op_t::kind2str(op_kind::AvgPoolBackward)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    bool exclude_pad = false;
    std::string data_format = "NCX";

    avg_op.set_attr(op_attr::strides, strides);
    avg_op.set_attr(op_attr::pads_begin, pads_begin);
    avg_op.set_attr(op_attr::pads_end, pads_end);
    avg_op.set_attr(op_attr::kernel, kernel);
    avg_op.set_attr(op_attr::exclude_pad, exclude_pad);
    avg_op.set_attr(op_attr::data_format, data_format);
    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        std::vector<int64_t> in_shape {1, 1, 6, 6};
        avg_op.set_attr(op_attr::auto_pad, auto_pad);
        avg_op.set_attr(op_attr::src_shape, in_shape);

        logical_tensor_t lt_output_delta
                = logical_tensor_init(id++, in_shape, data_type::f32);
        if (auto_pad == "VALID") {
            lt_output_delta
                    = logical_tensor_init(id++, {1, 1, 3, 3}, data_type::f32);
        }
        std::vector<logical_tensor_t *> in {&lt_output_delta};
        logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        // if output shape is unknown, infer output shape
        avg_op_schema->shape_infer(&avg_op, in, out);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();

        auto inferred_pads_begin
                = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        auto inferred_pads_end
                = avg_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
        EXPECT_EQ(inferred_out_shape, expected_out_shape);

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_pads_begin = {2, 2};
            const std::vector<int64_t> expected_pads_end = {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_pads_begin = {0, 0};
            const std::vector<int64_t> expected_pads_end = {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, Divide) {
    const op_kind_t op_kind_ = op_kind::Divide;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferDivideOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Divide;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferDivideOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Divide;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, Elu) {
    const op_kind_t op_kind_ = op_kind::Elu;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::alpha, true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferEluOutputShape) {
    const op_kind_t op_kind_ = op_kind::Elu;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, EluBackward) {
    const op_kind_t op_kind_ = op_kind::EluBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::alpha, true}, {op_attr::use_dst, false}};
    const size_t expected_attr_size = attrs_data.size();

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferEluBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::EluBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, End) {
    const op_schema_t *op_schema
            = op_schema_registry_t::get_op_schema(op_kind::End);

    op_t end_op {0, op_kind::End, "end"};
    logical_tensor_t lt_in_0 = logical_tensor_init(0, data_type::f32);

    end_op.add_input(lt_in_0);
    EXPECT_TRUE(op_schema->verify(&end_op));
}

TEST(OpSchema, Reorder) {
    const op_kind_t op_kind_ = op_kind::Reorder;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferReorderOutputShape) {
    const op_kind_t op_kind_ = op_kind::Reorder;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Exp) {
    const op_kind_t op_kind_ = op_kind::Exp;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferExpOutputShape) {
    const op_kind_t op_kind_ = op_kind::Exp;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Gelu) {
    const op_kind_t op_kind_ = op_kind::GELU;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferGeluOutputShape) {
    const op_kind_t op_kind_ = op_kind::GELU;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, GELUBackward) {
    const op_kind_t op_kind_ = op_kind::GELUBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferGeluBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::GELUBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, HardSwish) {
    const op_kind_t op_kind_ = op_kind::HardSwish;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferHardSwishOutputShape) {
    const op_kind_t op_kind_ = op_kind::HardSwish;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, HardSwishBackward) {
    const op_kind_t op_kind_ = op_kind::HardSwishBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferHardSwishBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::HardSwishBackward;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, LayerNorm) {
    const op_kind_t op_kind_ = op_kind::LayerNorm;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 4;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::keep_stats, false}, {op_attr::begin_norm_axis, false},
                    {op_attr::use_affine, false}, {op_attr::epsilon, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferLayerNormOutputShape) {
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind::LayerNorm);
    op_t op_ {op_kind::LayerNorm, op_t::kind2str(op_kind::LayerNorm)};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    // We test all available cases
    const std::vector<bool> keep_statses = {true, false};

    // TODO(qun) we should test multi begin_norm_axis attrs
    const int64_t begin_norm_axis = -1;

    op_.set_attr(op_attr::begin_norm_axis, begin_norm_axis);

    for (auto keep_stats : keep_statses) {
        op_.set_attr(op_attr::keep_stats, static_cast<bool>(keep_stats));
        for (const auto &ltype : layout_types) {
            logical_tensor_t lt_in1 = logical_tensor_init(
                    0, {1, 3, 416, 416}, data_type::f32, ltype);
            logical_tensor_t lt_out = logical_tensor_init(
                    1, data_type::f32, layout_type::strided);
            logical_tensor_t lt_mean = logical_tensor_init(
                    2, data_type::f32, layout_type::strided);
            logical_tensor_t lt_var = logical_tensor_init(
                    3, data_type::f32, layout_type::strided);

            std::vector<logical_tensor_t *> in {&lt_in1};
            std::vector<logical_tensor_t *> out {&lt_out, &lt_mean, &lt_var};

            op_schema_->shape_infer(&op_, in, out);

            const std::vector<int64_t> inferred_out_shape
                    = logical_tensor_wrapper_t(lt_out).vdims();
            const std::vector<int64_t> expected_out_shape = {1, 3, 416, 416};
            EXPECT_EQ(inferred_out_shape, expected_out_shape);

            const std::vector<int64_t> inferred_out_strides
                    = logical_tensor_wrapper_t(lt_out).vstrides();
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(inferred_out_strides, expected_out_strides);

            if (keep_stats) {
                // check inferred shape and strides for mean
                const std::vector<int64_t> inferred_mean_shape
                        = logical_tensor_wrapper_t(lt_mean).vdims();
                const std::vector<int64_t> expected_mean_shape = {1, 3, 416};
                EXPECT_EQ(inferred_mean_shape, expected_mean_shape);

                const std::vector<int64_t> inferred_mean_strides
                        = logical_tensor_wrapper_t(lt_mean).vstrides();
                const std::vector<int64_t> expected_mean_strides
                        = compute_dense_strides(expected_mean_shape);
                EXPECT_EQ(inferred_mean_strides, expected_mean_strides);

                // check inferred shape and strides for var
                const std::vector<int64_t> inferred_var_shape
                        = logical_tensor_wrapper_t(lt_var).vdims();
                const std::vector<int64_t> expected_var_shape = {1, 3, 416};
                EXPECT_EQ(inferred_var_shape, expected_var_shape);

                const std::vector<int64_t> inferred_var_strides
                        = logical_tensor_wrapper_t(lt_var).vstrides();
                const std::vector<int64_t> expected_var_strides
                        = compute_dense_strides(expected_var_shape);
                EXPECT_EQ(inferred_var_strides, expected_var_strides);
            }
        }
    }
}

TEST(OpSchema, LayerNormBackward) {
    const op_kind_t op_kind_ = op_kind::LayerNormBackward;
    const size_t expected_in_size = 6;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 3;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::begin_norm_axis, false}, {op_attr::use_affine, false},
                    {op_attr::epsilon, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferLayerNormBackwardOutputShape) {
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind::LayerNormBackward);
    op_t op_ {op_kind::LayerNormBackward,
            op_t::kind2str(op_kind::LayerNormBackward)};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    // We test all available cases
    const std::vector<bool> use_affines = {true, false};

    for (auto use_affine : use_affines) {
        op_.set_attr(op_attr::use_affine, static_cast<bool>(use_affine));
        for (const auto &ltype : layout_types) {
            logical_tensor_t lt_data = logical_tensor_init(
                    0, {1, 256, 64, 64}, data_type::f32, ltype);
            logical_tensor_t lt_out_delta = logical_tensor_init(
                    1, {1, 256, 64, 64}, data_type::f32, ltype);
            logical_tensor_t lt_mean
                    = logical_tensor_init(2, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_variance
                    = logical_tensor_init(3, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_gamma
                    = logical_tensor_init(4, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_beta
                    = logical_tensor_init(5, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_in_delta = logical_tensor_init(
                    6, data_type::f32, layout_type::strided);
            logical_tensor_t lt_gamma_delta = logical_tensor_init(
                    7, data_type::f32, layout_type::strided);
            logical_tensor_t lt_beta_delta = logical_tensor_init(
                    8, data_type::f32, layout_type::strided);
            std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_out_delta,
                    &lt_mean, &lt_variance, &lt_gamma, &lt_beta};
            std::vector<logical_tensor_t *> lt_out {
                    &lt_in_delta, &lt_gamma_delta, &lt_beta_delta};

            op_schema_->shape_infer(&op_, lt_in, lt_out);

            const std::vector<int64_t> inferred_in_delta_shape
                    = logical_tensor_wrapper_t(lt_in_delta).vdims();
            const std::vector<int64_t> expected_in_delta_shape
                    = {1, 256, 64, 64};
            EXPECT_EQ(inferred_in_delta_shape, expected_in_delta_shape);

            const std::vector<int64_t> inferred_in_delta_strides
                    = logical_tensor_wrapper_t(lt_in_delta).vstrides();
            const std::vector<int64_t> expected_in_delta_strides
                    = compute_dense_strides(expected_in_delta_shape);
            EXPECT_EQ(inferred_in_delta_strides, expected_in_delta_strides);

            if (use_affine) {
                const std::vector<int64_t> inferred_gamma_delta_shape
                        = logical_tensor_wrapper_t(lt_gamma_delta).vdims();
                const std::vector<int64_t> expected_gamma_delta_shape
                        = {1, 256};
                EXPECT_EQ(
                        inferred_gamma_delta_shape, expected_gamma_delta_shape);

                const std::vector<int64_t> inferred_gamma_delta_strides
                        = logical_tensor_wrapper_t(lt_gamma_delta).vstrides();
                const std::vector<int64_t> expected_gamma_delta_strides
                        = compute_dense_strides(expected_gamma_delta_shape);
                EXPECT_EQ(inferred_gamma_delta_strides,
                        expected_gamma_delta_strides);

                const std::vector<int64_t> inferred_beta_delta_shape
                        = logical_tensor_wrapper_t(lt_beta_delta).vdims();
                const std::vector<int64_t> expected_beta_delta_shape = {1, 256};
                EXPECT_EQ(inferred_beta_delta_shape, expected_beta_delta_shape);

                const std::vector<int64_t> inferred_beta_delta_strides
                        = logical_tensor_wrapper_t(lt_beta_delta).vstrides();
                const std::vector<int64_t> expected_beta_delta_strides
                        = compute_dense_strides(expected_beta_delta_shape);
                EXPECT_EQ(inferred_beta_delta_strides,
                        expected_beta_delta_strides);
            }
        }
    }
}

TEST(OpSchema, Log) {
    const op_kind_t op_kind_ = op_kind::Log;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferLogOutputShape) {
    const op_kind_t op_kind_ = op_kind::Log;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, LogSoftmax) {
    const op_kind_t op_kind_ = op_kind::LogSoftmax;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::axis, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferLogSoftmaxOutputShape) {
    const op_kind_t op_kind_ = op_kind::LogSoftmax;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, LogSoftmaxBackward) {
    const op_kind_t op_kind_ = op_kind::LogSoftmaxBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::axis, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferLogSoftmaxBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::LogSoftmaxBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Maximum) {
    const op_kind_t op_kind_ = op_kind::Maximum;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMaximumOutputShapeWithoutBroadcst) {
    const op_kind_t op_kind_ = op_kind::Maximum;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferMaximumOutputShapeWihBroadcast) {
    const op_kind_t op_kind_ = op_kind::Maximum;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, MaxPoolBackward) {
    const op_kind_t op_kind_ = op_kind::MaxPoolBackward;

    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 7;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::strides, true},
            {op_attr::pads_begin, true}, {op_attr::pads_end, true},
            {op_attr::kernel, true}, {op_attr::auto_pad, false},
            {op_attr::dilations, false}, {op_attr::data_format, false}};
    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMaxPoolBackwardAutoPadShape) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPoolBackward);
    op_t pool_op {
            op_kind::MaxPoolBackward, op_t::kind2str(op_kind::MaxPoolBackward)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";

    pool_op.set_attr(op_attr::strides, strides);
    pool_op.set_attr(op_attr::pads_begin, pads_begin);
    pool_op.set_attr(op_attr::pads_end, pads_end);
    pool_op.set_attr(op_attr::kernel, kernel);
    pool_op.set_attr(op_attr::dilations, dilations);

    pool_op.set_attr(op_attr::data_format, data_format);
    std::vector<std::string> auto_pads_vec {
            "NONE", "SAME_UPPER", "SAME_LOWER", "VALID"};
    int64_t id = 0;

    for (const auto &auto_pad : auto_pads_vec) {
        pool_op.set_attr(op_attr::auto_pad, auto_pad);
        logical_tensor_t lt_data
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);

        logical_tensor_t lt_output_delta
                = logical_tensor_init(id++, {1, 1, 6, 6}, data_type::f32);
        if (auto_pad == "VALID") {
            lt_output_delta
                    = logical_tensor_init(id++, {1, 1, 3, 3}, data_type::f32);
        }
        std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
        logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        // if output shape is unknown, infer output shape
        pool_op_schema->shape_infer(&pool_op, in, out);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();

        auto inferred_pads_begin
                = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
        auto inferred_pads_end
                = pool_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
        const std::vector<int64_t> expected_out_shape = {1, 1, 6, 6};
        EXPECT_EQ(inferred_out_shape, expected_out_shape);

        if (auto_pad == "NONE") {
            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_UPPER") {
            const std::vector<int64_t> expected_pads_begin = {1, 1};
            const std::vector<int64_t> expected_pads_end = {2, 2};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "SAME_LOWER") {
            const std::vector<int64_t> expected_pads_begin = {2, 2};
            const std::vector<int64_t> expected_pads_end = {1, 1};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        } else if (auto_pad == "VALID") {
            const std::vector<int64_t> expected_pads_begin = {0, 0};
            const std::vector<int64_t> expected_pads_end = {0, 0};
            EXPECT_EQ(inferred_pads_begin, expected_pads_begin);
            EXPECT_EQ(inferred_pads_end, expected_pads_end);
        }
    }
}

TEST(OpSchema, InferMaxPoolBackwardDilationsShape) {
    const op_schema_t *pool_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MaxPoolBackward);
    op_t pool_op {
            op_kind::MaxPoolBackward, op_t::kind2str(op_kind::MaxPoolBackward)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> kernel = {4, 4};
    std::vector<int64_t> pads_begin = {0, 0};
    std::vector<int64_t> pads_end = {0, 0};
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    const std::vector<int64_t> expected_out_shape = {1, 1, 16, 16};
    int64_t id = 0;

    pool_op.set_attr(op_attr::strides, strides);
    pool_op.set_attr(op_attr::pads_begin, pads_begin);
    pool_op.set_attr(op_attr::pads_end, pads_end);
    pool_op.set_attr(op_attr::kernel, kernel);
    pool_op.set_attr(op_attr::dilations, dilations);

    pool_op.set_attr(op_attr::data_format, data_format);
    pool_op.set_attr<std::string>(op_attr::auto_pad, "VALID");

    logical_tensor_t lt_data
            = logical_tensor_init(id++, {1, 1, 16, 16}, data_type::f32);

    logical_tensor_t lt_output_delta
            = logical_tensor_init(id++, {1, 1, 10, 10}, data_type::f32);

    std::vector<logical_tensor_t *> in {&lt_data, &lt_output_delta};
    logical_tensor_t lt_out = logical_tensor_init(id++, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // if output shape is unknown, infer output shape
    pool_op_schema->shape_infer(&pool_op, in, out);

    const std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, Minimum) {
    const op_kind_t op_kind_ = op_kind::Minimum;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMinimumOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Minimum;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferMinimumOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Minimum;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, Mish) {
    const op_kind_t op_kind_ = op_kind::Mish;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMishOutputShape) {
    const op_kind_t op_kind_ = op_kind::Mish;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, MishBackward) {
    const op_kind_t op_kind_ = op_kind::MishBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMishBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::MishBackward;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Multiply) {
    const op_kind_t op_kind_ = op_kind::Multiply;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferMultiplyOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Multiply;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferMultiplyOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Multiply;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, PReLU) {
    const op_kind_t op_kind_ = op_kind::PReLU;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::data_format, false},
            {op_attr::per_channel_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, PReLUOutputShape) {
    const op_kind_t op_kind_ = op_kind::PReLU;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Reduce) {
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::keep_dims, false}, {op_attr::axes, false}};
    const std::vector<op_kind_t> configs {op_kind::ReduceL1, op_kind::ReduceL2,
            op_kind::ReduceMax, op_kind::ReduceMean, op_kind::ReduceMin,
            op_kind::ReduceProd, op_kind::ReduceSum};

    for (auto op_kind_ : configs) {
        verify_op_schema(op_kind_, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferReduceOutputShapeFromInput) {
    const std::vector<op_kind_t> configs {op_kind::ReduceL1, op_kind::ReduceL2,
            op_kind::ReduceMax, op_kind::ReduceMean, op_kind::ReduceMin,
            op_kind::ReduceProd, op_kind::ReduceSum};

    for (auto op_kind_ : configs) {
        const op_schema_t *op_schema_
                = op_schema_registry_t::get_op_schema(op_kind_);
        op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

        logical_tensor_t lt_in
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        logical_tensor_t lt_axis_indices
                = logical_tensor_init(1, data_type::f32);
        std::vector<logical_tensor_t *> in {&lt_in, &lt_axis_indices};
        logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        auto ret = op_schema_->shape_infer(&op_, in, out);
        EXPECT_EQ(ret, status::unimplemented);
    }
}

TEST(OpSchema, InferReduceOutputShapeFromAttributeWithKeepingDims) {
    const std::vector<op_kind_t> configs {op_kind::ReduceL1, op_kind::ReduceL2,
            op_kind::ReduceMax, op_kind::ReduceMean, op_kind::ReduceMin,
            op_kind::ReduceProd, op_kind::ReduceSum};
    const std::vector<int64_t> axes {2};
    const std::vector<int64_t> expected_out_shape = {1, 3, 1, 224};

    for (auto op_kind_ : configs) {
        const op_schema_t *op_schema_
                = op_schema_registry_t::get_op_schema(op_kind_);
        op_t op_ {op_kind_, op_t::kind2str(op_kind_)};
        op_.set_attr(op_attr::keep_dims, true);
        op_.set_attr(op_attr::axes, axes);

        logical_tensor_t lt_in
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        logical_tensor_t lt_axis_indices
                = logical_tensor_init(1, data_type::f32);
        std::vector<logical_tensor_t *> in {&lt_in, &lt_axis_indices};
        logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        op_schema_->shape_infer(&op_, in, out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferReduceOutputShapeFromAttributeWithoutKeepingDims) {
    const std::vector<op_kind_t> configs {op_kind::ReduceL1, op_kind::ReduceL2,
            op_kind::ReduceMax, op_kind::ReduceMean, op_kind::ReduceMin,
            op_kind::ReduceProd, op_kind::ReduceSum};
    const std::vector<int64_t> axes {0, 1, 2, 3};
    const std::vector<int64_t> expected_out_shape = {};

    for (auto op_kind_ : configs) {
        const op_schema_t *op_schema_
                = op_schema_registry_t::get_op_schema(op_kind_);
        op_t op_ {op_kind_, op_t::kind2str(op_kind_)};
        op_.set_attr(op_attr::axes, axes);

        logical_tensor_t lt_in
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        logical_tensor_t lt_axis_indices
                = logical_tensor_init(1, data_type::f32);
        std::vector<logical_tensor_t *> in {&lt_in, &lt_axis_indices};
        logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        op_schema_->shape_infer(&op_, in, out);
        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
        lt_out = logical_tensor_init(2, {}, data_type::f32);
        EXPECT_EQ(
                op_schema_->shape_infer(&op_, in, out), graph::status::success);
    }
}

TEST(OpSchema, ReLU) {
    const op_kind_t op_kind_ = op_kind::ReLU;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferReluOutputShape) {
    const op_kind_t op_kind_ = op_kind::ReLU;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, ReLUBackward) {
    const op_kind_t op_kind_ = op_kind::ReLUBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::use_dst, false}};
    const size_t expected_attr_size = attrs_data.size();

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferReluBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::ReLUBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Round) {
    const op_kind_t op_kind_ = op_kind::Round;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferRoundOutputShape) {
    const op_kind_t op_kind_ = op_kind::Round;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Select) {
    const op_kind_t op_kind_ = op_kind::Select;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSelectOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Select;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};
    const std::string no_broadcast_attr_val = "none";
    op_.set_attr(op_attr::auto_broadcast, no_broadcast_attr_val);

    // positive case
    logical_tensor_t lt_in0
            = logical_tensor_init(0, {3, 3, 64, 128}, data_type::boolean);
    logical_tensor_t lt_in1
            = logical_tensor_init(1, {3, 3, 64, 128}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(2, {3, 3, 64, 128}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in0, &lt_in1, &lt_in2};
    logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    status_t ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_EQ(ret, status::success);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {3, 3, 64, 128};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    // negative case
    logical_tensor_t lt_in2_neg
            = logical_tensor_init(4, {3, 3, 64, 64}, data_type::f32);
    std::vector<logical_tensor_t *> in_neg {&lt_in0, &lt_in1, &lt_in2_neg};
    logical_tensor_t lt_out_neg = logical_tensor_init(5, data_type::f32);
    std::vector<logical_tensor_t *> out_neg {&lt_out_neg};
    ret = op_schema_->shape_infer(&op_, in_neg, out_neg);
    EXPECT_EQ(ret, status::invalid_shape);
}

#define for_ for
TEST(OpSchema, InferSelectOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Select;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};
    const std::vector<std::vector<int64_t>> in0_shapes
            = {{2, 3, 64, 64}, {1, 64}, {3, 1, 64}, {1}};
    const std::vector<std::vector<int64_t>> in1_shapes
            = {{2, 1, 64, 1}, {2, 3, 64, 64}};
    const std::vector<std::vector<int64_t>> in2_shapes
            = {{3, 1, 64}, {2, 3, 64, 64}};
    const std::vector<std::vector<int64_t>> expected_out_shapes
            = {{2, 3, 64, 64}};

    for_(const auto &in0_shape : in0_shapes)
    for_(const auto &in1_shape : in1_shapes)
    for_(const auto &in2_shape : in2_shapes)
    for (const auto &expected_out_shape : expected_out_shapes) {
        logical_tensor_t lt_in0
                = logical_tensor_init(0, in0_shape, data_type::boolean);
        logical_tensor_t lt_in1
                = logical_tensor_init(1, in1_shape, data_type::f32);
        logical_tensor_t lt_in2
                = logical_tensor_init(2, in2_shape, data_type::f32);
        std::vector<logical_tensor_t *> in {&lt_in0, &lt_in1, &lt_in2};
        logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
        std::vector<logical_tensor_t *> out {&lt_out};

        // shape inference without explicitly setting auto_broadcast
        // should be enabled by default
        op_schema_->shape_infer(&op_, in, out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        // explicitly setting auto_broadcast
        const std::string with_broadcast_attr_val = "numpy";
        op_.set_attr(op_attr::auto_broadcast, with_broadcast_attr_val);
        logical_tensor_t lt_out_expl = logical_tensor_init(4, data_type::f32);
        std::vector<logical_tensor_t *> out_expl {&lt_out_expl};

        op_schema_->shape_infer(&op_, in, out_expl);
        const std::vector<int64_t> infered_out_shape_expl
                = logical_tensor_wrapper_t(lt_out_expl).vdims();
        EXPECT_EQ(infered_out_shape_expl, expected_out_shape);
    }
}
#undef for_

TEST(OpSchema, Sigmoid) {
    const op_kind_t op_kind_ = op_kind::Sigmoid;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSigmoidOutputShape) {
    const op_kind_t op_kind_ = op_kind::Sigmoid;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, SigmoidBackward) {
    const op_kind_t op_kind_ = op_kind::SigmoidBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::use_dst, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSigmoidBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::SigmoidBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, SoftMax) {
    const op_kind_t op_kind_ = op_kind::SoftMax;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::axis, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSoftMaxOutputShape) {
    const op_kind_t op_kind_ = op_kind::SoftMax;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, SoftMaxBackward) {
    const op_kind_t op_kind_ = op_kind::SoftMaxBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::axis, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSoftMaxBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::SoftMaxBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, InferSqrtOutputShape) {
    const op_kind_t op_kind_ = op_kind::Sqrt;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Sqrt) {
    const op_kind_t op_kind_ = op_kind::Sqrt;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, SoftPlus) {
    const op_kind_t op_kind_ = op_kind::SoftPlus;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::beta, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSoftPlusOutputShape) {
    const op_kind_t op_kind_ = op_kind::SoftPlus;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, SoftPlusBackward) {
    const op_kind_t op_kind_ = op_kind::SoftPlusBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::beta, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSoftPlusBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::SoftPlusBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, SqrtBackward) {
    const op_kind_t op_kind_ = op_kind::SqrtBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::use_dst, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, SquaredDifference) {
    const op_kind_t op_kind_ = op_kind::SquaredDifference;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, Subtract) {
    const op_kind_t op_kind_ = op_kind::Subtract;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSubtractOutputShapeWithoutBroadcast) {
    const op_kind_t op_kind_ = op_kind::Subtract;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(OpSchema, InferSubtractOutputShapeWithBroadcast) {
    const op_kind_t op_kind_ = op_kind::Subtract;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(OpSchema, Tanh) {
    const op_kind_t op_kind_ = op_kind::Tanh;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}
TEST(OpSchema, InferSqrtBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::SqrtBackward;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Square) {
    const op_kind_t op_kind_ = op_kind::Square;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<op_attr_t, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferSquareOutputShape) {
    const op_kind_t op_kind_ = op_kind::Square;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, InferTanhOutputShape) {
    const op_kind_t op_kind_ = op_kind::Tanh;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, TanhBackward) {
    const op_kind_t op_kind_ = op_kind::TanhBackward;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::use_dst, false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferTanhBackwardOutputShape) {
    const op_kind_t op_kind_ = op_kind::TanhBackward;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(OpSchema, Wildcard) {
    const op_schema_t *op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Wildcard);
    auto inputs_option = op_schema->get_inputs_option();
    auto outputs_option = op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema_t::param_num_option::variadic);
    EXPECT_TRUE(outputs_option == op_schema_t::param_num_option::variadic);

    op_t wildcard_op {0, op_kind::Wildcard, std::string("wildcard")};
    logical_tensor_t lt_in_0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_in_1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out_0 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_out_1 = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_out_2 = logical_tensor_init(4, data_type::f32);

    wildcard_op.add_input(lt_in_0);
    wildcard_op.add_input(lt_in_1);
    wildcard_op.add_output(lt_out_0);
    wildcard_op.add_output(lt_out_1);
    wildcard_op.add_output(lt_out_2);

    EXPECT_TRUE(op_schema->verify(&wildcard_op));
}

TEST(OpSchema, InferWildcardOutputShape) {
    const op_kind_t op_kind_ = op_kind::Wildcard;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    logical_tensor_t lt_in = logical_tensor_init(0, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    auto ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_EQ(ret, status::unimplemented);
}

TEST(OpSchema, BatchNormOptionalInput) {
    const op_schema_t *bn_op_schema = op_schema_registry_t::get_op_schema(
            op_kind::BatchNormForwardTraining);
    auto inputs_option = bn_op_schema->get_inputs_option();
    auto outputs_option = bn_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema_t::param_num_option::optional);
    EXPECT_TRUE(outputs_option == op_schema_t::param_num_option::fixed);

    op_t bn_op {0, op_kind::BatchNormForwardTraining, std::string("bn")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_viance = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_running_mean = logical_tensor_init(4, data_type::f32);
    logical_tensor_t lt_running_viance = logical_tensor_init(5, data_type::f32);
    logical_tensor_t lt_batch_mean = logical_tensor_init(6, data_type::f32);
    logical_tensor_t lt_batch_viance = logical_tensor_init(7, data_type::f32);

    bn_op.add_input(lt_data);
    bn_op.add_input(lt_mean);
    bn_op.add_input(lt_viance);
    bn_op.add_output(lt_output);
    bn_op.add_output(lt_running_mean);
    bn_op.add_output(lt_running_viance);
    bn_op.add_output(lt_batch_mean);
    bn_op.add_output(lt_batch_viance);

    bn_op.set_attr<float>(op_attr::epsilon, 0.001f);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));

    logical_tensor_t lt_gamma = logical_tensor_init(8, data_type::f32);
    bn_op.add_input(lt_gamma);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));
    logical_tensor_t lt_beta = logical_tensor_init(9, data_type::f32);
    bn_op.add_input(lt_beta);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));

    logical_tensor_t lt_false = logical_tensor_init(10, data_type::f32);
    bn_op.add_input(lt_false);
    EXPECT_FALSE(bn_op_schema->verify(&bn_op));
}

TEST(OpSchema, ConcatVariadicInput) {
    const op_schema_t *concat_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Concat);
    auto inputs_option = concat_op_schema->get_inputs_option();
    auto outputs_option = concat_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema_t::param_num_option::variadic);
    EXPECT_TRUE(outputs_option == op_schema_t::param_num_option::fixed);

    op_t concat_op {0, op_kind::Concat, std::string("concat")};
    logical_tensor_t lt_data_0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_data_2 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);

    concat_op.add_input(lt_data_0);
    concat_op.add_input(lt_data_1);
    concat_op.add_input(lt_data_2);
    concat_op.add_output(lt_output);

    concat_op.set_attr(op_attr::axis, int64_t(0));
    EXPECT_TRUE(concat_op_schema->verify(&concat_op));
}

TEST(OpSchema, ConcatVariadicInputNegative) {
    const op_schema_t *concat_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Concat);

    op_t concat_op {0, op_kind::Concat, std::string("concat")};
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);

    concat_op.add_output(lt_output);

    concat_op.set_attr(op_attr::axis, int64_t(0));
    EXPECT_FALSE(concat_op_schema->verify(&concat_op));
}

TEST(OpSchema, LayerNormOptionalInputs) {
    const op_schema_t *ln_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::LayerNorm);
    auto inputs_option = ln_op_schema->get_inputs_option();
    auto outputs_option = ln_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema_t::param_num_option::optional);
    EXPECT_TRUE(outputs_option == op_schema_t::param_num_option::optional);

    op_t ln_op {0, op_kind::LayerNorm, std::string("ln")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);

    logical_tensor_t lt_output = logical_tensor_init(1, data_type::f32);

    ln_op.add_input(lt_data);

    ln_op.add_output(lt_output);

    // Since number of output is only 1,
    // which doesn't fit requirement when keep_stats is true
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::f32);
    ln_op.add_input(lt_beta);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_gamma = logical_tensor_init(3, data_type::f32);
    ln_op.add_input(lt_gamma);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_mean = logical_tensor_init(4, data_type::f32);
    ln_op.add_output(lt_mean);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_variance = logical_tensor_init(5, data_type::f32);
    ln_op.add_output(lt_variance);
    EXPECT_TRUE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_false = logical_tensor_init(6, data_type::f32);
    ln_op.add_input(lt_false);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));
}

TEST(OpSchema, AddDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Add;
    op_t tmp_op {0, tmp_op_kind, std::string("add")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, AvgpoolDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::AvgPool;
    op_t tmp_op {0, tmp_op_kind, std::string("avgpool")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::rounding_type, &sval);
    EXPECT_EQ(*sval, "floor");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");

    tmp_op.get_attr<std::string>(op_attr::exclude_pad, &sval);
    EXPECT_EQ(*sval, "None");

    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    tmp_op.add_input(lt_data);
    tmp_op.add_output(lt_out);
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> kernel = {3, 3};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    bool exclude_pad = false;

    tmp_op.set_attr(op_attr::strides, strides);
    tmp_op.set_attr(op_attr::pads_begin, pads_begin);
    tmp_op.set_attr(op_attr::pads_end, pads_end);
    tmp_op.set_attr(op_attr::kernel, kernel);
    tmp_op.set_attr(op_attr::exclude_pad, exclude_pad);

    EXPECT_TRUE(opm->verify(&tmp_op));
}

TEST(OpSchema, AvgBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::AvgPoolBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("avgpool_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");
}

TEST(OpSchema, BatchNormInferenceDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::BatchNormInference;
    op_t tmp_op {0, tmp_op_kind, std::string("bn_inference")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, BatchNormForwardTrainingDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::BatchNormForwardTraining;
    op_t tmp_op {0, tmp_op_kind, std::string("bn_fwd_training")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, BatchNormTrainingBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::BatchNormTrainingBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("bn_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, BiasaddDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::BiasAdd;
    op_t tmp_op {0, tmp_op_kind, std::string("bias_add")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, BiasaddBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::BiasAddBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("bias_add_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, ConvolutionDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Convolution;
    op_t tmp_op {0, tmp_op_kind, std::string("conv")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::weights_format, &sval);
    EXPECT_EQ(*sval, "XIO");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::groups, &ival);
    int64_t int_value {1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, ConvolutionBackwardDataDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::ConvolutionBackwardData;
    op_t tmp_op {0, tmp_op_kind, std::string("conv_bpd")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::vector<int64_t> *vval {nullptr};
    tmp_op.get_attr<std::vector<int64_t>>(op_attr::output_padding, &vval);
    std::vector<int64_t> vector_value(DNNL_MAX_NDIMS, 0);
    EXPECT_EQ(*vval, vector_value);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::weights_format, &sval);
    EXPECT_EQ(*sval, "XIO");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::groups, &ival);
    int64_t int_value {1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, ConvolutionBackwardWeightsDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::ConvolutionBackwardWeights;
    op_t tmp_op {0, tmp_op_kind, std::string("conv_bpf")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::weights_format, &sval);
    EXPECT_EQ(*sval, "XIO");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::groups, &ival);
    int64_t int_value {1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, DivideDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Divide;
    op_t tmp_op {0, tmp_op_kind, std::string("divide")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, InterpolateDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Interpolate;
    op_t tmp_op {0, tmp_op_kind, std::string("interpolate")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(
            op_attr::coordinate_transformation_mode, &sval);
    EXPECT_EQ(*sval, "half_pixel");

    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, InterpolateBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::InterpolateBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("interpolate_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(
            op_attr::coordinate_transformation_mode, &sval);
    EXPECT_EQ(*sval, "half_pixel");

    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");
}

TEST(OpSchema, LayerNormDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::LayerNorm;
    op_t tmp_op {0, tmp_op_kind, std::string("ln")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::keep_stats, &bval);
    EXPECT_TRUE(bval);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::begin_norm_axis, &ival);
    int64_t int_value {-1};
    EXPECT_EQ(*ival, int_value);

    tmp_op.get_attr<bool>(op_attr::use_affine, &bval);
    EXPECT_TRUE(bval);

    const float *fval {nullptr};
    tmp_op.get_attr<float>(op_attr::epsilon, &fval);
    float float_value {1e-5f};
    EXPECT_FLOAT_EQ(*fval, float_value);
}

TEST(OpSchema, LayerNormBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::LayerNormBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("ln_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::use_affine, &bval);
    EXPECT_TRUE(bval);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::begin_norm_axis, &ival);
    int64_t int_value {-1};
    EXPECT_EQ(*ival, int_value);

    const float *fval {nullptr};
    tmp_op.get_attr<float>(op_attr::epsilon, &fval);
    float float_value {1e-5f};
    EXPECT_FLOAT_EQ(*fval, float_value);
}

TEST(OpSchema, LogSoftmaxDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::LogSoftmax;
    op_t tmp_op {0, tmp_op_kind, std::string("log_softmax")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::axis, &ival);
    int64_t int_value {-1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, LogSoftmaxBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::LogSoftmaxBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("logsoftmax_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::axis, &ival);
    int64_t int_value {-1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, MatmulDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::MatMul;
    op_t tmp_op {0, tmp_op_kind, std::string("matmul")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::transpose_a, &bval);
    EXPECT_FALSE(*bval);

    tmp_op.get_attr<bool>(op_attr::transpose_b, &bval);
    EXPECT_FALSE(*bval);
}

TEST(OpSchema, MaxPoolDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::MaxPool;
    op_t tmp_op {0, tmp_op_kind, std::string("max_pool")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    tmp_op.get_attr<std::string>(op_attr::rounding_type, &sval);
    EXPECT_EQ(*sval, "floor");

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");

    const std::vector<int64_t> *vval {nullptr};
    tmp_op.get_attr<std::vector<int64_t>>(op_attr::dilations, &vval);
    std::vector<int64_t> vector_value(DNNL_MAX_NDIMS, 1);
    EXPECT_EQ(*vval, vector_value);
}

TEST(OpSchema, MaxPoolBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::MaxPoolBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("max_pool_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::data_format, &sval);
    EXPECT_EQ(*sval, "NXC");

    const std::vector<int64_t> *vval {nullptr};
    tmp_op.get_attr<std::vector<int64_t>>(op_attr::dilations, &vval);
    std::vector<int64_t> vector_value(DNNL_MAX_NDIMS, 1);
    EXPECT_EQ(*vval, vector_value);

    tmp_op.get_attr<std::string>(op_attr::auto_pad, &sval);
    EXPECT_EQ(*sval, "None");
}

TEST(OpSchema, MaximumDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Maximum;
    op_t tmp_op {0, tmp_op_kind, std::string("max")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, MinimumDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Minimum;
    op_t tmp_op {0, tmp_op_kind, std::string("min")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, MultiplyDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Multiply;
    op_t tmp_op {0, tmp_op_kind, std::string("mul")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, ReduceDefaultAttribute) {
    const std::vector<op_kind_t> configs {op_kind::ReduceL1, op_kind::ReduceL2,
            op_kind::ReduceMax, op_kind::ReduceMean, op_kind::ReduceMin,
            op_kind::ReduceProd, op_kind::ReduceSum};

    for (auto tmp_op_kind : configs) {
        op_t tmp_op {0, tmp_op_kind, std::string("reduce")};

        const op_schema_t *opm
                = op_schema_registry_t::get_op_schema(tmp_op_kind);
        EXPECT_TRUE(opm != nullptr);
        opm->set_default_attribute(&tmp_op);

        const bool *bval {nullptr};
        tmp_op.get_attr<bool>(op_attr::keep_dims, &bval);
        EXPECT_FALSE(*bval);

        const std::vector<int64_t> *vval {nullptr};
        tmp_op.get_attr<std::vector<int64_t>>(op_attr::axes, &vval);
        EXPECT_TRUE(vval->empty());
    }
}

TEST(OpSchema, SigmoidBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SigmoidBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("sig_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::use_dst, &bval);
    EXPECT_TRUE(bval);
}

TEST(OpSchema, SoftMaxDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SoftMax;
    op_t tmp_op {0, tmp_op_kind, std::string("softmax")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::axis, &ival);
    int64_t int_value {1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, SoftMaxBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SoftMaxBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("softmax_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const int64_t *ival {nullptr};
    tmp_op.get_attr<int64_t>(op_attr::axis, &ival);
    int64_t int_value {1};
    EXPECT_EQ(*ival, int_value);
}

TEST(OpSchema, SoftPlusDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SoftPlus;
    op_t tmp_op {0, tmp_op_kind, std::string("softplus")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    float ival = tmp_op.get_attr<float>(op_attr::beta);
    EXPECT_EQ(ival, 1.f);
}

TEST(OpSchema, SoftPlusBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SoftPlusBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("softplus_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    float ival = tmp_op.get_attr<float>(op_attr::beta);
    EXPECT_EQ(ival, 1.f);
}

TEST(OpSchema, SqrtBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::SqrtBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("sqrt_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::use_dst, &bval);
    EXPECT_TRUE(bval);
}

TEST(OpSchema, SubtractDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::Subtract;
    op_t tmp_op {0, tmp_op_kind, std::string("sub")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const std::string *sval {nullptr};
    tmp_op.get_attr<std::string>(op_attr::auto_broadcast, &sval);
    EXPECT_EQ(*sval, "numpy");
}

TEST(OpSchema, TanhBackwardDefaultAttribute) {
    op_kind_t tmp_op_kind = op_kind::TanhBackward;
    op_t tmp_op {0, tmp_op_kind, std::string("tanh_bp")};

    const op_schema_t *opm = op_schema_registry_t::get_op_schema(tmp_op_kind);
    EXPECT_TRUE(opm != nullptr);
    opm->set_default_attribute(&tmp_op);

    const bool *bval {nullptr};
    tmp_op.get_attr<bool>(op_attr::use_dst, &bval);
    EXPECT_TRUE(bval);
}

TEST(OpSchema, MatmulTypeConstraints) {
    const op_schema_t *matmul_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::MatMul);
    op_t matmul_op {0, op_kind::MatMul, std::string("matmul")};
    logical_tensor_t lt_data_a = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_b = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::s8);

    matmul_op.add_input(lt_data_a);
    matmul_op.add_input(lt_data_b);
    matmul_op.add_output(lt_out);
    // MatMul op doesn't support s8 output
    EXPECT_FALSE(matmul_op_schema->verify(&matmul_op));
}

TEST(OpSchema, Quantize) {
    const op_schema_t *quant_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Quantize);
    op_t quant_op {0, op_kind::Quantize, std::string("quantize")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::s8);

    quant_op.add_input(lt_data);
    quant_op.add_output(lt_out);
    quant_op.set_attr(op_attr::zps, std::vector<int64_t> {1});
    quant_op.set_attr(op_attr::scales, std::vector<float> {0.1f});
    EXPECT_TRUE(quant_op_schema->verify(&quant_op));
}

TEST(OpSchema, QuantizeWithFloatZps) {
    const op_schema_t *quant_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Quantize);
    op_t quant_op {0, op_kind::Quantize, std::string("quantize")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::s8);

    quant_op.add_input(lt_data);
    quant_op.add_output(lt_out);
    quant_op.set_attr(op_attr::scales, std::vector<int64_t> {1});
    quant_op.set_attr(op_attr::zps, std::vector<float> {0.1f});

    //Quantize op does not support float zps and int64 scales
    EXPECT_FALSE(quant_op_schema->verify(&quant_op));
}

TEST(OpSchema, Dequantize) {
    const op_schema_t *dequant_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Dequantize);
    op_t dequant_op {0, op_kind::Dequantize, std::string("dequantize")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);

    dequant_op.add_input(lt_data);
    dequant_op.add_output(lt_out);
    dequant_op.set_attr(op_attr::zps, std::vector<int64_t> {1});
    dequant_op.set_attr(op_attr::scales, std::vector<float> {0.1f});
    EXPECT_TRUE(dequant_op_schema->verify(&dequant_op));
}

TEST(OpSchema, LayerNormBf16) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::LayerNorm);

    op_t lnorm {0, op_kind::LayerNorm, std::string("layer_norm")};
    lnorm.set_attr(op_attr::keep_stats, false);
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::bf16);

    lnorm.add_input(lt_data);
    lnorm.add_input(lt_gamma);
    lnorm.add_input(lt_beta);
    lnorm.add_output(lt_output);

    EXPECT_TRUE(schema->verify(&lnorm));
}

TEST(OpSchema, LayerNormBf16WithGamma) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::LayerNorm);

    op_t lnorm {0, op_kind::LayerNorm, std::string("layer_norm")};
    lnorm.set_attr(op_attr::keep_stats, false);
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::bf16);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::bf16);

    lnorm.add_input(lt_data);
    lnorm.add_input(lt_gamma);
    lnorm.add_input(lt_beta);
    lnorm.add_output(lt_output);

    EXPECT_TRUE(schema->verify(&lnorm));
}

TEST(OpSchema, SoftmaxBf16) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::SoftMax);

    op_t softmax {0, op_kind::SoftMax, std::string("softmax")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(1, data_type::bf16);

    softmax.add_input(lt_data);
    softmax.add_output(lt_output);

    softmax.set_attr<int64_t>(op_attr::axis, 1);
    EXPECT_TRUE(schema->verify(&softmax));
}

TEST(OpSchema, LogSoftmaxBf16) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::LogSoftmax);

    op_t logsoftmax {0, op_kind::LogSoftmax, std::string("logsoftmax")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(1, data_type::bf16);

    logsoftmax.add_input(lt_data);
    logsoftmax.add_output(lt_output);

    logsoftmax.set_attr<int64_t>(op_attr::axis, 1);
    EXPECT_TRUE(schema->verify(&logsoftmax));
}

TEST(OpSchema, TypeCast) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::TypeCast);
    EXPECT_TRUE(schema != nullptr);

    op_t typecast {0, op_kind::TypeCast, std::string("typecast")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(1, data_type::f32);

    typecast.add_input(lt_data);
    typecast.add_output(lt_output);

    EXPECT_TRUE(schema->verify(&typecast));
}

TEST(OpSchema, BatchNormInferenceWithBf16Data) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::BatchNormInference);

    op_t bn {0, op_kind::BatchNormInference, std::string("bn")};
    bn.set_attr(op_attr::epsilon, 0.001f);
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_var = logical_tensor_init(4, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::bf16);

    bn.add_input(lt_data);
    bn.add_input(lt_gamma);
    bn.add_input(lt_beta);
    bn.add_input(lt_mean);
    bn.add_input(lt_var);
    bn.add_output(lt_output);

    EXPECT_TRUE(schema->verify(&bn));
}

TEST(OpSchema, BatchNormInferenceWithBf16Inputs) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::BatchNormInference);

    op_t bn {0, op_kind::BatchNormInference, std::string("bn")};
    bn.set_attr(op_attr::epsilon, 0.001f);
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::bf16);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::bf16);
    logical_tensor_t lt_mean = logical_tensor_init(3, data_type::bf16);
    logical_tensor_t lt_var = logical_tensor_init(4, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::bf16);

    bn.add_input(lt_data);
    bn.add_input(lt_gamma);
    bn.add_input(lt_beta);
    bn.add_input(lt_mean);
    bn.add_input(lt_var);
    bn.add_output(lt_output);

    EXPECT_TRUE(schema->verify(&bn));
}

TEST(OpSchema, StaticTranspose) {
    const op_kind_t op_kind_ = op_kind::StaticTranspose;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::order, true}};
    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, StaticReshape) {
    const op_kind_t op_kind_ = op_kind::StaticReshape;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::shape, true}, {op_attr::special_zero, true}};
    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferStaticTransposeShape) {
    const op_schema_t *static_transpose_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::StaticTranspose);

    op_t static_transpose_op {
            op_kind::StaticTranspose, op_t::kind2str(op_kind::StaticTranspose)};
    static_transpose_op.set_attr(
            op_attr::order, std::vector<int64_t> {2, 0, 1});

    logical_tensor_t lt_in1
            = logical_tensor_init(0, {1024, 64, 32}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1};
    logical_tensor_t lt_o1
            = logical_tensor_init(1, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out1);

    const std::vector<int64_t> inferred_out_shape1
            = logical_tensor_wrapper_t(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {32, 1024, 64};
    EXPECT_EQ(inferred_out_shape1, expected_out_shape1);

    // negative order
    static_transpose_op.set_attr(
            op_attr::order, std::vector<int64_t> {-2, 2, 0});
    lt_in1 = logical_tensor_init(0, {2, 4, 1024}, data_type::f32);
    logical_tensor_t lt_o2
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out2);

    const std::vector<int64_t> inferred_out_shape2
            = logical_tensor_wrapper_t(lt_o2).vdims();
    const std::vector<int64_t> expected_out_shape2 = {4, 1024, 2};
    EXPECT_EQ(inferred_out_shape2, expected_out_shape2);

    // repeat order
    static_transpose_op.set_attr(
            op_attr::order, std::vector<int64_t> {1, 1, 0});
    logical_tensor_t lt_o3
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    status_t infer_status = static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out3);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // order not cover all input axis
    static_transpose_op.set_attr(op_attr::order, std::vector<int64_t> {1, 0});
    logical_tensor_t lt_o4
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out4 {&lt_o4};
    infer_status = static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out4);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // order out of range
    static_transpose_op.set_attr(
            op_attr::order, std::vector<int64_t> {1, 3, 0});
    logical_tensor_t lt_o5
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out5 {&lt_o5};
    infer_status = static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out5);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // order is empty
    static_transpose_op.set_attr(op_attr::order, std::vector<int64_t> {});
    logical_tensor_t lt_o6
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out6 {&lt_o6};
    infer_status = static_transpose_op_schema->shape_infer(
            &static_transpose_op, lt_in, lt_out6);

    const std::vector<int64_t> inferred_out_shape6
            = logical_tensor_wrapper_t(lt_o6).vdims();
    const std::vector<int64_t> expected_out_shape6 = {1024, 4, 2};
    EXPECT_EQ(inferred_out_shape6, expected_out_shape6);
}

TEST(OpSchema, InferStaticReshapeShape) {
    const op_schema_t *static_reshape_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::StaticReshape);

    op_t static_reshape_op {
            op_kind::StaticReshape, op_t::kind2str(op_kind::StaticReshape)};

    std::vector<int64_t> out_shape {16, 4, 8};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    static_reshape_op.set_attr(op_attr::special_zero, false);

    logical_tensor_t lt_in1
            = logical_tensor_init(0, {2, 8, 32}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1};
    logical_tensor_t lt_o1
            = logical_tensor_init(1, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    static_reshape_op_schema->shape_infer(&static_reshape_op, lt_in, lt_out1);

    std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_o1).vdims();
    EXPECT_EQ(inferred_out_shape, out_shape);

    // test special zero true
    out_shape = {4, 0, 16};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    static_reshape_op.set_attr(op_attr::special_zero, true);
    lt_o1 = logical_tensor_init(2, data_type::f32, layout_type::strided);

    static_reshape_op_schema->shape_infer(&static_reshape_op, lt_in, lt_out1);

    inferred_out_shape = logical_tensor_wrapper_t(lt_o1).vdims();
    std::vector<int64_t> expected_out_shape = {4, 8, 16};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    // test special zero false
    out_shape = {4, 0, 16};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    static_reshape_op.set_attr(op_attr::special_zero, false);
    lt_o1 = logical_tensor_init(3, data_type::f32, layout_type::strided);

    status_t infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out1);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // test -1 in shape
    out_shape = {8, 0, -1};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    static_reshape_op.set_attr(op_attr::special_zero, true);
    lt_o1 = logical_tensor_init(4, data_type::f32, layout_type::strided);

    static_reshape_op_schema->shape_infer(&static_reshape_op, lt_in, lt_out1);

    inferred_out_shape = logical_tensor_wrapper_t(lt_o1).vdims();
    expected_out_shape = {8, 8, 8};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    // test input/output with different shape size
    out_shape = {4, 6, 16};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    lt_o1 = logical_tensor_init(5, data_type::f32, layout_type::strided);

    infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out1);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // test invalid shape: more than one -1
    out_shape = {-1, -1, 16};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    lt_o1 = logical_tensor_init(6, data_type::f32, layout_type::strided);

    infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out1);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // test invalid shape: < -1
    out_shape = {4, -6, 16};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    lt_o1 = logical_tensor_init(7, data_type::f32, layout_type::strided);

    infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out1);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // test shape contains 0-D
    out_shape = {0, 4, 7};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    static_reshape_op.set_attr(op_attr::special_zero, false);

    logical_tensor_t lt_in8
            = logical_tensor_init(8, {0, 2, 8, 6}, data_type::f32);
    lt_in = {&lt_in8};
    logical_tensor_t lt_o9
            = logical_tensor_init(9, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out9 {&lt_o9};

    static_reshape_op_schema->shape_infer(&static_reshape_op, lt_in, lt_out9);

    inferred_out_shape = logical_tensor_wrapper_t(lt_o9).vdims();
    EXPECT_EQ(inferred_out_shape, out_shape);

    // test invalid shape case
    out_shape = {-1, 0};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    lt_o9 = logical_tensor_init(10, data_type::f32, layout_type::strided);

    infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out9);
    EXPECT_EQ(infer_status, status::invalid_shape);

    // test input/output with different shape size
    out_shape = {-1, 0};
    static_reshape_op.set_attr(op_attr::shape, out_shape);
    lt_o9 = logical_tensor_init(11, data_type::f32, layout_type::strided);

    infer_status = static_reshape_op_schema->shape_infer(
            &static_reshape_op, lt_in, lt_out9);
    EXPECT_EQ(infer_status, status::invalid_shape);
}

TEST(OpSchema, FailToAddWildcard) {
    const op_schema_t *wildcard_op_schema
            = op_schema_registry_t::get_op_schema(op_kind::Wildcard);
    op_t wildcard_op {0, op_kind::Wildcard, std::string("wildcard")};
    logical_tensor_t lt_data_a = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_b = logical_tensor_init(1, data_type::f16);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::bf16);

    wildcard_op.add_input(lt_data_a);
    wildcard_op.add_input(lt_data_b);
    wildcard_op.add_output(lt_out);
    EXPECT_TRUE(wildcard_op_schema->verify(&wildcard_op));
}

struct dynamic_quantization_params_t {
    data_type_t data_type;
    data_type_t scales_type;
    data_type_t zps_type;
    data_type_t out_type;
    std::string qtype;
    bool asymmetric;
    int64_t axis;
    bool supported;
};

template <op_kind_t kind>
class dynamic_quantization_t
    : public ::testing::TestWithParam<dynamic_quantization_params_t> {
public:
    void Test() {
        const op_schema_t *schema = op_schema_registry_t::get_op_schema(kind);

        auto params = ::testing::TestWithParam<
                dynamic_quantization_params_t>::GetParam();

        op_t dynamic_quantize {0, kind, "dynamic_quantize"};
        logical_tensor_t lt_data = logical_tensor_init(0, params.data_type);
        logical_tensor_t lt_scales = logical_tensor_init(1, params.scales_type);
        logical_tensor_t lt_zps = logical_tensor_init(2, params.zps_type);
        logical_tensor_t lt_output = logical_tensor_init(3, params.out_type);

        dynamic_quantize.add_input(lt_data);
        dynamic_quantize.add_input(lt_scales);
        dynamic_quantize.add_output(lt_output);

        if (params.asymmetric) dynamic_quantize.add_input(lt_zps);

        dynamic_quantize.set_attr<std::string>(op_attr::qtype, params.qtype);
        if (params.qtype == "per_channel")
            dynamic_quantize.set_attr<int64_t>(op_attr::axis, params.axis);

        EXPECT_EQ(schema->verify(&dynamic_quantize), params.supported);
    }
};

using dynamic_quantize_t = dynamic_quantization_t<op_kind::DynamicQuantize>;

TEST_P(dynamic_quantize_t, TestDynamicQuantize) {
    Test();
}

static auto QuantizeCases = []() {
    std::vector<dynamic_quantization_params_t> cases;

    cases.push_back({data_type::f32, data_type::f32, data_type::f32,
            data_type::s8, "per_tensor", false, 1, true});
    cases.push_back({data_type::f32, data_type::f32, data_type::f32,
            data_type::s8, "per_channel", false, 1, true});

    cases.push_back({data_type::f32, data_type::f32, data_type::f32,
            data_type::u8, "per_tensor", false, 1, true});
    cases.push_back({data_type::f32, data_type::f32, data_type::f32,
            data_type::u8, "per_channel", false, 1, true});

    cases.push_back({data_type::f32, data_type::f32, data_type::s8,
            data_type::s8, "per_tensor", true, 1, true});
    cases.push_back({data_type::f32, data_type::f32, data_type::s8,
            data_type::s8, "per_channel", true, 1, true});

    cases.push_back({data_type::f32, data_type::f32, data_type::u8,
            data_type::u8, "per_tensor", true, 1, true});
    cases.push_back({data_type::f32, data_type::f32, data_type::u8,
            data_type::u8, "per_channel", true, 1, true});

    cases.push_back({data_type::f32, data_type::f32, data_type::s32,
            data_type::s8, "per_tensor", true, 1, true});
    cases.push_back({data_type::f32, data_type::f32, data_type::s32,
            data_type::s8, "per_channel", true, 1, true});

    // negative cases
    cases.push_back({data_type::f16, data_type::f32, data_type::s32,
            data_type::s8, "per_tensor", true, 1, false});
    cases.push_back({data_type::f16, data_type::f16, data_type::s32,
            data_type::s8, "per_channel", true, 1, false});
    cases.push_back({data_type::bf16, data_type::bf16, data_type::s32,
            data_type::s8, "per_channel", true, 1, false});
    cases.push_back({data_type::bf16, data_type::bf16, data_type::s32,
            data_type::u8, "per_tensor", false, 1, false});

    return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(OpSchema, dynamic_quantize_t, QuantizeCases());

using dynamic_dequantize_t = dynamic_quantization_t<op_kind::DynamicDequantize>;

TEST_P(dynamic_dequantize_t, TestDynamicDequantize) {
    Test();
}

static auto DequantizeCases = []() {
    std::vector<dynamic_quantization_params_t> cases;

    cases.push_back({data_type::s8, data_type::f32, data_type::f32,
            data_type::f32, "per_tensor", false, 1, true});
    cases.push_back({data_type::s8, data_type::f32, data_type::f32,
            data_type::f32, "per_channel", false, 1, true});

    cases.push_back({data_type::u8, data_type::f32, data_type::f32,
            data_type::f32, "per_tensor", false, 1, true});
    cases.push_back({data_type::u8, data_type::f32, data_type::f32,
            data_type::f32, "per_channel", false, 1, true});

    cases.push_back({data_type::s8, data_type::f32, data_type::s8,
            data_type::f32, "per_tensor", true, 1, true});
    cases.push_back({data_type::s8, data_type::f32, data_type::s8,
            data_type::f32, "per_channel", true, 1, true});

    cases.push_back({data_type::u8, data_type::f32, data_type::u8,
            data_type::f32, "per_tensor", true, 1, true});
    cases.push_back({data_type::u8, data_type::f32, data_type::u8,
            data_type::f32, "per_channel", true, 1, true});

    cases.push_back({data_type::s8, data_type::f32, data_type::s32,
            data_type::f32, "per_tensor", true, 1, true});
    cases.push_back({data_type::s8, data_type::f32, data_type::s32,
            data_type::f32, "per_channel", true, 1, true});

    // negative cases
    cases.push_back({data_type::s8, data_type::f32, data_type::s32,
            data_type::f16, "per_tensor", true, 1, false});
    cases.push_back({data_type::s8, data_type::f16, data_type::s32,
            data_type::f16, "per_channel", true, 1, false});
    cases.push_back({data_type::s8, data_type::bf16, data_type::s32,
            data_type::bf16, "per_channel", true, 1, false});
    cases.push_back({data_type::u8, data_type::bf16, data_type::s32,
            data_type::bf16, "per_tensor", false, 1, false});

    return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(OpSchema, dynamic_dequantize_t, DequantizeCases());

TEST(OpSchema, InferInterpolateShape) {
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind::Interpolate);
    op_t op_ {0, op_kind::Interpolate, std::string("Interpolate")};

    // test normal ncx sizes case
    op_.set_attr<dims>(op_attr::sizes, {10, 20});
    op_.set_attr<std::string>(op_attr::data_format, "NXC");
    op_.set_attr<std::string>(
            op_attr::coordinate_transformation_mode, "align_corners");
    logical_tensor_t lt_in = logical_tensor_init(
            0, {6, 7, 8, 9}, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out
            = logical_tensor_init(1, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> out {&lt_out};
    status_t ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_EQ(ret, status::success);
    std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    std::vector<int64_t> expected_out_shape = {6, 10, 20, 9};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    // sizes and scales should not be valid at the same time
    op_.set_attr<std::string>(op_attr::data_format, "NCX");
    op_.set_attr<std::vector<float>>(op_attr::scales, {0.5f, 0.6f});
    lt_out = logical_tensor_init(2, data_type::f32, layout_type::strided);
    EXPECT_NE(op_schema_->verify(&op_), true);

    // test normal scales case
    op_.set_attr<dims>(op_attr::sizes, {});
    lt_out = logical_tensor_init(3, data_type::f32, layout_type::strided);
    ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_EQ(ret, status::success);
    inferred_out_shape = logical_tensor_wrapper_t(lt_out).vdims();
    expected_out_shape = {6, 7, 4, 5};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);

    // scales.size() == in.ndims() -2
    op_.set_attr<std::vector<float>>(op_attr::scales, {0.5f, 0.6f, 0.7f});
    lt_out = logical_tensor_init(2, data_type::f32, layout_type::strided);
    ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_NE(ret, status::success);

    // sizes.size() == in.ndims() -2
    op_.set_attr<std::vector<float>>(op_attr::scales, {});
    op_.set_attr<dims>(op_attr::sizes, {20});
    lt_out = logical_tensor_init(2, data_type::f32, layout_type::strided);
    ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_NE(ret, status::success);
}

TEST(OpSchema, InferInterpolateBackwardShape) {
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind::InterpolateBackward);
    op_t op_ {0, op_kind::InterpolateBackward,
            std::string("InterpolateBackward")};
    logical_tensor_t lt_in1 = logical_tensor_init(
            0, {8, 7, 6, 5}, data_type::f32, layout_type::strided);
    logical_tensor_t lt_in2 = logical_tensor_init(
            1, {8, 7, 3, 2}, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
    logical_tensor_t lt_out
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> out {&lt_out};
    op_.set_attr<dims>(op_attr::sizes, {6, 5});

    status_t ret = op_schema_->shape_infer(&op_, in, out);
    EXPECT_EQ(ret, status::success);
    std::vector<int64_t> inferred_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    std::vector<int64_t> expected_out_shape = {8, 7, 6, 5};
    EXPECT_EQ(inferred_out_shape, expected_out_shape);
}

TEST(OpSchema, FailToTypecastBf16ToF16) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::TypeCast);

    op_t typecast {0, op_kind::TypeCast, std::string("typecast")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(1, data_type::f16);

    typecast.add_input(lt_data);
    typecast.add_output(lt_output);

    EXPECT_FALSE(schema->verify(&typecast));
}

TEST(OpSchema, FailToTypecastF32ToF32) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::TypeCast);

    op_t typecast {0, op_kind::TypeCast, std::string("typecast")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(1, data_type::f32);

    typecast.add_input(lt_data);
    typecast.add_output(lt_output);

    EXPECT_FALSE(schema->verify(&typecast));
}

TEST(OpSchema, FailToAddBn) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::BatchNormInference);

    op_t bn {0, op_kind::BatchNormInference, std::string("bn")};
    bn.set_attr(op_attr::epsilon, 0.001f);
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::bf16);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::bf16);
    logical_tensor_t lt_mean = logical_tensor_init(3, data_type::bf16);
    logical_tensor_t lt_var = logical_tensor_init(4, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::f32);

    bn.add_input(lt_data);
    bn.add_input(lt_gamma);
    bn.add_input(lt_beta);
    bn.add_input(lt_mean);
    bn.add_input(lt_var);
    bn.add_output(lt_output);

    EXPECT_FALSE(schema->verify(&bn));
}

TEST(OpSchema, FailToAddLn) {
    const op_schema_t *schema
            = op_schema_registry_t::get_op_schema(op_kind::LayerNorm);

    op_t ln {0, op_kind::LayerNorm, std::string("ln")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_gamma = logical_tensor_init(1, data_type::bf16);
    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::bf16);
    logical_tensor_t lt_output = logical_tensor_init(5, data_type::f32);

    ln.add_input(lt_data);
    ln.add_input(lt_gamma);
    ln.add_input(lt_beta);
    ln.add_output(lt_output);

    EXPECT_FALSE(schema->verify(&ln));
}

TEST(OpSchema, Pow) {
    const op_kind_t op_kind_ = op_kind::Pow;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<op_attr_t, bool> attrs_data = {{op_attr::beta, true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, InferPowOutputShape) {
    const op_kind_t op_kind_ = op_kind::Pow;

    verify_single_in_identity_shape_infer(op_kind_);
}
