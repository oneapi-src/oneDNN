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

#include <gtest/gtest.h>

#include "interface/graph.hpp"
#include "interface/op_schema.hpp"
#include "utils.hpp"

using namespace llga::impl;
using namespace llga::tests::unit::utils;

namespace {

/**
 * This function verifies op schema. Should be used as a test helper.
 * attrs_data argument should contain all attributes (as keys) associated with op_kind,
 * along with information (as value) whether they are required or not.
 * Please treat the op_schema_test.Convolution as an example.
 */
void verify_op_schema(const op_kind_t op_kind_, const size_t expected_in_size,
        const size_t expected_out_size, const size_t expected_attr_size,
        const std::map<std::string, bool> &attrs_data) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    EXPECT_TRUE(nullptr != op_schema_);

    const size_t input_size = op_schema_->get_inputs().size();
    EXPECT_EQ(input_size, expected_in_size);

    size_t output_size = op_schema_->get_outputs().size();
    EXPECT_EQ(output_size, expected_out_size);

    size_t attr_size = op_schema_->get_attrs().size();
    EXPECT_EQ(attr_size, expected_attr_size);

    for (const auto &attr_data : attrs_data) {
        const auto &attr_name = attr_data.first;
        const auto is_required = attr_data.second;
        EXPECT_EQ(op_schema_->get_attrs().count(attr_name), 1);
        EXPECT_EQ(op_schema_->get_attrs().at(attr_name).required_, is_required);
    }
}

void verify_shape_infer_for_arithmetic_op_no_broadcast(
        const op_kind_t op_kind_) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};
    const std::string no_broadcast_attr_val = "none";
    node_.set_attr("auto_broadcast", no_broadcast_attr_val);

    // In valid situation, the inputs layout should only be strided or opaque,
    // so we need to test both of these two input layout type
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1 = logical_tensor_init(
                0, {1, 3, 416, 416}, data_type::f32, ltype);
        logical_tensor_t lt_in2 = logical_tensor_init(
                1, {1, 3, 416, 416}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        logical_tensor_t lt_out
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&node_, in, out);
        EXPECT_EQ(ret, status::success);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = {1, 3, 416, 416};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper(lt_out).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);

        // negative case - non-matching input dims
        logical_tensor_t lt_in2_neg
                = logical_tensor_init(1, {1, 3, 32, 32}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in_neg {&lt_in1, &lt_in2_neg};
        logical_tensor_t lt_out_neg
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out_neg {&lt_out_neg};
        ret = op_schema_->shape_infer(&node_, in_neg, out_neg);
        EXPECT_EQ(ret, status::invalid_shape);
    }
}

void verify_shape_infer_for_arithmetic_op_with_broadcast(
        const op_kind_t op_kind_) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1
                = logical_tensor_init(0, {2, 3, 64, 64}, data_type::f32, ltype);
        logical_tensor_t lt_in2
                = logical_tensor_init(1, {3, 1, 64}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        logical_tensor_t lt_out
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out {&lt_out};

        // shape inference without explicitly setting auto_broadcast
        // should be enabled by default
        op_schema_->shape_infer(&node_, in, out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = {2, 3, 64, 64};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper(lt_out).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);

        // explicitly setting auto_broadcast
        const std::string with_broadcast_attr_val = "numpy";
        node_.set_attr("auto_broadcast", with_broadcast_attr_val);
        logical_tensor_t lt_out_expl
                = logical_tensor_init(3, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out_expl {&lt_out_expl};

        op_schema_->shape_infer(&node_, in, out_expl);
        const std::vector<int64_t> infered_out_shape_expl
                = logical_tensor_wrapper(lt_out_expl).vdims();
        EXPECT_EQ(infered_out_shape_expl, expected_out_shape);

        const std::vector<int64_t> infered_out_strides2
                = logical_tensor_wrapper(lt_out).vstrides();
        EXPECT_EQ(infered_out_strides2, expected_out_strides);
    }
}

void set_conv_common_attr(node_t &node_, std::vector<int64_t> &strides,
        std::vector<int64_t> &pads_begin, std::vector<int64_t> &pads_end,
        std::vector<int64_t> &dilations, std::string auto_pad,
        std::string data_format, std::string filter_format, int64_t groups) {
    node_.set_attr("strides", strides);
    node_.set_attr("pads_begin", pads_begin);
    node_.set_attr("pads_end", pads_end);
    node_.set_attr("dilations", dilations);
    node_.set_attr("auto_pad", auto_pad);
    node_.set_attr("data_format", data_format);
    node_.set_attr("filter_format", filter_format);
    node_.set_attr("groups", groups);
}

void verify_shape_infer_for_conv(const op_kind_t op_kind_,
        std::string data_format, std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &expected_out_shape) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";

    set_conv_common_attr(node_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto &unchanged_pads_begin
            = node_.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end = node_.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    // if output shape is known, infer auto pad
    op_schema_->shape_infer(&node_, in, out);
    auto &infered_pads_begin
            = node_.get_attr<std::vector<int64_t>>("pads_begin");
    auto &infered_pads_end = node_.get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> expected_pads = {0, 0};
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

void verify_shape_infer_for_conv_bprop_data(const op_kind_t op_kind_,
        std::string data_format, std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &in_output_shape,
        const std::vector<int64_t> &expected_out_shape) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> output_padding = {3, 3};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";

    set_conv_common_attr(node_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);
    node_.set_attr("output_padding", output_padding);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    logical_tensor_t lt_output_shape
            = logical_tensor_init(2, in_output_shape, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight, &lt_output_shape};
    logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto &unchanged_pads_begin
            = node_.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end = node_.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    // if output shape is known, infer auto pad
    op_schema_->shape_infer(&node_, in, out);
    auto &infered_pads_begin
            = node_.get_attr<std::vector<int64_t>>("pads_begin");
    auto &infered_pads_end = node_.get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> expected_pads = {0, 0};
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

void verify_identity_shape_infer_(const op_kind_t op_kind_,
        const size_t out_lt_id, std::vector<logical_tensor_t *> &in) {
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_out = logical_tensor_init(
            out_lt_id, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_out).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

void verify_single_in_identity_shape_infer(const op_kind_t op_kind_) {
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in = logical_tensor_init(
                0, {1, 3, 224, 224}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in};
        verify_identity_shape_infer_(op_kind_, 1, in);
    }
}

void verify_two_ins_identity_shape_infer(const op_kind_t op_kind_) {
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1 = logical_tensor_init(
                0, {1, 3, 224, 224}, data_type::f32, ltype);
        logical_tensor_t lt_in2 = logical_tensor_init(
                1, {1, 3, 224, 224}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        verify_identity_shape_infer_(op_kind_, 2, in);
    }
}
} // namespace

#ifndef NDEBUG

TEST(op_schema_test, duplicated_attribute) {
    EXPECT_DEATH(op_schema()
                         .set_attr("kernel", "size of each filter", true)
                         .set_attr("kernel", "size of each filter", true),
            "provided attribute has already been set");
}

TEST(op_schema_test, duplicated_input) {
    EXPECT_DEATH(op_schema()
                         .set_num_inputs(5)
                         .set_input(3, "mean", "value for mean normalization")
                         .set_input(3, "mean", "value for mean normalization"),
            "provided `in_offset` has already been set");
}

TEST(op_schema_test, duplicated_output) {
    EXPECT_DEATH(op_schema()
                         .set_num_outputs(1)
                         .set_output(0, "output", "output tensor")
                         .set_output(0, "output", "output tensor"),
            "provided `out_offset` has already been set");
}

TEST(op_schema_test, set_input_before_num_inputs) {
    EXPECT_DEATH(op_schema()
                         .set_input(0, "a", "first input tensor")
                         .set_num_inputs(2),
            "input set before setting num_inputs_");
}

TEST(op_schema_test, set_output_before_num_outputs) {
    EXPECT_DEATH(op_schema()
                         .set_output(0, "output", "output tensor")
                         .set_num_outputs(1),
            "output set before setting num_outputs_");
}

TEST(op_schema_test, exceeded_num_inputs) {
    EXPECT_DEATH(op_schema()
                         .set_num_inputs(1)
                         .set_input(0, "a", "first input tensor")
                         .set_input(1, "b", "second input tensor"),
            "input offset exceeds declared num of inputs");
}

TEST(op_schema_test, exceeded_num_outputs) {
    EXPECT_DEATH(op_schema()
                         .set_num_outputs(1)
                         .set_output(0, "a", "first output tensor")
                         .set_output(1, "b", "second output tensor"),
            "output offset exceeds declared num of outputs");
}

#endif

TEST(op_schema_test, Convolution) {
    const op_kind_t op_kind_ = op_kind::Convolution;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<std::string, bool> attrs_data
            = {{"strides", true}, {"pads_begin", true}, {"pads_end", true},
                    {"dilations", true}, {"auto_pad", false}, {"groups", false},
                    {"data_format", false}, {"filter_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Conv_bias) {
    std::set<op_kind_t> conv_kinds = {op_kind::conv_bias,
            op_kind::conv_bias_abs, op_kind::conv_bias_relu,
            op_kind::conv_bias_sigmoid, op_kind::conv_bias_sqrt,
            op_kind::conv_bias_square, op_kind::conv_bias_tanh};
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<std::string, bool> attrs_data
            = {{"strides", true}, {"pads_begin", true}, {"pads_end", true},
                    {"dilations", true}, {"auto_pad", false}, {"groups", false},
                    {"data_format", false}, {"filter_format", false}};

    for (auto k : conv_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, conv_bias_infer_shape) {
    const op_schema *a_op_schema
            = op_schema_registry::get_op_schema(op_kind::conv_bias);
    EXPECT_TRUE(nullptr != a_op_schema);
    node_t a_node {op_kind::conv_bias};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_node, strides, pads_begin, pads_end, dilations,
            "None", data_format, filter_format, groups);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(3, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_node, lt_in, lt_out);
    auto &unchanged_pads_begin
            = a_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end
            = a_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);
    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(op_schema_test, Conv_bias_add_elu) {
    const op_kind_t conv_kind = op_kind::conv_bias_add_elu;
    const size_t expected_in_size = 4;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"groups", false}, {"data_format", false},
            {"filter_format", false}, {"alpha", true}};

    verify_op_schema(conv_kind, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Conv_bias_add_relu6) {
    const op_kind_t conv_kind = op_kind::conv_bias_add_relu6;
    const size_t expected_in_size = 4;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 10;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"groups", false}, {"data_format", false},
            {"filter_format", false}, {"min", true}, {"max", true}};

    verify_op_schema(conv_kind, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, conv_bias_add_infer_shape) {
    const op_schema *a_op_schema
            = op_schema_registry::get_op_schema(op_kind::conv_bias_add_elu);
    EXPECT_TRUE(nullptr != a_op_schema);
    node_t a_node {op_kind::conv_bias_add_elu};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_node, strides, pads_begin, pads_end, dilations,
            "None", data_format, filter_format, groups);
    a_node.set_attr("alpha", 1.f);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    logical_tensor_t lt_added
            = logical_tensor_init(3, {1, 512, 33, 33}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(4, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {
            &lt_data, &lt_weight, &lt_bias, &lt_added};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_node, lt_in, lt_out);
    auto &unchanged_pads_begin
            = a_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end
            = a_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(op_schema_test, Conv_bias_elu) {
    const op_kind_t conv_kind = op_kind::conv_bias_elu;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"groups", false}, {"data_format", false},
            {"filter_format", false}, {"alpha", true}};

    verify_op_schema(conv_kind, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Conv_bias_hardtanh) {
    std::set<op_kind_t> conv_kinds
            = {op_kind::conv_bias_hardtanh, op_kind::conv_bias_relu6};
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 10;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"groups", false}, {"data_format", false},
            {"filter_format", false}, {"min", true}, {"max", true}};

    for (auto k : conv_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, generate_default_attrib) {
    const op_schema *matmul_op_schema
            = op_schema_registry::get_op_schema(op_kind::MatMul);
    op_t matmul_op {0, kMatMul, std::string("matmul")};
    logical_tensor_t lt_data_a = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_b = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);

    matmul_op.add_input(&lt_data_a);
    matmul_op.add_input(&lt_data_b);
    matmul_op.add_output(&lt_out);
    EXPECT_TRUE(matmul_op_schema->verify(&matmul_op));

    matmul_op.set_attr("transpose_a", true);
    const bool *flag;
    const bool **ret_flag = &flag;
    matmul_op.attr<bool>("transpose_a", ret_flag);
    EXPECT_TRUE(ret_flag);
    EXPECT_EQ(matmul_op.attr<bool>("transpose_b", ret_flag),
            status::invalid_argument);

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(&matmul_op), status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_nodes(), 1);

    const auto &matmul_node = agraph.get_nodes()[0];
    EXPECT_TRUE(matmul_node->get_attr<bool>("transpose_a"));
    EXPECT_FALSE(matmul_node->get_attr<bool>("transpose_b"));
}

TEST(op_schema_test, verify_function) {
    const op_schema *conv_op_schema
            = op_schema_registry::get_op_schema(op_kind::Convolution);

    op_t conv_op {0, kConvolution, std::string("convolution")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);

    conv_op.add_input(&lt_data);
    conv_op.add_input(&lt_weight);
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    conv_op.add_output(&lt_out);
    EXPECT_FALSE(conv_op_schema->verify(&conv_op));

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    conv_op.set_attr("strides", strides);
    conv_op.set_attr("pads_begin", pads_begin);
    conv_op.set_attr("pads_end", pads_end);
    conv_op.set_attr("dilations", dilations);
    conv_op.set_attr("data_format", data_format);
    conv_op.set_attr("filter_format", filter_format);

    EXPECT_TRUE(conv_op_schema->verify(&conv_op));

    std::string auto_pad = "VALID";
    conv_op.set_attr("auto_pad", auto_pad);

    EXPECT_TRUE(conv_op_schema->verify(&conv_op));

    float arbitrary_value = 123.0;
    const std::string arbitrary_name {"arbitrary"};
    conv_op.set_attr("arbitrary", arbitrary_value);

    EXPECT_TRUE(conv_op_schema->verify(&conv_op));
}

TEST(op_schema_test, conv_ncx_oix_infer_shape) {
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
    }
}

void infer_conv_shape(op_kind_t kind) {
    using namespace llga::impl;
    const op_schema *conv_op_schema = op_schema_registry::get_op_schema(kind);

    node_t conv_node {kind};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";
    std::string data_format = "NXC";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(conv_node, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    // data shape {N, H, W, IC}
    logical_tensor_t lt_data_0
            = logical_tensor_init(0, {1, 224, 224, 3}, data_type::f32);
    // weight shape {OC, IC, KH, KW}
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {16, 3, 3, 3}, data_type::f32);
    // bias shape {OC}
    logical_tensor_t lt_bias = logical_tensor_init(2, {16}, data_type::f32);

    // add input
    logical_tensor_t lt_data_1
            = logical_tensor_init(3, {1, 16, 111, 111}, data_type::f32);

    std::vector<logical_tensor_t *> lt_in {
            &lt_data_0, &lt_weight, &lt_bias, &lt_data_1};
    logical_tensor_t lt_o
            = logical_tensor_init(4, {-1, -1, -1, -1}, data_type::f32);
    std::vector<logical_tensor_t *> lt_out {&lt_o};

    conv_op_schema->shape_infer(&conv_node, lt_in, lt_out);

    auto &unchanged_pads_begin
            = conv_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end
            = conv_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    // if output shape is known, infer auto pad
    conv_op_schema->shape_infer(&conv_node, lt_in, lt_out);
    auto &infered_pads_begin
            = conv_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &infered_pads_end
            = conv_node.get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> expected_pads = {0, 0};
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);

    const std::vector<int64_t> expect_output_shape = {1, 111, 111, 16};
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    EXPECT_EQ(infered_out_shape, expect_output_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expect_output_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(op_schema_test, conv_bias_add_relu_nxc_oix_infer_shape) {
    infer_conv_shape(op_kind::conv_bias);
    infer_conv_shape(op_kind::conv_bias_add);
    infer_conv_shape(op_kind::conv_bias_add_relu);
}

TEST(op_schema_test, conv_nxc_oix_infer_shape) {
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
    }
}

TEST(op_schema_test, conv_nxc_xio_infer_shape) {
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
    }
}

TEST(op_schema_test, conv_ncx_xio_infer_shape) {
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
    }
}

TEST(op_schema_test, PowBackpropExponent) {
    const op_kind_t op_kind_ = op_kind::PowBackpropExponent;
    const size_t expected_in_size = 4;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data;

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, exponent_infer_shape) {
    const op_schema *exponent_op_schema
            = op_schema_registry::get_op_schema(op_kind::PowBackpropExponent);

    node_t exponent_node {op_kind::PowBackpropExponent};

    logical_tensor_t lt_in1
            = logical_tensor_init(0, {64, 1024, 64}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(1, {64, 1024, 64}, data_type::f32);
    logical_tensor_t lt_in3
            = logical_tensor_init(2, {64, 1024, 64}, data_type::f32);
    logical_tensor_t lt_in4 = logical_tensor_init(3, {64}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1, &lt_in2, &lt_in3, &lt_in4};
    logical_tensor_t lt_o1
            = logical_tensor_init(4, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    exponent_op_schema->shape_infer(&exponent_node, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    const std::vector<int64_t> infered_out_strides1
            = logical_tensor_wrapper(lt_o1).vstrides();
    const std::vector<int64_t> expected_out_strides1
            = compute_dense_strides(expected_out_shape1);
    EXPECT_EQ(infered_out_strides1, expected_out_strides1);
}

TEST(op_schema_test, MaxPool) {
    const op_kind_t op_kind_ = op_kind::MaxPool;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"kernel", true}, {"pads_begin", true}, {"pads_end", true},
            {"dilations", false}, {"data_format", false}, {"auto_pad", false},
            {"rounding_type", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, maxpool_infer_shape) {
    const op_schema *pool_op_schema
            = op_schema_registry::get_op_schema(op_kind::MaxPool);

    node_t pool_node {op_kind::MaxPool};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> kernel = {3, 3};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "SAME_UPPER";
    std::string data_format = "NCX";

    pool_node.set_attr("strides", strides);
    pool_node.set_attr("pads_begin", pads_begin);
    pool_node.set_attr("pads_end", pads_end);
    pool_node.set_attr("kernel", kernel);
    pool_node.set_attr("dilations", dilations);
    pool_node.set_attr("auto_pad", auto_pad);
    pool_node.set_attr("data_format", data_format);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_data};
    logical_tensor_t lt_o
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out {&lt_o};

    // if output shape is unknown, infer output shape
    pool_op_schema->shape_infer(&pool_node, lt_in, lt_out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);

    // if output shape is known, infer auto pad
    pool_op_schema->shape_infer(&pool_node, lt_in, lt_out);
    auto &infered_pads_begin
            = pool_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &infered_pads_end
            = pool_node.get_attr<std::vector<int64_t>>("pads_end");
    const std::vector<int64_t> expected_pads_begin = {0, 0};
    const std::vector<int64_t> expected_pads_end = {1, 1};
    EXPECT_EQ(infered_pads_begin, expected_pads_begin);
    EXPECT_EQ(infered_pads_end, expected_pads_end);
}

TEST(op_schema_test, maxpool_ceil_mode) {
    const op_schema *pool_op_schema
            = op_schema_registry::get_op_schema(op_kind::MaxPool);

    std::set<std::string> rounding_types = {"ceil", "floor"};
    for (auto &rounding_type : rounding_types) {
        node_t pool_node {op_kind::MaxPool};
        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> kernel = {3, 3};
        std::vector<int64_t> pads_begin = {0, 0};
        std::vector<int64_t> pads_end = {0, 0};
        std::vector<int64_t> dilations = {1, 1};
        std::string data_format = "NCX";
        pool_node.set_attr("strides", strides);
        pool_node.set_attr("pads_begin", pads_begin);
        pool_node.set_attr("pads_end", pads_end);
        pool_node.set_attr("kernel", kernel);
        pool_node.set_attr("dilations", dilations);
        pool_node.set_attr("data_format", data_format);
        pool_node.set_attr("rounding_type", rounding_type);

        logical_tensor_t lt_data
                = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
        std::vector<logical_tensor_t *> lt_in {&lt_data};
        logical_tensor_t lt_o
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_out {&lt_o};

        // if output shape is unknown, infer output shape
        pool_op_schema->shape_infer(&pool_node, lt_in, lt_out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper(lt_o).vdims();
        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper(lt_o).vstrides();
        if (rounding_type == "ceil") {
            const std::vector<int64_t> expected_out_shape = {1, 3, 112, 112};
            EXPECT_EQ(infered_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(infered_out_strides, expected_out_strides);
        } else { // rounding_type = floor
            const std::vector<int64_t> expected_out_shape = {1, 3, 111, 111};
            EXPECT_EQ(infered_out_shape, expected_out_shape);
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(infered_out_strides, expected_out_strides);
        }

        auto &unchanged_pads_begin
                = pool_node.get_attr<std::vector<int64_t>>("pads_begin");
        auto &unchanged_pads_end
                = pool_node.get_attr<std::vector<int64_t>>("pads_end");
        EXPECT_EQ(unchanged_pads_begin, pads_begin);
        EXPECT_EQ(unchanged_pads_end, pads_end);

        // if output shape is known, infer auto pad
        pool_op_schema->shape_infer(&pool_node, lt_in, lt_out);
        auto &infered_pads_begin
                = pool_node.get_attr<std::vector<int64_t>>("pads_begin");
        auto &infered_pads_end
                = pool_node.get_attr<std::vector<int64_t>>("pads_end");

        if (rounding_type == "ceil") {
            const std::vector<int64_t> expected_pads_begin = {0, 0};
            const std::vector<int64_t> expected_pads_end = {1, 1};
            EXPECT_EQ(infered_pads_begin, expected_pads_begin);
            EXPECT_EQ(infered_pads_end, expected_pads_end);
        } else { // rounding_type = floor
            const std::vector<int64_t> expected_pads = {0, 0};
            EXPECT_EQ(infered_pads_begin, expected_pads);
            EXPECT_EQ(infered_pads_end, expected_pads);
        }
    }
}

TEST(op_schema_test, MatMul) {
    std::set<op_kind_t> matmul_kinds = {op_kind::MatMul, op_kind::matmul_relu,
            op_kind::matmul_elu, op_kind::matmul_sigmoid,
            op_kind::matmul_hardtanh, op_kind::matmul_gelu};
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"transpose_a", false}, {"transpose_b", false}};

    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, MatMul_bias) {
    std::set<op_kind_t> matmul_kinds = {op_kind::matmul_bias,
            op_kind::matmul_bias_relu, op_kind::matmul_bias_sigmoid};
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"transpose_a", false}, {"transpose_b", false}};

    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, MatMul_bias_add) {
    std::set<op_kind_t> matmul_kinds
            = {op_kind::matmul_bias_add, op_kind::matmul_bias_add_relu};
    const size_t expected_in_size = 4;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"transpose_a", false}, {"transpose_b", false}};
    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, MatMul_bias_bn) {
    op_kind_t matmul_kinds = op_kind::matmul_bias_bn;
    const size_t expected_in_size = 7;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data = {
            {"epsilon", true}, {"transpose_a", false}, {"transpose_b", false}};
    verify_op_schema(matmul_kinds, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, MatMul_bias_elu) {
    op_kind_t matmul_kinds = op_kind::matmul_bias_elu;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data
            = {{"alpha", true}, {"transpose_a", false}, {"transpose_b", false}};
    verify_op_schema(matmul_kinds, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, MatMul_bias_hardtanh) {
    std::set<op_kind_t> matmul_kinds
            = {op_kind::matmul_bias_hardtanh, op_kind::matmul_bias_relu6};
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 4;
    const std::map<std::string, bool> attrs_data = {{"min", true},
            {"max", true}, {"transpose_a", false}, {"transpose_b", false}};

    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(op_schema_test, matmul_infer_shape) {
    const op_schema *matmul_op_schema
            = op_schema_registry::get_op_schema(op_kind::MatMul);

    node_t matmul_node {op_kind::MatMul};
    bool transpose_a = true;
    matmul_node.set_attr("transpose_a", transpose_a);

    // test 2 dims matmul
    logical_tensor_t lt_in1
            = logical_tensor_init(0, {1024, 64}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(1, {1024, 1000}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1, &lt_in2};
    logical_tensor_t lt_o1
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    const std::vector<int64_t> infered_out_strides1
            = logical_tensor_wrapper(lt_o1).vstrides();
    const std::vector<int64_t> expected_out_strides1
            = compute_dense_strides(expected_out_shape1);
    EXPECT_EQ(infered_out_strides1, expected_out_strides1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    logical_tensor_t lt_o2
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out2);

    auto &infered_out_shape2 = lt_o2.dims;
    EXPECT_EQ(lt_o2.ndims, 1);
    EXPECT_EQ(infered_out_shape2[0], 1000);
    auto &infered_out_strides2 = lt_o2.layout.strides;
    EXPECT_EQ(infered_out_strides2[0], 1);

    // test >2 dims matmul
    lt_in1 = logical_tensor_init(0, {3, 1, 10, 1024, 64}, data_type::f32);
    lt_in2 = logical_tensor_init(1, {5, 1, 1024, 1000}, data_type::f32);
    logical_tensor_t lt_o3
            = logical_tensor_init(2, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out3);

    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(infered_out_shape3, expected_out_shape3);

    const std::vector<int64_t> infered_out_strides3
            = logical_tensor_wrapper(lt_o3).vstrides();
    const std::vector<int64_t> expected_out_strides3
            = compute_dense_strides(expected_out_shape3);
    EXPECT_EQ(infered_out_strides3, expected_out_strides3);
}

TEST(op_schema_test, matmul_bias_infer_shape) {
    const op_schema *matmul_op_schema
            = op_schema_registry::get_op_schema(op_kind::matmul_bias);

    node_t matmul_node {op_kind::matmul_bias};
    bool transpose_a = true;
    matmul_node.set_attr("transpose_a", transpose_a);

    // test 2 dims matmul
    logical_tensor_t lt_in1
            = logical_tensor_init(0, {1024, 64}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(1, {1024, 1000}, data_type::f32);
    logical_tensor_t lt_in3
            = logical_tensor_init(2, {64, 1000}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1, &lt_in2, &lt_in3};
    logical_tensor_t lt_o1 = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {1000}, data_type::f32);
    logical_tensor_t lt_o2 = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out2);

    auto &infered_out_shape2 = lt_o2.dims;
    EXPECT_EQ(lt_o2.ndims, 1);
    EXPECT_EQ(infered_out_shape2[0], 1000);

    // test >2 dims matmul
    lt_in1 = logical_tensor_init(0, {3, 1, 10, 1024, 64}, data_type::f32);
    lt_in2 = logical_tensor_init(1, {5, 1, 1024, 1000}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {3, 5, 10, 64, 1000}, data_type::f32);
    logical_tensor_t lt_o3 = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out3);

    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(infered_out_shape3, expected_out_shape3);
}

TEST(op_schema_test, matmul_bias_add_infer_shape) {
    const op_schema *matmul_op_schema
            = op_schema_registry::get_op_schema(op_kind::matmul_bias_add);

    node_t matmul_node {op_kind::matmul_bias_add};
    bool transpose_a = true;
    matmul_node.set_attr("transpose_a", transpose_a);

    // test 2 dims matmul
    logical_tensor_t lt_in1
            = logical_tensor_init(0, {1024, 64}, data_type::f32);
    logical_tensor_t lt_in2
            = logical_tensor_init(1, {1024, 1000}, data_type::f32);
    logical_tensor_t lt_in3
            = logical_tensor_init(2, {64, 1000}, data_type::f32);
    logical_tensor_t lt_in4
            = logical_tensor_init(3, {64, 1000}, data_type::f32);
    std::vector<logical_tensor_t *> lt_in {&lt_in1, &lt_in2, &lt_in3, &lt_in4};
    logical_tensor_t lt_o1 = logical_tensor_init(4, data_type::f32);
    std::vector<logical_tensor_t *> lt_out1 {&lt_o1};

    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {1000}, data_type::f32);
    lt_in4 = logical_tensor_init(3, {1000}, data_type::f32);
    logical_tensor_t lt_o2 = logical_tensor_init(4, data_type::f32);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out2);

    auto &infered_out_shape2 = lt_o2.dims;
    EXPECT_EQ(lt_o2.ndims, 1);
    EXPECT_EQ(infered_out_shape2[0], 1000);

    // test >2 dims matmul
    lt_in1 = logical_tensor_init(0, {3, 1, 10, 1024, 64}, data_type::f32);
    lt_in2 = logical_tensor_init(1, {5, 1, 1024, 1000}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {3, 5, 10, 64, 1000}, data_type::f32);
    lt_in4 = logical_tensor_init(3, {3, 5, 10, 64, 1000}, data_type::f32);
    logical_tensor_t lt_o3 = logical_tensor_init(4, data_type::f32);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    matmul_op_schema->shape_infer(&matmul_node, lt_in, lt_out3);

    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(infered_out_shape3, expected_out_shape3);
}

TEST(op_schema_test, BatchNormInference_infer_shape) {
    std::set<op_kind_t> bn_kinds
            = {op_kind::BatchNormInference, op_kind::bn_relu};
    for (auto cur_kind : bn_kinds) {
        const op_schema *bn_op_schema
                = op_schema_registry::get_op_schema(cur_kind);
        EXPECT_TRUE(nullptr != bn_op_schema);
        node_t bn_node {cur_kind};
        bn_node.set_attr("epsilon", 0.001f);

        logical_tensor_t lt_data
                = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
        logical_tensor_t lt_gamma
                = logical_tensor_init(1, {1, 256}, data_type::f32);
        logical_tensor_t lt_beta
                = logical_tensor_init(2, {1, 256}, data_type::f32);
        logical_tensor_t lt_mean
                = logical_tensor_init(3, {1, 256}, data_type::f32);
        logical_tensor_t lt_variance
                = logical_tensor_init(4, {1, 256}, data_type::f32);
        logical_tensor_t lt_o
                = logical_tensor_init(5, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {
                &lt_data, &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
        std::vector<logical_tensor_t *> lt_out {&lt_o};

        bn_op_schema->shape_infer(&bn_node, lt_in, lt_out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper(lt_o).vdims();
        const std::vector<int64_t> expected_out_shape = {1, 256, 64, 64};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper(lt_o).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);
    }
}

TEST(op_schema_test, conv_bn_relu_infer_shape) {
    const op_schema *a_op_schema
            = op_schema_registry::get_op_schema(op_kind::conv_bn_relu);
    EXPECT_TRUE(nullptr != a_op_schema);
    node_t a_node {op_kind::conv_bn_relu};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_node, strides, pads_begin, pads_end, dilations,
            "None", data_format, filter_format, groups);
    a_node.set_attr("epsilon", epsilon);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_gamma
            = logical_tensor_init(1, {1, 512}, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(2, {1, 512}, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(3, {1, 512}, data_type::f32);
    logical_tensor_t lt_variance
            = logical_tensor_init(4, {1, 512}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(5, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {
            &lt_data, &lt_weight, &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_node, lt_in, lt_out);
    auto &unchanged_pads_begin
            = a_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &unchanged_pads_end
            = a_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(op_schema_test, conv_bn_add_relu_infer_shape) {
    const op_schema *a_op_schema
            = op_schema_registry::get_op_schema(op_kind::conv_bn_add_relu);
    EXPECT_TRUE(nullptr != a_op_schema);
    node_t a_node {op_kind::conv_bn_add_relu};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_node, strides, pads_begin, pads_end, dilations,
            "None", data_format, filter_format, groups);
    a_node.set_attr("epsilon", epsilon);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_gamma
            = logical_tensor_init(1, {1, 512}, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(2, {1, 512}, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(3, {1, 512}, data_type::f32);
    logical_tensor_t lt_variance
            = logical_tensor_init(4, {1, 512}, data_type::f32);
    logical_tensor_t lt_add
            = logical_tensor_init(5, {1, 512, 33, 33}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(6, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_gamma,
            &lt_beta, &lt_mean, &lt_variance, &lt_add};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_node, lt_in, lt_out);
    auto &infered_pads_begin
            = a_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto &infered_pads_end = a_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(infered_pads_begin[0], 1);
    EXPECT_EQ(infered_pads_begin[1], 1);
    EXPECT_EQ(infered_pads_end[0], 2);
    EXPECT_EQ(infered_pads_end[1], 2);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(op_schema_test, Add) {
    const op_kind_t op_kind_ = op_kind::Add;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Add_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Add;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Add_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Add;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, BatchNormForwardTraining) {
    const op_kind_t op_kind_ = op_kind::BatchNormForwardTraining;
    const size_t expected_in_size = 5;
    const size_t expected_out_size = 5;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data
            = {{"epsilon", true}, {"momentum", false}, {"data_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, BatchNormForwardTraining_infer_shape) {
    const op_kind_t op_kind_ = op_kind::BatchNormForwardTraining;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

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

    node_.set_attr("epsilon", 0.01);
    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape0
            = logical_tensor_wrapper(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape0 = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape0, expected_out_shape0);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_r_mean).vdims();
    const std::vector<int64_t> infered_out_shape2
            = logical_tensor_wrapper(lt_r_var).vdims();
    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper(lt_b_mean).vdims();
    const std::vector<int64_t> infered_out_shape4
            = logical_tensor_wrapper(lt_b_var).vdims();
    const std::vector<int64_t> expected_out_shape_1D = {224};
    EXPECT_EQ(infered_out_shape1, expected_out_shape_1D);
    EXPECT_EQ(infered_out_shape2, expected_out_shape_1D);
    EXPECT_EQ(infered_out_shape3, expected_out_shape_1D);
    EXPECT_EQ(infered_out_shape4, expected_out_shape_1D);

    logical_tensor_t lt_out_not_filled = logical_tensor_init(8, data_type::f32);
    std::vector<logical_tensor_t *> out_partially_not_filled {
            &lt_out_not_filled, &lt_r_mean, &lt_r_var, &lt_b_mean, &lt_b_var};
    const std::string data_f = "NCX";
    node_.set_attr("data_format", data_f);
    auto result = op_schema_->shape_infer(&node_, in, out_partially_not_filled);
    EXPECT_EQ(result, status::invalid_shape);
}

TEST(op_schema_test, BatchNormTrainingBackprop) {
    const op_kind_t op_kind_ = op_kind::BatchNormTrainingBackprop;
    const size_t expected_in_size = 6;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data = {
            {"epsilon", true}, {"is_training", false}, {"data_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, BatchNormTrainingBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::BatchNormTrainingBackprop;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

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

    node_.set_attr("epsilon", 0.01);
    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape0
            = logical_tensor_wrapper(lt_input_delta).vdims();
    const std::vector<int64_t> expected_out_shape0 = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape0, expected_out_shape0);

    logical_tensor_t lt_gamma = logical_tensor_init(5, {224}, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(6, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in_with_options {&lt_in, &lt_output_delta,
            &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
    logical_tensor_t lt_gamma_delta = logical_tensor_init(7, data_type::f32);
    logical_tensor_t lt_beta_delta = logical_tensor_init(8, data_type::f32);
    std::vector<logical_tensor_t *> out_with_options {
            &lt_input_delta, &lt_gamma_delta, &lt_beta_delta};

    op_schema_->shape_infer(&node_, in_with_options, out_with_options);
    const std::vector<int64_t> expected_out_shape_1D = {224};
    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper(lt_gamma_delta).vdims();
    const std::vector<int64_t> infered_out_shape2
            = logical_tensor_wrapper(lt_beta_delta).vdims();
    EXPECT_EQ(infered_out_shape1, expected_out_shape_1D);
    EXPECT_EQ(infered_out_shape2, expected_out_shape_1D);
}

TEST(op_schema_test, BiasAdd) {
    const op_kind_t op_kind_ = op_kind::BiasAdd;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"data_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, BiasAdd_infer_shape) {
    const op_kind_t op_kind_ = op_kind::BiasAdd;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(1, {224}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_bias};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    logical_tensor_t lt_out_not_filled = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out_not_filled {&lt_out_not_filled};
    const std::string data_f = "NCX";
    node_.set_attr("data_format", data_f);
    auto result = op_schema_->shape_infer(&node_, in, out_not_filled);
    EXPECT_EQ(result, status::invalid_shape);
}

TEST(op_schema_test, BiasAddBackprop) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackprop;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"data_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, BiasAddBackprop_nxc_infer_shape) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackprop;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    const int64_t channels = 16;
    logical_tensor_t lt_in
            = logical_tensor_init(0, {3, 64, 64, channels}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // no need to set attribute, NXC value should be default
    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {channels};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    // explicitly setting data_format to NXC
    const std::string default_data_f = "NXC";
    node_.set_attr("data_format", default_data_f);
    logical_tensor_t lt_out_expl = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out_expl {&lt_out_expl};

    op_schema_->shape_infer(&node_, in, out_expl);
    const std::vector<int64_t> infered_out_shape_expl
            = logical_tensor_wrapper(lt_out_expl).vdims();
    EXPECT_EQ(infered_out_shape_expl, expected_out_shape);
}

TEST(op_schema_test, BiasAddBackprop_ncx_infer_shape) {
    const op_kind_t op_kind_ = op_kind::BiasAddBackprop;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    const int64_t channels = 16;
    logical_tensor_t lt_in
            = logical_tensor_init(0, {3, channels, 64, 64}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};
    const std::string data_f = "NCX";
    node_.set_attr("data_format", data_f);

    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {channels};
    EXPECT_EQ(infered_out_shape, expected_out_shape);
}

TEST(op_schema_test, Clamp) {
    const op_kind_t op_kind_ = op_kind::Clamp;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"min", true}, {"max", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Clamp_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Clamp;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, ClampBackprop) {
    const op_kind_t op_kind_ = op_kind::ClampBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"min", true}, {"max", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, ClampBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ClampBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, ConvolutionBackpropData) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropData;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 9;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"output_padding", false}, {"groups", false},
            {"data_format", false}, {"filter_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, conv_bprop_data_ncx_oix_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropData;

    std::string data_format = "NCX";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3};
        const std::vector<int64_t> &in_output_shape = {};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 452, 452};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(op_schema_test, conv_bprop_data_nxc_oix_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropData;

    std::string data_format = "NXC";
    std::string filter_format = "OIX";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 32};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {16, 32 / groups, 3, 3};
        const std::vector<int64_t> &in_output_shape = {};
        const std::vector<int64_t> &expected_out_shape = {1, 452, 452, 16};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(op_schema_test, conv_bprop_data_nxc_xio_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropData;

    std::string data_format = "NXC";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, H, W, IC}
        const std::vector<int64_t> &in_data = {1, 224, 224, 32};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {3, 3, 32 / groups, 16};
        const std::vector<int64_t> &in_output_shape = {};
        const std::vector<int64_t> &expected_out_shape = {1, 452, 452, 16};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(op_schema_test, conv_bprop_data_ncx_xio_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropData;

    std::string data_format = "NCX";
    std::string filter_format = "XIO";

    std::vector<int64_t> groups_vec {1, 2, 4};

    for (auto groups : groups_vec) {
        // data shape {N, IC, H, W}
        const std::vector<int64_t> &in_data = {1, 32, 224, 224};
        // weight shape {OC, IC, KH, KW}
        const std::vector<int64_t> &in_weight = {3, 3, 32 / groups, 16};
        const std::vector<int64_t> &in_output_shape = {};
        const std::vector<int64_t> &expected_out_shape = {1, 16, 452, 452};

        verify_shape_infer_for_conv_bprop_data(op_kind_, data_format,
                filter_format, groups, in_data, in_weight, in_output_shape,
                expected_out_shape);
    }
}

TEST(op_schema_test, ConvolutionBackpropFilters) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropFilters;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 8;
    const std::map<std::string, bool> attrs_data
            = {{"strides", true}, {"pads_begin", true}, {"pads_end", true},
                    {"dilations", true}, {"auto_pad", false}, {"groups", false},
                    {"data_format", false}, {"filter_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, ConvolutionBackpropFilters_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ConvolutionBackpropFilters;

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

    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    set_conv_common_attr(node_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight_spatial_dims
            = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_output_delta
            = logical_tensor_init(2, in_output_delta, data_type::f32);
    std::vector<logical_tensor_t *> in {
            &lt_data, &lt_weight_spatial_dims, &lt_output_delta};
    logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);
}

TEST(op_schema_test, Divide) {
    const op_kind_t op_kind_ = op_kind::Divide;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Divide_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Divide;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Divide_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Divide;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, Elu) {
    const op_kind_t op_kind_ = op_kind::Elu;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"alpha", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Elu_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Elu;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, EluBackprop) {
    const op_kind_t op_kind_ = op_kind::EluBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"alpha", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, EluBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::EluBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Erf) {
    const op_kind_t op_kind_ = op_kind::Erf;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Erf_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Erf;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Exp) {
    const op_kind_t op_kind_ = op_kind::Exp;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Exp_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Exp;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, GELU) {
    const op_kind_t op_kind_ = op_kind::GELU;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, GELU_infer_shape) {
    const op_kind_t op_kind_ = op_kind::GELU;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, GELUBackprop) {
    const op_kind_t op_kind_ = op_kind::GELUBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, GELUBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::GELUBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, HardTanh) {
    const op_kind_t op_kind_ = op_kind::HardTanh;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"min", true}, {"max", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, HardTanh_infer_shape) {
    const op_kind_t op_kind_ = op_kind::HardTanh;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, HardTanhBackprop) {
    const op_kind_t op_kind_ = op_kind::HardTanhBackprop;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"min", true}, {"max", true}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, HardTanhBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::HardTanhBackprop;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Index) {
    const op_kind_t op_kind_ = op_kind::Index;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data;

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Index_infer_shape_unsupported) {
    const op_kind_t op_kind_ = op_kind::Index;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {3, 64, 64, 64}, data_type::f32);
    logical_tensor_t lt_indices
            = logical_tensor_init(1, {2, 4}, data_type::s32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_indices};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    auto status = op_schema_->shape_infer(&node_, in, out);
    EXPECT_EQ(status, status::unsupported);
}

TEST(op_schema_test, LayerNorm) {
    const op_kind_t op_kind_ = op_kind::LayerNorm;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 4;
    const std::map<std::string, bool> attrs_data
            = {{"keep_stats", false}, {"begin_norm_axis", false},
                    {"use_affine", false}, {"epsilon", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, LayerNorm_infer_shape) {
    const op_schema *op_schema_
            = op_schema_registry::get_op_schema(op_kind::LayerNorm);
    node_t node_ {op_kind::LayerNorm};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    // We test all available cases
    const std::vector<bool> keep_statses = {true, false};

    // TODO(qun) we should test multi begin_norm_axis attrs
    const int64_t begin_norm_axis = -1;

    node_.set_attr("begin_norm_axis", begin_norm_axis);

    for (auto keep_stats : keep_statses) {
        node_.set_attr("keep_stats", static_cast<bool>(keep_stats));
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

            op_schema_->shape_infer(&node_, in, out);

            const std::vector<int64_t> infered_out_shape
                    = logical_tensor_wrapper(lt_out).vdims();
            const std::vector<int64_t> expected_out_shape = {1, 3, 416, 416};
            EXPECT_EQ(infered_out_shape, expected_out_shape);

            const std::vector<int64_t> infered_out_strides
                    = logical_tensor_wrapper(lt_out).vstrides();
            const std::vector<int64_t> expected_out_strides
                    = compute_dense_strides(expected_out_shape);
            EXPECT_EQ(infered_out_strides, expected_out_strides);

            if (keep_stats) {
                // check infered shape and strides for mean
                const std::vector<int64_t> infered_mean_shape
                        = logical_tensor_wrapper(lt_mean).vdims();
                const std::vector<int64_t> expected_mean_shape = {1, 3, 416};
                EXPECT_EQ(infered_mean_shape, expected_mean_shape);

                const std::vector<int64_t> infered_mean_strides
                        = logical_tensor_wrapper(lt_mean).vstrides();
                const std::vector<int64_t> expected_mean_strides
                        = compute_dense_strides(expected_mean_shape);
                EXPECT_EQ(infered_mean_strides, expected_mean_strides);

                // check infered shape and strides for var
                const std::vector<int64_t> infered_var_shape
                        = logical_tensor_wrapper(lt_var).vdims();
                const std::vector<int64_t> expected_var_shape = {1, 3, 416};
                EXPECT_EQ(infered_var_shape, expected_var_shape);

                const std::vector<int64_t> infered_var_strides
                        = logical_tensor_wrapper(lt_var).vstrides();
                const std::vector<int64_t> expected_var_strides
                        = compute_dense_strides(expected_var_shape);
                EXPECT_EQ(infered_var_strides, expected_var_strides);
            }
        }
    }
}

TEST(op_schema_test, LayerNormBackprop) {
    const op_kind_t op_kind_ = op_kind::LayerNormBackprop;
    const size_t expected_in_size = 5;
    const size_t expected_out_size = 3;
    const size_t expected_attr_size = 4;
    const std::map<std::string, bool> attrs_data = {{"begin_norm_axis", false},
            {"use_affine", false}, {"epsilon", false}, {"use_stats", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, LayerNormBackprop_infer_shape) {
    const op_schema *op_schema_
            = op_schema_registry::get_op_schema(op_kind::LayerNormBackprop);
    node_t node_ {op_kind::LayerNormBackprop};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    // We test all available cases
    const std::vector<bool> use_affines = {true, false};

    for (auto use_affine : use_affines) {
        node_.set_attr("use_affine", static_cast<bool>(use_affine));
        for (const auto &ltype : layout_types) {
            logical_tensor_t lt_data = logical_tensor_init(
                    0, {1, 256, 64, 64}, data_type::f32, ltype);
            logical_tensor_t lt_gamma
                    = logical_tensor_init(1, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_beta
                    = logical_tensor_init(2, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_mean
                    = logical_tensor_init(3, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_variance
                    = logical_tensor_init(4, {1, 256}, data_type::f32, ltype);
            logical_tensor_t lt_in_delta = logical_tensor_init(
                    5, data_type::f32, layout_type::strided);
            logical_tensor_t lt_gamma_delta = logical_tensor_init(
                    6, data_type::f32, layout_type::strided);
            logical_tensor_t lt_beta_delta = logical_tensor_init(
                    7, data_type::f32, layout_type::strided);
            std::vector<logical_tensor_t *> lt_in {
                    &lt_data, &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
            std::vector<logical_tensor_t *> lt_out {
                    &lt_in_delta, &lt_gamma_delta, &lt_beta_delta};

            op_schema_->shape_infer(&node_, lt_in, lt_out);

            const std::vector<int64_t> infered_in_delta_shape
                    = logical_tensor_wrapper(lt_in_delta).vdims();
            const std::vector<int64_t> expected_in_delta_shape
                    = {1, 256, 64, 64};
            EXPECT_EQ(infered_in_delta_shape, expected_in_delta_shape);

            const std::vector<int64_t> infered_in_delta_strides
                    = logical_tensor_wrapper(lt_in_delta).vstrides();
            const std::vector<int64_t> expected_in_delta_strides
                    = compute_dense_strides(expected_in_delta_shape);
            EXPECT_EQ(infered_in_delta_strides, expected_in_delta_strides);

            if (use_affine) {
                const std::vector<int64_t> infered_gamma_delta_shape
                        = logical_tensor_wrapper(lt_gamma_delta).vdims();
                const std::vector<int64_t> expected_gamma_delta_shape
                        = {1, 256};
                EXPECT_EQ(
                        infered_gamma_delta_shape, expected_gamma_delta_shape);

                const std::vector<int64_t> infered_gamma_delta_strides
                        = logical_tensor_wrapper(lt_gamma_delta).vstrides();
                const std::vector<int64_t> expected_gamma_delta_strides
                        = compute_dense_strides(expected_gamma_delta_shape);
                EXPECT_EQ(infered_gamma_delta_strides,
                        expected_gamma_delta_strides);

                const std::vector<int64_t> infered_beta_delta_shape
                        = logical_tensor_wrapper(lt_beta_delta).vdims();
                const std::vector<int64_t> expected_beta_delta_shape = {1, 256};
                EXPECT_EQ(infered_beta_delta_shape, expected_beta_delta_shape);

                const std::vector<int64_t> infered_beta_delta_strides
                        = logical_tensor_wrapper(lt_beta_delta).vstrides();
                const std::vector<int64_t> expected_beta_delta_strides
                        = compute_dense_strides(expected_beta_delta_shape);
                EXPECT_EQ(infered_beta_delta_strides,
                        expected_beta_delta_strides);
            }
        }
    }
}

TEST(op_schema_test, Log) {
    const op_kind_t op_kind_ = op_kind::Log;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Log_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Log;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, LogSoftmax) {
    const op_kind_t op_kind_ = op_kind::LogSoftmax;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"axis", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, LogSoftmax_infer_shape) {
    const op_kind_t op_kind_ = op_kind::LogSoftmax;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, LogSoftmaxBackprop) {
    const op_kind_t op_kind_ = op_kind::LogSoftmaxBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"axis", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, LogSoftmaxBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::LogSoftmaxBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Maximum) {
    const op_kind_t op_kind_ = op_kind::Maximum;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Maximum_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Maximum;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Maximum_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Maximum;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, MaxPoolBackprop) {
    const op_kind_t op_kind_ = op_kind::MaxPoolBackprop;

    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 7;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"kernel", true},
            {"auto_pad", false}, {"dilations", false}, {"data_format", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, MaxPoolBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::MaxPoolBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}
TEST(op_schema_test, Minimum) {
    const op_kind_t op_kind_ = op_kind::Minimum;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Minimum_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Minimum;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Minimum_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Minimum;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, Multiply) {
    const op_kind_t op_kind_ = op_kind::Multiply;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Multiply_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Multiply;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Multiply_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Multiply;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, Pow) {
    const op_kind_t op_kind_ = op_kind::Pow;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"auto_broadcast", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Pow_no_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Pow;

    verify_shape_infer_for_arithmetic_op_no_broadcast(op_kind_);
}

TEST(op_schema_test, Pow_with_broadcast_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Pow;

    verify_shape_infer_for_arithmetic_op_with_broadcast(op_kind_);
}

TEST(op_schema_test, PowBackprop) {
    const op_kind_t op_kind_ = op_kind::PowBackprop;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, PowBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::PowBackprop;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_out_delta
            = logical_tensor_init(1, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_exp
            = logical_tensor_init(2, {1, 1, 224}, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_out_delta, &lt_exp};
    logical_tensor_t lt_in_delta = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_in_delta};

    op_schema_->shape_infer(&node_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper(lt_in_delta).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape, expected_out_shape);
}

TEST(op_schema_test, ReduceSum) {
    const op_kind_t op_kind_ = op_kind::ReduceSum;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"keep_dims", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, ReduceSum_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ReduceSum;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_in
            = logical_tensor_init(0, {1, 3, 224, 224}, data_type::f32);
    logical_tensor_t lt_axis_indices = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in, &lt_axis_indices};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    auto ret = op_schema_->shape_infer(&node_, in, out);
    EXPECT_EQ(ret, status::unsupported);
}

TEST(op_schema_test, ReLU) {
    const op_kind_t op_kind_ = op_kind::ReLU;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Relu_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ReLU;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, ReLUBackprop) {
    const op_kind_t op_kind_ = op_kind::ReLUBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, ReluBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::ReLUBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Round) {
    const op_kind_t op_kind_ = op_kind::Round;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Round_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Round;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Sigmoid) {
    const op_kind_t op_kind_ = op_kind::Sigmoid;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Sigmoid_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Sigmoid;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, SigmoidBackprop) {
    const op_kind_t op_kind_ = op_kind::SigmoidBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"use_dst", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SigmoidBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SigmoidBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, SoftMax) {
    const op_kind_t op_kind_ = op_kind::SoftMax;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"axis", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SoftMax_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SoftMax;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, SoftMaxBackprop) {
    const op_kind_t op_kind_ = op_kind::SoftMaxBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"axis", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SoftMaxBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SoftMaxBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Sqrt_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Sqrt;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Sqrt) {
    const op_kind_t op_kind_ = op_kind::Sqrt;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SoftPlus) {
    const op_kind_t op_kind_ = op_kind::SoftPlus;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"beta", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SoftPlus_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SoftPlus;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, SoftPlusBackprop) {
    const op_kind_t op_kind_ = op_kind::SoftPlusBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"beta", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, SoftPlusBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SoftPlusBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, SqrtBackprop) {
    const op_kind_t op_kind_ = op_kind::SqrtBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"use_dst", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Tanh) {
    const op_kind_t op_kind_ = op_kind::Tanh;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}
TEST(op_schema_test, SqrtBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::SqrtBackprop;

    verify_two_ins_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Square) {
    const op_kind_t op_kind_ = op_kind::Square;
    const size_t expected_in_size = 1;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 0;
    const std::map<std::string, bool> attrs_data = {};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, Square_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Square;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Tanh_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Tanh;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, TanhBackprop) {
    const op_kind_t op_kind_ = op_kind::TanhBackprop;
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 1;
    const std::map<std::string, bool> attrs_data = {{"use_dst", false}};

    verify_op_schema(op_kind_, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(op_schema_test, TanhBackprop_infer_shape) {
    const op_kind_t op_kind_ = op_kind::TanhBackprop;

    verify_single_in_identity_shape_infer(op_kind_);
}

TEST(op_schema_test, Wildcard) {
    const op_schema *op_schema = op_schema_registry::get_op_schema(kWildcard);
    auto inputs_option = op_schema->get_inputs_option();
    auto outputs_option = op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema::param_num_option::variadic);
    EXPECT_TRUE(outputs_option == op_schema::param_num_option::variadic);

    op_t wildcard_op {0, kWildcard, std::string("wildcard")};
    logical_tensor_t lt_in_0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_in_1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_out_0 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_out_1 = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_out_2 = logical_tensor_init(4, data_type::f32);

    wildcard_op.add_input(&lt_in_0);
    wildcard_op.add_input(&lt_in_1);
    wildcard_op.add_output(&lt_out_0);
    wildcard_op.add_output(&lt_out_1);
    wildcard_op.add_output(&lt_out_2);

    EXPECT_TRUE(op_schema->verify(&wildcard_op));
}

TEST(op_schema_test, Wildcard_infer_shape) {
    const op_kind_t op_kind_ = op_kind::Wildcard;
    const op_schema *op_schema_ = op_schema_registry::get_op_schema(op_kind_);
    node_t node_ {op_kind_};

    logical_tensor_t lt_in = logical_tensor_init(0, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_in};
    logical_tensor_t lt_out = logical_tensor_init(1, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    auto ret = op_schema_->shape_infer(&node_, in, out);
    EXPECT_EQ(ret, status::unsupported);
}

TEST(op_schema_test, optional_input) {
    const op_schema *bn_op_schema
            = op_schema_registry::get_op_schema(kBatchNormForwardTraining);
    auto inputs_option = bn_op_schema->get_inputs_option();
    auto outputs_option = bn_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema::param_num_option::optional);
    EXPECT_TRUE(outputs_option == op_schema::param_num_option::fixed);

    op_t bn_op {0, kBatchNormForwardTraining, std::string("bn")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_viance = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);
    logical_tensor_t lt_running_mean = logical_tensor_init(4, data_type::f32);
    logical_tensor_t lt_running_viance = logical_tensor_init(5, data_type::f32);
    logical_tensor_t lt_batch_mean = logical_tensor_init(6, data_type::f32);
    logical_tensor_t lt_batch_viance = logical_tensor_init(7, data_type::f32);

    bn_op.add_input(&lt_data);
    bn_op.add_input(&lt_mean);
    bn_op.add_input(&lt_viance);
    bn_op.add_output(&lt_output);
    bn_op.add_output(&lt_running_mean);
    bn_op.add_output(&lt_running_viance);
    bn_op.add_output(&lt_batch_mean);
    bn_op.add_output(&lt_batch_viance);

    bn_op.set_attr("epsilon", 0.001f);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));

    logical_tensor_t lt_gamma = logical_tensor_init(8, data_type::f32);
    bn_op.add_input(&lt_gamma);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));
    logical_tensor_t lt_beta = logical_tensor_init(9, data_type::f32);
    bn_op.add_input(&lt_beta);
    EXPECT_TRUE(bn_op_schema->verify(&bn_op));

    logical_tensor_t lt_false = logical_tensor_init(10, data_type::f32);
    bn_op.add_input(&lt_false);
    EXPECT_FALSE(bn_op_schema->verify(&bn_op));
}

TEST(op_schema_test, variadic_input) {
    const op_schema *concat_op_schema
            = op_schema_registry::get_op_schema(kConcat);
    auto inputs_option = concat_op_schema->get_inputs_option();
    auto outputs_option = concat_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema::param_num_option::variadic);
    EXPECT_TRUE(outputs_option == op_schema::param_num_option::fixed);

    op_t concat_op {0, kConcat, std::string("concat")};
    logical_tensor_t lt_data_0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t lt_data_1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t lt_data_2 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);

    concat_op.add_input(&lt_data_0);
    concat_op.add_input(&lt_data_1);
    concat_op.add_input(&lt_data_2);
    concat_op.add_output(&lt_output);

    concat_op.set_attr("axis", int64_t(0));
    EXPECT_TRUE(concat_op_schema->verify(&concat_op));
}

TEST(op_schema_test, variadic_input_negative) {
    const op_schema *concat_op_schema
            = op_schema_registry::get_op_schema(kConcat);

    op_t concat_op {0, kConcat, std::string("concat")};
    logical_tensor_t lt_output = logical_tensor_init(3, data_type::f32);

    concat_op.add_output(&lt_output);

    concat_op.set_attr("axis", int64_t(0));
    EXPECT_FALSE(concat_op_schema->verify(&concat_op));
}

TEST(op_schema_test, test_layernorm_optional_inputs) {
    const op_schema *ln_op_schema
            = op_schema_registry::get_op_schema(kLayerNorm);
    auto inputs_option = ln_op_schema->get_inputs_option();
    auto outputs_option = ln_op_schema->get_outputs_option();
    EXPECT_TRUE(inputs_option == op_schema::param_num_option::optional);
    EXPECT_TRUE(outputs_option == op_schema::param_num_option::optional);

    op_t ln_op {0, kLayerNorm, std::string("ln")};
    logical_tensor_t lt_data = logical_tensor_init(0, data_type::f32);

    logical_tensor_t lt_output = logical_tensor_init(1, data_type::f32);

    ln_op.add_input(&lt_data);

    ln_op.add_output(&lt_output);

    ln_op.set_attr("keep_stats", true);
    EXPECT_TRUE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_beta = logical_tensor_init(2, data_type::f32);
    ln_op.add_input(&lt_beta);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_gamma = logical_tensor_init(3, data_type::f32);
    ln_op.add_input(&lt_gamma);
    EXPECT_TRUE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_mean = logical_tensor_init(4, data_type::f32);
    ln_op.add_output(&lt_mean);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_variance = logical_tensor_init(5, data_type::f32);
    ln_op.add_output(&lt_variance);
    EXPECT_TRUE(ln_op_schema->verify(&ln_op));

    logical_tensor_t lt_false = logical_tensor_init(6, data_type::f32);
    ln_op.add_input(&lt_false);
    EXPECT_FALSE(ln_op_schema->verify(&ln_op));
}
