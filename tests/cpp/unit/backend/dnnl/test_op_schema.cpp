/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "backend/dnnl/dnnl_shape_infer.hpp"
#include "backend/dnnl/internal_ops.hpp"

#include "cpp/unit/utils.hpp"

using namespace dnnl::graph::impl;
using namespace dnnl::graph::tests::unit::utils;

TEST(OpSchema, Int8Conv) {
    std::set<op_kind_t> conv_kinds = {impl::dnnl_impl::op_kind::int8_conv,
            impl::dnnl_impl::op_kind::int8_conv_relu};
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 12;
    const std::map<std::string, bool> attrs_data = {{"strides", true},
            {"pads_begin", true}, {"pads_end", true}, {"dilations", true},
            {"auto_pad", false}, {"groups", false}, {"data_format", false},
            {"filter_format", false}, {"qtype", false}, {"axis", false},
            {"scales", true}, {"zps", true}};

    for (auto k : conv_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }

    std::set<op_kind_t> conv_bias_kinds
            = {impl::dnnl_impl::op_kind::int8_conv_bias,
                    impl::dnnl_impl::op_kind::int8_conv_bias_relu};
    const size_t expected_in_size2 = 3;
    for (auto k : conv_bias_kinds) {
        verify_op_schema(k, expected_in_size2, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, Int8Matmul) {
    std::set<op_kind_t> matmul_kinds = {impl::dnnl_impl::op_kind::int8_matmul,
            impl::dnnl_impl::op_kind::int8_matmul_relu,
            impl::dnnl_impl::op_kind::int8_matmul_sigmoid,
            impl::dnnl_impl::op_kind::int8_matmul_gelu};
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 6;
    const std::map<std::string, bool> attrs_data
            = {{"transpose_a", false}, {"transpose_b", false}, {"qtype", false},
                    {"axis", false}, {"scales", true}, {"zps", true}};

    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }

    std::set<op_kind_t> matmul_bias_kinds
            = {impl::dnnl_impl::op_kind::int8_matmul_bias,
                    impl::dnnl_impl::op_kind::int8_matmul_bias_relu,
                    impl::dnnl_impl::op_kind::int8_matmul_bias_sigmoid,
                    impl::dnnl_impl::op_kind::int8_matmul_bias_gelu};
    const size_t expected_in_size2 = 3;
    for (auto k : matmul_bias_kinds) {
        verify_op_schema(k, expected_in_size2, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, InferConvBiasOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bias,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(3, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto unchanged_pads_begin
            = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto unchanged_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);
    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, InferConvBiasOutputShapeAutoPad) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bias,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {}; // empty pads_begin
    std::vector<int64_t> pads_end = {}; // empty pads_end
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            "SAME_UPPER", data_format, filter_format, groups);

    auto lt_data = logical_tensor_init(0, {1, 1, 5, 5}, data_type::f32);
    auto lt_weight = logical_tensor_init(1, {1, 1, 3, 3}, data_type::f32);
    auto lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    auto lt_o = logical_tensor_init(3, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);

    const auto infered_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape {1, 1, 5, 5};
    ASSERT_EQ(infered_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConvBiasOutputShapeAutoPadNoDefaultAttribute) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bias,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {};
    std::vector<int64_t> pads_end = {};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
            "SAME_LOWER", data_format, filter_format, groups);

    auto lt_data = logical_tensor_init(0, {1, 1, 5, 5}, data_type::f32);
    auto lt_weight = logical_tensor_init(1, {1, 1, 3, 3}, data_type::f32);
    auto lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    auto lt_o = logical_tensor_init(3, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);

    const auto infered_out_shape = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape {1, 1, 3, 3};
    ASSERT_EQ(infered_out_shape, expected_out_shape);
}

TEST(OpSchema, InferConv3dBiasOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_node {impl::dnnl_impl::op_kind::conv_bias,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias)};
    std::vector<int64_t> strides = {2, 2, 2};
    std::vector<int64_t> pads_begin = {1, 1, 1};
    std::vector<int64_t> pads_end = {2, 2, 2};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_node, strides, pads_begin, pads_end, dilations,
            "None", data_format, filter_format, groups);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(2, {1}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(3, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_node, lt_in, lt_out);
    auto unchanged_pads_begin
            = a_node.get_attr<std::vector<int64_t>>("pads_begin");
    auto unchanged_pads_end = a_node.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);
    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, ConvBiasAddElu) {
    const op_kind_t conv_kind = impl::dnnl_impl::op_kind::conv_bias_add_elu;
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

TEST(OpSchema, ConvBiasAddRelu6) {
    const op_kind_t conv_kind = impl::dnnl_impl::op_kind::conv_bias_add_relu6;
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

TEST(OpSchema, InferConvBiasAddEluOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias_add_elu);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bias_add_elu,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias_add_elu)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("alpha", 1.f);

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
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto unchanged_pads_begin
            = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto unchanged_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, ConvBiasElu) {
    const op_kind_t conv_kind = impl::dnnl_impl::op_kind::conv_bias_elu;
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

TEST(OpSchema, ConvBiasHardtanh) {
    std::set<op_kind_t> conv_kinds
            = {impl::dnnl_impl::op_kind::conv_bias_hardtanh,
                    impl::dnnl_impl::op_kind::conv_bias_relu6};
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

TEST(OpSchema, InferSqueezeOutputShape) {
    const op_kind_t kind = impl::dnnl_impl::op_kind::squeeze;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);
    std::vector<std::vector<int64_t>> axes_list {{1}, {1, 2}, {-1}, {-1, -2}};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 1, 4, 5}, {3, 1, 1, 4, 5}, {3, 4, 5, 1}, {3, 4, 5, 1, 1}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
    for (size_t i = 0; i < axes_list.size(); ++i) {
        op_t op {kind, "squeeze"};
        op.set_attr<std::vector<int64_t>>("axes", axes_list[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shapes[i], data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = dst_shapes[i];
        EXPECT_EQ(infered_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferExpandOutputShape) {
    const op_kind_t kind = impl::dnnl_impl::op_kind::expand;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);

    const std::vector<int64_t> src_shape {4};
    const int64_t expand_to {4};

    const std::vector<std::vector<int64_t>> dst_shapes {
            {1, 1, 1, 4}, {1, 1, 4, 1}};
    const std::vector<std::string> insert_1dim {"before", "after"};

    for (size_t i = 0; i < dst_shapes.size(); ++i) {
        op_t op {kind, "expand"};
        op.set_attr<int64_t>("expand_to", expand_to);
        op.set_attr<std::string>("insert_1dim", insert_1dim[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shape, data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = dst_shapes[i];
        EXPECT_EQ(infered_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferExpandOutputShapeBasedOnAxes) {
    const op_kind_t kind = impl::dnnl_impl::op_kind::expand;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);

    const std::vector<std::vector<int64_t>> axes_list {
            {1}, {1, 2}, {-1}, {-1, -2}};
    const std::vector<std::vector<int64_t>> src_shapes {
            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
    const std::vector<std::vector<int64_t>> dst_shapes {
            {3, 1, 4, 5}, {3, 1, 1, 4, 5}, {3, 4, 5, 1}, {3, 4, 5, 1, 1}};

    for (size_t i = 0; i < axes_list.size(); ++i) {
        op_t op {kind, "expand"};
        op.set_attr<std::vector<int64_t>>("axes", axes_list[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shapes[i], data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = dst_shapes[i];
        EXPECT_EQ(infered_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferConvBiasAddReluWithNxcFormat) {
    infer_conv_shape(impl::dnnl_impl::op_kind::conv_bias);
    infer_conv_shape(impl::dnnl_impl::op_kind::conv_bias_add);
    infer_conv_shape(impl::dnnl_impl::op_kind::conv_bias_add_relu);
}

TEST(OpSchema, MatmulFusion) {
    std::set<op_kind_t> matmul_kinds = {impl::dnnl_impl::op_kind::matmul_relu,
            impl::dnnl_impl::op_kind::matmul_elu,
            impl::dnnl_impl::op_kind::matmul_sigmoid,
            impl::dnnl_impl::op_kind::matmul_hardtanh,
            impl::dnnl_impl::op_kind::matmul_gelu};
    const size_t expected_in_size = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"transpose_a", false}, {"transpose_b", false}};

    for (auto k : matmul_kinds) {
        verify_op_schema(k, expected_in_size, expected_out_size,
                expected_attr_size, attrs_data);
    }

    const std::set<size_t> expected_in_sizes = {2, 3};
    for (auto s : expected_in_sizes) {
        verify_op_schema(op_kind::MatMul, s, expected_out_size,
                expected_attr_size, attrs_data);
    }
}

TEST(OpSchema, MatmulBiasFusion) {
    std::set<op_kind_t> matmul_kinds = {impl::dnnl_impl::op_kind::matmul_bias,
            impl::dnnl_impl::op_kind::matmul_bias_relu,
            impl::dnnl_impl::op_kind::matmul_bias_sigmoid,
            impl::dnnl_impl::op_kind::matmul_bias_gelu};
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

TEST(OpSchema, MatmulBiasAdd) {
    std::set<op_kind_t> matmul_kinds
            = {impl::dnnl_impl::op_kind::matmul_bias_add,
                    impl::dnnl_impl::op_kind::matmul_bias_add_relu};
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

TEST(OpSchema, MatmulBiasBn) {
    op_kind_t matmul_kinds = impl::dnnl_impl::op_kind::matmul_bias_bn;
    const size_t expected_in_size = 7;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data = {
            {"epsilon", true}, {"transpose_a", false}, {"transpose_b", false}};
    verify_op_schema(matmul_kinds, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, MatMulBiasElu) {
    op_kind_t matmul_kinds = impl::dnnl_impl::op_kind::matmul_bias_elu;
    const size_t expected_in_size = 3;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 3;
    const std::map<std::string, bool> attrs_data
            = {{"alpha", true}, {"transpose_a", false}, {"transpose_b", false}};
    verify_op_schema(matmul_kinds, expected_in_size, expected_out_size,
            expected_attr_size, attrs_data);
}

TEST(OpSchema, MatmulBiasHardtanh) {
    std::set<op_kind_t> matmul_kinds
            = {impl::dnnl_impl::op_kind::matmul_bias_hardtanh,
                    impl::dnnl_impl::op_kind::matmul_bias_relu6};
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

TEST(OpSchema, InferMatmulBiasOutputShape) {
    const op_schema_t *matmul_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::matmul_bias);

    op_t matmul_op {impl::dnnl_impl::op_kind::matmul_bias,
            op_t::kind2str(impl::dnnl_impl::op_kind::matmul_bias)};
    bool transpose_a = true;
    matmul_op.set_attr("transpose_a", transpose_a);

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

    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper_t(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {1000}, data_type::f32);
    logical_tensor_t lt_o2 = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out2);

    auto &infered_out_shape2 = lt_o2.dims;
    EXPECT_EQ(lt_o2.ndims, 1);
    EXPECT_EQ(infered_out_shape2[0], 1000);

    // test >2 dims matmul
    lt_in1 = logical_tensor_init(0, {3, 1, 10, 1024, 64}, data_type::f32);
    lt_in2 = logical_tensor_init(1, {5, 1, 1024, 1000}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {3, 5, 10, 64, 1000}, data_type::f32);
    logical_tensor_t lt_o3 = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> lt_out3 {&lt_o3};
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out3);

    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper_t(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(infered_out_shape3, expected_out_shape3);
}

TEST(OpSchema, InferMatmulBiasAddOutputShape) {
    const op_schema_t *matmul_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::matmul_bias_add);

    op_t matmul_op {impl::dnnl_impl::op_kind::matmul_bias_add,
            op_t::kind2str(impl::dnnl_impl::op_kind::matmul_bias_add)};
    bool transpose_a = true;
    matmul_op.set_attr("transpose_a", transpose_a);

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

    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out1);

    const std::vector<int64_t> infered_out_shape1
            = logical_tensor_wrapper_t(lt_o1).vdims();
    const std::vector<int64_t> expected_out_shape1 = {64, 1000};
    EXPECT_EQ(infered_out_shape1, expected_out_shape1);

    // test 1 dims matmul
    lt_in1 = logical_tensor_init(0, {1024}, data_type::f32);
    lt_in3 = logical_tensor_init(2, {1000}, data_type::f32);
    lt_in4 = logical_tensor_init(3, {1000}, data_type::f32);
    logical_tensor_t lt_o2 = logical_tensor_init(4, data_type::f32);
    std::vector<logical_tensor_t *> lt_out2 {&lt_o2};
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out2);

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
    matmul_op_schema->shape_infer(&matmul_op, lt_in, lt_out3);

    const std::vector<int64_t> infered_out_shape3
            = logical_tensor_wrapper_t(lt_o3).vdims();
    const std::vector<int64_t> expected_out_shape3 = {3, 5, 10, 64, 1000};
    EXPECT_EQ(infered_out_shape3, expected_out_shape3);
}

TEST(OpSchema, InferBatchNormInferenceBiasAddOutputShape) {
    std::set<op_kind_t> bn_kinds
            = {op_kind::BatchNormInference, impl::dnnl_impl::op_kind::bn_relu};
    for (auto cur_kind : bn_kinds) {
        const op_schema_t *bn_op_schema
                = op_schema_registry_t::get_op_schema(cur_kind);
        EXPECT_TRUE(nullptr != bn_op_schema);
        op_t bn_op {cur_kind, op_t::kind2str(cur_kind)};
        bn_op.set_attr<float>("epsilon", 0.001f);

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

        bn_op_schema->shape_infer(&bn_op, lt_in, lt_out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();
        const std::vector<int64_t> expected_out_shape = {1, 256, 64, 64};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper_t(lt_o).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);
    }
}

TEST(OpSchema, InferConvBnOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bn);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bn,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bn)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("epsilon", epsilon);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(1, {512}, data_type::f32);
    logical_tensor_t lt_gamma
            = logical_tensor_init(1, {1, 512}, data_type::f32);
    logical_tensor_t lt_beta = logical_tensor_init(2, {1, 512}, data_type::f32);
    logical_tensor_t lt_mean = logical_tensor_init(3, {1, 512}, data_type::f32);
    logical_tensor_t lt_variance
            = logical_tensor_init(4, {1, 512}, data_type::f32);
    logical_tensor_t lt_o
            = logical_tensor_init(5, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias,
            &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto unchanged_pads_begin
            = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto unchanged_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, InferConvBnAddOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bn_add);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bn_add,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bn_add)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("epsilon", epsilon);

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
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto infered_pads_begin = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(infered_pads_begin[0], 1);
    EXPECT_EQ(infered_pads_begin[1], 1);
    EXPECT_EQ(infered_pads_end[0], 2);
    EXPECT_EQ(infered_pads_end[1], 2);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, InferConvBnReluOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bn_relu);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bn_relu,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bn_relu)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("epsilon", epsilon);

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
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto unchanged_pads_begin
            = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto unchanged_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(unchanged_pads_begin, pads_begin);
    EXPECT_EQ(unchanged_pads_end, pads_end);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, InferConvBiasBnOutputShape) {
    std::set<op_kind_t> op_kinds = {impl::dnnl_impl::op_kind::conv_bias_bn,
            impl::dnnl_impl::op_kind::conv_bias_bn_relu};
    for (auto a_op_kind : op_kinds) {
        const op_schema_t *a_op_schema
                = op_schema_registry_t::get_op_schema(a_op_kind);
        EXPECT_TRUE(nullptr != a_op_schema);
        op_t a_op {a_op_kind, op_t::kind2str(a_op_kind)};
        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> pads_begin = {1, 1};
        std::vector<int64_t> pads_end = {2, 2};
        std::vector<int64_t> dilations = {1, 1};
        float epsilon = 0.001f;
        std::string data_format = "NCX";
        std::string filter_format = "OIX";
        int64_t groups = 1;

        set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations,
                "None", data_format, filter_format, groups);
        a_op.set_attr("epsilon", epsilon);

        logical_tensor_t lt_data
                = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
        logical_tensor_t lt_weight
                = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
        logical_tensor_t lt_bias
                = logical_tensor_init(1, {512}, data_type::f32);
        logical_tensor_t lt_gamma
                = logical_tensor_init(1, {1, 512}, data_type::f32);
        logical_tensor_t lt_beta
                = logical_tensor_init(2, {1, 512}, data_type::f32);
        logical_tensor_t lt_mean
                = logical_tensor_init(3, {1, 512}, data_type::f32);
        logical_tensor_t lt_variance
                = logical_tensor_init(4, {1, 512}, data_type::f32);
        logical_tensor_t lt_o
                = logical_tensor_init(5, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias,
                &lt_gamma, &lt_beta, &lt_mean, &lt_variance};
        std::vector<logical_tensor_t *> lt_out {&lt_o};
        a_op_schema->shape_infer(&a_op, lt_in, lt_out);
        auto unchanged_pads_begin
                = a_op.get_attr<std::vector<int64_t>>("pads_begin");
        auto unchanged_pads_end
                = a_op.get_attr<std::vector<int64_t>>("pads_end");
        EXPECT_EQ(unchanged_pads_begin, pads_begin);
        EXPECT_EQ(unchanged_pads_end, pads_end);

        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_o).vdims();
        const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper_t(lt_o).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);
    }
}

TEST(OpSchema, InferConvBnAddReluOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bn_add_relu);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bn_add_relu,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bn_add_relu)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("epsilon", epsilon);

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
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto infered_pads_begin = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(infered_pads_begin[0], 1);
    EXPECT_EQ(infered_pads_begin[1], 1);
    EXPECT_EQ(infered_pads_end[0], 2);
    EXPECT_EQ(infered_pads_end[1], 2);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, InferConvBiasBnAddReluOutputShape) {
    const op_schema_t *a_op_schema = op_schema_registry_t::get_op_schema(
            impl::dnnl_impl::op_kind::conv_bias_bn_add_relu);
    EXPECT_TRUE(nullptr != a_op_schema);
    op_t a_op {impl::dnnl_impl::op_kind::conv_bias_bn_add_relu,
            op_t::kind2str(impl::dnnl_impl::op_kind::conv_bias_bn_add_relu)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    float epsilon = 0.001f;
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(a_op, strides, pads_begin, pads_end, dilations, "None",
            data_format, filter_format, groups);
    a_op.set_attr("epsilon", epsilon);

    logical_tensor_t lt_data
            = logical_tensor_init(0, {1, 256, 64, 64}, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {512, 256, 3, 3}, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(1, {512}, data_type::f32);
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
    std::vector<logical_tensor_t *> lt_in {&lt_data, &lt_weight, &lt_bias,
            &lt_gamma, &lt_beta, &lt_mean, &lt_variance, &lt_add};
    std::vector<logical_tensor_t *> lt_out {&lt_o};
    a_op_schema->shape_infer(&a_op, lt_in, lt_out);
    auto infered_pads_begin = a_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = a_op.get_attr<std::vector<int64_t>>("pads_end");
    EXPECT_EQ(infered_pads_begin[0], 1);
    EXPECT_EQ(infered_pads_begin[1], 1);
    EXPECT_EQ(infered_pads_end[0], 2);
    EXPECT_EQ(infered_pads_end[1], 2);

    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 512, 33, 33};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

TEST(OpSchema, ConvBiasFusion) {
    std::set<op_kind_t> conv_kinds = {impl::dnnl_impl::op_kind::conv_bias,
            impl::dnnl_impl::op_kind::conv_bias_abs,
            impl::dnnl_impl::op_kind::conv_bias_relu,
            impl::dnnl_impl::op_kind::conv_bias_sigmoid,
            impl::dnnl_impl::op_kind::conv_bias_sqrt,
            impl::dnnl_impl::op_kind::conv_bias_square,
            impl::dnnl_impl::op_kind::conv_bias_tanh};
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

TEST(OpSchema, DnnlBinary) {
    op_kind_t op_kind = dnnl_impl::op_kind::dnnl_binary;
    const size_t expected_in_size_lower = 2;
    const size_t expected_out_size = 1;
    const size_t expected_attr_size = 2;
    const std::map<std::string, bool> attrs_data
            = {{"auto_broadcast", false}, {"alg_kind", true}};
    verify_op_schema(op_kind, expected_in_size_lower, expected_out_size,
            expected_attr_size, attrs_data);

    const size_t expected_in_size_upper = std::numeric_limits<size_t>::max();
    verify_op_schema(op_kind, expected_in_size_upper, expected_out_size,
            expected_attr_size, attrs_data);
}
