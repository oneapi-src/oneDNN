/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "interface/c_types_map.hpp"
#include "interface/shape_infer.hpp"

#include "backend/dnnl/common.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using graph::dims;

TEST(ShapeInfer, OneWayBroadcast) {
    using dims = graph::dims;
    dims src_shape {2, 3};
    dims dst1_shape {2};
    dims dst2_shape {2, 3};
    dims dst3_shape {4, 3};
    dims dst4_shape {1, 2, 3};

    ASSERT_EQ(graph::one_way_broadcast(dst1_shape, src_shape),
            graph::status::invalid_shape);

    ASSERT_EQ(graph::one_way_broadcast(dst2_shape, src_shape),
            graph::status::success);

    ASSERT_EQ(graph::one_way_broadcast(dst3_shape, src_shape),
            graph::status::invalid_shape);

    ASSERT_EQ(graph::one_way_broadcast(dst4_shape, src_shape),
            graph::status::success);
}

TEST(ShapeInfer, InvalidShapeForMatmul) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t weight0
            = utils::logical_tensor_init(1, {8}, graph::data_type::f32);
    graph::logical_tensor_t dst0
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs0 {&src0, &weight0};
    std::vector<graph::logical_tensor_t *> outputs0 {&dst0};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs0, outputs0),
            graph::status::invalid_shape);

    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t weight1
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t dst1
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs1 {&src1, &weight1};
    std::vector<graph::logical_tensor_t *> outputs1 {&dst1};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs1, outputs1),
            graph::status::invalid_shape);

    graph::logical_tensor_t src2
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight2
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t dst2
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs2 {&src2, &weight2};
    std::vector<graph::logical_tensor_t *> outputs2 {&dst2};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs2, outputs2),
            graph::status::invalid_shape);

    graph::logical_tensor_t src3
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight3
            = utils::logical_tensor_init(1, {4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst3
            = utils::logical_tensor_init(1, {4, 1}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs3 {&src3, &weight3};
    std::vector<graph::logical_tensor_t *> outputs3 {&dst3};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs3, outputs3),
            graph::status::invalid_shape);

    graph::logical_tensor_t src4
            = utils::logical_tensor_init(1, {8, 6, 3}, graph::data_type::f32);
    graph::logical_tensor_t weight4
            = utils::logical_tensor_init(1, {4, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst4
            = utils::logical_tensor_init(1, {4, 1, 2}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs4 {&src4, &weight4};
    std::vector<graph::logical_tensor_t *> outputs4 {&dst4};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs4, outputs4),
            graph::status::invalid_shape);

    graph::logical_tensor_t src5
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight5
            = utils::logical_tensor_init(1, {6, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst5
            = utils::logical_tensor_init(1, {8, 2}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs5 {&src5, &weight5};
    std::vector<graph::logical_tensor_t *> outputs5 {&dst5};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs5, outputs5),
            graph::status::invalid_shape);
}

TEST(ShapeInfer, InvalidShapeForConv) {
    using dims = graph::dnnl_impl::dims;

    graph::op_t conv_op {0, graph::op_kind::Convolution, std::string("conv")};
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 2);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(2,
            {8, 16, 222, 222}, graph::data_type::f32, graph::layout_type::any);

    std::vector<graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<graph::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            graph::status::invalid_shape);

    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            graph::status::invalid_shape);
}

TEST(ShapeInfer, InvalidShapeForPool) {
    graph::op_t max_pool_op {
            0, graph::op_kind::MaxPool, std::string("maxPool")};
    graph::op_t avg_pool_bk_op {
            0, graph::op_kind::AvgPoolBackward, std::string("avg_pool_bk_op")};
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t dst_lt2 = utils::logical_tensor_init(
            1, graph::data_type::f32, graph::layout_type::any);

    std::vector<graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<graph::logical_tensor_t *> outputs {&dst_lt};
    std::vector<graph::logical_tensor_t *> outputs2 {&dst_lt2};
    ASSERT_EQ(graph::infer_pool_bwd_output_shape(&max_pool_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_pool_bwd_output_shape(
                      &avg_pool_bk_op, inputs, outputs2),
            graph::status::unimplemented);
}

TEST(ShapeInferDeathTest, CanonicalizeError) {
#ifndef NDEBUG
    ASSERT_DEATH(graph::canonicalize({1, 2, 3, 4}, "XXXX"), "invalid format");
#endif
}

TEST(ShapeInfer, InferConvOuputShapeError) {
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    {
        std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<graph::dim_t> wei_shape = {8, 4, 1, 1};
        graph::data_type_t dtype = graph::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<graph::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<graph::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }

    {
        conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1, 1, 1});
        std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<graph::dim_t> wei_shape = {8, 8, 1, 1};
        graph::data_type_t dtype = graph::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<graph::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<graph::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }
}

TEST(ShapeInfer, InferConvBpropDataOuputShape) {
    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardData);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.set_attr<dims>(graph::op_attr::dst_shape, dims {8, 3, 224, 224});

    // prepare logical tensor
    graph::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, graph::data_type::f32);

    conv_op.add_input(diff_dst);
    conv_op.add_input(weights);
    conv_op.add_output(diff_src);
    std::vector<graph::logical_tensor_t *> inputs {&diff_src, &weights};
    std::vector<graph::logical_tensor_t *> outputs {&diff_dst};
    {
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "XXXXX");
        ASSERT_EQ(graph::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                graph::status::unimplemented);
    }
    {
        conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1, 1, 1});
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        ASSERT_EQ(graph::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }
}

TEST(ShapeInfer, InferConvtransposeNcxOixError) {
    graph::op_t conv {graph::op_kind::ConvTransposeBackwardData,
            graph::op_t::kind2str(graph::op_kind::ConvTransposeBackwardData)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";
    int64_t groups = 1;
    int64_t id = 0;
    conv.set_attr(graph::op_attr::strides, strides);
    conv.set_attr(graph::op_attr::pads_begin, pads_begin);
    conv.set_attr(graph::op_attr::pads_end, pads_end);
    conv.set_attr(graph::op_attr::dilations, dilations);
    conv.set_attr(graph::op_attr::auto_pad, auto_pad);
    conv.set_attr(graph::op_attr::data_format, data_format);
    conv.set_attr(graph::op_attr::weights_format, filter_format);
    conv.set_attr(graph::op_attr::groups, groups);

    auto lt_data = utils::logical_tensor_init(
            id++, {1, 1, 16, 16}, graph::data_type::f32);
    auto lt_weight = utils::logical_tensor_init(
            id++, {1, 1, 4, 4}, graph::data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 10, 10};

    auto lt_o = utils::logical_tensor_init(
            id++, graph::data_type::f32, graph::layout_type::strided);
    std::vector<graph::logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<graph::logical_tensor_t *> lt_out {&lt_o};
    {
        conv.set_attr<int64_t>(graph::op_attr::groups, 2);
        ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                graph::status::invalid_shape);
    }

    {
        conv.set_attr(graph::op_attr::groups, groups);
        conv.set_attr<graph::dims>(graph::op_attr::strides, {1, 1, 1, 1, 1});
        ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                graph::status::invalid_shape);
    }
}
