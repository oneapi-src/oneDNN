/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;
using impl::dims;

TEST(ShapeInfer, OneWayBroadcast) {
    using dims = impl::dims;
    dims src_shape {2, 3};
    dims dst1_shape {2};
    dims dst2_shape {2, 3};
    dims dst3_shape {4, 3};
    dims dst4_shape {1, 2, 3};

    ASSERT_EQ(impl::one_way_broadcast(dst1_shape, src_shape),
            impl::status::invalid_shape);

    ASSERT_EQ(impl::one_way_broadcast(dst2_shape, src_shape),
            impl::status::success);

    ASSERT_EQ(impl::one_way_broadcast(dst3_shape, src_shape),
            impl::status::invalid_shape);

    ASSERT_EQ(impl::one_way_broadcast(dst4_shape, src_shape),
            impl::status::success);
}

TEST(ShapeInfer, InvalidShapeForMatmul) {
    impl::op_t matmul {0, impl::op_kind::MatMul, std::string("matmul")};
    impl::logical_tensor_t src0
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);
    impl::logical_tensor_t weight0
            = utils::logical_tensor_init(1, {8}, impl::data_type::f32);
    impl::logical_tensor_t dst0
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs0 {&src0, &weight0};
    std::vector<impl::logical_tensor_t *> outputs0 {&dst0};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs0, outputs0),
            impl::status::invalid_shape);

    impl::logical_tensor_t src1
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);
    impl::logical_tensor_t weight1
            = utils::logical_tensor_init(1, {8, 6}, impl::data_type::f32);
    impl::logical_tensor_t dst1
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs1 {&src1, &weight1};
    std::vector<impl::logical_tensor_t *> outputs1 {&dst1};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs1, outputs1),
            impl::status::invalid_shape);

    impl::logical_tensor_t src2
            = utils::logical_tensor_init(1, {8, 6}, impl::data_type::f32);
    impl::logical_tensor_t weight2
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);
    impl::logical_tensor_t dst2
            = utils::logical_tensor_init(1, {4}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs2 {&src2, &weight2};
    std::vector<impl::logical_tensor_t *> outputs2 {&dst2};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs2, outputs2),
            impl::status::invalid_shape);

    impl::logical_tensor_t src3
            = utils::logical_tensor_init(1, {8, 6}, impl::data_type::f32);
    impl::logical_tensor_t weight3
            = utils::logical_tensor_init(1, {4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst3
            = utils::logical_tensor_init(1, {4, 1}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs3 {&src3, &weight3};
    std::vector<impl::logical_tensor_t *> outputs3 {&dst3};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs3, outputs3),
            impl::status::invalid_shape);

    impl::logical_tensor_t src4
            = utils::logical_tensor_init(1, {8, 6, 3}, impl::data_type::f32);
    impl::logical_tensor_t weight4
            = utils::logical_tensor_init(1, {4, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst4
            = utils::logical_tensor_init(1, {4, 1, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs4 {&src4, &weight4};
    std::vector<impl::logical_tensor_t *> outputs4 {&dst4};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs4, outputs4),
            impl::status::invalid_shape);

    impl::logical_tensor_t src5
            = utils::logical_tensor_init(1, {8, 6}, impl::data_type::f32);
    impl::logical_tensor_t weight5
            = utils::logical_tensor_init(1, {6, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst5
            = utils::logical_tensor_init(1, {8, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t *> inputs5 {&src5, &weight5};
    std::vector<impl::logical_tensor_t *> outputs5 {&dst5};
    ASSERT_EQ(impl::infer_matmul_output_shape(&matmul, inputs5, outputs5),
            impl::status::invalid_shape);
}

TEST(ShapeInfer, InvalidShapeForConv) {
    impl::op_t conv_op {0, impl::op_kind::Convolution, std::string("conv")};
    conv_op.set_attr<dims>(impl::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(impl::op_attr::groups, 2);
    conv_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(impl::op_attr::filter_format, "OIX");

    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<impl::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(impl::infer_conv_output_shape(&conv_op, inputs, outputs),
            impl::status::invalid_shape);
    ASSERT_EQ(impl::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            impl::status::invalid_shape);

    conv_op.set_attr<int64_t>(impl::op_attr::groups, 1);
    conv_op.set_attr<dims>(impl::op_attr::dilations, dims {1, 1, 1});
    ASSERT_EQ(impl::infer_conv_output_shape(&conv_op, inputs, outputs),
            impl::status::invalid_shape);
    ASSERT_EQ(impl::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            impl::status::invalid_shape);
}

TEST(ShapeInfer, InvalidShapeForPool) {
    impl::op_t max_pool_op {0, impl::op_kind::MaxPool, std::string("maxPool")};
    impl::op_t avg_pool_bk_op {
            0, impl::op_kind::AvgPoolBackprop, std::string("avg_pool_bk_op")};
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t dst_lt2 = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<impl::logical_tensor_t *> outputs {&dst_lt};
    std::vector<impl::logical_tensor_t *> outputs2 {&dst_lt2};
    ASSERT_EQ(impl::infer_pool_bwd_output_shape(&max_pool_op, inputs, outputs),
            impl::status::invalid_shape);
    ASSERT_EQ(impl::infer_pool_bwd_output_shape(
                      &avg_pool_bk_op, inputs, outputs2),
            impl::status::unimplemented);
}

TEST(ShapeInfer, CanonicalizeError) {
#ifndef NDEBUG
    ASSERT_DEATH(impl::canonicalize({1, 2, 3, 4}, "XXXX"), "invalid format");
#endif
}

TEST(ShapeInfer, InferConvOuputShapeError) {
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>(impl::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(impl::op_attr::groups, 1);
    conv_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(impl::op_attr::filter_format, "OIX");
    {
        std::vector<impl::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<impl::dim_t> wei_shape = {8, 4, 1, 1};
        impl::data_type_t dtype = impl::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<impl::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<impl::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(impl::infer_conv_output_shape(&conv_op, inputs, outputs),
                impl::status::invalid_shape);
    }

    {
        conv_op.set_attr<dims>(impl::op_attr::strides, dims {1, 1, 1, 1, 1});
        std::vector<impl::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<impl::dim_t> wei_shape = {8, 8, 1, 1};
        impl::data_type_t dtype = impl::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<impl::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<impl::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(impl::infer_conv_output_shape(&conv_op, inputs, outputs),
                impl::status::invalid_shape);
    }
}

TEST(ShapeInfer, InferConvBpropDataOuputShape) {
    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>(impl::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op.set_attr<int64_t>(impl::op_attr::groups, 1);
    conv_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(impl::op_attr::filter_format, "OIX");
    conv_op.set_attr<dims>(impl::op_attr::output_shape, dims {8, 3, 224, 224});

    // prepare logical tensor
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32);

    conv_op.add_input(diff_dst);
    conv_op.add_input(weights);
    conv_op.add_output(diff_src);
    std::vector<impl::logical_tensor_t *> inputs {&diff_src, &weights};
    std::vector<impl::logical_tensor_t *> outputs {&diff_dst};
    {
        conv_op.set_attr<std::string>(impl::op_attr::data_format, "XXXXX");
        ASSERT_EQ(impl::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                impl::status::unimplemented);
    }
    {
        conv_op.set_attr<dims>(impl::op_attr::strides, dims {1, 1, 1, 1, 1});
        conv_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
        ASSERT_EQ(impl::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                impl::status::invalid_shape);
    }
}

TEST(ShapeInfer, InferConvtransposeNcxOixError) {
    impl::op_t conv {impl::op_kind::ConvTransposeBackpropData,
            impl::op_t::kind2str(impl::op_kind::ConvTransposeBackpropData)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";
    int64_t groups = 1;
    int64_t id = 0;
    conv.set_attr(impl::op_attr::strides, strides);
    conv.set_attr(impl::op_attr::pads_begin, pads_begin);
    conv.set_attr(impl::op_attr::pads_end, pads_end);
    conv.set_attr(impl::op_attr::dilations, dilations);
    conv.set_attr(impl::op_attr::auto_pad, auto_pad);
    conv.set_attr(impl::op_attr::data_format, data_format);
    conv.set_attr(impl::op_attr::filter_format, filter_format);
    conv.set_attr(impl::op_attr::groups, groups);

    auto lt_data = utils::logical_tensor_init(
            id++, {1, 1, 16, 16}, impl::data_type::f32);
    auto lt_weight = utils::logical_tensor_init(
            id++, {1, 1, 4, 4}, impl::data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 10, 10};

    auto lt_o = utils::logical_tensor_init(
            id++, impl::data_type::f32, impl::layout_type::strided);
    std::vector<impl::logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<impl::logical_tensor_t *> lt_out {&lt_o};
    {
        conv.set_attr<int64_t>(impl::op_attr::groups, 2);
        ASSERT_EQ(impl::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                impl::status::invalid_shape);
    }

    {
        conv.set_attr(impl::op_attr::groups, groups);
        conv.set_attr<impl::dims>(impl::op_attr::strides, {1, 1, 1, 1, 1});
        ASSERT_EQ(impl::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                impl::status::invalid_shape);
    }
}
