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

#include <tuple>

#include "interface/shape_infer.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_shape_infer.hpp"
#include "backend/dnnl/internal_attrs.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;
namespace dnnl_impl = impl::dnnl_impl;

TEST(DnnlInferShape, InferDnnlConvOutputShape) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    using ltw = impl::logical_tensor_wrapper_t;

    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>(impl::op_attr::strides, {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::dilations, {1, 1});
    conv_op.set_attr<dims>(impl::op_attr::pads_begin, {0, 0});
    conv_op.set_attr<dims>(impl::op_attr::pads_end, {0, 0});
    conv_op.set_attr<int64_t>(impl::op_attr::groups, 1);
    conv_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(impl::op_attr::filter_format, "OIX");

    conv_op.set_attr<std::string>(impl::dnnl_impl::op_attr::dw_type, "k3s2p1");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 1, -1, -1}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    std::vector<impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(impl::dnnl_impl::infer_dnnl_conv_output_shape(
                      &conv_op, inputs, outputs),
            impl::status::success);

    auto output_dims = ltw(outputs[0]).vdims();
    dims output_dims_ref {1U, 1U, 1U, 1U};

    ASSERT_TRUE(std::equal(
            output_dims.begin(), output_dims.end(), output_dims_ref.begin()));

    auto output_strides = ltw(outputs[0]).vstrides();
    dims output_strides_ref {1U, 1U, 1U, 1U}; // add value
    ASSERT_TRUE(std::equal(output_strides.begin(), output_strides.end(),
            output_strides_ref.begin()));
}

TEST(DnnlShapeInfer, InferFromGroupOutputShape) {
    using namespace dnnl::graph::tests::unit::utils;
    size_t id = 0;

    {
        impl::dims dims {1, 2, 2, 2};
        auto op = std::make_shared<impl::op_t>(
                id++, impl::op_kind::Wildcard, "wildcard");
        impl::logical_tensor_t in_lt
                = logical_tensor_init(id++, dims, impl::data_type::f32);
        impl::logical_tensor_t out_lt = logical_tensor_init(
                id++, {1, -1, 2, 2}, impl::data_type::f32);
        op->set_attr<int64_t>(impl::op_attr::groups, 2);
        std::vector<impl::logical_tensor_t *> inputs {&in_lt};
        std::vector<impl::logical_tensor_t *> outputs {&out_lt};
        ASSERT_EQ(dnnl_impl::infer_from_group_output_shape(
                          op.get(), inputs, outputs),
                impl::status::success);
        const auto ret_dims = impl::logical_tensor_wrapper_t(&out_lt).vdims();
        ASSERT_EQ(ret_dims.size(), 3U);
        ASSERT_EQ(ret_dims[0], 4U);
    }

    {
        impl::dims dims {1, 2, 2, 2};
        auto op = std::make_shared<impl::op_t>(
                id++, impl::op_kind::Wildcard, "wildcard");
        impl::logical_tensor_t in_lt
                = logical_tensor_init(id++, dims, impl::data_type::f32);
        impl::logical_tensor_t out_lt = logical_tensor_init(
                id++, {1, -1, 2, 2}, impl::data_type::f32);
        op->set_attr<int64_t>(impl::op_attr::groups, 2);
        op->set_attr<bool>(dnnl_impl::op_attr::is_convtranspose, true);
        std::vector<impl::logical_tensor_t *> inputs {&in_lt};
        std::vector<impl::logical_tensor_t *> outputs {&out_lt};
        ASSERT_EQ(dnnl_impl::infer_from_group_output_shape(
                          op.get(), inputs, outputs),
                impl::status::success);
        const auto ret_dims = impl::logical_tensor_wrapper_t(&out_lt).vdims();
        ASSERT_EQ(ret_dims.size(), 3U);
        ASSERT_EQ(ret_dims[1], 4);
    }
}

TEST(DnnlShapeInfer, InferBnFoldingOutputShape) {
    using namespace dnnl::graph::tests::unit::utils;
    using item_type = std::tuple<impl::dims, impl::dims, impl::dims, impl::dims,
            impl::status_t>;
    std::vector<item_type> items {
            item_type(impl::dims {1, 2}, impl::dims {1, 2}, impl::dims {1, 2},
                    impl::dims {1, 2}, impl::status::success),
            item_type(impl::dims {1}, impl::dims {1, 2}, impl::dims {1, 2},
                    impl::dims {-1}, impl::status::invalid_shape),
            item_type(impl::dims {1, 2}, impl::dims {1}, impl::dims {-1},
                    impl::dims {1, 2}, impl::status::invalid_shape),
    };

    size_t id = 0;
    for (const auto &item : items) {
        auto &in0 = std::get<0>(item);
        auto &in1 = std::get<1>(item);
        auto &out0 = std::get<2>(item);
        auto &out1 = std::get<3>(item);
        auto status = std::get<4>(item);
        auto op = std::make_shared<impl::op_t>(
                id++, impl::op_kind::Wildcard, "wildcard");
        impl::logical_tensor_t in_lt0
                = logical_tensor_init(id++, in0, impl::data_type::f32);
        impl::logical_tensor_t in_lt1
                = logical_tensor_init(id++, in1, impl::data_type::f32);
        impl::logical_tensor_t out_lt0
                = logical_tensor_init(id++, out0, impl::data_type::f32);
        impl::logical_tensor_t out_lt1
                = logical_tensor_init(id++, out1, impl::data_type::f32);
        std::vector<impl::logical_tensor_t *> inputs {&in_lt0, &in_lt1};
        std::vector<impl::logical_tensor_t *> outputs {&out_lt0, &out_lt1};
        ASSERT_EQ(dnnl_impl::infer_bn_folding_output_shape(
                          op.get(), inputs, outputs),
                status);
    }
}

TEST(DnnlShapeInfer, InferDnnlConvBwdDataOutputShape) {
    using namespace dnnl::graph::tests::unit::utils;
    using dims = impl::dims;
    auto conv_op = std::make_shared<impl::op_t>(
            impl::op_kind::ConvolutionBackpropData);
    conv_op->set_attr<dims>(impl::op_attr::strides, dims {1, 1});
    conv_op->set_attr<dims>(impl::op_attr::dilations, dims {1, 1});
    conv_op->set_attr<dims>(impl::op_attr::pads_begin, dims {0, 0});
    conv_op->set_attr<dims>(impl::op_attr::pads_end, dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op->set_attr<int64_t>(impl::op_attr::groups, 2);
    conv_op->set_attr<std::string>(impl::op_attr::data_format, "NCX");
    conv_op->set_attr<std::string>(impl::op_attr::filter_format, "OIX");
    conv_op->set_attr<dims>(impl::op_attr::output_shape, dims {1, 6, 7, 7});

    // prepare logical tensor
    impl::logical_tensor_t diff_src
            = logical_tensor_init(0, {1, 4, 5, 5}, impl::data_type::f32);
    impl::logical_tensor_t weights
            = logical_tensor_init(1, {2, 3, 4, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst
            = logical_tensor_init(2, {1, 6, 7, 7}, impl::data_type::f32);

    conv_op->add_input(diff_dst);
    conv_op->add_input(weights);
    conv_op->add_output(diff_src);
    std::vector<impl::logical_tensor_t *> inputs {&diff_src, &weights};
    std::vector<impl::logical_tensor_t *> outputs {&diff_dst};

    ASSERT_EQ(dnnl_impl::infer_dnnl_conv_bwd_data_output_shape(
                      conv_op.get(), inputs, outputs),
            impl::status::success);
}
