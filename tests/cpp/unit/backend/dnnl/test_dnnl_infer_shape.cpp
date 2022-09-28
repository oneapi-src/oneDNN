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

#include "interface/shape_infer.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_shape_infer.hpp"
#include "backend/dnnl/internal_attrs.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(DnnlShapeInfer, InferDnnlConvOutputShape) {
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
    dims output_dims_ref {1, 1, 1, 1};

    ASSERT_TRUE(std::equal(
            output_dims.begin(), output_dims.end(), output_dims_ref.begin()));

    auto output_strides = ltw(outputs[0]).vstrides();
    dims output_strides_ref {1, 1, 1, 1}; // add value
    ASSERT_TRUE(std::equal(output_strides.begin(), output_strides.end(),
            output_strides_ref.begin()));
}
