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

#include "interface/type_constraint.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "gtest/gtest.h"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(TypeConstraint, CheckBnBwdDataType) {
    using dims = graph::dims;

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};
    {
        graph::op_t bn_op(graph::op_kind::BatchNormTrainingBackward);
        bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0);
        bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        // prepare logical tensor
        graph::logical_tensor_t src = utils::logical_tensor_init(0, src_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t scale = utils::logical_tensor_init(1,
                scale_dims, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t mean = utils::logical_tensor_init(2, mean_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t variance
                = utils::logical_tensor_init(3, variance_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_dst = utils::logical_tensor_init(4,
                src_dims, graph::data_type::f32, graph::layout_type::strided);

        bn_op.add_input(src);
        bn_op.add_input(diff_dst);
        bn_op.add_input(mean);
        bn_op.add_input(variance);
        bn_op.add_input(scale);

        ASSERT_EQ(graph::check_bn_bwd_data_type(&bn_op), true);
    }
    {
        graph::op_t bn_op(graph::op_kind::BatchNormTrainingBackward);
        bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0);
        bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        // prepare logical tensor
        graph::logical_tensor_t src = utils::logical_tensor_init(0, src_dims,
                graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t scale = utils::logical_tensor_init(1,
                scale_dims, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t mean = utils::logical_tensor_init(2, mean_dims,
                graph::data_type::bf16, graph::layout_type::strided);
        graph::logical_tensor_t variance
                = utils::logical_tensor_init(3, variance_dims,
                        graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t diff_dst = utils::logical_tensor_init(4,
                src_dims, graph::data_type::f32, graph::layout_type::strided);

        bn_op.add_input(src);
        bn_op.add_input(diff_dst);
        bn_op.add_input(mean);
        bn_op.add_input(variance);
        bn_op.add_input(scale);

        ASSERT_EQ(graph::check_bn_bwd_data_type(&bn_op), false);
    }
}

TEST(TypeConstraint, CheckLnDataType) {
    {
        graph::op_t layernorm_op(graph::op_kind::LayerNorm);
        layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 3, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 3, 2}, graph::data_type::f32);
        graph::logical_tensor_t mean_lt
                = utils::logical_tensor_init(4, {1, 3}, graph::data_type::f32);
        graph::logical_tensor_t variance_lt
                = utils::logical_tensor_init(5, {1, 3}, graph::data_type::f32);

        layernorm_op.add_input(src_lt);
        layernorm_op.add_output(dst_lt);
        layernorm_op.add_output(mean_lt);
        layernorm_op.add_output(variance_lt);
        ASSERT_EQ(graph::check_ln_data_type(&layernorm_op), true);
    }
    {
        graph::op_t layernorm_op(graph::op_kind::LayerNorm);
        layernorm_op.set_attr<float>(graph::op_attr::epsilon, 0);
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 3, 2}, graph::data_type::f32);
        graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 3, 2}, graph::data_type::f32);
        graph::logical_tensor_t mean_lt
                = utils::logical_tensor_init(4, {1, 3}, graph::data_type::bf16);
        graph::logical_tensor_t variance_lt
                = utils::logical_tensor_init(5, {1, 3}, graph::data_type::f32);

        layernorm_op.add_input(src_lt);
        layernorm_op.add_output(dst_lt);
        layernorm_op.add_output(mean_lt);
        layernorm_op.add_output(variance_lt);
        ASSERT_EQ(graph::check_ln_data_type(&layernorm_op), false);
    }
}

TEST(TypeConstraint, CheckTypecastDateType) {
    {
        graph::op_t typecast_op(graph::op_kind::TypeCast);

        graph::logical_tensor_t f32_lt = utils::logical_tensor_init(
                0, {2, 3}, {3, 1}, graph::data_type::f32);
        graph::logical_tensor_t bf16_lt = utils::logical_tensor_init(
                1, {2, 3}, {3, 1}, graph::data_type::bf16);

        typecast_op.add_input(f32_lt);
        typecast_op.add_output(bf16_lt);
        ASSERT_EQ(graph::check_typecast_data_type(&typecast_op), true);
    }
    {
        graph::op_t typecast_op(graph::op_kind::TypeCast);

        graph::logical_tensor_t f16_lt = utils::logical_tensor_init(
                0, {2, 3}, {3, 1}, graph::data_type::f16);
        graph::logical_tensor_t bf16_lt = utils::logical_tensor_init(
                1, {2, 3}, {3, 1}, graph::data_type::bf16);

        typecast_op.add_input(f16_lt);
        typecast_op.add_output(bf16_lt);
        ASSERT_EQ(graph::check_typecast_data_type(&typecast_op), false);
    }
}
