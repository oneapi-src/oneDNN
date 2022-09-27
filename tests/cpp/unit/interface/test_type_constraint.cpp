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

#include "interface/type_constraint.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

#include "gtest/gtest.h"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(TypeConstraint, CheckBnBwdDataType) {
    using dims = impl::dims;

    // Tensor dimensions.
    const impl::dim_t N = 1, // batch size
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
        impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
        bn_op.set_attr<float>(impl::op_attr::epsilon, 0.0);
        bn_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
        // prepare logical tensor
        impl::logical_tensor_t src = utils::logical_tensor_init(
                0, src_dims, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t scale = utils::logical_tensor_init(1, scale_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t mean = utils::logical_tensor_init(
                2, mean_dims, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t variance
                = utils::logical_tensor_init(3, variance_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
                4, src_dims, impl::data_type::f32, impl::layout_type::strided);

        bn_op.add_input(src);
        bn_op.add_input(diff_dst);
        bn_op.add_input(mean);
        bn_op.add_input(variance);
        bn_op.add_input(scale);

        ASSERT_EQ(impl::check_bn_bwd_data_type(&bn_op), true);
    }
    {
        impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
        bn_op.set_attr<float>(impl::op_attr::epsilon, 0.0);
        bn_op.set_attr<std::string>(impl::op_attr::data_format, "NCX");
        // prepare logical tensor
        impl::logical_tensor_t src = utils::logical_tensor_init(
                0, src_dims, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t scale = utils::logical_tensor_init(1, scale_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t mean = utils::logical_tensor_init(2, mean_dims,
                impl::data_type::bf16, impl::layout_type::strided);
        impl::logical_tensor_t variance
                = utils::logical_tensor_init(3, variance_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
                4, src_dims, impl::data_type::f32, impl::layout_type::strided);

        bn_op.add_input(src);
        bn_op.add_input(diff_dst);
        bn_op.add_input(mean);
        bn_op.add_input(variance);
        bn_op.add_input(scale);

        ASSERT_EQ(impl::check_bn_bwd_data_type(&bn_op), false);
    }
}

TEST(TypeConstraint, CheckLnDataType) {
    {
        impl::op_t layernorm_op(impl::op_kind::LayerNorm);
        layernorm_op.set_attr<float>(impl::op_attr::epsilon, 0);
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 3, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 3, 2}, impl::data_type::f32);
        impl::logical_tensor_t mean_lt
                = utils::logical_tensor_init(4, {1, 3}, impl::data_type::f32);
        impl::logical_tensor_t variance_lt
                = utils::logical_tensor_init(5, {1, 3}, impl::data_type::f32);

        layernorm_op.add_input(src_lt);
        layernorm_op.add_output(dst_lt);
        layernorm_op.add_output(mean_lt);
        layernorm_op.add_output(variance_lt);
        ASSERT_EQ(impl::check_ln_data_type(&layernorm_op), true);
    }
    {
        impl::op_t layernorm_op(impl::op_kind::LayerNorm);
        layernorm_op.set_attr<float>(impl::op_attr::epsilon, 0);
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 3, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 3, 2}, impl::data_type::f32);
        impl::logical_tensor_t mean_lt
                = utils::logical_tensor_init(4, {1, 3}, impl::data_type::bf16);
        impl::logical_tensor_t variance_lt
                = utils::logical_tensor_init(5, {1, 3}, impl::data_type::f32);

        layernorm_op.add_input(src_lt);
        layernorm_op.add_output(dst_lt);
        layernorm_op.add_output(mean_lt);
        layernorm_op.add_output(variance_lt);
        ASSERT_EQ(impl::check_ln_data_type(&layernorm_op), false);
    }
}

TEST(TypeConstraint, CheckTypecastDateType) {
    {
        impl::op_t typecast_op(impl::op_kind::TypeCast);

        impl::logical_tensor_t f32_lt = utils::logical_tensor_init(
                0, {2, 3}, {3, 1}, impl::data_type::f32);
        impl::logical_tensor_t bf16_lt = utils::logical_tensor_init(
                1, {2, 3}, {3, 1}, impl::data_type::bf16);

        typecast_op.add_input(f32_lt);
        typecast_op.add_output(bf16_lt);
        ASSERT_EQ(impl::check_typecast_data_type(&typecast_op), true);
    }
    {
        impl::op_t typecast_op(impl::op_kind::TypeCast);

        impl::logical_tensor_t f16_lt = utils::logical_tensor_init(
                0, {2, 3}, {3, 1}, impl::data_type::f16);
        impl::logical_tensor_t bf16_lt = utils::logical_tensor_init(
                1, {2, 3}, {3, 1}, impl::data_type::bf16);

        typecast_op.add_input(f16_lt);
        typecast_op.add_output(bf16_lt);
        ASSERT_EQ(impl::check_typecast_data_type(&typecast_op), false);
    }
}
