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

#include "oneapi/dnnl/dnnl_graph.h"

TEST(c_api_test, filter_conv_bn_standalone) {
    dnnl_graph_graph_t *agraph;
    dnnl_graph_op_t *conv2d;
    dnnl_graph_op_t *bn;
    dnnl_graph_engine_kind_t engine = dnnl_graph_cpu;
    dnnl_graph_op_kind_t op_kind = kConvolution;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    uint64_t part_num = 0;

    dnnl_graph_op_create(&conv2d, 1, op_kind, "conv2d");
    op_kind = kBatchNormInference;
    dnnl_graph_op_create(&bn, 2, op_kind, "bn");
    dnnl_graph_graph_create(&agraph, engine);

    //conv2d / bn params
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilation[] = {1, 1};
    float epsilon = 0.001f;
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "strides",
                      dnnl_graph_attribute_kind_is, stride, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "pads_begin",
                      dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "pads_end",
                      dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "dilations",
                      dnnl_graph_attribute_kind_is, dilation, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "data_format",
                      dnnl_graph_attribute_kind_s, "NCX", 1),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "filter_format",
                      dnnl_graph_attribute_kind_s, "OIX", 1),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(
                      bn, "epsilon", dnnl_graph_attribute_kind_f, &epsilon, 1),
            dnnl_graph_result_success);

    dnnl_graph_logical_tensor_t conv2d_src_desc, conv2d_weight_desc,
            conv2d_dst_desc, bn_src_desc, bn_gamma_desc, bn_beta_desc,
            bn_mean_desc, bn_var_desc, bn_dst_desc;
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_src_desc, 0,
                      dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_weight_desc, 1,
                      dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_dst_desc, 2,
                      dnnl_graph_f32, 0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_src_desc, 0, dnnl_graph_f32, 4,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_gamma_desc, 3, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_beta_desc, 4, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_mean_desc, 5, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_var_desc, 6, dnnl_graph_f32, 0,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_dst_desc, 7, dnnl_graph_f32, 4,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(conv2d, &conv2d_src_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(conv2d, &conv2d_weight_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(conv2d, &conv2d_dst_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_src_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_gamma_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_beta_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_mean_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_var_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(bn, &bn_dst_desc),
            dnnl_graph_result_success);

    // add op, build graph
    ASSERT_EQ(dnnl_graph_add_op(agraph, conv2d), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_add_op(agraph, bn), dnnl_graph_result_success);
    ASSERT_EQ(
            dnnl_graph_graph_filter(agraph, policy), dnnl_graph_result_success);

    dnnl_graph_graph_get_partition_num(agraph, &part_num);

    ASSERT_EQ(part_num, 2);

    // free memory
    ASSERT_EQ(dnnl_graph_graph_destroy(agraph), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(conv2d), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(bn), dnnl_graph_result_success);
}

TEST(c_api_test, filter_conv_bn_fused) {
    dnnl_graph_graph_t *agraph;
    dnnl_graph_op_t *conv2d;
    dnnl_graph_op_t *bn;
    dnnl_graph_engine_kind_t engine = dnnl_graph_cpu;
    dnnl_graph_op_kind_t op_kind = kConvolution;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    uint64_t part_num = 0;

    dnnl_graph_op_create(&conv2d, 1, op_kind, "conv2d");

    //conv2d params
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilation[] = {1, 1};
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "strides",
                      dnnl_graph_attribute_kind_is, stride, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "pads_begin",
                      dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "pads_end",
                      dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "dilations",
                      dnnl_graph_attribute_kind_is, dilation, 2),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "data_format",
                      dnnl_graph_attribute_kind_s, "NCX", 1),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_attr(conv2d, "filter_format",
                      dnnl_graph_attribute_kind_s, "OIX", 1),
            dnnl_graph_result_success);

    dnnl_graph_logical_tensor_t conv2d_src_desc, conv2d_weight_desc,
            conv2d_dst_desc;
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_src_desc, 0,
                      dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_weight_desc, 1,
                      dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&conv2d_dst_desc, 2,
                      dnnl_graph_f32, 0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(conv2d, &conv2d_src_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(conv2d, &conv2d_weight_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(conv2d, &conv2d_dst_desc),
            dnnl_graph_result_success);

    op_kind = kBatchNormInference;
    dnnl_graph_op_create(&bn, 2, op_kind, "bn");
    float epsilon = 0.001f;
    ASSERT_EQ(dnnl_graph_op_add_attr(
                      bn, "epsilon", dnnl_graph_attribute_kind_f, &epsilon, 1),
            dnnl_graph_result_success);
    dnnl_graph_logical_tensor_t bn_gamma_desc, bn_beta_desc, bn_mean_desc,
            bn_var_desc, bn_dst_desc;
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_gamma_desc, 3, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_beta_desc, 4, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_mean_desc, 5, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_var_desc, 6, dnnl_graph_f32, 0,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&bn_dst_desc, 7, dnnl_graph_f32, 4,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &conv2d_dst_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_gamma_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_beta_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_mean_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(bn, &bn_var_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(bn, &bn_dst_desc),
            dnnl_graph_result_success);

    dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ(dnnl_graph_add_op(agraph, conv2d), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_add_op(agraph, bn), dnnl_graph_result_success);
    ASSERT_EQ(
            dnnl_graph_graph_filter(agraph, policy), dnnl_graph_result_success);

    dnnl_graph_graph_get_partition_num(agraph, &part_num);

    ASSERT_EQ(part_num, 1);

    ASSERT_EQ(dnnl_graph_graph_destroy(agraph), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(conv2d), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(bn), dnnl_graph_result_success);
}

TEST(c_api_test, filter_relu_add) {
    // y = relu(x); z = x + y
    dnnl_graph_op_t *relu;
    dnnl_graph_op_kind_t op_kind = kReLU;
    dnnl_graph_op_create(&relu, 0, op_kind, "relu");

    dnnl_graph_op_t *add;
    op_kind = kAdd;
    dnnl_graph_op_create(&add, 1, op_kind, "add");

    dnnl_graph_logical_tensor_t relu_src_desc, relu_dst_desc, add_dst_desc;
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&relu_src_desc, 0, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&relu_dst_desc, 1, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&add_dst_desc, 2, dnnl_graph_f32,
                      0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(relu, &relu_src_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(relu, &relu_dst_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(add, &relu_src_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(add, &relu_dst_desc),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(add, &add_dst_desc),
            dnnl_graph_result_success);

    dnnl_graph_graph_t *agraph;
    dnnl_graph_engine_kind_t engine = dnnl_graph_cpu;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_graph_create(&agraph, engine);
    ASSERT_EQ(dnnl_graph_add_op(agraph, relu), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_add_op(agraph, add), dnnl_graph_result_success);
    ASSERT_EQ(
            dnnl_graph_graph_filter(agraph, policy), dnnl_graph_result_success);

    ASSERT_EQ(dnnl_graph_graph_destroy(agraph), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(relu), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(add), dnnl_graph_result_success);
}

TEST(c_api_test, different_lt_with_same_id) {
    dnnl_graph_op_t *add;
    dnnl_graph_op_kind_t op_kind = kAdd;
    dnnl_graph_op_create(&add, 1, op_kind, "add");

    dnnl_graph_logical_tensor_t x1, x2, y;
    // x1 and x2 are two different tensors (with different layout_type)
    // but with same id
    ASSERT_EQ(dnnl_graph_logical_tensor_init(&x1, 0, dnnl_graph_f32, 0,
                      dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(
                      &x2, 0, dnnl_graph_f32, 0, dnnl_graph_layout_type_any),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_logical_tensor_init(
                      &y, 1, dnnl_graph_f32, 0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(add, &x1), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_input(add, &x2), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_add_output(add, &y), dnnl_graph_result_success);

    dnnl_graph_graph_t *agraph;
    dnnl_graph_engine_kind_t engine = dnnl_graph_cpu;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_graph_create(&agraph, engine);
    ASSERT_EQ(dnnl_graph_add_op(agraph, add), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_graph_filter(agraph, policy),
            dnnl_graph_result_error_invalid_graph);

    ASSERT_EQ(dnnl_graph_graph_destroy(agraph), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_op_destroy(add), dnnl_graph_result_success);
}
