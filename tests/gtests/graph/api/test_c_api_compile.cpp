/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_graph.h"

#include "test_api_common.h"
#include "test_api_common.hpp"

TEST(CAPI, CompileBN) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t bn = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_batch_norm_inference;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t gamma;
    dnnl_graph_logical_tensor_t beta;
    dnnl_graph_logical_tensor_t mean;
    dnnl_graph_logical_tensor_t var;
    dnnl_graph_logical_tensor_t output;
    const int64_t input_dim[] = {1, 16, 64, 64};
    const int64_t bn_param_dim[] = {16};
    const int64_t output_dim[] = {1, 16, 64, 64};
    size_t part_num = 0;
    float epsilon = 0.0;

#define COMPILE_BN_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(bn); \
        bn = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_op_create(&bn, 1, op_kind, "bn");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&gamma, 2, dnnl_f32,
                           1, bn_param_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&beta, 3, dnnl_f32,
                           1, bn_param_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&mean, 4, dnnl_f32,
                           1, bn_param_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&var, 5, dnnl_f32,
                           1, bn_param_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 6, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_BN_DESTROY);

    dnnl_graph_op_add_input(bn, &input);
    dnnl_graph_op_add_input(bn, &gamma);
    dnnl_graph_op_add_input(bn, &beta);
    dnnl_graph_op_add_input(bn, &mean);
    dnnl_graph_op_add_input(bn, &var);
    dnnl_graph_op_add_output(bn, &output);

    const dnnl_graph_logical_tensor_t *inputs[5]
            = {&input, &gamma, &beta, &mean, &var};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_f32(
                           bn, dnnl_graph_op_attr_epsilon, &epsilon, 0),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           bn, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_add_op(agraph, bn), dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_BN_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           5, inputs, 1, outputs, e),
            dnnl_success, COMPILE_BN_DESTROY);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            0U); // Convolutional operator W/O sum has no in-place operation.

    COMPILE_BN_DESTROY;
#undef COMPILE_BN_DESTROY
}

TEST(CAPI, CompileConv2D) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_convolution;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t output;

#define COMPILED_CONV2D_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 3, 227, 227};
    const int64_t weight_dim[] = {64, 3, 11, 11};
    const int64_t output_dim[] = {1, 64, 55, 55};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 1;

    dnnl_graph_op_create(&conv2d, 1, op_kind, "conv2d");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32, 4,
                    weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 4, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_DESTROY);

    dnnl_graph_op_add_input(conv2d, &input);
    dnnl_graph_op_add_input(conv2d, &weight);
    dnnl_graph_op_add_output(conv2d, &output);

    const dnnl_graph_logical_tensor_t *inputs[2] = {&input, &weight};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILED_CONV2D_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILED_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILED_CONV2D_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           2, inputs, 1, outputs, e),
            dnnl_success, COMPILED_CONV2D_DESTROY);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            0U); // Convolutional operator W/O sum has no in-place operation.

    COMPILED_CONV2D_DESTROY;
#undef COMPILED_CONV2D_DESTROY
}

TEST(CAPI, CompileGroupedConv2D) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_convolution;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t output;

#define COMPILE_GROUND_CONV2D_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 8, 227, 227};
    const int64_t weight_dim[] = {64, 2, 11, 11};
    const int64_t output_dim[] = {1, 64, 55, 55};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 4;

    dnnl_graph_op_create(&conv2d, 1, op_kind, "conv2d");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32, 4,
                    weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 4, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);

    dnnl_graph_op_add_input(conv2d, &input);
    dnnl_graph_op_add_input(conv2d, &weight);
    dnnl_graph_op_add_output(conv2d, &output);

    const dnnl_graph_logical_tensor_t *inputs[2] = {&input, &weight};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           2, inputs, 1, outputs, e),
            dnnl_success, COMPILE_GROUND_CONV2D_DESTROY);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            0U); // Convolutional operator W/O sum has no in-place operation.

    COMPILE_GROUND_CONV2D_DESTROY;
#undef COMPILE_GROUND_CONV2D_DESTROY
}

TEST(CAPI, CompileConv2DBiasSum) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_graph_op_t bias_add = nullptr;
    dnnl_graph_op_t sum = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t bias;
    dnnl_graph_logical_tensor_t output;
    dnnl_graph_logical_tensor_t bias_add_output;
    dnnl_graph_logical_tensor_t sum_src2;
    dnnl_graph_logical_tensor_t sum_dst;

#define COMPILE_CONV2D_BIAS_SUM_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_op_destroy(bias_add); \
        bias_add = nullptr; \
        dnnl_graph_op_destroy(sum); \
        sum = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 3, 227, 227};
    const int64_t weight_dim[] = {64, 3, 11, 11};
    const int64_t bias_dim[] = {64};
    const int64_t output_dim[] = {1, 64, 55, 55};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 1;

    dnnl_graph_op_create(&conv2d, 1, dnnl_graph_op_convolution, "conv2d");
    dnnl_graph_op_create(&bias_add, 2, dnnl_graph_op_bias_add, "bias_add");
    dnnl_graph_op_create(&sum, 3, dnnl_graph_op_add, "sum");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32, 4,
                    weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 3, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&bias, 4, dnnl_f32,
                           1, bias_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bias_add_output, 5,
                    dnnl_f32, 4, output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&sum_src2, 6, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&sum_dst, 7, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);

    dnnl_graph_op_add_input(conv2d, &input);
    dnnl_graph_op_add_input(conv2d, &weight);
    dnnl_graph_op_add_output(conv2d, &output);
    dnnl_graph_op_add_input(bias_add, &output);
    dnnl_graph_op_add_input(bias_add, &bias);
    dnnl_graph_op_add_output(bias_add, &bias_add_output);
    dnnl_graph_op_add_input(sum, &bias_add_output);
    dnnl_graph_op_add_input(sum, &sum_src2);
    dnnl_graph_op_add_output(sum, &sum_dst);

    const dnnl_graph_logical_tensor_t *inputs[4]
            = {&input, &weight, &bias, &sum_src2};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&sum_dst};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, bias_add), dnnl_success,
            COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, sum), dnnl_success,
            COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           4, inputs, 1, outputs, e),
            dnnl_success, COMPILE_CONV2D_BIAS_SUM_DESTROY);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);

    // TODO(qun): re-enable this check once library and bridge align the inplace
    // logic
    // EXPECT_EQ(num_inplace_pairs, 1); // two strided lts also can inplace

    COMPILE_CONV2D_BIAS_SUM_DESTROY;
#undef COMPILE_CONV2D_BIAS_SUM_DESTROY
}

TEST(CAPI, CompileConv2DSumConv2D) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv0 = nullptr;
    dnnl_graph_op_t conv1 = nullptr;
    dnnl_graph_op_t sum = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t *partition = nullptr;
    dnnl_graph_compiled_partition_t *compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t conv0_input;
    dnnl_graph_logical_tensor_t conv0_weight;
    dnnl_graph_logical_tensor_t conv0_output;
    dnnl_graph_logical_tensor_t conv1_input;
    dnnl_graph_logical_tensor_t conv1_weight;
    dnnl_graph_logical_tensor_t conv1_output;
    dnnl_graph_logical_tensor_t sum_output;

#define COMPILE_CONV2D_SUM_CONV2D_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv0); \
        conv0 = nullptr; \
        dnnl_graph_op_destroy(conv1); \
        conv1 = nullptr; \
        dnnl_graph_op_destroy(sum); \
        sum = nullptr; \
    } while (0);

    const int64_t input_dim[] = {128, 3, 227, 227};
    const int64_t weight_dim[] = {16, 3, 11, 11};
    const int64_t output_dim[] = {128, 16, 55, 55};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 1;

    dnnl_graph_op_create(&conv0, 1, dnnl_graph_op_convolution, "conv0");
    dnnl_graph_op_create(&conv1, 2, dnnl_graph_op_convolution, "conv1");
    dnnl_graph_op_create(&sum, 3, dnnl_graph_op_add, "sum");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_input, 1,
                           dnnl_f32, 4, input_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_weight, 2,
                           dnnl_f32, 4, weight_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_output, 3,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv1_input, 4,
                           dnnl_f32, 4, input_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv1_weight, 5,
                           dnnl_f32, 4, weight_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv1_output, 6,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&sum_output, 7,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);

    dnnl_graph_op_add_input(conv0, &conv0_input);
    dnnl_graph_op_add_input(conv0, &conv0_weight);
    dnnl_graph_op_add_output(conv0, &conv0_output);
    dnnl_graph_op_add_input(conv1, &conv1_input);
    dnnl_graph_op_add_input(conv1, &conv1_weight);
    dnnl_graph_op_add_output(conv1, &conv1_output);
    dnnl_graph_op_add_input(sum, &conv0_output);
    dnnl_graph_op_add_input(sum, &conv1_output);
    dnnl_graph_op_add_output(sum, &sum_output);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv0, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv0, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv1, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv1, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv1, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv1, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv1, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv1, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv1, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv0), dnnl_success,
            COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv1), dnnl_success,
            COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, sum), dnnl_success,
            COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY);
    ASSERT_EQ_SAFE(part_num, 2U, COMPILE_CONV2D_SUM_CONV2D_DESTROY);

    partition = (dnnl_graph_partition_t *)malloc(
            part_num * sizeof(dnnl_graph_partition_t));
    if (partition == nullptr) {
        COMPILE_CONV2D_SUM_CONV2D_DESTROY;
        return;
    }
    compiled_partition = (dnnl_graph_compiled_partition_t *)malloc(
            part_num * sizeof(dnnl_graph_compiled_partition_t));
    if (compiled_partition == nullptr) {
        COMPILE_CONV2D_SUM_CONV2D_DESTROY;
        free(partition);
        partition = nullptr;
        return;
    }

#define COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS \
    do { \
        COMPILE_CONV2D_SUM_CONV2D_DESTROY; \
        for (size_t i = 0; i < part_num; ++i) { \
            dnnl_graph_partition_destroy(*(partition + i)); \
            dnnl_graph_compiled_partition_destroy(*(compiled_partition + i)); \
        } \
        if (partition) { \
            free(partition); \
            partition = nullptr; \
        } \
        if (compiled_partition) { \
            free(compiled_partition); \
            compiled_partition = nullptr; \
        } \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partitions(agraph, part_num, partition),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS);
    for (size_t i = 0; i < part_num; ++i) {
        ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                               compiled_partition + i, *(partition + i)),
                dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS);
    }

    const dnnl_graph_logical_tensor_t *p0_inputs[2]
            = {&conv1_input, &conv1_weight};
    const dnnl_graph_logical_tensor_t *p0_outputs[1] = {&conv1_output};

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(*partition, *compiled_partition,
                           2, p0_inputs, 1, p0_outputs, e),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS);

    dnnl_graph_logical_tensor_t opt_conv1_output;
    ASSERT_EQ_SAFE(
            dnnl_graph_compiled_partition_query_logical_tensor(
                    *compiled_partition, conv1_output.id, &opt_conv1_output),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS);

    const dnnl_graph_logical_tensor_t *p1_inputs[3]
            = {&conv0_input, &conv0_weight, &opt_conv1_output};
    const dnnl_graph_logical_tensor_t *p1_outputs[1] = {&sum_output};
    ASSERT_EQ_SAFE(
            dnnl_graph_partition_compile(*(partition + 1),
                    *(compiled_partition + 1), 3, p1_inputs, 1, p1_outputs, e),
            dnnl_success, COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      *compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            0U); // Convolutional operator W/O sum has no in-place operation.
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      *(compiled_partition + 1), &num_inplace_pairs,
                      &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            1U); // Convolutional operator W/ sum supports in-place operation.

    COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS;
#undef COMPILE_CONV2D_SUM_CONV2D_DESTROY
#undef COMPILE_CONV2D_SUM_CONV2D_DESTROY_PLUS
}

TEST(CAPI, CompileSumConv2DStridedBN) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv0 = nullptr;
    dnnl_graph_op_t bn = nullptr;
    dnnl_graph_op_t sum = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t *partition = nullptr;
    dnnl_graph_compiled_partition_t *compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t conv0_input;
    dnnl_graph_logical_tensor_t conv0_weight;
    dnnl_graph_logical_tensor_t conv0_output;
    dnnl_graph_logical_tensor_t bn_input;
    dnnl_graph_logical_tensor_t bn_gamma;
    dnnl_graph_logical_tensor_t bn_beta;
    dnnl_graph_logical_tensor_t bn_mean;
    dnnl_graph_logical_tensor_t bn_var;
    dnnl_graph_logical_tensor_t sum_input1;
    dnnl_graph_logical_tensor_t sum_output;

#define COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv0); \
        conv0 = nullptr; \
        dnnl_graph_op_destroy(bn); \
        bn = nullptr; \
        dnnl_graph_op_destroy(sum); \
        sum = nullptr; \
    } while (0);

    const int64_t N = 128;
    const int64_t IC = 3;
    const int64_t IH = 227;
    const int64_t IW = 227;

    const int64_t KH = 11;
    const int64_t KW = 11;

    // set OC to be much larger than IC to make sure that the conv's output will
    // have different format with input on all platforms. So the inplace pair
    // will always be empty.
    const int64_t OC = 128;
    const int64_t OH = 55;
    const int64_t OW = 55;

    const int64_t input_dim[] = {N, IC, IH, IW};
    const int64_t weight_dim[] = {OC, IC, KH, KW};
    const int64_t output_dim[] = {N, OC, OH, OW};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 1;

    float epsilon = 0.001f;

    dnnl_graph_op_create(&conv0, 1, dnnl_graph_op_convolution, "conv0");
    dnnl_graph_op_create(
            &bn, 2, dnnl_graph_op_batch_norm_inference, "batchnorm");
    dnnl_graph_op_create(&sum, 3, dnnl_graph_op_add, "sum");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_input, 1,
                           dnnl_f32, 4, input_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_weight, 2,
                           dnnl_f32, 4, weight_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&conv0_output, 3,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&sum_input1, 4,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&sum_output, 5,
                           dnnl_f32, 4, output_dim, dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_input, 6, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&bn_gamma, 7,
                           dnnl_f32, 1, &OC, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&bn_beta, 8,
                           dnnl_f32, 1, &OC, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&bn_mean, 9,
                           dnnl_f32, 1, &OC, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&bn_var, 10,
                           dnnl_f32, 1, &OC, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    dnnl_graph_op_add_input(conv0, &conv0_input);
    dnnl_graph_op_add_input(conv0, &conv0_weight);
    dnnl_graph_op_add_output(conv0, &conv0_output);
    dnnl_graph_op_add_input(bn, &bn_input);
    dnnl_graph_op_add_input(bn, &bn_gamma);
    dnnl_graph_op_add_input(bn, &bn_beta);
    dnnl_graph_op_add_input(bn, &bn_mean);
    dnnl_graph_op_add_input(bn, &bn_var);
    dnnl_graph_op_add_output(bn, &sum_input1);
    dnnl_graph_op_add_input(sum, &conv0_output);
    dnnl_graph_op_add_input(sum, &sum_input1);
    dnnl_graph_op_add_output(sum, &sum_output);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv0, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv0, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv0, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_f32(
                           bn, dnnl_graph_op_attr_epsilon, &epsilon, 0),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           bn, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv0), dnnl_success,
            COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, bn), dnnl_success,
            COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, sum), dnnl_success,
            COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);
    ASSERT_EQ_SAFE(part_num, 2U, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY);

    partition = (dnnl_graph_partition_t *)malloc(
            part_num * sizeof(dnnl_graph_partition_t));
    compiled_partition = (dnnl_graph_compiled_partition_t *)malloc(
            part_num * sizeof(dnnl_graph_compiled_partition_t));

#define COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS \
    do { \
        COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY; \
        for (size_t i = 0; i < part_num; ++i) { \
            dnnl_graph_partition_destroy(*(partition + i)); \
            dnnl_graph_compiled_partition_destroy(*(compiled_partition + i)); \
        } \
        if (partition) { \
            free(partition); \
            partition = nullptr; \
        } \
        if (compiled_partition) { \
            free(compiled_partition); \
            compiled_partition = nullptr; \
        } \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partitions(agraph, part_num, partition),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);
    for (size_t i = 0; i < part_num; ++i) {
        ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                               compiled_partition + i, *(partition + i)),
                dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);
    }

    const dnnl_graph_logical_tensor_t *p0_inputs[5]
            = {&bn_input, &bn_gamma, &bn_beta, &bn_mean, &bn_var};
    const dnnl_graph_logical_tensor_t *p0_outputs[1] = {&sum_input1};

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(*partition, *compiled_partition,
                           5, p0_inputs, 1, p0_outputs, e),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);
    dnnl_graph_logical_tensor_t opt_sum_input1;
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_query_logical_tensor(
                           *compiled_partition, sum_input1.id, &opt_sum_input1),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);
    ASSERT_EQ(opt_sum_input1.layout_type, dnnl_graph_layout_type_strided);

    const dnnl_graph_logical_tensor_t *p1_inputs[3]
            = {&conv0_input, &conv0_weight, &opt_sum_input1};
    const dnnl_graph_logical_tensor_t *p1_outputs[1] = {&sum_output};
    ASSERT_EQ_SAFE(
            dnnl_graph_partition_compile(*(partition + 1),
                    *(compiled_partition + 1), 3, p1_inputs, 1, p1_outputs, e),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);

    dnnl_graph_logical_tensor_t opaque_sum_output;
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_query_logical_tensor(
                           *(compiled_partition + 1), sum_output.id,
                           &opaque_sum_output),
            dnnl_success, COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS);
    ASSERT_EQ(opaque_sum_output.layout_type,
            engine == dnnl_gpu ? dnnl_graph_layout_type_opaque
                               : dnnl_graph_layout_type_strided);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      *compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs, 0U);

    // `opt_sum_input1` is of plain format, while `opaque_sum_output` is of
    // blocked format. Though they are both opaque tensor, in-place operation
    // is not supported as the different formats they are.
    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");
    num_inplace_pairs = 10;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      *(compiled_partition + 1), &num_inplace_pairs,
                      &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs, 0U);

    COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS;
#undef COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY
#undef COMPILE_SUM_CONV2D_STRIDED_BN_DESTROY_PLUS
}

TEST(CAPI, CompileConv2DWithUnknownShape) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_convolution;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t output;

#define COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 3, 227, 227};
    const int64_t weight_dim[] = {64, 3, 11, 11};
    const int64_t output_dim[]
            = {DNNL_GRAPH_UNKNOWN_DIM, DNNL_GRAPH_UNKNOWN_DIM,
                    DNNL_GRAPH_UNKNOWN_DIM, DNNL_GRAPH_UNKNOWN_DIM};
    size_t part_num = 0;
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    int64_t group = 1;

    dnnl_graph_op_create(&conv2d, 1, op_kind, "conv2d");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32, 4,
                    weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 4, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);

    dnnl_graph_op_add_input(conv2d, &input);
    dnnl_graph_op_add_input(conv2d, &weight);
    dnnl_graph_op_add_output(conv2d, &output);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);
    // support arbitrary order of inputs in infer_shape and compile APIs
    const dnnl_graph_logical_tensor_t *const_inputs[2] = {&weight, &input};
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_outputs[1] = {&output};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           2, const_inputs, 1, const_outputs, e),
            dnnl_success, COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      compiled_partition, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    EXPECT_EQ(num_inplace_pairs,
            0U); // Convolutional operator W/O sum has no in-place operation.

    COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY;
#undef COMPILED_CONV2D_WITH_UNKNOWN_SHAPE_DESTROY
}

TEST(CAPI, CompileMaxPool) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t maxpool = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_max_pool;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t output;

#define COMPILED_POOL_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(maxpool); \
        maxpool = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 2, 2, 1};
    size_t part_num = 0;
    int64_t stride[] = {2, 2};
    int64_t kernel[] = {2, 2};
    int64_t dilations[] = {1, 1};
    int64_t padding[] = {0, 0};
    // wrong output shape
    const int64_t output_dim[] = {1, 2, 2, 1};

    dnnl_graph_op_create(&maxpool, 1, op_kind, "maxpool");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&output, 2, dnnl_f32, 4,
                    output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_POOL_DESTROY);

    dnnl_graph_op_add_input(maxpool, &input);
    dnnl_graph_op_add_output(maxpool, &output);

    const dnnl_graph_logical_tensor_t *inputs[1] = {&input};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_kernel, kernel, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           maxpool, dnnl_graph_op_attr_data_format, "NXC", 1),
            dnnl_success, COMPILED_POOL_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, maxpool), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILED_POOL_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           1, inputs, 1, outputs, e),
            dnnl_invalid_shape, COMPILED_POOL_DESTROY);

    COMPILED_POOL_DESTROY;
#undef COMPILED_POOL_DESTROY
}

TEST(CAPI, CompileMaxPoolWithStridedOutput) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t maxpool = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_max_pool;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t output;

#define COMPILED_POOL_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(maxpool); \
        maxpool = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {1, 1, 5, 5};
    size_t part_num = 0;
    int64_t stride[] = {1, 1};
    int64_t kernel[] = {3, 3};
    int64_t dilations[] = {1, 1};
    int64_t padding[] = {0, 0};
    const int64_t output_dim[] = {1, 1, 3, 3};
    const int64_t output_strides[] = {18, 18, 6, 2}; // strided

    dnnl_graph_op_create(&maxpool, 1, op_kind, "maxpool");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           4, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_POOL_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_strides(&output, 2,
                           dnnl_f32, 4, output_dim, output_strides,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILED_POOL_DESTROY);

    dnnl_graph_op_add_input(maxpool, &input);
    dnnl_graph_op_add_output(maxpool, &output);

    const dnnl_graph_logical_tensor_t *inputs[1] = {&input};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           maxpool, dnnl_graph_op_attr_kernel, kernel, 2),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           maxpool, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILED_POOL_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, maxpool), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILED_POOL_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILED_POOL_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           1, inputs, 1, outputs, e),
            dnnl_success, COMPILED_POOL_DESTROY);

    size_t expected = 0;
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_get_mem_size(&output, &expected),
            dnnl_success, COMPILED_POOL_DESTROY);

    dnnl_graph_logical_tensor_t queried_output;
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_query_logical_tensor(
                           compiled_partition, 2, &queried_output),
            dnnl_success, COMPILED_POOL_DESTROY);

    size_t queried = 0;
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_get_mem_size(&queried_output, &queried),
            dnnl_success, COMPILED_POOL_DESTROY);

    ASSERT_EQ_SAFE(expected, queried, COMPILED_POOL_DESTROY);

    COMPILED_POOL_DESTROY;
#undef COMPILED_POOL_DESTROY
}

TEST(CAPI, CompileAdd) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t add = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_add;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;

#define COMPILE_ADD_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(add); \
        add = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t src0;
    dnnl_graph_logical_tensor_t src1;
    dnnl_graph_logical_tensor_t dst;

    const int64_t src0_dim[] = {1, 16, 64, 64};
    const int64_t src1_dim[] = {1, 16, 64, 64};
    const int64_t dst_dim[] = {1, 16, 64, 64};

    size_t part_num = 0;

    dnnl_graph_op_create(&add, 1, op_kind, "add");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src0, 0, dnnl_f32,
                           4, src0_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src1, 1, dnnl_f32,
                           4, src1_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&dst, 2, dnnl_f32,
                           4, dst_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_ADD_DESTROY);

    dnnl_graph_op_add_input(add, &src0);
    dnnl_graph_op_add_input(add, &src1);
    dnnl_graph_op_add_output(add, &dst);

    ASSERT_EQ_SAFE(
            dnnl_graph_add_op(agraph, add), dnnl_success, COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_ADD_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_ADD_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_ADD_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_inputs[2] = {&src0, &src1};
    const dnnl_graph_logical_tensor_t *const_dst[1] = {&dst};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           2, const_inputs, 1, const_dst, e),
            dnnl_success, COMPILE_ADD_DESTROY);

    COMPILE_ADD_DESTROY;
#undef COMPILE_ADD_DESTROY
}

TEST(CAPI, CompileConvBN) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_graph_op_t bn = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t conv_input;
    dnnl_graph_logical_tensor_t conv_weight;
    dnnl_graph_logical_tensor_t conv_output;
    dnnl_graph_logical_tensor_t bn_scale;
    dnnl_graph_logical_tensor_t bn_shift;
    dnnl_graph_logical_tensor_t bn_mean;
    dnnl_graph_logical_tensor_t bn_var;
    dnnl_graph_logical_tensor_t bn_output;

#define COMPILE_CONV_BN_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_op_destroy(bn); \
        bn = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t conv_input_dim[] = {1, 3, 227, 227};
    const int64_t conv_weight_dim[] = {64, 3, 11, 11};
    const int64_t conv_output_dim[] = {1, 64, 55, 55};
    const int64_t bn_scale_dim[] = {64};
    const int64_t bn_shift_dim[] = {64};
    const int64_t bn_mean_dim[] = {64};
    const int64_t bn_var_dim[] = {64};
    const int64_t bn_output_dim[] = {1, 64, 55, 55};

    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    float epsilon = 0.0;
    int64_t group = 1;
    size_t part_num = 0;

    dnnl_graph_op_create(&conv2d, 1, dnnl_graph_op_convolution, "conv2d");
    dnnl_graph_op_create(&bn, 2, dnnl_graph_op_batch_norm_inference, "bn");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_input, 1, dnnl_f32,
                    4, conv_input_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_weight, 2, dnnl_f32,
                    4, conv_weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_output, 4, dnnl_f32,
                    4, conv_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_scale, 5, dnnl_f32, 1,
                    bn_scale_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_shift, 6, dnnl_f32, 1,
                    bn_shift_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_mean, 7, dnnl_f32, 1,
                    bn_mean_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_var, 8, dnnl_f32, 1,
                    bn_var_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_output, 9, dnnl_f32, 4,
                    bn_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_DESTROY);

    dnnl_graph_op_add_input(conv2d, &conv_input);
    dnnl_graph_op_add_input(conv2d, &conv_weight);
    dnnl_graph_op_add_output(conv2d, &conv_output);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_CONV_BN_DESTROY);

    dnnl_graph_op_add_input(bn, &conv_output);
    dnnl_graph_op_add_input(bn, &bn_scale);
    dnnl_graph_op_add_input(bn, &bn_shift);
    dnnl_graph_op_add_input(bn, &bn_mean);
    dnnl_graph_op_add_input(bn, &bn_var);
    dnnl_graph_op_add_output(bn, &bn_output);

    const dnnl_graph_logical_tensor_t *inputs[6] = {
            &conv_input, &conv_weight, &bn_scale, &bn_shift, &bn_mean, &bn_var};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&bn_output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_f32(
                           bn, dnnl_graph_op_attr_epsilon, &epsilon, 0),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, bn), dnnl_success,
            COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_CONV_BN_DESTROY);

    dnnl_graph_graph_get_partition_num(agraph, &part_num);

    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_CONV_BN_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           6, inputs, 1, outputs, e),
            dnnl_success, COMPILE_CONV_BN_DESTROY);

    COMPILE_CONV_BN_DESTROY;
#undef COMPILE_CONV_BN_DESTROY
}

TEST(CAPI, CompileGroupedConvBN) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_graph_op_t bn = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t conv_input;
    dnnl_graph_logical_tensor_t conv_weight;
    dnnl_graph_logical_tensor_t conv_output;
    dnnl_graph_logical_tensor_t bn_scale;
    dnnl_graph_logical_tensor_t bn_shift;
    dnnl_graph_logical_tensor_t bn_mean;
    dnnl_graph_logical_tensor_t bn_var;
    dnnl_graph_logical_tensor_t bn_output;

#define COMPILE_GROUPED_CONV_BN_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_op_destroy(bn); \
        bn = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t conv_input_dim[] = {1, 8, 227, 227};
    const int64_t conv_weight_dim[] = {64, 2, 11, 11};
    const int64_t conv_output_dim[] = {1, 64, 55, 55};
    const int64_t bn_scale_dim[] = {64};
    const int64_t bn_shift_dim[] = {64};
    const int64_t bn_mean_dim[] = {64};
    const int64_t bn_var_dim[] = {64};
    const int64_t bn_output_dim[] = {1, 64, 55, 55};

    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    float epsilon = 0.0;
    int64_t group = 4;
    size_t part_num = 0;

    dnnl_graph_op_create(&conv2d, 1, dnnl_graph_op_convolution, "conv2d");
    dnnl_graph_op_create(&bn, 2, dnnl_graph_op_batch_norm_inference, "bn");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_input, 1, dnnl_f32,
                    4, conv_input_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_weight, 2, dnnl_f32,
                    4, conv_weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_output, 4, dnnl_f32,
                    4, conv_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_scale, 5, dnnl_f32, 1,
                    bn_scale_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_shift, 6, dnnl_f32, 1,
                    bn_shift_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_mean, 7, dnnl_f32, 1,
                    bn_mean_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_var, 8, dnnl_f32, 1,
                    bn_var_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_output, 9, dnnl_f32, 4,
                    bn_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);

    dnnl_graph_op_add_input(conv2d, &conv_input);
    dnnl_graph_op_add_input(conv2d, &conv_weight);
    dnnl_graph_op_add_output(conv2d, &conv_output);

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);

    dnnl_graph_op_add_input(bn, &conv_output);
    dnnl_graph_op_add_input(bn, &bn_scale);
    dnnl_graph_op_add_input(bn, &bn_shift);
    dnnl_graph_op_add_input(bn, &bn_mean);
    dnnl_graph_op_add_input(bn, &bn_var);
    dnnl_graph_op_add_output(bn, &bn_output);

    const dnnl_graph_logical_tensor_t *inputs[6] = {
            &conv_input, &conv_weight, &bn_scale, &bn_shift, &bn_mean, &bn_var};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&bn_output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_f32(
                           bn, dnnl_graph_op_attr_epsilon, &epsilon, 0),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, bn), dnnl_success,
            COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_GROUPED_CONV_BN_DESTROY);

    dnnl_graph_graph_get_partition_num(agraph, &part_num);

    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           6, inputs, 1, outputs, e),
            dnnl_success, COMPILE_GROUPED_CONV_BN_DESTROY);

    COMPILE_GROUPED_CONV_BN_DESTROY;
#undef COMPILE_GROUPED_CONV_BN_DESTROY
}

TEST(CAPI, CompileConvBNStandalone) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t conv2d = nullptr;
    dnnl_graph_op_t bn = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition[2] = {nullptr};
    dnnl_graph_compiled_partition_t compiled_partition[2] = {nullptr};
    dnnl_graph_logical_tensor_t conv_input;
    dnnl_graph_logical_tensor_t conv_weight;
    dnnl_graph_logical_tensor_t conv_output;

#define COMPILE_CONV_BN_STANDALONE_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(conv2d); \
        conv2d = nullptr; \
        dnnl_graph_op_destroy(bn); \
        bn = nullptr; \
        dnnl_graph_partition_destroy(partition[0]); \
        partition[0] = nullptr; \
        dnnl_graph_partition_destroy(partition[1]); \
        partition[1] = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition[0]); \
        compiled_partition[0] = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition[1]); \
        compiled_partition[1] = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t bn_input;
    dnnl_graph_logical_tensor_t bn_gamma;
    dnnl_graph_logical_tensor_t bn_beta;
    dnnl_graph_logical_tensor_t bn_mean;
    dnnl_graph_logical_tensor_t bn_var;
    dnnl_graph_logical_tensor_t bn_output;

    const int64_t conv_input_dim[] = {1, 3, 227, 227};
    const int64_t conv_weight_dim[] = {64, 3, 11, 11};
    const int64_t conv_output_dim[] = {1, 64, 55, 55};
    const int64_t bn_input_dim[] = {1, 64, 55, 55};
    const int64_t bn_params_dim[] = {64};
    const int64_t bn_output_dim[] = {1, 64, 55, 55};

    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilations[] = {1, 1};
    float epsilon = 0.001f;
    int64_t group = 1;
    size_t part_num = 0;
    size_t ops[10];

    dnnl_graph_op_create(&conv2d, 1, dnnl_graph_op_convolution, "conv2d");
    dnnl_graph_op_create(&bn, 2, dnnl_graph_op_batch_norm_inference, "bn");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_input, 1, dnnl_f32,
                    4, conv_input_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_weight, 2, dnnl_f32,
                    4, conv_weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&conv_output, 4, dnnl_f32,
                    4, conv_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);

    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_input, 5, dnnl_f32, 4,
                    bn_input_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_gamma, 6, dnnl_f32, 1,
                    bn_params_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_beta, 7, dnnl_f32, 1,
                    bn_params_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_mean, 8, dnnl_f32, 1,
                    bn_params_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_var, 9, dnnl_f32, 1,
                    bn_params_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&bn_output, 10, dnnl_f32,
                    4, bn_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);

    dnnl_graph_op_add_input(conv2d, &conv_input);
    dnnl_graph_op_add_input(conv2d, &conv_weight);
    dnnl_graph_op_add_output(conv2d, &conv_output);

    const dnnl_graph_logical_tensor_t *conv_inputs[2]
            = {&conv_input, &conv_weight};
    const dnnl_graph_logical_tensor_t *conv_outputs[1] = {&conv_output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_dilations, dilations, 2),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           conv2d, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           conv2d, dnnl_graph_op_attr_groups, &group, 0),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);

    dnnl_graph_op_add_input(bn, &bn_input);
    dnnl_graph_op_add_input(bn, &bn_gamma);
    dnnl_graph_op_add_input(bn, &bn_beta);
    dnnl_graph_op_add_input(bn, &bn_mean);
    dnnl_graph_op_add_input(bn, &bn_var);
    dnnl_graph_op_add_output(bn, &bn_output);

    const dnnl_graph_logical_tensor_t *bn_inputs[5]
            = {&bn_input, &bn_gamma, &bn_beta, &bn_mean, &bn_var};
    const dnnl_graph_logical_tensor_t *bn_outputs[1] = {&bn_output};

    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_f32(
                           bn, dnnl_graph_op_attr_epsilon, &epsilon, 0),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           bn, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, conv2d), dnnl_success,
            COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, bn), dnnl_success,
            COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_CONV_BN_STANDALONE_DESTROY);

    dnnl_graph_graph_get_partition_num(agraph, &part_num);

    ASSERT_EQ_SAFE(part_num, 2U, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partitions(agraph, part_num, partition),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition[0], partition[0]),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition[1], partition[1]),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_partition_get_ops(partition[0], 1, ops),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_partition_get_ops(partition[1], 1, ops),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(
            dnnl_graph_partition_compile(partition[0], compiled_partition[0], 2,
                    conv_inputs, 1, conv_outputs, e),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_partition_compile(partition[1], compiled_partition[1], 5,
                    bn_inputs, 1, bn_outputs, e),
            dnnl_success, COMPILE_CONV_BN_STANDALONE_DESTROY);

    COMPILE_CONV_BN_STANDALONE_DESTROY;
#undef COMPILE_CONV_BN_STANDALONE_DESTROY
}

TEST(CAPI, CompileMatmulAdd1D) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t matmul = nullptr;
    dnnl_graph_op_t add = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t matmul_output;
    dnnl_graph_logical_tensor_t other;
    dnnl_graph_logical_tensor_t add_output;

#define COMPILE_MATMUL_ADD_1D_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(matmul); \
        matmul = nullptr; \
        dnnl_graph_op_destroy(add); \
        add = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {8, 768};
    const int64_t weight_dim[] = {768, 768};
    const int64_t matmul_output_dim[] = {8, 768};
    const int64_t other_dim[] = {768};
    const int64_t add_output_dim[] = {8, 768};
    size_t part_num = 0;

    dnnl_graph_op_create(&matmul, 1, dnnl_graph_op_matmul, "matmul");
    dnnl_graph_op_create(&add, 2, dnnl_graph_op_add, "add");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32,
                           2, input_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32, 2,
                    weight_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&matmul_output, 3,
                           dnnl_f32, 2, matmul_output_dim,
                           dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&other, 4, dnnl_f32,
                           1, other_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_logical_tensor_init_with_dims(&add_output, 5, dnnl_f32,
                    2, add_output_dim, dnnl_graph_layout_type_strided,
                    dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);

    dnnl_graph_op_add_input(matmul, &input);
    dnnl_graph_op_add_input(matmul, &weight);
    dnnl_graph_op_add_output(matmul, &matmul_output);
    dnnl_graph_op_add_input(add, &matmul_output);
    dnnl_graph_op_add_input(add, &other);
    dnnl_graph_op_add_output(add, &add_output);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, matmul), dnnl_success,
            COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, add), dnnl_success,
            COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);

    const dnnl_graph_logical_tensor_t *inputs[3] = {&input, &weight, &other};
    const dnnl_graph_logical_tensor_t *outputs[1] = {&add_output};

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           3, inputs, 1, outputs, e),
            dnnl_success, COMPILE_MATMUL_ADD_1D_DESTROY);

    COMPILE_MATMUL_ADD_1D_DESTROY;
#undef COMPILE_MATMUL_ADD_1D_DESTROY
}

TEST(CAPI, CompileMatmulAddActivation) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t matmul = nullptr;
    dnnl_graph_op_t add = nullptr;
    dnnl_graph_op_t activation = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;
    dnnl_graph_logical_tensor_t input;
    dnnl_graph_logical_tensor_t weight;
    dnnl_graph_logical_tensor_t matmul_output;
    dnnl_graph_logical_tensor_t other;
    dnnl_graph_logical_tensor_t add_output;
    dnnl_graph_logical_tensor_t activation_output;

#define COMPILE_MATMUL_ADD_ACTIVATION_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(matmul); \
        matmul = nullptr; \
        dnnl_graph_op_destroy(add); \
        add = nullptr; \
        dnnl_graph_op_destroy(activation); \
        activation = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    const int64_t input_dim[] = {8, 768};
    const int64_t weight_dim[] = {768, 768};
    const int64_t matmul_output_dim[] = {8, 768};
    const int64_t other_dim[] = {768};
    const int64_t add_output_dim[] = {8, 768};
    const int64_t activation_output_dim[] = {8, 768};
    size_t part_num = 0;

    std::vector<dnnl_graph_op_kind_t> activation_kinds
            = {dnnl_graph_op_gelu, dnnl_graph_op_relu, dnnl_graph_op_sigmoid};
    for (size_t i = 0; i < activation_kinds.size(); i++) {
        dnnl_graph_op_create(&matmul, 1, dnnl_graph_op_matmul, "matmul");
        dnnl_graph_op_create(&add, 2, dnnl_graph_op_add, "add");
        dnnl_graph_op_create(&activation, 3, activation_kinds[i], "activation");
        api_test_dnnl_graph_graph_create(&agraph, engine);

        ASSERT_EQ_SAFE(
                dnnl_graph_logical_tensor_init_with_dims(&input, 1, dnnl_f32, 2,
                        input_dim, dnnl_graph_layout_type_strided,
                        dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(
                dnnl_graph_logical_tensor_init_with_dims(&weight, 2, dnnl_f32,
                        2, weight_dim, dnnl_graph_layout_type_strided,
                        dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&matmul_output,
                               3, dnnl_f32, 2, matmul_output_dim,
                               dnnl_graph_layout_type_strided,
                               dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(
                dnnl_graph_logical_tensor_init_with_dims(&other, 4, dnnl_f32, 1,
                        other_dim, dnnl_graph_layout_type_strided,
                        dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&add_output, 5,
                               dnnl_f32, 2, add_output_dim,
                               dnnl_graph_layout_type_strided,
                               dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(
                dnnl_graph_logical_tensor_init_with_dims(&activation_output, 6,
                        dnnl_f32, 2, activation_output_dim,
                        dnnl_graph_layout_type_strided,
                        dnnl_graph_tensor_property_undef),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);

        dnnl_graph_op_add_input(matmul, &input);
        dnnl_graph_op_add_input(matmul, &weight);
        dnnl_graph_op_add_output(matmul, &matmul_output);
        dnnl_graph_op_add_input(add, &matmul_output);
        dnnl_graph_op_add_input(add, &other);
        dnnl_graph_op_add_output(add, &add_output);
        dnnl_graph_op_add_input(activation, &add_output);
        dnnl_graph_op_add_output(activation, &activation_output);

        ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, matmul), dnnl_success,
                COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, add), dnnl_success,
                COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, activation), dnnl_success,
                COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
                COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
                COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(part_num, 1U, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(
                dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);
        ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                               &compiled_partition, partition),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);

        const dnnl_graph_logical_tensor_t *inputs[3]
                = {&input, &weight, &other};
        const dnnl_graph_logical_tensor_t *outputs[1] = {&activation_output};

        dnnl_engine_t e;
        api_test_dnnl_engine_create(&e, engine);
        ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition,
                               compiled_partition, 3, inputs, 1, outputs, e),
                dnnl_success, COMPILE_MATMUL_ADD_ACTIVATION_DESTROY);

        COMPILE_MATMUL_ADD_ACTIVATION_DESTROY;
    }
#undef COMPILE_MATMUL_ADD_ACTIVATION_DESTROY
}

TEST(CAPI, CompileSoftmax) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t softmax = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_softmax;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;

#define COMPILE_SOFTMAX_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(softmax); \
        softmax = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t src;
    dnnl_graph_logical_tensor_t dst;

    const int64_t src_dim[] = {1, 4};
    const int64_t dst_dim[] = {1, 4};

    size_t part_num = 0;

    dnnl_graph_op_create(&softmax, 1, op_kind, "softmax");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src, 0, dnnl_f32,
                           2, src_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&dst, 1, dnnl_f32,
                           2, dst_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);

    dnnl_graph_op_add_input(softmax, &src);
    dnnl_graph_op_add_output(softmax, &dst);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, softmax), dnnl_success,
            COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_SOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_inputs[1] = {&src};
    const dnnl_graph_logical_tensor_t *const_dst[1] = {&dst};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           1, const_inputs, 1, const_dst, e),
            dnnl_success, COMPILE_SOFTMAX_DESTROY);

    COMPILE_SOFTMAX_DESTROY;
#undef COMPILE_SOFTMAX_DESTROY
}

TEST(CAPI, CompileSoftmaxBackward) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t softmax = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_softmax_backward;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;

#define COMPILE_SOFTMAX_BWD_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(softmax); \
        softmax = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t src0;
    dnnl_graph_logical_tensor_t src1;
    dnnl_graph_logical_tensor_t dst;

    const int64_t src0_dim[] = {1, 4};
    const int64_t src1_dim[] = {1, 4};
    const int64_t dst_dim[] = {1, 4};

    size_t part_num = 0;

    dnnl_graph_op_create(&softmax, 1, op_kind, "softmax_bwd");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src0, 0, dnnl_f32,
                           2, src0_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src1, 1, dnnl_f32,
                           2, src1_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&dst, 2, dnnl_f32,
                           2, dst_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);

    dnnl_graph_op_add_input(softmax, &src0);
    dnnl_graph_op_add_input(softmax, &src1);
    dnnl_graph_op_add_output(softmax, &dst);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, softmax), dnnl_success,
            COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_SOFTMAX_BWD_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_inputs[2] = {&src0, &src1};
    const dnnl_graph_logical_tensor_t *const_dst[1] = {&dst};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           2, const_inputs, 1, const_dst, e),
            dnnl_success, COMPILE_SOFTMAX_BWD_DESTROY);

    COMPILE_SOFTMAX_BWD_DESTROY;
#undef COMPILE_SOFTMAX_BWD_DESTROY
}

TEST(CAPI, CompileLogSoftmax) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t logsoftmax = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_log_softmax;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;

#define COMPILE_LOGSOFTMAX_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(logsoftmax); \
        logsoftmax = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t src;
    dnnl_graph_logical_tensor_t dst;

    const int64_t src_dim[] = {1, 4};
    const int64_t dst_dim[] = {1, 4};

    size_t part_num = 0;

    dnnl_graph_op_create(&logsoftmax, 1, op_kind, "logsoftmax");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src, 0, dnnl_f32,
                           2, src_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&dst, 1, dnnl_f32,
                           2, dst_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);

    dnnl_graph_op_add_input(logsoftmax, &src);
    dnnl_graph_op_add_output(logsoftmax, &dst);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, logsoftmax), dnnl_success,
            COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_LOGSOFTMAX_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_inputs[1] = {&src};
    const dnnl_graph_logical_tensor_t *const_dst[1] = {&dst};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           1, const_inputs, 1, const_dst, e),
            dnnl_success, COMPILE_LOGSOFTMAX_DESTROY);

    COMPILE_LOGSOFTMAX_DESTROY;
#undef COMPILE_LOGSOFTMAX_DESTROY
}

TEST(CAPI, CompileWildcard) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t wildcard = nullptr;
    dnnl_engine_kind_t engine = api_test_engine_kind;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_wildcard;
    dnnl_graph_partition_policy_t policy = dnnl_graph_partition_policy_fusion;
    dnnl_graph_partition_t partition = nullptr;
    dnnl_graph_compiled_partition_t compiled_partition = nullptr;

#define COMPILE_WILDCARD_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = nullptr; \
        dnnl_graph_op_destroy(wildcard); \
        wildcard = nullptr; \
        dnnl_graph_partition_destroy(partition); \
        partition = nullptr; \
        dnnl_graph_compiled_partition_destroy(compiled_partition); \
        compiled_partition = nullptr; \
    } while (0);

    dnnl_graph_logical_tensor_t src;
    dnnl_graph_logical_tensor_t dst;

    const int64_t src_dim[] = {1, 4};
    const int64_t dst_dim[] = {1, 4};

    size_t part_num = 0;

    dnnl_graph_op_create(&wildcard, 1, op_kind, "wildcard");
    api_test_dnnl_graph_graph_create(&agraph, engine);

    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&src, 0, dnnl_f32,
                           2, src_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init_with_dims(&dst, 1, dnnl_f32,
                           2, dst_dim, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, COMPILE_WILDCARD_DESTROY);

    dnnl_graph_op_add_input(wildcard, &src);
    dnnl_graph_op_add_output(wildcard, &dst);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, wildcard), dnnl_success,
            COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_filter(agraph, policy), dnnl_success,
            COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_partition_num(agraph, &part_num),
            dnnl_success, COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(part_num, 1U, COMPILE_WILDCARD_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_graph_get_partitions(agraph, part_num, &partition),
            dnnl_success, COMPILE_WILDCARD_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_compiled_partition_create(
                           &compiled_partition, partition),
            dnnl_success, COMPILE_WILDCARD_DESTROY);

    dnnl_engine_t e;
    api_test_dnnl_engine_create(&e, engine);
    const dnnl_graph_logical_tensor_t *const_inputs[1] = {&src};
    const dnnl_graph_logical_tensor_t *const_dst[1] = {&dst};
    ASSERT_EQ_SAFE(dnnl_graph_partition_compile(partition, compiled_partition,
                           1, const_inputs, 1, const_dst, e),
            dnnl_invalid_arguments, COMPILE_WILDCARD_DESTROY);

    COMPILE_WILDCARD_DESTROY;
#undef COMPILE_WILDCARD_DESTROY
}

TEST(CAPI, PartitionInvalidArguments) {
    dnnl_status_t res = dnnl_success;
    size_t num = 0;
    size_t id = 0;
    dnnl_engine_kind_t kind = dnnl_cpu;
    uint8_t supported = 0;
    dnnl_graph_partition_t part = nullptr;

    res = dnnl_success;
    res = dnnl_graph_partition_create_with_op(nullptr, nullptr, kind);
    ASSERT_EQ(res, dnnl_invalid_arguments);

    res = dnnl_success;
    res = dnnl_graph_partition_get_op_num(nullptr, &num);
    ASSERT_EQ(res, dnnl_invalid_arguments);

    res = dnnl_success;
    res = dnnl_graph_partition_get_id(nullptr, &id);
    ASSERT_EQ(res, dnnl_invalid_arguments);

    res = dnnl_success;
    res = dnnl_graph_partition_is_supported(nullptr, &supported);
    ASSERT_EQ(res, dnnl_invalid_arguments);

    res = dnnl_success;
    res = dnnl_graph_partition_get_engine_kind(nullptr, &kind);
    ASSERT_EQ(res, dnnl_invalid_arguments);

    res = dnnl_success;
    res = dnnl_graph_partition_create_with_op(&part, nullptr, kind);
    if (res != dnnl_invalid_arguments) {
        dnnl_graph_partition_destroy(part);
        ASSERT_TRUE(false);
    }

    res = dnnl_success;
    res = dnnl_graph_partition_get_op_num(part, nullptr);
    if (res != dnnl_invalid_arguments) {
        dnnl_graph_partition_destroy(part);
        ASSERT_TRUE(false);
    }

    res = dnnl_success;
    res = dnnl_graph_partition_is_supported(part, nullptr);
    if (res != dnnl_invalid_arguments) {
        dnnl_graph_partition_destroy(part);
        ASSERT_TRUE(false);
    }

    res = dnnl_success;
    res = dnnl_graph_partition_compile(
            part, nullptr, 0, nullptr, 0, nullptr, nullptr);
    if (res != dnnl_invalid_arguments) {
        dnnl_graph_partition_destroy(part);
        ASSERT_TRUE(false);
    }

    dnnl_graph_partition_destroy(part);
}
