/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @example cpu_conv_bias_bn_add_relu.c
/// @copybrief cpu_conv_bias_bn_add_relu_c
/// > Annotated version: @ref cpu_conv_bias_bn_add_relu_c

/// @page cpu_conv_bias_bn_add_relu_c CPU example for conv+bias+bn+add_relu pattern
///
/// > Example code: @ref cpu_conv_bias_bn_add_relu.c

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/allocator.h"
#include "common/utils.h"

#include "oneapi/dnnl/dnnl_graph.h"

///////////////////////////////////////////////
// The original graph is:
// digraph G {
// Convolution_100002 -> BiasAdd_100003;
// BiasAdd_100003 -> BatchNormInference_100004;
// Convolution_100005 -> BiasAdd_100006;
// BiasAdd_100006 -> BatchNormInference_100007;
// BatchNormInference_100004 -> Add_100008;
// BatchNormInference_100007 -> Add_100008;
// Add_100008 -> ReLU_100009;
// }
//
//
// The optimoptimized graph is:
// digraph G {
// Conv_bias_bn_100120 -> Conv_bias_bn_add_relu_100038;
// }
////////////////////////////////////////////////

// Pre-define unique id for dnnl_graph_op,
// used to represent op in computation graph
#define CONV0_ID 0
#define CONV1_ID 1
#define BIAS_ADD0_ID 2
#define BIAS_ADD1_ID 3
#define BATCH_NORM0_ID 4
#define BATCH_NORM1_ID 5
#define ADD0_ID 6
#define RELU0_ID 7

// Predefine unique id for dnnl_graph_logical_tensor,
// used to represent edge in computation graph
#define CONV0_SRC_ID 0
#define CONV0_WEI_ID 1
#define CONV0_DST_ID 2
#define CONV0_BIAS_ID 3

#define CONV1_SRC_ID 4
#define CONV1_WEI_ID 5
#define CONV1_DST_ID 6
#define CONV1_BIAS_ID 7

#define BIAS_ADD0_DST_ID 8
#define BIAS_ADD1_DST_ID 9

#define BATCH_NORM0_SCALE_ID 10
#define BATCH_NORM0_SHIFT_ID 11
#define BATCH_NORM0_MEAN_ID 12
#define BATCH_NORM0_VARIANCE_ID 13
#define BATCH_NORM0_DST_ID 14

#define BATCH_NORM1_SCALE_ID 15
#define BATCH_NORM1_SHIFT_ID 16
#define BATCH_NORM1_MEAN_ID 17
#define BATCH_NORM1_VARIANCE_ID 18
#define BATCH_NORM1_DST_ID 19

#define ADD0_SRC0_ID 14
#define ADD0_SRC1_ID 19
#define ADD0_DST_ID 20

#define RELU0_SRC_ID 20
#define RELU0_DST_ID 21
// pre-defined net parameters
#define BATCH 8

#define CONV0_IC 3
#define CONV0_OC 96
#define CONV0_IH 227
#define CONV0_IW 227
#define CONV0_OH 55
#define CONV0_OW 55
#define CONV0_KS 11
#define CONV0_STRIDE 4
#define CONV0_PADDING 0
#define CONV0_DILATION 1
#define CONV0_GROUPS 1

#define CONV1_IC 3
#define CONV1_OC 96
#define CONV1_IH 227
#define CONV1_IW 227
#define CONV1_OH 55
#define CONV1_OW 55
#define CONV1_KS 11
#define CONV1_STRIDE 4
#define CONV1_PADDING 0
#define CONV1_DILATION 1
#define CONV1_GROUPS 1

int main(int argc, char **argv) {
    /// Get input args
    dnnl_graph_engine_kind_t engine_kind = parse_engine_kind(argc, argv);
    if (engine_kind == dnnl_graph_gpu) {
        printf("Don't support gpu now\n");
        return 0;
    }

    /// Step 1: create allocator
    printf("Step 1: Create allocator----------------");
    dnnl_graph_allocator_t *allocator;
    DNNL_GRAPH_CHECK(
            dnnl_graph_allocator_create(&allocator, allocate, deallocate));
    printf("Success!\n");

    /// Step 2: create an engine and set the allocator to it,
    /// the engine and allocator will used by dnnl graph backend to manage memory resource
    printf("Step 2: Create engine-------------------");
    dnnl_graph_engine_t *engine;
    const int32_t device_id = 0;
    DNNL_GRAPH_CHECK(dnnl_graph_engine_create(&engine, engine_kind, device_id));
    DNNL_GRAPH_CHECK(dnnl_graph_engine_set_allocator(engine, allocator));
    printf("Success!\n");

    /// Step 3: create dnnl_graph_op and add attrs
    printf("Step 3: Create op-----------------------");
    dnnl_graph_op_t *conv0, *conv1, *bias_add0, *bias_add1, *batch_norm0,
            *batch_norm1, *add0, *relu0;
    DNNL_GRAPH_CHECK(
            dnnl_graph_op_create(&conv0, CONV0_ID, kConvolution, "conv0"));
    DNNL_GRAPH_CHECK(
            dnnl_graph_op_create(&conv1, CONV1_ID, kConvolution, "conv1"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &bias_add0, BIAS_ADD0_ID, kBiasAdd, "bias_add0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &bias_add1, BIAS_ADD1_ID, kBiasAdd, "bias_add1"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &batch_norm0, BATCH_NORM0_ID, kBatchNormInference, "batch_norm0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &batch_norm1, BATCH_NORM1_ID, kBatchNormInference, "batch_norm1"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(&add0, ADD0_ID, kAdd, "add0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(&relu0, RELU0_ID, kReLU, "relu0"));

    int64_t conv0_stride[] = {CONV0_STRIDE, CONV0_STRIDE};
    int64_t conv0_padding[] = {CONV0_PADDING, CONV0_PADDING};
    int64_t conv0_dilation[] = {CONV0_DILATION, CONV0_DILATION};
    int64_t conv0_groups[] = {CONV0_GROUPS};
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv0, "strides", dnnl_graph_attribute_kind_is, conv0_stride, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(conv0, "pads_begin",
            dnnl_graph_attribute_kind_is, conv0_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv0, "pads_end", dnnl_graph_attribute_kind_is, conv0_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(conv0, "dilations",
            dnnl_graph_attribute_kind_is, conv0_dilation, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv0, "data_format", dnnl_graph_attribute_kind_s, "NCX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv0, "filter_format", dnnl_graph_attribute_kind_s, "OIX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv0, "groups", dnnl_graph_attribute_kind_i, conv0_groups, 1));

    int64_t conv1_stride[] = {CONV1_STRIDE, CONV1_STRIDE};
    int64_t conv1_padding[] = {CONV1_PADDING, CONV1_PADDING};
    int64_t conv1_dilation[] = {CONV1_DILATION, CONV1_DILATION};
    int64_t conv1_groups[] = {CONV1_GROUPS};

    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv1, "strides", dnnl_graph_attribute_kind_is, conv1_stride, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(conv1, "pads_begin",
            dnnl_graph_attribute_kind_is, conv1_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv1, "pads_end", dnnl_graph_attribute_kind_is, conv1_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(conv1, "dilations",
            dnnl_graph_attribute_kind_is, conv1_dilation, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv1, "data_format", dnnl_graph_attribute_kind_s, "NCX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv1, "filter_format", dnnl_graph_attribute_kind_s, "OIX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            conv1, "groups", dnnl_graph_attribute_kind_i, conv1_groups, 1));

    float epsilon = 0.0;

    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            batch_norm0, "epsilon", dnnl_graph_attribute_kind_f, &epsilon, 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_add_attr(
            batch_norm1, "epsilon", dnnl_graph_attribute_kind_f, &epsilon, 1));

    printf("Success!\n");

    /// Step 4: connect dnnl_graph_op by using logical tensor, and then add dnnl_graph_op
    /// into backend graph
    printf("Step 4: Add OP to graph-----------------");
    dnnl_graph_graph_t *graph;
    DNNL_GRAPH_CHECK(dnnl_graph_graph_create(&graph, engine_kind));
    {
        /// Here, we create dummy logical tensors, which only have valid ID.
        /// We use these dummy logical tensor to represent edge in computation graph.
        /// These dummy logical tensors will be copy into dnnl_graph_op, so we can destroy
        /// them immediately. When we need to use logical tensor later, we can create
        /// them again with more valid information
        /// \note If we can get all required information (such as id, ndims, dims,
        /// data_type, layout_id) before the add op step, we are also able to create
        /// logical tensors once with all required info, and keep them until the
        /// program exit. This depends on our preference
        dnnl_graph_logical_tensor_t conv0_src_desc, conv0_weight_desc,
                conv0_dst_desc, conv0_bias_desc, bias_add0_dst_desc,
                batch_norm0_scale_desc, batch_norm0_shift_desc,
                batch_norm0_mean_desc, batch_norm0_variance_desc,
                batch_norm0_dst_desc, conv1_src_desc, conv1_weight_desc,
                conv1_dst_desc, conv1_bias_desc, bias_add1_dst_desc,
                batch_norm1_scale_desc, batch_norm1_shift_desc,
                batch_norm1_mean_desc, batch_norm1_variance_desc,
                batch_norm1_dst_desc, add0_dst_desc, relu0_dst_desc;
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_src_desc,
                CONV0_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_weight_desc,
                CONV0_WEI_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_dst_desc,
                CONV0_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_bias_desc,
                CONV0_BIAS_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&bias_add0_dst_desc,
                BIAS_ADD0_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm0_scale_desc,
                BATCH_NORM0_SCALE_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm0_shift_desc,
                BATCH_NORM0_SHIFT_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm0_mean_desc,
                BATCH_NORM0_MEAN_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(
                &batch_norm0_variance_desc, BATCH_NORM0_VARIANCE_ID,
                dnnl_graph_f32, 0, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm0_dst_desc,
                BATCH_NORM0_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_src_desc,
                CONV1_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_weight_desc,
                CONV1_WEI_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_dst_desc,
                CONV1_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_bias_desc,
                CONV1_BIAS_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&bias_add1_dst_desc,
                BIAS_ADD1_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm1_scale_desc,
                BATCH_NORM1_SCALE_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm1_shift_desc,
                BATCH_NORM1_SHIFT_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm1_mean_desc,
                BATCH_NORM1_MEAN_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(
                &batch_norm1_variance_desc, BATCH_NORM1_VARIANCE_ID,
                dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&batch_norm1_dst_desc,
                BATCH_NORM1_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&add0_dst_desc,
                ADD0_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&relu0_dst_desc,
                RELU0_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));

        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv0, &conv0_src_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv0, &conv0_weight_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(conv0, &conv0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add0, &conv0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add0, &conv0_bias_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(bias_add0, &bias_add0_dst_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm0, &bias_add0_dst_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm0, &batch_norm0_scale_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm0, &batch_norm0_shift_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm0, &batch_norm0_mean_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(
                batch_norm0, &batch_norm0_variance_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(batch_norm0, &batch_norm0_dst_desc));

        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv1, &conv1_src_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv1, &conv1_weight_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(conv1, &conv1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add1, &conv1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add1, &conv1_bias_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(bias_add1, &bias_add1_dst_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm1, &bias_add1_dst_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm1, &batch_norm1_scale_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm1, &batch_norm1_shift_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_input(batch_norm1, &batch_norm1_mean_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(
                batch_norm1, &batch_norm1_variance_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(batch_norm1, &batch_norm1_dst_desc));

        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(add0, &batch_norm0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(add0, &batch_norm1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(add0, &add0_dst_desc));

        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(relu0, &add0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(relu0, &relu0_dst_desc));

        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, conv0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, bias_add0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, batch_norm0));

        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, conv1));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, bias_add1));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, batch_norm1));

        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, add0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, relu0));
    }
    printf("Success!\n");

    /// Step 5: optimize the graph and get partition from it

    /// This function run pass to optimize the graph. this will fuse
    /// some ops into one op, so the graph will be rewrited
    printf("Step 5: Filter and get partition--------");
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_filter(graph, dnnl_graph_partition_policy_fusion));

    uint64_t partitions_num;
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_get_partition_num(graph, &partitions_num));
    if (partitions_num != 2) {
        printf("Error: partitions number is not equal to %llu\n",
                (unsigned long long)partitions_num);
        exit(1);
    }

    /// Get partition from the optimized graph. Each partition will be composed
    /// of a single op (fused or unfused op)
    dnnl_graph_partition_t *partitions[2];
    DNNL_GRAPH_CHECK(dnnl_graph_partition_create(&partitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_partition_create(&partitions[1]));
    DNNL_GRAPH_CHECK(dnnl_graph_graph_get_partitions(graph, 2, partitions));
    printf("Success!\n");

    /// Step 6: compile partitions
    printf("Step 6: Compile the partitions----------");
    dnnl_graph_compiled_partition_t *cpartitions[2];

    /// Compile partition[0]
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_create(
            &cpartitions[0], partitions[0]));

    int64_t conv1_src_dims[] = {BATCH, CONV1_IC, CONV1_IH, CONV1_IW};
    int64_t conv1_weight_dims[] = {CONV1_OC, CONV1_IC, CONV1_KS, CONV1_KS};
    int64_t conv1_bias_dims[] = {CONV1_OC};
    int64_t batch_norm1_scale_dims[] = {CONV1_OC};
    int64_t batch_norm1_shift_dims[] = {CONV1_OC};
    int64_t batch_norm1_mean_dims[] = {CONV1_OC};
    int64_t batch_norm1_variance_dims[] = {CONV1_OC};
    int64_t batch_norm1_dst_dims[] = {BATCH, CONV1_OC, CONV1_OH, CONV1_OW};
    /// Get cpartition[1]'s output layout id, which has been filled in compile process
    dnnl_graph_logical_tensor_t conv1_src_desc, conv1_weight_desc,
            conv1_bias_desc, batch_norm1_scale_desc, batch_norm1_shift_desc,
            batch_norm1_mean_desc, batch_norm1_variance_desc,
            batch_norm1_dst_desc;
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&conv1_src_desc,
            CONV1_SRC_ID, dnnl_graph_f32, 4, conv1_src_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &conv1_weight_desc, CONV1_WEI_ID, dnnl_graph_f32, 4,
            conv1_weight_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&conv1_bias_desc,
            CONV1_BIAS_ID, dnnl_graph_f32, 1, conv1_bias_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm1_scale_desc, BATCH_NORM1_SCALE_ID, dnnl_graph_f32, 1,
            batch_norm1_scale_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm1_shift_desc, BATCH_NORM1_SHIFT_ID, dnnl_graph_f32, 1,
            batch_norm1_shift_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm1_mean_desc, BATCH_NORM1_MEAN_ID, dnnl_graph_f32, 1,
            batch_norm1_mean_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm1_variance_desc, BATCH_NORM1_VARIANCE_ID, dnnl_graph_f32,
            1, batch_norm1_variance_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    /// \note we set the output layout id to dnnl_graph_any to tell dnnl graph backend that the logical tensor's
    /// layout is allowed to be reset to optimal layout by itself in compilation
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm1_dst_desc, BATCH_NORM1_DST_ID, dnnl_graph_f32, 4,
            batch_norm1_dst_dims, dnnl_graph_layout_type_any,
            dnnl_graph_tensor_property_undef));

    /// The inputs have to contain all logical tensors required by the partition, while
    /// the outputs have to contain all logical tensors generated by the partition
    /// \note Here, we only give the required logical tensor of this partition. But as
    /// long as the above requirements are met, the inputs and outputs could contain more
    /// logical tensor than required. The compile function will search the partition's
    /// required logical tensors in the given ones.
    const dnnl_graph_logical_tensor_t *partition0_inputs[]
            = {&conv1_src_desc, &conv1_weight_desc, &conv1_bias_desc,
                    &batch_norm1_scale_desc, &batch_norm1_shift_desc,
                    &batch_norm1_mean_desc, &batch_norm1_variance_desc};
    const dnnl_graph_logical_tensor_t *partition0_outputs[]
            = {&batch_norm1_dst_desc};
    DNNL_GRAPH_CHECK(dnnl_graph_partition_compile(partitions[0], cpartitions[0],
            7, partition0_inputs, 1, partition0_outputs, engine));

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_DST_ID, &batch_norm1_dst_desc));

    /// Compile partition[1]
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_create(
            &cpartitions[1], partitions[1]));

    int64_t conv0_src_dims[] = {BATCH, CONV0_IC, CONV0_IH, CONV0_IW};
    int64_t conv0_weight_dims[] = {CONV0_OC, CONV0_IC, CONV0_KS, CONV0_KS};
    int64_t conv0_bias_dims[] = {CONV0_OC};
    int64_t batch_norm0_scale_dims[] = {CONV0_OC};
    int64_t batch_norm0_shift_dims[] = {CONV0_OC};
    int64_t batch_norm0_mean_dims[] = {CONV0_OC};
    int64_t batch_norm0_variance_dims[] = {CONV0_OC};
    int64_t relu0_dst_dims[] = {BATCH, CONV0_OC, CONV0_OH, CONV0_OW};

    dnnl_graph_logical_tensor_t conv0_src_desc, conv0_weight_desc,
            conv0_bias_desc, batch_norm0_scale_desc, batch_norm0_shift_desc,
            batch_norm0_mean_desc, batch_norm0_variance_desc, add0_src1_desc,
            relu0_dst_desc;
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&conv0_src_desc,
            CONV0_SRC_ID, dnnl_graph_f32, 4, conv0_src_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &conv0_weight_desc, CONV0_WEI_ID, dnnl_graph_f32, 4,
            conv0_weight_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&conv0_bias_desc,
            CONV0_BIAS_ID, dnnl_graph_f32, 1, conv0_bias_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm0_scale_desc, BATCH_NORM0_SCALE_ID, dnnl_graph_f32, 1,
            batch_norm0_scale_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm0_shift_desc, BATCH_NORM0_SHIFT_ID, dnnl_graph_f32, 1,
            batch_norm0_shift_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm0_mean_desc, BATCH_NORM0_MEAN_ID, dnnl_graph_f32, 1,
            batch_norm0_mean_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &batch_norm0_variance_desc, BATCH_NORM0_VARIANCE_ID, dnnl_graph_f32,
            1, batch_norm0_variance_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&add0_src1_desc,
            ADD0_SRC1_ID, dnnl_graph_f32, 4, batch_norm1_dst_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&relu0_dst_desc,
            RELU0_DST_ID, dnnl_graph_f32, 4, relu0_dst_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));

    /// Reset the input data's layout id, since it's the output of cpartition[1] and its
    /// layout id has been set to optimal by cpartition[1], so cpartition[0] can directly
    /// use this layout
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], ADD0_SRC1_ID, &add0_src1_desc));

    const dnnl_graph_logical_tensor_t *partition1_inputs[] = {&conv0_src_desc,
            &conv0_weight_desc, &conv0_bias_desc, &batch_norm0_scale_desc,
            &batch_norm0_shift_desc, &batch_norm0_mean_desc,
            &batch_norm0_variance_desc, &add0_src1_desc};
    const dnnl_graph_logical_tensor_t *partition1_outputs[] = {&relu0_dst_desc};
    DNNL_GRAPH_CHECK(dnnl_graph_partition_compile(partitions[1], cpartitions[1],
            8, partition1_inputs, 1, partition1_outputs, engine));
    printf("Success!\n");
    /// Step 7: alloc memory and execute compiled partition
    printf("Step 7: Alloc memory and execute--------");
    dnnl_graph_stream_t *stream = NULL;
    DNNL_GRAPH_CHECK(dnnl_graph_stream_create(&stream, engine));

    /// Execute compiled_partition[0]
    /// Alloc used buffer
    size_t conv1_src_size, conv1_weight_size, conv1_bias_size,
            batch_norm1_scale_size, batch_norm1_shift_size,
            batch_norm1_mean_size, batch_norm1_variance_size,
            batch_norm1_dst_size;
    float *conv1_src_data = NULL, *conv1_weight_data = NULL,
          *conv1_bias_data = NULL, *batch_norm1_scale_data = NULL,
          *batch_norm1_shift_data = NULL, *batch_norm1_mean_data = NULL,
          *batch_norm1_variance_data = NULL, *batch_norm1_dst_data = NULL;

    dnnl_graph_logical_tensor_t temp;
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV1_SRC_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv1_src_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV1_WEI_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv1_weight_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV1_BIAS_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv1_bias_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_SCALE_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm1_scale_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_SHIFT_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm1_shift_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_MEAN_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm1_mean_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_VARIANCE_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm1_variance_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], BATCH_NORM1_DST_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm1_dst_size));

    conv1_src_data = (float *)malloc(conv1_src_size);
    conv1_weight_data = (float *)malloc(conv1_weight_size);
    conv1_bias_data = (float *)malloc(conv1_bias_size);
    batch_norm1_scale_data = (float *)malloc(batch_norm1_scale_size);
    batch_norm1_shift_data = (float *)malloc(batch_norm1_shift_size);
    batch_norm1_mean_data = (float *)malloc(batch_norm1_mean_size);
    batch_norm1_variance_data = (float *)malloc(batch_norm1_variance_size);
    batch_norm1_dst_data = (float *)malloc(batch_norm1_dst_size);

    if (!conv1_src_data || !conv1_weight_data || !conv1_bias_data
            || !batch_norm1_scale_data || !batch_norm1_shift_data
            || !batch_norm1_mean_data || !batch_norm1_variance_data
            || !batch_norm1_dst_data) {
        printf("Error: alloc memory failed\n");
        exit(1);
    }
    /// Set value of conv1 inputs (data, weight, bias)
    for (int i = 0; i < conv1_src_size / sizeof(float); i++) {
        conv1_src_data[i] = 1.0f;
    }
    for (int i = 0; i < conv1_weight_size / sizeof(float); i++) {
        conv1_weight_data[i] = 1.0f;
    }
    for (int i = 0; i < conv1_bias_size / sizeof(float); i++) {
        conv1_bias_data[i] = 1.0f;
    }
    for (int i = 0; i < batch_norm1_scale_size / sizeof(float); i++) {
        batch_norm1_scale_data[i] = 1.0f;
    }
    for (int i = 0; i < batch_norm1_shift_size / sizeof(float); i++) {
        batch_norm1_shift_data[i] = 0.0f;
    }
    for (int i = 0; i < batch_norm1_mean_size / sizeof(float); i++) {
        batch_norm1_mean_data[i] = 0.0f;
    }
    for (int i = 0; i < batch_norm1_variance_size / sizeof(float); i++) {
        batch_norm1_variance_data[i] = 1.0f;
    }

    /// Wrap buffer and dnnl_graph_logical_tensor to dnnl_graph_tensor
    dnnl_graph_tensor_t *conv1_src, *conv1_weight, *conv1_bias,
            *batch_norm1_scale, *batch_norm1_shift, *batch_norm1_mean,
            *batch_norm1_variance, *batch_norm1_dst;
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv1_src, &conv1_src_desc, engine, conv1_src_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv1_weight, &conv1_weight_desc, engine, conv1_weight_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv1_bias, &conv1_bias_desc, engine, conv1_bias_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm1_scale,
            &batch_norm1_scale_desc, engine, batch_norm1_scale_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm1_shift,
            &batch_norm1_shift_desc, engine, batch_norm1_shift_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm1_mean,
            &batch_norm1_mean_desc, engine, batch_norm1_mean_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm1_variance,
            &batch_norm1_variance_desc, engine, batch_norm1_variance_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm1_dst,
            &batch_norm1_dst_desc, engine, batch_norm1_dst_data));

    /// Execute the compiled partition
    /// \note The given inputs and outputs tensors should correspond
    /// to the logical tensors in the compile process one-to-one. And
    /// their order should be same too. Because the dnnl graph implementation recorded the index
    /// of logical tensor it needs in the given ones, and the index will
    /// used here to take out required tensors.
    const dnnl_graph_tensor_t *cpartition0_inputs[]
            = {conv1_src, conv1_weight, conv1_bias, batch_norm1_scale,
                    batch_norm1_shift, batch_norm1_mean, batch_norm1_variance};
    const dnnl_graph_tensor_t *cpartition0_outputs[] = {batch_norm1_dst};

    /// Execute compiled_partition[0]
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_execute(cpartitions[0],
            stream, 7, cpartition0_inputs, 1, cpartition0_outputs));

    /// Alloc used buffer
    size_t conv0_src_size, conv0_weight_size, conv0_bias_size,
            batch_norm0_scale_size, batch_norm0_shift_size,
            batch_norm0_mean_size, batch_norm0_variance_size, add0_src1_size,
            relu0_dst_size;
    float *conv0_src_data = NULL, *conv0_weight_data = NULL,
          *conv0_bias_data = NULL, *batch_norm0_scale_data = NULL,
          *batch_norm0_shift_data = NULL, *batch_norm0_mean_data = NULL,
          *batch_norm0_variance_data = NULL, *add0_src1_data = NULL,
          *relu0_dst_data = NULL;

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], CONV0_SRC_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_src_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], CONV0_WEI_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_weight_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], CONV0_BIAS_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_bias_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], BATCH_NORM0_SCALE_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm0_scale_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], BATCH_NORM0_SHIFT_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm0_shift_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], BATCH_NORM0_MEAN_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm0_mean_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], BATCH_NORM0_VARIANCE_ID, &temp));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_get_mem_size(
            &temp, &batch_norm0_variance_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], ADD0_SRC1_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &add0_src1_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], RELU0_DST_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &relu0_dst_size));

    conv0_src_data = (float *)malloc(conv0_src_size);
    conv0_weight_data = (float *)malloc(conv0_weight_size);
    conv0_bias_data = (float *)malloc(conv0_bias_size);
    batch_norm0_scale_data = (float *)malloc(batch_norm0_scale_size);
    batch_norm0_shift_data = (float *)malloc(batch_norm0_shift_size);
    batch_norm0_mean_data = (float *)malloc(batch_norm0_mean_size);
    batch_norm0_variance_data = (float *)malloc(batch_norm0_variance_size);
    add0_src1_data = (float *)malloc(add0_src1_size);
    relu0_dst_data = (float *)malloc(relu0_dst_size);

    if (!conv0_src_data || !conv0_weight_data || !conv0_bias_data
            || !batch_norm0_scale_data || !batch_norm0_shift_data
            || !batch_norm0_mean_data || !batch_norm0_variance_data
            || !add0_src1_data || !relu0_dst_data) {
        printf("Error: alloc memory failed\n");
        exit(1);
    }
    /// Set value of conv0 inputs (data, weight, bias)
    for (int i = 0; i < conv0_src_size / sizeof(float); i++) {
        conv0_src_data[i] = 1.0f;
    }
    for (int i = 0; i < conv0_weight_size / sizeof(float); i++) {
        conv0_weight_data[i] = 1.0f;
    }
    for (int i = 0; i < conv0_bias_size / sizeof(float); i++) {
        conv0_bias_data[i] = 1.0f;
    }
    for (int i = 0; i < batch_norm0_scale_size / sizeof(float); i++) {
        batch_norm0_scale_data[i] = 1.0f;
    }
    for (int i = 0; i < batch_norm0_shift_size / sizeof(float); i++) {
        batch_norm0_shift_data[i] = 0.0f;
    }
    for (int i = 0; i < batch_norm0_mean_size / sizeof(float); i++) {
        batch_norm0_mean_data[i] = 0.0f;
    }
    for (int i = 0; i < batch_norm0_variance_size / sizeof(float); i++) {
        batch_norm0_variance_data[i] = 1.0f;
    }

    /// Wrap buffer and dnnl_graph_logical_tensor to dnnl_graph_tensor
    dnnl_graph_tensor_t *conv0_src, *conv0_weight, *conv0_bias,
            *batch_norm0_scale, *batch_norm0_shift, *batch_norm0_mean,
            *batch_norm0_variance, *relu0_dst;
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_src, &conv0_src_desc, engine, conv0_src_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_weight, &conv0_weight_desc, engine, conv0_weight_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_bias, &conv0_bias_desc, engine, conv0_bias_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm0_scale,
            &batch_norm0_scale_desc, engine, batch_norm0_scale_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm0_shift,
            &batch_norm0_shift_desc, engine, batch_norm0_shift_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm0_mean,
            &batch_norm0_mean_desc, engine, batch_norm0_mean_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(&batch_norm0_variance,
            &batch_norm0_variance_desc, engine, batch_norm0_variance_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &relu0_dst, &relu0_dst_desc, engine, relu0_dst_data));

    /// Execute compiled_partition[1]
    const dnnl_graph_tensor_t *cpartition1_inputs[] = {conv0_src, conv0_weight,
            conv0_bias, batch_norm0_scale, batch_norm0_shift, batch_norm0_mean,
            batch_norm0_variance, batch_norm1_dst};
    const dnnl_graph_tensor_t *cpartition1_outputs[] = {relu0_dst};
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_execute(cpartitions[1],
            stream, 8, cpartition1_inputs, 1, cpartition1_outputs));

    DNNL_GRAPH_CHECK(dnnl_graph_stream_destroy(stream));
    printf("Success!\n");

    /// Step 8: Check correctness of the output results
    printf("Step 8: Check correctness---------------");
    float excepted_result = (1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) * 2;
    for (int i = 0; i < relu0_dst_size / sizeof(float); i++) {
        if (fabsf(excepted_result - relu0_dst_data[i]) > 1e-6f) {
            printf("Error: output result is not equal to excepted "
                   "results\n");
            exit(1);
        }
    }
    printf("Success!\n");

    /// Release resource
    free(conv0_src_data);
    free(conv0_weight_data);
    free(conv0_bias_data);
    free(batch_norm0_scale_data);
    free(batch_norm0_shift_data);
    free(batch_norm0_mean_data);
    free(batch_norm0_variance_data);

    free(conv1_src_data);
    free(conv1_weight_data);
    free(conv1_bias_data);
    free(batch_norm1_scale_data);
    free(batch_norm1_shift_data);
    free(batch_norm1_mean_data);
    free(batch_norm1_variance_data);
    free(batch_norm1_dst_data);

    free(add0_src1_data);
    free(relu0_dst_data);

    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_src));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_weight));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_bias));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm0_scale));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm0_shift));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm0_mean));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm0_variance));

    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_src));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_weight));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_bias));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm1_scale));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm1_shift));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm1_mean));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm1_variance));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(batch_norm1_dst));

    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(relu0_dst));

    DNNL_GRAPH_CHECK(dnnl_graph_partition_destroy(partitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_partition_destroy(partitions[1]));

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_destroy(cpartitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_destroy(cpartitions[1]));

    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(conv0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(bias_add0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(batch_norm0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(conv1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(bias_add1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(batch_norm1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(add0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(relu0));

    DNNL_GRAPH_CHECK(dnnl_graph_graph_destroy(graph));
    DNNL_GRAPH_CHECK(dnnl_graph_engine_destroy(engine));
    DNNL_GRAPH_CHECK(dnnl_graph_allocator_destroy(allocator));

    printf("Example pass\n");

    return 0;
}
