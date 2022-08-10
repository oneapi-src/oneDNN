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

/// @example cpu_multi_times_inference.c
/// @copybrief cpu_multi_times_inference_c
/// > Annotated version: @ref cpu_multi_times_inference_c

/// @page cpu_multi_times_inference_c CPU example for conv+bias+relu+conv+bias+relu pattern
///
/// > Example code: @ref cpu_multi_times_inference.c

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/allocator.h"
#include "common/utils.h"

#include "oneapi/dnnl/dnnl_graph.h"

///////////////////////////////////////////////
//    0 (conv0_src)
//    |    1 (conv0_weight)
//    |   /
//    |  /
//   op0 (conv0)                ---
//    |       2 (conv0_bias)      |
//    |     /                     |
//   op4 (bias_add0)              |
//    |                           | partition 0
//    |                           |
//    3 (bias_add0_dst/relu0_src) |
//    |                           |
//   op1 (relu0)                ---
//    |
//    4 (relu0_dst/conv1_src)
//    |    5 (conv1_weight)
//    |   /
//    |  /
//   op2 (conv1)                ---
//    |       6 (conv1_bias)      |
//    |     /                     |
//   op5 (bias_add1)              |
//    |                           | partition 1
//    |                           |
//    7 (bias_add1_dst/relu1_src) |
//    |                           |
//   op3 (relu1)                ---
//    |
//    8 (relu1-dst)
////////////////////////////////////////////////

// Pre-define unique id for dnnl_graph_op,
// used to represent op in computation graph
#define CONV0_ID 0
#define RELU0_ID 1
#define CONV1_ID 2
#define RELU1_ID 3
#define BIAS_ADD0_ID 4
#define BIAS_ADD1_ID 5

// Predefine unique id for dnnl_graph_logical_tensor,
// used to represent edge in computation graph
#define CONV0_SRC_ID 0
#define CONV0_WEI_ID 1
#define CONV0_BIAS_ID 2
#define CONV0_DST_ID 3

#define RELU0_SRC_ID 3
#define RELU0_DST_ID 4

#define CONV1_SRC_ID 4
#define CONV1_WEI_ID 5
#define CONV1_BIAS_ID 6
#define CONV1_DST_ID 7

#define RELU1_SRC_ID 7
#define RELU1_DST_ID 8

#define BIAS_ADD0_DST_ID 9
#define BIAS_ADD1_DST_ID 10

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

#define CONV1_IC 96
#define CONV1_OC 96
#define CONV1_IH 55
#define CONV1_IW 55
#define CONV1_OH 55
#define CONV1_OW 55
#define CONV1_KS 1
#define CONV1_STRIDE 1
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
    dnnl_graph_allocator_t allocator;
    DNNL_GRAPH_CHECK(
            dnnl_graph_allocator_create(&allocator, allocate, deallocate));
    printf("Success!\n");

    /// Step 2: create an engine and set the allocator to it,
    /// the engine and allocator will used by dnnl graph backend to manage memory resource
    printf("Step 2: Create engine-------------------");
    dnnl_graph_engine_t engine;
    const int32_t device_id = 0;
    DNNL_GRAPH_CHECK(dnnl_graph_engine_create_with_allocator(
            &engine, engine_kind, device_id, allocator));
    printf("Success!\n");

    /// Step 3: create dnnl_graph_op and add attrs
    printf("Step 3: Create op-----------------------");
    dnnl_graph_op_t conv0, relu0, conv1, relu1, bias_add0, bias_add1;
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &conv0, CONV0_ID, dnnl_graph_op_convolution, "conv0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &relu0, RELU0_ID, dnnl_graph_op_relu, "relu0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &conv1, CONV1_ID, dnnl_graph_op_convolution, "conv1"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &relu1, RELU1_ID, dnnl_graph_op_relu, "relu1"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &bias_add0, BIAS_ADD0_ID, dnnl_graph_op_bias_add, "bias_add0"));
    DNNL_GRAPH_CHECK(dnnl_graph_op_create(
            &bias_add1, BIAS_ADD1_ID, dnnl_graph_op_bias_add, "bias_add1"));

    int64_t conv0_stride[] = {CONV0_STRIDE, CONV0_STRIDE};
    int64_t conv0_padding[] = {CONV0_PADDING, CONV0_PADDING};
    int64_t conv0_dilation[] = {CONV0_DILATION, CONV0_DILATION};
    int64_t conv0_groups[] = {CONV0_GROUPS};
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv0, dnnl_graph_op_attr_strides, conv0_stride, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv0, dnnl_graph_op_attr_pads_begin, conv0_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv0, dnnl_graph_op_attr_pads_end, conv0_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv0, dnnl_graph_op_attr_dilations, conv0_dilation, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            conv0, dnnl_graph_op_attr_data_format, "NCX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            conv0, dnnl_graph_op_attr_filter_format, "OIX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv0, dnnl_graph_op_attr_groups, conv0_groups, 0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            bias_add0, dnnl_graph_op_attr_data_format, "NCX", 1));

    int64_t conv1_stride[] = {CONV1_STRIDE, CONV1_STRIDE};
    int64_t conv1_padding[] = {CONV1_PADDING, CONV1_PADDING};
    int64_t conv1_dilation[] = {CONV1_DILATION, CONV1_DILATION};
    int64_t conv1_groups[] = {CONV1_GROUPS};
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv1, dnnl_graph_op_attr_strides, conv1_stride, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv1, dnnl_graph_op_attr_pads_begin, conv1_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv1, dnnl_graph_op_attr_pads_end, conv1_padding, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv1, dnnl_graph_op_attr_dilations, conv1_dilation, 2));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            conv1, dnnl_graph_op_attr_data_format, "NCX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            conv1, dnnl_graph_op_attr_filter_format, "OIX", 1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_s64(
            conv1, dnnl_graph_op_attr_groups, conv1_groups, 0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_set_attr_str(
            bias_add1, dnnl_graph_op_attr_data_format, "NCX", 1));
    printf("Success!\n");

    /// Step 4: connect dnnl_graph_op by using logical tensor, and then add dnnl_graph_op
    /// into backend graph
    printf("Step 4: Add OP to graph-----------------");
    dnnl_graph_graph_t graph;
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
                conv0_bias_desc, conv0_dst_desc, relu0_src_desc, relu0_dst_desc,
                conv1_src_desc, conv1_weight_desc, conv1_bias_desc,
                conv1_dst_desc, relu1_src_desc, relu1_dst_desc,
                bias_add0_dst_desc, bias_add1_dst_desc;
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_src_desc,
                CONV0_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_weight_desc,
                CONV0_WEI_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_bias_desc,
                CONV0_BIAS_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv0_dst_desc,
                CONV0_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&relu0_src_desc,
                RELU0_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&relu0_dst_desc,
                RELU0_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_src_desc,
                CONV1_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_weight_desc,
                CONV1_WEI_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_bias_desc,
                CONV1_BIAS_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&conv1_dst_desc,
                CONV1_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&relu1_src_desc,
                RELU1_SRC_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&relu1_dst_desc,
                RELU1_DST_ID, dnnl_graph_f32, -1, dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&bias_add0_dst_desc,
                BIAS_ADD0_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));
        DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init(&bias_add1_dst_desc,
                BIAS_ADD1_DST_ID, dnnl_graph_f32, -1,
                dnnl_graph_layout_type_undef,
                dnnl_graph_tensor_property_undef));

        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv0, &conv0_src_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv0, &conv0_weight_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(conv0, &conv0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add0, &conv0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add0, &conv0_bias_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(bias_add0, &bias_add0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(relu0, &bias_add0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(relu0, &relu0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv1, &relu0_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(conv1, &conv1_weight_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(conv1, &conv1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add1, &conv1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(bias_add1, &conv1_bias_desc));
        DNNL_GRAPH_CHECK(
                dnnl_graph_op_add_output(bias_add1, &bias_add1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_input(relu1, &bias_add1_dst_desc));
        DNNL_GRAPH_CHECK(dnnl_graph_op_add_output(relu1, &relu1_dst_desc));

        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, conv0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, bias_add0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, relu0));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, conv1));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, bias_add1));
        DNNL_GRAPH_CHECK(dnnl_graph_add_op(graph, relu1));
    }
    printf("Success!\n");

    /// Step 5: optimize the graph and get partition from it

    /// This function run pass to optimize the graph. this will fuse
    /// some ops into one op, so the graph will be rewrited
    printf("Step 5: Filter and get partition--------");
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_filter(graph, dnnl_graph_partition_policy_fusion));

    size_t partitions_num;
    DNNL_GRAPH_CHECK(
            dnnl_graph_graph_get_partition_num(graph, &partitions_num));
    if (partitions_num != 2) {
        printf("Error: partitions number is not equal to %llu\n",
                (unsigned long long)partitions_num);
        return -1;
    }

    /// Get partition from the optimized graph. Each partition will be composed
    /// of a single op (fused or unfused op)
    dnnl_graph_partition_t partitions[2];
    DNNL_GRAPH_CHECK(dnnl_graph_partition_create(&partitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_partition_create(&partitions[1]));
    DNNL_GRAPH_CHECK(dnnl_graph_graph_get_partitions(graph, 2, partitions));
    printf("Success!\n");

    /// Step 6: compile partitions
    printf("Step 6: Compile the partitions----------");
    dnnl_graph_compiled_partition_t cpartitions[2];

    /// Compile partition[0]
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_create(
            &cpartitions[0], partitions[0]));

    dnnl_graph_dims_t conv0_src_dims = {BATCH, CONV0_IC, CONV0_IH, CONV0_IW};
    dnnl_graph_dims_t conv0_weight_dims
            = {CONV0_OC, CONV0_IC, CONV0_KS, CONV0_KS};
    dnnl_graph_dims_t conv0_bias_dims = {CONV0_OC};
    dnnl_graph_dims_t conv0_dst_dims = {BATCH, CONV0_OC, CONV0_OH, CONV0_OW};
    dnnl_graph_dims_t relu0_dst_dims = {conv0_dst_dims[0], conv0_dst_dims[1],
            conv0_dst_dims[2], conv0_dst_dims[3]};

    dnnl_graph_logical_tensor_t conv0_src_desc, conv0_weight_desc,
            conv0_bias_desc, relu0_dst_desc;
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
    /// \note we set the output layout id to dnnl_graph_any to tell dnnl graph backend that the logical tensor's
    /// layout is allowed to be reset to optimal layout by itself in compilation
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&relu0_dst_desc,
            RELU0_DST_ID, dnnl_graph_f32, 4, relu0_dst_dims,
            dnnl_graph_layout_type_any, dnnl_graph_tensor_property_undef));

    /// The inputs have to contain all logical tensors required by the partition, while
    /// the outputs have to contain all logical tensors generated by the partition
    /// \note Here, we only give the required logical tensor of this partition. But as
    /// long as the above requirements are met, the inputs and outputs could contain more
    /// logical tensor than required. The compile function will search the partition's
    /// required logical tensors in the given ones.
    const dnnl_graph_logical_tensor_t *partition0_inputs[]
            = {&conv0_src_desc, &conv0_weight_desc, &conv0_bias_desc};
    const dnnl_graph_logical_tensor_t *partition0_outputs[] = {&relu0_dst_desc};
    DNNL_GRAPH_CHECK(dnnl_graph_partition_compile(partitions[0], cpartitions[0],
            3, partition0_inputs, 1, partition0_outputs, engine));

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], RELU0_DST_ID, &relu0_dst_desc));

    /// Compile partition[1]
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_create(
            &cpartitions[1], partitions[1]));

    dnnl_graph_dims_t conv1_weight_dims
            = {CONV1_OC, CONV1_IC, CONV1_KS, CONV1_KS};
    dnnl_graph_dims_t conv1_bias_dims = {CONV1_OC};
    dnnl_graph_dims_t conv1_dst_dims = {BATCH, CONV1_OC, CONV1_OH, CONV1_OW};
    dnnl_graph_dims_t relu1_dst_dims = {conv1_dst_dims[0], conv1_dst_dims[1],
            conv1_dst_dims[2], conv1_dst_dims[3]};

    /// Get cpartition[0]'s output layout id, which has been filled in compile process
    dnnl_graph_logical_tensor_t conv1_src_desc, conv1_weight_desc,
            conv1_bias_desc, relu1_dst_desc;
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(
            &conv1_weight_desc, CONV1_WEI_ID, dnnl_graph_f32, 4,
            conv1_weight_dims, dnnl_graph_layout_type_strided,
            dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&conv1_bias_desc,
            CONV1_BIAS_ID, dnnl_graph_f32, 1, conv1_bias_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));
    DNNL_GRAPH_CHECK(dnnl_graph_logical_tensor_init_with_dims(&relu1_dst_desc,
            RELU1_DST_ID, dnnl_graph_f32, 4, relu1_dst_dims,
            dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef));

    /// Reset the input data's layout id, since it's the output of cpartition[0] and its
    /// layout id has been set to optimal by cpartition[0], so cpartition[1] can directly
    /// use this layout
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV1_SRC_ID, &conv1_src_desc));

    const dnnl_graph_logical_tensor_t *partition1_inputs[]
            = {&conv1_src_desc, &conv1_weight_desc, &conv1_bias_desc};
    const dnnl_graph_logical_tensor_t *partition1_outputs[] = {&relu1_dst_desc};
    DNNL_GRAPH_CHECK(dnnl_graph_partition_compile(partitions[1], cpartitions[1],
            3, partition1_inputs, 1, partition1_outputs, engine));
    printf("Success!\n");

    /// Step 7: alloc memory and execute compiled partition
    printf("Step 7: Alloc memory and execute--------\n");
    dnnl_graph_stream_t stream = NULL;
    DNNL_GRAPH_CHECK(dnnl_graph_stream_create(&stream, engine));

    /// Alloc used buffer for compiled_partition[0]
    size_t conv0_src_size, conv0_weight_size, conv0_bias_size, relu0_dst_size;
    float *conv0_src_data, *conv0_weight_data, *conv0_bias_data,
            *relu0_dst_data;

    dnnl_graph_logical_tensor_t temp;
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV0_SRC_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_src_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV0_WEI_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_weight_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], CONV0_BIAS_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv0_bias_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[0], RELU0_DST_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &relu0_dst_size));

    conv0_src_data = (float *)malloc(conv0_src_size);
    conv0_weight_data = (float *)malloc(conv0_weight_size);
    conv0_bias_data = (float *)malloc(conv0_bias_size);
    relu0_dst_data = (float *)malloc(relu0_dst_size);

    /// Wrap buffer and dnnl_graph_logical_tensor to dnnl_graph_tensor
    dnnl_graph_tensor_t conv0_src, conv0_weight, conv0_bias, relu0_dst;
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_src, &conv0_src_desc, engine, conv0_src_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_weight, &conv0_weight_desc, engine, conv0_weight_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv0_bias, &conv0_bias_desc, engine, conv0_bias_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &relu0_dst, &relu0_dst_desc, engine, relu0_dst_data));

    /// Alloc used buffer for compiled_partition[1]
    size_t conv1_weight_size, conv1_bias_size, relu1_dst_size;
    float *conv1_weight_data, *conv1_bias_data, *relu1_dst_data;

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], CONV1_WEI_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv1_weight_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], CONV1_BIAS_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &conv1_bias_size));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_query_logical_tensor(
            cpartitions[1], RELU1_DST_ID, &temp));
    DNNL_GRAPH_CHECK(
            dnnl_graph_logical_tensor_get_mem_size(&temp, &relu1_dst_size));

    conv1_weight_data = (float *)malloc(conv1_weight_size);
    conv1_bias_data = (float *)malloc(conv1_bias_size);
    relu1_dst_data = (float *)malloc(relu1_dst_size);

    /// Wrap buffer and dnnl_graph_logical_tensor to dnnl_graph_tensor
    dnnl_graph_tensor_t conv1_src, conv1_weight, conv1_bias, relu1_dst;
    DNNL_GRAPH_CHECK(
            dnnl_graph_tensor_create(&conv1_src, &conv1_src_desc, engine,
                    relu0_dst_data)); // use compiled partition[0] output buffer
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv1_weight, &conv1_weight_desc, engine, conv1_weight_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &conv1_bias, &conv1_bias_desc, engine, conv1_bias_data));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_create(
            &relu1_dst, &relu1_dst_desc, engine, relu1_dst_data));

    /// Execute the compiled partition
    clock_t start, end;
    double duration, fps;
    for (int iter = 0; iter < 5; iter++) {
        printf("----------Iter %d------------\n", iter);
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
        for (int i = 0; i < conv1_weight_size / sizeof(float); i++) {
            conv1_weight_data[i] = 1.0f;
        }
        for (int i = 0; i < conv1_bias_size / sizeof(float); i++) {
            conv1_bias_data[i] = 1.0f;
        }

        start = clock();
        /// Execute the compiled_partition[0]
        /// \note The given inputs and outputs tensors should correspond
        /// to the logical tensors in the compile process one-to-one. And
        /// their order should be same too. Because dnnl graph implementation recorded the index
        /// of logical tensor it needs in the given ones, and the index will
        /// used here to take out required tensors.
        const_dnnl_graph_tensor_t cpartition0_inputs[]
                = {conv0_src, conv0_weight, conv0_bias};
        const_dnnl_graph_tensor_t cpartition0_outputs[] = {relu0_dst};
        DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_execute(cpartitions[0],
                stream, 3, cpartition0_inputs, 1, cpartition0_outputs));

        /// Execute the compiled_partition[1]
        const_dnnl_graph_tensor_t cpartition1_inputs[]
                = {relu0_dst, conv1_weight, conv1_bias};
        const_dnnl_graph_tensor_t cpartition1_outputs[] = {relu1_dst};
        DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_execute(cpartitions[1],
                stream, 3, cpartition1_inputs, 1, cpartition1_outputs));
        end = clock();
        duration = (double)(end - start) / CLOCKS_PER_SEC; // second
        fps = 1 / duration;
        printf("time: %gms, fps: %g\n", duration * 1000, fps);

        /// Check correctness of the output results
        float excepted_result
                = (1 * 11 * 11 * 3 + /* conv0 bias */ 1.0f) * (1 * 1 * 96)
                + /* conv1 bias */ 1.0f;
        for (int i = 0; i < relu1_dst_size / sizeof(float); i++) {
            if (fabsf(excepted_result - relu1_dst_data[i]) > 1e-6f) {
                printf("Error: output result is not equal to excepted "
                       "results\n");
                return -2;
            }
        }
        printf("Check correctness success!\n");
    }

    DNNL_GRAPH_CHECK(dnnl_graph_stream_destroy(stream));
    printf("Success!\n");

    /// Release resource
    free(conv0_src_data);
    free(conv0_weight_data);
    free(conv0_bias_data);
    free(relu0_dst_data);
    free(conv1_weight_data);
    free(conv1_bias_data);
    free(relu1_dst_data);

    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_src));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_weight));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv0_bias));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(relu0_dst));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_src));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_weight));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(conv1_bias));
    DNNL_GRAPH_CHECK(dnnl_graph_tensor_destroy(relu1_dst));

    DNNL_GRAPH_CHECK(dnnl_graph_partition_destroy(partitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_partition_destroy(partitions[1]));

    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_destroy(cpartitions[0]));
    DNNL_GRAPH_CHECK(dnnl_graph_compiled_partition_destroy(cpartitions[1]));

    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(conv0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(relu0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(conv1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(relu1));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(bias_add0));
    DNNL_GRAPH_CHECK(dnnl_graph_op_destroy(bias_add1));

    DNNL_GRAPH_CHECK(dnnl_graph_graph_destroy(graph));
    DNNL_GRAPH_CHECK(dnnl_graph_engine_destroy(engine));
    DNNL_GRAPH_CHECK(dnnl_graph_allocator_destroy(allocator));

    printf("Example pass\n");

    return 0;
}
