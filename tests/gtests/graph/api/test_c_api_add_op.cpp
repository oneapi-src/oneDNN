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

#include "oneapi/dnnl/dnnl_graph.h"

#include "test_api_common.h"

TEST(CAPI, AddOp) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_graph_op_t op0 = nullptr;
    dnnl_graph_op_t op1 = nullptr;
    dnnl_engine_kind_t engine = dnnl_cpu;

#define ADD_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op0); \
        op0 = NULL; \
        dnnl_graph_op_destroy(op1); \
        op1 = NULL; \
        dnnl_graph_graph_destroy(agraph); \
        agraph = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(
            dnnl_graph_op_create(&op0, 1, dnnl_graph_op_convolution, "conv2d"),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op1, 2, dnnl_graph_op_log, "log"),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_create(&agraph, engine), dnnl_success,
            ADD_OP_DESTROY);

    //op0 params
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilation[] = {1, 1};
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op0, dnnl_graph_op_attr_strides, stride, 2),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op0, dnnl_graph_op_attr_pads_begin, padding, 2),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op0, dnnl_graph_op_attr_pads_end, padding, 2),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op0, dnnl_graph_op_attr_dilations, dilation, 2),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           op0, dnnl_graph_op_attr_data_format, "NCX", 1),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           op0, dnnl_graph_op_attr_weights_format, "OIX", 1),
            dnnl_success, ADD_OP_DESTROY);

    dnnl_graph_logical_tensor_t op0_src_desc, op0_weight_desc, op0_dst_desc;
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_src_desc, 0, dnnl_f32, 4,
                           dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_weight_desc, 1, dnnl_f32,
                           4, dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_dst_desc, 2, dnnl_f32, 0,
                           dnnl_graph_layout_type_strided,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op0, &op0_src_desc), dnnl_success,
            ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op0, &op0_weight_desc), dnnl_success,
            ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_output(op0, &op0_dst_desc), dnnl_success,
            ADD_OP_DESTROY);

    ASSERT_EQ_SAFE(
            dnnl_graph_add_op(agraph, op0), dnnl_success, ADD_OP_DESTROY);

    dnnl_graph_logical_tensor_t op1_src_desc, op1_dst_desc;
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op1_src_desc, 0, dnnl_f32, 4,
                           dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op1_dst_desc, 1, dnnl_f32, 4,
                           dnnl_graph_layout_type_any,
                           dnnl_graph_tensor_property_undef),
            dnnl_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op1, &op1_src_desc), dnnl_success,
            ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_output(op1, &op1_dst_desc), dnnl_success,
            ADD_OP_DESTROY);

    ASSERT_EQ_SAFE(
            dnnl_graph_add_op(agraph, op1), dnnl_success, ADD_OP_DESTROY);

    ADD_OP_DESTROY;
#undef ADD_OP_DESTROY
}
