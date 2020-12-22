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

#include "test_api_common.h"

TEST(c_api_test, add_op) {
    dnnl_graph_graph_t *agraph = NULL;
    dnnl_graph_op_t *op0 = NULL;
    dnnl_graph_op_t *op1 = NULL;
    dnnl_graph_engine_kind_t engine = dnnl_graph_cpu;

#define ADD_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op0); \
        op0 = NULL; \
        dnnl_graph_op_destroy(op1); \
        op1 = NULL; \
        dnnl_graph_graph_destroy(agraph); \
        agraph = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op0, 1, kConvolution, "conv2d"),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op1, 2, kLog, "log"),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_create(&agraph, engine),
            dnnl_graph_result_success, ADD_OP_DESTROY);

    //op0 params
    int64_t stride[] = {4, 4};
    int64_t padding[] = {0, 0};
    int64_t dilation[] = {1, 1};
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "strides",
                           dnnl_graph_attribute_kind_is, stride, 2),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "pads_begin",
                           dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "pads_end",
                           dnnl_graph_attribute_kind_is, padding, 2),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "dilations",
                           dnnl_graph_attribute_kind_is, dilation, 2),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "data_format",
                           dnnl_graph_attribute_kind_s, "NCX", 1),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op0, "filter_format",
                           dnnl_graph_attribute_kind_s, "OIX", 1),
            dnnl_graph_result_success, ADD_OP_DESTROY);

    dnnl_graph_logical_tensor_t op0_src_desc, op0_weight_desc, op0_dst_desc;
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_src_desc, 0,
                           dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_weight_desc, 1,
                           dnnl_graph_f32, 4, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op0_dst_desc, 2,
                           dnnl_graph_f32, 0, dnnl_graph_layout_type_strided),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op0, &op0_src_desc),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op0, &op0_weight_desc),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_output(op0, &op0_dst_desc),
            dnnl_graph_result_success, ADD_OP_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, op0), dnnl_graph_result_success,
            ADD_OP_DESTROY);

    dnnl_graph_logical_tensor_t op1_src_desc, op1_dst_desc;
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op1_src_desc, 0,
                           dnnl_graph_f32, 4, dnnl_graph_layout_type_any),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_logical_tensor_init(&op1_dst_desc, 1,
                           dnnl_graph_f32, 4, dnnl_graph_layout_type_any),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_input(op1, &op1_src_desc),
            dnnl_graph_result_success, ADD_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_output(op1, &op1_dst_desc),
            dnnl_graph_result_success, ADD_OP_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_add_op(agraph, op1), dnnl_graph_result_success,
            ADD_OP_DESTROY);

    ADD_OP_DESTROY;
#undef ADD_OP_DESTROY
}
