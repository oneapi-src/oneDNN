/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

/**
 * TODO: cover more op kind
*/
TEST(CAPI, CreateOp) {
    dnnl_graph_op_t op = nullptr;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_convolution;

#define CREATE_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = nullptr; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_success, CREATE_OP_DESTROY);
    CREATE_OP_DESTROY;
#undef CREATE_OP_DESTROY
}

TEST(CAPI, OpAttr) {
    dnnl_graph_op_t op = nullptr;
    dnnl_graph_op_t matmul_op = nullptr;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_convolution;

#define OP_ATTR_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = nullptr; \
        dnnl_graph_op_destroy(matmul_op); \
        matmul_op = nullptr; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_success, OP_ATTR_DESTROY);

    int64_t strides[] = {4, 4};
    const char *auto_pad = "same_upper";
    int64_t groups = 2;
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op, dnnl_graph_op_attr_strides, strides, 2),
            dnnl_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_str(
                           op, dnnl_graph_op_attr_auto_pad, auto_pad, 1),
            dnnl_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_set_attr_s64(
                           op, dnnl_graph_op_attr_groups, &groups, 1),
            dnnl_success, OP_ATTR_DESTROY);

    ASSERT_EQ_SAFE(
            dnnl_graph_op_create(&matmul_op, 2, dnnl_graph_op_matmul, "matmul"),
            dnnl_success, OP_ATTR_DESTROY);
    bool transpose_a = true;
    bool transpose_b = false;
    ASSERT_EQ_SAFE(
            dnnl_graph_op_set_attr_bool(matmul_op,
                    dnnl_graph_op_attr_transpose_a, (uint8_t *)&transpose_a, 1),
            dnnl_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_op_set_attr_bool(matmul_op,
                    dnnl_graph_op_attr_transpose_b, (uint8_t *)&transpose_b, 1),
            dnnl_success, OP_ATTR_DESTROY);

    OP_ATTR_DESTROY;
#undef OP_ATTR_DESTROY
}

TEST(CAPI, DnnlGraphOpGetKind) {
    dnnl_graph_op_t op = nullptr;
    dnnl_graph_op_kind_t op_kind = dnnl_graph_op_wildcard;

#define CREATE_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = nullptr; \
    } while (0);
    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_success, CREATE_OP_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_get_kind(nullptr, &op_kind),
            dnnl_invalid_arguments, CREATE_OP_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_get_kind(op, nullptr), dnnl_invalid_arguments,
            CREATE_OP_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_get_kind(op, &op_kind), dnnl_success,
            CREATE_OP_DESTROY);
    ASSERT_EQ_SAFE(op_kind, dnnl_graph_op_wildcard, CREATE_OP_DESTROY);
    CREATE_OP_DESTROY;
#undef CREATE_OP_DESTROY
}
