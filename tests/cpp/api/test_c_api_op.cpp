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

/**
 * TODO: cover more op kind
*/
TEST(c_api_test, create_op) {
    dnnl_graph_op_t *op = NULL;
    dnnl_graph_op_kind_t op_kind = kConvolution;

#define CREATE_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_graph_result_success, CREATE_OP_DESTROY);
    CREATE_OP_DESTROY;
#undef CREATE_OP_DESTROY
}

TEST(c_api_test, op_attr) {
    dnnl_graph_op_t *op = NULL;
    dnnl_graph_op_t *matmul_op = NULL;
    dnnl_graph_op_kind_t op_kind = kConvolution;

#define OP_ATTR_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = NULL; \
        dnnl_graph_op_destroy(matmul_op); \
        matmul_op = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    int64_t strides[] = {4, 4};
    const char *auto_pad = "same_upper";
    int64_t groups = 2;
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "strides",
                           dnnl_graph_attribute_kind_is, &strides, 2),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "auto_pad",
                           dnnl_graph_attribute_kind_s, auto_pad, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "groups",
                           dnnl_graph_attribute_kind_i, &groups, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    const void *got_strides {};
    int64_t strides_num {};
    const void *got_auto_pad {};
    int64_t auto_pad_num {};
    const void *got_groups {};
    int64_t groups_num {};
    ASSERT_EQ_SAFE(
            dnnl_graph_op_get_attr(op, "strides", dnnl_graph_attribute_kind_is,
                    &got_strides, &strides_num),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    const auto new_strides = reinterpret_cast<const int64_t *>(got_strides);
    ASSERT_EQ_SAFE(strides_num, 2, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(*new_strides, 4, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(*(new_strides + 1), 4, OP_ATTR_DESTROY);

    ASSERT_EQ(
            dnnl_graph_op_get_attr(op, "auto_pad", dnnl_graph_attribute_kind_s,
                    &got_auto_pad, &auto_pad_num),
            dnnl_graph_result_success);
    const auto new_auto_pad = reinterpret_cast<const char *>(got_auto_pad);
    ASSERT_EQ(std::string(new_auto_pad), "same_upper");

    ASSERT_EQ(dnnl_graph_op_get_attr(op, "groups", dnnl_graph_attribute_kind_i,
                      &got_groups, &groups_num),
            dnnl_graph_result_success);
    const auto new_groups = reinterpret_cast<const int64_t *>(got_groups);
    ASSERT_EQ_SAFE(*new_groups, 2, OP_ATTR_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&matmul_op, 2, kMatMul, "matmul"),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    bool transpose_a = true;
    bool transpose_b = false;
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(matmul_op, "transpose_a",
                           dnnl_graph_attribute_kind_b, &transpose_a, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(matmul_op, "transpose_b",
                           dnnl_graph_attribute_kind_b, &transpose_b, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    dnnl_graph_attribute_kind_t got_transpose_a_kind
            = dnnl_graph_attribute_kind_f;
    const void *got_transpose_a {};
    int64_t transpose_a_num {};
    ASSERT_EQ_SAFE(dnnl_graph_op_get_attr_kind(
                           matmul_op, "transpose_a", &got_transpose_a_kind),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_op_get_attr(matmul_op, "transpose_a",
                    got_transpose_a_kind, &got_transpose_a, &transpose_a_num),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    const auto new_transpose_a
            = reinterpret_cast<const bool *>(got_transpose_a);
    ASSERT_EQ_SAFE(*new_transpose_a, true, OP_ATTR_DESTROY);

    dnnl_graph_attribute_kind_t got_transpose_b_kind
            = dnnl_graph_attribute_kind_f;
    const void *got_transpose_b {};
    int64_t transpose_b_num {};
    ASSERT_EQ_SAFE(dnnl_graph_op_get_attr_kind(
                           matmul_op, "transpose_b", &got_transpose_b_kind),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(
            dnnl_graph_op_get_attr(matmul_op, "transpose_b",
                    got_transpose_b_kind, &got_transpose_b, &transpose_b_num),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    const auto new_transpose_b
            = reinterpret_cast<const bool *>(got_transpose_b);
    ASSERT_EQ_SAFE(*new_transpose_b, false, OP_ATTR_DESTROY);

    OP_ATTR_DESTROY;
#undef OP_ATTR_DESTROY
}
