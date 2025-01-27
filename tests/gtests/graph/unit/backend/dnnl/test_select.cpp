/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using dims = std::vector<dim_t>;

TEST(test_select_execute, TestSelect) {

    graph::engine_t *engine = get_engine();
    std::vector<bool> cond(128, true);
    std::vector<float> src0(1, -1);
    std::vector<float> src1(12 * 128 * 128, 1);
    std::vector<float> dst(12 * 128 * 128);
    for (int i = 0; i < 64; i++)
        cond[i] = false;
    graph::op_t select_op(graph::op_kind::Select);

    graph::logical_tensor_t cond_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 128}, graph::data_type::boolean);

    graph::logical_tensor_t src0_lt
            = utils::logical_tensor_init(1, {1}, graph::data_type::f32);

    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            2, {1, 12, 128, 128}, graph::data_type::f32);

    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 12, 128, 128}, graph::data_type::f32);

    select_op.add_input(cond_lt);
    select_op.add_input(src0_lt);
    select_op.add_input(src1_lt);
    select_op.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&select_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("select_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &cond_lt, &src0_lt, &src1_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    test_tensor_t cond_ts(cond_lt, engine, cond);
    test_tensor_t src0_ts(src0_lt, engine, src0);
    test_tensor_t src1_ts(src1_lt, engine, src1);
    test_tensor_t dst_ts(dst_lt, engine, dst);

    ASSERT_EQ(cp.execute(stream, {cond_ts.get(), src0_ts.get(), src1_ts.get()},
                      {dst_ts.get()}),
            graph::status::success);
    stream->wait();
    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < 12 * 128; ++i) {
        for (size_t j = 0; j < 128; ++j) {
            if (j < 64)
                ASSERT_EQ(dst[i * 128 + j], 1);
            else
                ASSERT_EQ(dst[i * 128 + j], -1);
        }
    }
}

TEST(test_select_execute, MatmulSelect) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "MatMul");
    graph::op_t div_op(1, graph::op_kind::Divide, "div_op");
    graph::op_t select_op(2, graph::op_kind::Select, "Select");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data(32 * 56, 1);
    std::vector<float> weight_data(56 * 32, 1);
    std::vector<float> div_src1_data(1, 5);
    std::vector<float> select_src1_data(1, -1);
    std::vector<bool> cond_data(32 * 32);
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            cond_data[i * 32 + j] = j > 1 ? true : false;

    std::vector<float> ref_dst_data(32 * 32);
    for (int i = 0; i < 1 * 32; i++)
        for (int j = 0; j < 32; j++)
            ref_dst_data[i * 32 + j] = j > 1 ? -1 : 11.2;
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {1, 1, 32, 56}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {1, 1, 56, 32}, graph::data_type::f32);
    graph::logical_tensor_t matmul_dst = utils::logical_tensor_init(
            2, {1, 1, 32, 32}, graph::data_type::f32);
    graph::logical_tensor_t div_src1
            = utils::logical_tensor_init(3, {1}, graph::data_type::f32);
    graph::logical_tensor_t div_dst = utils::logical_tensor_init(
            4, {1, 1, 32, 32}, graph::data_type::f32);
    graph::logical_tensor_t cond = utils::logical_tensor_init(
            5, {1, 1, 32, 32}, graph::data_type::boolean);
    graph::logical_tensor_t select_src0
            = utils::logical_tensor_init(6, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            7, {1, 1, 32, 32}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(matmul_dst);

    div_op.add_input(matmul_dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);

    select_op.add_input(cond);
    select_op.add_input(select_src0);
    select_op.add_input(div_dst);
    select_op.add_output(dst);

    graph::graph_t g(engine->kind());
    g.add_op(&matmul_op);
    g.add_op(&div_op);
    g.add_op(&select_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    ASSERT_EQ(part->get_ops().size(), 3U);

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &div_src1, &cond, &select_src0};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t weight_ts(weight, engine, weight_data);
    test_tensor_t div_src1_ts(div_src1, engine, div_src1_data);
    test_tensor_t cond_ts(cond, engine, cond_data);
    test_tensor_t select_src0_ts(select_src0, engine, select_src1_data);
    test_tensor_t dst_ts(dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm,
                      {src_ts.get(), weight_ts.get(), div_src1_ts.get(),
                              cond_ts.get(), select_src0_ts.get()},
                      {dst_ts.get()}),
            graph::status::success);

    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < 32; ++i)
        for (size_t j = 0; j < 32; ++j)
            ASSERT_EQ(dst_data[i * 32 + j], ref_dst_data[i * 32 + j]);
}
