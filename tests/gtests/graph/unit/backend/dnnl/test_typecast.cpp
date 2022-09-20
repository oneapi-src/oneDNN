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

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, Typecast) {
    graph::engine_t *engine = get_engine();

    test::vector<float> f32_val {
            12.5234537f, 0.f, -32.6735142f, -1.f, -2.8765223f, 66.66f};
    test::vector<uint16_t> bf16_val(f32_val.size(), 10);

    graph::op_t typecast_op(graph::op_kind::TypeCast);

    // prepare input/output logical tensor
    // shape (2, 3), stride (3, 1), plain format
    graph::logical_tensor_t f32_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::f32);
    graph::logical_tensor_t bf16_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, graph::data_type::bf16);

    typecast_op.add_input(f32_lt);
    typecast_op.add_output(bf16_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&typecast_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&f32_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&bf16_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    graph::stream_t *stream = get_stream();
    graph::tensor_t f32_ts(f32_lt, engine, f32_val.data());
    graph::tensor_t bf16_ts(bf16_lt, engine, bf16_val.data());

    // f32 --> bf16
    ASSERT_EQ(cp.execute(stream, {f32_ts}, {bf16_ts}), graph::status::success);
    stream->wait();
}

TEST(Compile, TypecastNegativeInput) {
    graph::engine_t *engine = get_engine();

    graph::op_t typecast_op(graph::op_kind::TypeCast);
    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, graph::data_type::f32);
    // failure case : different shape (dims)
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, graph::data_type::bf16);
    typecast_op.add_input(src_lt);
    typecast_op.add_output(dst_lt);

    graph::graph_t g(engine->kind());
    g.add_op(&typecast_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
            graph::status::invalid_shape);
}
