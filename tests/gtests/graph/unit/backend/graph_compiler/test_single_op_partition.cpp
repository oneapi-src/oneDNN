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
#include "backend/graph_compiler/compiler_backend.hpp"
#include "interface/allocator.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <runtime/context.hpp>

namespace impl = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::impl::graph::tests::unit::compiler::utils;

TEST(GCSingleOpTest, BinaryOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 16}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            id_gen.get_id(), {1, 16}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 16}, graph::data_type::f32);

    std::vector<graph::op_kind_t> binary_ops = {graph::op_kind::Add,
            graph::op_kind::Subtract, graph::op_kind::Multiply,
            graph::op_kind::Divide, graph::op_kind::Maximum};

    for (size_t i = 0; i < binary_ops.size(); i++) {
        graph::op_t binary_op(id_gen.get_id(), binary_ops[i], "binary");

        binary_op.add_input(src0_lt);
        binary_op.add_input(src1_lt);
        binary_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&binary_op);
        g.finalize();

        auto &compiler_backend_ptr
                = impl::compiler_impl::compiler_backend_t::get_singleton();
        compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
        auto partitions = g.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);

        auto part = partitions[0];
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        std::vector<float> src0(16 * 16), src1(16), dst(16 * 16);

        test_tensor_t src0_ts(src0_lt, eng, src0);
        test_tensor_t src1_ts(src1_lt, eng, src1);
        test_tensor_t dst_ts(dst_lt, eng, dst);

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src0_ts.get(), src1_ts.get()}, {dst_ts.get()});
        strm->wait();
    }
}

TEST(GCSingleOpTest, UnaryActivationOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 16}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 16}, graph::data_type::f32);

    std::vector<graph::op_kind_t> unary_ops = {graph::op_kind::ReLU,
            graph::op_kind::Sigmoid, graph::op_kind::GELU, graph::op_kind::Tanh,
            graph::op_kind::SoftMax};

    for (size_t i = 0; i < unary_ops.size(); i++) {
        graph::op_t unary_op(id_gen.get_id(), unary_ops[i], "unary");

        unary_op.add_input(src_lt);
        unary_op.add_output(dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&unary_op);
        g.finalize();

        auto &compiler_backend_ptr
                = impl::compiler_impl::compiler_backend_t::get_singleton();
        compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
        auto partitions = g.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);

        auto part = partitions[0];
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
        std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, eng);

        std::vector<float> src(16 * 16), dst(16 * 16);

        test_tensor_t src_ts(src_lt, eng, src);
        test_tensor_t dst_ts(dst_lt, eng, dst);

        graph::stream_t *strm = get_stream();
        cp.execute(strm, {src_ts.get()}, {dst_ts.get()});
        strm->wait();
    }
}

TEST(GCSingleOpTest, StaticReshapeOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 16}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 8, 4, 4}, graph::data_type::f32);

    graph::op_t static_reshape_op(
            id_gen.get_id(), graph::op_kind::StaticReshape, "static_reshape");
    static_reshape_op.set_attr(
            graph::op_attr::shape, std::vector<graph::dim_t> {2, 8, 4, 4});
    static_reshape_op.set_attr(graph::op_attr::special_zero, false);
    static_reshape_op.add_input(src_lt);
    static_reshape_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&static_reshape_op);
    g.finalize();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
    auto partitions = g.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    auto part = partitions[0];
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    std::vector<float> src(16 * 16), dst(16 * 16);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get()}, {dst_ts.get()});
    strm->wait();
}

TEST(GCSingleOpTest, StaticTransposeOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 8, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 4, 4, 8}, graph::data_type::f32);

    graph::op_t static_transpose_op(id_gen.get_id(),
            graph::op_kind::StaticTranspose, "static_transpose");
    static_transpose_op.set_attr(
            graph::op_attr::order, std::vector<graph::dim_t> {0, 2, 3, 1});
    static_transpose_op.add_input(src_lt);
    static_transpose_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&static_transpose_op);
    g.finalize();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
    auto partitions = g.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    auto part = partitions[0];
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    std::vector<float> src(16 * 16), dst(16 * 16);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get()}, {dst_ts.get()});
    strm->wait();
}

TEST(GCSingleOpTest, SelectOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t cond_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 1, 1, 5}, graph::data_type::boolean);
    graph::logical_tensor_t then_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 3, 4, 5}, graph::data_type::f32);
    graph::logical_tensor_t else_lt = utils::logical_tensor_init(
            id_gen.get_id(), {1}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {2, 3, 4, 5}, graph::data_type::f32);

    graph::op_t select_op(id_gen.get_id(), graph::op_kind::Select, "select_op");
    select_op.add_input(cond_lt);
    select_op.add_input(then_lt);
    select_op.add_input(else_lt);
    select_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&select_op);
    g.finalize();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
    auto partitions = g.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    auto part = partitions[0];
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &cond_lt, &then_lt, &else_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    std::vector<float> then(2 * 3 * 4 * 5), els(1), dst(2 * 3 * 4 * 5);
    std::vector<char> cond(2 * 5);

    test_tensor_t cond_ts(cond_lt, eng, cond);
    test_tensor_t then_ts(then_lt, eng, then);
    test_tensor_t else_ts(else_lt, eng, els);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {cond_ts.get(), then_ts.get(), else_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(GCSingleOpTest, ConcatOpCompileExecution_CPU) {
    REQUIRE_SINGLE_OP_PATTERN();
    utils::id_generator_t id_gen;
    REQUIRE_CPU_ENGINE();
    graph::engine_t *eng = get_engine();

    graph::logical_tensor_t src0_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 8, 1, 8}, graph::data_type::f32);
    graph::logical_tensor_t src1_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 8, 2, 8}, graph::data_type::f32);
    graph::logical_tensor_t src2_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 8, 3, 8}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            id_gen.get_id(), {16, 8, 6, 8}, graph::data_type::f32);

    graph::op_t concat_op(id_gen.get_id(), graph::op_kind::Concat, "concat_op");
    concat_op.set_attr<int64_t>(graph::op_attr::axis, -2);
    concat_op.add_input(src0_lt);
    concat_op.add_input(src1_lt);
    concat_op.add_input(src2_lt);
    concat_op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&concat_op);
    g.finalize();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(g, impl::partition_policy::fusion);
    auto partitions = g.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    auto part = partitions[0];
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &src2_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, eng);

    std::vector<float> src0(16 * 8 * 1 * 8), src1(16 * 8 * 2 * 8),
            src2(16 * 8 * 3 * 8), dst(16 * 8 * 6 * 8);

    test_tensor_t src0_ts(src0_lt, eng, src0);
    test_tensor_t src1_ts(src1_lt, eng, src1);
    test_tensor_t src2_ts(src2_lt, eng, src2);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src0_ts.get(), src1_ts.get(), src2_ts.get()},
            {dst_ts.get()});
    strm->wait();
}
