/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include <functional>
#include <random>

#include "interface/c_types_map.hpp"

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "backend/dnnl/dnnl_constant_tensor_cache.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_matmul_execute, MatmulFp32) {
    graph::op_t matmul_op(graph::op_kind::MatMul);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);
    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {-2.0, -1.5};
    std::vector<float> bias_data {3.75};
    std::vector<float> ref_dst_data {10};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor bias_ts(bias, eng, bias_data);
    test_tensor dst_ts(dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }

    // test with second iteration to see
    // if memory cache works correctly
    std::vector<float> src_data2 {-2.0, -1.5};
    std::vector<float> weight_data2 {-1.0, -1.0};
    std::vector<float> bias_data2 {1.5};
    std::vector<float> ref_dst_data2 {5};
    std::vector<float> dst_data2(ref_dst_data2.size(), 0.0);

    test_tensor src_ts2(src, eng, src_data2);
    test_tensor weight_ts2(weight, eng, weight_data2);
    test_tensor bias_ts2(bias, eng, bias_data2);
    test_tensor dst_ts2(dst, eng, dst_data2);

    cp.execute(strm, {src_ts2.get(), weight_ts2.get(), bias_ts2.get()},
            {dst_ts2.get()});
    strm->wait();
    dst_data2 = dst_ts2.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data2.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data2[i], ref_dst_data2[i]);
    }
}

TEST(test_matmul_execute, MatmulF16F16F16_GPU) {
    graph::op_t matmul_op(graph::op_kind::MatMul);

    graph::engine_t *eng = get_engine();
    SKIP_IF(eng->kind() != graph::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");
#ifndef NDEBUG
    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Temporarily skip for primitive bug.");
#endif

    // in real use case, the data type should be half
    std::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    std::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    std::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f16);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f16);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f16);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute, MatmulBf16Bf16Bf16) {
    graph::op_t matmul_op(graph::op_kind::MatMul);

    graph::engine_t *eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    // in real use case, the data type should be half
    std::vector<bfloat16_t> src_data {-2.0, -1.5};
    std::vector<bfloat16_t> weight_data {-2.0, -1.5};
    std::vector<bfloat16_t> dst_data {0.0};

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::bf16);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, graph::data_type::bf16);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1}, graph::data_type::bf16);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_compile, MatmulMatmulBf16Bf16Bf16) {
    graph::op_t matmul_op0(0, graph::op_kind::MatMul, "matmul_0");
    graph::op_t matmul_op1(1, graph::op_kind::MatMul, "matmul_1");

    graph::engine_t *eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {24576, 1024}, graph::data_type::bf16);
    graph::logical_tensor_t weight0 = utils::logical_tensor_init(
            1, {1024, 1024}, graph::data_type::bf16);
    graph::logical_tensor_t dst0 = utils::logical_tensor_init(
            2, {24576, 1024}, graph::data_type::bf16, graph::layout_type::any);
    graph::logical_tensor_t weight1 = utils::logical_tensor_init(
            3, {1024, 4096}, graph::data_type::bf16);
    graph::logical_tensor_t dst1 = utils::logical_tensor_init(
            4, {24576, 4096}, graph::data_type::bf16, graph::layout_type::any);

    matmul_op0.add_input(src);
    matmul_op0.add_input(weight0);
    matmul_op0.add_output(dst0);
    matmul_op1.add_input(dst0);
    matmul_op1.add_input(weight1);
    matmul_op1.add_output(dst1);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op0);
    g.add_op(&matmul_op1);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 2U);
    auto parts = g.get_partitions();

    // compile
    graph::partition_t p0;
    p0.init(parts[0]);
    graph::compiled_partition_t cp0(p0);
    std::vector<const graph::logical_tensor_t *> inputs0 {&src, &weight0};
    std::vector<const graph::logical_tensor_t *> outputs0 {&dst0};
    ASSERT_EQ(p0.compile(&cp0, inputs0, outputs0, eng), graph::status::success);

    graph::partition_t p1;
    p1.init(parts[1]);
    graph::compiled_partition_t cp1(p1);
    graph::logical_tensor_t opaque_lt;
    cp0.query_logical_tensor(2, &opaque_lt);
    ASSERT_EQ(opaque_lt.layout_type, graph::layout_type::strided);
    std::vector<const graph::logical_tensor_t *> inputs1 {&opaque_lt, &weight1};
    std::vector<const graph::logical_tensor_t *> outputs1 {&dst1};
    ASSERT_EQ(p1.compile(&cp1, inputs1, outputs1, eng), graph::status::success);
}

TEST(test_matmul_compile, MatmulBlocked) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul");

    graph::engine_t *eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {24576, 1024}, graph::data_type::bf16);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {1024, 4096}, graph::data_type::bf16);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            2, {24576, 4096}, graph::data_type::bf16, graph::layout_type::any);

    // simulate user given block src
    dnnl::memory::desc src_md({24576, 1024}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::AB48a16b);
    auto &backend_ptr = graph::dnnl_impl::dnnl_backend::get_singleton();
    graph::utils::optional_t<size_t> layout_id
            = backend_ptr.set_mem_desc(src_md);
    src.layout.layout_id = graph::backend_registry_t::encode_layout_id(
            layout_id.value(), backend_ptr.get_id());
    src.layout_type = graph::layout_type::opaque;

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto parts = g.get_partitions();

    // compile
    graph::partition_t p;
    p.init(parts[0]);
    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t opaque_lt;
    cp.query_logical_tensor(2, &opaque_lt);
    ASSERT_EQ(opaque_lt.layout_type, graph::layout_type::strided);
}

TEST(test_matmul_execute, MatmulNdx1d) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::vector<int64_t>> weight_shapes {{16}, {16, 1}};
    std::vector<std::vector<int64_t>> src_shapes {
            {4, 2, 16}, {6, 4, 2, 16}, {8, 6, 4, 2, 16}};
    std::vector<std::vector<int64_t>> dst_shapes {{4, 2}, {6, 4, 2},
            {8, 6, 4, 2}, {4, 2, 1}, {6, 4, 2, 1}, {8, 6, 4, 2, 1}};

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (size_t w_idx = 0; w_idx < weight_shapes.size(); ++w_idx) {
        for (size_t idx = 0; idx < src_shapes.size(); idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[idx + w_idx * src_shapes.size()];

            std::vector<float> src_data(product(src_shape));
            std::vector<float> weight_data(product(weight_shapes[w_idx]));
            std::vector<float> dst_data(product(dst_shape));
            std::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            graph::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, graph::data_type::f32);
            graph::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], graph::data_type::f32);
            graph::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, graph::data_type::f32);

            graph::op_t matmul_op(graph::op_kind::MatMul);
            matmul_op.add_input(src);
            matmul_op.add_input(weight);
            matmul_op.add_output(dst);

            graph::graph_t g(engine->kind());
            g.add_op(&matmul_op);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            graph::compiled_partition_t cp(p);

            std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
            std::vector<const graph::logical_tensor_t *> outputs {&dst};

            ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                    graph::status::success);

            test_tensor src_ts(src, engine, src_data);
            test_tensor weight_ts(weight, engine, weight_data);
            test_tensor dst_ts(dst, engine, dst_data);

            ASSERT_EQ(cp.execute(strm, {src_ts.get(), weight_ts.get()},
                              {dst_ts.get()}),
                    graph::status::success);
            strm->wait();

            // compute the ref results
            size_t M = product(src_shape)
                    / static_cast<size_t>(src_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].size() == 1
                    ? static_cast<size_t>(1)
                    : static_cast<size_t>(weight_shapes[w_idx].back());
            // size_t N = static_cast<size_t>(1);
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n]
                                += src_data[m * K + k] * weight_data[k * N + n];
                }
            }
            dst_data = dst_ts.as_vec_type<float>();
            // FIXME(qun) the max error will be bigger than 1e-6
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(test_matmul_execute, Matmul1dxNd) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::vector<int64_t>> src_shapes {{16}, {1, 16}};
    std::vector<std::vector<int64_t>> weight_shapes {
            {4, 16, 2}, {6, 4, 16, 2}, {8, 6, 4, 16, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {{4, 2}, {6, 4, 2},
            {8, 6, 4, 2}, {4, 1, 2}, {6, 4, 1, 2}, {8, 6, 4, 1, 2}};

    // Initialize
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t idx = 0; idx < src_shapes.size(); ++idx) {
        for (size_t w_idx = 0; w_idx < weight_shapes.size(); w_idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[w_idx + idx * weight_shapes.size()];

            std::vector<float> src_data(product(src_shape));
            std::vector<float> weight_data(product(weight_shapes[w_idx]));
            std::vector<float> dst_data(product(dst_shape));
            std::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            graph::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, graph::data_type::f32);
            graph::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], graph::data_type::f32);
            graph::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, graph::data_type::f32);

            graph::op_t matmul_op(graph::op_kind::MatMul);
            matmul_op.add_input(src);
            matmul_op.add_input(weight);
            matmul_op.add_output(dst);

            graph::graph_t g(engine->kind());
            g.add_op(&matmul_op);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            graph::compiled_partition_t cp(p);

            std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
            std::vector<const graph::logical_tensor_t *> outputs {&dst};

            p.compile(&cp, inputs, outputs, engine);

            test_tensor src_ts(src, engine, src_data);
            test_tensor weight_ts(weight, engine, weight_data);
            test_tensor dst_ts(dst, engine, dst_data);

            cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()});
            strm->wait();

            // compute the ref results
            size_t M = product(dst_shape)
                    / static_cast<size_t>(dst_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].back();
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n] += src_data[k]
                                * weight_data[m * K * N + k * N + n];
                }
            }

            // FIXME(qun) the max error will be bigger than 1e-6
            dst_data = dst_ts.as_vec_type<float>();
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(test_matmul_execute, Matmul3dx3d) {
    graph::op_t matmul_op(graph::op_kind::MatMul);
    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> weight_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> ref_dst_data {2.0, 2.0, 2.0, 2.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {4, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {4, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(2, {4, 1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasAdd) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::BiasAdd, "add_op");
    add_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> post_src_data {-2.0};
    std::vector<float> ref_dst_data {-3.75};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            4, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t add_dst = utils::logical_tensor_init(
            5, graph::data_type::f32, graph::layout_type::strided);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(post_src);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &post_src};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);
    cp.query_logical_tensor(add_dst.id, &add_dst);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor post_src_ts(post_src, engine, post_src_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), post_src_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasAddPerTensorBroadcast) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0, 2.0};
    std::vector<float> post_src_data {-2.0};
    std::vector<float> ref_dst_data {-5.0, 3.0, -4.0, 2.25};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<graph::dims> post_src_shapes = {{1}, {1, 1}};

    for (auto &post_src_shape : post_src_shapes) {
        graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
        graph::op_t add_op(1, graph::op_kind::Add, "add_op");

        // prepare logical tensor
        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, graph::data_type::f32);
        graph::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, graph::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
        ASSERT_EQ(g.add_op(&add_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

        p.compile(&cp, inputs, outputs, engine);

        test_tensor src_ts(src, engine, src_data);
        test_tensor weight_ts(weight, engine, weight_data);
        test_tensor bias_ts(bias, engine, bias_data);
        test_tensor post_src_ts(post_src, engine, post_src_data);
        test_tensor dst_ts(add_dst, engine, dst_data);

        cp.execute(strm,
                {src_ts.get(), weight_ts.get(), bias_ts.get(),
                        post_src_ts.get()},
                {dst_ts.get()});
        strm->wait();
        dst_data = dst_ts.as_vec_type<float>();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(test_matmul_execute, MatmulBiasAddPerChannelBroadcast) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0, 2.0};
    std::vector<float> post_src_data {-2.0, -1.0};
    std::vector<float> ref_dst_data {-5.0, 4.0, -4.0, 3.25};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<graph::dims> post_src_shapes = {{2}, {1, 2}};

    for (auto &post_src_shape : post_src_shapes) {
        graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
        graph::op_t add_op(1, graph::op_kind::Add, "add_op");
        // prepare logical tensor
        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, graph::data_type::f32);
        graph::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, graph::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
        ASSERT_EQ(g.add_op(&add_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

        p.compile(&cp, inputs, outputs, engine);

        test_tensor src_ts(src, engine, src_data);
        test_tensor weight_ts(weight, engine, weight_data);
        test_tensor bias_ts(bias, engine, bias_data);
        test_tensor post_src_ts(post_src, engine, post_src_data);
        test_tensor dst_ts(add_dst, engine, dst_data);

        cp.execute(strm,
                {src_ts.get(), weight_ts.get(), bias_ts.get(),
                        post_src_ts.get()},
                {dst_ts.get()});
        strm->wait();
        dst_data = dst_ts.as_vec_type<float>();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(test_matmul_compile, MatmulBiasAddUnsupportedBroadcast) {
    graph::engine_t *engine = get_engine();

    std::vector<graph::dims> post_src_shapes = {{3}, {1, 3}};

    for (auto &post_src_shape : post_src_shapes) {
        graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
        graph::op_t add_op(1, graph::op_kind::Add, "add_op");
        // prepare logical tensor
        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, graph::data_type::f32);
        graph::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, graph::data_type::f32);
        graph::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, graph::data_type::f32);
        graph::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, graph::data_type::f32);
        graph::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, graph::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        graph::graph_t g(engine->kind());
        ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
        ASSERT_EQ(g.add_op(&add_op), graph::status::success);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                graph::status::invalid_shape);
    }
}

TEST(test_matmul_compile, MatmulInt8WeightScaleSupport) {
    graph::engine_t *engine = get_engine();

    std::vector<int64_t> src_shape = {3, 3, 8, 4};
    std::vector<int64_t> weight_shape = {3, 3, 4, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape = {3, 3, 8, 2};
    size_t scales_wei_sizes = dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    std::vector<size_t> axes = {0, 1, 2, 3};
    for (auto &axis : axes) {
        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0});
        dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {1});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, axis);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {0});
        qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {1});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 3);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};
        // Matmul only support applying scale per channel along the last
        // dimension for DNNL_ARG_WEIGHTS.
        if (axis == weight_shape.size() - 1) {
            ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                    graph::status::success);
        } else {
            ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                    graph::status::unimplemented);
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulNdx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);

        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 4U);

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get()},
                {dst_s8_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.01f,
                            /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulU8U8) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<uint8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 127.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 127.f; // map to 0~127
        int64_t zp_src = 0;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_u8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::u8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_u8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_u8_ts(weight_u8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);

        // -------------------------case 1----------------------------------
        std::vector<float> case1_out_data(product(dst_shape));
        test_tensor dst_f32_ts(dst_f32, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_u8_ts, bias_f32_ts},
                          {dst_f32_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 3U);

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_u8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<float> case2_out_data(product(dst_shape));
        test_tensor dst_f32_case2_ts(dst_f32, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_u8_ts.get(), bias_f32_ts.get()},
                {dst_f32_case2_ts.get()});
        strm->wait();
        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<float>(dst_f32_ts, dst_f32_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose<float>(dst_f32_ts, dst_f32_case2_ts,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulNdx1d) {
    // compare results between: case 1: [quantize] - [dequantize] -
    // [fp32_matmul] - [quantize] case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::vector<int64_t>> src_shapes {{3, 8, 4}, {8, 4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 1}, {4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 8, 1}, {8, 1}, {3, 8}, {8}};

    for (size_t i = 0; i < weight_shapes.size(); ++i) {
        for (size_t j = 0; j < src_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[j];
            std::vector<int64_t> weight_shape = weight_shapes[i];
            std::vector<int64_t> dst_shape
                    = dst_shapes[j + i * src_shapes.size()];

            std::vector<uint8_t> src_data(product(src_shape));
            std::vector<int8_t> weight_data(product(weight_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
            std::uniform_real_distribution<float> s8_distribution(
                    -127.0f, 128.0f);
            std::generate(src_data.begin(), src_data.end(), [&]() {
                return static_cast<uint8_t>(u8_distribution(generator));
            });
            std::generate(weight_data.begin(), weight_data.end(), [&]() {
                return static_cast<int8_t>(s8_distribution(generator));
            });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

            graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(
                    graph::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            graph::op_t dqweight_op(
                    2, graph::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    graph::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

            graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
            qout_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_out});
            qout_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_out});
            qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            // prepare logical tensor
            graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, graph::data_type::u8);
            graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, graph::data_type::f32);
            graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, graph::data_type::s8);
            graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, graph::data_type::f32);
            graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, graph::data_type::f32);
            graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, graph::data_type::s8);

            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            graph::graph_t g(engine->kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.finalize();

            test_tensor src_u8_ts(src_u8, engine, src_data);
            test_tensor weight_s8_ts(weight_s8, engine, weight_data);
            // -------------------------case 1----------------------------------
            std::vector<int8_t> case1_out_data(product(dst_shape));
            test_tensor dst_s8_case1_ts(dst_s8, engine, case1_out_data);
            ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_s8_case1_ts},
                              *engine, *strm),
                    graph::status::success);

            // -------------------------case 2----------------------------------
            graph::pass::pass_base_ptr apass
                    = get_pass("x8x8x_matmul_post_ops");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            graph::compiled_partition_t cp(p);

            std::vector<const graph::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8};
            std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, engine);

            std::vector<int8_t> case2_out_data(product(dst_shape));
            test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
            cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get()},
                    {dst_s8_case2_ts.get()});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose<int8_t>(dst_s8_case1_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose<int8_t>(dst_s8_case1_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulNdx2dWithTranspose) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};

    for (size_t i = 0; i < src_shapes.size(); ++i) {
        for (size_t j = 0; j < weight_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[i];
            std::vector<int64_t> weight_shape = weight_shapes[j];
            std::vector<int64_t> bias_shape {1};
            std::vector<int64_t> dst_shape = dst_shapes[i];

            std::vector<uint8_t> src_data(product(src_shape));
            std::vector<int8_t> weight_data(product(weight_shape));
            std::vector<float> bias_data(product(bias_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
            std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
            std::uniform_real_distribution<float> s8_distribution(
                    -127.0f, 128.0f);
            std::generate(src_data.begin(), src_data.end(), [&]() {
                return static_cast<uint8_t>(u8_distribution(generator));
            });
            std::generate(weight_data.begin(), weight_data.end(), [&]() {
                return static_cast<int8_t>(s8_distribution(generator));
            });
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

            // -------------------------case 1----------------------------------
            graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>(
                    graph::op_attr::qtype, "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_src});
            dqdata_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_src});
            dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            graph::op_t dqweight_op(
                    2, graph::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>(
                    graph::op_attr::qtype, "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_wei});
            dqweight_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_wei});
            dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
            matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

            graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
            qout_op.set_attr<std::vector<int64_t>>(
                    graph::op_attr::zps, {zp_out});
            qout_op.set_attr<std::vector<float>>(
                    graph::op_attr::scales, {scale_out});
            qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

            // prepare logical tensor
            graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, graph::data_type::u8);
            graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, graph::data_type::f32);
            graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, graph::data_type::s8);
            graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, graph::data_type::f32);
            graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, graph::data_type::f32);
            graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, graph::data_type::f32);
            graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, graph::data_type::s8);

            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            matmul_op.add_input(bias_f32);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            graph::graph_t g(engine->kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.finalize();

            test_tensor src_u8_ts(src_u8, engine, src_data);
            test_tensor weight_s8_ts(weight_s8, engine, weight_data);
            test_tensor bias_f32_ts = test_tensor(bias_f32, engine, bias_data);
            // -------------------------case 1----------------------------------
            std::vector<int8_t> case1_out_data(product(dst_shape));
            test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
            ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                              {dst_s8_ts}, *engine, *strm),
                    graph::status::success);

            // -------------------------case 2----------------------------------
            graph::pass::pass_base_ptr apass
                    = get_pass("x8x8x_matmul_post_ops");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            graph::compiled_partition_t cp(p);

            std::vector<const graph::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8, &bias_f32};
            std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, engine);

            std::vector<int8_t> case2_out_data(product(dst_shape));
            test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
            cp.execute(strm,
                    {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get()},
                    {dst_s8_case2_ts.get()});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasSumNdx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(engine->kind() == graph::engine_kind::cpu
                    && dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(const auto &other_qtype : other_qtypes)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));
        std::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        // post-sum and reorder didn't support zps on gpu
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqother_op(5, graph::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_op(6, graph::op_kind::Add, "add_op");

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_s8 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        test_tensor other_s8_ts(other_s8, engine, other_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_matmul_add_post_ops_gpu"
                                : "x8x8x8_matmul_add_post_ops_cpu");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 6U);

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get(),
                        other_s8_ts.get()},
                {dst_s8_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.01f,
                            /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasBinary) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(engine->kind() == graph::engine_kind::cpu
                    && dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");

    std::vector<std::string> qtypes {"per_channel"};
    std::vector<graph::op_kind_t> binary_kinds {graph::op_kind::Multiply,
            graph::op_kind::Divide, graph::op_kind::Maximum,
            graph::op_kind::Minimum, graph::op_kind::Subtract};
    std::vector<std::vector<int64_t>> src_shapes {{3, 3, 8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {{3, 3, 8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(size_t j = 0; j < weight_shapes.size(); ++j)
    for (const auto &binary_kind : binary_kinds) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));
        std::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t binary_op(6, binary_kind, "binary_op");

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_f32 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_binary_f32 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        binary_op.add_input(dst_f32);
        binary_op.add_input(other_f32);
        binary_op.add_output(dst_binary_f32);

        qout_op.add_input(dst_binary_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&binary_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        test_tensor other_f32_ts(other_f32, engine, other_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 5U);

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get(),
                        other_f32_ts.get()},
                {dst_s8_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.01f,
                            /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasAddMul) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(engine->kind() == graph::engine_kind::cpu
                    && dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(const auto &other_qtype : other_qtypes)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // the following cmd is not implemented for gpu:
        // ./benchdnn --matmul --cfg=u8s8f32 --engine=gpu --stag=ab
        // --wtag=ab --dtag=ab --attr-oscale=common:0.000031
        // --attr-post-ops=sum:1+mul:f32:common:ab --attr-zero-points
        // =wei:per_oc:1 8x4:4x2:8x2
        if (engine->kind() == graph::engine_kind::gpu && qtype == "per_channel")
            continue;
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));
        std::vector<int8_t> add_other_data(product(dst_shape));
        std::vector<float> mul_other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(add_other_data.begin(), add_other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(mul_other_data.begin(), mul_other_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        // post-sum and reorder didn't support zps on gpu
        int64_t zp_other = other_qtype == "symmetric"
                        || engine->kind() == graph::engine_kind::gpu
                ? 0
                : 128;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqother_op(5, graph::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_op(6, graph::op_kind::Add, "add_op");

        graph::op_t mul_op(7, graph::op_kind::Multiply, "mul_op");

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_s8 = utils::logical_tensor_init(
                10, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t mul_src_f32 = utils::logical_tensor_init(
                13, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_mul_f32 = utils::logical_tensor_init(
                14, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        mul_op.add_input(dst_add_f32);
        mul_op.add_input(mul_src_f32);
        mul_op.add_output(dst_mul_f32);

        qout_op.add_input(dst_mul_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&mul_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        test_tensor add_other_s8_ts(other_s8, engine, add_other_data);
        test_tensor mul_other_f32_ts(mul_src_f32, engine, mul_other_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts,
                                  add_other_s8_ts, mul_other_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_matmul_add_post_ops_gpu"
                                : "x8x8x8_matmul_add_post_ops_cpu");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 7U);

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_s8, &mul_src_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get(),
                        add_other_s8_ts.get(), mul_other_f32_ts.get()},
                {dst_s8_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(
                    allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.01f,
                            /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasNdx2dX8s8f32) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 0;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        // -------------------------case 1----------------------------------
        std::vector<float> case1_out_data(product(dst_shape));
        test_tensor dst_f32_ts(dst_f32, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<float> case2_out_data(product(dst_shape));
        test_tensor dst_f32_case2_ts(dst_f32, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get()},
                {dst_f32_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<float>(dst_f32_ts, dst_f32_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose<float>(dst_f32_ts, dst_f32_case2_ts,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulNdx2dX8s8f32) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 0;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_output(dst_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        // -------------------------case 1----------------------------------
        std::vector<float> case1_out_data(product(dst_shape));
        test_tensor dst_f32_ts(dst_f32, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_f32_ts}, *engine,
                          *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<float> case2_out_data(product(dst_shape));
        test_tensor dst_f32_case2_ts(dst_f32, engine, case2_out_data);
        cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get()},
                {dst_f32_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<float>(dst_f32_ts, dst_f32_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose<float>(dst_f32_ts, dst_f32_case2_ts,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasGeluNdx2dX8s8f32) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = 0;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t gelu_op(4, graph::op_kind::GELU, "gelu_op");

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t gelu_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        gelu_op.add_input(dst_f32);
        gelu_op.add_output(gelu_f32);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&gelu_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        // -------------------------case 1----------------------------------
        std::vector<float> case1_out_data(product(dst_shape));
        test_tensor dst_f32_ts(gelu_f32, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&gelu_f32};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<float> case2_out_data(product(dst_shape));
        test_tensor dst_f32_case2_ts(gelu_f32, engine, case2_out_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get()},
                {dst_f32_case2_ts.get()});
        strm->wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (engine->kind() == graph::engine_kind::cpu
                && isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(
                    allclose<float>(dst_f32_ts, dst_f32_case2_ts, /*rtol*/ 0.1f,
                            /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose<float>(dst_f32_ts, dst_f32_case2_ts,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(test_matmul_compile, MatmulAddGetInplacePair) {
    graph::engine_t *eng = get_engine();

    graph::graph_t agraph(eng->kind());
    graph::op_t mm {0, graph::op_kind::MatMul, "matmul"};
    graph::op_t add {1, graph::op_kind::Add, "add"};
    graph::op_t mm2 {2, graph::op_kind::MatMul, "matmul2"};

    graph::logical_tensor_t lt_mm_src = utils::logical_tensor_init(
            1, {1, 16, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_mm_weight
            = utils::logical_tensor_init(2, {4, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_mm_out = utils::logical_tensor_init(
            3, {1, 16, 4, 4}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t lt_mm_src2 = utils::logical_tensor_init(
            4, {1, 16, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_mm_weight2
            = utils::logical_tensor_init(5, {4, 4}, graph::data_type::f32);
    graph::logical_tensor_t lt_mm_out2 = utils::logical_tensor_init(
            6, {1, 16, 4, 4}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t lt_add_out = utils::logical_tensor_init(
            7, {1, 16, 4, 4}, graph::data_type::f32, graph::layout_type::any);
    mm.add_input(lt_mm_src);
    mm.add_input(lt_mm_weight);
    mm.add_output(lt_mm_out);
    mm2.add_input(lt_mm_src2);
    mm2.add_input(lt_mm_weight2);
    mm2.add_output(lt_mm_out2);
    add.add_input(lt_mm_out);
    add.add_input(lt_mm_out2);
    add.add_output(lt_add_out);

    ASSERT_EQ(agraph.add_op(&mm), graph::status::success);
    ASSERT_EQ(agraph.add_op(&mm2), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add), graph::status::success);
    agraph.finalize();
    ASSERT_EQ(agraph.num_ops(), 3U);

    graph::pass::pass_base_ptr apass1 = get_pass("fp_matmul_post_ops");
    graph::pass::pass_base_ptr apass2 = get_pass("matmul_pass");

    apass1->run(agraph);
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);
    auto part1 = agraph.get_partitions()[0]; // matmul_add
    auto part2 = agraph.get_partitions()[1]; // matmul
    graph::partition_t p1, p2;
    p1.init(part1);
    p2.init(part2);

    graph::compiled_partition_t cp1(p1), cp2(p2);

    // compile matmul partition
    std::vector<const graph::logical_tensor_t *> inputs2 {
            &lt_mm_src2, &lt_mm_weight2};
    std::vector<const graph::logical_tensor_t *> outputs2 {&lt_mm_out2};
    p2.compile(&cp2, inputs2, outputs2, eng);
    cp2.query_logical_tensor(lt_mm_out2.id, &lt_mm_out2);

    // compile matmul_add partition
    std::vector<const graph::logical_tensor_t *> inputs1 {
            &lt_mm_src, &lt_mm_weight, &lt_mm_out2};
    std::vector<const graph::logical_tensor_t *> outputs1 {&lt_add_out};
    p1.compile(&cp1, inputs1, outputs1, eng);

    std::vector<graph::inplace_pair_t> inplace_pairs = cp1.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1U);
    ASSERT_EQ(inplace_pairs[0].input_id, lt_mm_out2.id);
    ASSERT_EQ(inplace_pairs[0].output_id, lt_add_out.id);
}

TEST(test_matmul_execute_subgraph_int8, Matmul2dx3dWithTranspose) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::vector<int64_t>> src_shapes {{8, 4}};
    std::vector<std::vector<int64_t>> weight_shapes {{8, 2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {{8, 8, 2}};

    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {1};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> int8_distribution(0.0f, 100.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(int8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(int8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f;
        float scale_wei = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_wei = 0;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        graph::op_t qdata_op(0, graph::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        qdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        qdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qweight_op.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_wei});
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_wei});
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqweight_op.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_wei});
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_wei});
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

        graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        // execute
        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        std::vector<int8_t> dst_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, dst_data);
        cp.execute(strm,
                {src_u8_ts.get(), weight_s8_ts.get(), bias_f32_ts.get()},
                {dst_s8_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasSumGetInplacePair_CPU) {
    graph::engine_t *engine = get_engine();
    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "Skip for GPU - no inplace for layout mismatch.");
    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(const auto &other_qtype : other_qtypes)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t add_op(8, graph::op_kind::Add, "add_op");

        graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqdata_op2(9, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op2.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op2.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_src});
        dqdata_op2.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op2.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op2(10, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op2.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op2.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, zp_wei);
        dqweight_op2.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op2.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op2(11, graph::op_kind::MatMul, "matmul_op");
        matmul_op2.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op2.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op2(6, graph::op_kind::Quantize, "qother_op");
        qout_op2.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op2.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        qout_op2.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        qout_op2.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqout_o2p(7, graph::op_kind::Dequantize, "dqother_op");
        dqout_o2p.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqout_o2p.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqout_o2p.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqout_o2p.set_attr<int64_t>(graph::op_attr::axis, 0);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);

        graph::logical_tensor_t src_u8_2 = utils::logical_tensor_init(
                21, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq_2 = utils::logical_tensor_init(
                22, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8_2 = utils::logical_tensor_init(
                24, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq_2 = utils::logical_tensor_init(
                25, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32_2 = utils::logical_tensor_init(
                26, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32_2 = utils::logical_tensor_init(
                27, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8_2 = utils::logical_tensor_init(
                28, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t dst_f32_dq_2 = utils::logical_tensor_init(
                29, dst_shape, graph::data_type::f32);

        graph::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);

        dqdata_op2.add_input(src_u8_2);
        dqdata_op2.add_output(src_f32_dq_2);
        dqweight_op2.add_input(weight_s8_2);
        dqweight_op2.add_output(weight_f32_dq_2);
        matmul_op2.add_input(src_f32_dq_2);
        matmul_op2.add_input(weight_f32_dq_2);
        matmul_op2.add_input(bias_f32_2);
        matmul_op2.add_output(dst_f32_2);
        qout_op2.add_input(dst_f32_2);
        qout_op2.add_output(dst_s8_2);
        dqout_o2p.add_input(dst_s8_2);
        dqout_o2p.add_output(dst_f32_dq_2);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);
        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);
        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);
        add_op.add_input(dst_f32);
        add_op.add_input(dst_f32_dq_2);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op2);
        g.add_op(&dqweight_op2);
        g.add_op(&matmul_op2);
        g.add_op(&qout_op2);
        g.add_op(&dqout_o2p);

        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.finalize();

        graph::pass::pass_base_ptr apass1
                = get_pass("x8x8x8_matmul_add_post_ops_cpu");
        graph::pass::pass_base_ptr apass2 = get_pass("x8x8x_matmul_post_ops");
        apass1->run(g);
        apass2->run(g);
        ASSERT_EQ(g.get_num_partitions(), 2U);
        auto part2 = g.get_partitions()[0]; // int8_mamtul_bias_sum
        auto part1 = g.get_partitions()[1]; // int8_matmul_bias

        // compile
        graph::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        graph::compiled_partition_t cp1(p1);
        graph::compiled_partition_t cp2(p2);

        dst_s8_2.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_ins1 {
                &src_u8_2, &weight_s8_2, &bias_f32_2};
        std::vector<const graph::logical_tensor_t *> lt_outs1 {&dst_s8_2};
        p1.compile(&cp1, lt_ins1, lt_outs1, engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        dst_s8.layout_type = graph::layout_type::any;
        std::vector<const graph::logical_tensor_t *> lt_ins2 {
                &src_u8, &weight_s8, &bias_f32, &dst_s8_2};
        std::vector<const graph::logical_tensor_t *> lt_outs2 {&dst_s8};
        p2.compile(&cp2, lt_ins2, lt_outs2, engine);

        std::vector<graph::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1U);
        ASSERT_EQ(inplace_pairs[0].input_id, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output_id, dst_s8.id);
    }
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasU8s8bf16_CPU) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // gpu doesn't support mixed int8-bf16 matmul with runtime zero points
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));
    std::vector<bfloat16_t> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(dst_bf16);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_s8, engine, weight_data);
    test_tensor bias_bf16_ts(bias_bf16, engine, bias_data);
    test_tensor dst_ts(dst_bf16, engine);
    cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get(), bias_bf16_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulU8U8bf16) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_u8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::u8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(dst_bf16);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_u8, &weight_u8};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_u8, engine, weight_data);
    test_tensor dst_ts(dst_bf16, engine);
    cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get()}, {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasAddBF16U8s8bf16_CPU) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // gpu doesn't support mixed int8-bf16 matmul with runtime zero points
    SKIP_IF(engine->kind() == graph::engine_kind::gpu, "skip on gpu");

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));
    std::vector<bfloat16_t> other_data(product(dst_shape));
    std::vector<bfloat16_t> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(other_data.begin(), other_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t add_op(5, graph::op_kind::Add, "add_op");

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t other_bf16
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    add_op.add_input(other_bf16);
    add_op.add_input(matmul_bf16);
    add_op.add_output(dst_bf16);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_bf16, &other_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_s8, engine, weight_data);
    test_tensor bias_bf16_ts(bias_bf16, engine, bias_data);
    test_tensor other_bf16_ts(other_bf16, engine, other_data);
    test_tensor dst_ts(dst_bf16, engine);
    cp.execute(strm,
            {src_u8_ts.get(), weight_s8_ts.get(), bias_bf16_ts.get(),
                    other_bf16_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasaddAddBF16U8s8bf16_CPU) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // gpu doesn't support mixed int8-bf16 matmul
    SKIP_IF(engine->kind() == graph::engine_kind::gpu,
            "skip on gpu for unsupported mixed int8-bf16 matmul with runtime ");
    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<bfloat16_t> other_data(product(dst_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(other_data.begin(), other_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-1.0f, 1.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t tc_bias_op(5, graph::op_kind::TypeCast, "typecast_bias");

    graph::op_t biasadd_op(6, graph::op_kind::BiasAdd, "biasadd_op");
    // matmul bias should add to the last dim
    biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    graph::op_t add_op(7, graph::op_kind::Add, "add_op");

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            30, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(7, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_out_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t other_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(11, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(matmul_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    biasadd_op.add_input(matmul_bf16);
    biasadd_op.add_input(bias_bf16);
    biasadd_op.add_output(bias_out_bf16);

    add_op.add_input(other_bf16);
    add_op.add_input(bias_out_bf16);
    add_op.add_output(dst_bf16);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&qweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tc_bias_op);
    g.add_op(&biasadd_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&add_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32, &other_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor other_bf16_ts(other_bf16, engine, other_data);
    test_tensor dst_ts(dst_bf16, engine);
    cp.execute(strm,
            {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get(),
                    other_bf16_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasU8s8u8MixBf16) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));
    std::vector<bfloat16_t> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t tcdst_op {5, graph::op_kind::TypeCast, "typecast_dst"};

    graph::op_t qout_op(6, graph::op_kind::Quantize, "qdout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_f32
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_u8
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    tcdst_op.add_input(matmul_bf16);
    tcdst_op.add_output(matmul_f32);

    qout_op.add_input(matmul_f32);
    qout_op.add_output(dst_u8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcdst_op);
    g.add_op(&qout_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    std::vector<uint8_t> dst_data(product(dst_shape));
    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_s8, engine, weight_data);
    test_tensor bias_bf16_ts(bias_bf16, engine, bias_data);
    test_tensor dst_ts(dst_u8, engine, dst_data);
    cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get(), bias_bf16_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasaddU8s8u8MixBf16) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t tc_bias_op(5, graph::op_kind::TypeCast, "typecast_bias");

    graph::op_t biasadd_op(6, graph::op_kind::BiasAdd, "biasadd_op");
    // matmul bias should add to the last dim
    biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    graph::op_t tcdst_op {7, graph::op_kind::TypeCast, "typecast_dst"};

    graph::op_t qout_op(8, graph::op_kind::Quantize, "qdout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            30, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(7, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_out_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_f32
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_u8
            = utils::logical_tensor_init(111, dst_shape, graph::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(matmul_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    biasadd_op.add_input(matmul_bf16);
    biasadd_op.add_input(bias_bf16);
    biasadd_op.add_output(bias_out_bf16);

    tcdst_op.add_input(bias_out_bf16);
    tcdst_op.add_output(matmul_f32);

    qout_op.add_input(matmul_f32);
    qout_op.add_output(dst_u8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tc_bias_op);
    g.add_op(&biasadd_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcdst_op);
    g.add_op(&qout_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    std::vector<uint8_t> dst_data(product(dst_shape));
    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor dst_ts(dst_u8, engine, dst_data);
    cp.execute(strm, {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasGeluU8s8u8MixBf16) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // Note(zhitao): Although this UT contains BF16 data type, but it works
    // well on old platforms such as AVX2 which does not support BF16, as the
    // graph will be lowered to an INT8 primitive. However, the library does
    // not expect users to provide unsupported data types, hence skip the case.
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && engine->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));
    std::vector<bfloat16_t> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t gelu_op {5, graph::op_kind::GELU, "gelu_op"};
    graph::op_t tcgelu_op {6, graph::op_kind::TypeCast, "typecast_gelu"};

    graph::op_t qout_op(7, graph::op_kind::Quantize, "qdout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_f32
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_u8
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    gelu_op.add_input(matmul_bf16);
    gelu_op.add_output(gelu_bf16);

    tcgelu_op.add_input(gelu_bf16);
    tcgelu_op.add_output(gelu_f32);

    qout_op.add_input(gelu_f32);
    qout_op.add_output(dst_u8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&gelu_op);
    g.add_op(&tcgelu_op);
    g.add_op(&qout_op);
    g.finalize();

    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    pm.run_passes(g, "", graph::partition_policy::fusion);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_s8, engine, weight_data);
    test_tensor bias_bf16_ts(bias_bf16, engine, bias_data);
    test_tensor dst_ts(dst_u8, engine);
    cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get(), bias_bf16_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulBiasaddGeluU8s8u8MixBf16) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(10, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

    graph::op_t tc_bias_op(5, graph::op_kind::TypeCast, "typecast_bias");

    graph::op_t biasadd_op(6, graph::op_kind::BiasAdd, "biasadd_op");
    // matmul bias should add to the last dim
    biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    graph::op_t gelu_op {7, graph::op_kind::GELU, "gelu_op"};
    graph::op_t tcgelu_op {8, graph::op_kind::TypeCast, "typecast_gelu"};

    graph::op_t qout_op(9, graph::op_kind::Quantize, "qdout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            30, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(7, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_out_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_bf16
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_f32
            = utils::logical_tensor_init(11, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_u8
            = utils::logical_tensor_init(12, dst_shape, graph::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(matmul_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    biasadd_op.add_input(matmul_bf16);
    biasadd_op.add_input(bias_bf16);
    biasadd_op.add_output(bias_out_bf16);

    gelu_op.add_input(bias_out_bf16);
    gelu_op.add_output(gelu_bf16);

    tcgelu_op.add_input(gelu_bf16);
    tcgelu_op.add_output(gelu_f32);

    qout_op.add_input(gelu_f32);
    qout_op.add_output(dst_u8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tc_bias_op);
    g.add_op(&biasadd_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&gelu_op);
    g.add_op(&tcgelu_op);
    g.add_op(&qout_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_bf32_ts(bias_f32, engine, bias_data);
    test_tensor dst_ts(dst_u8, engine);
    cp.execute(strm, {src_u8_ts.get(), weight_f32_ts.get(), bias_bf32_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute, MatmulTransposeReorder) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reorder_stride {16, 8, 2, 1};

    // prepare fp32 data
    std::vector<float> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });

    graph::op_t matmul_op(1, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t transpose_op(
            2, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reorder_op(3, graph::op_kind::Reorder, "reorder_op");

    // prepare logical tensor
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            1, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(2, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            3, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t reorder_f32 = utils::logical_tensor_init(
            4, transpose_shape, reorder_stride, graph::data_type::f32);

    matmul_op.add_input(src_f32);
    matmul_op.add_input(weight_f32);
    matmul_op.add_output(dst_f32);

    transpose_op.add_input(dst_f32);
    transpose_op.add_output(transpose_f32);

    reorder_op.add_input(transpose_f32);
    reorder_op.add_output(reorder_f32);

    graph::graph_t g(engine->kind());
    g.add_op(&matmul_op);
    g.add_op(&transpose_op);
    g.add_op(&reorder_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_transpose_reorder");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_f32, &weight_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&reorder_f32};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_f32_ts(src_f32, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor dst_ts(reorder_f32, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(
                strm, {src_f32_ts.get(), weight_f32_ts.get()}, {dst_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8, QuantWeiMatmulBiasTransposeReorder) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reorder_stride {16, 8, 2, 1};

    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 255.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 3);

    graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 3);

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t transpose_op(
            5, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reorder_op(6, graph::op_kind::Reorder, "reorder_op");

    graph::op_t qout_op(7, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            3, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            8, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t reorder_f32 = utils::logical_tensor_init(
            10, transpose_shape, reorder_stride, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            15, transpose_shape, reorder_stride, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    transpose_op.add_input(dst_f32);
    transpose_op.add_output(transpose_f32);

    reorder_op.add_input(transpose_f32);
    reorder_op.add_output(reorder_f32);

    qout_op.add_input(reorder_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&transpose_op);
    g.add_op(&reorder_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x_matmul_transpose_reorder");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    std::vector<int8_t> dst_out_data(product(dst_shape));
    test_tensor dst_ts(dst_s8, engine, dst_out_data);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8, QuantWeiMixBf16MatmulTransposeReorder) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reorder_stride {16, 8, 2, 1};

    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 255.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op(2, graph::op_kind::TypeCast, "tcdata_op");

    graph::op_t qweight_op(3, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 3);

    graph::op_t dqweight_op(4, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 3);

    graph::op_t tcweight_op(5, graph::op_kind::TypeCast, "tcweight_op");

    graph::op_t matmul_op(6, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t transpose_op(
            7, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reorder_op(8, graph::op_kind::Reorder, "reorder_op");

    graph::op_t tcout_op(9, graph::op_kind::TypeCast, "tcout_op");

    graph::op_t qout_op(10, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16_dq
            = utils::logical_tensor_init(3, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(5, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            6, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16_dq = utils::logical_tensor_init(
            7, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t transpose_bf16 = utils::logical_tensor_init(
            9, transpose_shape, graph::data_type::bf16);
    graph::logical_tensor_t reorder_bf16 = utils::logical_tensor_init(
            10, transpose_shape, reorder_stride, graph::data_type::bf16);
    graph::logical_tensor_t reorder_f32 = utils::logical_tensor_init(
            11, transpose_shape, reorder_stride, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            12, transpose_shape, reorder_stride, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16_dq);

    matmul_op.add_input(src_bf16_dq);
    matmul_op.add_input(weight_bf16_dq);
    matmul_op.add_output(dst_bf16);

    transpose_op.add_input(dst_bf16);
    transpose_op.add_output(transpose_bf16);

    reorder_op.add_input(transpose_bf16);
    reorder_op.add_output(reorder_bf16);

    tcout_op.add_input(reorder_bf16);
    tcout_op.add_output(reorder_f32);

    qout_op.add_input(reorder_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&tcdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&tcweight_op);
    g.add_op(&matmul_op);
    g.add_op(&transpose_op);
    g.add_op(&reorder_op);
    g.add_op(&tcout_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x_tc_matmul_transpose_reorder");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_u8, &weight_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    std::vector<int8_t> dst_out_data(product(dst_shape));
    test_tensor dst_ts(dst_s8, engine, dst_out_data);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(
                strm, {src_u8_ts.get(), weight_f32_ts.get()}, {dst_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute, MatmulScalarOutput) {
    graph::op_t matmul_op(graph::op_kind::MatMul);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);
    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {-2.0, -1.5, 1.0};
    std::vector<float> weight_data {-2.0, -1.5, 1.0};
    std::vector<float> ref_dst_data {7.25};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {3}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            2, graph::data_type::f32, graph::layout_type::any);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    graph::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 0);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(scalar_lt, eng, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute_subgraph_int8, QuantWeiMatmulBiasSumNdx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> weight_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    std::vector<float> scales = {1.f, 1 / 127.f};
    std::vector<int64_t> zps = {0, 110};
    for_(const auto &scale : scales)
    for_(const auto &zp : zps)
    for_(const auto &qtype : qtypes)
    for_(const auto &wei_qtype : weight_qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<float> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));
        std::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 127.0f);
        std::uniform_real_distribution<float> s8_distribution(0.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = scale; // map to 0~255
        float scale_other = scale;
        float scale_out = scale;
        // reorder with zps is not supported on GPU
        int64_t zp_src = engine->kind() == graph::engine_kind::gpu ? 0 : zp;
        int64_t zp_other = engine->kind() == graph::engine_kind::gpu ? 0 : zp;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : zp;

        auto generate_zps = [&]() {
            // backend integration doesn't support per_channel asym quant now.
            if (qtype == "per_channel" || wei_qtype == "symmetric"
                    || engine->kind() == graph::engine_kind::gpu)
                return 0;
            else
                return (int)zp;
        };

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, scale);
        std::vector<int64_t> zp_wei(scales_wei_sizes, generate_zps());

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqother_op(6, graph::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>(
                graph::op_attr::zps, {zp_other});
        dqother_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_other});
        dqother_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_op(7, graph::op_kind::Add, "add_op");

        // The logical tensor in graph stage should only have valid id and dtype
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, graph::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.finalize();

        // prepare in/out full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        // set weight to be constant
        weight_f32.property = graph::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, graph::data_type::s8);
        bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        // set bias to be constant
        bias_f32.property = graph::property_type::constant;

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_f32_ts(weight_f32, engine, weight_data);
        test_tensor other_s8_ts(other_s8, engine, other_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass
                = get_pass(engine->kind() == graph::engine_kind::gpu
                                ? "x8s8x8_matmul_add_post_ops_gpu"
                                : "x8x8x8_matmul_add_post_ops_cpu");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32, &other_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        for (size_t iter = 0; iter < 5; iter++) {
            cp.execute(strm,
                    {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get(),
                            other_s8_ts.get()},
                    {dst_s8_case2_ts.get()});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, U8S8U8MatmulAddF32) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {{8, 4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {{8, 2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<float> weight_data(product(weight_shape));
        std::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 127.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 127.f; // map to 0~127
        float scale_out = 1 / 255.f;
        int64_t zp_src = 90;
        int64_t zp_out = 78;

        std::vector<float> scale_wei(dst_shape.back(), 1 / 127.f);
        std::vector<int64_t> zp_wei(dst_shape.back(), 0);

        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t add_op(6, graph::op_kind::Add, "add_op");

        // The logical tensor in graph stage should only have valid id and dtype
        auto src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        weight_f32.property = graph::property_type::constant;
        auto weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                6, dst_shape, graph::data_type::f32);
        auto other_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::f32);
        auto dst_u8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_output(dst_f32);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_u8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_f32_ts(weight_f32, engine, weight_data);
        test_tensor other_f32_ts(other_f32, engine, other_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_u8_ts(dst_u8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_f32_ts, other_f32_ts},
                          {dst_u8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &other_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_u8_case2_ts(dst_u8, engine, case2_out_data);
        for (size_t iter = 0; iter < 5; iter++) {
            cp.execute(strm,
                    {src_u8_ts.get(), weight_f32_ts.get(), other_f32_ts.get()},
                    {dst_u8_case2_ts.get()});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose<uint8_t>(dst_u8_ts, dst_u8_case2_ts,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose<uint8_t>(dst_u8_ts, dst_u8_case2_ts,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, QuantWeiMatmulBiasNdx2dWithTranspose) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};

    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<float> weight_data(product(weight_shape));
        std::vector<float> bias_data(product(bias_shape));
        std::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);

        graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        // prepare logical tensor
        // The logical tensor in graph stage should only have valid id and dtype
        auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, graph::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, graph::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, graph::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::s8);
        auto bias_f32 = utils::logical_tensor_init(6, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.finalize();

        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        // set weight to be constant
        weight_f32.property = graph::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);
        bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        // set bias to be constant
        bias_f32.property = graph::property_type::constant;

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_f32_ts(weight_f32, engine, weight_data);
        test_tensor bias_f32_ts(bias_f32, engine, bias_data);
        // -------------------------case 1----------------------------------
        std::vector<int8_t> case1_out_data(product(dst_shape));
        test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);
        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        std::vector<int8_t> case2_out_data(product(dst_shape));
        test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
        for (size_t iter = 0; iter < 1; iter++) {
            cp.execute(strm,
                    {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                    {dst_s8_case2_ts.get()});
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute_subgraph_int8, QuantWeiMatmulBiasReluNdx2d) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<bool> with_bias_types {true, false};
    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto with_bias : with_bias_types)
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
        dqdata_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_src});
        dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        qweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>(graph::op_attr::qtype, qtype);
        dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
        dqweight_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, scale_wei);
        dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

        graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
        matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

        graph::op_t relu_op(5, graph::op_kind::ReLU, "relu_op");

        graph::op_t qout_op(6, graph::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
        qout_op.set_attr<std::vector<float>>(
                graph::op_attr::scales, {scale_out});
        qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

        // prepare logical tensor
        graph::logical_tensor_t src_u8 = utils::logical_tensor_init(
                1, src_shape, graph::data_type::u8);
        graph::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, graph::data_type::s8);
        graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, graph::data_type::f32);
        graph::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, graph::data_type::f32);
        graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                8, dst_shape, graph::data_type::s8);
        graph::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                9, dst_shape, graph::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        if (with_bias) matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        relu_op.add_input(dst_f32);
        relu_op.add_output(dst_relu_f32);

        qout_op.add_input(dst_relu_f32);
        qout_op.add_output(dst_s8);

        graph::graph_t g(engine->kind());
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&relu_op);
        g.add_op(&qout_op);
        g.finalize();

        test_tensor src_u8_ts(src_u8, engine);
        src_u8_ts.fill<uint8_t>(20, 5);
        test_tensor weight_f32_ts(weight_f32, engine);
        weight_f32_ts.fill<float>();
        test_tensor bias_f32_ts(bias_f32, engine);
        bias_f32_ts.fill<float>();
        // -------------------------case 1----------------------------------
        test_tensor dst_s8_ts(dst_s8, engine);
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                          {dst_s8_ts}, *engine, *strm),
                graph::status::success);

        // -------------------------case 2----------------------------------
        graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        if (!with_bias) lt_ins.pop_back();
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test_tensor dst_s8_case2_ts(dst_s8, engine);
        for (size_t iter = 0; iter < 5; iter++) {
            if (with_bias) {
                cp.execute(strm,
                        {src_u8_ts.get(), weight_f32_ts.get(),
                                bias_f32_ts.get()},
                        {dst_s8_case2_ts.get()});
            } else {
                cp.execute(strm, {src_u8_ts.get(), weight_f32_ts.get()},
                        {dst_s8_case2_ts.get()});
            }
            strm->wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (engine->kind() == graph::engine_kind::cpu
                    && isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(allclose<int8_t>(dst_s8_case2_ts, dst_s8_ts,
                        /*rtol*/ 0.1f,
                        /*atol*/ 1.f));
            else
                ASSERT_TRUE(allclose<int8_t>(dst_s8_case2_ts, dst_s8_ts,
                        /*rtol*/ 0.01f,
                        /*atol*/ 1.f));
        }
    }
}

TEST(test_matmul_execute, MatmulReluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t relu_op(1, graph::op_kind::ReLU, "relu_op");

    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, 1.5};
    std::vector<float> ref_dst_data {0.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    relu_op.add_input(dst);
    relu_op.add_output(relu_dst);

    graph::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&relu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(relu_dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasFusion) {
    graph::op_t matmul_op(graph::op_kind::MatMul);

    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    std::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {7.25, 4.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor bias_ts(bias, eng, bias_data);
    test_tensor dst_ts(dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulSumBroadcast1d) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");

    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    std::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    std::vector<float> add_src1_data {1.0};
    std::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &add_src1};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor add_src1_ts(add_src1, engine, add_src1_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), add_src1_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulSumFusion) {
    graph::engine_t *engine = get_engine();

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");

    std::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    std::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    std::vector<float> add_src1_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &add_src1};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor add_src1_ts(add_src1, engine, add_src1_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), add_src1_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulSumGeluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");
    graph::op_t gelu_op(2, graph::op_kind::GELU, "gelu_op");

    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    std::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    std::vector<float> other_data {1.0};
    std::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t gelu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(other);
    add_op.add_output(add_dst);
    gelu_op.add_input(add_dst);
    gelu_op.add_output(gelu_dst);

    graph::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &other};
    std::vector<const graph::logical_tensor_t *> outputs {&gelu_dst};

    p.compile(&cp, inputs, outputs, eng);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor other_ts(other, eng, other_data);
    test_tensor dst_ts(gelu_dst, eng, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), other_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulSumReluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");
    graph::op_t relu_op(2, graph::op_kind::ReLU, "relu_op");

    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    std::vector<float> weight_data {-1.0, -2.0, 3.0, 4.0};
    std::vector<float> other_data {2.5};
    std::vector<float> ref_dst_data {0, 27.5f};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(other);
    add_op.add_output(add_dst);
    relu_op.add_input(add_dst);
    relu_op.add_output(relu_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    ASSERT_EQ(g.add_op(&relu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &other};
    std::vector<const graph::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor other_ts(other, engine, other_data);
    test_tensor dst_ts(relu_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), other_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasReluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t relu_op(1, graph::op_kind::ReLU, "relu_op");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {0.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    relu_op.add_input(dst);
    relu_op.add_output(relu_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&relu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(relu_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasGeluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t gelu_op(1, graph::op_kind::GELU, "gelu_op");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    std::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, graph::data_type::f32);
    graph::logical_tensor_t gelu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    gelu_op.add_input(dst);
    gelu_op.add_output(gelu_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&gelu_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(gelu_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasRelu6Fusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t clamp_op(1, graph::op_kind::Clamp, "clamp_op");
    clamp_op.set_attr<float>(graph::op_attr::min, 0.0);
    clamp_op.set_attr<float>(graph::op_attr::max, 6.0);

    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {-2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {6.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t clamp_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    clamp_op.add_input(dst);
    clamp_op.add_output(clamp_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&clamp_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&clamp_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(clamp_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasClampFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t clamp_op(1, graph::op_kind::Clamp, "clamp_op");
    clamp_op.set_attr<float>(graph::op_attr::min, -3.0);
    clamp_op.set_attr<float>(graph::op_attr::max, 3.0);

    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {-2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {3.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t clamp_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    clamp_op.add_input(dst);
    clamp_op.add_output(clamp_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&clamp_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&clamp_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(clamp_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasEluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t elu_op(1, graph::op_kind::Elu, "elu_op");
    elu_op.set_attr<float>(graph::op_attr::alpha, 1.f);

    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {-0.75};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(exp(-0.75) - 1);
    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t elu_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    elu_op.add_input(dst);
    elu_op.add_output(elu_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&elu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&elu_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(elu_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasSigmoidFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t sigmoid_op(1, graph::op_kind::Sigmoid, "sigmoid_op");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> ref_dst_data {-0.75};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(1 / (exp(-ref_dst_data[0]) + 1));
    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t sigmoid_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    sigmoid_op.add_input(dst);
    sigmoid_op.add_output(sigmoid_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const graph::logical_tensor_t *> outputs {&sigmoid_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor dst_ts(sigmoid_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), bias_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasAddFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> post_src_data {-2.0};
    std::vector<float> ref_dst_data {-2.75};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(post_src);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &bias, &post_src};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor post_src_ts(post_src, engine, post_src_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), weight_ts.get(), bias_ts.get(), post_src_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulDivFusion) {
    graph::engine_t *engine = get_engine();

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t div_op(1, graph::op_kind::Divide, "div_op");

    std::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    std::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    std::vector<float> div_src1_data {-1.0};
    std::vector<float> ref_dst_data {
            -4.0, -3.0, -3.0, -2.25, -3.0, -3.0, -0.5, -0.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t div_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&div_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &div_src1};
    std::vector<const graph::logical_tensor_t *> outputs {&div_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor div_src1_ts(div_src1, engine, div_src1_data);
    test_tensor dst_ts(div_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), weight_ts.get(), div_src1_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulDivAddFusion) {
    graph::engine_t *engine = get_engine();

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t div_op(1, graph::op_kind::Divide, "div_op");
    graph::op_t add_op(2, graph::op_kind::Add, "add_op");

    std::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    std::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    std::vector<float> div_src1_data {-1.0};
    std::vector<float> add_src1_data {1.0, 2.0};
    std::vector<float> ref_dst_data {
            -3.0, -1.0, -2.0, -0.25, -2.0, -1.0, 0.5, 1.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {1, 2, 2, 1}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {1, 2, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t div_dst = utils::logical_tensor_init(
            4, {1, 2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_src1 = utils::logical_tensor_init(
            5, {1, 1, 1, 2}, graph::data_type::f32);
    graph::logical_tensor_t add_dst = utils::logical_tensor_init(
            6, {1, 2, 2, 2}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);
    add_op.add_input(div_dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&div_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &div_src1, &add_src1};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor div_src1_ts(div_src1, engine, div_src1_data);
    test_tensor add_src1_ts(add_src1, engine, add_src1_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), weight_ts.get(), div_src1_ts.get(),
                    add_src1_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulSwapBinaryMulAddFusion) {
    graph::engine_t *engine = get_engine();

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t div_op(1, graph::op_kind::Multiply, "div_op");
    graph::op_t add_op(2, graph::op_kind::Add, "add_op");

    std::vector<int64_t> src_shape {2, 2, 2, 2};
    std::vector<int64_t> div_shape {1};
    std::vector<int64_t> add_shape {2, 1, 2, 2};

    std::vector<float> src_data(product(src_shape));
    std::vector<float> weight_data(product(src_shape));
    std::vector<float> div_src1_data(product(div_shape));
    std::vector<float> add_src1_data(product(add_shape));
    std::vector<float> dst_data(product(src_shape), 0.0);

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(), [&]() {
        return static_cast<uint8_t>(f32_distribution(generator));
    });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });
    std::generate(div_src1_data.begin(), div_src1_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });
    std::generate(add_src1_data.begin(), add_src1_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, div_shape, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(3, src_shape, graph::data_type::f32);
    graph::logical_tensor_t div_dst
            = utils::logical_tensor_init(4, src_shape, graph::data_type::f32);
    graph::logical_tensor_t add_src1
            = utils::logical_tensor_init(5, add_shape, graph::data_type::f32);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(6, src_shape, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);
    add_op.add_input(add_src1);
    add_op.add_input(div_dst);
    add_op.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&div_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &div_src1, &add_src1};
    std::vector<const graph::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor div_src1_ts(div_src1, engine, div_src1_data);
    test_tensor add_src1_ts(add_src1, engine, add_src1_data);
    test_tensor dst_ts(add_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), weight_ts.get(), div_src1_ts.get(),
                    add_src1_ts.get()},
            {dst_ts.get()});
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, MatmulReluFusion) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [relu] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    // prepare fp32 data
    std::vector<int64_t> src_shape {8, 6};
    std::vector<int64_t> weight_shape {6, 4};
    std::vector<int64_t> dst_shape {8, 4};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> weight_data(product(weight_shape));

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_wei = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_wei = 0;

    int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

    // -------------------------case 1----------------------------------
    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_wei});
    dqweight_op.set_attr<std::vector<float>>(
            graph::op_attr::scales, {scale_wei});
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t relu_op(4, graph::op_kind::ReLU, "relu_op");

    graph::op_t qout_op(5, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_relu_f32
            = utils::logical_tensor_init(8, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    relu_op.add_input(dst_f32);
    relu_op.add_output(dst_relu_f32);

    qout_op.add_input(dst_relu_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&relu_op);
    g.add_op(&qout_op);
    g.finalize();

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_s8_ts(weight_s8, engine, weight_data);
    // -------------------------case 1----------------------------------
    std::vector<int8_t> case1_out_data(product(dst_shape));
    test_tensor dst_s8_ts(dst_s8, engine, case1_out_data);
    ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_s8_ts}, *engine,
                      *strm),
            graph::status::success);
    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src_u8, &weight_s8};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    std::vector<int8_t> case2_out_data(product(dst_shape));
    test_tensor dst_s8_case2_ts(dst_s8, engine, case2_out_data);
    cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get()},
            {dst_s8_case2_ts.get()});
    strm->wait();

    static auto isa = dnnl_get_effective_cpu_isa();
    if (isa < dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    else
        ASSERT_TRUE(allclose<int8_t>(dst_s8_ts, dst_s8_case2_ts, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}

TEST(test_matmul_execute_subgraph_int8,
        QuantWeiMatmulBiasReshapeTransposeQuantize) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {8, 4};
    std::vector<int64_t> weight_shape {4, 4};
    std::vector<int64_t> bias_shape {4};
    std::vector<int64_t> dst_shape {8, 4};
    std::vector<int64_t> reshape_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 127.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t reshape_op(5, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    graph::op_t transpose_op(
            6, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t qout_op(7, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            3, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t reshape_f32 = utils::logical_tensor_init(
            8, reshape_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            9, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            10, transpose_shape, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    reshape_op.add_input(dst_f32);
    reshape_op.add_output(reshape_f32);

    transpose_op.add_input(reshape_f32);
    transpose_op.add_output(transpose_f32);

    qout_op.add_input(transpose_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&reshape_op);
    g.add_op(&transpose_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x8_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor dst_s8_ts(dst_s8, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_s8_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8,
        QuantWeiMatmulBiasTransposeReshapeQuantize) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reshape_shape {4, 8};

    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 127.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t qweight_op(2, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(3, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t matmul_op(4, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t transpose_op(
            5, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reshape_op(6, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    graph::op_t qout_op(7, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            3, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            8, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t reshape_f32 = utils::logical_tensor_init(
            9, reshape_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            10, reshape_shape, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    transpose_op.add_input(dst_f32);
    transpose_op.add_output(transpose_f32);

    reshape_op.add_input(transpose_f32);
    reshape_op.add_output(reshape_f32);

    qout_op.add_input(reshape_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&transpose_op);
    g.add_op(&reshape_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x8_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    std::vector<int8_t> dst_out_data(product(dst_shape));
    test_tensor dst_s8_ts(dst_s8, engine, dst_out_data);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_s8_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8,
        QuantWeiMixBf16MatmulBiasReshapeTransposeQuantize) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {8, 4};
    std::vector<int64_t> weight_shape {4, 4};
    std::vector<int64_t> bias_shape {4};
    std::vector<int64_t> dst_shape {8, 4};
    std::vector<int64_t> reshape_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 127.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op(2, graph::op_kind::TypeCast, "tcdata_op");

    graph::op_t qweight_op(3, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(4, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t tcweight_op(5, graph::op_kind::TypeCast, "tcweight_op");

    graph::op_t matmul_op(6, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t tc_bias_op(7, graph::op_kind::TypeCast, "tcbias_op");

    graph::op_t biasadd_op(8, graph::op_kind::BiasAdd, "biasadd_op");
    biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    graph::op_t reshape_op(9, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    graph::op_t transpose_op(
            10, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t tcout_op(11, graph::op_kind::TypeCast, "tcout_op");

    graph::op_t qout_op(12, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16_dq
            = utils::logical_tensor_init(3, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(5, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            6, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16_dq = utils::logical_tensor_init(
            7, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(8, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(9, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t biasadd_bf16
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(11, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t reshape_bf16 = utils::logical_tensor_init(
            12, reshape_shape, graph::data_type::bf16);
    graph::logical_tensor_t transpose_bf16 = utils::logical_tensor_init(
            13, transpose_shape, graph::data_type::bf16);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            14, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            15, transpose_shape, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16_dq);

    matmul_op.add_input(src_bf16_dq);
    matmul_op.add_input(weight_bf16_dq);
    matmul_op.add_output(dst_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    biasadd_op.add_input(dst_bf16);
    biasadd_op.add_input(bias_bf16);
    biasadd_op.add_output(biasadd_bf16);

    reshape_op.add_input(biasadd_bf16);
    reshape_op.add_output(reshape_bf16);

    transpose_op.add_input(reshape_bf16);
    transpose_op.add_output(transpose_bf16);

    tcout_op.add_input(transpose_bf16);
    tcout_op.add_output(transpose_f32);

    qout_op.add_input(transpose_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&tcdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&tcweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tc_bias_op);
    g.add_op(&biasadd_op);
    g.add_op(&reshape_op);
    g.add_op(&transpose_op);
    g.add_op(&tcout_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x8_tc_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor dst_s8_ts(dst_s8, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_s8_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute_subgraph_int8,
        QuantWeiMixBf16MatmulBiasTransposeReshapeQuantize) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reshape_shape {4, 8};

    // prepare fp32 data
    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1 / 127.f;
    int64_t zp_src = 121;
    int64_t zp_out = 0;

    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op(2, graph::op_kind::TypeCast, "tcdata_op");

    graph::op_t qweight_op(3, graph::op_kind::Quantize, "qweight_op");
    qweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    qweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    qweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(4, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t tcweight_op(5, graph::op_kind::TypeCast, "tcweight_op");

    graph::op_t matmul_op(6, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t tc_bias_op(7, graph::op_kind::TypeCast, "tcbias_op");

    graph::op_t biasadd_op(8, graph::op_kind::BiasAdd, "biasadd_op");
    biasadd_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    graph::op_t transpose_op(
            9, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reshape_op(10, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    graph::op_t tcout_op(11, graph::op_kind::TypeCast, "tcout_op");

    graph::op_t qout_op(12, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16_dq
            = utils::logical_tensor_init(3, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(5, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            6, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16_dq = utils::logical_tensor_init(
            7, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(8, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(9, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t biasadd_bf16
            = utils::logical_tensor_init(10, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(11, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t transpose_bf16 = utils::logical_tensor_init(
            12, transpose_shape, graph::data_type::bf16);
    graph::logical_tensor_t reshape_bf16 = utils::logical_tensor_init(
            13, reshape_shape, graph::data_type::bf16);
    graph::logical_tensor_t reshape_f32 = utils::logical_tensor_init(
            14, reshape_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_s8 = utils::logical_tensor_init(
            15, reshape_shape, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16_dq);

    qweight_op.add_input(weight_f32);
    qweight_op.add_output(weight_s8);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16_dq);

    matmul_op.add_input(src_bf16_dq);
    matmul_op.add_input(weight_bf16_dq);
    matmul_op.add_output(dst_bf16);

    tc_bias_op.add_input(bias_f32);
    tc_bias_op.add_output(bias_bf16);

    biasadd_op.add_input(dst_bf16);
    biasadd_op.add_input(bias_bf16);
    biasadd_op.add_output(biasadd_bf16);

    transpose_op.add_input(biasadd_bf16);
    transpose_op.add_output(transpose_bf16);

    reshape_op.add_input(transpose_bf16);
    reshape_op.add_output(reshape_bf16);

    tcout_op.add_input(reshape_bf16);
    tcout_op.add_output(reshape_f32);

    qout_op.add_input(reshape_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&tcdata_op);
    g.add_op(&qweight_op);
    g.add_op(&dqweight_op);
    g.add_op(&tcweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tc_bias_op);
    g.add_op(&biasadd_op);
    g.add_op(&transpose_op);
    g.add_op(&reshape_op);
    g.add_op(&tcout_op);
    g.add_op(&qout_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("x8x8x8_tc_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_u8, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_u8_ts(src_u8, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor dst_s8_ts(dst_s8, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_u8_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_s8_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute, MatmulBiasReshapeTranspose) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {8, 4};
    std::vector<int64_t> weight_shape {4, 4};
    std::vector<int64_t> bias_shape {4};
    std::vector<int64_t> dst_shape {8, 4};
    std::vector<int64_t> reshape_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    // prepare fp32 data
    std::vector<float> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t reshape_op(1, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    graph::op_t transpose_op(
            2, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    // prepare logical tensor
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            1, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(2, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(3, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t reshape_f32 = utils::logical_tensor_init(
            4, reshape_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            5, transpose_shape, graph::data_type::f32, graph::layout_type::any);

    matmul_op.add_input(src_f32);
    matmul_op.add_input(weight_f32);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    reshape_op.add_input(dst_f32);
    reshape_op.add_output(reshape_f32);

    transpose_op.add_input(reshape_f32);
    transpose_op.add_output(transpose_f32);

    graph::graph_t g(engine->kind());
    g.add_op(&matmul_op);
    g.add_op(&reshape_op);
    g.add_op(&transpose_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("fp_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_f32, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&transpose_f32};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_f32_ts(src_f32, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    graph::logical_tensor_t lt;
    cp.query_logical_tensor(transpose_f32.id, &lt);
    test_tensor dst_f32_ts(lt, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_f32_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_f32_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute, MatmulBiasTransposeReshape) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> src_shape {2, 4, 2, 2};
    std::vector<int64_t> weight_shape {2, 4, 2, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape {2, 4, 2, 2};
    std::vector<int64_t> transpose_order {0, 2, 1, 3};
    std::vector<int64_t> transpose_shape {2, 2, 4, 2};
    std::vector<int64_t> reshape_shape {4, 8};

    // prepare fp32 data
    std::vector<float> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight and bias data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return f32_distribution(generator); });

    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t transpose_op(
            1, graph::op_kind::StaticTranspose, "transpose_op");
    transpose_op.set_attr(graph::op_attr::order, transpose_order);

    graph::op_t reshape_op(2, graph::op_kind::StaticReshape, "reshape_op");
    reshape_op.set_attr(graph::op_attr::shape, reshape_shape);
    reshape_op.set_attr(graph::op_attr::special_zero, false);

    // prepare logical tensor
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_f32 = utils::logical_tensor_init(
            1, weight_shape, graph::data_type::f32);
    weight_f32.property = graph::property_type::constant;
    graph::logical_tensor_t bias_f32
            = utils::logical_tensor_init(2, bias_shape, graph::data_type::f32);
    bias_f32.property = graph::property_type::constant;
    graph::logical_tensor_t dst_f32
            = utils::logical_tensor_init(3, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t transpose_f32 = utils::logical_tensor_init(
            4, transpose_shape, graph::data_type::f32);
    graph::logical_tensor_t reshape_f32 = utils::logical_tensor_init(
            5, reshape_shape, graph::data_type::f32);

    matmul_op.add_input(src_f32);
    matmul_op.add_input(weight_f32);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_f32);

    transpose_op.add_input(dst_f32);
    transpose_op.add_output(transpose_f32);

    reshape_op.add_input(transpose_f32);
    reshape_op.add_output(reshape_f32);

    graph::graph_t g(engine->kind());
    g.add_op(&matmul_op);
    g.add_op(&transpose_op);
    g.add_op(&reshape_op);
    g.finalize();

    // -------------------------case 2----------------------------------
    graph::pass::pass_base_ptr apass
            = get_pass("fp_matmul_reshape_transpose_reshape");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {
            &src_f32, &weight_f32, &bias_f32};
    std::vector<const graph::logical_tensor_t *> lt_outs {&reshape_f32};

    p.compile(&cp, lt_ins, lt_outs, engine);

    test_tensor src_f32_ts(src_f32, engine, src_data);
    test_tensor weight_f32_ts(weight_f32, engine, weight_data);
    test_tensor bias_f32_ts(bias_f32, engine, bias_data);
    test_tensor dst_f32_ts(reshape_f32, engine);
    for (size_t iter = 0; iter < 5; iter++) {
        cp.execute(strm,
                {src_f32_ts.get(), weight_f32_ts.get(), bias_f32_ts.get()},
                {dst_f32_ts.get()});
        strm->wait();
    }
}

TEST(test_matmul_execute, MatmulStridedScalarOutput) {
    graph::op_t matmul_op(graph::op_kind::MatMul);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, true);
    graph::engine_t *eng = get_engine();

    std::vector<float> src_data {-2.0, -1.5, 1.0};
    std::vector<float> weight_data {-2.0, -1.5, 1.0};
    std::vector<float> ref_dst_data {7.25};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {3}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            2, {}, graph::data_type::f32, graph::layout_type::strided);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    // output should be a scalar (ndims=0, layout_type=strided)
    graph::logical_tensor_t scalar_lt;
    cp.query_logical_tensor(dst.id, &scalar_lt);
    ASSERT_EQ(scalar_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(scalar_lt.ndims, 0);

    test_tensor src_ts(src, eng, src_data);
    test_tensor weight_ts(weight, eng, weight_data);
    test_tensor dst_ts(scalar_lt, eng, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulBiasAddReluFusion) {
    graph::op_t matmul_op(0, graph::op_kind::MatMul, "matmul_op");
    graph::op_t add_op(1, graph::op_kind::Add, "add_op");
    graph::op_t relu_op(2, graph::op_kind::ReLU, "relu_op");
    graph::engine_t *engine = get_engine();

    std::vector<float> src_data {-2.0, -1.5};
    std::vector<float> weight_data {2.0, -1.5};
    std::vector<float> bias_data {1.0};
    std::vector<float> post_src_data {-2.0};
    std::vector<float> ref_dst_data {0.0};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, graph::data_type::f32);
    graph::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t add_dst
            = utils::logical_tensor_init(5, {1, 1}, graph::data_type::f32);
    graph::logical_tensor_t relu_dst
            = utils::logical_tensor_init(6, {1, 1}, graph::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(post_src);
    add_op.add_output(add_dst);
    relu_op.add_input(add_dst);
    relu_op.add_output(relu_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&matmul_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    ASSERT_EQ(g.add_op(&relu_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &weight, &bias, &post_src};
    std::vector<const graph::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, engine);

    test_tensor src_ts(src, engine, src_data);
    test_tensor weight_ts(weight, engine, weight_data);
    test_tensor bias_ts(bias, engine, bias_data);
    test_tensor post_src_ts(post_src, engine, post_src_data);
    test_tensor dst_ts(relu_dst, engine, dst_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), weight_ts.get(), bias_ts.get(), post_src_ts.get()},
            {dst_ts.get()});
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_matmul_execute, MatmulEmptyInput) {
    graph::op_t matmul_op(graph::op_kind::MatMul);
    graph::engine_t *eng = get_engine();

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3, 4}, graph::data_type::f32);
    graph::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 4, 0}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            3, graph::data_type::f32, graph::layout_type::any);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&matmul_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t empty_lt;
    cp.query_logical_tensor(dst.id, &empty_lt);
    ASSERT_EQ(empty_lt.layout_type, graph::layout_type::strided);
    ASSERT_EQ(empty_lt.ndims, 3);
    ASSERT_EQ(empty_lt.dims[0], 2);
    ASSERT_EQ(empty_lt.dims[1], 3);
    ASSERT_EQ(empty_lt.dims[2], 0);
    cp.query_logical_tensor(weight.id, &weight);
    cp.query_logical_tensor(dst.id, &dst);

    test_tensor src_ts(src, eng);
    test_tensor weight_ts(weight, eng);
    test_tensor dst_ts(dst, eng);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get(), weight_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
}

TEST(test_matmul_execute_subgraph_int8, ShareCachedWeight) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();
    std::string qtype = "per_channel";

    std::vector<int64_t> weight_shape = {4096, 4096};

    float scale_src = 1 / 255.f; // map to 0~255
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_out = engine->kind() == graph::engine_kind::gpu ? 0 : 78;

    size_t scales_wei_sizes = weight_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(1, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t dqweight_op(2, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t matmul_op(3, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(graph::op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(graph::op_attr::transpose_b, false);

    graph::op_t qout_op(4, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    auto src_u8 = utils::logical_tensor_init(1, graph::data_type::u8);
    auto src_f32_dq = utils::logical_tensor_init(2, graph::data_type::f32);
    auto weight_s8
            = utils::logical_tensor_init(4, weight_shape, graph::data_type::s8);
    weight_s8.property = graph::property_type::constant;
    auto weight_f32_dq = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::f32);
    auto dst_f32 = utils::logical_tensor_init(7, graph::data_type::f32);
    auto dst_s8 = utils::logical_tensor_init(8, graph::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    qout_op.add_input(dst_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&qout_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    ASSERT_EQ(part->get_ops().size(), 4U);

    graph::partition_t p;
    p.init(part);

    std::vector<std::vector<int64_t>> src_shapes {{4, 4096}, {32, 4096}};
    std::vector<std::vector<int64_t>> dst_shapes {{4, 4096}, {32, 4096}};
    std::vector<std::shared_ptr<graph::compiled_partition_t>> cps;
    size_t prv_cache_size = 0;
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        src_u8 = utils::logical_tensor_init(1, src_shape, graph::data_type::u8);
        dst_s8 = utils::logical_tensor_init(8, dst_shape, graph::data_type::s8);
        std::vector<const graph::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8};
        std::vector<const graph::logical_tensor_t *> lt_outs {&dst_s8};

        cps.push_back(std::make_shared<graph::compiled_partition_t>(p));
        auto &cp = *cps.back();

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine),
                graph::status::success);

        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(dst_s8.id, &compiled_output);

        std::vector<uint8_t> src_data(product(src_shape));
        std::vector<int8_t> weight_data(product(weight_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });

        test_tensor src_u8_ts(src_u8, engine, src_data);
        test_tensor weight_s8_ts(weight_s8, engine, weight_data);
        test_tensor dst_s8_ts(compiled_output, engine);
        ASSERT_EQ(cp.execute(strm, {src_u8_ts.get(), weight_s8_ts.get()},
                          {dst_s8_ts.get()}),
                graph::status::success);

        size_t curr_cache_size = graph::get_constant_tensor_cache(
                engine->kind(), engine->index())
                                         ->get_size();

        if (i != 0) {
            // cache size should not change since no new weight cached
            ASSERT_EQ(prv_cache_size, curr_cache_size);
        }
        prv_cache_size = curr_cache_size;

        strm->wait();
    }
}
