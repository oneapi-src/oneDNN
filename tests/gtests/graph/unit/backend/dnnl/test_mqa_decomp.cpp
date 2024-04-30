/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using dims = std::vector<dim_t>;

static inline void custom_setenv(
        const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    SetEnvironmentVariable(name, value);
#else
    ::setenv(name, value, overwrite);
#endif
}

TEST(test_mqa_decomp_execute, F32MqaDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::f32, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    // Force enable to avoid some unexpected env settings
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> inputs_ts, outputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

TEST(test_mqa_decomp_execute, Bf16MqaDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::bf16, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    // Force enable to avoid some unexpected env settings
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> inputs_ts, outputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs_ts)),
            graph::status::success);
    strm->wait();
}

// Test multiple thread execute
TEST(test_mqa_decomp_execute, MultithreaMqaDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::f32, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    // Force enable to avoid some unexpected env settings
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> inputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    auto func = [&]() {
        std::vector<test_tensor> outputs_ts;
        for (auto &lt : outputs) {
            graph::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            outputs_ts.emplace_back(compiled_output, eng);
        }
        for (int i = 0; i < 5; i++) {
            ASSERT_EQ(cp.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                              test_tensor::to_graph_tensor(outputs_ts)),
                    graph::status::success);
        }
        strm->wait();
    };

    std::thread t1(func);
    std::thread t2(func);
    std::thread t3(func);
    std::thread t4(func);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

// Test correctness
TEST(test_mqa_decomp_execute, F32MqaCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::f32, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    std::vector<test_tensor> inputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    // ---------------------------case1-----------------------------------------
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
    graph::compiled_partition_t cp1(p);
    ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> outputs1_ts;
    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp1.query_logical_tensor(lt->id, &compiled_output);
        outputs1_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp1.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs1_ts)),
            graph::status::success);
    strm->wait();

    // ---------------------------case2-----------------------------------------
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
    graph::compiled_partition_t cp2(p);
    ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> outputs2_ts;
    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp2.query_logical_tensor(lt->id, &compiled_output);
        outputs2_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp2.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs2_ts)),
            graph::status::success);
    strm->wait();

    ASSERT_TRUE(allclose<float>(outputs1_ts[0], outputs2_ts[0],
            /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}

// Test correctness
TEST(test_mqa_decomp_execute, Bf16MqaCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::bf16, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    std::vector<test_tensor> inputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    // ---------------------------case1-----------------------------------------
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
    graph::compiled_partition_t cp1(p);
    ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> outputs1_ts;
    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp1.query_logical_tensor(lt->id, &compiled_output);
        outputs1_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp1.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs1_ts)),
            graph::status::success);
    strm->wait();

    // ---------------------------case2-----------------------------------------
    custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
    graph::compiled_partition_t cp2(p);
    ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor> outputs2_ts;
    for (auto &lt : outputs) {
        graph::logical_tensor_t compiled_output;
        cp2.query_logical_tensor(lt->id, &compiled_output);
        outputs2_ts.emplace_back(compiled_output, eng);
    }
    ASSERT_EQ(cp2.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                      test_tensor::to_graph_tensor(outputs2_ts)),
            graph::status::success);
    strm->wait();

    ASSERT_TRUE(allclose<float>(outputs1_ts[0], outputs2_ts[0],
            /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}
