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

TEST(test_sdp_decomp_execute, F32SdpDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 1024,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};
    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_dnnl_float_MHA(&g, dnnl::impl::data_type::f32,
                    batch_size, seq_len, num_head, head_dim, transpose_b[i],
                    attention_mask_vec[j]);
            g.finalize();

            if (attention_mask_vec[j])
                ASSERT_EQ(g.get_ops().size(), 7U);
            else
                ASSERT_EQ(g.get_ops().size(), 6U);

            graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();
            if (attention_mask_vec[j])
                ASSERT_EQ(partition_inputs.size(), 5U);
            else
                ASSERT_EQ(partition_inputs.size(), 4U);
            ASSERT_EQ(partition_outputs.size(), 1U);

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                // set output to be strided
                lt = utils::logical_tensor_init(
                        lt.id, lt.data_type, graph::layout_type::strided);
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

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
    }
}

TEST(test_sdp_decomp_execute, Bf16SdpDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 256,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_dnnl_float_MHA(&g, dnnl::impl::data_type::bf16,
                    batch_size, seq_len, num_head, head_dim, transpose_b[i],
                    attention_mask_vec[j]);
            g.finalize();

            if (attention_mask_vec[j])
                ASSERT_EQ(g.get_ops().size(), 7U);
            else
                ASSERT_EQ(g.get_ops().size(), 6U);

            graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();
            if (attention_mask_vec[j])
                ASSERT_EQ(partition_inputs.size(), 5U);
            else
                ASSERT_EQ(partition_inputs.size(), 4U);
            ASSERT_EQ(partition_outputs.size(), 1U);

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                // set output to be strided
                lt = utils::logical_tensor_init(
                        lt.id, lt.data_type, graph::layout_type::strided);
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

            std::vector<test_tensor> inputs_ts, outputs_ts;
            for (auto &lt : inputs) {
                inputs_ts.emplace_back(*lt, eng);
                inputs_ts.back().fill<bfloat16_t>();
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
    }
}

TEST(test_sdp_decomp_execute, Int8SdpDecomp_CPU) {
    // for sdp decompose test
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 1024,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_int8_MHA(&g, batch_size, seq_len, num_head,
                    head_dim, transpose_b[i], attention_mask_vec[j]);
            g.finalize();

            if (attention_mask_vec[j])
                ASSERT_EQ(g.get_ops().size(), 13U);
            else
                ASSERT_EQ(g.get_ops().size(), 12U);
            graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();
            if (attention_mask_vec[j])
                ASSERT_EQ(partition_inputs.size(), 5U);
            else
                ASSERT_EQ(partition_inputs.size(), 4U);
            ASSERT_EQ(partition_outputs.size(), 1U);

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                // set output to be strided
                lt = utils::logical_tensor_init(
                        lt.id, lt.data_type, graph::layout_type::strided);
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

            std::vector<test_tensor> inputs_ts, outputs_ts;
            for (auto &lt : inputs) {
                inputs_ts.emplace_back(*lt, eng);
                inputs_ts.back().fill<uint8_t>();
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
    }
}

TEST(test_sdp_decomp_execute, Int8Bf16SdpDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 1024,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_int8_bf16_MHA(&g, batch_size, seq_len, num_head,
                    head_dim, transpose_b[i], attention_mask_vec[j]);
            g.finalize();

            if (attention_mask_vec[j])
                ASSERT_EQ(g.get_ops().size(), 19U);
            else
                ASSERT_EQ(g.get_ops().size(), 18U);

            graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();
            if (attention_mask_vec[j])
                ASSERT_EQ(partition_inputs.size(), 5U);
            else
                ASSERT_EQ(partition_inputs.size(), 4U);
            ASSERT_EQ(partition_outputs.size(), 1U);

            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

            std::vector<test_tensor> inputs_ts, outputs_ts;
            for (auto &lt : inputs) {
                inputs_ts.emplace_back(*lt, eng);
                inputs_ts.back().fill<bfloat16_t>();
            }

            for (auto &lt : partition_outputs) {
                outputs_ts.emplace_back(lt, eng);
            }

            ASSERT_EQ(cp.execute(strm, test_tensor::to_graph_tensor(inputs_ts),
                              test_tensor::to_graph_tensor(outputs_ts)),
                    graph::status::success);

            strm->wait();
        }
    }
}

// Test multiple thread execute
TEST(test_sdp_decomp_execute, MultithreaSdpDecomp_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 1024,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_int8_bf16_MHA(
                &g, batch_size, seq_len, num_head, head_dim, transpose_b[i]);
        g.finalize();

        ASSERT_EQ(g.get_ops().size(), 19U);

        graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();
        ASSERT_EQ(partition_inputs.size(), 5U);
        ASSERT_EQ(partition_outputs.size(), 1U);

        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[4].layout.strides);

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        for (auto &lt : partition_inputs) {
            inputs.emplace_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            outputs.emplace_back(&lt);
        }

        graph::compiled_partition_t cp(p);
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
        std::vector<test_tensor> inputs_ts;
        for (auto &lt : inputs) {
            inputs_ts.emplace_back(*lt, eng);
            inputs_ts.back().fill<uint8_t>();
        }

        auto func = [&]() {
            std::vector<test_tensor> outputs_ts;
            for (auto &lt : partition_outputs) {
                outputs_ts.emplace_back(lt, eng);
            }
            for (int i = 0; i < 10; i++)
                ASSERT_EQ(cp.execute(strm,
                                  test_tensor::to_graph_tensor(inputs_ts),
                                  test_tensor::to_graph_tensor(outputs_ts)),
                        graph::status::success);
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
}

// Test correctness
TEST(test_sdp_decomp_execute, F32SdpCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 384, num_head = 16, head_dim = 1024,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_dnnl_float_MHA(&g, dnnl::impl::data_type::f32,
                    batch_size, seq_len, num_head, head_dim, transpose_b[i],
                    attention_mask_vec[j]);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                // set output to be strided
                lt = utils::logical_tensor_init(
                        lt.id, lt.data_type, graph::layout_type::strided);
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

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

            // -------------------------case 1----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
            graph::compiled_partition_t cp1(p);
            ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng),
                    graph::status::success);
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

            // -------------------------case 2----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
            graph::compiled_partition_t cp2(p);
            ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng),
                    graph::status::success);
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
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, F32DistilBertSdpCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 128, num_head = 12, head_dim = 768,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_select_float_MHA(&g, dnnl::impl::data_type::f32,
                batch_size, seq_len, num_head, head_dim, transpose_b[i]);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[5].layout.strides);

        for (auto &lt : partition_inputs) {
            inputs.emplace_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            // set output to be strided
            lt = utils::logical_tensor_init(
                    lt.id, lt.data_type, graph::layout_type::strided);
            outputs.emplace_back(&lt);
        }

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

        // -------------------------case 1----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
        graph::compiled_partition_t cp1(p);
        ASSERT_EQ(
                p.compile(&cp1, inputs, outputs, eng), graph::status::success);
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

        // -------------------------case 2----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
        graph::compiled_partition_t cp2(p);
        ASSERT_EQ(
                p.compile(&cp2, inputs, outputs, eng), graph::status::success);
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
}

// Test correctness
TEST(test_sdp_decomp_execute, Bf16SdpCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 2, num_head = 2, head_dim = 64,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_dnnl_float_MHA(&g, dnnl::impl::data_type::bf16,
                    batch_size, seq_len, num_head, head_dim, transpose_b[i],
                    attention_mask_vec[j]);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

            for (auto &lt : partition_inputs) {
                inputs.emplace_back(&lt);
            }
            for (auto &lt : partition_outputs) {
                // set output to be strided
                lt = utils::logical_tensor_init(
                        lt.id, lt.data_type, graph::layout_type::strided);
                outputs.emplace_back(&lt);
            }

            graph::compiled_partition_t cp(p);
            ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                    graph::status::success);

            std::vector<test_tensor> inputs_ts, outputs_ts;
            for (auto &lt : inputs) {
                inputs_ts.emplace_back(*lt, eng);
                inputs_ts.back().fill<bfloat16_t>();
            }

            for (auto &lt : outputs) {
                graph::logical_tensor_t compiled_output;
                cp.query_logical_tensor(lt->id, &compiled_output);
                outputs_ts.emplace_back(compiled_output, eng);
            }

            // -------------------------case 1----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
            graph::compiled_partition_t cp1(p);
            ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng),
                    graph::status::success);
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

            // -------------------------case 2----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
            graph::compiled_partition_t cp2(p);
            ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng),
                    graph::status::success);
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

            ASSERT_TRUE(allclose<bfloat16_t>(outputs1_ts[0], outputs2_ts[0],
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
        }
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, Bf16DistilBertSdpCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 128, num_head = 12, head_dim = 768,
        size_per_head = head_dim / num_head;
    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_select_float_MHA(&g, dnnl::impl::data_type::bf16,
                batch_size, seq_len, num_head, head_dim, transpose_b[i]);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("float_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[5].layout.strides);

        for (auto &lt : partition_inputs) {
            inputs.emplace_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            // set output to be strided
            lt = utils::logical_tensor_init(
                    lt.id, lt.data_type, graph::layout_type::strided);
            outputs.emplace_back(&lt);
        }

        graph::compiled_partition_t cp(p);
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        std::vector<test_tensor> inputs_ts, outputs_ts;
        for (auto &lt : inputs) {
            inputs_ts.emplace_back(*lt, eng);
            inputs_ts.back().fill<bfloat16_t>();
        }

        for (auto &lt : outputs) {
            graph::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            outputs_ts.emplace_back(compiled_output, eng);
        }

        // -------------------------case 1----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
        graph::compiled_partition_t cp1(p);
        ASSERT_EQ(
                p.compile(&cp1, inputs, outputs, eng), graph::status::success);
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

        // -------------------------case 2----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
        graph::compiled_partition_t cp2(p);
        ASSERT_EQ(
                p.compile(&cp2, inputs, outputs, eng), graph::status::success);
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

        ASSERT_TRUE(allclose<bfloat16_t>(outputs1_ts[0], outputs2_ts[0],
                /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, Int8SdpCorr_CPU) {
    // for sdp decompose test
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 2, num_head = 2, head_dim = 64,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_int8_MHA(&g, batch_size, seq_len, num_head,
                    head_dim, transpose_b[i], attention_mask_vec[j]);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

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
                inputs_ts.back().fill<uint8_t>();
            }
            // -------------------------case 1----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
            graph::compiled_partition_t cp1(p);
            ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng),
                    graph::status::success);
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

            // -------------------------case 2----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
            graph::compiled_partition_t cp2(p);
            ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng),
                    graph::status::success);
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

            ASSERT_TRUE(allclose<int8_t>(outputs1_ts[0], outputs2_ts[0],
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
        }
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, Int8Bf16SdpCorr_CPU) {
    // for sdp decompose test
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 2, num_head = 2, head_dim = 64,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        for (size_t j = 0; j < attention_mask_vec.size(); ++j) {
            graph::graph_t g(eng->kind());
            utils::construct_int8_bf16_MHA(&g, batch_size, seq_len, num_head,
                    head_dim, transpose_b[i], attention_mask_vec[j]);
            g.finalize();

            graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1U);
            auto part = g.get_partitions()[0];

            // compile
            graph::partition_t p;
            p.init(part);

            auto partition_inputs = p.get_inputs();
            auto partition_outputs = p.get_outputs();

            std::vector<const graph::logical_tensor_t *> inputs, outputs;
            //mm1 src format tag: acbd, abcd
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[0].layout.strides);
            //mm1 wei format tag: adbc, abcd
            std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                    partition_inputs[1].layout.strides);
            //mm2 wei format tag: acbd, abcd
            size_t mm2_wei_in_offset = attention_mask_vec[j] ? 4 : 3;
            std::copy(QUERY_VALUE_STRIDES[i].begin(),
                    QUERY_VALUE_STRIDES[i].begin() + ndims,
                    partition_inputs[mm2_wei_in_offset].layout.strides);

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
                inputs_ts.back().fill<uint8_t>();
            }
            // -------------------------case 1----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
            graph::compiled_partition_t cp1(p);
            ASSERT_EQ(p.compile(&cp1, inputs, outputs, eng),
                    graph::status::success);
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

            // -------------------------case 2----------------------------------
            custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
            graph::compiled_partition_t cp2(p);
            ASSERT_EQ(p.compile(&cp2, inputs, outputs, eng),
                    graph::status::success);
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

            ASSERT_TRUE(allclose<int8_t>(outputs1_ts[0], outputs2_ts[0],
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
        }
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, Int8DistilBertSdpCorr_CPU) {
    // for sdp decompose test
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 128, num_head = 12, head_dim = 768,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_select_int8_MHA(
                &g, batch_size, seq_len, num_head, head_dim, transpose_b[i]);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("int8_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[5].layout.strides);

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
            inputs_ts.back().fill<uint8_t>();
        }
        // -------------------------case 1----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
        graph::compiled_partition_t cp1(p);
        ASSERT_EQ(
                p.compile(&cp1, inputs, outputs, eng), graph::status::success);
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

        // -------------------------case 2----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
        graph::compiled_partition_t cp2(p);
        ASSERT_EQ(
                p.compile(&cp2, inputs, outputs, eng), graph::status::success);
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

        ASSERT_TRUE(allclose<int8_t>(outputs1_ts[0], outputs2_ts[0],
                /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, Int8Bf16DistilBertSdpCorr_CPU) {
    // for sdp decompose test
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 128, num_head = 12, head_dim = 768,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};
    std::vector<bool> attention_mask_vec = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_int8_bf16_MHA(&g, batch_size, seq_len, num_head,
                head_dim, transpose_b[i], false, true);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[5].layout.strides);

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
            inputs_ts.back().fill<uint8_t>();
        }
        // -------------------------case 1----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
        graph::compiled_partition_t cp1(p);
        ASSERT_EQ(
                p.compile(&cp1, inputs, outputs, eng), graph::status::success);
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

        // -------------------------case 2----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "1", 1);
        graph::compiled_partition_t cp2(p);
        ASSERT_EQ(
                p.compile(&cp2, inputs, outputs, eng), graph::status::success);
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

        ASSERT_TRUE(allclose<int8_t>(outputs1_ts[0], outputs2_ts[0],
                /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

// Test correctness
TEST(test_sdp_decomp_execute, MultithreaSdpDecompCorr_CPU) {
    graph::engine_t *eng = get_engine();
    graph::stream_t *strm = get_stream();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu, "skip on gpu");

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF((isa < dnnl_cpu_isa_avx512_core)
                    && eng->kind() == graph::engine_kind::cpu,
            "Skip bf16 tests for systems that do not support avx512_core.");

    size_t ndims = 4;
    int batch_size = 56, seq_len = 2, num_head = 2, head_dim = 64,
        size_per_head = head_dim / num_head;

    //query,value format tag: acbd, abcd
    std::vector<dims> QUERY_VALUE_STRIDES = {
            {seq_len * head_dim, size_per_head, head_dim, 1},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    //key format tag: adbc, abcd
    std::vector<dims> KEY_STRIDES = {
            {seq_len * head_dim, size_per_head, 1, head_dim},
            {seq_len * head_dim, size_per_head * seq_len, size_per_head, 1}};
    std::vector<bool> transpose_b = {false, true};

    for (size_t i = 0; i < KEY_STRIDES.size(); ++i) {
        graph::graph_t g(eng->kind());
        utils::construct_int8_bf16_MHA(
                &g, batch_size, seq_len, num_head, head_dim, transpose_b[i]);
        g.finalize();

        ASSERT_EQ(g.get_ops().size(), 19U);

        graph::pass::pass_base_ptr apass = get_pass("int8_bf16_sdp_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();
        ASSERT_EQ(partition_inputs.size(), 5U);
        ASSERT_EQ(partition_outputs.size(), 1U);

        //mm1 src format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[0].layout.strides);
        //mm1 wei format tag: adbc, abcd
        std::copy(KEY_STRIDES[i].begin(), KEY_STRIDES[i].begin() + ndims,
                partition_inputs[1].layout.strides);
        //mm2 wei format tag: acbd, abcd
        std::copy(QUERY_VALUE_STRIDES[i].begin(),
                QUERY_VALUE_STRIDES[i].begin() + ndims,
                partition_inputs[4].layout.strides);

        std::vector<const graph::logical_tensor_t *> inputs, outputs;
        for (auto &lt : partition_inputs) {
            inputs.emplace_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            outputs.emplace_back(&lt);
        }

        graph::compiled_partition_t cp(p);
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);
        std::vector<test_tensor> inputs_ts;
        for (auto &lt : inputs) {
            inputs_ts.emplace_back(*lt, eng);
            inputs_ts.back().fill<uint8_t>();
        }

        // -------------------------case 1----------------------------------
        custom_setenv("_ONEDNN_ENABLE_SDP_DECOMP", "0", 1);
        graph::compiled_partition_t cp1(p);
        ASSERT_EQ(
                p.compile(&cp1, inputs, outputs, eng), graph::status::success);
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

        auto func = [&]() {
            std::vector<test_tensor> outputs_ts;
            for (auto &lt : partition_outputs) {
                outputs_ts.emplace_back(lt, eng);
            }
            for (int i = 0; i < 10; i++)
                ASSERT_EQ(cp.execute(strm,
                                  test_tensor::to_graph_tensor(inputs_ts),
                                  test_tensor::to_graph_tensor(outputs_ts)),
                        graph::status::success);
            ASSERT_TRUE(allclose<int8_t>(outputs1_ts[0], outputs_ts[0],
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
        };

        std::thread t1(func);
        std::thread t2(func);
        t1.join();
        t2.join();
    }
}
