/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::graph::tests::unit::compiler::utils;

using ltsr_vec = std::vector<impl::logical_tensor_t>;
static void set_mha_dynamic_parti_ltsrs(std::vector<int64_t> batch_sizes,
        std::vector<int64_t> seq_lengths, int iteration,
        ltsr_vec &parti_inputs) {
    parti_inputs[0].dims[0] = batch_sizes[iteration];
    parti_inputs[0].dims[2] = seq_lengths[iteration];
    parti_inputs[1].dims[0] = batch_sizes[iteration];
    parti_inputs[1].dims[3] = seq_lengths[iteration];
    parti_inputs[3].dims[0] = batch_sizes[iteration];
    parti_inputs[3].dims[3] = seq_lengths[iteration];
    parti_inputs[4].dims[0] = batch_sizes[iteration];
    parti_inputs[4].dims[2] = seq_lengths[iteration];
}

static void set_distill_bert_mha_dynamic_parti_ltsrs(
        std::vector<int64_t> batch_sizes, std::vector<int64_t> seq_lengths,
        int iteration, ltsr_vec &parti_inputs) {
    parti_inputs[0].dims[0] = batch_sizes[iteration];
    parti_inputs[0].dims[2] = seq_lengths[iteration];
    parti_inputs[1].dims[0] = batch_sizes[iteration];
    parti_inputs[1].dims[3] = seq_lengths[iteration];
    parti_inputs[2].dims[0] = batch_sizes[iteration];
    parti_inputs[2].dims[3] = seq_lengths[iteration];
    parti_inputs[4].dims[0] = batch_sizes[iteration];
    parti_inputs[4].dims[2] = seq_lengths[iteration];
}

static void set_mlp_dynamic_parti_ltsrs(std::vector<int64_t> batch_sizes,
        int iteration, ltsr_vec &parti_inputs) {
    parti_inputs[0].dims[0] = batch_sizes[iteration];
}

static void set_bart_mha_dynamic_parti_ltsrs(std::vector<int64_t> seq_lengths,
        int iteration, ltsr_vec &parti_inputs) {
    parti_inputs[0].dims[1] = seq_lengths[iteration];
    parti_inputs[1].dims[2] = seq_lengths[iteration];
    parti_inputs[2].dims[1] = seq_lengths[iteration];
}

static void compile_execution_pipeline(impl::graph_t &agraph,
        int expected_part_size,
        std::function<void(int, ltsr_vec &)> dynamic_callback = nullptr,
        int total_iterations = -1) {
    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), static_cast<size_t>(expected_part_size));
    if (dynamic_callback) { ASSERT_EQ(expected_part_size, 1); }
    // TODO(yifei): generalize the logic here
    // sort partitions to run forward first according to num ops
    std::sort(partitions.begin(), partitions.end(),
            [](std::shared_ptr<impl::partition_impl_t> a,
                    std::shared_ptr<impl::partition_impl_t> b) {
                return a->get_ops().size() < b->get_ops().size();
            });

    std::unordered_map<size_t, impl::logical_tensor_t> lt_info_map;

    for (size_t i = 0; i < partitions.size(); ++i) {
        impl::partition_t p;
        p.init(partitions[i]);
        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        // replace partition inputs info if needed
        for (size_t i = 0; i < partition_inputs.size(); ++i) {
            if (lt_info_map.find(partition_inputs[i].id) != lt_info_map.end()) {
                partition_inputs[i] = lt_info_map[partition_inputs[i].id];
            }
        }

        std::vector<const impl::logical_tensor_t *> compile_inputs;
        std::vector<const impl::logical_tensor_t *> compile_outputs;
        for (auto &lt : partition_inputs) {
            compile_inputs.push_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            compile_outputs.push_back(&lt);
        }
        impl::compiled_partition_t cp(p);
        impl::engine_t &eng = get_engine();
        ASSERT_EQ(p.compile(&cp, compile_inputs, compile_outputs, &eng),
                impl::status::success);
        size_t total_mem_size = 0;
        auto execute_stage = [&](size_t size) {
            std::vector<impl::tensor_t> execution_inputs, execution_outputs;
            test::vector<char> data(size);
            size = 0;
            for (auto &lt : partition_inputs) {
                impl::tensor_t placeholder(lt, &eng, data.data() + size);
                execution_inputs.push_back(placeholder);
                size += compiler_backend_ptr.get_mem_size(lt);
            }
            for (auto &lt : partition_outputs) {
                impl::tensor_t placeholder(lt, &eng, data.data() + size);
                execution_outputs.push_back(placeholder);
                size += compiler_backend_ptr.get_mem_size(lt);
            }
            impl::stream_t &strm = get_stream();
            ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
                    impl::status::success);
            strm.wait();
        };
        if (!dynamic_callback) {
            for (auto &lt : partition_inputs) {
                total_mem_size += compiler_backend_ptr.get_mem_size(lt);
                assert(lt.ndims > -1);
                lt_info_map[lt.id] = lt;
            }
            partition_outputs.clear();
            for (auto &lt : compile_outputs) {
                impl::logical_tensor_t compiled_output;
                cp.query_logical_tensor(lt->id, &compiled_output);
                total_mem_size
                        += compiler_backend_ptr.get_mem_size(compiled_output);
                partition_outputs.push_back(compiled_output);
                assert(compiled_output.ndims > -1);
                lt_info_map[compiled_output.id] = compiled_output;
            }
            execute_stage(total_mem_size);
        } else {
            // set concrete input shape for dynamic case
            for (int shape_idx = 0; shape_idx < total_iterations; ++shape_idx) {
                dynamic_callback(shape_idx, partition_inputs);
                for (auto &lt : partition_inputs) {
                    total_mem_size += compiler_backend_ptr.get_mem_size(lt);
                    assert(lt.ndims > -1);
                    lt_info_map[lt.id] = lt;
                }
                std::vector<const impl::logical_tensor_t *> queried_inputs;
                std::vector<impl::logical_tensor_t *> queried_outputs;
                for (auto &lt : partition_inputs) {
                    queried_inputs.push_back(&lt);
                }
                for (auto &lt : partition_outputs) {
                    queried_outputs.push_back(&lt);
                }
                cp.query_dynamic_outputs(queried_outputs, queried_inputs);
                // caculate mem size for queried partition outputs
                for (auto &lt : partition_outputs) {
                    total_mem_size += compiler_backend_ptr.get_mem_size(lt);
                    lt_info_map[lt.id] = lt;
                }
                execute_stage(total_mem_size);
            }
        }
    }
}

// test fp32 get partition + compile + execution of MHA graph
TEST(GCGraphTest, FP32MHACompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHACompileExecution2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHACompileExecution3) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative2(&agraph);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, FP32MHACompileExecution4) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, false, false, impl::op_kind::StaticReshape);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionFake) {
    REQUIRE_AVX512(); // fake int8, so it only requires avx512
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative2(&agraph, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test int8 get partition + compile + execution of MHA graph
TEST(GCGraphTest, INT8MHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, true, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution2) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, true, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, INT8BF16MHACompileExecution3) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, true, true, impl::op_kind::StaticReshape);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test bf16 get partition + compile + execution of MHA graph
TEST(GCGraphTest, BF16MHACompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test infer shape + compile + execute for fp32 MHA
// the created graph is without intermediate shapes
// if infer shape failed, the compilation will also fail
TEST(GCGraphTest, FP32MHAInfershapeCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_infer_shape(&agraph);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

// test compile + multithreading execution for fp32 MHA
TEST(GCGraphTest, FP32MHACompileExecutionMultiThreading) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    size_t total_buffer_size = 0;
    for (auto &lt : partition_inputs) {
        total_buffer_size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        total_buffer_size += compiler_backend_ptr.get_mem_size(lt);
    }

    int thread_num = 8;

    auto thread_func = [&](size_t tid) {
        test::vector<char> data(total_buffer_size);

        std::vector<impl::tensor_t> execution_inputs;
        std::vector<impl::tensor_t> execution_outputs;

        size_t size = 0;
        for (auto &lt : partition_inputs) {
            impl::tensor_t placeholder(lt, &eng, data.data() + size);
            execution_inputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(lt);
        }
        for (auto &lt : partition_outputs) {
            impl::tensor_t placeholder(lt, &eng, data.data() + size);
            execution_outputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(lt);
        }
        impl::stream_t &strm = get_stream();
        ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
                impl::status::success);
    };

    std::vector<std::thread> workers;
    for (int t_num = 0; t_num < thread_num; t_num++) {
        workers.emplace_back(thread_func, t_num);
    }

    for (int t_num = 0; t_num < thread_num; t_num++) {
        workers[t_num].join();
    }
}

namespace sc {
void release_runtime_memory(runtime::engine_t *engine);
}

// test allocator release before compiled partition destruction
TEST(GCGraphTest, AllocatorEarlyRelease) {
    REQUIRE_AVX512();
    sc::release_runtime_memory(nullptr);
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    impl::compiled_partition_t cp(p);
    impl::allocator_t *allocator = impl::allocator_t::create();
    // create a new engine rather than use test engine here to avoid release the
    // default allocator of the test engine。
    impl::engine_t eng(impl::engine_kind::cpu, 0, allocator);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    allocator->release(); // release the allocator

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;
    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}

TEST(GCGraphTest, FP32MHAAlternativeCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32MHAAlternativeCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHAAlternativeCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32DistillBertMHACompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, false, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16DistillBertMHACompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16DistillBertMHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32BartMHACompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, false, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16BartMHACompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BartMHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16BartMHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32BartMHADynamicCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, false, false, 1, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_bart_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 6, 20}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, BF16BartMHADynamicCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, true, false, 1, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_bart_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 6, 20}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8BartMHADynamicCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, false, true, 1, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_bart_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 6, 20}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8BF16BartMHADynamicCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_bart_MHA(&agraph, true, true, 1, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_bart_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 6, 20}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, FP32DistillBertMHACompileExecution2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, false, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16DistillBertMHACompileExecution2) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16DistillBertMHACompileExecution2) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16MHAAlternativeCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHADynamicGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    // dynamic both batchsize and sequence length
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, false, false, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8BF16MHADynamicGraphCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    // dynamic both batchsize and sequence length
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, true, true, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, BF16MHADynamicGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    // dynamic both batchsize and sequence length
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, true, false, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, FP32DistillBertMHADynamicGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(
            &agraph, false, false, false, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_distill_bert_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, BF16DistillBertMHADynamicGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(
            &agraph, true, false, false, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_distill_bert_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8BF16DistillBertMHADynamicGraphCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(
            &agraph, true, true, false, impl::op_kind::Reorder, -2, -2);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_distill_bert_mha_dynamic_parti_ltsrs,
                    std::vector<int64_t> {64, 96, 128},
                    std::vector<int64_t> {128, 384, 384}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, FP32MLPCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MLPCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16MLPCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, true, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MLPDynamicGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, -2, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 64, 128}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8MLPDynamicGraphCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(&agraph, -2, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 64, 128}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, INT8BF16MLPDynamicGraphCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(
            &agraph, {-2, 384}, 1, {1024, 1024}, {impl::op_kind::ReLU}, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 64, 128}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, BF16MLPDynamicGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, true, -2, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs,
                    std::vector<int64_t> {1, 64, 128}, std::placeholders::_1,
                    std::placeholders::_2),
            3);
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop},
            false, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MLPTrainingGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop},
            true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, false, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution2) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, true, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckCompileExecution) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckCompileExecution) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecutionNXC) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NXC", "XIO");
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8ConvolutionalBottleneckCompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecutionNCX) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}}, false, false,
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NCX", "OIX");
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckTrainingCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16IdenticalBottleneckTrainingCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {64, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}}, true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16ConvolutionalBottleneckTrainingCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {64, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32InstanceNormCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "VALID",
            {0, 0, 0}, {0, 0, 0}, {1, 2, 3});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32InstanceNormReluCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "VALID",
            {0, 0, 0}, {0, 0, 0}, {1, 2, 3}, "ReLU");
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32InstanceNormLeakyReluCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "VALID",
            {0, 0, 0}, {0, 0, 0}, {1, 2, 3}, "LeakyReLU");
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16InstanceNormCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "None",
            {1, 1, 1}, {1, 1, 1}, {1, 2, 3}, "", true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16InstanceNormReluCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "None",
            {1, 1, 1}, {1, 1, 1}, {1, 2, 3}, "ReLU", true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16InstanceNormLeakyReluCompileExecution) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_instance_norm_subgraph(agraph, id_gen,
            {6, 224, 224, 160, 32}, {1, 1, 1}, {1, 1, 1, 32, 4}, {1}, "VALID",
            {0, 0, 0}, {0, 0, 0}, {1, 2, 3}, "LeakyReLU", true);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
}
