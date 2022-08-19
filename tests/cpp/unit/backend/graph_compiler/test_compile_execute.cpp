/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

static void compile_execution_pipeline(
        impl::graph_t &agraph, int expected_part_size) {
    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), expected_part_size);

    // TODO(yifei): generalize the logic here
    // sort partitions to run forward first according to num ops
    std::sort(partitions.begin(), partitions.end(),
            [](std::shared_ptr<impl::partition_impl_t> a,
                    std::shared_ptr<impl::partition_impl_t> b) {
                return a->get_ops().size() < b->get_ops().size();
            });

    std::unordered_map<int, impl::logical_tensor_t> lt_info_map;

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

        std::vector<impl::tensor_t> execution_inputs;
        std::vector<impl::tensor_t> execution_outputs;
        partition_outputs.clear();
        size_t size = 0;
        for (auto &lt : partition_inputs) {
            size += compiler_backend_ptr.get_mem_size(lt);
            assert(lt.ndims > 0);
            lt_info_map[lt.id] = lt;
        }
        for (auto &lt : outputs) {
            impl::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            size += compiler_backend_ptr.get_mem_size(compiled_output);
            partition_outputs.push_back(compiled_output);
            assert(compiled_output.ndims > 0);
            lt_info_map[compiled_output.id] = compiled_output;
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
    ASSERT_EQ(partitions.size(), 1);

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
    for (size_t t_num = 0; t_num < thread_num; t_num++) {
        workers.emplace_back(thread_func, t_num);
    }

    for (size_t t_num = 0; t_num < thread_num; t_num++) {
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
    ASSERT_EQ(partitions.size(), 1);

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

TEST(GCGraphTest, BF16MHAAlternativeCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.build_graph();

    compile_execution_pipeline(agraph, 1);
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

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop});
    agraph.build_graph();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MLPTrainingGraphCompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 1, 5,
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
