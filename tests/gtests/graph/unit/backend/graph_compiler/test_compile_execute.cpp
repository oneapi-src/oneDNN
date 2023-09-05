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
#include "interface/allocator.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>
#include <runtime/context.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
struct gc_env_initializer {
    gc_env_initializer() {
        dnnl::impl::graph::gc::runtime::get_default_stream = []() {
            static auto the_stream = []() {
                dnnl::impl::graph::gc::runtime::stream_t ret
                        = dnnl::impl::graph::gc::runtime::default_stream;
                ret.vtable_.stream = ::get_stream();
                return ret;
            }();
            return &the_stream;
        };
    }
};
static gc_env_initializer gc_test_init;
#endif

namespace impl = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::impl::graph::tests::unit::compiler::utils;

using ltsr_vec = std::vector<impl::logical_tensor_t>;
static void set_mlp_dynamic_parti_ltsrs(int64_t real_batch_size,
        ltsr_vec &parti_inputs, ltsr_vec &parti_outputs) {
    parti_inputs[0].dims[0] = real_batch_size;
    parti_outputs[0].dims[0] = real_batch_size;
}

static void compile_execution_pipeline(impl::graph_t &agraph,
        int expected_part_size,
        std::function<void(ltsr_vec &, ltsr_vec &)> dynamic_callback
        = nullptr) {
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

        std::vector<const impl::logical_tensor_t *> inputs;
        std::vector<const impl::logical_tensor_t *> outputs;
        for (auto &lt : partition_inputs) {
            inputs.push_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            outputs.push_back(&lt);
        }
        impl::compiled_partition_t cp(p);
        impl::engine_t &eng = *get_engine();
        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        std::vector<impl::tensor_t> execution_inputs;
        std::vector<impl::tensor_t> execution_outputs;
        partition_outputs.clear();
        for (auto &lt : outputs) {
            impl::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            partition_outputs.push_back(compiled_output);
            assert(compiled_output.ndims > -1);
        }
        if (dynamic_callback) {
            dynamic_callback(partition_inputs, partition_outputs);
        }
        size_t size = 0;
        for (auto &lt : partition_inputs) {
            size += compiler_backend_ptr.get_mem_size(lt);
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
        }
        for (auto &lt : partition_outputs) {
            size += compiler_backend_ptr.get_mem_size(lt);
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
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

        impl::stream_t &strm = *get_stream();
        ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
                impl::status::success);
        strm.wait();
    }
}

// test fp32 get partition + compile + execution of MHA graph
TEST(GCGraphTest, FP32MHACompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHACompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHACompileExecution3_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, FP32MHACompileExecution4_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, false, false, impl::op_kind::StaticReshape);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(
            &agraph, false, true, false, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(
            &agraph, true, true, false, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionDynamicQuantize2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true,
            impl::op_kind::Reorder, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecutionDynamicQuantize2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true,
            impl::op_kind::Reorder, 128, 384, 16, 1024, true);
    agraph.finalize();
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHACompileExecutionFake_CPU) {
    REQUIRE_AVX512(); // fake int8, so it only requires avx512
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test int8 get partition + compile + execution of MHA graph
TEST(GCGraphTest, INT8MHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHACompileExecution2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test ITEX pattern ends with StaticReshape
TEST(GCGraphTest, INT8BF16MHACompileExecution3_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(
            &agraph, true, true, impl::op_kind::StaticReshape);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MLPCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test bf16 get partition + compile + execution of MHA graph
TEST(GCGraphTest, BF16MHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test infer shape + compile + execute for fp32 MHA
// the created graph is without intermediate shapes
// if infer shape failed, the compilation will also fail
TEST(GCGraphTest, FP32MHAInfershapeCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_infer_shape(&agraph);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

// test compile + multithreading execution for fp32 MHA
TEST(GCGraphTest, FP32MHACompileExecutionMultiThreading_CPU) {
    REQUIRE_AVX512();
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
    GTEST_SKIP();
#endif
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.finalize();

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
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    size_t total_buffer_size = 0;
    for (auto &lt : partition_inputs) {
        total_buffer_size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt.id, &compiled_output);
        total_buffer_size += compiler_backend_ptr.get_mem_size(compiled_output);
    }

    int thread_num = 8;

    auto thread_func = [&](size_t tid) {
        test::vector<char> data(total_buffer_size);

        std::vector<impl::tensor_t> execution_inputs;
        std::vector<impl::tensor_t> execution_outputs;

        size_t size = 0;
        for (auto &lt : partition_inputs) {
            impl::tensor_t placeholder(lt, engine, data.data() + size);
            execution_inputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(lt);
        }
        for (auto &lt : partition_outputs) {
            graph::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt.id, &compiled_output);
            graph::tensor_t placeholder(
                    compiled_output, engine, data.data() + size);
            execution_outputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(compiled_output);
        }
        impl::stream_t &strm = *get_stream();
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

TEST(GCGraphTest, FP32MHAAlternativeCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32MHAAlternativeCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16MHAAlternativeCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16MHAAlternativeCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MHAAlternative4CompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MHAAlternative4CompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32BartMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16BartMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BartMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16BartMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32DistillBertMHA_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16DistillBertMHA_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32DistillBertMHA_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16DistillBertMHA_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16MLPCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, true, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, INT8MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, BF16MLPDynamicGraphCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, true, -1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1,
            std::bind(set_mlp_dynamic_parti_ltsrs, static_cast<int64_t>(1),
                    std::placeholders::_1, std::placeholders::_2));
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MLPTrainingGraphCompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward},
            false, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MLPTrainingGraphCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 128, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            {impl::op_kind::SigmoidBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward, impl::op_kind::ReLUBackward,
                    impl::op_kind::ReLUBackward},
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32BartMLPResidualCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_mlp_residual_subgraph(
            &agraph, false, false, 1, 17);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16BartMLPResidualCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_mlp_residual_subgraph(&agraph, true, false, 1, 17);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

#if 0
TEST(GCGraphTest, INT8BartMLPResidualCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_mlp_residual_subgraph(&agraph, false, true, 1, 17);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16BartMLPResidualCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_bart_mlp_residual_subgraph(&agraph, true, true, 1, 17);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}
#endif

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32MHATrainingGraphCompileExecution2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16MHATrainingGraphCompileExecution2_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecutionNXC_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NXC", "XIO");
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8ConvolutionalBottleneckCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32IdenticalBottleneckTrainingCompileExecutionNCX_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}}, false, false,
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NCX", "OIX");
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, FP32ConvolutionalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16IdenticalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {64, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}}, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, BF16ConvolutionalBottleneckTrainingCompileExecution_CPU) {
    REQUIRE_AMXBF16();
#if SC_BUILTIN_JIT_ENABLED
    if (::dnnl::impl::graph::gc::get_default_context()->flags_.jit_kind_
            == ::dnnl::impl::graph::gc::jit_kind::xbyak) {
        GTEST_SKIP();
        return;
    }
#endif
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {64, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 2);
}

TEST(GCGraphTest, INT8IdenticalBottleneckCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 256, 56, 56},
            {{64, 256, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NCX", "OIX",
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest,
        INT8IdenticalBottleneckCompileExecutionDynamicQuantizeNXC_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_identical_bottleneck_resblock(&agraph,
            id_gen, {1, 56, 56, 256},
            {{1, 1, 256, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}},
            {{1, 1}, {1, 1}, {1, 1}}, {{0, 0}, {1, 1}, {0, 0}}, "NXC", "XIO",
            true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest,
        INT8ConvolutionalBottleneckCompileExecutionDynamicQuantize_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            {{2, 2}, {1, 1}, {2, 2}, {1, 1}}, {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
            "NCX", "OIX", true);

    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8MulQuantizeCompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_mul_quantize_subgraph(
            &agraph, id_gen, {4, 1, 4096});
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32GPTMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16GPTMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32GPTMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16GPTMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32LLAMAMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16LLAMAMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32LLAMAMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16LLAMAMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32GPTMLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16GPTMLPCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32GPTMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16GPTMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32LLAMAMLPCompileExecution_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16LLAMAMLPCompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32LLAMAMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16LLAMAMLPCompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, GptjBf16Concat_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph);
    agraph.finalize();

    // should hit add_to_concat_permute_concat_to pattern
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, GptjInt8Bf16Concat_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph, true);
    agraph.finalize();

    // should hit mul_mul_add_concat_permute_concat_quant pattern
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, LlamaInt8Bf16Concat_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::add_llama_concat_subgraph(&agraph, true);
    agraph.finalize();

    // should hit add_typecast_concat_typecasts_quant pattern
    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, FP32STARCODERMHACompileExecution_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, BF16STARCODERMHACompileExecution_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, false);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8FP32STARCODERMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}

TEST(GCGraphTest, INT8BF16STARCODERMHACompileExecution_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    impl::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, true);
    agraph.finalize();

    compile_execution_pipeline(agraph, 1);
}
