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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "backend/graph_compiler/compiler_backend.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "test_utils.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

namespace impl = dnnl::impl;
namespace compiler_impl = dnnl::impl::graph::compiler_impl;
namespace pass = dnnl::impl::graph::pass;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::impl::graph::tests::unit::compiler::utils;

typedef struct {
    unsigned int num_ops;
    unsigned int num_inputs;
    unsigned int num_outputs;
} partition_info_t;

pass::pass_base_ptr get_pass(compiler_impl::compiler_backend_t &backend_ptr,
        const std::string &pass_name) {
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const pass::pass_base_ptr &p) -> bool {
                return p->get_pass_name() == pass_name;
            });
    if (find == passes.end()) { return nullptr; }
    return *find;
}

void test_pattern_matched(graph::graph_t &agraph,
        const std::vector<std::string> &pass_names, unsigned int partition_num,
        const std::vector<partition_info_t> &partition_infos) {
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    for (const auto &pass_name : pass_names) {
        pass::pass_base_ptr apass = get_pass(compiler_backend_ptr, pass_name);
        apass->run(agraph);
    }

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), partition_num);

    for (unsigned int i = 0; i < partition_num; ++i) {
        graph::partition_t p;
        p.init(partitions[i]);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        ASSERT_EQ(p.num_ops(), partition_infos[i].num_ops);
        ASSERT_EQ(partition_inputs.size(), partition_infos[i].num_inputs);
        ASSERT_EQ(partition_outputs.size(), partition_infos[i].num_outputs);
    }
}

// test int8 MHA pattern (optimized graph)
TEST(GCPatternTests, INT8MHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{20, 5, 1}});
}

TEST(GCPatternTests, INT8MHAPattern2_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{19, 5, 1}});
}

// test fp32 MHA pattern
TEST(GCPatternTests, FP32MHAPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern"}, 1,
            std::vector<partition_info_t> {{14, 5, 1}});
}

TEST(GCPatternTests, FP32MHAPattern2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, false, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern"}, 1,
            std::vector<partition_info_t> {{13, 5, 1}});
}

// test fp32 MHA pattern alternative
TEST(GCPatternTests, FP32MHAPatternAlternative_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern_alternative"}, 1,
            std::vector<partition_info_t> {{7, 5, 1}});
}

TEST(GCPatternTests, INT8FP32MHAPatternAlternative_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern_alternative"}, 1,
            std::vector<partition_info_t> {{13, 5, 1}});
}

TEST(GCPatternTests, INT8BF16MHAPatternAlternative_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_mha_pattern_alternative"}, 1,
            std::vector<partition_info_t> {{19, 5, 1}});
}

// test fp32 MHA pattern (no reshape)
TEST(GCPatternTests, FP32MHAPatternOptionalReshape_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    utils::construct_f32_MHA(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern"}, 1,
            std::vector<partition_info_t> {{13, 5, 1}});
}

TEST(GCPatternTests, INT8BF16MHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_mha_pattern"}, 1,
            std::vector<partition_info_t> {{25, 5, 1}});
}

// test bf16 MHA pattern
TEST(GCPatternTests, BF16MHAPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph(&agraph, true, false);
    agraph.finalize();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    // it shall not match fp32 pass
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0U);

    apass = get_pass(compiler_backend_ptr, "bf16_mha_pattern");
    REQUIRE_BF16_AMXBF16();
    apass->run(agraph);
    partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    graph::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test bf16 MHA pattern alternative
TEST(GCPatternTests, BF16MHAPatternAlternative_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.finalize();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    // it shall not match fp32 alternative pass
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0U);

    apass = get_pass(compiler_backend_ptr, "bf16_mha_pattern_alternative");
    REQUIRE_BF16_AMXBF16();
    apass->run(agraph);
    partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    graph::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test MHA pattern matcher v2 on graph variations
TEST(GCPatternTests, INT8MHAPatternVariation1_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{20, 5, 1}});
}

TEST(GCPatternTests, INT8MHAPatternVariation2_CPU) {
    // replace divide with multiply
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{20, 5, 1}});
}

TEST(GCPatternTests, INT8MHAPatternVariation3_CPU) {
    // set rescale output as Add's second input
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph, true,
            std::vector<compiler_utils::quantize_position_t>(
                    4, compiler_utils::RESHAPE_INCLUDED),
            1);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{20, 5, 1}});
}

// test fp32 distill_bert MHA pattern
TEST(GCPatternTests, FP32DistillBertMHAPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_distill_bert_mha_pattern"}, 1,
            std::vector<partition_info_t> {{6, 5, 1}});
}

// test bf16 distill_bert MHA pattern
TEST(GCPatternTests, BF16DistillBertMHAPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_distill_bert_mha_pattern"}, 1,
            std::vector<partition_info_t> {{6, 5, 1}});
}

TEST(GCPatternTests, INT8FP32DistillBertMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_distill_bert_mha_pattern"}, 1,
            std::vector<partition_info_t> {{12, 5, 1}});
}

// test int8-bf16 distill_Bert MHA pattern
TEST(GCPatternTests, INT8BF16DistillBertMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_distill_bert_MHA(&agraph, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_distill_bert_mha_pattern"}, 1,
            std::vector<partition_info_t> {{18, 5, 1}});
}

TEST(GCPatternTests, FP32DLRMBottom_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 3, {13, 512, 256, 128},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU});
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mlp_forward_pattern"}, 1,
            std::vector<partition_info_t> {{6, 7, 1}});
}

TEST(GCPatternTests, FP32DLRMTop_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU,
                    graph::op_kind::ReLU, graph::op_kind::Sigmoid});
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mlp_forward_pattern"}, 1,
            std::vector<partition_info_t> {{10, 11, 1}});
}

TEST(GCPatternTests, INT8DLRMBottom_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 3, {13, 512, 256, 128},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU});
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mlp_pattern"}, 1,
            std::vector<partition_info_t> {{15, 7, 1}});
}

TEST(GCPatternTests, INT8DLRMTop_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU,
                    graph::op_kind::ReLU, graph::op_kind::Sigmoid});
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mlp_pattern"}, 1,
            std::vector<partition_info_t> {{25, 11, 1}});
}

TEST(GCPatternTests, FP32MLPSeparateAdd_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU,
                    graph::op_kind::ReLU, graph::op_kind::Sigmoid},
            true);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mlp_forward_pattern"}, 1,
            std::vector<partition_info_t> {{15, 11, 1}});
}

TEST(GCPatternTests, FP32MLPNoActivation_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard});
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mlp_forward_pattern"}, 1,
            std::vector<partition_info_t> {{5, 11, 1}});
}

TEST(GCPatternTests, FP32MLPSeparateAddNoActivation_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard},
            true);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mlp_forward_pattern"}, 1,
            std::vector<partition_info_t> {{10, 11, 1}});
}

TEST(GCPatternTests, INT8MLPNoActivation_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard, graph::op_kind::Wildcard,
                    graph::op_kind::Wildcard});
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mlp_pattern"}, 1,
            std::vector<partition_info_t> {{20, 11, 1}});
}

TEST(GCPatternTests, FP32MLPTraining_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_mlp_training_graph(&agraph, 1, 3, {13, 512, 256, 128},
            {graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::ReLU},
            {graph::op_kind::ReLUBackward, graph::op_kind::ReLUBackward,
                    graph::op_kind::ReLUBackward});
    agraph.finalize();

    test_pattern_matched(agraph,
            {"fp32_mlp_forward_pattern", "fp32_mlp_backward_pattern_v2"}, 2,
            std::vector<partition_info_t> {{6, 7, 3}, {13, 7, 3}});
}

TEST(GCPatternTests, FP32MHATrainingPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false);
    agraph.finalize();

    test_pattern_matched(agraph,
            {"fp32_mha_forward_pattern", "fp32_mha_backward_pattern"}, 2,
            std::vector<partition_info_t> {{8, 6, 3}, {11, 8, 3}});
}

TEST(GCPatternTests, FP32MHATrainingPattern2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph,
            {"fp32_mha_forward_pattern", "fp32_mha_backward_pattern"}, 2,
            std::vector<partition_info_t> {{9, 7, 3}, {12, 9, 3}});
}

TEST(GCPatternTests, BF16MHATrainingPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true);
    agraph.finalize();

    test_pattern_matched(agraph,
            {"bf16_mha_forward_pattern", "bf16_mha_backward_pattern"}, 2,
            std::vector<partition_info_t> {{8, 6, 3}, {11, 8, 3}});
}

TEST(GCPatternTests, BF16MHATrainingPattern2_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_training_subgraph(&agraph, true, true);
    agraph.finalize();

    test_pattern_matched(agraph,
            {"bf16_mha_forward_pattern", "bf16_mha_backward_pattern"}, 2,
            std::vector<partition_info_t> {{9, 7, 3}, {12, 9, 3}});
}

TEST(GCPatternTests, FP32IdenticalBottleneckPattern1_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56}, {{64, 64, 1, 1}, {64, 64, 1, 1}}, false,
            {{1, 1}, {1, 1}}, {{0, 0}, {0, 0}});
    agraph.finalize();

    test_pattern_matched(agraph, {"f32_identical_bottleneck"}, 1,
            std::vector<partition_info_t> {{5, 5, 1}});
}

TEST(GCPatternTests, FP32IdenticalBottleneckPattern2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}},
            false, {{1, 1}, {1, 1}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.finalize();

    test_pattern_matched(agraph, {"f32_identical_bottleneck"}, 1,
            std::vector<partition_info_t> {{9, 9, 1}});
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckPattern1_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56}, {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}},
            false, {{2, 2}, {1, 1}, {2, 2}}, {{0, 0}, {0, 0}, {0, 0}});
    agraph.finalize();

    test_pattern_matched(agraph, {"f32_convolutional_bottleneck"}, 1,
            std::vector<partition_info_t> {{6, 7, 1}});
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckPattern2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1},
                    {64, 64, 1, 1}},
            false, {{2, 2}, {1, 1}, {2, 2}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.finalize();

    test_pattern_matched(agraph, {"f32_convolutional_bottleneck"}, 1,
            std::vector<partition_info_t> {{10, 11, 1}});
}

TEST(GCPatternTests, BF16IdenticalBottleneckPattern_CPU) {
    REQUIRE_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 256, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}}, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_identical_bottleneck"}, 1,
            std::vector<partition_info_t> {{7, 7, 1}});
}

TEST(GCPatternTests, INT8ConvolutionalBottleneckPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1},
                    {64, 64, 1, 1}},
            {{2, 2}, {1, 1}, {2, 2}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_convolutional_bottleneck"}, 1,
            std::vector<partition_info_t> {{26, 11, 1}});
}

TEST(GCPatternTests, BF16ConvolutionalBottleneckPattern_CPU) {
    REQUIRE_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            true);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_convolutional_bottleneck"}, 1,
            std::vector<partition_info_t> {{8, 9, 1}});
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckTrainingPattern_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.finalize();

    test_pattern_matched(agraph,
            {"f32_convolutional_bottleneck_forward",
                    "f32_convolutional_bottleneck_backward_v1"},
            2, std::vector<partition_info_t> {{12, 21, 23}, {16, 25, 13}});
}

TEST(GCPatternTests, FP32IdenticalBottleneckTrainingPattern_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 64}});
    agraph.finalize();

    test_pattern_matched(agraph,
            {"f32_identical_bottleneck_forward",
                    "f32_identical_bottleneck_backward_v1"},
            2, std::vector<partition_info_t> {{10, 16, 18}, {13, 20, 10}});
}

TEST(GCPatternTests, FP32MatMulSoftmaxPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_matmul_softmax_fusion"}, 1,
            std::vector<partition_info_t> {{4, 4, 1}});
}

TEST(GCPatternTests, BF16MatMulSoftmaxPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_matmul_softmax_fusion"}, 1,
            std::vector<partition_info_t> {{4, 4, 1}});
}

TEST(GCPatternTests, INT8MatMulSoftmaxPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_matmul_softmax_fusion"}, 1,
            std::vector<partition_info_t> {{7, 4, 1}});
}

TEST(GCPatternTests, INT8BF16MatMulSoftmaxPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_matmul_softmax_fusion"}, 1,
            std::vector<partition_info_t> {{10, 4, 1}});
}

TEST(GCPatternTests, INT8MulQuantizePattern_CPU) {
    {
        REQUIRE_AVX512();
        utils::id_generator id_gen;
        REQUIRE_CPU_ENGINE();
        graph::graph_t agraph(engine->kind());
        compiler_utils::construct_mul_quantize_subgraph(
                &agraph, id_gen, {4, 1, 4096});
        agraph.finalize();

        auto ops = agraph.get_ops();

        for (auto &op : ops) {
            if (op->get_kind() == graph::op_kind::Multiply) {
                auto relu = agraph.create_op(graph::op_kind::ReLU, "relu");
                auto in_val = op->get_input_value(0);
                in_val->remove_consumer(*op, 0);
                in_val->add_consumer(*relu, 0);
                relu->add_input(in_val);
                auto relu_out = in_val->get_logical_tensor();
                relu_out.id = id_gen.get_id();
                auto new_val = std::make_shared<graph::value_t>(
                        *relu, 0, relu_out, false);
                relu->add_output(new_val);
                new_val->add_consumer(*op, 0);
                op->connect_input(0, new_val);
                break;
            }
        }

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "mul_typecast_quantize");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 0U);
    }
    {
        REQUIRE_AVX512();
        utils::id_generator id_gen;
        REQUIRE_CPU_ENGINE();
        graph::graph_t agraph(engine->kind());
        compiler_utils::construct_mul_quantize_subgraph(
                &agraph, id_gen, {4, 1, 4096});
        agraph.finalize();

        auto ops = agraph.get_ops();

        for (auto &op : ops) {
            if (op->get_kind() == graph::op_kind::Multiply) {
                auto wildcard = agraph.create_op(
                        graph::op_kind::Wildcard, "wildcard");
                auto in_val = op->get_input_value(0);
                wildcard->add_output(in_val);
                in_val->set_producer(*wildcard);
                break;
            }
        }

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "mul_typecast_quantize");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);

        graph::partition_t p;
        p.init(partitions[0]);

        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();
        ASSERT_EQ(p.num_ops(), 2U);
        ASSERT_EQ(partition_inputs.size(), 2U);
        ASSERT_EQ(partition_outputs.size(), 1U);
    }
}

TEST(GCPatternTests, FP32GPTMHAPattern_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_gpt_mha"}, 1,
            std::vector<partition_info_t> {{8, 7, 1}});
}

TEST(GCPatternTests, BF16GPTMHAPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_gpt_mha"}, 1,
            std::vector<partition_info_t> {{8, 7, 1}});
}

TEST(GCPatternTests, INT8FP32GPTMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_gpt_mha"}, 1,
            std::vector<partition_info_t> {{14, 7, 1}});
}

TEST(GCPatternTests, INT8BF16GPTMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_gpt_mha"}, 1,
            std::vector<partition_info_t> {{20, 7, 1}});
}

TEST(GCPatternTests, FP32LLAMAMHAPattern_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_llama_mha"}, 1,
            std::vector<partition_info_t> {{6, 6, 1}});
}

TEST(GCPatternTests, BF16LLAMAMHAPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_llama_mha"}, 1,
            std::vector<partition_info_t> {{8, 6, 1}});
}

TEST(GCPatternTests, INT8FP32LLAMAMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_llama_mha"}, 1,
            std::vector<partition_info_t> {{11, 6, 1}});
}

TEST(GCPatternTests, INT8BF16LLAMAMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mha_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_llama_mha"}, 1,
            std::vector<partition_info_t> {{16, 6, 1}});
}

TEST(GCPatternTests, FP32GPTMLPPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_gpt_mlp"}, 1,
            std::vector<partition_info_t> {{7, 10, 1}});
}

TEST(GCPatternTests, INT8GPTMLPPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_gpt_mlp"}, 1,
            std::vector<partition_info_t> {{15, 10, 1}});
}

TEST(GCPatternTests, BF16GPTMLPPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_gpt_mlp"}, 1,
            std::vector<partition_info_t> {{7, 10, 1}});
}

TEST(GCPatternTests, INT8BF16GPTMLPPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_gpt_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_gpt_mlp"}, 1,
            std::vector<partition_info_t> {{23, 10, 1}});
}

TEST(GCPatternTests, FP32LLAMAMLPPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_llama_mlp"}, 1,
            std::vector<partition_info_t> {{21, 10, 1}});
}

TEST(GCPatternTests, INT8LLAMAMLPPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_llama_mlp"}, 1,
            std::vector<partition_info_t> {{32, 10, 1}});
}

TEST(GCPatternTests, BF16LLAMAMLPPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_AMX();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_llama_mlp"}, 1,
            std::vector<partition_info_t> {{25, 10, 1}});
}

TEST(GCPatternTests, INT8BF16LLAMAMLPPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_llama_mlp_subgraph(&agraph, id_gen, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_llama_mlp"}, 1,
            std::vector<partition_info_t> {{47, 10, 1}});
}

TEST(GCPatternTests, FP32MHAPatternAlternative2_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern_alternative2"}, 1,
            std::vector<partition_info_t> {{14, 4, 1}});
}

TEST(GCPatternTests, FAKEINT8MHAPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative2(&agraph, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"fake_int8_mha_pattern"}, 1,
            std::vector<partition_info_t> {{15, 5, 1}});
}

TEST(GCPatternTests, FP32BartMHAPattern_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern_alternative3"}, 1,
            std::vector<partition_info_t> {{3, 3, 1}});
}

TEST(GCPatternTests, BF16BartMHAPattern_CPU) {
    REQUIRE_AMXBF16();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_mha_pattern_alternative3"}, 1,
            std::vector<partition_info_t> {{3, 3, 1}});
}

TEST(GCPatternTests, INT8FP32BartMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern_alternative3"}, 1,
            std::vector<partition_info_t> {{9, 3, 1}});
}

TEST(GCPatternTests, INT8BF16BartMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_bart_MHA(&agraph, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_mha_pattern_alternative3"}, 1,
            std::vector<partition_info_t> {{15, 3, 1}});
}

TEST(GCPatternTests, FP32MHAPatternAlternative4_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_mha_pattern_alternative4"}, 1,
            std::vector<partition_info_t> {{7, 5, 1}});
}

TEST(GCPatternTests, INT8MHAPatternAlternative4_CPU) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_MHA_subgraph_alternative4(&agraph, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_mha_pattern_alternative4"}, 1,
            std::vector<partition_info_t> {{13, 5, 1}});
}

TEST(GCPatternTests, add_to_concat_permute_concat_to_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"add_to_concat_permute_concat_to"}, 1,
            std::vector<partition_info_t> {{6, 4, 1}});
}

TEST(GCPatternTests, add_to_concat_permute_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"add_to_concat_permute"}, 1,
            std::vector<partition_info_t> {{4, 3, 1}});
}

TEST(GCPatternTests, permute_concat_to_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph);
    agraph.finalize();

    test_pattern_matched(agraph, {"permute_concat_to"}, 1,
            std::vector<partition_info_t> {{3, 2, 1}});
}

TEST(GCPatternTests, mul_mul_add_concat_permute_concat_quant_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_gptj_concat_subgraph(&agraph, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"mul_mul_add_concat_permute_concat_quant"}, 1,
            std::vector<partition_info_t> {{9, 4, 1}});
}

TEST(GCPatternTests, add_typecast_concat_typecasts_quant_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::add_llama_concat_subgraph(&agraph, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"add_typecast_concat_typecasts_quant"}, 1,
            std::vector<partition_info_t> {{6, 3, 1}});
}

TEST(GCPatternTests, FP32STARCODERMHAPattern_CPU) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"fp32_starcoder_mha"}, 1,
            std::vector<partition_info_t> {{5, 6, 1}});
}

TEST(GCPatternTests, BF16STARCODERMHAPattern_CPU) {
    REQUIRE_BF16_AMXBF16();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, false);
    agraph.finalize();

    test_pattern_matched(agraph, {"bf16_starcoder_mha"}, 1,
            std::vector<partition_info_t> {{5, 6, 1}});
}

TEST(GCPatternTests, INT8FP32STARCODERMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, false, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_starcoder_mha"}, 1,
            std::vector<partition_info_t> {{10, 6, 1}});
}

TEST(GCPatternTests, INT8BF16STARCODERMHAPattern_CPU) {
    REQUIRE_VNNI_AMXINT8();
    utils::id_generator id_gen;
    REQUIRE_CPU_ENGINE();
    graph::graph_t agraph(engine->kind());
    compiler_utils::construct_starcoder_mha_subgraph(
            &agraph, id_gen, true, true);
    agraph.finalize();

    test_pattern_matched(agraph, {"int8_bf16_starcoder_mha"}, 1,
            std::vector<partition_info_t> {{15, 6, 1}});
}
