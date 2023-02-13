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
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "test_utils.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

namespace impl = dnnl::graph::impl;
namespace compiler_impl = dnnl::graph::impl::compiler_impl;
namespace pass = dnnl::graph::impl::pass;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = dnnl::graph::tests::unit::compiler::utils;

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

// test int8 MHA pattern (optimized graph)
TEST(GCPatternTests, INT8MHAPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 20U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8MHAPattern2) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 19U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test fp32 MHA pattern
TEST(GCPatternTests, FP32MHAPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 14U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32MHAPattern2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, false, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 13U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test fp32 MHA pattern alternative
TEST(GCPatternTests, FP32MHAPatternAlternative) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 7U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test fp32 distill_bert MHA pattern
TEST(GCPatternTests, FP32DistillBertMHAPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, false, false);
    agraph.build_graph();
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_distill_bert_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1UL);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 6U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test bf16 distill_bert MHA pattern
TEST(GCPatternTests, BF16DistillBertMHAPattern) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, false);
    agraph.build_graph();
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "bf16_distill_bert_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 6U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test int8-bf16 distill_Bert MHA pattern
TEST(GCPatternTests, INT8BF16DistillBertMHAPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, true);
    agraph.build_graph();
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass = get_pass(
            compiler_backend_ptr, "int8_bf16_distill_bert_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 18U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8BF16DistillBertMHAPattern2) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, true, true);
    agraph.build_graph();
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass = get_pass(
            compiler_backend_ptr, "int8_bf16_distill_bert_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 18U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// create a case where select op's input does not meet
// the constriant specified in the pattern
TEST(GCPatternTests, INT8BF16DistillBertMHAPatternFailureCase) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_distill_bert_MHA(&agraph, true, true, true);
    for (auto &op : agraph.get_ops()) {
        if (op->get_kind() == impl::op_kind::Select) {
            auto if_value = op->get_input_value(1);
            auto else_value = op->get_input_value(2);
            // set if to be the same shape as else
            auto else_lt = else_value->get_logical_tensor();
            else_lt.id = if_value->get_logical_tensor().id;
            if_value->set_logical_tensor(else_lt);
        }
    }
    agraph.build_graph();
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass = get_pass(
            compiler_backend_ptr, "int8_bf16_distill_bert_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0U);
}

// test fp32 MHA pattern (no reshape)
TEST(GCPatternTests, FP32MHAPatternOptionalReshape) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    utils::construct_f32_MHA(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 13U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8BF16MHAPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, true, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_bf16_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 25U);
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test bf16 MHA pattern
TEST(GCPatternTests, BF16MHAPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph(&agraph, true, false);
    agraph.build_graph();

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

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test bf16 MHA pattern alternative
TEST(GCPatternTests, BF16MHAPatternAlternative) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.build_graph();

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

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

// test MHA pattern matcher v2 on graph variations
TEST(GCPatternTests, INT8MHAPatternVariation1) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20U);
}

TEST(GCPatternTests, INT8MHAPatternVariation2) {
    // replace divide with multiply
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20U);
}

TEST(GCPatternTests, INT8MHAPatternVariation3) {
    // set rescale output as Add's second input
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::get_int8_MHA_subgraph_varients(&agraph, true,
            std::vector<compiler_utils::quantize_position_t>(
                    4, compiler_utils::RESHAPE_INCLUDED),
            1);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20U);
}

TEST(GCPatternTests, FP32DLRMBottom) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 6U);

    ASSERT_EQ(partition_inputs.size(), 7U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32DLRMTop) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 10U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8DLRMBottom) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mlp_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 15U);

    ASSERT_EQ(partition_inputs.size(), 7U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8DLRMTop) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mlp_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 25U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32MLPSeparateAdd) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid},
            true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 15U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32MLPNoActivation) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 5U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32MLPSeparateAddNoActivation) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_subgraph(&agraph, false, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard},
            true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 10U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8MLPNoActivation) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_int8_mlp_subgraph(&agraph, 1, 5,
            {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard, impl::op_kind::Wildcard,
                    impl::op_kind::Wildcard});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mlp_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 20U);

    ASSERT_EQ(partition_inputs.size(), 11U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, FP32MLPTraining) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_mlp_training_graph(&agraph, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU},
            {impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass_fwd
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");
    pass::pass_base_ptr apass_bwd
            = get_pass(compiler_backend_ptr, "fp32_mlp_backward_pattern_v2");

    apass_fwd->run(agraph);
    apass_bwd->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 6U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 13U);
}

TEST(GCPatternTests, FP32MHATrainingPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    pass::pass_base_ptr fwd_apass
            = get_pass(compiler_backend_ptr, "fp32_mha_forward_pattern");
    pass::pass_base_ptr bwd_apass
            = get_pass(compiler_backend_ptr, "fp32_mha_backward_pattern");
    fwd_apass->run(agraph);
    bwd_apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 8U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 6U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 11U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 8U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3U);
}

TEST(GCPatternTests, FP32MHATrainingPattern2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    pass::pass_base_ptr fwd_apass
            = get_pass(compiler_backend_ptr, "fp32_mha_forward_pattern");
    pass::pass_base_ptr bwd_apass
            = get_pass(compiler_backend_ptr, "fp32_mha_backward_pattern");
    fwd_apass->run(agraph);
    bwd_apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 9U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 7U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 12U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 9U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3U);
}

TEST(GCPatternTests, BF16MHATrainingPattern) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr fwd_apass
            = get_pass(compiler_backend_ptr, "bf16_mha_forward_pattern");
    pass::pass_base_ptr bwd_apass
            = get_pass(compiler_backend_ptr, "bf16_mha_backward_pattern");
    fwd_apass->run(agraph);
    bwd_apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 8U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 6U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 11U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 8U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3U);
}

TEST(GCPatternTests, BF16MHATrainingPattern2) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_training_subgraph(&agraph, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr fwd_apass
            = get_pass(compiler_backend_ptr, "bf16_mha_forward_pattern");
    pass::pass_base_ptr bwd_apass
            = get_pass(compiler_backend_ptr, "bf16_mha_backward_pattern");
    fwd_apass->run(agraph);
    bwd_apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 9U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 7U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 12U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 9U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3U);
}

TEST(GCPatternTests, FP32IdenticalBottleneckPattern1) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56}, {{64, 64, 1, 1}, {64, 64, 1, 1}}, false,
            {{1, 1}, {1, 1}}, {{0, 0}, {0, 0}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "f32_identical_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 5U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 5U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, FP32IdenticalBottleneckPattern2) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}},
            false, {{1, 1}, {1, 1}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "f32_identical_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 9U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 9U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckPattern1) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56}, {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}},
            false, {{2, 2}, {1, 1}, {2, 2}}, {{0, 0}, {0, 0}, {0, 0}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "f32_convolutional_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 6U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 7U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckPattern2) {
    REQUIRE_AVX512();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1},
                    {64, 64, 1, 1}},
            false, {{2, 2}, {1, 1}, {2, 2}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "f32_convolutional_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 10U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 11U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, BF16IdenticalBottleneckPattern) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_resblock(&agraph, id_gen,
            {1, 256, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}}, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "bf16_identical_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 7U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 7U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, INT8ConvolutionalBottleneckPattern) {
    REQUIRE_VNNI_AMXINT8();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_int8_convolutional_bottleneck_resblock(&agraph,
            id_gen, {1, 64, 56, 56},
            {{64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 1, 1},
                    {64, 64, 1, 1}},
            {{2, 2}, {1, 1}, {2, 2}, {1, 1}, {1, 1}},
            {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_convolutional_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 26U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 11U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, BF16ConvolutionalBottleneckPattern) {
    REQUIRE_BF16_AMXBF16();
    REQUIRE_SINGLE_THREAD();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_resblock(&agraph, id_gen,
            {1, 64, 56, 56},
            {{256, 64, 1, 1}, {64, 64, 1, 1}, {64, 64, 3, 3}, {256, 64, 1, 1}},
            true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "bf16_convolutional_bottleneck");
    apass->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 8U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 9U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 1U);
}

TEST(GCPatternTests, FP32ConvolutionalBottleneckTrainingPattern) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_convolutional_bottleneck_training_subgraph(
            &agraph, id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 256}, {1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 256}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass_fwd = get_pass(
            compiler_backend_ptr, "f32_convolutional_bottleneck_forward");
    pass::pass_base_ptr apass_bwd = get_pass(
            compiler_backend_ptr, "f32_convolutional_bottleneck_backward_v1");

    apass_fwd->run(agraph);
    apass_bwd->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 12U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 21U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 23U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 16U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 25U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 13U);
}

TEST(GCPatternTests, FP32IdenticalBottleneckTrainingPattern) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 64}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass_fwd = get_pass(
            compiler_backend_ptr, "f32_identical_bottleneck_forward");
    pass::pass_base_ptr apass_bwd = get_pass(
            compiler_backend_ptr, "f32_identical_bottleneck_backward_v1");

    apass_fwd->run(agraph);
    apass_bwd->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2U);
    ASSERT_EQ(partitions[0]->get_ops().size(), 10U);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 16U);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 18U);
    ASSERT_EQ(partitions[1]->get_ops().size(), 13U);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 20U);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 10U);
}

TEST(GCPatternTests, StaticOnlyPartitions) {
    REQUIRE_AVX512();
    utils::id_generator id_gen;
    impl::graph_t agraph;
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {1, 56, 56, 64},
            {{1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 64}});
    compiler_utils::construct_identical_bottleneck_training_subgraph(&agraph,
            id_gen, {-2, 56, 56, 64},
            {{1, 1, 64, 64}, {3, 3, 64, 64}, {1, 1, 64, 64}});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass = get_pass(
            compiler_backend_ptr, "f32_identical_bottleneck_forward");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);
}

TEST(GCPatternTests, DynamicOnlyPartitions) {
    REQUIRE_AVX512();
    {
        impl::graph_t agraph;
        compiler_utils::add_mlp_subgraph(
                &agraph, false, -2, 1, {13, 512}, {impl::op_kind::ReLU});
        agraph.build_graph();

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "fp32_mlp_single_layer");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);
        ASSERT_EQ(partitions[0]->get_ops().size(), 2U);
    }
    {
        impl::graph_t agraph;
        compiler_utils::add_mlp_subgraph(
                &agraph, false, 1, 1, {13, 512}, {impl::op_kind::ReLU});
        agraph.build_graph();

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "fp32_mlp_single_layer");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 0U);
    }
    REQUIRE_VNNI_AMXINT8();
    {
        impl::graph_t agraph;
        compiler_utils::add_int8_mlp_subgraph(
                &agraph, -2, 1, {13, 512}, {impl::op_kind::ReLU});
        agraph.build_graph();

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "int8_mlp_single_layer");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);
        ASSERT_EQ(partitions[0]->get_ops().size(), 5U);
    }
    {
        impl::graph_t agraph;
        compiler_utils::add_int8_mlp_subgraph(
                &agraph, 1, 1, {13, 512}, {impl::op_kind::ReLU});
        agraph.build_graph();

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "int8_mlp_single_layer");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 0U);
    }
    {
        impl::graph_t agraph;
        compiler_utils::add_int8_mlp_subgraph(
                &agraph, -2, 1, {13, 512}, {impl::op_kind::ReLU}, true);
        agraph.build_graph();

        auto &compiler_backend_ptr
                = compiler_impl::compiler_backend_t::get_singleton();
        pass::pass_base_ptr apass
                = get_pass(compiler_backend_ptr, "int8_bf16_mlp_single_layer");

        apass->run(agraph);
        auto partitions = agraph.get_partitions();
        ASSERT_EQ(partitions.size(), 1U);
    }
}

TEST(GCPatternTests, FP32MatMulSoftmaxPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_matmul_softmax_fusion");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 4U);
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, BF16MatMulSoftmaxPattern) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "bf16_matmul_softmax_fusion");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 4U);
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8MatMulSoftmaxPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_matmul_softmax_fusion");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 7U);
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}

TEST(GCPatternTests, INT8BF16MatMulSoftmaxPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    compiler_utils::add_MHA_subgraph_alternative(&agraph, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_bf16_matmul_softmax_fusion");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 10U);
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);
}
