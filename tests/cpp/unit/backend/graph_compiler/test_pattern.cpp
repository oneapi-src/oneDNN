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
    add_MHA_subgraph(&agraph, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 20);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, INT8MHAPattern2) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 19);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test fp32 MHA pattern
TEST(GCPatternTests, FP32MHAPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 14);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32MHAPattern2) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false, false, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 13);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test fp32 MHA pattern alternative
TEST(GCPatternTests, FP32MHAPatternAlternative) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph_alternative(&agraph, false, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 7);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
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
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 13);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, INT8BF16MHAPattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, true, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_bf16_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 25);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test bf16 MHA pattern
TEST(GCPatternTests, BF16MHAPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, true, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    // it shall not match fp32 pass
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0);

    apass = get_pass(compiler_backend_ptr, "bf16_mha_pattern");
    REQUIRE_BF16_AMXBF16();
    apass->run(agraph);
    partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test bf16 MHA pattern alternative
TEST(GCPatternTests, BF16MHAPatternAlternative) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph_alternative(&agraph, true, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    // it shall not match fp32 alternative pass
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0);

    apass = get_pass(compiler_backend_ptr, "bf16_mha_pattern_alternative");
    REQUIRE_BF16_AMXBF16();
    apass->run(agraph);
    partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test MHA pattern matcher v2 on graph variations
TEST(GCPatternTests, INT8MHAPatternVariation1) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}

TEST(GCPatternTests, INT8MHAPatternVariation2) {
    // replace divide with multiply
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}

TEST(GCPatternTests, INT8MHAPatternVariation3) {
    // set rescale output as Add's second input
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph, true,
            std::vector<quantize_position_t>(4, RESHAPE_INCLUDED), 1);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}

TEST(GCPatternTests, FP32DLRMBottom) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_subgraph(&agraph, false, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 6);

    ASSERT_EQ(partition_inputs.size(), 7);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32DLRMTop) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_subgraph(&agraph, false, 1, 5, {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 10);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, INT8DLRMBottom) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_int8_mlp_subgraph(&agraph, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mlp_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 15);

    ASSERT_EQ(partition_inputs.size(), 7);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, INT8DLRMTop) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_int8_mlp_subgraph(&agraph, 1, 5, {479, 1024, 1024, 512, 256, 1},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU,
                    impl::op_kind::ReLU, impl::op_kind::Sigmoid});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mlp_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 25);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32MLPSeparateAdd) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_subgraph(&agraph, false, 1, 5, {479, 1024, 1024, 512, 256, 1},
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
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 15);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32MLPNoActivation) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_subgraph(&agraph, false, 1, 5, {479, 1024, 1024, 512, 256, 1},
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
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 5);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32MLPSeparateAddNoActivation) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_subgraph(&agraph, false, 1, 5, {479, 1024, 1024, 512, 256, 1},
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
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 10);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, INT8MLPNoActivation) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_int8_mlp_subgraph(&agraph, 1, 5, {479, 1024, 1024, 512, 256, 1},
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
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    ASSERT_EQ(p.num_ops(), 20);

    ASSERT_EQ(partition_inputs.size(), 11);
    ASSERT_EQ(partition_outputs.size(), 1);
}

TEST(GCPatternTests, FP32MLPTraining) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_mlp_training_graph(&agraph, 1, 3, {13, 512, 256, 128},
            {impl::op_kind::ReLU, impl::op_kind::ReLU, impl::op_kind::ReLU},
            {impl::op_kind::ReLUBackprop, impl::op_kind::ReLUBackprop,
                    impl::op_kind::ReLUBackprop});
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass_fwd
            = get_pass(compiler_backend_ptr, "fp32_mlp_forward_pattern");
    pass::pass_base_ptr apass_bwd
            = get_pass(compiler_backend_ptr, "fp32_mlp_backward_pattern");

    apass_fwd->run(agraph);
    apass_bwd->run(agraph);

    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions[0]->get_ops().size(), 6);
    ASSERT_EQ(partitions[1]->get_ops().size(), 18);
}

TEST(GCPatternTests, FP32MHATrainingPattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_training_subgraph(&agraph, false);
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
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions[0]->get_ops().size(), 8);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 6);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3);
    ASSERT_EQ(partitions[1]->get_ops().size(), 11);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 8);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3);
}

TEST(GCPatternTests, BF16MHATrainingPattern) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    add_MHA_training_subgraph(&agraph, true);
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
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions[0]->get_ops().size(), 8);
    ASSERT_EQ(partitions[0]->get_inputs().size(), 6);
    ASSERT_EQ(partitions[0]->get_outputs().size(), 3);
    ASSERT_EQ(partitions[1]->get_ops().size(), 11);
    ASSERT_EQ(partitions[1]->get_inputs().size(), 8);
    ASSERT_EQ(partitions[1]->get_outputs().size(), 3);
}
