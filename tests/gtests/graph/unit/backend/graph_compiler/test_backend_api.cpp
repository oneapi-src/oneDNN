/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include <memory>
#include <gtest/gtest.h>

#include "backend/graph_compiler/compiler_backend.hpp"
#include "backend/graph_compiler/compiler_partition_impl.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#include "interface/partition.hpp"
#include "test_utils.hpp"

namespace impl = dnnl::impl;
namespace utils = dnnl::graph::tests::unit::utils;
namespace compiler_utils = impl::graph::tests::unit::compiler::utils;

TEST(GCBackendApi, GetMemSize_CPU) {
    graph::logical_tensor_t a, b, c, d, e;
    const std::vector<graph::dim_t> a_dim {1, 4, 3};
    const std::vector<graph::dim_t> b_dim {32, 16, 64, 64};
    const std::vector<graph::dim_t> c_dim {32};
    const std::vector<graph::dim_t> d_dim {1, 1};

    a = utils::logical_tensor_init(
            0, a_dim, graph::data_type::u8, graph::layout_type::strided);
    b = utils::logical_tensor_init(
            1, b_dim, graph::data_type::s8, graph::layout_type::strided);
    c = utils::logical_tensor_init(
            2, c_dim, graph::data_type::f32, graph::layout_type::strided);
    d = utils::logical_tensor_init(
            3, d_dim, graph::data_type::s32, graph::layout_type::strided);
    e = utils::logical_tensor_init(
            4, {}, graph::data_type::f32, graph::layout_type::strided);

    auto &compiler_backend_ptr
            = graph::compiler_impl::compiler_backend_t::get_singleton();

    size_t a_mem_res = utils::product(a_dim) * sizeof(signed char);
    size_t b_mem_res = utils::product(b_dim) * sizeof(signed char);
    size_t c_mem_res = utils::product(c_dim) * sizeof(float);
    size_t d_mem_res = utils::product(d_dim) * sizeof(int32_t);
    size_t e_mem_res = sizeof(float);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(a), a_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(b), b_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(c), c_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(d), d_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(e), e_mem_res);
}

TEST(GCBackendApi, CompilerBackendRegistration_CPU) {
    std::vector<const graph::backend_t *> &backends
            = graph::backend_registry_t::get_singleton()
                      .get_registered_backends();
    auto compiler_backend = std::find_if(
            backends.begin(), backends.end(), [](const graph::backend_t *bkd) {
                return bkd->get_name() == "compiler_backend";
            });
    ASSERT_NE(compiler_backend, backends.end());
    EXPECT_FLOAT_EQ((*compiler_backend)->get_priority(), 2.0);
}

TEST(GCBackendApi, TestRewriteOutputLayout_CPU) {
    REQUIRE_AVX512();
    using namespace impl::graph;
    graph_t agraph;
    compiler_utils::add_MHA_infer_shape(&agraph);
    agraph.finalize();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, partition_policy::fusion);
    auto partitions = agraph.get_partitions();

    partition_t p;
    p.init(partitions[0]);
    std::vector<const logical_tensor_t *> inputs;
    std::vector<logical_tensor_t *> outputs;
    for (auto &lt : p.get_inputs()) {
        inputs.push_back(&lt);
    }
    for (auto &lt : p.get_outputs()) {
        outputs.push_back(const_cast<logical_tensor_t *>(&lt));
    }
    // replace output node to be unknown shape + any format
    outputs[0]->layout_type = graph::layout_type::any;
    outputs[0]->ndims = -1;

    // latest update will not overwrite output layout
    p.infer_shape(inputs, outputs);
    EXPECT_EQ(outputs[0]->layout_type, graph::layout_type::any);
}

// Test output tensor inplace
static void build_conv_add_partition(graph::graph_t &agraph,
        const graph::dims &input_shape, const graph::dims &filter_shape,
        const graph::dims &strides, const graph::dims &output_shape) {
    using dims = graph::dims;
    // construct conv-add graph
    utils::id_generator_t id_gen;
    auto dtype = graph::data_type::f32;
    graph::logical_tensor_t input0
            = utils::logical_tensor_init(id_gen.get_id(), input_shape, dtype);
    graph::logical_tensor_t weight0
            = utils::logical_tensor_init(id_gen.get_id(), filter_shape, dtype);
    graph::logical_tensor_t output0
            = utils::logical_tensor_init(id_gen.get_id(), output_shape, dtype);
    graph::op_t conv_fwd(
            id_gen.get_id(), graph::op_kind::Convolution, "conv_fwd0");
    conv_fwd.set_attr<dims>(graph::op_attr::strides, strides);
    conv_fwd.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_fwd.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_fwd.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_fwd.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_fwd.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_fwd.add_input(input0);
    conv_fwd.add_input(weight0);
    conv_fwd.add_output(output0);
    agraph.add_op(&conv_fwd);

    graph::logical_tensor_t input1
            = utils::logical_tensor_init(id_gen.get_id(), output_shape, dtype);
    graph::logical_tensor_t output
            = utils::logical_tensor_init(id_gen.get_id(), output_shape, dtype);
    graph::op_t add {id_gen.get_id(), graph::op_kind::Add, "add"};
    add.add_input(output0);
    add.add_input(input1);
    add.add_output(output);
    agraph.add_op(&add);
    agraph.finalize();

    // add {conv, add} to partitions
    auto conv_add_partition
            = std::make_shared<graph::compiler_impl::compiler_partition_impl_t>(
                    agraph.get_engine_kind(), agraph.get_fpmath_mode(),
                    graph::partition_kind_t::convolution_post_ops, "conv_add");
    std::vector<graph::op_t *> ops;
    dnnl::impl::graph::topo_order_visit(
            agraph.get_output_ops(), [&ops](graph::op_t *op) {
                ops.push_back(op);
                return graph::status::success;
            });
    for (const auto &op : ops) {
        conv_add_partition->add_op(op->shared_from_this());
        op->set_partition(conv_add_partition.get());
        for (const auto &value : op->get_input_values()) {
            if (!value->has_producer()) {
                conv_add_partition->add_input_tensor(value);
            }
        }
        for (const auto &value : op->get_output_values()) {
            if (value->get_consumers().empty()) {
                conv_add_partition->add_output_tensor(value);
            }
        }
    }
    agraph.add_partition(conv_add_partition);
}

TEST(GCBackendApi, ConvAdd_Inplace0_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();

    graph::graph_t agraph(engine->kind());
    const graph::dims input_shape = {128, 3, 227, 227};
    const graph::dims filter_shape = {16, 3, 11, 11};
    const graph::dims strides = {4, 4};
    const graph::dims output_shape = {128, 16, 55, 55};
    build_conv_add_partition(
            agraph, input_shape, filter_shape, strides, output_shape);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    graph::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    std::vector<const graph::logical_tensor_t *> inputs;
    std::vector<const graph::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    // Check in-place pairs
    size_t num_inplace_pairs = 10; // Initialized with an impossible value.
    const dnnl_graph_inplace_pair_t *inplace_pairs = nullptr;
    EXPECT_EQ(dnnl_graph_compiled_partition_get_inplace_ports(
                      &cp, &num_inplace_pairs, &inplace_pairs),
            dnnl_success);
    /*
    Main entry:
    * @param logical_tensor_5 [f32 [128, 16, 55, 55] @ ABCD]
    * @param logical_tensor_0 [f32 [128, 3, 227, 227] @ ABCD]
    * @param logical_tensor_1 [f32 [16, 3, 11, 11] @ ABCD]
    * @param logical_tensor_4 [f32 [128, 16, 55, 55] @ ABCD]
    func conv_add_100004(logical_tensor_5: [f32 * 6195200UL], logical_tensor_0: [f32 * 19787136UL],
            logical_tensor_1: [f32 * 5808UL], logical_tensor_4: [f32 * 6195200UL]): void {
        // [f32 [1, 1, 11, 11, 3, 16] @ ABCD3b16a]
        tensor buffer_3: [f32 * 5808UL]
        evaluate{reorder_5(buffer_3, logical_tensor_1)}
        // [f32 [128, 1, 55, 55, 16] @ ABCD16b]
        tensor buffer_4: [f32 * 6195200UL]
        evaluate{reorder_6(buffer_4, logical_tensor_4)}
        // [f32 [128, 1, 227, 227, 3] @ ABCD3b]
        tensor buffer_5: [f32 * 19787136UL]
        evaluate{reorder_4(buffer_5, logical_tensor_0)}
        // [f32 [128, 1, 55, 55, 16] @ ABCD16b]
        tensor buffer_6: [f32 * 6195200UL]
        evaluate{outerloop_1X128X1X55X1_partition_conv_fwd_core_add_8(buffer_6, buffer_5, buffer_3, buffer_4)}
        evaluate{reorder_7(logical_tensor_5, buffer_6)}
    }
    Inplace: out buf: logical_tensor_5, in buf: logical_tensor_4
    */
    // This feature is disabled temporarily.
    EXPECT_EQ(num_inplace_pairs, 0U);
    /*
    auto pair0 = *(inplace_pairs);
    EXPECT_EQ(pair0.input_id,
            agraph.get_input_values()[2]->get_logical_tensor().id);
    EXPECT_EQ(pair0.output_id,
            agraph.get_output_values()[0]->get_logical_tensor().id);
    */
}

TEST(GCBackendApi, ConvAdd_Inplace1_CPU) {
    REQUIRE_AVX512();
    REQUIRE_CPU_ENGINE();

    graph::graph_t agraph(engine->kind());
    const graph::dims input_shape = {1, 1, 4, 4};
    const graph::dims filter_shape = {1, 1, 1, 1};
    const graph::dims strides = {1, 1};
    const graph::dims output_shape = {1, 1, 4, 4};
    build_conv_add_partition(
            agraph, input_shape, filter_shape, strides, output_shape);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U);

    graph::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    std::vector<const graph::logical_tensor_t *> inputs;
    std::vector<const graph::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    /*
    Main entry:
    * @param logical_tensor_5 [f32 [1, 1, 4, 4] @ ABCD]
    * @param logical_tensor_0 [f32 [1, 1, 4, 4] @ ABCD]
    * @param logical_tensor_1 [f32 [1, 1, 1, 1] @ ABCD]
    * @param logical_tensor_4 [f32 [1, 1, 4, 4] @ ABCD]
    func conv_add_100004(logical_tensor_5: [f32 * 16UL], logical_tensor_0: [f32 * 16UL], 
            logical_tensor_1: [f32 * 1UL], logical_tensor_4: [f32 * 16UL]): void {
        evaluate{outerloop_1X1X1X4_partition_conv_fwd_core_add_8(&logical_tensor_5[0UL],
                &logical_tensor_0[0UL], &logical_tensor_1[0UL], &logical_tensor_4[0UL])}
    }
    Inplace: out buf: logical_tensor_5, in buf: logical_tensor_4
    */
    // This feature is disabled temporarily.
    // check inplace pairs
    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 0U);
    /*
    ASSERT_EQ(inplace_pairs[0].input_id,
            agraph.get_input_values()[2]->get_logical_tensor().id);
    ASSERT_EQ(inplace_pairs[0].output_id,
            agraph.get_output_values()[0]->get_logical_tensor().id);
    */
}
