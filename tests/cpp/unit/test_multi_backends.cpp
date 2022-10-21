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

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "interface/partition.hpp"

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;
namespace pass = dnnl::graph::impl::pass;

#ifdef DNNL_GRAPH_ENABLE_COMPILER_BACKEND
#include "backend/graph_compiler/compiler_backend.hpp"
namespace compiler_impl = dnnl::graph::impl::compiler_impl;
#endif

namespace {
template <typename backend_t>
pass::pass_base_ptr get_pass(
        backend_t &backend_ptr, const std::string &pass_name) {
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const pass::pass_base_ptr &p) -> bool {
                return p->get_pass_name() == pass_name;
            });
    if (find == passes.end()) { return nullptr; }
    return *find;
}
} // namespace

TEST(Execute, MixUseMultipleBackends) {
    /*
        MatMul
          |
         MHA
    */
    using ltw = impl::logical_tensor_wrapper_t;
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::graph_t g(eng.kind());

    const int bs = 1;
    const int seq_len = 384;
    const int num_head = 16;
    const int head_dim = 1024;
    int size_per_head = head_dim / num_head;
    impl::dims RESHAPED_SHAPE = {bs, seq_len, num_head, size_per_head};
    impl::dims TRANSPOSED_SHAPE = {bs, num_head, seq_len, size_per_head};
    impl::dims TRANSPOSED_ORDER = {0, 2, 1, 3};
    utils::construct_dnnl_f32_MHA(&g, bs, seq_len, num_head, head_dim);

    auto mha_in_egde = g.get_input_values()[0]->get_logical_tensor();

    impl::dims matmul_src_shape {bs, seq_len, 2};
    impl::dims matmul_weight_shape {bs, 2, head_dim};
    impl::dims matmul_result_shape {bs, seq_len, head_dim};

    impl::logical_tensor_t matmul_src = utils::logical_tensor_init(100,
            matmul_src_shape, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t matmul_weight
            = utils::logical_tensor_init(101, matmul_weight_shape,
                    impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t matmul_result = utils::logical_tensor_init(
            102, matmul_result_shape, impl::data_type::f32);
    impl::logical_tensor_t reshape_out = utils::logical_tensor_init(
            103, RESHAPED_SHAPE, impl::data_type::f32);

    // create op matmul
    impl::op_t matmul {100, impl::op_kind::MatMul, "matmul"};
    // reshape + transpose for query + key
    impl::op_t reshape_op {101, impl::op_kind::StaticReshape, "query_reshape"};
    reshape_op.set_attr(impl::op_attr::special_zero, false);
    reshape_op.set_attr<std::vector<int64_t>>(
            impl::op_attr::shape, RESHAPED_SHAPE);

    impl::op_t transpose_op {
            102, impl::op_kind::StaticTranspose, "query_transpose"};
    transpose_op.set_attr<std::vector<int64_t>>(
            impl::op_attr::order, TRANSPOSED_ORDER);

    matmul.add_input(matmul_src);
    matmul.add_input(matmul_weight);
    matmul.add_output(matmul_result);

    reshape_op.add_input(matmul_result);
    reshape_op.add_output(reshape_out);

    transpose_op.add_input(reshape_out);
    transpose_op.add_output(mha_in_egde);

    ASSERT_EQ(g.add_op(&matmul), impl::status::success);
    ASSERT_EQ(g.add_op(&reshape_op), impl::status::success);
    ASSERT_EQ(g.add_op(&transpose_op), impl::status::success);
    g.build_graph();
    ASSERT_EQ(g.get_ops().size(), 10U);

    /*----------- partitioning stage ----------------------*/
#ifdef DNNL_GRAPH_ENABLE_COMPILER_BACKEND
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip compiler backend fp32 mha related cases for systems that do "
            "not support avx512_core.");
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr compiler_bkd_pass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");
    compiler_bkd_pass->run(g);
#endif

    auto &dnnl_backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    pass::pass_base_ptr dnnl_bkd_pass
            = get_pass(dnnl_backend_ptr, "f32_MHA_fusion");
    dnnl_bkd_pass->run(g);
    dnnl_bkd_pass = get_pass(
            dnnl_backend_ptr, "matmul_transpose_optional_reshape_fusion");
    dnnl_bkd_pass->run(g);

    size_t num_partitions = g.get_num_partitions();
    ASSERT_EQ(num_partitions, 2U);

    impl::partition_t part0, part1;
    std::vector<impl::partition_t *> partitions {&part0, &part1};
    g.get_ordered_partitions(partitions);

    /*------------ compilation stage: single matmul --------*/
    auto partition_inputs = part0.get_inputs();
    auto partition_outputs = part0.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 2U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set any to allow backend to decide the layout
        lt.layout_type = impl::layout_type::any;
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp0(part0);
    ASSERT_EQ(
            part0.compile(&cp0, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t queried_matmul_output_lt;
    cp0.query_logical_tensor(mha_in_egde.id, &queried_matmul_output_lt);
    ASSERT_EQ(queried_matmul_output_lt.layout_type, impl::layout_type::strided);

    /*------------ execution stage: single matmul ---------*/
    std::vector<test::vector<float>> matmul_inputs_data, matmul_outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        matmul_inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, &eng, matmul_inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        size_t num_elem = utils::product(ltw(lt).vdims());
        if (queried_matmul_output_lt.id == lt.id)
            num_elem = static_cast<size_t>(
                    ltw(queried_matmul_output_lt).size() / sizeof(float));
        matmul_outputs_data.emplace_back(test::vector<float>(num_elem));
        outputs_ts.emplace_back(lt, &eng, matmul_outputs_data.back().data());
    }

    ASSERT_EQ(cp0.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();

    /*------------ compilation stage: MHA --------------*/
    partition_inputs = part1.get_inputs();
    partition_outputs = part1.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    inputs.clear();
    outputs.clear();
    for (auto &lt : partition_inputs) {
        if (lt.id == queried_matmul_output_lt.id)
            inputs.emplace_back(&queried_matmul_output_lt);
        else
            inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp1(part1);
    ASSERT_EQ(
            part1.compile(&cp1, inputs, outputs, &eng), impl::status::success);

    /*------------ execution stage: MHA ----------------*/
    std::vector<test::vector<float>> mha_inputs_data, mha_outputs_data;
    inputs_ts.clear();
    outputs_ts.clear();

    for (auto &lt : partition_inputs) {
        mha_inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, &eng, mha_inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        mha_outputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, &eng, mha_outputs_data.back().data());
    }

    ASSERT_EQ(cp1.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, MixUseMultipleBackendsReverseOrder) {
    /*
         MHA
          |
        MatMul
    */
    using ltw = impl::logical_tensor_wrapper_t;
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::graph_t g(eng.kind());

    const int bs = 1;
    const int seq_len = 384;
    const int num_head = 16;
    const int head_dim = 1024;
    utils::construct_dnnl_f32_MHA(&g, bs, seq_len, num_head, head_dim);

    auto mha_out_egde = g.get_output_values().back()->get_logical_tensor();

    int size_per_head = head_dim / num_head;
    // matmuls rc shape: {bs, seq_len, num_head, size_per_head}
    impl::dims matmul_weight_shape {bs, seq_len, size_per_head, 2};
    impl::dims matmul_dst_shape {bs, seq_len, num_head, 2};

    impl::logical_tensor_t matmul_weight
            = utils::logical_tensor_init(100, matmul_weight_shape,
                    impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t matmul_dst = utils::logical_tensor_init(101,
            matmul_dst_shape, impl::data_type::f32, impl::layout_type::strided);

    // create op matmul
    impl::op_t matmul {100, impl::op_kind::MatMul, "matmul"};
    matmul.add_input(mha_out_egde);
    matmul.add_input(matmul_weight);
    matmul.add_output(matmul_dst);

    ASSERT_EQ(g.add_op(&matmul), impl::status::success);

    g.build_graph();
    ASSERT_EQ(g.get_ops().size(), 8U);

    /*----------- partitioning stage ----------------------*/
#ifdef DNNL_GRAPH_ENABLE_COMPILER_BACKEND
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip compiler backend fp32 mha related cases for systems that do "
            "not support avx512_core.");
    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr compiler_bkd_pass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern_alternative");
    compiler_bkd_pass->run(g);
#endif

    auto &dnnl_backend_ptr = dnnl_impl::dnnl_backend::get_singleton();
    pass::pass_base_ptr dnnl_bkd_pass
            = get_pass(dnnl_backend_ptr, "f32_MHA_fusion");
    dnnl_bkd_pass->run(g);
    dnnl_bkd_pass = get_pass(dnnl_backend_ptr, "matmul_pass");
    dnnl_bkd_pass->run(g);

    size_t num_partitions = g.get_num_partitions();
    ASSERT_EQ(num_partitions, 2U);

    impl::partition_t part0, part1;
    std::vector<impl::partition_t *> partitions {&part0, &part1};
    g.get_ordered_partitions(partitions);

    /*------------ compilation stage: MHA --------------*/
    auto partition_inputs = part0.get_inputs();
    auto partition_outputs = part0.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set any to allow backend to decide the layout
        lt.layout_type = impl::layout_type::any;
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp0(part0);
    ASSERT_EQ(
            part0.compile(&cp0, inputs, outputs, &eng), impl::status::success);

    /*------------ execution stage: MHA ----------------*/
    std::vector<test::vector<float>> mha_inputs_data, mha_outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        mha_inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, &eng, mha_inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        mha_outputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, &eng, mha_outputs_data.back().data());
    }

    ASSERT_EQ(cp0.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();

    /*------------ compilation stage: single matmul --------*/
    impl::logical_tensor_t queried_mha_output_lt;
    cp0.query_logical_tensor(mha_out_egde.id, &queried_mha_output_lt);
    ASSERT_EQ(queried_mha_output_lt.layout_type, impl::layout_type::strided);

    partition_inputs = part1.get_inputs();
    partition_outputs = part1.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 2U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    inputs.clear();
    outputs.clear();
    for (auto &lt : partition_inputs) {
        if (queried_mha_output_lt.id == lt.id)
            inputs.emplace_back(&queried_mha_output_lt);
        else
            inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp1(part1);
    ASSERT_EQ(
            part1.compile(&cp1, inputs, outputs, &eng), impl::status::success);

    /*------------ execution stage: single matmul ---------*/
    std::vector<test::vector<float>> matmul_inputs_data, matmul_outputs_data;
    inputs_ts.clear();
    outputs_ts.clear();

    for (auto &lt : partition_inputs) {
        size_t num_elem = utils::product(ltw(lt).vdims());
        if (queried_mha_output_lt.id == lt.id)
            num_elem = static_cast<size_t>(
                    ltw(queried_mha_output_lt).size() / sizeof(float));
        matmul_inputs_data.emplace_back(test::vector<float>(num_elem));
        inputs_ts.emplace_back(lt, &eng, matmul_inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        matmul_outputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, &eng, matmul_outputs_data.back().data());
    }

    ASSERT_EQ(cp1.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}
