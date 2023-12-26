/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include <iostream>
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/loop_merge.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/matmul_core.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_graph_binding_axis, TestMHAPostOps) {
    sc_graph_t graph;
    SET_THREADS_OR_SKIP(8);
    // build test graph
    auto input0 = graph.make_input(test_utils::make_tsr({8, 16, 16, 16, 32}));
    auto relu0 = graph.make("relu", input0->get_outputs(), {}, {});
    auto reduce0 = graph.make("reduce", relu0->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2, 4}}, {"rd_op", 0}});
    auto relu1 = graph.make("relu", reduce0->get_outputs(), {}, {});
    auto mul0 = graph.make(
            "mul", {relu0->get_outputs()[0], relu1->get_outputs()[0]}, {}, {});
    graph.make_output(mul0->get_outputs());
    // get test context
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    graph_driver_before_fusion(graph, ctx);
    // diable partition optimization
    ctx->flags_.opt_level_ = sc_opt_level::lv0;
    mixed_partition(graph, ctx);
    auto mixed_op = get_mixed_op_from_graph(graph);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    // get func of parti
    func_c f = mixed_op->parti_list_[0]->func_;
    ir_simplifier_t simp {false};
    loop_merger_t lm;
    // simplify TIR
    f = simp(f);
    // do loop merge
    f = lm(f);
    // get lanes
    int lanes = ctx->get_max_vector_lanes(sc_data_etype::S32);
    builder::ir_builder_t builder;
    _function_(datatypes::boolean, expected,
            _arg_("out0", datatypes::s32, {8UL, 16UL, 16UL, 16UL, 32UL}),
            _arg_("inp0", datatypes::s32, {8UL, 16UL, 16UL, 16UL, 32UL})) {
        _bind_(out0, inp0);
        _for_(bs, 0, 8UL) {
            _for_(m_o, 0, 16UL) {
                _tensor_(_relu_out_1, datatypes::s32,
                        {8UL, 16UL, 1UL, 16UL, 1UL});
                _tensor_(_relu_out_0, datatypes::s32,
                        {8UL, 16UL, 16UL, 16UL, 32UL});
                _tensor_(_reduce_out_0, datatypes::s32,
                        {8UL, 16UL, 1UL, 16UL, 1UL});
                // fuse relu
                _for_(n_o, 0, 16UL) {
                    _for_(m_i, 0, 16UL) {
                        _for_(n_i, 0, 32UL, lanes) {
                            builder::tensor_ptr(_relu_out_0,
                                    {bs, m_o, n_o, m_i, 0}, {},
                                    true)[span_t({0, 0, 0, 0, n_i}, lanes)]
                                    = builder::make_max(
                                            make_expr<constant_node>((int64_t)0,
                                                    sc_data_type_t::s32(lanes)),
                                            builder::tensor_ptr(inp0,
                                                    {bs, m_o, n_o, m_i, 0}, {},
                                                    true)[span_t(
                                                    {0, 0, 0, 0, n_i}, lanes)]);
                        }
                    }
                }
                // fuse reduce
                _for_(m_i, 0, 16UL) {
                    _var_(reduce_sum, sc_data_type_t::s32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            int64_t(0), sc_data_type_t::s32(lanes));
                    _for_(n_o, 0, 16UL) {
                        _for_(n_i, 0, 32UL, lanes) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(_relu_out_0,
                                            {bs, m_o, 0, 0, 0}, {},
                                            true)[span_t(
                                            {0, 0, n_o, m_i, n_i}, lanes)],
                                    reduce_sum);
                        }
                    }
                    builder::tensor_ptr(_reduce_out_0, {bs, m_o, 0, 0, 0}, {},
                            true)[{0, 0, 0, m_i, 0}]
                            = builder::make_reduce_add(reduce_sum);
                    /** NOTE: Even though m_o loop has been transformed several
                     * times for both `reduce` and `relu` op, they can be merged
                     * safely, ensured by binding axis system. **/
                    // fuse relu, and merge two loops: `m_o` and `m_o`
                    builder::tensor_ptr(_relu_out_1, {bs, m_o, 0, 0, 0}, {},
                            true)[{0, 0, 0, m_i, 0}]
                            = builder::make_max(
                                    make_expr<constant_node>(
                                            (int64_t)0, sc_data_type_t::s32()),
                                    builder::tensor_ptr(_reduce_out_0,
                                            {bs, m_o, 0, 0, 0}, {},
                                            true)[{0, 0, 0, m_i, 0}]);
                }
                /** NOTE: `m_o` and `n_o` loop come from different binding axis
                 * respectively, as the result, they could never be merged
                 * although two loops have same range `16` and become nearby
                 * after ir simplify pass. **/
                // fuse broadcast mul but could not merge `m_o` and `n_o` loops
                _for_(n_o, 0, 16UL) {
                    _for_(m_i, 0, 16UL) {
                        _for_(n_i, 0, 32UL, lanes) {
                            builder::tensor_ptr(out0, {bs, m_o, n_o, m_i, 0},
                                    {}, true)[span_t({0, 0, 0, 0, n_i}, lanes)]
                                    = builder::tensor_ptr(_relu_out_0,
                                              {bs, m_o, n_o, m_i, 0}, {},
                                              true)[span_t(
                                              {0, 0, 0, 0, n_i}, lanes)]
                                    * builder::make_broadcast(
                                            builder::tensor_ptr(_relu_out_1,
                                                    {bs, m_o, 0, m_i, 0}, {},
                                                    true)[{0, 0, 0, 0, 0}],
                                            lanes);
                        }
                    }
                }
            }
        }
        _return_(true);
    }
    // compare final TIR
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(f, expected, false));
}
