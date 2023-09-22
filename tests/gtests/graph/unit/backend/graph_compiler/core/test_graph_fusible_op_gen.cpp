/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <iostream>
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <runtime/config.hpp>

#define CMP_SIMPLIFIED_IR(AAA, BBB) \
    ir_simplifier_t simp {false}; \
    ir_comparer cmper(true); \
    EXPECT_TRUE(cmper.compare(simp(AAA), simp(BBB), false));

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorAdd) {
    REQUIRE_PARALLEL();
    sc_graph_t mgr;
    auto ins = mgr.make_input({graph_tensor::make({32, 16, 64}),
            graph_tensor::make({32, 16, 64})});
    auto addop = mgr.make("add", ins->get_outputs(), {}, {});
    mgr.make_output(addop->get_outputs());
    auto addf = lower_graph(get_test_ctx(), mgr, {})->get_func("add__1");
    ASSERT_TRUE(addf);
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    builder::ir_builder_t builder;
    for_loop l0, l1;
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {32UL, 16UL, 64UL}),
            _arg_("in0", datatypes::f32, {32UL, 16UL, 64UL}),
            _arg_("in1", datatypes::f32, {32UL, 16UL, 64UL})) {
        _bind_(out, in0, in1);
        _named_for_(l0, ii, 0, 32UL, 1, for_type::PARALLEL) {
            _named_for_(l1, jj, 0, 16UL) {
                _for_(f1, 0, 1) {
                    _for_(f2, 0, 1) {
                        _for_(f3, 0, UINT64_C(64), simd_len) {
                            auto op = builder::tensor_ptr(
                                    out, {ii, jj, 0}, {}, true);
                            auto ip0 = builder::tensor_ptr(
                                    in0, {ii, jj, 0}, {}, true);
                            auto ip1 = builder::tensor_ptr(
                                    in1, {ii, jj, 0}, {}, true);
                            op[span_t({f1, f2, f3}, simd_len)]
                                    = ip0[span_t({f1, f2, f3}, simd_len)]
                                    + ip1[span_t({f1, f2, f3}, simd_len)];
                        }
                    }
                }
            }
        }
        _return_(true);
    }
    l0->fuse(l1);
    CMP_SIMPLIFIED_IR(addf, bbb);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorAdd2) {
    REQUIRE_PARALLEL();
    REQUIRE_AVX();
    bool is_builtin = is_builtin_test_ctx();
    sc_graph_t mgr;
    auto ins = mgr.make_input(
            {graph_tensor::make({1030}), graph_tensor::make({1030})});
    auto addop = mgr.make("add", ins->get_outputs(), {}, {});
    mgr.make_output(addop->get_outputs());
    auto addf = lower_graph(get_test_ctx(), mgr, {})->get_func("add__1");
    ASSERT_TRUE(addf);
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    builder::ir_builder_t builder;
    _function_(datatypes::boolean, bbb, _arg_("out", datatypes::f32, {1030UL}),
            _arg_("in0", datatypes::f32, {1030UL}),
            _arg_("in1", datatypes::f32, {1030UL})) {
        _bind_(out, in0, in1);
        auto op = builder::tensor_ptr(out, {0}, {}, true);
        auto ip0 = builder::tensor_ptr(in0, {0}, {}, true);
        auto ip1 = builder::tensor_ptr(in1, {0}, {}, true);
        auto iter = builder::make_var(datatypes::index, "iter");
        auto iter1 = builder::make_var(datatypes::index, "iter1");
        auto iter2 = builder::make_var(datatypes::index, "iter2");

        auto mask = last_dim_generate_mask(
                iter, expr(1024UL), 1030, simd_len, true);

        auto loop = builder::make_for_loop_unattached(iter, 0, UINT64_C(1024),
                simd_len,
                builder::make_stmts_unattached({builder::make_assign_unattached(
                        op[span_t({iter}, simd_len)],
                        ip0[span_t({iter}, simd_len)]
                                + ip1[span_t({iter}, simd_len)])}),
                true, for_type::PARALLEL);
        auto loop1 = builder::make_for_loop_unattached(iter1, expr(1024UL),
                UINT64_C(1030), simd_len,
                builder::make_stmts_unattached({builder::make_assign_unattached(
                        op[span_t({iter1}, simd_len, mask)],
                        ip0[span_t({iter1}, simd_len, mask)]
                                + ip1[span_t({iter1}, simd_len, mask)])}),
                true, for_type::PARALLEL);
        auto loop2 = builder::make_for_loop_unattached(iter2, expr(1024UL),
                UINT64_C(1030), 1,
                builder::make_stmts_unattached({builder::make_assign_unattached(
                        op[span_t({iter2}, 1)],
                        ip0[span_t({iter2}, 1)] + ip1[span_t({iter2}, 1)])}),
                true, for_type::NORMAL);
        builder.emit(loop);
        if (is_builtin) {
            builder.emit(loop1);
        } else {
            builder.emit(loop2);
        }

        _return_(true);
    }
    CMP_SIMPLIFIED_IR(addf, bbb);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorReorder) {
    SET_THREADS_OR_SKIP(56);

    sc_graph_t mgr;
    auto ins = mgr.make_input(
            {graph_tensor::make({32, 64}, sc_data_format_t::MKmk(4, 8))});
    auto reorderop = mgr.make("reorder", ins->get_outputs(),
            {graph_tensor::make({32, 64}, sc_data_format_t::MKmk(16, 16))}, {});
    mgr.make_output(reorderop->get_outputs());
    auto reorderf
            = lower_graph(get_test_ctx(), mgr, {})->get_func("reorder__1");
    ASSERT_TRUE(reorderf);
    builder::ir_builder_t builder;
    for_loop l0, l1, l2;
    int lanes = std::min(
            (int)get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32), 8);
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {2UL, 4UL, 16UL, 16UL}),
            _arg_("in0", datatypes::f32, {8UL, 8UL, 4UL, 8UL})) {
        _bind_(out, in0);
        auto ip0 = builder::tensor_ptr(in0, {0, 0, 0, 0}, {}, true);
        _named_for_(l0, ii, 0, 8UL, 1, for_type::PARALLEL) {
            _named_for_(l1, jj, 0, 8) {
                _named_for_(l2, kk, 0, 4) {
                    _for_(ll, 0, 8UL, lanes) {
                        out[span_t({(((0 + (kk + 0)) + ((ii + 0) * 4)) / 16),
                                           (((0 + (ll + 0)) + ((jj + 0) * 8))
                                                   / 16),
                                           (((0 + (kk + 0)) + ((ii + 0) * 4))
                                                   % 16),
                                           (((0 + (ll + 0)) + ((jj + 0) * 8))
                                                   % 16)},
                                lanes)]
                                = ip0[span_t({ii, jj, kk, ll}, lanes)];
                    }
                }
            }
        }
        _return_(true);
    }
    l0->fuse(l1)->fuse(l2);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(reorderf, bbb, false));
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorReorder2) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(32);
    sc_graph_t mgr;
    auto ins = mgr.make_input({graph_tensor::make(
            {2048, 32, 64, 16}, sc_data_format_t(format_kinds::ACBD))});
    auto reorderop = mgr.make("reorder", ins->get_outputs(),
            {graph_tensor::make({2048, 32, 64, 16},
                    sc_data_format_t(format_kinds::ABCDdc, {16, 16, 0, 0}))},
            {});
    mgr.make_output(reorderop->get_outputs());
    auto reorderf
            = lower_graph(get_test_ctx(), mgr, {})->get_func("reorder__1");
    ASSERT_TRUE(reorderf);
    constant_folder_t pass;
    auto_caster_t pass2;
    auto reorderf2 = pass(pass2(reorderf));

    builder::ir_builder_t builder;
    for_loop li, lj;
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {2048UL, 32UL, 4UL, 1UL, 16UL, 16UL}),
            _arg_("in0", datatypes::f32, {2048UL, 64UL, 32UL, 16UL})) {
        _bind_(out, in0);
        _named_for_(li, l0, 0UL, 2048UL, 1UL, for_type::PARALLEL) {
            _named_for_(lj, l1, 0UL, 32UL, 1UL) _for_(l2, 0UL, 4UL, 1UL)
                    _for_(l3, 0UL, 1UL, 1UL) _for_(l4, 0UL, 16UL, 8UL)
                            _for_(l5, 0UL, 16UL, 8UL) {
                auto ip0 = in0;
                _var_(row0, sc_data_type_t::f32(8));
                _var_(row1, sc_data_type_t::f32(8));
                _var_(row2, sc_data_type_t::f32(8));
                _var_(row3, sc_data_type_t::f32(8));
                _var_(row4, sc_data_type_t::f32(8));
                _var_(row5, sc_data_type_t::f32(8));
                _var_(row6, sc_data_type_t::f32(8));
                _var_(row7, sc_data_type_t::f32(8));
                _var_(row8, sc_data_type_t::f32(8));
                _var_(row9, sc_data_type_t::f32(8));
                _var_(row10, sc_data_type_t::f32(8));
                _var_(row11, sc_data_type_t::f32(8));
                row0 = ip0[span_t({l0, l5 + l2 * UINT64_C(16), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row1 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(1), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row2 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(2), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row3 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(3), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row4 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(4), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row5 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(5), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row6 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(6), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row7 = ip0[span_t({l0, l5 + l2 * UINT64_C(16) + UINT64_C(7), l1,
                                          l4 + l3 * UINT64_C(16)},
                        8)];
                row8 = builder::make_unpack_low(row0, row1);
                row0 = builder::make_unpack_high(row0, row1);
                row9 = builder::make_unpack_low(row2, row3);
                row1 = builder::make_unpack_high(row2, row3);
                row10 = builder::make_unpack_low(row4, row5);
                row2 = builder::make_unpack_high(row4, row5);
                row11 = builder::make_unpack_low(row6, row7);
                row3 = builder::make_unpack_high(row6, row7);

                row4 = builder::make_shuffle(row8, row9, 68, 32);
                row5 = builder::make_shuffle(row8, row9, 238, 32);
                row6 = builder::make_shuffle(row0, row1, 68, 32);
                row7 = builder::make_shuffle(row0, row1, 238, 32);
                row8 = builder::make_shuffle(row10, row11, 68, 32);
                row9 = builder::make_shuffle(row10, row11, 238, 32);
                row10 = builder::make_shuffle(row2, row3, 68, 32);
                row11 = builder::make_shuffle(row2, row3, 238, 32);

                row0 = builder::make_permute(row4, row8, 32, 128);
                row1 = builder::make_permute(row5, row9, 32, 128);
                row2 = builder::make_permute(row6, row10, 32, 128);
                row3 = builder::make_permute(row7, row11, 32, 128);
                row4 = builder::make_permute(row4, row8, 49, 128);
                row5 = builder::make_permute(row5, row9, 49, 128);
                row6 = builder::make_permute(row6, row10, 49, 128);
                row7 = builder::make_permute(row7, row11, 49, 128);
                out[span_t({l0, l1, l2, l3, l4, l5}, 8)] = row0;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(1), l5}, 8)] = row1;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(2), l5}, 8)] = row2;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(3), l5}, 8)] = row3;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(4), l5}, 8)] = row4;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(5), l5}, 8)] = row5;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(6), l5}, 8)] = row6;
                out[span_t({l0, l1, l2, l3, l4 + UINT64_C(7), l5}, 8)] = row7;
            }
        }
        _return_(true);
    }
    if (runtime_config_t::get().get_num_threads() == 224) {
        // outer 2 loops will be fused when OMP_NUM_THREADS=224
        li->fuse(lj);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(reorderf2, bbb, false));
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorReduce) {
    REQUIRE_PARALLEL();
    sc_graph_t mgr;
    auto ins = mgr.make_input(
            {graph_tensor::make({32, 32, 64, 64}, sc_data_format_t::NCHW())});
    auto addop = mgr.make("reduce", ins->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    mgr.make_output(addop->get_outputs());
    auto reducef = lower_graph(get_test_ctx(), mgr, {})->get_func("reduce__1");
    ASSERT_TRUE(reducef);
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    builder::ir_builder_t builder;
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {32UL, 1UL, 64UL, 64UL}),
            _arg_("in", datatypes::f32, {32UL, 32UL, 64UL, 64UL})) {
        _bind_(out, in);
        _for_(fused, 0UL, 2048UL, 1UL, for_type::PARALLEL) {
            _for_(itr1, 0, 32UL, 1) {}
            auto op = builder::tensor_ptr(out,
                    {(fused / UINT64_C(64)), 0, (fused % UINT64_C(64)), 0}, {},
                    true);
            auto ip0 = builder::tensor_ptr(in,
                    {(fused / UINT64_C(64)), 0, (fused % UINT64_C(64)), 0}, {},
                    true);
            _for_(f7, 0, 1) {
                _for_(f9, 0, 1) {
                    _for_(f10, 0, 64UL, simd_len) {
                        _var_(reduce_v, sc_data_type_t::f32(simd_len));
                        reduce_v = make_expr<constant_node>(
                                0UL, sc_data_type_t::f32(simd_len));
                        _for_(f8, 0, 32UL, 1) {
                            reduce_v = ip0[span_t({f7, f8, f9, f10}, simd_len)]
                                    + reduce_v;
                        }
                        op[span_t({f7, 0, f9, f10}, simd_len)] = reduce_v;
                    }
                }
            }
        }
        _return_(true);
    }

    CMP_SIMPLIFIED_IR(reducef, bbb);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorReduceBroadcast) {
    auto has_fused_op = [](sc_graph_t &g) {
        for (auto &op : g.ops_) {
            if (op->isa<fused_op_t>()) { return true; }
        }
        return false;
    };
    {
        // reduce is connected to broadcast input of add
        sc_graph_t mgr;
        auto ins = mgr.make_input({graph_tensor::make({16, 1, 64, 1})});
        auto ins2 = mgr.make_input({graph_tensor::make({16, 64, 32})});
        auto rdop = mgr.make("reduce", ins->get_outputs(), {},
                {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0},
                        {"keep_dims", false}});
        auto addop = mgr.make("add",
                {rdop->get_outputs()[0], ins2->get_outputs()[0]}, {}, {});
        mgr.make_output(addop->get_outputs());
        fuse_ops(mgr, get_test_ctx());
        // check no fused
        ASSERT_EQ(mgr.ops_.size(), 5UL);
        ASSERT_FALSE(has_fused_op(mgr));
    }

    {
        // reduce is connected to non-broadcast input of add, ok to fuse
        sc_graph_t mgr;
        auto ins = mgr.make_input({graph_tensor::make({16, 32, 64, 32})});
        auto ins2 = mgr.make_input({graph_tensor::make({1, 64, 1})});
        auto rdop = mgr.make("reduce", ins->get_outputs(), {},
                {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0},
                        {"keep_dims", false}});
        auto addop = mgr.make("add",
                {rdop->get_outputs()[0], ins2->get_outputs()[0]}, {}, {});
        mgr.make_output(addop->get_outputs());
        fuse_ops(mgr, get_test_ctx());
        // check fused
        ASSERT_EQ(mgr.ops_.size(), 4UL);
        ASSERT_TRUE(has_fused_op(mgr));
    }

    {
        // reduce is connected to broadcast output
        sc_graph_t mgr;
        auto ins = mgr.make_input({graph_tensor::make({16, 32, 64, 32})});
        auto ins2 = mgr.make_input({graph_tensor::make({1, 64, 1})});
        auto addop = mgr.make(
                "add", {ins->get_outputs()[0], ins2->get_outputs()[0]}, {}, {});
        auto rdop = mgr.make("reduce", addop->get_outputs(), {},
                {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0},
                        {"keep_dims", false}});
        mgr.make_output(rdop->get_outputs());
        fuse_ops(mgr, get_test_ctx());
        // check non fused
        ASSERT_EQ(mgr.ops_.size(), 5UL);
        ASSERT_FALSE(has_fused_op(mgr));
    }
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorFuse) {
    REQUIRE_PARALLEL();
    sc_graph_t mgr;
    auto ins = mgr.make_input({graph_tensor::make({32, 16}),
            graph_tensor::make({32, 16}), graph_tensor::make({32, 16})});
    std::vector<graph_tensor_ptr> add1args {
            ins->get_outputs()[0], ins->get_outputs()[1]};
    auto addop = mgr.make("add", add1args, {}, {});
    auto addop2 = mgr.make(
            "add", {addop->get_outputs()[0], ins->get_outputs()[2]}, {}, {});
    auto reluop = mgr.make("relu", {addop2->get_outputs()[0]}, {}, {});
    auto reduceop = mgr.make("reduce", {reluop->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {0}}, {"rd_op", 0}});
    mgr.make_output(reduceop->get_outputs());
    fuse_ops(mgr);
    mgr.reset_op_ids();
    auto fusedf = lower_graph(get_test_ctx(), mgr, {})
                          ->get_func("add_add_relu_reduce__2");
    ASSERT_TRUE(fusedf);
    builder::ir_builder_t builder;
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {1UL, 16UL}),
            _arg_("in0", datatypes::f32, {32UL, 16UL}),
            _arg_("in1", datatypes::f32, {32UL, 16UL}),
            _arg_("in2", datatypes::f32, {32UL, 16UL})) {
        _bind_(out, in0, in1, in2);
        _tensor_(buf0, datatypes::f32, 32UL, 16UL);
        _for_(itr0, 0, 32UL, 1, for_type::PARALLEL) {
            _for_(f1, 0, 1) _for_(f2, 0, 16UL, simd_len) {
                auto bp1 = builder::tensor_ptr(buf0, {itr0, 0}, {}, true);
                auto ip0 = builder::tensor_ptr(in0, {itr0, 0}, {}, true);
                auto ip1 = builder::tensor_ptr(in1, {itr0, 0}, {}, true);
                bp1[span_t({f1, f2}, simd_len)]
                        = ip0[span_t({f1, f2}, simd_len)]
                        + ip1[span_t({f1, f2}, simd_len)];
                bp1 = builder::tensor_ptr(buf0, {itr0, f2}, {}, true);
                auto ip2 = builder::tensor_ptr(in2, {itr0, f2}, {}, true);
                builder::tensor_ptr(
                        buf0, {itr0, f2}, {}, true)[span_t({f1, 0}, simd_len)]
                        = bp1[span_t({f1, 0}, simd_len)]
                        + ip2[span_t({f1, 0}, simd_len)];
                auto vec_zero = make_expr<constant_node>(
                        0UL, sc_data_type_t::f32(simd_len));
                auto tsr_index = builder::tensor_ptr(
                        buf0, {itr0, f2}, {}, true)[span_t({f1, 0}, simd_len)];
                tsr_index = builder::make_max(tsr_index, vec_zero);
            }
        }
        _for_(f6, 0, 16UL, simd_len, for_type::PARALLEL) {
            _var_(reduce_v, sc_data_type_t::f32(simd_len));
            auto op1 = builder::tensor_ptr(out, {0, 0}, {}, true);
            auto bp0 = builder::tensor_ptr(buf0, {0, 0}, {}, true);
            reduce_v = make_expr<constant_node>(
                    0UL, sc_data_type_t::f32(simd_len));
            _for_(f4, 0, 32UL, 1) {
                reduce_v = bp0[span_t({f4, f6}, simd_len)] + reduce_v;
            }
            op1[span_t({0, f6}, simd_len)] = reduce_v;
        }
        _return_(true);
    }
    CMP_SIMPLIFIED_IR(fusedf, bbb);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorNoAxisOptim) {
    sc_graph_t mgr;
    auto run_threads = runtime_config_t::get().get_num_threads();
    // This N value will ensure parallel_num > run_threads in any cases
    int N = run_threads + 1;
    auto ins = mgr.make_input(
            {graph_tensor::make({N, 32, 64, 64}, sc_data_format_t::NCHW())});

    auto addop = mgr.make(
            "add", {ins->get_outputs()[0], ins->get_outputs()[0]}, {}, {});
    auto tviewop = mgr.make("tensor_view", addop->get_outputs(), {},
            {{"shape", sc_dims {N, 32, 64, 64}}});
    auto rdop = mgr.make("reduce", tviewop->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});

    mgr.make_output(rdop->get_outputs());
    fuse_ops(mgr);
    auto a = lower_graph(get_test_ctx(), mgr, {});
    auto reducef = a->get_func("add_tensor_view_reduce__2");
    ASSERT_TRUE(reducef);
    EXPECT_EQ(get_expr_as_int(reducef->body_.checked_as<stmts>()
                                      ->seq_[0]
                                      .checked_as<for_loop>()
                                      ->iter_end_),
            N);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorForcedAxisOptim) {
    REQUIRE_PARALLEL();
    sc_graph_t mgr;
    auto ins = mgr.make_input({graph_tensor::make(
            {1 * 64, 8 * 64}, sc_data_format_t::MKmk(64, 64))});

    auto addop = mgr.make(
            "add", {ins->get_outputs()[0], ins->get_outputs()[0]}, {}, {});
    auto rdop = mgr.make("reduce", addop->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto reo = mgr.make("reorder", rdop->get_outputs(), {},
            {{"out_format", sc_data_format_t::MK()}});

    mgr.make_output(reo->get_outputs());
    fuse_ops(mgr);
    auto a = lower_graph(get_test_ctx(), mgr, {});
    auto add_reducef = a->get_func("add_reduce__3");
    ASSERT_TRUE(add_reducef);
    auto reorderf = a->get_func("reorder__1");
    ASSERT_TRUE(reorderf);
    EXPECT_EQ(get_expr_as_int(add_reducef->body_.checked_as<stmts>()
                                      ->seq_[0]
                                      .checked_as<for_loop>()
                                      ->iter_end_),
            64);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorLoopConflict) {
    sc_graph_t mgr;
    auto ins = mgr.make_input(
            {graph_tensor::make({32, 32, 64, 64}, sc_data_format_t::NCHW())});
    auto constop = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(std::vector<int> {666})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}}});
    // expected to be `no_fused`
    auto reo = mgr.make("reorder", ins->get_outputs(), {},
            {{"out_format", sc_data_format_t::NCHWc(16)}});
    auto addop = mgr.make(
            "add", {reo->get_outputs()[0], constop->get_outputs()[0]}, {}, {});

    mgr.make_output(addop->get_outputs());
    fuse_ops(mgr);
    EXPECT_EQ(reo->attrs_.get_or_else(op_attr_key::no_fuse, false), true);
}

TEST(GCCore_CPU_fusible_op_gen, TestFusibleOpGeneratorExpMask) {
    REQUIRE_PARALLEL();
    REQUIRE_AVX();
    sc_graph_t mgr;
    auto ins = mgr.make_input(
            {graph_tensor::make({32, 35}, sc_data_format_t::MKmk(32, 32))});
    auto expop = mgr.make("exp", ins->get_outputs(), {}, {});
    mgr.make_output(expop->get_outputs());
    fuse_ops(mgr);
    auto a = lower_graph(get_test_ctx(), mgr, {});
    auto expf = a->get_func("exp__1");
    ASSERT_TRUE(expf);
    auto expf2 = constant_folder_t()(expf);
    builder::ir_builder_t builder;
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    // 8 for windows ci
    auto mask_dtype = simd_len == 16 ? datatypes::u16 : datatypes::u8;
    auto full_mask = simd_len == 16
            ? builder::make_constant({UINT64_C(0xffff)}, mask_dtype)
            : builder::make_constant({UINT64_C(0xff)}, mask_dtype);
    auto empty_mask = builder::make_constant({UINT64_C(0)}, mask_dtype);
    auto mask_zero
            = builder::make_constant(std::vector<union_val>(simd_len, 0.f),
                    sc_data_type_t::f32(simd_len));
    _function_(datatypes::boolean, bbb,
            _arg_("out", datatypes::f32, {1UL, 2UL, 32UL, 32UL}),
            _arg_("in", datatypes::f32, {1UL, 2UL, 32UL, 32UL})) {
        _bind_(out, in);
        _for_(fused, 0UL, 64UL, 1UL, for_type::PARALLEL) {
            _for_(iter3, 0, UINT64_C(32), simd_len) {
                auto in_ptr = builder::tensor_ptr(in,
                        {fused / UINT64_C(64),
                                fused / UINT64_C(32) % UINT64_C(2),
                                fused % UINT64_C(32), 0},
                        {}, true);
                auto out_ptr = builder::tensor_ptr(out,
                        {fused / UINT64_C(64),
                                fused / UINT64_C(32) % UINT64_C(2),
                                fused % UINT64_C(32), 0},
                        {}, true);
                expr cur_idx = (iter3
                        + (0 + fused / UINT64_C(32) % UINT64_C(2)) * 32);
                expr offset = builder::make_min(
                        builder::make_max(35
                                        - builder::make_cast(
                                                datatypes::s32, cur_idx),
                                0),
                        simd_len);
                expr mask_select = builder::make_select(offset == 0, empty_mask,
                        builder::make_select(offset == simd_len, full_mask,
                                full_mask >> builder::make_cast(
                                        mask_dtype, simd_len - offset)));
                out_ptr[span_t({0, 0, 0, iter3}, simd_len)] = builder::make_exp(
                        in_ptr[span_t({0, 0, 0, iter3}, simd_len)]);
                _var_init_(mask_var, mask_dtype, mask_select);
                out_ptr[span_t({0, 0, 0, iter3}, simd_len)]
                        = builder::make_select(mask_var,
                                out_ptr[span_t({0, 0, 0, iter3}, simd_len)],
                                mask_zero);
            }
        }
        _return_(true);
    }
    CMP_SIMPLIFIED_IR(expf2, constant_folder_t()(bbb));
}
