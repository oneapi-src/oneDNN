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

#include <chrono>
#include <iostream>
#include <utility>
#include <vector>
#include "context.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_anchor.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/buffer_schedule_utils.hpp>
#include <compiler/ir/transform/loop_merge.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/matmul_core.hpp>
#include <ops/reduce_graph_op.hpp>
#include <ops/templates/matmul_core.hpp>
#include <reference/act_ref.hpp>
#include <test_utils.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
using namespace std::chrono;

#define CMP_SIMPLIFIED_IR(AAA, BBB) \
    ir_simplifier_t simp {false}; \
    ir_comparer cmper(true); \
    loop_merger_t lm; \
    auto A = simp(lm(AAA)); \
    auto B = simp(lm(BBB)); \
    EXPECT_TRUE(cmper.compare(A, B, false));

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerElemBlock) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(test_utils::make_tsr({100}, datatypes::s32));
    auto frelu = g.make("relu", finput->get_outputs(), {}, {});
    auto fmul = g.make(
            "mul", {frelu->get_outputs()[0], finput->get_outputs()[0]}, {}, {});
    auto foutput = g.make_output(fmul->get_outputs());
    EXPECT_EQ(fmul->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], slice_range {{0, 32}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);
    int32_t lanes
            = frelu->dyn_cast<const unary_elementwise_op_impl_t>()->get_lanes();

    ///// Expected func:
    _function_(datatypes::s32, bbb, _arg_("out", datatypes::s32, {100UL}),
            _arg_("buf", datatypes::s32, {100UL})) {
        _bind_(out, buf);
        _tensor_(tbuf, datatypes::s32, {100UL});
        auto tb = tensor_ptr(tbuf, {0}, {}, true);
        auto b = tensor_ptr(buf, {0}, {}, true);
        _for_(i, 0, 32, lanes) {
            tb[span_t({i}, lanes)] = builder::make_max(
                    builder::make_constant(
                            {0UL}, sc_data_type_t(sc_data_etype::S32, lanes)),
                    b[span_t({i}, lanes)]);
            auto o = tensor_ptr(out, {0}, {}, true);
            tb = tensor_ptr(tbuf, {0}, {}, true);
            b = tensor_ptr(buf, {0}, {}, true);
            o[span_t({i}, lanes)]
                    = tb[span_t({i}, lanes)] * b[span_t({i}, lanes)];
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestConcatOP) {
    REQUIRE_AVX512(); // vec lane is 16 for s32 dtype
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput0 = g.make_input(test_utils::make_tsr({100, 200, 10}));
    auto finput1 = g.make_input(test_utils::make_tsr({100, 300, 10}));
    auto finput2 = g.make_input(test_utils::make_tsr({100, 400, 10}));
    auto fconcat = g.make("concat",
            {finput0->get_outputs()[0], finput1->get_outputs()[0],
                    finput2->get_outputs()[0]},
            {}, {{"axis", 1}});
    auto fout = g.make_output(fconcat->get_outputs());
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                slice_map {{finput0->get_outputs()[0].get(),
                                   {{/*dim1*/ {0, 10}, /*dim2*/ {0, 200},
                                           /*dim3*/ {0, 6}}}},
                        {finput1->get_outputs()[0].get(),
                                {{/*dim1*/ {0, 10}, /*dim2*/ {0, 300},
                                        /*dim3*/ {0, 6}}}},
                        {finput2->get_outputs()[0].get(),
                                {{/*dim1*/ {0, 10}, /*dim2*/ {0, 400},
                                        /*dim3*/ {0, 6}}}}});
        _return_(123);
    }

    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes = 16; // for avx512 and s32 dtype
    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::s32, {100UL, 900UL, 10UL}),
            _arg_("inp0", datatypes::s32, {100UL, 200UL, 10UL}),
            _arg_("inp1", datatypes::s32, {100UL, 300UL, 10UL}),
            _arg_("inp2", datatypes::s32, {100UL, 400UL, 10UL})) {
        _bind_(out0, inp0, inp1, inp2);
        auto out0_ptr = builder::tensor_ptr(out0, {0, 0, 0}, {}, true);
        auto inp0_ptr = builder::tensor_ptr(inp0, {0, 0, 0}, {}, true);
        auto inp1_ptr = builder::tensor_ptr(inp1, {0, 0, 0}, {}, true);
        auto inp2_ptr = builder::tensor_ptr(inp2, {0, 0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj0, 0, 200) {
                _for_(hh0, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh0, 0, 6, int(lanes), true);
                    out0_ptr[span_t({ii, jj0 + expr(0), hh0}, lanes, mask)]
                            = inp0_ptr[span_t({ii, jj0, hh0}, lanes, mask)];
                }
            }
            _for_(jj1, 0, 300) {
                _for_(hh1, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh1, 0, 6, int(lanes), true);
                    out0_ptr[span_t({ii, jj1 + (expr(0) + expr(200)), hh1},
                            lanes, mask)]
                            = inp1_ptr[span_t({ii, jj1, hh1}, lanes, mask)];
                }
            }
            _for_(jj2, 0, 400) {
                _for_(hh2, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh2, 0, 6, int(lanes), true);
                    out0_ptr[span_t(
                            {ii, jj2 + (expr(0) + expr(200) + expr(300)), hh2},
                            lanes, mask)]
                            = inp2_ptr[span_t({ii, jj2, hh2}, lanes, mask)];
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

#ifdef __AVX512F__

TEST(GCCore_CPU_fuse_mgr_cpp, TestLeakyReluOP) {
    auto check_leaky_relu = [&](sc_data_type_t type, const int M, const int K,
                                    float alpha) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input(test_utils::make_tsr({M, K}, type));
        auto flkrelu = g.make(
                "leaky_relu", fin->get_outputs(), {}, {{"alpha", alpha}});
        auto fout = g.make_output(flkrelu->get_outputs());

        _function_(datatypes::void_t, testf) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {{0, M}, {0, K}});
        }
        commit_graph_to_func(g, testf, fusion);

        auto testf_mod
                = ir_module_t::from_entry_func(get_default_context(), testf);
        auto testf_ptr = jit_engine_t::make(get_test_ctx())
                                 ->get_entry_func(testf_mod, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        if (type == datatypes::bf16) {
            std::vector<bf16_t> cast_in_buf(M * K);
            std::vector<bf16_t> cast_out_buf(M * K);
            std::vector<bf16_t> ref_cast_out_buf(M * K);

            for (int i = 0; i < M * K; i++) {
                cast_in_buf[i] = bf16_t(in_buf[i]);
            }
            testf_ptr->call_default(cast_out_buf.data(), cast_in_buf.data());
            for (int i = 0; i < M * K; i++) {
                out_buf[i] = cast_out_buf[i];
            }
            ref_leaky_relu(
                    ref_cast_out_buf.data(), cast_in_buf.data(), M * K, alpha);
            for (int i = 0; i < M * K; i++) {
                ref_out_buf[i] = ref_cast_out_buf[i];
            }
        } else {
            testf_ptr->call_default(out_buf.data(), in_buf.data());
            ref_leaky_relu(ref_out_buf.data(), in_buf.data(), M * K, alpha);
        }

        test_utils::compare_data(out_buf, ref_out_buf, 1e-4f, 1e-4f);
    };

    check_leaky_relu(datatypes::f32, 100, 200, 0.01);
    check_leaky_relu(datatypes::f32, 100, 256, 0.01);
    check_leaky_relu(datatypes::f32, 100, 200, 0.5f);
    check_leaky_relu(datatypes::f32, 100, 256, 0.5f);

    check_leaky_relu(datatypes::bf16, 100, 200, 0.01);
    check_leaky_relu(datatypes::bf16, 100, 256, 0.01);
    check_leaky_relu(datatypes::bf16, 100, 200, 0.5f);
    check_leaky_relu(datatypes::bf16, 100, 256, 0.5f);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestTanhOP) {
    auto check_tanh = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input(test_utils::make_tsr({M, K}, datatypes::f32));
        auto ftanh = g.make("tanh", fin->get_outputs(), {}, {});
        auto fout = g.make_output(ftanh->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {{0, M}, {0, K}});
        }
        commit_graph_to_func(g, aaa, fusion);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -500.f;
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = 500.f;

        fptr->call_default(out_buf.data(), in_buf.data());
        ref_tanh(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 5e-3f, 1e-4f);
    };
    // scalar version
    check_tanh(100, 200);

    // vectorization version
    check_tanh(100, 256);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestErfOP) {
    auto check_erf = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input(test_utils::make_tsr({M, K}, datatypes::f32));
        auto ffwd = g.make("erf", fin->get_outputs(), {}, {});
        auto fout = g.make_output(ffwd->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {{0, M}, {0, K}});
        }

        commit_graph_to_func(g, aaa, fusion);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -500.f;
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = 500.f;

        fptr->call_default(out_buf.data(), in_buf.data());
        ref_erf(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 5e-3f, 1e-4f);
    };
    // scalar version
    check_erf(128, 200);

    // vectorization version
    check_erf(128, 256);
}
#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast1) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    int bc_block = 64;
    auto finput = g.make_input(test_utils::make_tsr({50, 100, 200}));
    auto finput_add
            = g.make_input(test_utils::make_tsr({1, 100, 200}, datatypes::s32));
    auto fadd = g.make("add",
            {finput->get_outputs()[0], finput_add->get_outputs()[0]}, {}, {});
    auto foutput = g.make_output(fadd->get_outputs());
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(finput->get_outputs()[0],
                {{0, 50}, {0, bc_block}, {0, bc_block}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = fadd->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("buf", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("bc_args_add", datatypes::s32, {1UL, 100UL, 200UL})) {
        _bind_(out, buf, bc_args_add);
        auto bc_args_add_tptr
                = builder::tensor_ptr(bc_args_add, {0, 0, 0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + bc_args_add_tptr[span_t({0, i, j}, lanes)];
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast2) {
    bool is_builtin = is_builtin_test_ctx();
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    int bc_block = 64;
    auto finput = g.make_input(test_utils::make_tsr({50, 100, 200}));
    auto finput_add
            = g.make_input(test_utils::make_tsr({1, 100, 1}, datatypes::s32));
    auto fadd = g.make("add",
            {finput->get_outputs()[0], finput_add->get_outputs()[0]}, {}, {});
    auto foutput = g.make_output(fadd->get_outputs());
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{0, 50}, {0, bc_block}, {0, 50}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = fadd->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("buf", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("bc_args_add", datatypes::s32, {1UL, 100UL, 1UL})) {
        _bind_(out, buf, bc_args_add);
        auto bc_args_add_tptr
                = builder::tensor_ptr(bc_args_add, {0, 0, 0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block) {
                _for_(j, 0, 48, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + builder::make_broadcast(
                                    bc_args_add_tptr[{0, i, 0}],
                                    static_cast<int>(lanes));
                }
                if (is_builtin) {
                    _for_(j, 48, 50, int(lanes)) {
                        auto mask = last_dim_generate_mask(
                                j, 48, 50, int(lanes), true);
                        out_tptr[span_t({n, i, j}, lanes, mask)]
                                = buf_tptr[span_t({n, i, j}, lanes, mask)]
                                + builder::make_broadcast(
                                        bc_args_add_tptr[{0, i, 0}],
                                        static_cast<int>(lanes));
                    }
                } else {
                    _for_(j, 48, 50, 1) {
                        out_tptr[span_t({n, i, j}, 1)]
                                = buf_tptr[span_t({n, i, j}, 1)]
                                + bc_args_add_tptr[{0, i, 0}];
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast3) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    int bc_block = 64;
    auto finput = g.make_input(test_utils::make_tsr({50, 100, 200}));
    auto finput_add = g.make_input(test_utils::make_tsr({100}, datatypes::s32));
    auto fadd = g.make("add",
            {finput->get_outputs()[0], finput_add->get_outputs()[0]}, {},
            any_map_t {{"bc_axis", std::vector<int> {1}}});
    auto foutput = g.make_output(fadd->get_outputs());
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(finput->get_outputs()[0],
                {{0, 50}, {0, bc_block}, {0, bc_block}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = fadd->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("buf", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("bc_args_add", datatypes::s32, {100UL})) {
        _bind_(out, buf, bc_args_add);
        auto bc_args_add_tptr = builder::tensor_ptr(bc_args_add, {0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + builder::make_broadcast(bc_args_add_tptr[i],
                                    static_cast<int>(lanes));
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast4) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    int bc_block = 64;
    auto finput = g.make_input(test_utils::make_tsr({100}));
    auto finput_add = g.make_input(test_utils::make_tsr({50, 100, 200}));
    auto fadd = g.make("sub",
            {finput->get_outputs()[0], finput_add->get_outputs()[0]}, {},
            any_map_t {{"bc_axis", std::vector<int> {1}}});
    auto foutput = g.make_output(fadd->get_outputs());
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                slice_map {{finput_add->get_outputs()[0].get(),
                                   {{{0, 50}, {0, bc_block}, {0, bc_block}}}},
                        {finput->get_outputs()[0].get(), {{{0, bc_block}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = fadd->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {50UL, 100UL, 200UL}),
            _arg_("bc_args_add", datatypes::s32, {100UL}),
            _arg_("buf", datatypes::s32, {50UL, 100UL, 200UL})) {
        _bind_(out, bc_args_add, buf);
        auto bc_args_add_tptr = builder::tensor_ptr(bc_args_add, {0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = builder::make_broadcast(bc_args_add_tptr[i],
                                      static_cast<int>(lanes))
                            - buf_tptr[span_t({n, i, j}, lanes)];
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerVectorizedReLU) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(test_utils::make_tsr({100, 256}));
    auto frelu = g.make("relu", finput->get_outputs(), {}, {});
    auto foutput = g.make_output(frelu->get_outputs());
    EXPECT_EQ(frelu->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{0, 100}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = frelu->dyn_cast<const unary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {100UL, 256UL}),
            _arg_("buf", datatypes::s32, {100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0}, {}, true);
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 256, int(lanes)) {
                out_tptr[span_t({i, j}, lanes)]
                        = builder::make_max(make_expr<constant_node>((int64_t)0,
                                                    sc_data_type_t::s32(lanes)),
                                buf_tptr[span_t({i, j}, lanes)]);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerExp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(test_utils::make_tsr({100, 256}));
    auto fexp = g.make<exp_op_t>(finput->get_outputs()[0]);
    auto foutput = g.make_output(fexp->get_outputs());
    EXPECT_EQ(fexp->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::f32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{0, 100}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    uint32_t lanes
            = fexp->dyn_cast<const unary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::f32, bbb,
            _arg_("out", datatypes::s32, {100UL, 256UL}),
            _arg_("buf", datatypes::s32, {100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0}, {}, true);
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 256, int(lanes)) {
                out_tptr[span_t({i, j}, lanes)]
                        = builder::make_exp(buf_tptr[span_t({i, j}, lanes)]);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestSigmoidOP) {
    REQUIRE_AVX2();
    auto check_sigmoid = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input(test_utils::make_tsr({M, K}, datatypes::f32));
        auto fsig = g.make("sigmoid", fin->get_outputs(), {}, {});
        auto fout = g.make_output(fsig->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {{0, M}, {0, K}});
        }
        commit_graph_to_func(g, aaa, fusion);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -std::numeric_limits<float>::infinity();
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = std::numeric_limits<float>::infinity();

        in_buf[6] = std::numeric_limits<float>::min();
        in_buf[7] = std::numeric_limits<float>::max();

        in_buf[8] = -std::numeric_limits<float>::min();
        in_buf[9] = -std::numeric_limits<float>::max();

        fptr->call_default(out_buf.data(), in_buf.data());

        ref_sigmoid(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 1e-5f, 1e-5f);
    };
    // scalar version
    check_sigmoid(100, 200);

    // vectorization version
    check_sigmoid(100, 256);
}

// #define BENCH_SIGMOID

#ifdef BENCH_SIGMOID
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedSigmoidOP) {
    auto bench_sigmoid = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input();
        auto fsig = g.make("sigmoid", fin->get_outputs(), {}, {});
        auto fout = g.make_output(fsig->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    finput->get_outputs()[0], {{0, M}, {0, K}});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(out_buf.data(), in_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_sigmoid(100, 256);

    // vectorization version
    bench_sigmoid(100, 2560);

    // vectorization version
    bench_sigmoid(100, 25600);

    // vectorization version
    bench_sigmoid(100, 512000);
}
#endif

//#define BENCH_ROUND

#ifdef BENCH_ROUND
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedRoundOP) {
    auto bench_round = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input();
        auto fround = g.make("round", fin->get_outputs(), {}, {});
        auto fout = g.make_output(fround->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    finput->get_outputs()[0], {{0, M}, {0, K}});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(out_buf.data(), in_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_round(100, 256);

    // vectorization version
    bench_round(100, 2560);

    // vectorization version
    bench_round(100, 25600);

    // vectorization version
    bench_round(100, 512000);
}
#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceSum1) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(
            test_utils::make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = g.make("reduce", finput->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2}},
                    {"rd_op", static_cast<int>(reduce_operator::add)},
                    {"keep_dims", false}});
    auto foutput = g.make_output(fsum->get_outputs());
    EXPECT_EQ(fsum->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{10, 1}, {30, 1}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::s32, bbb, _arg_("out", datatypes::f32, {20UL, 100UL}),
            _arg_("buf", datatypes::f32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_value, sc_data_type_t::f32(lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(lanes));
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    reduce_value = builder::make_add(
                            buf_tptr[span_t({i, j, k}, lanes)], reduce_value);
                }
                out_tptr[{i, j}] = builder::make_reduce_add(reduce_value);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceSum2) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(
            test_utils::make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = g.make("reduce", finput->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}},
                    {"rd_op", static_cast<int>(reduce_operator::add)},
                    {"keep_dims", true}});
    auto foutput = g.make_output(fsum->get_outputs());
    EXPECT_EQ(fsum->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{10, 1}, {0, 100}, {20, 1}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::f32, {20UL, 1UL, 256UL}),
            _arg_("buf", datatypes::f32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 0, 20}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 0, 20}, {}, true);
        _for_(i, 0, 1) {
            _for_(k, 0, 1) {
                _var_(reduce_sum, sc_data_type_t::f32());
                reduce_sum = 0.f;
                _for_(j, 0, 100) {
                    reduce_sum = builder::make_add(
                            buf_tptr[{i, j, k}], reduce_sum);
                }
                out_tptr[{i, 0, k}] = reduce_sum;
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceProd) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(
            test_utils::make_tsr({20, 100, 256}, datatypes::f32));
    auto fprod = g.make("reduce", finput->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2}},
                    {"rd_op", static_cast<int>(reduce_operator::mul)},
                    {"keep_dims", false}});
    auto foutput = g.make_output(fprod->get_outputs());
    EXPECT_EQ(fprod->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{10, 1}, {30, 1}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    uint32_t lanes = fprod->dyn_cast<reduce_op_t>()->get_lanes();
    ///// Expected func:
    _function_(datatypes::s32, bbb, _arg_("out", datatypes::f32, {20UL, 100UL}),
            _arg_("buf", datatypes::f32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_mul, sc_data_type_t::f32(lanes));
                reduce_mul = make_expr<constant_node>(
                        1.0f, sc_data_type_t::f32(lanes));
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    reduce_mul = builder::make_mul(
                            buf_tptr[span_t({i, j, k}, lanes)], reduce_mul);
                }
                out_tptr[{i, j}] = builder::make_reduce_mul(reduce_mul);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredDiff1) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(
            test_utils::make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = g.make("reduce", finput->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2}},
                    {"rd_op", static_cast<int>(reduce_operator::add)},
                    {"keep_dims", true}});
    auto fsquare_diff = g.make("squared_diff",
            {finput->get_outputs()[0], fsum->get_outputs()[0]}, {}, {});
    auto foutput = g.make_output(fsquare_diff->get_outputs());
    EXPECT_EQ(fsquare_diff->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{10, 1}, {30, 1}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t rd_lanes
            = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    uint32_t sd_lanes
            = fsquare_diff->dyn_cast<const binary_elementwise_op_impl_t>()
                      ->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::f32, {20UL, 100UL, 256UL}),
            _arg_("buf", datatypes::f32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 100UL, 1UL});
        auto fuse_buf_tptr
                = builder::tensor_ptr(_fuse_buf_0, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_value, sc_data_type_t::f32(rd_lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(rd_lanes));
                _for_(k, 0, 256, static_cast<int>(rd_lanes)) {
                    reduce_value = builder::make_add(
                            buf_tptr[span_t({i, j, k}, rd_lanes)],
                            reduce_value);
                }
                fuse_buf_tptr[{i, j, 0}]
                        = builder::make_reduce_add(reduce_value);
            }
        }

        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _for_(k, 0, 256, static_cast<int>(sd_lanes)) {
                    out_tptr[span_t({i, j, k}, sd_lanes)]
                            = (buf_tptr[span_t({i, j, k}, sd_lanes)]
                                      - builder::make_broadcast(
                                              builder::tensor_ptr(_fuse_buf_0,
                                                      {10, 30, 0}, {},
                                                      true)[{i, j, 0}],
                                              static_cast<int>(sd_lanes)))
                            * (buf_tptr[span_t({i, j, k}, sd_lanes)]
                                    - builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_0,
                                                    {10, 30, 0}, {},
                                                    true)[{i, j, 0}],
                                            static_cast<int>(sd_lanes)));
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredDiff2) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(test_utils::make_tsr({20, 100, 256}));
    auto fmean = g.make_input(test_utils::make_tsr({20, 100, 256}));
    auto fsquare_diff = g.make("squared_diff",
            {finput->get_outputs()[0], fmean->get_outputs()[0]}, {}, {});
    auto foutput = g.make_output(fsquare_diff->get_outputs());
    EXPECT_EQ(fsquare_diff->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                slice_map {{finput->get_outputs()[0].get(),
                                   {{{10, 1}, {30, 50}, {0, 256}}}},
                        {fmean->get_outputs()[0].get(),
                                {{{10, 1}, {30, 50}, {0, 256}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes
            = fsquare_diff->dyn_cast<const binary_elementwise_op_impl_t>()
                      ->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {20UL, 100UL, 256UL}),
            _arg_("buf", datatypes::s32, {20UL, 100UL, 256UL}),
            _arg_("sqd_mean", datatypes::s32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf, sqd_mean);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        auto sqd_mean_tptr
                = builder::tensor_ptr(sqd_mean, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 50) {
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    out_tptr[span_t({i, j, k}, lanes)]
                            = (buf_tptr[span_t({i, j, k}, lanes)]
                                      - sqd_mean_tptr[span_t({i, j, k}, lanes)])
                            * (buf_tptr[span_t({i, j, k}, lanes)]
                                    - sqd_mean_tptr[span_t({i, j, k}, lanes)]);
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredRoot1) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(test_utils::make_tsr({20, 100, 256}));
    auto fsquare_root = g.make("squared_root", finput->get_outputs(), {}, {});
    auto foutput = g.make_output(fsquare_root->get_outputs());
    EXPECT_EQ(fsquare_root->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                finput->get_outputs()[0], {{10, 1}, {30, 50}, {0, 256}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    uint32_t lanes = fsquare_root->dyn_cast<const unary_elementwise_op_impl_t>()
                             ->get_lanes();

    _function_(datatypes::s32, bbb,
            _arg_("out", datatypes::s32, {20UL, 100UL, 256UL}),
            _arg_("buf", datatypes::s32, {20UL, 100UL, 256UL})) {
        _bind_(out, buf);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 50) {
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    out_tptr[span_t({i, j, k}, lanes)]
                            = make_sqrt(buf_tptr[span_t({i, j, k}, lanes)]);
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerCastBF16) {
    REQUIRE_AVX512();
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finputf32
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::f32));
    auto finputbf16
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::bf16));

    auto fcastf32tobf16 = g.make(
            "cast", finputf32->get_outputs(), {}, {{"dtype", datatypes::bf16}});
    auto fcastbf16tof32 = g.make(
            "cast", finputbf16->get_outputs(), {}, {{"dtype", datatypes::f32}});

    auto foutf32tobf16 = g.make_output(fcastf32tobf16->get_outputs());
    auto foutbf16tof32 = g.make_output(fcastbf16tof32->get_outputs());

    EXPECT_EQ(foutf32tobf16->get_inputs()[0], fcastf32tobf16->get_outputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finputf32->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputbf16->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    std::vector<float> inf32(20 * 259), outbf16tof32(20 * 259);
    std::vector<bf16_t> inbf16(20 * 259), outf32tobf16(20 * 259);
    test_utils::fill_data<float>(inf32.data(), 20 * 259);
    for (int i = 0; i < 20 * 259; i++) {
        inbf16[i] = inf32[i];
    }

    auto fptr = jit_engine_t::make(get_test_ctx())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                                 get_default_context(), aaa),
                                true);
    fptr->call_default(outbf16tof32.data(), outf32tobf16.data(), inbf16.data(),
            inf32.data());

    for (unsigned i = 0; i < 20 * 259; i++) {
        if (std::abs(outf32tobf16[i] - inbf16[i]) >= 1e-5f) {
            std::cout << outf32tobf16[i] << "\n";
        }
        EXPECT_TRUE(
                std::abs(outf32tobf16[i] - float(bf16_t(inf32[i]))) < 1e-5f);
        EXPECT_TRUE(std::abs(outbf16tof32[i] - inbf16[i]) < 1e-5f);
    }
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerCastU8S8) {
    BUILTIN_REQUIRE_AVX512(); // AVX2 no cast instruction
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finputf32tou8
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::f32));
    auto finputf32tos8
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::f32));
    auto finputs32tou8 = g.make_input(test_utils::make_tsr({20, 259}));
    auto finputs32tos8 = g.make_input(test_utils::make_tsr({20, 259}));
    auto finputu8
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::u8));
    auto finputs8
            = g.make_input(test_utils::make_tsr({20, 259}, datatypes::s8));

    auto fcastf32tos32 = g.make("cast", finputf32tos8->get_outputs(), {},
            {{"dtype", datatypes::s32}});
    auto fcastf32tou8 = g.make("cast", finputf32tou8->get_outputs(), {},
            {{"dtype", datatypes::u8}});
    auto fcastf32tos8 = g.make("cast", finputf32tos8->get_outputs(), {},
            {{"dtype", datatypes::s8}});

    auto fcasts32tof32 = g.make("cast", finputs32tos8->get_outputs(), {},
            {{"dtype", datatypes::f32}});
    auto fcasts32tou8 = g.make("cast", finputs32tou8->get_outputs(), {},
            {{"dtype", datatypes::u8}});
    auto fcasts32tos8 = g.make("cast", finputs32tos8->get_outputs(), {},
            {{"dtype", datatypes::s8}});
    auto fcastu8tos32 = g.make(
            "cast", finputu8->get_outputs(), {}, {{"dtype", datatypes::s32}});
    auto fcastu8tof32 = g.make(
            "cast", finputu8->get_outputs(), {}, {{"dtype", datatypes::f32}});
    auto fcasts8tos32 = g.make(
            "cast", finputs8->get_outputs(), {}, {{"dtype", datatypes::s32}});
    auto fcasts8tof32 = g.make(
            "cast", finputs8->get_outputs(), {}, {{"dtype", datatypes::f32}});

    auto foutf32tos32 = g.make_output(fcastf32tos32->get_outputs());
    auto foutf32tou8 = g.make_output(fcastf32tou8->get_outputs());
    auto foutf32tos8 = g.make_output(fcastf32tos8->get_outputs());

    auto fouts32tof32 = g.make_output(fcasts32tof32->get_outputs());
    auto fouts32tou8 = g.make_output(fcasts32tou8->get_outputs());
    auto fouts32tos8 = g.make_output(fcasts32tos8->get_outputs());

    auto foutu8tof32 = g.make_output(fcastu8tof32->get_outputs());
    auto foutu8tos32 = g.make_output(fcastu8tos32->get_outputs());

    auto fouts8tof32 = g.make_output(fcasts8tof32->get_outputs());
    auto fouts8tos32 = g.make_output(fcasts8tos32->get_outputs());

    EXPECT_EQ(fouts8tof32->get_inputs()[0], fcasts8tof32->get_outputs()[0]);
    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finputf32tou8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputf32tos8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputs32tou8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputs32tos8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputf32tos8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputu8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}},
                {finputs8->get_outputs()[0].get(), {{{0, 20}, {0, 259}}}}});
        _return_(123);
    }

    commit_graph_to_func(g, aaa, fusion);

    std::vector<float> outs32tof32(20 * 259), outu8tof32(20 * 259),
            outs8tof32(20 * 259);
    std::vector<int32_t> outf32tos32(20 * 259, 1), outu8tos32(20 * 259),
            outs8tos32(20 * 259);
    std::vector<uint8_t> inu8(20 * 259), outf32tou8(20 * 259),
            outs32tou8(20 * 259);
    std::vector<int8_t> ins8(20 * 259), outf32tos8(20 * 259),
            outs32tos8(20 * 259);
    test_utils::fill_data<uint8_t>(inu8.data(), 20 * 259);
    test_utils::fill_data<int8_t>(ins8.data(), 20 * 259);
    std::vector<float> inf32tou8(inu8.begin(), inu8.end()),
            inf32tos8(ins8.begin(), ins8.end());
    std::vector<int32_t> ins32tou8(inu8.begin(), inu8.end()),
            ins32tos8(ins8.begin(), ins8.end());

    auto fptr = jit_engine_t::make(get_test_ctx())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                                 get_default_context(), aaa),
                                true);
    fptr->call_default(outs8tof32.data(), outs8tos32.data(), outu8tof32.data(),
            outu8tos32.data(), outs32tos8.data(), outs32tof32.data(),
            outs32tou8.data(), outf32tos8.data(), outf32tos32.data(),
            outf32tou8.data(), ins8.data(), inu8.data(), ins32tos8.data(),
            ins32tou8.data(), inf32tos8.data(), inf32tou8.data());
    for (unsigned i = 0; i < 20 * 259; i++) {
        EXPECT_TRUE(outf32tos32[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outf32tou8[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outf32tos8[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(
                static_cast<int>(outs32tof32[i]) == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outs32tou8[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outs32tos8[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(
                static_cast<int>(outu8tof32[i]) == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outu8tos32[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(
                static_cast<int>(outs8tof32[i]) == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outs8tos32[i] == static_cast<int>(ins8[i]));
    }
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerMultiAnchor) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput = g.make_input(
            test_utils::make_tsr({20, 30, 40, 64}, datatypes::f32));
    auto fexp = g.make("exp", finput->get_outputs(), {}, {});
    auto fsum = g.make("reduce", {fexp->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1, 3}}, {"rd_op", 0},
                    {"keep_dims", true}});
    auto fadd = g.make(
            "add", {finput->get_outputs()[0], fsum->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fadd->get_outputs());
    _function_(datatypes::s32, aaa) {
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_fusion_anchor(finput->get_outputs()[0],
                        {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}});
            }
            // Anchor 1
            fusion.create_fusion_anchor(finput->get_outputs()[0],
                    {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}});
        }
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    auto ctx = get_test_ctx();
    uint32_t lanes = ctx->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::f32, {20UL, 30UL, 40UL, 64UL}),
            _arg_("inp0", datatypes::f32, {20UL, 30UL, 40UL, 64UL})) {
        _bind_(out0, inp0);
        _for_(m_o, 0, 20) {
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 30UL, 40UL, 64UL});
            _for_(n_o, 0, 30) {
                // fuse exp
                _for_(i, 0, 1) {
                    _for_(j, 0, 1) {
                        _for_(ii, 0, 40) {
                            _for_(jj, 0, 64, static_cast<int>(lanes)) {
                                builder::tensor_ptr(_fuse_buf_0,
                                        {m_o, n_o, 0, 0}, {},
                                        true)[span_t({i, j, ii, jj}, lanes)]
                                        = make_exp(tensor_ptr(inp0,
                                                {m_o, n_o, 0, 0}, {},
                                                true)[span_t(
                                                {i, j, ii, jj}, lanes)]);
                            }
                        }
                    }
                }
            }
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(_fuse_buf_0,
                                            {m_o, 0, 0, 0}, {}, true)[span_t(
                                            {i, j, ii, jj}, lanes)],
                                    reduce_sum);
                        }
                    }
                    builder::tensor_ptr(_fuse_buf_1, {m_o, 0, 0, 0}, {},
                            true)[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                }
            }
            // fuse add
            _for_(i, 0, 1) {
                _for_(j, 0, 30) {
                    _for_(ii, 0, 40) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            builder::tensor_ptr(out0, {m_o, j, ii, 0}, {},
                                    true)[span_t({i, 0, 0, jj}, lanes)]
                                    = builder::tensor_ptr(inp0, {m_o, j, ii, 0},
                                              {}, true)[span_t(
                                              {i, 0, 0, jj}, lanes)]
                                    + builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_1,
                                                    {m_o, 0, ii, 0}, {},
                                                    true)[{i, 0, 0, 0}],
                                            static_cast<int>(lanes));
                        }
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerMultiAnchorShrink) {
    sc_graph_t mgr;
    SET_THREADS_OR_SKIP(56);

    // gemm + exp + reduce fusion pattern
    auto x = graph_tensor::make({8192, 256});
    auto w = graph_tensor::make({256, 128});
    auto input = mgr.make_input({x, w});
    input->attrs_.set("constant", const_kind::local_const);

    const ops::matmul_core_config_t gemm_cfg = {32, 32, 32};
    auto gemm = mgr.make("matmul_core", input->get_outputs(), {}, {});
    gemm->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(gemm_cfg));
    auto exp = mgr.make("exp", {gemm->get_outputs()[0]}, {}, {});
    auto reduce = mgr.make("reduce", {exp->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto add = mgr.make(
            "add", {gemm->get_outputs()[0], reduce->get_outputs()[0]}, {}, {});
    mgr.make_output(add->get_outputs());
    layout_propagation(mgr, get_test_ctx());
    mixed_partition(mgr, get_test_ctx());

    auto ir_mod = lower_graph(get_test_ctx(), mgr, {});
    auto func = ir_mod->get_func(
            "outerloop_256_partition_reorder_matmul_core_exp_reduce_compute_"
            "reduce_collect_add_reorder_3");
    COMPILE_ASSERT(func, "no function got");
    tensor_shrinker_t pass;
    auto ss = pass(func->body_).checked_as<stmts>()->seq_;
    COMPILE_ASSERT(ss.size() > 1, "Unexpected stmts size found");
    auto cur_loop = ss[0].checked_as<for_loop>().get();
    while (true) {
        auto nextloop = get_inner_for_loop(cur_loop).get();
        if (!nextloop) break;
        cur_loop = nextloop;
    }
    ss = cur_loop->body_.checked_as<stmts>()->seq_;
    bool found = false;
    for (auto &s : ss) {
        if (!s.isa<define>()) break;
        auto tsr = s.checked_as<define>()->var_.checked_as<tensor>();
        bool name_eq = (tsr->name_.find("matmul_core") != std::string::npos
                && tsr->name_.find("shr") != std::string::npos);
        if (!name_eq) continue;
        int N_num_block = 128 / gemm_cfg.N_block;
        found |= (get_expr_to_dims(tsr->dims_)
                == sc_dims {
                        1, N_num_block, gemm_cfg.M_block, gemm_cfg.N_block});
    }
    EXPECT_TRUE(found);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestBinaryElementwiseOp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput0 = g.make_input(test_utils::make_tsr({100, 200}));
    auto finput1 = g.make_input(test_utils::make_tsr({100, 200}));
    auto fmin = g.make("min",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmax = g.make(
            "max", {finput0->get_outputs()[0], fmin->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fmax->get_outputs());

    EXPECT_EQ(fmax->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(
                slice_map {{finput0->get_outputs()[0].get(),
                                   {{/*dim1*/ {0, 10}, /*dim2*/ {0, 128}}}},
                        {finput1->get_outputs()[0].get(),
                                {{/*dim1*/ {0, 10}, /*dim2*/ {0, 128}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    auto ctx = get_test_ctx();
    int lanes = ctx->get_max_vector_lanes(sc_data_etype::F32);
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::s32, {100UL, 200UL}),
            _arg_("out0", datatypes::s32, {100UL, 200UL}),
            _arg_("inp1", datatypes::s32, {100UL, 200UL})) {
        _bind_(out0, inp0, inp1);
        _tensor_(tmp_out, datatypes::s32, {100UL, 200UL});
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto tout_tptr = builder::tensor_ptr(tmp_out, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 128, lanes) {
                tout_tptr[span_t({ii, jj}, lanes)]
                        = builder::make_min(inp0_tptr[span_t({ii, jj}, lanes)],
                                inp1_tptr[span_t({ii, jj}, lanes)]);
                auto out0_tptr = builder::tensor_ptr(out0, {ii, 0}, {}, true);
                inp0_tptr = builder::tensor_ptr(inp0, {ii, 0}, {}, true);
                tout_tptr = builder::tensor_ptr(tmp_out, {ii, 0}, {}, true);
                out0_tptr[span_t({0, jj}, lanes)]
                        = builder::make_max(inp0_tptr[span_t({0, jj}, lanes)],
                                tout_tptr[span_t({0, jj}, lanes)]);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestReshapeCopyOp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput0 = g.make_input(test_utils::make_tsr({100, 200}));
    auto freshape = g.make("reshape", finput0->get_outputs(), {},
            {{"shape", sc_dims {10, 10, 20, 10}}});
    auto fout = g.make_output(freshape->get_outputs());

    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(finput0->get_outputs()[0],
                {/*dim1*/ {0, 100UL}, /*dim2*/ {0, 200UL}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::s32, {10UL, 10UL, 20UL, 10UL}),
            _arg_("inp0", datatypes::s32, {100UL, 200UL})) {
        _bind_(out0, inp0);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto out0_tptr = builder::tensor_ptr(out0, {0, 0, 0, 0}, {}, true);
        _for_(ii, 0, 10 * 10 * 20 * UINT64_C(10)) {
            out0_tptr[{ii / (10 * 20 * UINT64_C(10)),
                    ii % (10 * 20 * UINT64_C(10)) / (20 * UINT64_C(10)),
                    ii % (20 * UINT64_C(10)) / UINT64_C(10), ii % UINT64_C(10)}]
                    = inp0_tptr[{ii / UINT64_C(200), ii % UINT64_C(200)}];
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestVecterizedClampOP) {
    REQUIRE_AVX2();
    auto check_clamp = [&](const int M, const int K, bool vectorized,
                               const float clamp_min, const float clamp_max) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input(test_utils::make_tsr({M, K}, datatypes::f32));
        auto fclamp = g.make("clamp", fin->get_outputs(), {},
                {{"min", clamp_min}, {"max", clamp_max}});
        auto fout = g.make_output(fclamp->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {/*dim1*/ {0, M}, /*dim2*/ {0, K}});
        }
        commit_graph_to_func(g, aaa, fusion);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -std::numeric_limits<float>::infinity();
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = std::numeric_limits<float>::infinity();

        in_buf[6] = std::numeric_limits<float>::min();
        in_buf[7] = std::numeric_limits<float>::max();

        in_buf[8] = -std::numeric_limits<float>::min();
        in_buf[9] = -std::numeric_limits<float>::max();

        fptr->call_default(out_buf.data(), in_buf.data());

        ref_clamp(
                ref_out_buf.data(), in_buf.data(), M * K, clamp_min, clamp_max);

        test_utils::compare_data(out_buf, ref_out_buf, 1e-5f, 1e-5f);
    };
    // scalar version
    // check_clamp(100, 200, false, 0.1f, 0.5f);

    // vectorization version
    check_clamp(100, 256, true, 0.1f, 0.5f);
}

// #define BENCH_CLAMP

#ifdef BENCH_CLAMP
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedClampOP) {
    auto bench_clamp = [&](const int M, const int K, bool vectorized,
                               const float clamp_min, const float clamp_max) {
        builder::ir_builder_t builder;
        fusion_anchor_mgr_t fusion;
        sc_graph_t g;
        auto fin = g.make_input();
        auto fclamp = g.make("clamp", fin->get_outputs(),
                {{"min", clamp_min}, {"max", clamp_max}});
        auto fout = g.make_output(fclamp->get_outputs());

        _function_(datatypes::void_t, aaa) {
            fusion.create_fusion_anchor(
                    fin->get_outputs()[0], {{0, M}, {0, K}});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(out_buf.data(), in_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_clamp(100, 256, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 2560, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 25600, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 512000, true, 0.1f, 0.5f);
}

#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionUnaryOp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    sc_dims arg_dims = {100, 256};
    auto finput0 = g.make_input(test_utils::make_tsr(arg_dims, datatypes::s32));
    auto finput1 = g.make_input(test_utils::make_tsr(arg_dims, datatypes::s32));
    auto frelu = g.make("relu", finput1->get_outputs(), {}, {});
    auto fadd = g.make("add",
            {finput0->get_outputs()[0], frelu->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fadd->get_outputs());

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finput0->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}},
                {finput1->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    auto lanes
            = frelu->dyn_cast<const unary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp1", datatypes::s32, {100UL, 256UL})) {
        _bind_(out0, inp0, inp1);

        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);

        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)] = builder::make_max(
                        make_expr<constant_node>(
                                (uint64_t)0, sc_data_type_t::s32(lanes)),
                        inp0_tptr[span_t({ii, jj}, lanes)]);
                auto out0_tptr = builder::tensor_ptr(out0, {ii, 0}, {}, true);
                auto inp1_tptr = builder::tensor_ptr(inp1, {ii, 0}, {}, true);
                fuse_tptr = builder::tensor_ptr(_fuse_buf, {ii, 0}, {}, true);
                out0_tptr[span_t({0, jj}, lanes)]
                        = inp1_tptr[span_t({0, jj}, lanes)]
                        + fuse_tptr[span_t({0, jj}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionBinaryOp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    sc_dims arg_dims_1 = {100, 256};
    auto finput0
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::s32));
    auto finput1
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::s32));
    auto finput2
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::s32));
    auto fsub = g.make("sub",
            {finput1->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});
    auto fadd = g.make(
            "add", {finput0->get_outputs()[0], fsub->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fadd->get_outputs());

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finput0->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}},
                {finput1->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}},
                {finput2->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    auto lanes
            = fsub->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp1", datatypes::s32, {100UL, 256UL}),
            _arg_("inp2", datatypes::s32, {100UL, 256UL})) {
        _bind_(out0, inp0, inp1, inp2);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        - inp1_tptr[span_t({ii, jj}, lanes)];
                auto out0_tptr = builder::tensor_ptr(out0, {ii, 0}, {}, true);
                auto inp2_tptr = builder::tensor_ptr(inp2, {ii, 0}, {}, true);
                fuse_tptr = builder::tensor_ptr(_fuse_buf, {ii, 0}, {}, true);
                out0_tptr[span_t({0, jj}, lanes)]
                        = inp2_tptr[span_t({0, jj}, lanes)]
                        + fuse_tptr[span_t({0, jj}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionBinaryOpWithBroadCast) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    sc_dims arg_dims_1 = {100, 256};
    sc_dims arg_dims_2 = {1, 256};
    auto finput0
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::s32));
    auto finput1
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::s32));
    auto finput2
            = g.make_input(test_utils::make_tsr(arg_dims_2, datatypes::s32));
    auto fmul = g.make("mul",
            {finput1->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});
    auto fadd = g.make(
            "add", {finput0->get_outputs()[0], fmul->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fadd->get_outputs());

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finput0->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}},
                {finput1->get_outputs()[0].get(), {{{0, 10}, {0, 256}}}},
                {finput2->get_outputs()[0].get(), {{{0, 1}, {0, 256}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    auto lanes
            = fmul->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::s32, bbb,
            _arg_("out0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp0", datatypes::s32, {100UL, 256UL}),
            _arg_("inp1", datatypes::s32, {1UL, 256UL}),
            _arg_("inp2", datatypes::s32, {100UL, 256UL})) {
        _bind_(out0, inp0, inp1, inp2);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        * builder::make_indexing(inp1_tptr, {0, jj}, lanes);
                fuse_tptr = builder::tensor_ptr(_fuse_buf, {ii, 0}, {}, true);
                auto inp2_tptr = builder::tensor_ptr(inp2, {ii, 0}, {}, true);
                auto out0_tptr = builder::tensor_ptr(out0, {ii, 0}, {}, true);
                out0_tptr[span_t({00, jj}, lanes)]
                        = inp2_tptr[span_t({00, jj}, lanes)]
                        + fuse_tptr[span_t({00, jj}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionReduceOp) {
    builder::ir_builder_t builder;
    fusion_anchor_mgr_t fusion;
    sc_graph_t g;
    auto finput0 = g.make_input(test_utils::make_tsr({16, 16}, datatypes::f32));
    sc_dims arg_dims_1 = {10, 16, 10, 16};
    auto finput1
            = g.make_input(test_utils::make_tsr(arg_dims_1, datatypes::f32));
    auto freduce = g.make("reduce", finput1->get_outputs(), {},
            {{"rd_axis", std::vector<int> {0, 2}},
                    {"rd_op", static_cast<int>(reduce_operator::add)},
                    {"keep_dims", false}});
    auto fadd = g.make("add",
            {finput0->get_outputs()[0], freduce->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fadd->get_outputs());
    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::f32, aaa) {
        fusion.create_fusion_anchor(slice_map {
                {finput0->get_outputs()[0].get(), {{{0, 16}, {0, 16}}}},
                {finput1->get_outputs()[0].get(),
                        {{{0, 10}, {0, 16}, {0, 10}, {0, 16}}}}});
        _return_(123);
    }
    commit_graph_to_func(g, aaa, fusion);

    ///// Expected func:
    auto rd_lanes = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    auto add_lanes
            = fadd->dyn_cast<const binary_elementwise_op_impl_t>()->get_lanes();
    _function_(datatypes::f32, bbb, _arg_("out0", datatypes::f32, {16UL, 16UL}),
            _arg_("inp0", datatypes::f32, {10UL, 16UL, 10UL, 16UL}),
            _arg_("inp1", datatypes::f32, {16UL, 16UL})) {
        _bind_(out0, inp0, inp1);

        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0, 0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::f32, {16UL, 16UL});
        auto fuse_buf_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(j, 0, 16) {
            _for_(jj, 0, 16, static_cast<int>(rd_lanes)) {
                _var_(reduce_value, sc_data_type_t::f32(rd_lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(rd_lanes));
                _for_(i, 0, 10) {
                    _for_(ii, 0, 10) {
                        reduce_value = builder::make_add(
                                inp0_tptr[span_t({i, j, ii, jj}, rd_lanes)],
                                reduce_value);
                    }
                }
                fuse_buf_tptr[span_t({j, jj}, rd_lanes)] = reduce_value;
                auto inp1_tptr = builder::tensor_ptr(inp1, {j, 0}, {}, true);
                auto out0_tptr = builder::tensor_ptr(out0, {j, 0}, {}, true);
                fuse_buf_tptr
                        = builder::tensor_ptr(_fuse_buf, {j, 0}, {}, true);
                out0_tptr[span_t({0, jj}, add_lanes)]
                        = inp1_tptr[span_t({0, jj}, add_lanes)]
                        + fuse_buf_tptr[span_t({0, jj}, add_lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerDeclareAndShrinkForTensorPtr) {
    sc_graph_t mgr;
    // gemm + exp + reduce fusion pattern
    auto input_A = mgr.make_input({graph_tensor::make({16, 384, 64})});
    auto input_B = mgr.make_input({graph_tensor::make({16, 64, 384})});
    auto matmul0 = mgr.make("matmul_core",
            {input_A->get_outputs()[0], input_B->get_outputs()[0]}, {}, {});
    auto cast0 = mgr.make("cast", {matmul0->get_outputs()[0]}, {},
            {{"dtype", datatypes::u8}});
    auto cast1 = mgr.make(
            "cast", {cast0->get_outputs()[0]}, {}, {{"dtype", datatypes::f32}});
    auto tv0 = mgr.make("tensor_view", {cast1->get_outputs()[0]}, {},
            {{"shape", sc_dims {16 * 384, 384}}});
    auto softmax = mgr.make("reduce", {tv0->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto output = mgr.make_output(softmax->get_outputs());
    layout_propagation(mgr, get_test_ctx());
    mixed_partition(mgr, get_test_ctx());

    auto ir_mod = lower_graph(get_test_ctx(), mgr, {});
    auto func = ir_mod->get_func(
            "outerloop_16X6_partition_matmul_core_cast_cast_tensor_view_reduce_"
            "compute_reduce_collect_reorder_3");
    ASSERT_TRUE(func);
    tensor_shrinker_t pass;
    auto newbody = pass(func->body_);
    auto ss = newbody.checked_as<stmts>()->seq_;
    ASSERT_TRUE(ss.size() > 1);
    auto cur_loop = ss[0].checked_as<for_loop>().get();
    ss = cur_loop->body_.checked_as<stmts>()->seq_;
    ASSERT_TRUE(!ss.empty());
    for (auto &s : ss) {
        if (!s.isa<define>()) break;
        auto tsr = s.as<define>()->var_.as<tensor>();
        if (tsr->name_.find("reduce_compute") != std::string::npos
                && tsr->name_.find("shr") != std::string::npos) {
            bool dim_eq = (get_expr_to_dims(tsr->dims_)
                    == sc_dims {1, 1, 1, 64, 1,
                            get_test_ctx()->get_max_vector_lanes(
                                    sc_data_etype::F32)});
            EXPECT_TRUE(dim_eq);
        }
    }
}
