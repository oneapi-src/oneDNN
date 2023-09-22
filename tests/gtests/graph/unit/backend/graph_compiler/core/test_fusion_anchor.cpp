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
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusion_anchor.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/visitor.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <ops/fusible/binary_elemwise.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static void do_commit(const sc_graph_t &g, const func_t &func,
        const fusion_anchor_mgr_t &fmgr) {
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    auto parti = std::make_shared<mixed_parti_t>(
            ctx, func, fmgr, std::make_shared<op_dep_matrix_t>(g));
    op_visitor_t visitor
            = op_visitor_t::dfs_topology_speculative_sort(g.ops_.size());
    visitor.visit_graph(
            g, [&parti](op_visitor_t *visitor, const sc_op_ptr &op) {
                if (op->isa<input_op>() || op->isa<output_op>()) return;
                if (parti->empty()) parti->buf_alloc_.allocate_buffer(op.get());
                parti->add(op);
            });
    parti->transform_to_mixed_op();
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestGroupedFusionAnchor) {
    sc_graph_t g;
    sc_dims input_dims = {20, 35, 32, 16};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});

    auto fadd = g.make("add",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmul = g.make(
            "mul", {fadd->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});

    auto fout = g.make_output(fmul->get_outputs());

    fusion_anchor_mgr_t fmgr;
    builder::ir_builder_t bld;
    _function_(datatypes::boolean, aaa) {
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 32) {
                // Grouped Anchor 0
                fmgr.create_fusion_anchor(/*group_id*/ 0,
                        slice_map {{fadd->get_inputs()[0].get(),
                                {{{m_o, 1}, {n_o, 1}, {0, 32}, {0, 16}}}}});
            }
            _for_(n_o, 0, 3) {
                // Grouped Anchor 0
                fmgr.create_fusion_anchor(/*group_id*/ 0,
                        slice_map {{fadd->get_inputs()[0].get(),
                                {{{m_o, 1}, {n_o, 1}, {0, 32}, {0, 16}}}}});
            }
        }
        _return_(true);
    }

    // commit graph to existed TIR function body with fusion anchor mgr
    do_commit(g, aaa, fmgr);

    int lanes = fadd->stc_cast<binary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::boolean, bbb,
            _arg_("mul_out", datatypes::f32, {20UL, 35UL, 32UL, 16UL}),
            _arg_("add_in_0", datatypes::f32, {20UL, 35UL, 32UL, 16UL}),
            _arg_("add_in_1", datatypes::f32, {20UL, 35UL, 32UL, 16UL}),
            _arg_("mul_in_1", datatypes::f32, {20UL, 35UL, 32UL, 16UL})) {
        _bind_(mul_out, add_in_0, add_in_1, mul_in_1);
        _for_(m_o, 0, 20) {
            // Common parent node is right position for tensor define
            _tensor_(add_out, datatypes::f32, {20UL, 35UL, 32UL, 16UL});
            _for_(n_o, 0, 32) {
                // Grouped Anchor 0
                bld.push_scope();
                {
                    auto mul_out_tptr = builder::tensor_ptr(
                            mul_out, {m_o, n_o, 0, 0}, {}, true);
                    auto add_out_tptr = builder::tensor_ptr(
                            add_out, {m_o, n_o, 0, 0}, {}, true);
                    auto add_in_0_tptr = builder::tensor_ptr(
                            add_in_0, {m_o, n_o, 0, 0}, {}, true);
                    auto add_in_1_tptr = builder::tensor_ptr(
                            add_in_1, {m_o, n_o, 0, 0}, {}, true);
                    auto mul_in_1_tptr = builder::tensor_ptr(
                            mul_in_1, {m_o, n_o, 0, 0}, {}, true);

                    _for_(ii, 0, 32) {
                        _for_(jj, 0, 16, lanes) {
                            add_out_tptr[span_t({0, 0, ii, jj}, lanes)]
                                    = add_in_0_tptr[span_t(
                                              {0, 0, ii, jj}, lanes)]
                                    + add_in_1_tptr[span_t(
                                            {0, 0, ii, jj}, lanes)];
                        }
                    }

                    _for_(ii, 0, 32) {
                        _for_(jj, 0, 16, lanes) {
                            mul_out_tptr[span_t({0, 0, ii, jj}, lanes)]
                                    = add_out_tptr[span_t(
                                              {0, 0, ii, jj}, lanes)]
                                    * mul_in_1_tptr[span_t(
                                            {0, 0, ii, jj}, lanes)];
                        }
                    }
                }
                bld.emit(bld.pop_scope());
            }
            _for_(n_o, 0, 3) {
                // Grouped Anchor 0
                bld.push_scope();
                {
                    auto mul_out_tptr = builder::tensor_ptr(
                            mul_out, {m_o, n_o, 0, 0}, {}, true);
                    auto add_out_tptr = builder::tensor_ptr(
                            add_out, {m_o, n_o, 0, 0}, {}, true);
                    auto add_in_0_tptr = builder::tensor_ptr(
                            add_in_0, {m_o, n_o, 0, 0}, {}, true);
                    auto add_in_1_tptr = builder::tensor_ptr(
                            add_in_1, {m_o, n_o, 0, 0}, {}, true);
                    auto mul_in_1_tptr = builder::tensor_ptr(
                            mul_in_1, {m_o, n_o, 0, 0}, {}, true);

                    _for_(ii, 0, 32) {
                        _for_(jj, 0, 16, lanes) {
                            add_out_tptr[span_t({0, 0, ii, jj}, lanes)]
                                    = add_in_0_tptr[span_t(
                                              {0, 0, ii, jj}, lanes)]
                                    + add_in_1_tptr[span_t(
                                            {0, 0, ii, jj}, lanes)];
                        }
                    }

                    _for_(ii, 0, 32) {
                        _for_(jj, 0, 16, lanes) {
                            mul_out_tptr[span_t({0, 0, ii, jj}, lanes)]
                                    = add_out_tptr[span_t(
                                              {0, 0, ii, jj}, lanes)]
                                    * mul_in_1_tptr[span_t(
                                            {0, 0, ii, jj}, lanes)];
                        }
                    }
                }
                bld.emit(bld.pop_scope());
            }
        }
        _return_(true);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestDynamicFusionAnchor) {
    sc_graph_t g;
    sc_dims input_dims = {20, 35, 16};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});

    auto fadd = g.make("add",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmul = g.make(
            "mul", {fadd->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});

    auto fout = g.make_output(fmul->get_outputs());

    fusion_anchor_mgr_t fmgr;
    builder::ir_builder_t bld;
    _function_(datatypes::boolean, aaa) {
        _for_(m_o, 0, 20) {
            // dynamic shape according to `m_o` value
            _var_(dyn_shape, sc_data_type_t::s32());
            dyn_shape = builder::make_select(m_o > 10, expr(32), expr(3));
            // Dynamic Anchor with dynamic slice shape
            fmgr.create_fusion_anchor(slice_map {{fadd->get_inputs()[0].get(),
                    {{{m_o, 1}, {0, dyn_shape}, {0, 16}}}}});
        }
        _return_(true);
    }

    // commit graph to existed TIR function body with fusion anchor mgr
    do_commit(g, aaa, fmgr);

    int lanes = fadd->stc_cast<binary_elementwise_op_impl_t>()->get_lanes();

    _function_(datatypes::boolean, bbb,
            _arg_("mul_out", datatypes::f32, {20UL, 35UL, 16UL}),
            _arg_("add_in_0", datatypes::f32, {20UL, 35UL, 16UL}),
            _arg_("add_in_1", datatypes::f32, {20UL, 35UL, 16UL}),
            _arg_("mul_in_1", datatypes::f32, {20UL, 35UL, 16UL})) {
        _bind_(mul_out, add_in_0, add_in_1, mul_in_1);
        _for_(m_o, 0, 20) {
            _var_(dyn_shape, sc_data_type_t::s32());
            _tensor_(add_out, datatypes::f32, {20UL, 35UL, 16UL});
            dyn_shape = builder::make_select(m_o > 10, expr(32), expr(3));
            bld.push_scope();
            {
                auto mul_out_tptr
                        = builder::tensor_ptr(mul_out, {m_o, 0, 0}, {}, true);
                auto add_out_tptr
                        = builder::tensor_ptr(add_out, {m_o, 0, 0}, {}, true);
                auto add_in_0_tptr
                        = builder::tensor_ptr(add_in_0, {m_o, 0, 0}, {}, true);
                auto add_in_1_tptr
                        = builder::tensor_ptr(add_in_1, {m_o, 0, 0}, {}, true);
                auto mul_in_1_tptr
                        = builder::tensor_ptr(mul_in_1, {m_o, 0, 0}, {}, true);
                _for_(j, 0, dyn_shape) {
                    _for_(k, 0, 16, lanes) {
                        add_out_tptr[span_t({0, j, k}, static_cast<int>(lanes))]
                                = add_in_0_tptr[span_t({0, j, k}, lanes)]
                                + add_in_1_tptr[span_t({0, j, k}, lanes)];
                    }
                }
                _for_(j, 0, dyn_shape) {
                    _for_(k, 0, 16, lanes) {
                        mul_out_tptr[span_t({0, j, k}, static_cast<int>(lanes))]
                                = add_out_tptr[span_t({0, j, k}, lanes)]
                                * mul_in_1_tptr[span_t({0, j, k}, lanes)];
                    }
                }
            }
            bld.emit(bld.pop_scope());
        }
        _return_(true);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestSplitGroupedFusionAnchor1) {
    SET_THREADS_OR_SKIP(20);
    sc_graph_t g;
    // The last dim 55 could not be divided by max lanes, as the result, it
    // should be auto split into grouped anchor in avoid of loop merge break
    sc_dims input_dims = {20, 30, 55};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});

    auto fadd = g.make("add",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmul = g.make(
            "mul", {fadd->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});

    auto fout = g.make_output(fmul->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    do_mixed_partition(ctx, g);

    auto mixed_op = get_mixed_op_from_graph(g);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto parti = mixed_op->parti_list_[0];
    ASSERT_TRUE(parti->fanchors_.size() == 1);
    auto fanchor = parti->fanchors_[0];
    // grouped anchor is expected
    auto grouped_anchor = fanchor->dyn_cast<fuse_grouped_anchor_map_t>();
    ASSERT_TRUE(grouped_anchor);
    // own two group size
    ASSERT_TRUE(
            grouped_anchor->anchor_position_.checked_as<stmts>()->seq_.size()
            == 2);
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestSplitGroupedFusionAnchor2) {
    SET_THREADS_OR_SKIP(20);
    sc_graph_t g;
    // The last dim 55 could not be divided by max lanes, as the result, it
    // should be auto split into grouped anchor in avoid of loop merge break
    sc_dims input_dims = {20, 30, 55};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});

    auto fadd = g.make("add",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmul = g.make(
            "mul", {fadd->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});
    // However, this reduce op would break split anchor optimization, which may
    // drive reduce op to select larger fusion anchor than non-optimized version
    auto freduce = g.make("reduce", fmul->get_outputs(), {},
            {{"rd_axis", std::vector<int> {2}}, {"rd_op", 0}});
    auto fout = g.make_output(freduce->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    do_mixed_partition(ctx, g);

    auto mixed_op = get_mixed_op_from_graph(g);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto parti = mixed_op->parti_list_[0];
    ASSERT_TRUE(parti->fanchors_.size() == 1);
    auto fanchor = parti->fanchors_[0];
    // grouped anchor is not expected
    auto grouped_anchor = fanchor->dyn_cast<fuse_grouped_anchor_map_t>();
    ASSERT_FALSE(grouped_anchor);
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestSplitGroupedFusionAnchor3) {
    SET_THREADS_OR_SKIP(20);
    sc_graph_t g;
    // The last dim 55 could not be divided by max lanes, as the result, it
    // should be auto split into grouped anchor in avoid of loop merge break
    sc_dims input_dims = {20, 30, 55};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});

    auto frelu0 = g.make("relu", {finput0->get_outputs()[0]}, {}, {});
    auto frelu1 = g.make("relu", {finput1->get_outputs()[0]}, {}, {});
    auto fadd = g.make("add",
            {frelu1->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});
    auto fmul = g.make(
            "mul", {frelu0->get_outputs()[0], fadd->get_outputs()[0]}, {}, {});
    auto fout = g.make_output(fmul->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    do_mixed_partition(ctx, g);

    auto mixed_op = get_mixed_op_from_graph(g);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto parti = mixed_op->parti_list_[0];
    // No grouped anchor is expected explicitly
    ASSERT_TRUE(std::all_of(parti->fanchors_.begin(), parti->fanchors_.end(),
            [](const fuse_anchor_map_ptr &fanchor) {
                return !fanchor->isa<fuse_grouped_anchor_map_t>();
            }));
    // check tensor shrink info for fadd
    auto body
            = parti->get_outer_loops().back()->body_.checked_as<stmts>()->seq_;
    EXPECT_TRUE(std::any_of(body.begin(), body.end(), [](const stmt &s) {
        return s.cast<define>()
                .map([](const define &d) { return d->var_.as<tensor>(); })
                .filter([](const tensor &t) {
                    // check output buffer of add whether has tensor shrink attr
                    return t->name_.find("add") != std::string::npos
                            && t->attr().has_key(
                                    tensor_shrinker_attrs::should_shrink);
                })
                .has_value();
    }));
}

TEST(GCCore_CPU_fusion_anchor_cpp, TestSplitGroupedFusionAnchor4) {
    SET_THREADS_OR_SKIP(20);
    sc_graph_t g;
    // The last dim 55 could not be divided by max lanes, as the result, it
    // should be auto split into grouped anchor in avoid of loop merge break
    sc_dims input_dims = {20, 30, 55};
    auto finput0 = g.make_input({graph_tensor::make(input_dims)});
    auto finput1 = g.make_input({graph_tensor::make(input_dims)});
    auto finput2 = g.make_input({graph_tensor::make(input_dims)});
    auto finput3 = g.make_input({graph_tensor::make(input_dims)});

    auto fadd0 = g.make("add",
            {finput0->get_outputs()[0], finput1->get_outputs()[0]}, {}, {});
    auto fmul0 = g.make("mul",
            {fadd0->get_outputs()[0], finput2->get_outputs()[0]}, {}, {});
    auto frelu0 = g.make("relu", finput3->get_outputs(), {}, {});
    auto fadd1 = g.make(
            "add", {frelu0->get_outputs()[0], fmul0->get_outputs()[0]}, {}, {});
    auto frelu1 = g.make("relu", fadd1->get_outputs(), {}, {});

    auto fout = g.make_output(frelu1->get_outputs());

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = true;
    ctx->flags_.use_cost_model_ = true;
    do_mixed_partition(ctx, g);

    auto mixed_op = get_mixed_op_from_graph(g);
    ASSERT_TRUE(mixed_op && mixed_op->parti_list_.size() == 1);
    auto parti = mixed_op->parti_list_[0];
    ASSERT_TRUE(parti->fanchors_.size() == 1);
    auto fanchor = parti->fanchors_[0];
    // implicit grouped anchor is expected
    ASSERT_TRUE(std::any_of(fanchor->fsmap_.datamap_.begin(),
            fanchor->fsmap_.datamap_.end(),
            [](const std::pair<graph_tensor *, slice_range_list> &kv) {
                return kv.second.size() > 1;
            }));
}
