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

#include <functional>
#include <iostream>
#include <vector>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/viewer.hpp>
#include <util/any_map.hpp>
using namespace dnnl::impl::graph::gc;

#include "gtest/gtest.h"

// test 3 functionalities
// 1. whether dispatch() for a base case can be dispatched to a sub-class
// 2. whether default visit() calls dispatch() for each of the field
// 3. when a sub-node is changed, whether the parent node calls remake()?

struct {
    std::vector<expr_c> exprlist;
    std::vector<stmt_c> stmtlist;
    func_t f;
    void reset() {
        exprlist.clear();
        stmtlist.clear();
        f = nullptr;
    }
} history;

ir_comparer cmper;

using expr_v = std::vector<expr_c>;

namespace std {
bool operator==(const std::vector<expr_c> &a, const std::vector<expr_c> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (!a.at(i).ptr_same(b.at(i))) return false;
    }
    return true;
}
} // namespace std

/**
 * Check the visitor/inplace visitor/viewer with the input.
 * T should be expr or stmt. The leaf nodes of the inputshould be
 * constants. It will check that:
 *  if the output after dispatch() is ptr_same of original
 *  if the output equals original
 *  if inplace visitor's changed_ field is "should_inplace_change"
 *  if the leaf nodes are the same of expected
 *
 * The "expected" functor takes an input node and generates a sequence
 * of leaf nodes that are expected
 * */
template <typename T, typename TVisitor, typename TIPVisitor, typename TViewer>
void type_check(T input, bool should_inplace_change,
        std::function<expr_v(T)> expected) {
    {
        T tmp = input->remake();
        auto expected_v = expected(tmp);
        TVisitor vis;
        auto ret = vis.dispatch(tmp);
        EXPECT_TRUE(cmper.compare(ret, input));
        EXPECT_FALSE(ret.ptr_same(tmp));
        EXPECT_EQ(history.exprlist, expected_v);
        history.reset();
    }
    {
        T tmp = input->remake();
        auto expected_v = expected(tmp);
        TIPVisitor vis;
        auto ret = vis.dispatch_impl(tmp);
        EXPECT_EQ(vis.changed_, should_inplace_change);
        EXPECT_TRUE(ret.ptr_same(tmp));
        EXPECT_TRUE(cmper.compare(ret, input));
        EXPECT_EQ(history.exprlist, expected_v);
        history.reset();
    }
    {
        T tmp = input->remake();
        TViewer vis;
        EXPECT_TRUE(vis.dispatch(tmp).ptr_same(tmp));
        EXPECT_EQ(history.exprlist, expected(tmp));
        history.reset();
    }
}

template <typename T>
void simple_type_check(expr input) {
    using T_c = decltype(T().to_const());
    class visitor_t : public ir_visitor_t {
        expr_c visit(T_c c) override {
            history.exprlist.push_back(c);
            return c.remove_const()->remake();
        }
    };

    class ipvisitor_t : public ir_inplace_visitor_t {
        expr visit_impl(T c) override {
            history.exprlist.push_back(c);
            return c;
        }
    };

    class viewer_t : public ir_viewer_t {
        void view(T_c c) override { history.exprlist.push_back(c); }
    };

    type_check<expr, visitor_t, ipvisitor_t, viewer_t>(
            input, false, [](const expr &e) { return expr_v {e}; });
}

class visitor_test_base_t : public ir_visitor_t {
public:
    using ir_visitor_t::visit;
    expr_c visit(constant_c c) override {
        history.exprlist.emplace_back(c);
        return c.remove_const()->remake();
    }
    expr_c visit(var_c c) override {
        history.exprlist.emplace_back(c);
        return c.remove_const()->remake();
    }
};

class ip_visitor_test_base_t : public ir_inplace_visitor_t {
public:
    using ir_inplace_visitor_t::visit_impl;
    expr visit_impl(constant c) override {
        history.exprlist.emplace_back(c);
        return c->remake();
    }
    expr visit_impl(var c) override {
        history.exprlist.emplace_back(c);
        return c->remake();
    }
};

class viewer_test_base_t : public ir_viewer_t {
public:
    void view(constant_c c) override { history.exprlist.emplace_back(c); }
    void view(var_c c) override { history.exprlist.emplace_back(c); }
};

TEST(GCCore_CPU_visitor_cpp, TestVisitorConst) {
    simple_type_check<constant>(expr(1));
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorVar) {
    simple_type_check<var>(builder::make_var(datatypes::bf16, "a"));
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorCast) {
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(builder::make_cast(datatypes::f16, 32), true,
            [](const expr &e) { return expr_v {e.as<cast>()->in_}; });
}

// if we override base and return non-empty, check if the sub-class is never
// called
template <typename Base, typename Derived>
void check_visit_in_inherited(expr input) {
    using Derived_c = decltype(Derived().to_const());
    class visitor_t : public visitor_test_base_t {
    public:
        using visitor_test_base_t::visit;
        expr_c visit(Derived_c c) override {
            history.exprlist.push_back(c);
            return c.remove_const()->remake();
        }
    };

    class ip_visitor_t : public ip_visitor_test_base_t {
    public:
        using ip_visitor_test_base_t::visit_impl;
        expr visit_impl(Derived c) override {
            history.exprlist.push_back(c);
            return c;
        }
    };

    class viewer_t : public viewer_test_base_t {
    public:
        using viewer_test_base_t::view;
        void view(Derived_c c) override { history.exprlist.push_back(c); }
    };
    type_check<expr, visitor_t, ip_visitor_t, viewer_t>(
            input, false, [](const expr &e) { return expr_v {e}; });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorBinary) {
    check_visit_in_inherited<binary, add>(expr(12) + expr(13));
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorCmp) {
    check_visit_in_inherited<cmp, cmp_eq>(expr(12) == expr(13));
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorLogic) {
    check_visit_in_inherited<logic, logic_and>(expr(false) && expr(false));
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorBinaries) {
    expr v1 = 12;
    expr v2 = 13;
    auto check_op_bin = [&](expr (*op)(const expr_c &, const expr_c &)) {
        type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
                viewer_test_base_t>(op(v1, v2), true, [](const expr &e) {
            return expr_v {
                    e.static_as<binary>()->l_, e.static_as<binary>()->r_};
        });
    };
    check_op_bin(builder::make_add);
    check_op_bin(builder::make_sub);
    check_op_bin(builder::make_mul);
    check_op_bin(builder::make_div);
    check_op_bin(builder::make_mod);

    auto check_op_cmp = [&](expr (*op)(const expr_c &, const expr_c &)) {
        type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
                viewer_test_base_t>(op(v1, v2), true, [](const expr &e) {
            return expr_v {e.static_as<cmp>()->l_, e.static_as<cmp>()->r_};
        });
    };

    check_op_cmp(builder::make_cmp_lt);
    check_op_cmp(builder::make_cmp_le);
    check_op_cmp(builder::make_cmp_gt);
    check_op_cmp(builder::make_cmp_ge);
    check_op_cmp(builder::make_cmp_eq);
    check_op_cmp(builder::make_cmp_ne);

    auto check_op_logic = [&](expr (*op)(const expr_c &, const expr_c &)) {
        type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
                viewer_test_base_t>(op(v1, v2), true, [](const expr &e) {
            return expr_v {e.static_as<logic>()->l_, e.static_as<logic>()->r_};
        });
    };

    check_op_logic(builder::make_logic_and);
    check_op_logic(builder::make_logic_or);
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorLogicNot) {
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(!expr(false), true,
            [](const expr &e) { return expr_v {e.as<logic_not>()->in_}; });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorCondition) {
    namespace gc = dnnl::impl::graph::gc;
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(
            builder::make_select(true, 1, 0), true, [](const expr &e) {
                return expr_v {e.static_as<gc::select>()->cond_,
                        e.static_as<gc::select>()->l_,
                        e.static_as<gc::select>()->r_};
            });
    auto cond_l = builder::make_tensor("cond_l", {16}, datatypes::f32);
    auto cond_r = builder::make_tensor("cond_r", {16}, datatypes::f32);
    auto l = builder::make_tensor("l", {16}, datatypes::f32);
    auto r = builder::make_tensor("r", {16}, datatypes::f32);
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(builder::make_select(cond_l[span_t({0}, 16)]
                                                < cond_r[span_t({0}, 16)],
                                        l[span_t({0}, 16)], r[span_t({0}, 16)]),
            false, [](const expr &e) {
                auto cond
                        = e.static_as<gc::select>()->cond_.static_as<cmp_lt>();
                auto cl = cond->l_.static_as<indexing>();
                auto cl_t = cl->ptr_.static_as<tensor>();
                auto cr = cond->r_.static_as<indexing>();
                auto cr_t = cr->ptr_.static_as<tensor>();
                auto rl = e.static_as<gc::select>()->l_.static_as<indexing>();
                auto rl_t = rl->ptr_.static_as<tensor>();
                auto rr = e.static_as<gc::select>()->r_.static_as<indexing>();
                auto rr_t = rr->ptr_.static_as<tensor>();
                return expr_v {cl_t->dims_[0], cl_t->strides_[0], cl->idx_[0],
                        cr_t->dims_[0], cr_t->strides_[0], cr->idx_[0],
                        rl_t->dims_[0], rl_t->strides_[0], rl->idx_[0],
                        rr_t->dims_[0], rr_t->strides_[0], rr->idx_[0]};
            });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorCall) {
    builder::ir_builder_t b;
    _decl_func_(datatypes::f32, aaaa, _arg_("c1", datatypes::s32),
            _arg_("c2", datatypes::s32));
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(
            aaaa(expr(1), expr(2)), true, [](const expr &e) {
                auto call_e = e.as<call>();
                return expr_v {call_e->args_[0], call_e->args_[1]};
            });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorIntrinCall) {
    builder::ir_builder_t b;
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(
            make_expr<intrin_call_node>(intrin_type::max,
                    std::vector<expr> {expr(1), 2}, any_map_t()),
            true, [](const expr &e) {
                auto call_e = e.as<intrin_call>();
                return expr_v {call_e->args_[0], call_e->args_[1]};
            });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorIndexing) {
    auto tsr = builder::make_stensor(
            "AAA", std::vector<expr> {100, 200}, {200, 1}, datatypes::s32);
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(tsr[{10, 20}], true, [](const expr &e) {
        auto call_e = e.as<indexing>();
        auto t = call_e->ptr_.as<tensor>();
        return expr_v {t->dims_[0], t->dims_[1], t->strides_[0], t->strides_[1],
                call_e->idx_[0], call_e->idx_[1]};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorTensorPtr) {
    auto tsr = builder::make_stensor(
            "AAA", std::vector<expr> {100, 200}, {200, 1}, datatypes::s32);
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(
            builder::tensor_ptr(tsr, {10, 20}), false, [](const expr &e) {
                auto call_e = e.as<tensorptr>()->base_.as<indexing>();
                auto t = call_e->ptr_.as<tensor>();
                return expr_v {t->dims_[0], t->dims_[1], t->strides_[0],
                        t->strides_[1], call_e->idx_[0], call_e->idx_[1]};
            });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorTensor) {
    auto tsr = builder::make_stensor("AAA", std::vector<expr> {100, 200},
            {200, 1}, datatypes::s32, address_space::device);
    type_check<expr, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(tsr, true, [](const expr &e) {
        auto t = e.as<tensor>();
        return expr_v {
                t->dims_[0], t->dims_[1], t->strides_[0], t->strides_[1]};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorAssign) {
    auto tsr = builder::make_stensor(
            "AAA", std::vector<expr> {100, 200}, {200, 1}, datatypes::s32);
    assign asn = make_stmt<assign_node_t>(tsr[{10, 20}].get(), expr(123));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true, [](const stmt &e) {
        auto asn = e.as<assign>();
        auto call_e = asn->var_.as<indexing>();
        auto t = call_e->ptr_.as<tensor>();
        return expr_v {t->dims_[0], t->dims_[1], t->strides_[0], t->strides_[1],
                call_e->idx_[0], call_e->idx_[1], asn->value_};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorStmts) {
    std::vector<stmt> s {make_stmt<evaluate_node_t>(expr(1)),
            make_stmt<evaluate_node_t>(expr(2)),
            make_stmt<evaluate_node_t>(expr(3))};
    stmt asn = make_stmt<stmts_node_t>(std::move(s));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true, [](const stmt &e) {
        auto asn = e.as<stmts>();
        return expr_v {asn->seq_[0].as<evaluate>()->value_,
                asn->seq_[1].as<evaluate>()->value_,
                asn->seq_[2].as<evaluate>()->value_};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorEval) {
    stmt asn = make_stmt<evaluate_node_t>(expr(1));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true,
            [](const stmt &e) { return expr_v {e.as<evaluate>()->value_}; });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorReturn) {
    stmt asn = make_stmt<returns_node_t>(expr(1));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true,
            [](const stmt &e) { return expr_v {e.as<returns>()->value_}; });

    asn = make_stmt<returns_node_t>(expr());
    auto ret = visitor_test_base_t().dispatch(asn);
    EXPECT_TRUE(cmper.compare(ret, asn));
    EXPECT_TRUE(ret.ptr_same(asn));
    EXPECT_TRUE(history.exprlist.empty());
    ret = ip_visitor_test_base_t().dispatch_impl(asn);
    EXPECT_TRUE(cmper.compare(ret, asn));
    EXPECT_TRUE(ret.ptr_same(asn));
    EXPECT_TRUE(history.exprlist.empty());
    ret = viewer_test_base_t().dispatch(asn);
    EXPECT_TRUE(cmper.compare(ret, asn));
    EXPECT_TRUE(ret.ptr_same(asn));
    EXPECT_TRUE(history.exprlist.empty());
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorVarTensorDef) {
    stmt asn = builder::make_var_tensor_def_unattached(
            builder::make_var(datatypes::f32, "a"));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true,
            [](const stmt &e) { return expr_v {e.as<define>()->var_}; });

    asn = builder::make_var_tensor_def_unattached(
            builder::make_var(datatypes::f32, "a"), linkage::local, expr(1.0f));
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(asn, true, [](const stmt &e) {
        auto def = e.as<define>();
        return expr_v {def->init_, def->var_};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorIfElse) {
    builder::ir_builder_t bui;
    bui.push_scope();
    _if_(expr(false)) { bui.push_evaluate(12); }
    _if_(expr(true)) { bui.push_evaluate(14); }
    _else_ { bui.push_evaluate(15); }
    auto s = bui.pop_scope();
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(s, true, [](const stmt &e) {
        expr_v ret;
        stmts lst = e.as<stmts>();
        auto ifelse1 = lst->seq_[0].as<if_else>();
        ret.push_back(ifelse1->condition_);
        ret.push_back(ifelse1->then_case_.as<stmts>()
                              ->seq_[0]
                              .as<evaluate>()
                              ->value_);
        ifelse1 = lst->seq_[1].as<if_else>();
        ret.push_back(ifelse1->condition_);
        ret.push_back(ifelse1->then_case_.as<stmts>()
                              ->seq_[0]
                              .as<evaluate>()
                              ->value_);
        ret.push_back(ifelse1->else_case_.as<stmts>()
                              ->seq_[0]
                              .as<evaluate>()
                              ->value_);
        return ret;
    });
}

TEST(GCCore_CPU_visitor_cpp, TestVisitorFor) {
    builder::ir_builder_t bui;
    bui.push_scope();
    _for_(i, 0, 100, 1) { bui.push_evaluate(123); }
    auto s = bui.pop_scope();
    type_check<stmt, visitor_test_base_t, ip_visitor_test_base_t,
            viewer_test_base_t>(s, true, [](const stmt &e) {
        auto loop = e.as<stmts>()->seq_[0].as<for_loop>();
        return expr_v {loop->var_, loop->iter_begin_, loop->iter_end_,
                loop->step_,
                loop->body_.as<stmts>()->seq_[0].as<evaluate>()->value_};
    });
}

TEST(GCCore_CPU_visitor_cpp, TestConsistentVisitor) {
    builder::ir_builder_t bui;
    bui.push_scope();
    _var_(AAA, datatypes::f32);
    _tensor_(BBB, datatypes::f32, {100});
    bui.push_assign(AAA, 1.2f);
    BBB[10] = 1.2f;
    auto s = bui.pop_scope().as<stmts>();
    // save the old pointers
    expr AAA2 = AAA;
    expr BBB2 = BBB;
    class vis_t : public ir_consistent_visitor_t {
    public:
        bool change_tensor = false;
        bool change_var = false;
        expr_c visit(tensor_c f) override {
            if (change_tensor && f->name_ == "BBB") {
                return builder::make_tensor("BBB2", {200}, datatypes::f32);
            }
            return std::move(f);
        }

        expr_c visit(var_c f) override {
            if (change_var && f->name_ == "AAA") {
                return builder::make_var(datatypes::f32, "AAA2");
            }
            return std::move(f);
        }
    } v;
    v.dispatch(s);
    EXPECT_TRUE(s->seq_[0].checked_as<define>()->var_.ptr_same(AAA2));
    EXPECT_TRUE(s->seq_[1].checked_as<define>()->var_.ptr_same(BBB2));
    EXPECT_TRUE(s->seq_[2].checked_as<assign>()->var_.ptr_same(AAA2));
    EXPECT_TRUE(s->seq_[3]
                        .checked_as<assign>()
                        ->var_.checked_as<indexing>()
                        ->ptr_.ptr_same(BBB2));

    v.change_tensor = true;
    s = v.dispatch(s).checked_as<stmts>();
    EXPECT_TRUE(s->seq_[0].checked_as<define>()->var_.ptr_same(AAA2));
    // BBB is changed to BBB2
    EXPECT_FALSE(s->seq_[1].checked_as<define>()->var_.ptr_same(BBB2));
    BBB2 = s->seq_[1].checked_as<define>()->var_;
    EXPECT_TRUE(BBB2.isa<tensor>());
    EXPECT_TRUE(s->seq_[2].checked_as<assign>()->var_.ptr_same(AAA2));
    // The assignment using BBB should be changed to BBB2
    EXPECT_TRUE(s->seq_[3]
                        .checked_as<assign>()
                        ->var_.checked_as<indexing>()
                        ->ptr_.ptr_same(BBB2));

    v.change_tensor = false;
    v.change_var = true;
    s = v.dispatch(s).checked_as<stmts>();
    EXPECT_FALSE(s->seq_[0].checked_as<define>()->var_.ptr_same(AAA2));
    AAA2 = s->seq_[0].checked_as<define>()->var_;
    EXPECT_TRUE(AAA2.isa<var>());
    EXPECT_TRUE(s->seq_[1].checked_as<define>()->var_.ptr_same(BBB2));
    EXPECT_TRUE(s->seq_[2].checked_as<assign>()->var_.ptr_same(AAA2));
    EXPECT_TRUE(s->seq_[3]
                        .checked_as<assign>()
                        ->var_.checked_as<indexing>()
                        ->ptr_.ptr_same(BBB2));
}
