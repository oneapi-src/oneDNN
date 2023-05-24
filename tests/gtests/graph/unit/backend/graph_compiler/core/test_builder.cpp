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

#include <iostream>
#include <compiler/config/context.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass/validator.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/index_flatten.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
TEST(GCCore_CPU_ir_builder_cpp, TestIRBuilder) {
    ir_builder_t builder;
    auto callee = make_func("some_func", {}, stmt(), datatypes::f32);
    builder.push_scope();
    {
        auto i = make_var(datatypes::s32, "i");
        auto buf = make_tensor("buf", {100}, datatypes::s32);
        builder.push_evaluate(buf);
        builder.push_scope();
        {
            auto ptr = make_indexing(buf, i);
            builder.push_assign(ptr, 23);
            builder.push_assign(ptr, 24);
        }
        auto body = builder.pop_scope();
        builder.push_for_loop(i, 10, 20, 1, body, true, for_type::NORMAL);

        builder.push_scope();
        {
            auto ptr = make_indexing(buf, 1);
            builder.push_assign(ptr, 23);
        }
        auto true_block = builder.pop_scope();

        builder.push_scope();
        {
            auto ptr = make_indexing(buf, 1);
            builder.push_assign(ptr, 100);
            builder.push_evaluate(callee(1.2f));
        }
        auto false_block = builder.pop_scope();

        builder.push_if_else(
                make_cmp_eq(make_indexing(buf, 1), 0), true_block, false_block);
        builder.push_returns();
    }
    auto body = builder.pop_scope();

    {
        std::vector<stmt> expected;
        auto i = make_expr<var_node>(datatypes::s32, "i");
        auto buf = make_expr<tensor_node>(datatypes::s32, "buf",
                std::vector<expr> {make_expr<constant_node>(INT64_C(100))});
        expected.emplace_back(make_stmt<evaluate_node_t>(buf));
        stmts for_body;
        {
            std::vector<stmt> for_body_s;
            auto ptr = make_expr<indexing_node>(
                    buf, std::vector<expr> {i}, expr());
            for_body_s.emplace_back(make_stmt<assign_node_t>(
                    ptr, make_expr<constant_node>(INT64_C(23))));
            for_body_s.emplace_back(make_stmt<assign_node_t>(
                    ptr, make_expr<constant_node>(INT64_C(24))));
            for_body = make_stmt<stmts_node_t>(std::move(for_body_s));
        }
        expected.emplace_back(make_stmt<for_loop_node_t>(i,
                make_expr<constant_node>(INT64_C(10)),
                make_expr<constant_node>(INT64_C(20)),
                make_expr<constant_node>(INT64_C(1)), for_body, true,
                for_type::NORMAL));

        stmts true_block;
        {
            auto ptr = make_expr<indexing_node>(buf,
                    std::vector<expr> {make_expr<constant_node>(INT64_C(1))},
                    expr());
            true_block = make_stmt<stmts_node_t>(
                    std::vector<stmt> {make_stmt<assign_node_t>(
                            ptr, make_expr<constant_node>(INT64_C(23)))});
        }

        stmts false_block;
        {
            auto ptr = make_expr<indexing_node>(buf,
                    std::vector<expr> {make_expr<constant_node>(INT64_C(1))},
                    expr());
            false_block = make_stmt<stmts_node_t>(std::vector<stmt> {
                    make_stmt<assign_node_t>(
                            ptr, make_expr<constant_node>(INT64_C(100))),
                    make_stmt<evaluate_node_t>(make_expr<call_node>(callee,
                            std::vector<expr> {
                                    make_expr<constant_node>(1.2f)}))});
        }
        expected.emplace_back(make_stmt<if_else_node_t>(
                make_expr<cmp_eq_node>(
                        make_expr<indexing_node>(buf,
                                std::vector<expr> {
                                        make_expr<constant_node>(INT64_C(1))},
                                expr()),
                        make_expr<constant_node>(INT64_C(0))),
                true_block, false_block));
        expected.emplace_back(make_stmt<returns_node_t>(expr()));
        stmts expected_stmt = make_stmt<stmts_node_t>(std::move(expected));
        ir_comparer cmper;
        EXPECT_TRUE(expected_stmt->equals(body, cmper));
    }
}

TEST(GCCore_CPU_ir_builder_cpp, TestEasyBuilder) {
    builder::ir_builder_t builder;
    for_loop loop;
    _function_(datatypes::f32, aaa, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32),
            _arg_("buf", datatypes::s32, {100, 200})) {
        _bind_(ii, jj, buf);
        _tensor_(buf2, datatypes::s32, {100, 200});
        _tensor_(buf3, datatypes::s32, {100, 200});
        _tensor_(buf4, datatypes::s32, {100, 200, 300, 15});
        _var_(v1, datatypes::f32);
        _var_(v2, datatypes::f32);
        buf3[{ii, 1}] = buf3[{jj, 2}];
        _named_for_(loop, i, 0, 20, 1, for_type::PARALLEL) {
            _for_(j, 0, 50) {
                _for_(k, 0, 30) {
                    buf3[{i, j}] = buf3[{i, j}] + buf[{i, k}] * buf2[{k, j}];
                    buf3[{i, j % UINT64_C(3)}] = 3;
                    _if_(buf3[{i, j}] == 0) { buf3[{i, j}] = 3; }
                    _else_ _if_(buf3[{i, j}] == 2) { buf3[{i, j}] = 4; }
                    _else_ {
                        buf3[{k, j}] = 4;
                        buf4[{i, j, k, 3}] = 12;
                        buf3[span_t({i, j}, 16)] = buf3[span_t({i, j}, 16)];
                    }
                }
            }
        }
        _return_(1.2f);
    }

    // generate expected
    auto ii = make_var(datatypes::s32, "ii");
    auto jj = make_var(datatypes::s32, "jj");
    auto buf = make_tensor("buf", {100, 200}, datatypes::s32);
    builder.push_scope();
    {
        auto buf2 = make_tensor("buf2", {100, 200}, datatypes::s32);
        builder.push_evaluate(buf2);
        auto buf3 = make_tensor("buf3", {100, 200}, datatypes::s32);
        builder.push_evaluate(buf3);
        auto buf4 = make_tensor("buf4", {100, 200, 300, 15}, datatypes::s32);
        builder.push_evaluate(buf4);
        auto v1 = make_var(datatypes::f32, "v1");
        builder.push_evaluate(v1);
        auto v2 = make_var(datatypes::f32, "v2");
        builder.push_evaluate(v2);
        builder.push_assign(
                make_indexing(buf3, {ii, 1}), make_indexing(buf3, {jj, 2}));

        auto i = make_var(datatypes::index, "i");
        builder.push_scope();
        {
            auto j = make_var(datatypes::index, "j");
            builder.push_scope();
            {
                auto k = make_var(datatypes::index, "k");
                builder.push_scope();
                {
                    auto v = make_mul(make_indexing(buf, {i, k}),
                            make_indexing(buf2, {k, j}));
                    v = make_add(make_indexing(buf3, {i, j}), v);
                    builder.push_assign(make_indexing(buf3, {i, j}), v);
                    builder.push_assign(
                            make_indexing(buf3, {i, j % UINT64_C(3)}), 3);
                    builder.push_scope();
                    { builder.push_assign(make_indexing(buf3, {i, j}), 3); }
                    auto tblock = builder.pop_scope();
                    builder.push_scope();
                    {
                        builder.push_scope();
                        { builder.push_assign(make_indexing(buf3, {i, j}), 4); }
                        auto inner_tblock = builder.pop_scope();
                        builder.push_scope();
                        {
                            builder.push_assign(make_indexing(buf3, {k, j}), 4);
                            builder.push_assign(
                                    make_indexing(buf4, {i, j, k, 3}), 12);
                            builder.push_assign(make_indexing(buf3, {i, j}, 16),
                                    make_indexing(buf3, {i, j}, 16));
                        }
                        auto inner_fblock = builder.pop_scope();
                        builder.push_if_else(
                                make_cmp_eq(make_indexing(buf3, {i, j}), 2),
                                inner_tblock, inner_fblock);
                    }
                    auto fblock = builder.pop_scope();
                    builder.push_if_else(
                            make_cmp_eq(make_indexing(buf3, {i, j}), 0), tblock,
                            fblock);
                }
                builder.push_for_loop(k, 0, 30, 1, builder.pop_scope(), true,
                        for_type::NORMAL);
            }
            builder.push_for_loop(
                    j, 0, 50, 1, builder.pop_scope(), true, for_type::NORMAL);
        }
        builder.push_for_loop(
                i, 0, 20, 1, builder.pop_scope(), true, for_type::PARALLEL);
    }
    builder.push_returns(1.2f);
    auto fbody = builder.pop_scope();
    auto expected = make_func("aaa", {ii, jj, buf}, fbody, datatypes::f32);

    ir_comparer cmper(true);
    expected->equals(aaa, cmper);
    EXPECT_TRUE(cmper.same_);
}

TEST(GCCore_CPU_ir_builder_cpp, TestEasyBuilderNestedLoops) {
    ir_builder_t builder;
    builder.push_scope();
    _tensor_(buf2, datatypes::s32, {100, 200});
    _tensor_(buf3, datatypes::s32, {100, 200});
    for_loop a, b;
    _nested_for_(range(a, 0, 20, 1, for_type::PARALLEL), range(0, 50),
            range(0, 30)) {
        _iter_var_(i);
        _iter_var_(j);
        _iter_var_(k);
        buf3[{i, j}] = buf3[{i, j}] + buf2[{k, j}];
        buf3[{i, j % UINT64_C(3)}] = 3;
    }
    _named_for_(b, i, 0, 20, 1, for_type::PARALLEL) {
        _for_(j, 0, 50) {
            _for_(k, 0, 30) {
                buf3[{i, j}] = buf3[{i, j}] + buf2[{k, j}];
                buf3[{i, j % UINT64_C(3)}] = 3;
            }
        }
    }
    ir_comparer cmper(false, true);
    EXPECT_TRUE(a->equals(b, cmper));
}

TEST(GCCore_CPU_ir_builder_cpp, TestLValue) {
    builder::ir_builder_t builder;
    builder.push_scope();
    _var_(a, datatypes::s32);
    var va = a.data_.checked_as<var>();
    EXPECT_TRUE(a.get().ptr_same(va));
    EXPECT_TRUE(a.get().ptr_same(va));
    a = 123;
    EXPECT_TRUE(a.get().ptr_same(va));
    _var_(b, datatypes::s32);
    a = b;
    EXPECT_TRUE(a.get().ptr_same(va));

    auto s = builder.pop_scope();

    // create expected
    builder.push_scope();
    expr vara = builder::make_var(datatypes::s32, "a");
    builder.push_var_tensor_def(vara);
    builder.push_assign(vara, 123);
    expr varb = builder::make_var(datatypes::s32, "b");
    builder.push_var_tensor_def(varb);
    builder.push_assign(vara, varb);
    auto exp = builder.pop_scope();

    ir_comparer cmp(false, true);
    EXPECT_TRUE(cmp.compare(s, exp));
}

TEST(GCCore_CPU_ir_builder_cpp, TestArgs) {
    builder::ir_builder_t builder;
    std::vector<expr> args = {builder::make_var(datatypes::bf16, "A"),
            builder::make_var(datatypes::bf16, "B")};
    _function_(datatypes::boolean, AAA, _arg_("v", datatypes::s32),
            _varg_(args)) {}
    EXPECT_EQ(AAA->params_.size(), 3u);
    EXPECT_TRUE(AAA->params_[0].as<var>().defined());
    EXPECT_TRUE(AAA->params_[1].ptr_same(args[0]));
    EXPECT_TRUE(AAA->params_[2].ptr_same(args[1]));
}
