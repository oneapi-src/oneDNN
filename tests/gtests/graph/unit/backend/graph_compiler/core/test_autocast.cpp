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

#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

TEST(GCCore_CPU_auto_cast_cpp, TestAutoCastBinary) {
    builder::ir_builder_t builder;
    builder.push_scope();
    ir_comparer cmp(false, true, true);

    _tensor_(t, datatypes::f32, {100});
    _var_(tptr, datatypes::pointer);
    tptr = t;
    EXPECT_EQ(auto_caster_t()(builder.get_current_scope().body.back())
                      .checked_as<assign>()
                      ->value_->dtype_,
            datatypes::pointer);

    _var_(v, datatypes::index);
    expr add_result = (v + 1);
    EXPECT_TRUE(cmp.compare(auto_caster_t()(add_result + add_result),
            (v + make_cast(datatypes::index, 1))
                    + (v + make_cast(datatypes::index, 1))));

    // binary
    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1.0f) + 1),
            expr(1.0f) + make_cast(datatypes::f32, 1)));
    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1.0f) + UINT64_C(1)),
            expr(1.0f) + make_cast(datatypes::f32, UINT64_C(1))));

    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1) + UINT64_C(1)),
            make_cast(datatypes::index, 1) + UINT64_C(1)));
    EXPECT_TRUE(cmp.compare(
            auto_caster_t()(
                    make_expr<constant_node>(INT64_C(1), datatypes::s8) + 1),
            make_cast(datatypes::s32,
                    make_expr<constant_node>(INT64_C(1), datatypes::s8))
                    + 1));
    // f16 to f32
    EXPECT_TRUE(cmp.compare(
            auto_caster_t()(
                    make_expr<constant_node>(1.2f, datatypes::f16) + 1.2f),
            make_cast(datatypes::f32,
                    make_expr<constant_node>(1.2f, datatypes::f16))
                    + 1.2f));

    // cmp
    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1.0f) > 1),
            expr(1.0f) > make_cast(datatypes::f32, 1)));
    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1.0f) > UINT64_C(1)),
            expr(1.0f) > make_cast(datatypes::f32, UINT64_C(1))));

    EXPECT_TRUE(cmp.compare(auto_caster_t()(expr(1) > UINT64_C(1)),
            make_cast(datatypes::index, 1) > UINT64_C(1)));
    EXPECT_TRUE(cmp.compare(
            auto_caster_t()(
                    make_expr<constant_node>(INT64_C(1), datatypes::s8) > 1),
            make_cast(datatypes::s32,
                    make_expr<constant_node>(INT64_C(1), datatypes::s8))
                    > 1));

    _var_(aaa, datatypes::f32);
    EXPECT_TRUE(cmp.compare(auto_caster_t()(make_stmt<assign_node_t>(aaa, 1)),
            make_stmt<assign_node_t>(aaa, make_cast(datatypes::f32, 1))));
    EXPECT_TRUE(cmp.compare(
            auto_caster_t()(make_stmt<assign_node_t>(
                    aaa, make_expr<constant_node>(INT64_C(1), datatypes::s8))),
            make_stmt<assign_node_t>(aaa,
                    make_cast(datatypes::f32,
                            make_expr<constant_node>(
                                    INT64_C(1), datatypes::s8)))));
    builder.get_current_scope().body.clear();

    // intrin
    any_map_t attr;
    attr["hidden"] = true;
    auto intrin_out = auto_caster_t()(make_expr<intrin_call_node>(
            intrin_type::min, std::vector<expr> {expr(1.0f), expr(1)}, attr));
    auto intrin_expected = make_expr<intrin_call_node>(intrin_type::min,
            std::vector<expr> {expr(1.0f), make_cast(datatypes::f32, expr(1))},
            attr);
    EXPECT_TRUE(cmp.compare(intrin_out, intrin_expected));
    EXPECT_TRUE(
            intrin_out.checked_as<intrin_call>()->intrin_attrs_->get_or_else(
                    "hidden", false));
}

TEST(GCCore_CPU_auto_cast_cpp, TestAutoCastCall) {
    builder::ir_builder_t builder;
    builder.push_scope();
    ir_comparer cmp(false, true, true);
    _decl_func_(datatypes::void_t, abc, _arg_("a", datatypes::f32),
            _arg_("b", datatypes::index));

    EXPECT_TRUE(cmp.compare(auto_caster_t()(abc(1, 1)),
            abc(make_cast(datatypes::f32, 1), make_cast(datatypes::index, 1))));
}

TEST(GCCore_CPU_auto_cast_cpp, TestAutoCastFor) {
    builder::ir_builder_t builder;
    builder.push_scope();
    ir_comparer cmp(false, true, false);
    for_loop loop;
    _named_for_(loop, i, 1, 200UL, 20) {}
    auto expected = builder.push_for_loop(make_var(datatypes::index, "i"),
            make_cast(datatypes::index, 1), 200UL,
            make_cast(datatypes::index, 20),
            make_stmt<stmts_node_t>(std::vector<stmt> {}), true,
            for_type::NORMAL);
    _named_for_(loop, i, 1UL, 200UL, 20UL) {}
    expected = builder.push_for_loop(make_var(datatypes::index, "i"), 1UL,
            200UL, 20UL, make_stmt<stmts_node_t>(std::vector<stmt> {}), true,
            for_type::NORMAL);
    stmt_c after = auto_caster_t()(loop);
    EXPECT_TRUE(after.ptr_same(loop));
    EXPECT_TRUE(cmp.compare(after, expected));
}

TEST(GCCore_CPU_auto_cast_cpp, TestAutoCastIndexing) {
    builder::ir_builder_t builder;
    builder.push_scope();
    ir_comparer cmp(false, true, false);
    _tensor_(tsr, datatypes::f32, {100, 200, 200});
    expr before = tsr[{10, 20, 20}];
    expr_c after = auto_caster_t()(before);
    EXPECT_TRUE(after.ptr_same(before));
    before = tsr[{10, 20, 20UL}];
    after = auto_caster_t()(before);
    EXPECT_TRUE(cmp.compare(after,
            tsr[{make_cast(datatypes::index, 10),
                        make_cast(datatypes::index, 20), 20UL}]
                    .get()));
}
