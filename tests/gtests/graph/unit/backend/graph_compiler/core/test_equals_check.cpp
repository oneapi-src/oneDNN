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
#include <compiler/ir/builder.hpp>

#include "gtest/gtest.h"
#include <compiler/ir/ir_comparer.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

template <typename T, typename T2>
void run_binary() {
    ir_comparer cmper {};
    // binary
    { // good
        expr a = make_expr<T>(
                make_expr<constant_node>(INT64_C(1), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<T>(
                make_expr<constant_node>(INT64_C(1), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        EXPECT_TRUE(a->equals(b));
        EXPECT_TRUE(cmper.compare(a, b));
    }
    { // different op
        expr a = make_expr<T>(
                make_expr<constant_node>(INT64_C(1), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<T2>(
                make_expr<constant_node>(INT64_C(1), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        EXPECT_FALSE(a->equals(b));
    }
    { // different operand
        expr a = make_expr<T>(
                make_expr<constant_node>(INT64_C(1), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<T>(
                make_expr<constant_node>(INT64_C(1111), datatypes::s32),
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        EXPECT_FALSE(a->equals(b));
    }
}

TEST(GCCore_CPU_test_equals_check, TestEqualsExpr) {
    ir_comparer cmper {};
    // const
    {
        expr a = make_expr<constant_node>(12.3f, datatypes::f16);
        auto b = make_expr<constant_node>(12.3f, datatypes::f16);
        EXPECT_TRUE(a->equals(b));
    }
    {
        auto a = make_expr<constant_node>(12.3f, datatypes::f16);
        auto b = make_expr<constant_node>(12.3f, datatypes::f32);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    {
        auto a = make_expr<constant_node>(INT64_C(123), datatypes::s32);
        auto b = make_expr<constant_node>(INT64_C(124), datatypes::s32);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    // const vector
    {
        expr a = make_expr<constant_node>(
                std::vector<union_val> {12.3f, 12.3f, 12.3f, 12.3f},
                sc_data_type_t::f32(4));
        auto b = make_expr<constant_node>(
                std::vector<union_val> {12.3f, 12.3f, 12.3f, 12.3f},
                sc_data_type_t::f32(4));
        EXPECT_TRUE(a->equals(b));
    }
    {
        auto a = make_expr<constant_node>(
                std::vector<union_val> {12.0f, 12.0f, 12.0f, 12.0f},
                sc_data_type_t::f32(4));
        auto b = make_expr<constant_node>(
                std::vector<union_val> {
                        INT64_C(12), INT64_C(12), INT64_C(12), INT64_C(12)},
                sc_data_type_t::s32(4));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    {
        auto a = make_expr<constant_node>(
                std::vector<union_val> {
                        INT64_C(123), INT64_C(123), INT64_C(123), INT64_C(123)},
                sc_data_type_t::s32(4));
        auto b = make_expr<constant_node>(
                std::vector<union_val> {
                        INT64_C(123), INT64_C(123), INT64_C(123), INT64_C(124)},
                sc_data_type_t::s32(4));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    // var
    {
        auto a = make_expr<var_node>(datatypes::f16, "a");
        auto b = make_expr<var_node>(datatypes::f16, "b");
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
        ir_comparer cmp2(true, true, false, true);
        EXPECT_FALSE(a->equals(b, cmp2));
        EXPECT_EQ(cmp2.diff->first_diff_expr_.first.get(), a.get());
        EXPECT_EQ(cmp2.diff->first_diff_expr_.second.get(), b.get());

        auto c = make_expr<var_node>(datatypes::f16, "a");
        ir_comparer cmp3(true, true, true, true);
        EXPECT_TRUE(a->equals(c, cmper));
        EXPECT_TRUE(a->equals(c, cmp2));
        cmper.reset();
        EXPECT_FALSE(a->equals(c, cmp3));
    }
    {
        auto a = make_expr<var_node>(datatypes::f16, "a");
        auto b = make_expr<var_node>(datatypes::s32, "b");
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // cast
    { // good
        auto a = make_expr<cast_node>(datatypes::f16,
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<cast_node>(datatypes::f16,
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different val
        auto a = make_expr<cast_node>(datatypes::f16,
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<cast_node>(datatypes::f16,
                make_expr<constant_node>(INT64_C(124), datatypes::s32));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different type
        auto a = make_expr<cast_node>(datatypes::f16,
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        auto b = make_expr<cast_node>(datatypes::u8,
                make_expr<constant_node>(INT64_C(123), datatypes::s32));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    run_binary<add_node, mul_node>();
    run_binary<logic_and_node, logic_or_node>();
    run_binary<cmp_eq_node, cmp_lt_node>();

    // logic_not_node
    { // good
        auto a = make_expr<logic_not_node>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean));
        auto b = make_expr<logic_not_node>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // NE
        auto a = make_expr<logic_not_node>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean));
        auto b = make_expr<logic_not_node>(make_expr<logic_not_node>(
                make_expr<constant_node>(UINT64_C(0), datatypes::boolean)));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // call
    func_t f1 = func_t(new func_base("AA", {}, stmt(), datatypes::f16));
    func_t f2 = func_t(new func_base("AA", {}, stmt(), datatypes::f16));
    expr ptr1 = make_expr<var_node>(datatypes::pointer, "p1");
    ptr1->attr()["prototype"] = f1;
    expr ptr2 = make_expr<var_node>(datatypes::pointer, "p2");
    ptr2->attr()["prototype"] = f2;
    { // good
        auto a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();

        a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                std::vector<call_node::parallel_attr_t> {{1, 2, 3}});
        b = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                std::vector<call_node::parallel_attr_t> {{1, 2, 3}});
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();

        // check call function pointer
        a = make_expr<call_node>(ptr1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        b = make_expr<call_node>(ptr2,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different func_t (even if they are actually the same)
        auto a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<call_node>(f2,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        // check call function ptr v.s. call function
        b = make_expr<call_node>(ptr1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different arg num
        auto a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<call_node>(f2,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(UINT64_C(2), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different arg
        auto a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<call_node>(f2,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(UINT64_C(2), datatypes::s32)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // different para_attr
        auto a = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                std::vector<call_node::parallel_attr_t> {{1, 2, 3}});
        auto b = make_expr<call_node>(f1,
                std::vector<expr> {make_expr<constant_node>(
                                           UINT64_C(1), datatypes::boolean),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b->para_attr_ = {{2, 2, 3}};
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // tensor
    { // good
        auto a = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<tensor_node>(datatypes::f32, "BB",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();

        ir_comparer cmp2(true, true, false, true);
        EXPECT_FALSE(a->equals(b, cmp2));
        EXPECT_EQ(cmp2.diff->first_diff_expr_.first.get(), a.get());
        EXPECT_EQ(cmp2.diff->first_diff_expr_.second.get(), b.get());
        cmp2.reset();

        auto c = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});

        EXPECT_TRUE(a->equals(c, cmper));
        cmper.reset();
        EXPECT_TRUE(a->equals(c, cmp2));
        ir_comparer cmp3(true, true, true, true);
        EXPECT_FALSE(a->equals(c, cmp3));
        cmp3.set_expr_mapping(a, c);
        EXPECT_TRUE(a->equals(c, cmp3));

        a->address_space_ = address_space::device;
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff dim
        auto a = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(4), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff type
        auto a = make_expr<tensor_node>(datatypes::s32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff init_val
        auto a = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {make_expr<constant_node>(UINT64_C(10))});
        int a_v[] = {1, 2, 3, 4};
        a->init_value_ = std::make_shared<static_data_t>(a_v, sizeof(a_v));
        auto c = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {make_expr<constant_node>(UINT64_C(10))});
        c->init_value_ = std::make_shared<static_data_t>(a_v, sizeof(a_v));
        auto b = make_expr<tensor_node>(datatypes::f32, "AA",
                std::vector<expr> {make_expr<constant_node>(UINT64_C(10))});
        int b_v[] = {1, 2, 3, 6};
        b->init_value_ = std::make_shared<static_data_t>(b_v, sizeof(b_v));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
        EXPECT_TRUE(a->equals(c, cmper));
        cmper.reset();
    }
    // indexing

    auto tensora = make_expr<tensor_node>(datatypes::f32, "AA",
            std::vector<expr> {
                    make_expr<constant_node>(UINT64_C(1), datatypes::index),
                    make_expr<constant_node>(UINT64_C(2), datatypes::index)});
    auto tensorb = make_expr<tensor_node>(datatypes::f32, "AA",
            std::vector<expr> {
                    make_expr<constant_node>(UINT64_C(1), datatypes::index),
                    make_expr<constant_node>(UINT64_C(4), datatypes::index)});
    { // good
        auto a = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        auto b = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();

        a = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                make_expr<constant_node>(UINT64_C(2), datatypes::index));
        b = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                make_expr<constant_node>(UINT64_C(2), datatypes::index));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff tensor
        auto a = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        auto b = make_expr<indexing_node>(tensorb,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff index
        auto a = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(1), datatypes::index)},
                expr());
        auto b = make_expr<indexing_node>(tensorb,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff mask
        auto a = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(1), datatypes::index)},
                make_expr<constant_node>(UINT64_C(1), datatypes::index));
        auto b = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)},
                expr());
        auto c = make_expr<indexing_node>(tensora,
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(1), datatypes::index)},
                make_expr<constant_node>(UINT64_C(2), datatypes::index));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
        EXPECT_FALSE(b->equals(a, cmper));
        cmper.reset();
        EXPECT_FALSE(a->equals(c, cmper));
        cmper.reset();
    }

    auto make_tensor_ptr = [](const expr &tsr, uint64_t idx1, uint64_t idx2) {
        return make_expr<tensorptr_node>(
                make_expr<indexing_node>(tsr,
                        std::vector<expr> {make_expr<constant_node>(
                                                   idx1, datatypes::index),
                                make_expr<constant_node>(
                                        idx2, datatypes::index)},
                        expr()),
                std::vector<expr> {expr(1), 2}, false);
    };
    { // good
        auto a = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        auto b = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
        // diff shape
        a->shape_ = {};
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
        b->shape_ = {};
    }
    { // diff tensor
        auto a = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        auto b = make_tensor_ptr(tensorb, UINT64_C(1), UINT64_C(2));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff index
        auto a = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(1));
        auto b = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff is_slice
        auto a = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        auto b = make_tensor_ptr(tensora, UINT64_C(1), UINT64_C(2));
        a->is_slice_ = true;
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    auto make_intrin_call = [](uint64_t idx1, uint64_t idx2,
                                    intrin_type name = intrin_type::max) {
        return make_expr<intrin_call_node>(
                name, std::vector<expr> {expr(idx1), idx2}, any_map_t());
    };
    { // good
        auto a = make_intrin_call(UINT64_C(1), UINT64_C(2));
        auto b = make_intrin_call(UINT64_C(1), UINT64_C(2));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff param
        auto a = make_intrin_call(UINT64_C(1), UINT64_C(2));
        auto b = make_intrin_call(UINT64_C(2), UINT64_C(2));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff name
        auto a = make_intrin_call(UINT64_C(1), UINT64_C(2));
        auto b = make_intrin_call(UINT64_C(1), UINT64_C(2), intrin_type::min);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    { // good
        auto a = make_expr<func_addr_node>(f1);
        auto b = make_expr<func_addr_node>(f1);
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff func_t
        auto a = make_expr<func_addr_node>(f1);
        auto b = make_expr<func_addr_node>(f2);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    { // good
        auto a = make_expr<ssa_phi_node>(std::vector<expr> {1, 2, 3}, false);
        auto b = make_expr<ssa_phi_node>(std::vector<expr> {1, 2, 3}, false);
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff
        auto a = make_expr<ssa_phi_node>(std::vector<expr> {1}, false);
        auto b = make_expr<ssa_phi_node>(std::vector<expr> {1, 2, 3}, false);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
}

TEST(GCCore_CPU_test_equals_check, TestEqualsAutoReset) {
    ir_comparer cmper {};
    auto var1 = make_expr<var_node>(datatypes::f32, "A");
    auto var2 = make_expr<var_node>(datatypes::f32, "B");
    {
        stmt a = make_stmt<assign_node_t>(
                var1, make_expr<constant_node>(1.0f, datatypes::f16));
        auto b = make_stmt<assign_node_t>(
                var2, make_expr<constant_node>(1.0f, datatypes::f16));
        EXPECT_TRUE(cmper.compare(a, b, false));
        EXPECT_TRUE(cmper.get_expr_mapping(var1, var2));
        cmper.reset();
        EXPECT_TRUE(cmper.compare(a, b));
        EXPECT_FALSE(cmper.get_expr_mapping(var1, var2));
    }
}

TEST(GCCore_CPU_test_equals_check, TestEqualsStmt) {
    ir_comparer cmper {};
    // assign
    auto var1 = make_expr<var_node>(datatypes::f32, "A");
    auto var2 = make_expr<var_node>(datatypes::f32, "B");
    auto var3 = make_expr<var_node>(datatypes::f16, "A");
    {
        stmt a = make_stmt<assign_node_t>(
                var1, make_expr<constant_node>(1.0f, datatypes::f16));
        auto b = make_stmt<assign_node_t>(
                var2, make_expr<constant_node>(1.0f, datatypes::f16));
        EXPECT_TRUE(a->equals(b));
        EXPECT_TRUE(cmper.compare(a, b));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff var
        auto a = make_stmt<assign_node_t>(
                var1, make_expr<constant_node>(1.0f, datatypes::f16));
        auto b = make_stmt<assign_node_t>(
                var3, make_expr<constant_node>(1.0f, datatypes::f16));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff val
        auto a = make_stmt<assign_node_t>(
                var1, make_expr<constant_node>(1.0f, datatypes::f16));
        auto b = make_stmt<assign_node_t>(
                var3, make_expr<constant_node>(1.2f, datatypes::f16));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // stmts
    {
        auto a = make_stmt<stmts_node_t>(
                std::vector<stmt> {make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16))});
        auto b = make_stmt<stmts_node_t>(
                std::vector<stmt> {make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16))});
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    {
        auto a = make_stmt<stmts_node_t>(
                std::vector<stmt> {make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16))});
        auto b = make_stmt<stmts_node_t>(
                std::vector<stmt> {make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.4f, datatypes::f16))});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
        b = make_stmt<stmts_node_t>(std::vector<stmt> {});
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // ifelse
    {
        auto a = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                stmt());
        auto b = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                stmt());
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
        a = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)));
        b = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // cond
        auto a = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(2), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                stmt());
        auto b = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                stmt());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // then
        auto a = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.4f, datatypes::f16)),
                stmt());
        auto b = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                stmt());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }
    { // else
        auto a = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.4f, datatypes::f16)),
                stmt());
        auto b = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.4f, datatypes::f16)));
        auto c = make_stmt<if_else_node_t>(
                make_expr<constant_node>(UINT64_C(1), datatypes::boolean),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.2f, datatypes::f16)),
                make_stmt<evaluate_node_t>(
                        make_expr<constant_node>(1.8f, datatypes::f16)));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
        EXPECT_FALSE(b->equals(a, cmper));
        cmper.reset();
        EXPECT_FALSE(b->equals(c, cmper));
        cmper.reset();
    }

    // evaluate
    {
        auto a = make_stmt<evaluate_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        auto b = make_stmt<evaluate_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    {
        auto a = make_stmt<evaluate_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        auto b = make_stmt<evaluate_node_t>(
                make_expr<constant_node>(1.2f, datatypes::s32));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // var def
    {
        auto a = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr(1));
        auto b = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr(1));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    { // diff linkage
        auto a = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr(1));
        auto b = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"),
                linkage::private_global, expr(1));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    { // diff init v
        auto a = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr());
        auto b = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr(1));
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    { // diff var
        auto a = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f16, "a"), linkage::local,
                expr());
        auto b = make_stmt<define_node_t>(
                make_expr<var_node>(datatypes::f32, "a"), linkage::local,
                expr());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // returns
    {
        auto a = make_stmt<returns_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        auto b = make_stmt<returns_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();
    }
    {
        auto a = make_stmt<returns_node_t>(
                make_expr<constant_node>(1.2f, datatypes::f16));
        auto b = make_stmt<returns_node_t>(expr());
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    // for-loop
    auto const1 = make_expr<constant_node>(UINT64_C(123), datatypes::index);
    auto const2 = make_expr<constant_node>(UINT64_C(321), datatypes::index);
    auto const3 = make_expr<constant_node>(UINT64_C(1321), datatypes::index);
    {
        auto a = make_stmt<for_loop_node_t>(var1, const1, const2, const3,
                make_stmt<evaluate_node_t>(const1), true, for_type::NORMAL);
        auto b = make_stmt<for_loop_node_t>(var2, const1, const2, const3,
                make_stmt<evaluate_node_t>(const1), true, for_type::NORMAL);
        EXPECT_TRUE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var3, const1, const2, const3,
                make_stmt<evaluate_node_t>(const1), true, for_type::NORMAL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var1, const2, const2, const3,
                make_stmt<evaluate_node_t>(const1), true, for_type::NORMAL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var1, const1, const2, const2,
                make_stmt<evaluate_node_t>(const1), true, for_type::NORMAL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var1, const1, const2, const3,
                make_stmt<evaluate_node_t>(const2), true, for_type::NORMAL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var1, const1, const2, const3,
                make_stmt<evaluate_node_t>(const1), false, for_type::NORMAL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();

        b = make_stmt<for_loop_node_t>(var1, const1, const2, const3,
                make_stmt<evaluate_node_t>(const1), true, for_type::PARALLEL);
        EXPECT_FALSE(a->equals(b, cmper));
        cmper.reset();
    }

    { // var order checking
        // var a; var b; a=b
        // should be different from:
        // var a; var b; b=a
        auto a = make_expr<var_node>(datatypes::s32, "AA1");
        auto b = make_expr<var_node>(datatypes::s32, "AA2");
        std::vector<stmt> body1 {
                make_stmt<evaluate_node_t>(a),
                make_stmt<evaluate_node_t>(b),
                make_stmt<assign_node_t>(a, b),
        };

        std::vector<stmt> body2 {
                make_stmt<evaluate_node_t>(b),
                make_stmt<evaluate_node_t>(a),
                make_stmt<assign_node_t>(a, b),
        };
        EXPECT_FALSE(make_stmt<stmts_node_t>(std::move(body1))
                             ->equals(make_stmt<stmts_node_t>(std::move(body2)),
                                     cmper));
        cmper.reset();
    }

    { // tensor order checking
        auto a = make_expr<tensor_node>(datatypes::s32, "AA1",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        auto b = make_expr<tensor_node>(datatypes::s32, "AA2",
                std::vector<expr> {
                        make_expr<constant_node>(UINT64_C(1), datatypes::index),
                        make_expr<constant_node>(
                                UINT64_C(2), datatypes::index)});
        std::vector<stmt> body1 {
                make_stmt<evaluate_node_t>(a),
                make_stmt<evaluate_node_t>(b),
                make_stmt<assign_node_t>(a, b),
        };

        std::vector<stmt> body2 {
                make_stmt<evaluate_node_t>(b),
                make_stmt<evaluate_node_t>(a),
                make_stmt<assign_node_t>(a, b),
        };
        EXPECT_FALSE(make_stmt<stmts_node_t>(std::move(body1))
                             ->equals(make_stmt<stmts_node_t>(std::move(body2)),
                                     cmper));
        cmper.reset();
    }
}

TEST(GCCore_CPU_test_equals_check, TestEqualsFunction) {
    ir_comparer cmper {};
    auto var1 = make_expr<var_node>(datatypes::f32, "A");
    auto var2 = make_expr<var_node>(datatypes::s32, "B");
    auto var3 = make_expr<var_node>(datatypes::f16, "A");
    auto const1 = make_stmt<evaluate_node_t>(
            make_expr<constant_node>(UINT64_C(123), datatypes::index));
    auto const2 = make_stmt<evaluate_node_t>(
            make_expr<constant_node>(UINT64_C(321), datatypes::index));

    func_t f1 = func_t(new func_base(
            "AA", std::vector<expr> {var1, var2}, const1, datatypes::f16));
    func_t f2 = func_t(new func_base(
            "AA", std::vector<expr> {var1, var2}, const1, datatypes::f16));
    EXPECT_TRUE(f1->equals(f2, cmper));
    cmper.reset();
    EXPECT_TRUE(f1->equals(f2));

    f2 = func_t(new func_base(
            "AA2", std::vector<expr> {var1, var2}, const1, datatypes::f16));
    EXPECT_TRUE(f1->equals(f2, cmper));
    cmper.reset();
    {
        ir_comparer cmp2(true, true, false, true);
        EXPECT_FALSE(f1->equals(f2, cmp2));
        EXPECT_EQ(cmp2.diff->first_diff_func_.first.get(), f1.get());
        EXPECT_EQ(cmp2.diff->first_diff_func_.second.get(), f2.get());
        EXPECT_FALSE(cmp2.same_);
        cmp2.reset();
        EXPECT_EQ(cmp2.diff->first_diff_func_.first.get(), nullptr);
    }

    f2 = func_t(new func_base(
            "AA", std::vector<expr> {var1, var3}, const1, datatypes::f16));
    EXPECT_FALSE(f1->equals(f2, cmper));
    cmper.reset();
    {
        ir_comparer cmp2(true, true, false, true);
        EXPECT_FALSE(f1->equals(f2, cmp2));
        EXPECT_EQ(cmp2.diff->first_diff_expr_.first.get(), var2.get());
        EXPECT_EQ(cmp2.diff->first_diff_expr_.second.get(), var3.get());
    }

    f2 = func_t(new func_base(
            "AA", std::vector<expr> {var1, var2}, const2, datatypes::f16));
    EXPECT_FALSE(f1->equals(f2, cmper));
    cmper.reset();

    f2 = func_t(new func_base(
            "AA", std::vector<expr> {var1, var2}, const1, datatypes::bf16));
    EXPECT_FALSE(f1->equals(f2, cmper));
    cmper.reset();
}

TEST(GCCore_CPU_test_equals_check, TestEqualsComm) {
    ir_comparer cmper {true, false, true, false, true};
    ir_comparer cmper2 {true, false, true, false, false};
    auto var1 = make_expr<var_node>(datatypes::f32, "A");
    auto var2 = make_expr<var_node>(datatypes::f32, "B");
    expr v1 = var1 + var2;
    expr v2 = var2 + var1;

    // binary
    EXPECT_TRUE(cmper.compare(v1, v2));
    EXPECT_FALSE(cmper2.compare(v1, v2));
    EXPECT_FALSE(cmper.compare(var1 - var2, var2 - var1));

    // cmp
    EXPECT_TRUE(cmper.compare(var1 == var2, var2 == var1));
    EXPECT_FALSE(cmper.compare(var1 < var2, var2 < var1));

    auto var3 = make_expr<var_node>(datatypes::boolean, "A");
    auto var4 = make_expr<var_node>(datatypes::boolean, "B");

    EXPECT_TRUE(cmper.compare(var3 && var4, var4 && var3));
}
