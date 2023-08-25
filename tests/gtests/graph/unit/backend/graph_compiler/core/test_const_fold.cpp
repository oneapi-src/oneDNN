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
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass/validator.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/index_flatten.hpp>

#include <algorithm>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_const_fold_cpp, TestConstCompute) {
    constant_folder_t f;
    var va = make_expr<var_node>(datatypes::s32, "a");
#define MAKE_TEST(op, a, b) \
    EXPECT_TRUE(f(expr(a) op expr(b))->equals(expr(a op b)))
#define MAKE_TEST2(op, a, b) \
    EXPECT_TRUE(f(builder::make_##op(expr(a), expr(b))) \
                        ->equals(expr(std::op(a, b))))
#define MAKE_TESTS(op, a, b) \
    MAKE_TEST(op, a, b); \
    MAKE_TEST(op, (uint64_t)(a), (uint64_t)(b)); \
    MAKE_TEST(op, (float)(a), (float)(b));
#define MAKE_TESTS2(op, a, b) \
    MAKE_TEST2(op, a, b); \
    MAKE_TEST2(op, (uint64_t)(a), (uint64_t)(b)); \
    MAKE_TEST2(op, (float)(a), (float)(b));

    MAKE_TESTS(+, 11, 2);
    MAKE_TESTS(-, 11, 2);
    MAKE_TEST(*, 11, 2);
    MAKE_TEST(/, 11, 2);
    MAKE_TEST(%, 11, 2);
    MAKE_TEST2(min, 11, 2);
    MAKE_TEST2(max, 11, 2);
    MAKE_TEST(&, 11, 2);
    MAKE_TEST(|, 11, 2);

    // cmp
    MAKE_TESTS(>, 11, 2);
    MAKE_TESTS(>=, 11, 2);
    MAKE_TESTS(<, 11, 2);
    MAKE_TESTS(<=, 11, 2);
    MAKE_TESTS(==, 11, 2);
    MAKE_TESTS(!=, 11, 2);

    // logic
    MAKE_TEST(&&, true, false);
    MAKE_TEST(&&, true, true);
    MAKE_TEST(&&, false, false);
    MAKE_TEST(&&, false, true);
    MAKE_TEST(||, true, false);
    MAKE_TEST(||, true, true);
    MAKE_TEST(||, false, false);
    MAKE_TEST(||, false, true);

    EXPECT_TRUE(f(!expr(false))->equals(expr(true)));
    EXPECT_TRUE(f(!expr(true))->equals(expr(false)));

    // cast from s32
    EXPECT_TRUE(
            f(builder::make_cast(datatypes::index, 32))->equals(expr(32UL)));
    EXPECT_TRUE(f(builder::make_cast(datatypes::f32, 32))->equals(expr(32.0f)));
    EXPECT_TRUE(f(builder::make_cast(datatypes::s32, 32))->equals(expr(32)));

    // cast from f32
    EXPECT_TRUE(
            f(builder::make_cast(datatypes::index, 32.0f))->equals(expr(32UL)));
    EXPECT_TRUE(
            f(builder::make_cast(datatypes::f32, 32.0f))->equals(expr(32.0f)));
    EXPECT_TRUE(f(builder::make_cast(datatypes::s32, 32.0f))->equals(expr(32)));

    // cast from index
    EXPECT_TRUE(f(builder::make_cast(datatypes::index, expr(32UL)))
                        ->equals(expr(32UL)));
    EXPECT_TRUE(f(builder::make_cast(datatypes::f32, expr(32UL)))
                        ->equals(expr(32.0f)));
    EXPECT_TRUE(f(builder::make_cast(datatypes::s32, expr(32UL)))
                        ->equals(expr(32)));

    // SHR. SHL
    EXPECT_TRUE(f(expr(UINT64_C(1234)) >> expr(UINT64_C(3)))
                        ->equals(expr(uint64_t(1234 >> 3))));
    EXPECT_TRUE(f(expr(INT32_C(-1234)) >> expr(INT32_C(3)))
                        ->equals(expr(int32_t((-1234) >> 3))));
    EXPECT_TRUE(f(expr(UINT64_C(1234)) << expr(UINT64_C(3)))
                        ->equals(expr(uint64_t(1234 << 3))));
    EXPECT_TRUE(f(expr(INT32_C(1234)) << expr(INT32_C(3)))
                        ->equals(expr(int32_t((1234) << 3))));
    // failures
    expr tmp = builder::make_cast(datatypes::s32, va);
    EXPECT_TRUE(f(tmp).ptr_same(tmp));
    tmp = va + 2;
    EXPECT_TRUE(f(tmp).ptr_same(tmp));
    tmp = va == 2;
    EXPECT_TRUE(f(tmp).ptr_same(tmp));
    tmp = va && va->remake();
    EXPECT_TRUE(f(tmp).ptr_same(tmp));
    tmp = !va;
    EXPECT_TRUE(f(tmp).ptr_same(tmp));
#undef MAKE_TEST
#undef MAKE_TESTS
#undef MAKE_TEST2
#undef MAKE_TESTS2
}

#define EXPECT_NO_CHANGE(exp) \
    tmp = (exp); \
    EXPECT_TRUE(f(tmp).ptr_same(tmp))

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldRotation) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    constant_folder_t f {false};
    ir_comparer cmper(false, true, true);
    expr tmp;
    // c + x => x + c
    EXPECT_TRUE(cmper.compare(f(2 * va), va * 2));
    EXPECT_NO_CHANGE(va * 2);
    EXPECT_NO_CHANGE(2 - va);

    // (x + c1) + c2 => x + (c1 + c2)
    EXPECT_TRUE(cmper.compare(f((va * 2) * 3), va * 6));

    // (x + c) + y => (x + y) + c
    EXPECT_TRUE(cmper.compare(f((va * 2) * va), va * va * 2));
    EXPECT_NO_CHANGE(va + va + 3);

    // x + (y + c) => (x + y) + c
    EXPECT_TRUE(cmper.compare(f(va * (va * 2)), va * va * 2));
    EXPECT_TRUE(cmper.compare(f(va + (va - 2)), va + va - 2));
    EXPECT_NO_CHANGE(va + va + va);

    // (x + c1) + (y + c2) => (x + y) + (c1 + c2)
    EXPECT_TRUE(cmper.compare(f((va * 4) * (va * 2)), va * va * 8));
    EXPECT_TRUE(cmper.compare(f((va + 4) + (va - 2)), va + va + 2));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldSpecialConst) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var fa = make_expr<var_node>(datatypes::f32, "fa");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    var vc = make_expr<var_node>(datatypes::boolean, "c");
    constant_folder_t f {false};
    ir_comparer cmper(false, true, true);
    expr tmp;

    // scalar
    EXPECT_TRUE(cmper.compare(f(false && vb), expr(false)));
    EXPECT_TRUE(cmper.compare(f(true && vb), vb));
    EXPECT_NO_CHANGE(vb && vc);

    EXPECT_TRUE(cmper.compare(f(false || vb), vb));
    EXPECT_TRUE(cmper.compare(f(true || vb), expr(true)));
    EXPECT_NO_CHANGE(vb || vc);

    EXPECT_TRUE(cmper.compare(f(va & va), va));
    EXPECT_TRUE(cmper.compare(f(va | va), va));
    EXPECT_TRUE(cmper.compare(f(0 + va), va));
    EXPECT_TRUE(cmper.compare(f(0.0f + fa), fa));
    EXPECT_TRUE(cmper.compare(f(va - 0), va));
    EXPECT_TRUE(cmper.compare(f(fa - 0.0f), fa));
    EXPECT_TRUE(cmper.compare(f(0 * va), expr(0)));
    EXPECT_TRUE(cmper.compare(f(0.0f * fa), expr(0.0f)));
    EXPECT_TRUE(cmper.compare(f(1 * va), va));
    EXPECT_TRUE(cmper.compare(f(1.0f * fa), fa));
    EXPECT_TRUE(cmper.compare(f(va / 1), va));
    EXPECT_TRUE(cmper.compare(f(fa / 1.0f), fa));
    EXPECT_TRUE(cmper.compare(f(va % 1), expr(0)));

    // vec
    var vax = make_expr<var_node>(sc_data_type_t::s32(16), "a");
    var fax = make_expr<var_node>(sc_data_type_t::f32(16), "fa");
    var vbx = make_expr<var_node>(sc_data_type_t::boolean(16), "b");
    var vcx = make_expr<var_node>(sc_data_type_t::boolean(16), "c");
    expr zero_i = builder::make_constant({0UL}, sc_data_type_t::s32(16));
    expr one_i = builder::make_constant({1UL}, sc_data_type_t::s32(16));
    expr zero_f = builder::make_constant({0.f}, sc_data_type_t::f32(16));
    expr one_f = builder::make_constant({1.f}, sc_data_type_t::f32(16));
    expr zero_b = builder::make_constant({0UL}, sc_data_type_t::boolean(16));
    expr one_b = builder::make_constant({1UL}, sc_data_type_t::boolean(16));

    EXPECT_TRUE(cmper.compare(f(zero_b && vbx), zero_b));
    EXPECT_TRUE(cmper.compare(f(one_b && vbx), vbx));
    EXPECT_NO_CHANGE(vbx && vcx);

    EXPECT_TRUE(cmper.compare(f(zero_b || vbx), vbx));
    EXPECT_TRUE(cmper.compare(f(one_b || vbx), one_b));
    EXPECT_NO_CHANGE(vbx || vcx);

    EXPECT_TRUE(cmper.compare(f(vax & vax), vax));
    EXPECT_TRUE(cmper.compare(f(vax | vax), vax));
    EXPECT_TRUE(cmper.compare(f(zero_i + vax), vax));
    EXPECT_TRUE(cmper.compare(f(zero_f + fax), fax));
    EXPECT_TRUE(cmper.compare(f(vax - zero_i), vax));
    EXPECT_TRUE(cmper.compare(f(fax - zero_f), fax));
    EXPECT_TRUE(cmper.compare(f(zero_i * vax), zero_i));
    EXPECT_TRUE(cmper.compare(f(zero_f * fax), zero_f));
    EXPECT_TRUE(cmper.compare(f(one_i * vax), vax));
    EXPECT_TRUE(cmper.compare(f(one_f * fax), fax));
    EXPECT_TRUE(cmper.compare(f(vax / one_i), vax));
    EXPECT_TRUE(cmper.compare(f(fax / one_f), fax));
    EXPECT_TRUE(cmper.compare(f(vax % one_i), zero_i));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldSpecialExpr) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var fa = make_expr<var_node>(datatypes::f32, "fa");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    var vc = make_expr<var_node>(datatypes::boolean, "c");
    constant_folder_t f {false};
    ir_comparer cmper(false, true, true);
    expr tmp;
    EXPECT_TRUE(cmper.compare(f(va * expr(8UL) / expr(4UL)), va * expr(2UL)));
    EXPECT_TRUE(cmper.compare(f(va - va), expr(0)));
    EXPECT_TRUE(cmper.compare(f(fa - fa), expr(0.0f)));
    EXPECT_TRUE(cmper.compare(f(va / va), expr(1)));
    EXPECT_TRUE(cmper.compare(f(fa / fa), expr(1.0f)));
    EXPECT_TRUE(cmper.compare(f(va % va), expr(0)));
    EXPECT_TRUE(cmper.compare(f(vb && vb), vb));
    EXPECT_TRUE(cmper.compare(f(vb || vb), vb));
    EXPECT_TRUE(cmper.compare(f(va * expr(13) * expr(5) % expr(5)), expr(0)));
    EXPECT_TRUE(cmper.compare(f(va % expr(12) % expr(12)), va % expr(12)));
    EXPECT_TRUE(cmper.compare(f(builder::make_min(va, va)), va));
    EXPECT_TRUE(cmper.compare(f(builder::make_max(va, va)), va));
    EXPECT_TRUE(cmper.compare(f(builder::make_min(fa, fa)), fa));
    EXPECT_TRUE(cmper.compare(f(builder::make_max(fa, fa)), fa));

    EXPECT_TRUE(cmper.compare(f(va > va), expr(false)));
    EXPECT_TRUE(cmper.compare(f(fa > fa), expr(false)));
    EXPECT_TRUE(cmper.compare(f(va < va), expr(false)));
    EXPECT_TRUE(cmper.compare(f(fa < fa), expr(false)));
    EXPECT_TRUE(cmper.compare(f(va != va), expr(false)));
    EXPECT_TRUE(cmper.compare(f(fa != fa), expr(false)));
    EXPECT_TRUE(cmper.compare(f(va >= va), expr(true)));
    EXPECT_TRUE(cmper.compare(f(fa >= fa), expr(true)));
    EXPECT_TRUE(cmper.compare(f(va <= va), expr(true)));
    EXPECT_TRUE(cmper.compare(f(fa <= fa), expr(true)));
    EXPECT_TRUE(cmper.compare(f(va == va), expr(true)));
    EXPECT_TRUE(cmper.compare(f(fa == fa), expr(true)));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldFmadd) {
#define ADD(a, b) builder::make_add(a, b)
#define MUL(a, b) builder::make_mul(a, b)
#define FMADD(a, b, c) builder::make_fmadd(a, b, c)
    var fax = make_expr<var_node>(sc_data_type_t::f32(16), "fax");
    var fbx = make_expr<var_node>(sc_data_type_t::f32(16), "fbx");
    var fcx = make_expr<var_node>(sc_data_type_t::f32(16), "fcx");
    expr zero_f = builder::make_constant({0.f}, sc_data_type_t::f32(16));
    expr one_f = builder::make_constant({1.f}, sc_data_type_t::f32(16));

    constant_folder_t f;
    ir_comparer cmper(false, true, true);
    expr tmp;

    EXPECT_NO_CHANGE(FMADD(fax, fbx, fcx));

    EXPECT_TRUE(cmper.compare(f(FMADD(fax, fbx, zero_f)), MUL(fax, fbx)));
    EXPECT_TRUE(cmper.compare(f(FMADD(one_f, fbx, fcx)), ADD(fbx, fcx)));
    EXPECT_TRUE(cmper.compare(f(FMADD(fax, one_f, fcx)), ADD(fax, fcx)));
    EXPECT_TRUE(cmper.compare(f(FMADD(one_f, fbx, zero_f)), fbx));
    EXPECT_TRUE(cmper.compare(f(FMADD(zero_f, fbx, fcx)), fcx));
    EXPECT_TRUE(cmper.compare(f(FMADD(fax, zero_f, fcx)), fcx));
#undef ADD
#undef MUL
#undef FMADD
}

TEST(GCCore_CPU_const_fold_cpp, TestCanonialize) {
    var a = make_expr<var_node>(datatypes::s32, "a");
    var b = make_expr<var_node>(datatypes::s32, "b");
    var c = make_expr<var_node>(datatypes::s32, "c");
    var d = make_expr<var_node>(datatypes::index, "d");
    var e = make_expr<var_node>(datatypes::index, "e");
    auto tsr = builder::make_tensor("C", {100}, datatypes::f32);
    constant_folder_t folder {false};
    // complex add/sub
    EXPECT_TRUE(folder((2 + a) + ((b + a) - (a + b - 10 - (a - b))))
                        ->equals(((a - b) + a) + 12));
    //((a - b) + a) + 12

    // test unchanged
    expr inp = (b * a) * c;
    EXPECT_TRUE(folder(inp).ptr_same(inp));
    // if the order is already sorted, but is not chained to the right
    EXPECT_TRUE(folder(b * (a * c))->equals(inp));

    // add negative => sub
    EXPECT_TRUE(folder(10 + (a + 99 - 200))->equals(a - 91));
    EXPECT_TRUE(folder(UINT64_C(10) + (d + UINT64_C(99) - UINT64_C(200)))
                        ->equals(d - UINT64_C(91)));
    EXPECT_TRUE(folder(UINT64_C(10) + (d + UINT64_C(99) - UINT64_C(100)))
                        ->equals(d + UINT64_C(9)));
    // add zero
    EXPECT_TRUE(folder(10 + (a + 190 - 200))->equals(a));
    EXPECT_TRUE(folder(UINT64_C(10) + (d + UINT64_C(190) - UINT64_C(200)))
                        ->equals(d));

    // fold to zero
    EXPECT_TRUE(folder(a + a - a - a + 10 - 10)->equals(0));

    // fold to negative start
    EXPECT_TRUE(folder(a + b - a - b - a - 10)->equals(0 - a - 10));
    EXPECT_TRUE(folder(e + d - e - d - e - UINT64_C(190))
                        ->equals(UINT64_C(0) - e - UINT64_C(190)));

    // in other expr
    EXPECT_TRUE(folder(tsr[a + 10 - b - a])->equals(tsr[10 - b]));

    // mixed
    EXPECT_TRUE(folder(a + 10 * (10 + b + c) + 10 + b)
                        ->equals(((b + a) + ((b + c) * 10)) + 110));

    EXPECT_TRUE(folder(tsr[a + (10 - a)] + tsr[b + (10 - a)])
                        ->equals(tsr[10] + tsr[((b - a) + 10)]));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldMutiLevel) {
    var a = make_expr<var_node>(datatypes::s32, "a");
    var b = make_expr<var_node>(datatypes::s32, "b");
    var c = make_expr<var_node>(datatypes::s32, "c");
    EXPECT_TRUE(constant_folder_t(false)(a + 2 + (a + 4) * 2 + (4 + a * 0))
                        ->equals(a + a * 2 + 14));
    EXPECT_TRUE(constant_folder_t(false)((2 + a) + (4 + a) + (6 + a))
                        ->equals(a + a + a + 12));
    EXPECT_TRUE(constant_folder_t(false)(
            builder::make_max(1, builder::make_max(a, 2)))
                        ->equals(builder::make_max(a, 2)));
    // special values
    EXPECT_TRUE(constant_folder_t(false)(
            (0 + a) + (a - 0) + (a * 1) + (a * 0) + (a / 1))
                        ->equals(a + a + a + a));
    EXPECT_TRUE(constant_folder_t(false)((b && expr(true)) || expr(true))
                        ->equals(expr(true)));

    // special exprs
    EXPECT_TRUE(constant_folder_t(false)((a - 12) - (a - 12))->equals(expr(0)));
    EXPECT_TRUE(constant_folder_t(false)((b && b) || (b || b))->equals(b));
    EXPECT_TRUE(constant_folder_t(false)(a + c - a)->equals(c));
    EXPECT_TRUE(constant_folder_t(false)(c + a - a)->equals(c));
}

static const uint32_t lanes = 4;
TEST(GCCore_CPU_const_fold_cpp, TestConstVectorCompute) {
    constant_folder_t f;

    auto make_constf = [](std::initializer_list<float> v,
                               sc_data_type_t type = sc_data_type_t::f32(4)) {
        std::vector<union_val> val;
        for (auto i : v) {
            val.push_back(i);
        }
        return make_expr<constant_node>(val, type);
    };
    auto make_consti = [](std::initializer_list<int> v) {
        std::vector<union_val> val;
        for (auto i : v) {
            val.push_back(static_cast<int64_t>(i));
        }
        return make_expr<constant_node>(val, sc_data_type_t::s32(lanes));
    };
    auto make_constb = [](std::initializer_list<bool> v) {
        std::vector<union_val> val;
        for (auto i : v) {
            val.push_back(static_cast<uint64_t>(i));
        }
        return make_expr<constant_node>(val, sc_data_type_t::boolean(lanes));
    };
    constant ca_vec_s32 = make_consti({1, 2, 3, 4});
    constant cb_vec_s32 = make_consti({0});
    constant ca_vec_f32 = make_constf({1.0f, 2.0f, 3.0f, 4.0f});
    constant cb_vec_f32 = make_constf({0.0f});

    constant all_true = make_constb({true});
    constant all_false = make_constb({false});

    EXPECT_TRUE(f(ca_vec_s32 + cb_vec_s32)->equals(ca_vec_s32));
    EXPECT_TRUE(f(ca_vec_s32 - cb_vec_s32)->equals(ca_vec_s32));
    EXPECT_TRUE(f(ca_vec_s32 - ca_vec_s32)->equals(cb_vec_s32));
    EXPECT_TRUE(f(ca_vec_s32 * cb_vec_s32)->equals(cb_vec_s32));
    EXPECT_TRUE(f(cb_vec_s32 / ca_vec_s32)->equals(cb_vec_s32));
    EXPECT_TRUE(f(cb_vec_s32 % ca_vec_s32)->equals(cb_vec_s32));
    EXPECT_TRUE(f(ca_vec_s32 > cb_vec_s32)->equals(all_true));
    EXPECT_TRUE(f(ca_vec_s32 >= cb_vec_s32)->equals(all_true));
    EXPECT_TRUE(f(ca_vec_s32 == cb_vec_s32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_s32 < cb_vec_s32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_s32 <= cb_vec_s32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_s32 != cb_vec_s32)->equals(all_true));
    EXPECT_TRUE(
            f(builder::make_max(ca_vec_s32, cb_vec_s32))->equals(ca_vec_s32));
    EXPECT_TRUE(
            f(builder::make_min(ca_vec_s32, cb_vec_s32))->equals(cb_vec_s32));

    EXPECT_TRUE(f(ca_vec_f32 + cb_vec_f32)->equals(ca_vec_f32));
    EXPECT_TRUE(f(ca_vec_f32 - cb_vec_f32)->equals(ca_vec_f32));
    EXPECT_TRUE(f(ca_vec_f32 * cb_vec_f32)->equals(cb_vec_f32));
    EXPECT_TRUE(f(cb_vec_f32 / ca_vec_f32)->equals(cb_vec_f32));
    EXPECT_TRUE(f(ca_vec_f32 > cb_vec_f32)->equals(all_true));
    EXPECT_TRUE(f(ca_vec_f32 >= cb_vec_f32)->equals(all_true));
    EXPECT_TRUE(f(ca_vec_f32 == cb_vec_f32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_f32 < cb_vec_f32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_f32 <= cb_vec_f32)->equals(all_false));
    EXPECT_TRUE(f(ca_vec_f32 != cb_vec_f32)->equals(all_true));
    EXPECT_TRUE(
            f(builder::make_max(ca_vec_f32, cb_vec_f32))->equals(ca_vec_f32));
    EXPECT_TRUE(
            f(builder::make_min(ca_vec_f32, cb_vec_f32))->equals(cb_vec_f32));

    // compute complex result
    EXPECT_TRUE(f(make_constf({0, 1, 3, 4}) < make_constf({1, 2, 4, 3}))
                        ->equals(make_constb({true, true, true, false})));

    // compute complex result
    EXPECT_TRUE(f(make_constf({0, 1, 3, 4}) + make_constf({1, 2, 4, 3}))
                        ->equals(make_constf({1, 3, 7, 7})));

    EXPECT_TRUE(f(builder::make_cast(
                          sc_data_type_t::s32(4), make_constf({0, 1, 3, 4})))
                        ->equals(make_consti({0, 1, 3, 4})));

    // const f32=>bf16 cast
    EXPECT_TRUE(
            f(builder::make_cast(
                      sc_data_type_t::bf16(lanes), make_constf({1.0f})))
                    ->equals(make_constf({1.0f}, sc_data_type_t::bf16(lanes))));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldwithPolynomialExpansion) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var vb = make_expr<var_node>(datatypes::s32, "b");
    var vd = make_expr<var_node>(datatypes::s32, "d");
    var iva = make_expr<var_node>(datatypes::index, "a");
    var ivb = make_expr<var_node>(datatypes::index, "b");
    var ivd = make_expr<var_node>(datatypes::index, "d");
    //
    constant_folder_t f {false};
    ir_comparer cmper(false, true, true);
    expr tmp;

    // ((a+b)*20+d)*30 = a*600+b*600+d*30
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((va + vb) * expr(20) + vd) * expr(30), 2),
            va * expr(600) + vb * expr(600) + vd * expr(30)));
    // (a + 5) / 10 = (a + 5) / 10 (sint div remain same)
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((va + expr(5)) / expr(10)), 2), //
            ((va + expr(5)) / expr(10))));
    // (a - 26) % 3 = ((a % 3) - 2) % 3
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((va - expr(26)) % expr(3)), 2), //
            ((va % expr(3) - expr(2)) % expr(3))));
    // (a - 26) % 3 = (a - 26) % 3 (skip sint mod expand)
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((va - expr(26)) % expr(3)), 2, true), //
            ((va - expr(26)) % expr(3))));

    // ((ia+ib)*20+id)*30 = ia*600+ib*600+id*30
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(
                    ((iva + ivb) * expr(20UL) + ivd) * expr(30UL), 2, true),
            iva * expr(600UL) + ivb * expr(600UL) + ivd * expr(30UL)));
    // (ia + 5) / 10 = (ia + 5) / 10 (uint div remain same)
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((iva + expr(5UL)) / expr(10UL)), 2), //
            ((iva + expr(5UL)) / expr(10UL))));
    // (ia - 26) % 3 = (ia - 26) % 3 (uint mod remain same)
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((iva - expr(26UL)) % expr(3UL)), 2), //
            ((iva - expr(26UL)) % expr(3UL))));
    // (ia + 26) % 3 = ((a % 3) + 2) % 3
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((iva + expr(26UL)) % expr(3UL)), 2), //
            (((iva % expr(3UL)) + expr(2UL)) % expr(3UL))));
    // (ia + 26) % 3 = (ia + 26) % 3 (skip uint mod expand)
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((iva + expr(26UL)) % expr(3UL)), 2, true), //
            ((iva + expr(26UL)) % expr(3UL))));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldSuccessiveDiv) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var fa = make_expr<var_node>(datatypes::f32, "fa");
    var ua = make_expr<var_node>(datatypes::index, "ua");

    constant_folder_t f;
    ir_comparer cmper(false, true, true);
    expr tmp;

    EXPECT_TRUE(cmper.compare(f(va / 2 / 3), va / 6));
    EXPECT_TRUE(cmper.compare(f(fa / 1.4f / 1.5f / 1.6f), fa / 3.36f));
    EXPECT_TRUE(cmper.compare(f(ua / UINT64_C(128) / UINT64_C(512) % UINT64_C(2)
                                      / UINT64_C(64) / UINT64_C(1024)),
            ua / UINT64_C(65536) % UINT64_C(2) / UINT64_C(65536)));
}

#define U64(c) UINT64_C(c)

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldRange) {
    builder::ir_builder_t bld;
    _function_(datatypes::void_t, aaa) {
        _var_(a, datatypes::index);
        a = U64(1);

        _var_(c, datatypes::index);
        c = U64(1);
        c = U64(2); // not single assign

        _var_(d, datatypes::index);
        d = c;
        _for_(i, UINT64_C(0), UINT64_C(10)) {
            _var_ex_(b, datatypes::index, linkage::local, a * U64(2) + i);

            _if_(i < U64(100)) {
                d = builder::make_select(b < U64(100), U64(10), U64(20));
            }
            d = builder::make_select(b < U64(10), U64(10), U64(20));
            d = builder::make_select(b < U64(2), U64(10), U64(20));
            d = builder::make_select(
                    (i / U64(5) + U64(2)) / U64(4) == U64(0), U64(10), U64(20));
            d = builder::make_select(
                    d % (i + U64(10)) / U64(10) == U64(0), U64(10), U64(20));
            d = i % U64(10);
        }
    }
    constant_folder_t f {false};
    ir_comparer cmper {true};
    _function_(datatypes::void_t, expected) {
        _var_(a, datatypes::index);
        a = U64(1);

        _var_(c, datatypes::index);
        c = U64(1);
        c = U64(2); // not single assign

        _var_(d, datatypes::index);
        d = c;
        _for_(i, UINT64_C(0), UINT64_C(10)) {
            _var_ex_(b, datatypes::index, linkage::local, i + U64(2));

            bld.push_scope();
            { d = U64(10); }
            bld.emit(bld.pop_scope());
            d = builder::make_select(b < U64(10), U64(10), U64(20));
            d = U64(20);
            d = U64(10);
            d = U64(10);
            d = i;
        }
    }
    EXPECT_TRUE(cmper.compare(f(aaa), expected, false));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldRangeGEGT) {
    builder::ir_builder_t bld;
    _function_(datatypes::void_t, aaa) {
        _var_(p, datatypes::s32);
        _for_(i, UINT64_C(0), UINT64_C(3)) {
            _if_(i + U64(5) >= U64(7)) { p = 1; }
            _else_ { p = 2; }
            _if_(i + U64(5) > U64(7)) { p = 1; }
            _else_ { p = 2; }
        }
    }
    constant_folder_t f {false};
    auto out = f(aaa);

    _function_(datatypes::void_t, expected) {
        _var_(p, datatypes::s32);
        _for_(i, UINT64_C(0), UINT64_C(3)) {
            _if_(i + U64(5) >= U64(7)) { p = 1; }
            _else_ { p = 2; }
            bld.push_scope();
            { p = 2; }
            bld.emit(bld.pop_scope());
        }
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_const_fold_cpp, TestConstFoldElseBlock) {
    builder::ir_builder_t bld;
    _function_(datatypes::void_t, aaa) {
        _var_(p, datatypes::s32);
        _if_(false) { p = 1; }
        _else_ { p = expr(2) + expr(3); }
    }

    constant_folder_t f {true};
    auto out = f(aaa);

    _function_(datatypes::void_t, expected) {
        _var_(p, datatypes::s32);
        bld.push_scope();
        { p = 5; }
        bld.emit(bld.pop_scope());
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
