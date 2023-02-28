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

TEST(GCCore_const_fold_cpp, TestConstCompute) {
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

TEST(GCCore_const_fold_cpp, TestConstFoldRotation) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    constant_folder_t f;
    ir_comparer cmper(false, true, true);
    expr tmp;
    // c + x => x + c
    EXPECT_TRUE(cmper.compare(f(2 + va), va + 2));
    EXPECT_NO_CHANGE(va + 2);
    EXPECT_NO_CHANGE(2 - va);

    // (x + c1) + c2 => x + (c1 + c2)
    EXPECT_TRUE(cmper.compare(f((va + 2) + 3), va + 5));

    // (x + c) + y => (x + y) + c
    EXPECT_TRUE(cmper.compare(f((va + 2) + va), va + va + 2));
    EXPECT_NO_CHANGE(va + va + 3);

    // x + (y + c) => (x + y) + c
    EXPECT_TRUE(cmper.compare(f(va + (va + 2)), va + va + 2));
    EXPECT_NO_CHANGE(va + (va - 2));
    EXPECT_NO_CHANGE(va + va + va);

    // (x + c1) + (y + c2) => (x + y) + (c1 + c2)
    EXPECT_TRUE(cmper.compare(f((va + 4) + (va + 2)), va + va + 6));
    EXPECT_TRUE(cmper.compare(f((va + 4) + (va - 2)), va + (va - 2) + 4));
}

TEST(GCCore_const_fold_cpp, TestConstFoldSpecialConst) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var fa = make_expr<var_node>(datatypes::f32, "fa");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    var vc = make_expr<var_node>(datatypes::boolean, "c");
    constant_folder_t f;
    ir_comparer cmper(false, true, true);
    expr tmp;

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
}

TEST(GCCore_const_fold_cpp, TestConstFoldSpecialExpr) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var fa = make_expr<var_node>(datatypes::f32, "fa");
    var vb = make_expr<var_node>(datatypes::boolean, "b");
    var vc = make_expr<var_node>(datatypes::boolean, "c");
    constant_folder_t f;
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

TEST(GCCore_const_fold_cpp, TestConstFoldMutiLevel) {
    var a = make_expr<var_node>(datatypes::s32, "a");
    var b = make_expr<var_node>(datatypes::boolean, "b");
    EXPECT_TRUE(constant_folder_t()(a + 2 + (a + 4) * 2 + (4 + a * 0))
                        ->equals(a + (a + 4) * 2 + 6));
    EXPECT_TRUE(constant_folder_t()((2 + a) + (4 + a) + (6 + a))
                        ->equals(a + a + a + 12));
    EXPECT_TRUE(
            constant_folder_t()(builder::make_max(1, builder::make_max(a, 2)))
                    ->equals(builder::make_max(a, 2)));
    // special values
    EXPECT_TRUE(
            constant_folder_t()((0 + a) + (a - 0) + (a * 1) + (a * 0) + (a / 1))
                    ->equals(a + a + a + a));
    EXPECT_TRUE(constant_folder_t()((b && expr(true)) || expr(true))
                        ->equals(expr(true)));

    // special exprs
    EXPECT_TRUE(constant_folder_t()((a - 12) - (a - 12))->equals(expr(0)));
    EXPECT_TRUE(constant_folder_t()((b && b) || (b || b))->equals(b));
    EXPECT_TRUE(constant_folder_t()(a + b - a)->equals(b));
    EXPECT_TRUE(constant_folder_t()(b + a - a)->equals(b));
}

static const uint32_t lanes = 4;
TEST(GCCore_const_fold_cpp, TestConstVectorCompute) {
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

TEST(GCCore_const_fold_cpp, TestConstFoldwithPolynomialExpansion) {
    var va = make_expr<var_node>(datatypes::s32, "a");
    var vb = make_expr<var_node>(datatypes::s32, "b");
    var vd = make_expr<var_node>(datatypes::s32, "d");
    constant_folder_t f;
    ir_comparer cmper(false, true, true);
    expr tmp;

    // ((a+b)*20+d)*30 = a*600+b*600+d*30
    EXPECT_TRUE(cmper.compare(
            f.expand_polynomial(((va + vb) * expr(20) + vd) * expr(30), 2),
            va * expr(600) + vb * expr(600) + vd * expr(30)));
}

TEST(GCCore_const_fold_cpp, TestConstFoldSuccessiveDiv) {
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

TEST(GCCore_const_fold_cpp, TestConstFoldRange) {
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

TEST(GCCore_const_fold_cpp, TestConstFoldRangeGEGT) {
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
