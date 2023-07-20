/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
constexpr auto s32 = datatypes::s32;

TEST(GCCore_CPU_ssa_transform, TestSSATransform) {
    builder::ir_builder_t builder;
    func_t print_int_f = builder::make_func("print_int",
            {builder::make_var(s32, "v")}, stmt(), datatypes::void_t);
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(v, s32);
        _var_(b, s32);
        _var_init_(c, s32, 20);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        g = 1;
        v = A[0]; // indexing for read
        v = (v + 1) * (v * 2); // complex expr
        b = 1; // check phi
        c = b + v;
        a = a + 1; // assign and read params
        _if_(v < 10) {
            v = 3 + A[1];
            b = 2; // check phi
            A[v + 3] = A[b]; // indexing for read & write
        }
        _tensor_(gtsr, s32, 100);
        builder::get_current_builder()
                ->get_current_scope()
                .body.back()
                .checked_as<define>()
                ->init_
                = builder::tensor_ptr(A, {100}); // simulate global tensor
        gtsr[b] = a;
        _evaluate_call_(print_int_f, a);
        _return_(v + b + c + a + g);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    auto get_expected = [&print_int_f](bool aftergc) {
        _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
            _bind_(A, a);
            if (!aftergc) { _var_init_(c, s32, 20); }
            _var_(g, s32);
            g->attr()[attr_keys::module_global_offset] = size_t(1);
            _var_init_(t0, s32, 1);
            g = t0;
            _var_init_(t1, s32, 0);
            _var_init_(v0, s32, A[t1]); // v = A[0]

            // start of v = (v + 1) * (v * 2);
            _var_init_(t3, s32, 1);
            _var_init_(t4, s32, v0 + t3);
            _var_init_(t5, s32, 2);
            _var_init_(t6, s32, v0 * t5);
            _var_init_(v1, s32, t4 * t6);
            // end of v = (v + 1) * (v * 2);

            _var_init_(b2, s32, 1); // b = 1; // check phi
            _var_init_(c3, s32, b2 + v1); // c = b + v;

            _var_init_(t10, s32, 1);
            _var_init_(a4, s32, a + t10); // a = a + 1;

            _var_init_(t12, s32, 10);
            // start of if(){}
            expr b6_, v5_;
            _var_init_(t13, datatypes::boolean, v1 < t12);
            _if_(t13) {
                _var_init_(t14, s32, 3);
                _var_init_(t15, s32, 1);
                _var_init_(t16, s32, A[t15]);
                _var_init_copy_(v5, s32,
                        t14 + t16); // v = 3 + A[1];
                _var_init_copy_(b6, s32, 2); // b=2

                _var_init_(t19, s32, 3);
                _var_init_(t20, s32, v5 + t19);
                _var_init_(t21, s32, A[b6]);
                A[t20] = t21; // indexing for read & write
            }

            _var_init_(t22, s32, builder::make_phi({b2, b6_}));
            _var_init_(t23, s32, builder::make_phi({v1, v5_}));

            _var_init_(t100, s32, 100);
            _tensor_(gtsr, s32, 100);
            builder::get_current_builder()
                    ->get_current_scope()
                    .body.back()
                    .checked_as<define>()
                    ->init_
                    = builder::tensor_ptr(A, {t100});
            gtsr[t22] = a4;

            _var_init_(teval, datatypes::void_t, print_int_f(a4));
            builder::get_current_builder()->push_evaluate(teval);
            _var_init_(t24, s32, t23 + t22);
            _var_init_(t25, s32, t24 + c3);
            _var_init_(t26, s32, t25 + a4);
            _var_init_(t_g, s32, g);
            _var_init_(t27, s32, t26 + t_g);
            _return_(t27);
        }
        return expected;
    };
    ir_comparer cmper {true};
    ASSERT_TRUE(cmper.compare(out, get_expected(false)));

    // find if GC works
    ssa_visitor_t ssa_gc;
    out = ssa_gc.top_level_dispatch(out);
    EXPECT_TRUE(cmper.compare(out, get_expected(true), false));
}

TEST(GCCore_CPU_ssa_transform, TestSSATransformIfElse) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(v, s32);
        _var_(b, s32);
        v = A[0];
        b = a;
        _if_(v < 10) {
            b = v + 2;
            A[v] = b;
        }
        _else_ {
            b = v + b;
            A[v] = b;
        }
        _return_(b);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(t1, s32, 0);
        _var_init_(v0, s32, A[t1]);
        _var_init_(t2, s32, 10);
        _var_init_(t3, datatypes::boolean, v0 < t2);
        expr b1_, b2_;
        _if_(t3) {
            _var_init_(t4, s32, 2);
            _var_init_copy_(b1, s32, v0 + t4);
            A[v0] = b1;
        }
        _else_ {
            _var_init_copy_(b2, s32, v0 + a);
            A[v0] = b2;
        }
        _var_init_(bout, s32, builder::make_phi({b1_, b2_}));
        _return_(bout);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
    // find if GC works
    ssa_visitor_t ssa_gc;
    out = ssa_gc.top_level_dispatch(out);
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ssa_transform, TestSSATransformAssignmentToCopy) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(v, s32);
        _var_(b, s32);
        v = A[0];
        b = a;
        _if_(v < 10) { b = v; }
        _return_(b);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(t1, s32, 0);
        _var_init_(v0, s32, A[t1]);
        _var_init_(t2, s32, 10);
        _var_init_(t3, datatypes::boolean, v0 < t2);
        expr b1_;
        _if_(t3) { _var_init_copy_(b1, s32, v0); }
        _var_init_(bout, s32, builder::make_phi({a, b1_}));
        _return_(bout);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ssa_transform, TestSSATransformLoopPhiAfterIf) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(v, s32);
        _var_(b, datatypes::index);
        b = UINT64_C(0);
        _for_(i, 0, 100) {
            _if_(i < 10) { b = i; }
            _else_ { A[i] = i; }
            A[i] = b;
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(b0, datatypes::index, UINT64_C(0));
        _var_init_(t1, s32, 0);
        _var_init_(t2, s32, 100);
        _var_init_(t3, s32, 1);
        auto b3 = builder::make_var(datatypes::index, "b3");
        _for_(i, t1, t2, t3) {
            _var_init_(t4, s32, 10);
            _var_init_(t5, datatypes::boolean, i < t4);
            _var_init_(b2, datatypes::index, builder::make_phi({b0, b3}, true));
            expr b1_;
            _if_(t5) { _var_init_copy_(b1, datatypes::index, i); }
            _else_ { A[i] = i; }
            builder::get_current_builder()->push_var_tensor_def(
                    b3, linkage::local, builder::make_phi({b1_, b2}));
            A[i] = b3;
        }
        _var_init_(b4, datatypes::index, builder::make_phi({b0, b3}));
        _var_init_(t10, s32, 0);
        _return_(t10);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ssa_transform, TestSSATransformForLoop) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(v, s32, 0);
        _var_(b, s32);
        _var_init_(c, s32, 0);
        b = a;
        _for_(i, 0, 100, 1) {
            _if_(c == c) {
                v = v + 1;
                b = v + builder::make_cast(s32, i);
                A[v] = b;
            }
            A[b] = c;
        }
        _return_(b);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    auto get_expected = [](bool aftergc) {
        _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
            _bind_(A, a);
            _var_init_(v, s32, 0);
            _var_init_(c, s32, 0);
            _var_init_(t0, s32, 0);
            _var_init_(t1, s32, 100);
            _var_init_(t2, s32, 1);
            auto v6 = builder::make_var(s32, "v6");
            auto b5 = builder::make_var(s32, "b5");
            expr b4_, b3_, v2_, c0_;
            _for_(i, t0, t1, t2) {
                _var_init_(v1, s32, builder::make_phi({v, v6}, true));
                _var_init_copy_(c0, s32, builder::make_phi({c}));
                _var_init_(tcmp, datatypes::boolean, c0 == c0);
                _var_init_copy_(b4, s32, builder::make_phi({a, b5}, true));
                _if_(tcmp) {
                    _var_init_(t3, s32, 1);
                    _var_init_copy_(v2, s32, v1 + t3);
                    _var_init_(t4, s32, builder::make_cast(s32, i));
                    _var_init_copy_(b3, s32, v2 + t4);
                    A[v2] = b3;
                }
                builder::get_current_builder()->push_var_tensor_def(
                        b5, linkage::local, builder::make_phi({b4, b3_}));
                builder::get_current_builder()->push_var_tensor_def(
                        v6, linkage::local, builder::make_phi({v1, v2_}));
                A[b5] = c0;
            }
            _var_init_(b6, s32, builder::make_phi({a, b5}));
            if (!aftergc) { _var_init_(v7, s32, builder::make_phi({v, v6})); }
            _return_(b6);
        }
        return expected;
    };
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, get_expected(false)));

    // find if GC works
    ssa_visitor_t ssa_gc;
    out = ssa_gc.top_level_dispatch(out);
    ASSERT_TRUE(cmper.compare(out, get_expected(true)));
}

// test for nested if+for
TEST(GCCore_CPU_ssa_transform, TestSSATransformForLoop2) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(v, s32, 0);
        _var_init_(c, s32, 0);
        _for_(i, 0, a, 1) {
            _if_(c == c) {
                _if_(c == c) { v = v + 1; }
                _else_ { v = v + 2; }
            }
        }
        _return_(v);
    }
    ssa_transform_t s;
    auto out = s(ccc);
    auto get_expected = [](bool aftergc) {
        _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
            _bind_(A, a);
            _var_init_(v, s32, 0);
            _var_init_(c, s32, 0);
            _var_init_(t0, s32, 0);
            _var_init_(t1, s32, 1);
            auto v6 = builder::make_var(s32, "v6");
            expr c0_;
            _for_(i, t0, a, t1) {
                expr v5_;
                _var_init_(v1, s32, builder::make_phi({v, v6}, true));
                _var_init_copy_(c0, s32, builder::make_phi({c}));
                _var_init_(t3, datatypes::boolean, c0 == c0);
                _if_(t3) {
                    _var_init_(t4, datatypes::boolean, c0 == c0);
                    expr v2_, v4_;
                    _if_(t4) {
                        _var_init_(t6, s32, 1);
                        _var_init_copy_(v2, s32, v1 + t6);
                    }
                    _else_ {
                        _var_init_(t6, s32, 2);
                        _var_init_copy_(v4, s32, v1 + t6);
                    }
                    _var_init_copy_(v5, s32, builder::make_phi({v2_, v4_}));
                }
                builder::get_current_builder()->push_var_tensor_def(
                        v6, linkage::local, builder::make_phi({v1, v5_}));
            }
            _var_init_(v7, s32, builder::make_phi({v, v6}));
            _return_(v7);
        }
        return expected;
    };
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, get_expected(false)));
    // find if GC works
    ssa_visitor_t ssa_gc;
    out = ssa_gc.top_level_dispatch(out);
    ASSERT_TRUE(cmper.compare(out, get_expected(true)));
}

// test for nested if+for
TEST(GCCore_CPU_ssa_transform, TestSSATransformGCReferenced) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(v, s32, 0);
        A[0] = v;
        _var_init_(b, s32, v);
    }
    ssa_transform_t s;
    ssa_visitor_t ssa_gc;
    auto out = s(ccc);
    out = ssa_gc.top_level_dispatch(out);
    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(v, s32, 0);
        _var_init_(tmp, s32, 0);
        A[tmp] = v;
    }
    ir_comparer cmper {true};
    ASSERT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ssa_transform, TestSameVarCompare) {
    builder::ir_builder_t builder;
    // Same named var should be renamed by simpilfier
    // However, ssa_transform need to tell apart different var node
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _for_(iter, 0, 100) {
            auto &l1_iter = iter;
            _for_(iter, 0, 100) {
                auto &l2_iter = iter;
                A[l1_iter + l2_iter] = l1_iter + l2_iter;
            }
        }
    }
    ssa_transform_t s;
    auto out = s(ccc);
    _function_(
            datatypes::void_t, expected, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _var_init_(t0, datatypes::s32, 0);
        _var_init_(t1, datatypes::s32, 100);
        _var_init_(t2, datatypes::s32, 1);
        _for_(iter, t0, t1, t2) {
            auto &l1_iter = iter;
            _var_init_(t3, datatypes::s32, 0);
            _var_init_(t4, datatypes::s32, 100);
            _var_init_(t5, datatypes::s32, 1);
            _for_(iter, t3, t4, t5) {
                auto &l2_iter = iter;
                _var_init_(it0, datatypes::index,
                        builder::make_phi({l1_iter}, false));
                _var_init_(t7, datatypes::index, it0 + l2_iter);
                _var_init_(t8, datatypes::index, it0 + l2_iter);
                A[t7] = t8;
            }
        }
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ssa_transform, TestMultiLevelForPHI) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _var_init_(a, datatypes::s32, 1);
        _for_(x, 0, 100, 1) {
            _for_(y, 0, 100, 1) {
                _for_(i, 0, 100, 1) {
                    _if_(i == 0) { A[i] = a; }
                }
                _for_(j, 0, 100, 1) { a = a + 1; }
            }
        }
    }
    ssa_transform_t s;
    auto out = s(ccc);
    _function_(
            datatypes::void_t, expected, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _var_init_(a, datatypes::s32, 1);
        _var_init_(t0, datatypes::s32, 0);
        _var_init_(t1, datatypes::s32, 100);
        _var_init_(t2, datatypes::s32, 1);
        auto a4 = builder::make_var(datatypes::s32, "a4");
        auto a5 = builder::make_var(datatypes::s32, "a5");
        auto a6 = builder::make_var(datatypes::s32, "a6");
        _for_(x, t0, t1, t2) {
            _var_init_(a0, datatypes::s32, builder::make_phi({a, a6}, true));
            _var_init_(t3, datatypes::s32, 0);
            _var_init_(t4, datatypes::s32, 100);
            _var_init_(t5, datatypes::s32, 1);
            _for_(y, t3, t4, t5) {
                _var_init_(
                        a1, datatypes::s32, builder::make_phi({a0, a5}, true));
                _var_init_(t6, datatypes::s32, 0);
                _var_init_(t7, datatypes::s32, 100);
                _var_init_(t8, datatypes::s32, 1);
                _for_(i, t6, t7, t8) {
                    _var_init_(
                            a2, datatypes::s32, builder::make_phi({a1}, false));
                    _var_init_(cmp0, datatypes::s32, 0);
                    _var_init_(cmpv, datatypes::boolean, i == cmp0);
                    _if_(cmpv) { A[i] = a2; }
                }
                _var_init_(t9, datatypes::s32, 0);
                _var_init_(t10, datatypes::s32, 100);
                _var_init_(t11, datatypes::s32, 1);
                _for_(j, t9, t10, t11) {
                    _var_init_(a3, datatypes::s32,
                            builder::make_phi({a1, a4}, true));
                    _var_init_(t12, datatypes::s32, 1);
                    builder.push_var_tensor_def(a4, linkage::local, a3 + t12);
                }

                builder.push_var_tensor_def(
                        a5, linkage::local, builder::make_phi({a1, a4}));
            }
            builder.push_var_tensor_def(
                    a6, linkage::local, builder::make_phi({a0, a5}));
        }
        builder.push_var_tensor_def(
                a6, linkage::local, builder::make_phi({a, a6}));
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
