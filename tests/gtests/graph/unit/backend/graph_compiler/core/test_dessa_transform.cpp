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
#include <compiler/ir/transform/dessa_transform.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
constexpr auto s32 = datatypes::s32;

TEST(GCCore_CPU_dessa_transform, TestDeSSATransform) {
    builder::ir_builder_t builder;
    func_t print_int_f = builder::make_func("print_int",
            {builder::make_var(s32, "v")}, stmt(), datatypes::void_t);
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(c, s32, 20);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        g = 1;
        _if_(a < 10) { a = a + 1; }
        A[0] = a;
        _evaluate_call_(print_int_f, a);
        _return_(a + c);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    dessa_transform_t de;
    out = de(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(a0, s32);
        _var_init_(a1, s32, a);
        _var_init_(c, s32, 20);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        _var_init_(t0, s32, 1);
        g = t0;
        _var_init_(t1, s32, 10);
        _var_init_(t2, datatypes::boolean, a < t1);
        _if_(t2) {
            _var_init_(t3, s32, 1);
            a0 = a + t3;
            a1 = a0;
        }
        _var_init_(t6, s32, 0);
        A[t6] = a1;
        _evaluate_call_(print_int_f, a1);
        _var_init_(t7, s32, a1 + c);
        _return_(t7);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_dessa_transform, TestDeSSATransformForLostCopy) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(c, s32);
        _var_(d, s32);
        d = 0;
        _for_(i, 0, 10) {
            c = d;
            d = d + 1;
            A[0] = c;
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    dessa_transform_t de;
    out = de(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(c2, s32);
        _var_(c2_s, s32);
        _var_init_(d0, s32, 0);
        c2_s = d0;
        _var_init_(t9, s32, 0);
        _var_init_(t10, s32, 10);
        _var_init_(t11, s32, 1);
        _for_(i, t9, t10, t11) {
            c2 = c2_s;
            _var_init_(t13, s32, 1);
            _var_init_(d3, s32, c2 + t13);
            c2_s = d3;
            _var_init_(t15, s32, 0);
            A[t15] = c2;
        }
        _var_init_(t18, s32, 0);
        _return_(t18);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_dessa_transform, TestDeSSATransformForNoRedundantCopy) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(c, s32);
        _var_init_(d, s32, 1);
        _for_(i, 0, 10) {
            _if_(d == 0) { c = 1; }
            _else_ { c = 2; }
            A[0] = c;
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    dessa_transform_t de;
    out = de(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(d, s32, 1);
        _var_init_(t9, s32, 0);
        _var_init_(t10, s32, 10);
        _var_init_(t11, s32, 1);
        _for_(i, t9, t10, t11) {
            _var_(c1, s32);
            _var_(c2, s32);
            _var_(c3, s32);
            _var_init_(d0, s32, d);
            _var_init_(t23, s32, 0);
            _var_init_(t24, datatypes::boolean, d0 == t23);
            _if_(t24) {
                c1 = 1;
                c3 = c1;
            }
            _else_ {
                c2 = 2;
                c3 = c2;
            }
            _var_init_(t28, s32, 0);
            A[t28] = c3;
        }
        _var_init_(t29, s32, 0);
        _return_(t29);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_dessa_transform, TestDeSSATransformForSwapProblem) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_init_(c, s32, 20);
        _var_init_(d, s32, 1);
        d = 0;
        _for_(i, 0, 10) {
            _var_init_(tmp, s32, c);
            c = d;
            d = tmp;
        }
        _return_(c + d);
    }

    ssa_transform_t s;
    auto out = s(ccc);

    dessa_transform_t de;
    out = de(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(c1, s32);
        _var_(c1_s, s32);
        _var_(d4, s32);
        _var_(c3, s32);
        _var_(c3_s, s32);
        _var_(c5, s32);
        _var_(d6, s32);
        _var_init_(c, s32, 20);
        c1_s = c;
        c5 = c;
        _var_init_(d0, s32, 0);
        c3_s = d0;
        d6 = d0;
        _var_init_(t9, s32, 0);
        _var_init_(t10, s32, 10);
        _var_init_(t11, s32, 1);
        _for_(i, t9, t10, t11) {
            c1 = c1_s;
            c3 = c3_s;
            d4 = c1;
            c3_s = d4;
            d6 = d4;
            c1_s = c3;
            c5 = c3;
        }
        _var_init_(t40, s32, c5 + d6);
        _return_(t40);
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
