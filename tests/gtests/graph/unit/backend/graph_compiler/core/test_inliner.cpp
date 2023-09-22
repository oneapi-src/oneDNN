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
#include <compiler/ir/transform/func_inline.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
TEST(GCCore_CPU_func_inline_cpp, TestInlineAt) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222, 333}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(D, datatypes::f32, 10, 20);
        _tensor_(F, datatypes::f32, 100, len);
        _for_(i, 0, len) {
            A[{i, 10}] = B[{10, i, 4}] + D[{1, i}] + F[{11, i}];
        }
        _return_(12);
    }

    auto get_func = [&aaa]() -> func_t {
        _function_(datatypes::s32, mainfunc,
                _arg_("A1", datatypes::f32, {123, 321}),
                _arg_("B1", datatypes::f32, {111, 222, 333}),
                _arg_("len1", datatypes::s32)) {
            _bind_(A, B, len);
            A[{0, 0}] = 1;
            A[{0, 0}] = with_attr(aaa(A, B, len), "inline_level", 2);
        }
        return mainfunc;
    };

    func_t mainfunc = get_func();
    stmts body = mainfunc->body_.as<stmts>();
    assign the_assign = body->seq_[1].as<assign>();
    call callnode = the_assign->value_.as<call>();
    func_inliner_t inliner;
    the_assign->value_
            = inliner.inline_at(callnode, body->seq_, 1).remove_const();

    _function_(datatypes::s32, reference,
            _arg_("A1", datatypes::f32, {123, 321}),
            _arg_("B1", datatypes::f32, {111, 222, 333}),
            _arg_("len1", datatypes::s32)) {
        _bind_(A, B, len);
        A[{0, 0}] = 1;
        _var_(retval, datatypes::s32);
        builder.push_scope();
        {
            _tensor_(D, datatypes::f32, 10, 20);
            _tensor_(F, datatypes::f32, 100, len);
            _for_(i, 0, len) {
                A[{i, 10}] = B[{10, i, 4}] + D[{1, i}] + F[{11, i}];
            }
            retval = 12;
        }
        builder.emit(builder.pop_scope());
        A[{0, 0}] = retval;
    }

    ir_comparer cmp;
    EXPECT_TRUE(cmp.compare(mainfunc, reference));

    func_inliner_t inl;
    func_c f = get_func();
    f = inl(f);
    EXPECT_TRUE(cmp.compare(f, reference));
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineTensor) {
    builder::ir_builder_t builder;

    _function_(
            datatypes::void_t, aaa, _arg_("A", datatypes::f32, {100 * 200})) {
        _bind_(A);
        _for_(i, 0, 100) {
            _for_(j, 0, 200) { A[i * 200 + j] = A[i * 200 + j] + 1; }
        }
        _var_init_(ptr, datatypes::pointer, builder::tensor_ptr(A, {0}));
    }

    _function_(datatypes::s32, mainfunc,
            _arg_("A1", datatypes::f32, {200 * 400})) {
        _bind_(A);
        A[0] = 1;
        builder.push_evaluate(
                with_attr(aaa(tensor_ptr(A, {100})), "inline_level", 2));
    }
    func_inliner_t inl;

    _function_(datatypes::s32, expected,
            _arg_("A1", datatypes::f32, {200 * 400})) {
        _bind_(A);
        A[0] = 1;
        builder.push_scope();
        {
            _for_(i, 0, 100) {
                _for_(j, 0, 200) {
                    A[100 + (i * 200 + j)] = A[100 + (i * 200 + j)] + 1;
                }
            }
        }
        _var_init_(ptr, datatypes::pointer,
                builder::tensor_ptr(A, {100 + expr(0)}));
        builder.emit(builder.pop_scope());

        builder.push_scope();
        builder.emit(builder.pop_scope());
    }
    auto out = inl(mainfunc);
    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(out, expected, false));
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineFailure) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, aaa) {
        _for_(i, 0, 100) { _return_(12); }
        _return_(12);
    }
    func_inliner_t inliner;
    std::vector<stmt> seq;
    EXPECT_SC_ERROR(inliner.inline_at(aaa().as<call_c>(), seq, 0),
            "return_node should be the last statement in the IR, got");
    _function_(datatypes::s32, aaa2) {
        _return_(12);
        builder.push_evaluate(1);
    }
    EXPECT_SC_ERROR(inliner.inline_at(aaa2().as<call_c>(), seq, 0),
            "return_node should be the last statement in the IR, got");

    _function_(datatypes::void_t, aaa3) {
        _var_(a, datatypes::s32);
        _return_(12);
    }
    EXPECT_SC_ERROR(inliner.inline_at(aaa3().as<call_c>(), seq, 0),
            "The function to inline returns a value, but ret_var_ is");
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineSingleExpr) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(v + 1);
    }
    func_inliner_t inliner;
    _function_(datatypes::void_t, aaa) {
        _var_(a, datatypes::s32);
        a = with_attr(add1(123), "inline_level", 2);
    }

    _function_(datatypes::void_t, expected) {
        _var_(a, datatypes::s32);
        a = 123 + expr(1);
    }

    auto bbb = inliner(aaa);
    ir_comparer cmp;
    EXPECT_TRUE(cmp.compare(bbb, expected));
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineNestedSimple) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(v + 1);
    }

    _function_(datatypes::s32, sub1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(with_attr(add1(v), "inline_level", 2) - 1);
    }
    func_inliner_t inliner;
    _function_(datatypes::void_t, aaa) {
        _var_(a, datatypes::s32);
        a = with_attr(sub1(123), "inline_level", 2);
    }

    _function_(datatypes::void_t, expected) {
        _var_(a, datatypes::s32);
        a = expr(123) + expr(1) - 1;
    }

    auto bbb = inliner(aaa);
    ir_comparer cmp;
    EXPECT_TRUE(cmp.compare(bbb, expected));
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineNested) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _var_(b, datatypes::s32);
        b = v + 1;
        _return_(b + 2);
    }

    _function_(datatypes::s32, sub1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _var_(b, datatypes::s32);
        b = with_attr(add1(v), "inline_level", 2);
        _return_(b - 1);
    }

    func_inliner_t inliner;
    _function_(datatypes::void_t, aaa) {
        _var_(a, datatypes::s32);
        a = with_attr(sub1(a), "inline_level", 2);
    }

    _function_(datatypes::void_t, expected) {
        _var_(a, datatypes::s32);
        _var_(ret1, datatypes::s32);
        builder.push_scope();
        {
            _var_(localv, datatypes::s32);
            _var_(ret2, datatypes::s32);
            builder.push_scope();
            {
                _var_(b, datatypes::s32);
                b = a + 1;
                ret2 = b + 2;
            }
            builder.emit(builder.pop_scope());
            localv = ret2;
            ret1 = localv - 1;
        }
        builder.emit(builder.pop_scope());
        a = ret1;
    }

    auto bbb = inliner(aaa);
    ir_comparer cmp;
    EXPECT_TRUE(cmp.compare(bbb, expected, false));
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineNestedExceed) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {}

    _function_(datatypes::s32, sub1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _var_(b, datatypes::s32);
        b = with_attr(add1(v), "inline_level", 2);
        _return_(b - 1);
    }

    builder.push_scope();
    {
        expr::lvalue_proxy_t v(add1->params_[0], false);
        _return_(with_attr(sub1(v), "inline_level", 2));
    }
    add1->body_ = builder.pop_scope();

    func_inliner_t inliner;
    EXPECT_SC_ERROR(inliner(add1), "Reached max inline recursion depth");
    // clear the body, or there is a loop in shared_ptr
    add1->body_.checked_as<stmts>()->seq_.clear();
}

TEST(GCCore_CPU_func_inline_cpp, TestInlineCorrectDecl) {
    builder::ir_builder_t builder;

    _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(v + 1);
    }

    func_t add2;
    {
        _function_(datatypes::s32, add1, _arg_("v", datatypes::s32)) {
            _bind_(v);
            _return_(v + 2);
        }
        add2 = add1;
    }

    _function_(datatypes::s32, sub1, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(with_attr(add1(v), "inline_level", 2) - 1);
    }
    auto mod = std::make_shared<ir_module_t>(
            get_default_context(), std::vector<func_t> {sub1});
    mod->add_func({add1});
    // simulate a pass that changes add1 to add2, without changing the reference
    // in sub1
    mod->get_contents()[1] = add2;
    func_inliner_t inliner;

    _function_(datatypes::s32, expected, _arg_("v", datatypes::s32)) {
        _bind_(v);
        _return_(v + 2 - 1);
    }

    auto bbb = inliner(mod);
    ir_comparer cmp;
    EXPECT_TRUE(cmp.compare(bbb->get_contents()[0], expected));
}
