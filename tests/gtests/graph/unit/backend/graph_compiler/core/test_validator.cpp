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
#include <compiler/ir/pass/validator.hpp>

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

TEST(GCCore_CPU_validator_cpp, TestValidatorBinaryTensor) {
    validator_t vali;
    ir_builder_t builder;
    builder.push_scope();
    _tensor_(b, datatypes::f32, {1000});
    EXPECT_SC_ERROR(vali(tensor_ptr(b, {100, 200})),
            "Indexing node should have the same dimemsion of the tensor");
    EXPECT_SC_ERROR(vali(tensor_ptr(b, {100.0f})), "Expecting an integer");

    vali(tensor_ptr(b, {100}));
    vali(tensor_ptr(b, {100UL}));
}

TEST(GCCore_CPU_validator_cpp, TestValidatorBinaryBaseTest) {
    validator_t vali;
    ir_builder_t builder;
    builder.push_scope();
    const char *expected = "Invalid type: met undef/void";
    _var_(a, datatypes::void_t);
    _var_(b, datatypes::void_t);
    // void
    EXPECT_SC_ERROR(vali(a + b), expected);
    // undef
    EXPECT_SC_ERROR(vali(a + 0), expected);

    _var_(c, datatypes::s32);
    _var_(d, datatypes::index);
    expr tmp = c + d;
    tmp->dtype_ = datatypes::index;
    EXPECT_SC_ERROR(vali(tmp),
            "The types of LHS and RHS should be the same: s32 v.s. index, "
            "expr");
    EXPECT_SC_ERROR(vali(expr(1.23f) % 2.3f),
            "%% operator cannot be applied on this type: ");

    // zero-length vector
    _var_(e, sc_data_type_t::s32(0));
    _var_(f, sc_data_type_t::s32(0));
    EXPECT_SC_ERROR(vali(e + f), "met undef");

    vali(c % 10);
}

TEST(GCCore_CPU_validator_cpp, TestCmp) {
    validator_t vali;
    expr tmp = expr(1) == expr(2);
    tmp->dtype_ = datatypes::s32;
    EXPECT_SC_ERROR(vali(tmp),
            "The type of cmp should be boolean, got: s32. The expr is ");
    EXPECT_SC_ERROR(vali(expr(12) >= expr(14UL)),
            "The type of LHS and RHS should be the same: ");
    expr a = make_var(datatypes::pointer, "a");
    const char *expected
            = "comparison expressions should have valid type, got ";
    EXPECT_SC_ERROR(vali(a < a), expected);

    vali(expr(12) <= expr(14));
}

TEST(GCCore_CPU_validator_cpp, TestIntrinsics) {
    validator_t vali;
    auto a = make_min(1, 2.5f);
    EXPECT_SC_ERROR(vali(a), "Invalid type: met undef/void/zero-length vector");
    a->dtype_ = datatypes::bf16;
    EXPECT_SC_ERROR(vali(a), "The types of LHS and RHS should be the same");

    a = make_abs(1.23f);
    vali(a);
    a = make_floor(1.23f);
    vali(a);
    a = make_ceil(1.23f);
    vali(a);
    a = make_exp(1.23f);
    vali(a);

    {
        auto b = expr(1.2f) & 1.3f;
        EXPECT_SC_ERROR(vali(b), "int_and and int_or only supports ints, got");
    }
    {
        auto b = expr(1.2f) | 1.3f;
        EXPECT_SC_ERROR(vali(b), "int_and and int_or only supports ints, got");
    }
    a = make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4))
            | make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4));
    vali(a);
    a.as<intrin_call>()->type_ = intrin_type::int_and;
    vali(a);

    // SHL, SHR
    {
        auto b = expr(123) << 1.23f;
        EXPECT_SC_ERROR(vali(b), "operands of shl and shr should be ints, got");
        b = expr(123) >> 1.23f;
        EXPECT_SC_ERROR(vali(b), "operands of shl and shr should be ints, got");
        b = expr(12.f) << 1;
        EXPECT_SC_ERROR(vali(b), "operands of shl and shr should be ints, got");
        b = expr(12.f) >> 1;
        EXPECT_SC_ERROR(vali(b), "operands of shl and shr should be ints, got");
        b = make_constant({1UL, 2UL, 3UL, 4UL, 1UL, 2UL, 3UL, 4UL},
                    sc_data_type_t::s32(8))
                >> make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4));
        EXPECT_SC_ERROR(vali(b),
                "shl shr does not support A << B or A >> B where A is a scalar "
                "and B is a vector ");
        b = expr(12)
                >> make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4));
        EXPECT_SC_ERROR(vali(b),
                "shl shr does not support A << B or A >> B where A is a scalar "
                "and B is a vector ");
    }
    a = make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4))
            << make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4));
    vali(a);
    a = make_constant({1UL, 2UL, 3UL, 4UL}, sc_data_type_t::s32(4)) << 3;
    vali(a);
}

TEST(GCCore_CPU_validator_cpp, TestLogic) {
    validator_t vali;
    expr tmp = expr(true) || expr(false);
    tmp->dtype_ = datatypes::s32;
    EXPECT_SC_ERROR(vali(tmp), "The type of logic should be boolean, got: ");
    EXPECT_SC_ERROR(vali(expr(true) || expr(1)),
            "The type of RHS should be a boolean expr: ");
    EXPECT_SC_ERROR(vali(expr(1) || expr(true)),
            "The type of LHS should be a boolean expr: ");
    vali(expr(true) || expr(false));
}

TEST(GCCore_CPU_validator_cpp, TestLogicNot) {
    validator_t vali;
    logic_not tmp = make_logic_not(expr(true)).as<logic_not>();
    tmp->dtype_ = datatypes::s32;
    EXPECT_SC_ERROR(
            vali(tmp), "The type of logic not should be boolean, got: ");
    tmp = make_logic_not(expr(true)).as<logic_not>();
    tmp->in_ = expr(1);
    EXPECT_SC_ERROR(vali(tmp), "The type of in_ should be a boolean expr: ");
    vali(!expr(true));
}

TEST(GCCore_CPU_validator_cpp, TestSelect) {
    validator_t vali;
    dnnl::impl::graph::gc::select tmp
            = make_select(expr(1), expr(1), expr(0))
                      .as<dnnl::impl::graph::gc::select>();
    tmp->cond_ = expr(true);
    tmp->l_ = make_tensor("a", {16}, datatypes::f32);
    tmp->l_ = make_indexing(tmp->l_, {0}, 16);
    EXPECT_SC_ERROR(
            vali(tmp), "The two candidates in select should have same dtype");
    tmp->r_ = make_tensor("b", {16}, datatypes::f32);
    tmp->r_ = make_indexing(tmp->r_, {0}, 16);
    tmp->cond_ = builder::make_constant(16);
    EXPECT_SC_ERROR(vali(tmp),
            "When condition is bit mask, its number of bit should equal to "
            "number of left/right hand vector");
    tmp->cond_ = make_tensor("c", {16}, datatypes::boolean);
    tmp->cond_ = make_indexing(tmp->cond_, {0}, 16);
    vali(!expr(true));
}

TEST(GCCore_CPU_validator_cpp, TestIndexing) {
    validator_t vali;
    expr tsr = make_tensor("aaa", {100}, datatypes::f32);
    expr tmp = tsr[12];
    tmp.as<indexing>()->ptr_ = expr(10);
    EXPECT_SC_ERROR(vali(tmp),
            "Indexing node is expecting a tensor/tensorptr as the ptr:");
    tmp = tsr[10];
    tmp->dtype_ = datatypes::bf16;
    EXPECT_SC_ERROR(vali(tmp),
            "Indexing node should have the same type of the tensor element, "
            "got bf16 and f32");
    tmp = tsr[{10, 20}];
    EXPECT_SC_ERROR(vali(tmp),
            "Indexing node should have the same dimemsion of the tensor "
            "element, expecting 1, got 2");
    tmp = tsr[true];
    EXPECT_SC_ERROR(vali(tmp),
            "The 1-th index of the indexing expr has type bool. Expecting "
            "an integer: ");
    expr tsr2 = make_tensor("aaa", {100, 200}, datatypes::f32);
    EXPECT_SC_ERROR(vali(tsr2[{10, 20UL}]),
            "Expecting all the indices within the indexing expression "
            "having the same dtype. Current dimemsion: 2");

    vali(tsr[10]);
    vali(tsr[10UL]);
    vali(tsr2[{10, 20}]);
    vali(tsr2[{10UL, 20UL}]);
}

TEST(GCCore_CPU_validator_cpp, TestCall) {
    validator_t vali;
    ir_builder_t builder;
    _decl_func_(datatypes::undef, AAA, _arg_("len", datatypes::s32),
            _arg_("len2", datatypes::f32));
    EXPECT_SC_ERROR(vali(AAA(12)), "Met undef");
    AAA->ret_type_ = datatypes::index;
    EXPECT_SC_ERROR(
            vali(AAA(12)), "Wrong number of parameters, given 1, expecting 2");
    EXPECT_SC_ERROR(
            vali(AAA()), "Wrong number of parameters, given 0, expecting 2");
    EXPECT_SC_ERROR(vali(AAA(1, 2, 3)),
            "Wrong number of parameters, given 3, expecting 2");
    EXPECT_SC_ERROR(vali(AAA(1, 2UL)),
            "Unmatched types for parameter 2 : given index, expecting f32");
    expr tmp = AAA(1, 2.0f);
    tmp->dtype_ = datatypes::f16;
    EXPECT_SC_ERROR(vali(tmp), "Unmatched types of call node and the func_t:");

    vali(AAA(1, 2.0f));
}

TEST(GCCore_CPU_validator_cpp, TestTensor) {
    validator_t vali;
    ir_builder_t builder;
    builder.push_scope();
    _tensor_(tsr, datatypes::f32, {199});
    tsr->dtype_ = datatypes::f16;
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {10}, sc_data_type_t::s32(10))),
            "tensor cannot contain vector types");
    EXPECT_SC_ERROR(vali(tsr), "Tensor should have tensor type, got: f16");
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {10}, datatypes::void_t)),
            "Invalid type: met undef/void");
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {10}, sc_data_type_t::s32(0))),
            "Invalid type: met undef/void");
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {}, datatypes::s32)),
            "Expecting the dimension > 0: ");
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {1.2f}, datatypes::s32)),
            "The 1-th index of the tensor has type f32. Expecting an integer");
    EXPECT_SC_ERROR(vali(make_tensor("aaa", {1, 2UL}, datatypes::s32)),
            "Expecting the all dimemsions within the tensor definition "
            "having the same dtype. Current dimemsion: 2");
}

TEST(GCCore_CPU_validator_cpp, TestVar) {
    validator_t vali;
    ir_builder_t builder;
    builder.push_scope();
    _var_(v, datatypes::void_t);
    EXPECT_SC_ERROR(vali(v), "Invalid type: met undef/void");
}

TEST(GCCore_CPU_validator_cpp, TestVarTensorDef) {
    validator_t vali;
    ir_builder_t builder;
    EXPECT_SC_ERROR(vali(builder::make_var_tensor_def_unattached(expr(1))),
            "Expecting var/tensor");
    EXPECT_SC_ERROR(vali(builder::make_var_tensor_def_unattached(
                            builder::make_var(datatypes::bf16, "a"),
                            linkage::local, expr(1))),
            "The init val has different type from the var definition");
    EXPECT_SC_ERROR(vali(builder::make_var_tensor_def_unattached(
                            builder::make_tensor("a", {1}, datatypes::f32),
                            linkage::local, expr(1))),
            "The init val of tensor should come from dynamic");
}

TEST(GCCore_CPU_validator_cpp, TestAssign) {
    validator_t vali;
    ir_builder_t builder;
    builder.push_scope();
    EXPECT_SC_ERROR(vali(builder.push_assign(expr(1), expr(1))),
            "Assignment only supports tensor or var, got: 1");
    _var_(a, datatypes::bf16);
    EXPECT_SC_ERROR(vali(builder.push_assign(a, expr(1))),
            "Assignment expects the LHS and RHS of the same type, but got bf16 "
            "and s32");
}

TEST(GCCore_CPU_validator_cpp, TestReturnAndFunc) {
    validator_t vali;
    {
        ir_builder_t builder;
        _function_(datatypes::void_t, aaa) { _return_(1); }
        EXPECT_SC_ERROR(vali(aaa),
                "The current function should return void, but got s32");
    }
    {
        ir_builder_t builder;
        _function_(datatypes::f16, aaa) { _return_(1); }
        EXPECT_SC_ERROR(vali(aaa),
                "The current function should return f16, but got s32");
    }
    {
        ir_builder_t builder;
        _function_(datatypes::f16, aaa) { _return_(); }
        EXPECT_SC_ERROR(vali(aaa), "Returning void in a non-void function:");
    }
    {
        ir_builder_t builder;
        _function_(datatypes::s32, aaa) {
            _for_(i, 0, 10) { _return_(1); }
        }
        EXPECT_SC_ERROR(vali(aaa), "Cannot return in a for-loop: ");
    }
    {
        ir_builder_t builder;
        builder.push_scope();
        vali(builder.push_returns());
    }
    {
        ir_builder_t builder;
        _function_(datatypes::void_t, aaa) { _return_(); }
        vali(aaa);
    }
    {
        ir_builder_t builder;
        _function_(datatypes::s32, aaa) { _return_(1); }
        vali(aaa);
    }
}

TEST(GCCore_CPU_validator_cpp, TestIf) {
    ir_builder_t builder;
    validator_t vali;
    builder.push_scope();
    _if_(expr(1)) {}
    stmt s = builder.pop_scope();
    EXPECT_SC_ERROR(vali(s),
            "If-else node expects an boolean expr as the condition, got ");
}

TEST(GCCore_CPU_validator_cpp, TestFor) {
    ir_builder_t builder;
    validator_t vali;
    builder.push_scope();
    for_loop loop;
    _named_for_(loop, i, 0, 10UL, 1UL) {}
    loop->var_ = expr();
    EXPECT_SC_ERROR(vali(loop), "met an invalid for-loop");

    loop->var_ = make_var(datatypes::bf16, "a");
    EXPECT_SC_ERROR(vali(loop),
            "for_loop node expects an index or s32 itervar, got bf16");

    _named_for_(loop, i, 0, 10UL, 1UL) {}
    EXPECT_SC_ERROR(vali(loop),
            "iter_begin of for_loop node expects an index as the itervar, got "
            "s32");

    _named_for_(loop, i, 0UL, 10, 1UL) {}
    EXPECT_SC_ERROR(vali(loop),
            "iter_end of for_loop node expects an index as the itervar, got "
            "s32");

    _named_for_(loop, i, 0UL, 10UL, 1) {}
    EXPECT_SC_ERROR(vali(loop),
            "step of for_loop node expects an index as the itervar, got "
            "s32");
}

TEST(GCCore_CPU_validator_cpp, TestVarDefCheck) {
    ir_builder_t builder;
    {
        validator_t vali;
        expr vaaa = builder::make_var(datatypes::f32, "vaaa");
        _function_(datatypes::void_t, aaa) {
            builder.push_assign(vaaa, 2);
            _return_();
        }
        EXPECT_SC_ERROR(vali(aaa), "Use before define:");
    }

    {
        validator_t vali;
        _function_(datatypes::void_t, aaa) {
            _var_(va, datatypes::f32);
            builder.push_var_tensor_def(va);
            _return_();
        }
        EXPECT_SC_ERROR(vali(aaa), "is already defined");
    }

    {
        validator_t vali;
        _function_(datatypes::void_t, aaa) {
            _var_(va, datatypes::s32);
            builder.push_for_loop(va, 0, 1, 1,
                    builder::make_stmts_unattached({}), true, for_type::NORMAL);
            _return_();
        }
        EXPECT_SC_ERROR(vali(aaa), "is already defined");
    }

    // good case
    {
        validator_t vali;
        _function_(datatypes::void_t, aaa, _arg_("len", datatypes::s32)) {
            _bind_(len);
            _var_(va, datatypes::s32);
            len = 1;
            va = 2;
            _return_();
        }
        vali(aaa);
    }

    {
        validator_t vali;
        ir_module_ptr mod
                = std::make_shared<ir_module_t>(get_default_context());
        _global_tensor_(mod, tsr, datatypes::f32, 100);
        _function_(datatypes::void_t, aaa, _arg_("len", datatypes::s32)) {
            _bind_(len);
            tsr[len] = 1.0f;
            _return_();
        }
        mod->add_func({aaa});
        vali(mod);
    }
}
