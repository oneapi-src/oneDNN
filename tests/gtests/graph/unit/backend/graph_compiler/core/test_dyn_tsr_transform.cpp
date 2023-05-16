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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>

#include <iostream>
#include "context.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

expr dyn_var0 = builder::make_var(datatypes::index, "dyn_var0");
expr dyn_var1 = builder::make_var(datatypes::index, "dyn_var1");
expr dyn_var2 = builder::make_var(datatypes::index, "dyn_var2");

TEST(GCCore_CPU_dyn_tsr_transform_cpp, TestFunctionDefinitionParams) {
    REQUIRE_AVX2();
    builder::ir_builder_t builder;
    _function_(datatypes::index, ccc,
            _arg_("A", datatypes::f32, {dyn_var0, 100}),
            _arg_("B", datatypes::f32, {200, dyn_var0}),
            _arg_("C", datatypes::f32, {dyn_var1, 300}),
            _arg_("D", datatypes::f32, {400, 500})) {
        _bind_(A, B, C, D);
        A->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var0, 100});
        B->attr().set(attr_keys::plain_dims, std::vector<expr> {200, dyn_var0});
        C->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var1, 300});

        D[{0, 0}] = A[{0, 0}] + B[{0, 1}] + C[{0, 2}];
        _return_(dyn_var0 + dyn_var1);
    }
    expr sz = sizeof(runtime::dynamic_tensor_t);
    _function_(datatypes::index, expected, _arg_("dyn_A", datatypes::u8, {sz}),
            _arg_("dyn_B", datatypes::u8, {sz}),
            _arg_("dyn_C", datatypes::u8, {sz}),
            _arg_("D", datatypes::f32, {400, 500})) {
        _bind_(dyn_A, dyn_B, dyn_C, D);
        expr A = builder::make_tensor("A", {dyn_var0, 100}, datatypes::f32);
        expr A_shape = builder::make_tensor(
                "dyn_shape_A", {UINT64_C(2)}, datatypes::index);
        expr B = builder::make_tensor("B", {200, dyn_var0}, datatypes::f32);
        expr C = builder::make_tensor("C", {dyn_var1, 300}, datatypes::f32);
        expr C_shape = builder::make_tensor(
                "dyn_shape_C", {UINT64_C(2)}, datatypes::index);
        builder.push_var_tensor_def(A_shape, linkage::local,
                builder::make_read_struct(dyn_A, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder.push_var_tensor_def(dyn_var0, linkage::local,
                builder::make_indexing(A_shape, {UINT64_C(0)}));
        builder.push_var_tensor_def(A, linkage::local,
                builder::make_read_struct(dyn_A, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder.push_var_tensor_def(B, linkage::local,
                builder::make_read_struct(dyn_B, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder.push_var_tensor_def(C_shape, linkage::local,
                builder::make_read_struct(dyn_C, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder.push_var_tensor_def(dyn_var1, linkage::local,
                builder::make_indexing(C_shape, {UINT64_C(0)}));
        builder.push_var_tensor_def(C, linkage::local,
                builder::make_read_struct(dyn_C, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        D[{0, 0}] = builder::make_indexing(A, {0, 0})
                + builder::make_indexing(B, {0, 1})
                + builder::make_indexing(C, {0, 2});
        _return_(dyn_var0 + dyn_var1);
    }

    dyn_tensor_transformer_t transformer;
    func_c newf = transformer(ccc);
    ir_comparer cmper;
    ASSERT_TRUE(cmper.compare(newf, expected));
    std::vector<float> data(10000);

    runtime::dynamic_tensor_t A, B, C;

    auto jitf = jit_engine_t::make(get_test_ctx())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                get_test_ctx(), ccc));
    std::vector<float> in(10000), out(10000);
    in[0] = 3, in[1] = 4, in[2] = 6;
    int64_t A_shape[2] = {256, 100};
    int64_t C_shape[2] = {512, 300};
    A.data_ = in.data();
    B.data_ = in.data();
    C.data_ = in.data();
    A.dims_ = A_shape;
    C.dims_ = C_shape;
    uint64_t ret = jitf->call<uint64_t>(&A, &B, &C, out.data());
    ASSERT_EQ(out[0], 13);
    ASSERT_EQ(ret, 768UL);
}

TEST(GCCore_CPU_dyn_tsr_transform_cpp, TestFunctionCaller) {
    REQUIRE_AVX2();
    builder::ir_builder_t builder;
    _function_(datatypes::index, ccc,
            _arg_("A", datatypes::f32, {dyn_var0, 100}),
            _arg_("B", datatypes::f32, {200, dyn_var0}),
            _arg_("C", datatypes::f32, {dyn_var1, 300}),
            _arg_("D", datatypes::f32, {dyn_var2, dyn_var0})) {
        _bind_(A, B, C, D);
        A->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var0, 100});
        B->attr().set(attr_keys::plain_dims, std::vector<expr> {200, dyn_var0});
        C->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var1, 300});
        D->attr().set(
                attr_keys::plain_dims, std::vector<expr> {dyn_var2, dyn_var0});
        D[{0, 0}] = A[{0, 0}] + B[{0, 1}] + C[{0, 2}];
        _return_(dyn_var0 + dyn_var1 + dyn_var2);
    }

    _function_(datatypes::index, the_main,
            _arg_("A", datatypes::f32, {dyn_var0, 100}),
            _arg_("B", datatypes::f32, {200, dyn_var0}),
            _arg_("C", datatypes::f32, {400, dyn_var1})) {
        _bind_(A, B, C);
        A->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var0, 100});
        B->attr().set(attr_keys::plain_dims, std::vector<expr> {200, dyn_var0});
        C->attr().set(attr_keys::plain_dims, std::vector<expr> {400, dyn_var1});
        _var_init_(dyn_var2, datatypes::index, UINT64_C(123));
        _tensor_(D, datatypes::f32, {dyn_var1, 300});
        D->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var1, 300});
        _tensor_(E, datatypes::f32, {dyn_var2, dyn_var0});
        E->attr().set(
                attr_keys::plain_dims, std::vector<expr> {dyn_var2, dyn_var0});
        _var_init_(ret, datatypes::index,
                builder::make_call(ccc->decl_, {A, B, D, E}));
        _return_(ret);
    }

    dyn_tensor_transformer_t transformer;
    auto mod = ir_module_t::from_entry_func(get_test_ctx(), the_main);
    mod->add_func({ccc});

    auto new_mod = transformer(mod);
    auto new_main = new_mod->get_entry_func();
    auto new_ccc = new_mod->get_func("ccc");

    expr sz = sizeof(runtime::dynamic_tensor_t);
    _function_(datatypes::index, expected_main,
            _arg_("dyn_A", datatypes::u8, {sz}),
            _arg_("dyn_B", datatypes::u8, {sz}),
            _arg_("dyn_C", datatypes::u8, {sz})) {
        _bind_(dyn_A, dyn_B, dyn_C);
        expr A = builder::make_tensor("A", {dyn_var0, 100}, datatypes::f32);
        expr A_shape = builder::make_tensor(
                "dyn_shape_A", {UINT64_C(2)}, datatypes::index);
        expr B = builder::make_tensor("B", {200, dyn_var0}, datatypes::f32);
        expr C = builder::make_tensor("C", {400, dyn_var1}, datatypes::f32);
        expr C_shape = builder::make_tensor(
                "dyn_shape_C", {UINT64_C(2)}, datatypes::index);
        builder.push_var_tensor_def(A_shape, linkage::local,
                builder::make_read_struct(dyn_A, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder.push_var_tensor_def(dyn_var0, linkage::local,
                builder::make_indexing(A_shape, {UINT64_C(0)}));
        builder.push_var_tensor_def(A, linkage::local,
                builder::make_read_struct(dyn_A, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder.push_var_tensor_def(B, linkage::local,
                builder::make_read_struct(dyn_B, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder.push_var_tensor_def(C_shape, linkage::local,
                builder::make_read_struct(dyn_C, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder.push_var_tensor_def(dyn_var1, linkage::local,
                builder::make_indexing(C_shape, {UINT64_C(1)}));
        builder.push_var_tensor_def(C, linkage::local,
                builder::make_read_struct(dyn_C, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        _var_init_(dyn_var2, datatypes::index, UINT64_C(123));
        _tensor_(D, datatypes::f32, {dyn_var1, 300});
        D->attr().set(attr_keys::plain_dims, std::vector<expr> {dyn_var1, 300});
        _tensor_(E, datatypes::f32, {dyn_var2, dyn_var0});
        E->attr().set(
                attr_keys::plain_dims, std::vector<expr> {dyn_var2, dyn_var0});
        _tensor_(dyn_D, datatypes::u8, {sz});
        _tensor_(dyn_shape_D, datatypes::index, {UINT64_C(2)});
        dyn_shape_D[UINT64_C(0)] = dyn_var1;
        dyn_shape_D[UINT64_C(1)] = 300;
        _var_init_(dyn_mask_D, datatypes::u8,
                builder::make_constant({UINT64_C(1)}, datatypes::u8));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_D, D, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_D, dyn_shape_D,
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_D, 2, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::ndims));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_D,
                        builder::make_constant({UINT64_C(4)}, datatypes::u32),
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dtype));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_D, dyn_mask_D,
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dyn_mask));
        _tensor_(dyn_E, datatypes::u8, {sz});
        _tensor_(dyn_shape_E, datatypes::index, {UINT64_C(2)});
        dyn_shape_E[UINT64_C(0)] = dyn_var2;
        dyn_shape_E[UINT64_C(1)] = dyn_var0;
        _var_init_(dyn_mask_E, datatypes::u8,
                builder::make_constant({UINT64_C(3)}, datatypes::u8));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_E, E, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::data_ptr));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_E, dyn_shape_E,
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dim_ptr));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_E, 2, dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::ndims));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_E,
                        builder::make_constant({UINT64_C(4)}, datatypes::u32),
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dtype));
        builder::get_current_builder()->push_evaluate(
                builder::make_write_struct(dyn_E, dyn_mask_E,
                        dyn_tsr_struct_t::name,
                        dyn_tsr_struct_t::fields::dyn_mask));
        _var_init_(ret, datatypes::index,
                builder::make_call(
                        new_ccc->decl_, {dyn_A, dyn_B, dyn_D, dyn_E}));
        _return_(ret);
    }

    ir_comparer cmper;
    ASSERT_TRUE(cmper.compare(new_main, expected_main));

    runtime::dynamic_tensor_t A, B, C;
    auto jitf = jit_engine_t::make(get_test_ctx())->get_entry_func(mod);
    std::vector<float> in(1000);
    int64_t A_shape[2] = {256, 100};
    int64_t C_shape[2] = {300, 512};
    A.data_ = in.data();
    B.data_ = in.data();
    C.data_ = in.data();
    A.dims_ = A_shape;
    C.dims_ = C_shape;
    uint64_t ret = jitf->call<uint64_t>(&A, &B, &C);
    ASSERT_EQ(ret, 768UL + 123);
}
