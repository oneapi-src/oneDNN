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

#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/jit/jit.hpp>
#include <test_utils.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_builtin_cpp, TestBrgemmOnednn) {
    REQUIRE_AVX512();
    builder::ir_builder_t builder;
    const int M = 32;
    const int N = 64;
    const int K = 16;
    const int blocks = 10;
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);
    _function_(datatypes::boolean, brgemm_test,
            _arg_("A", datatypes::f32, {blocks, M, K}),
            _arg_("B", datatypes::f32, {blocks, N, K}),
            _arg_("C", datatypes::f32, {M, N})) {
        _bind_(A, B, C);
        _evaluate_call_(
                builtin::get_brgemm_update_funcs(
                        builtin::brgemm_mode::stride, scflags_t::brgemm_t::dnnl)
                        .second,
                builder::tensor_ptr(A, {0, 0, 0}),
                builder::tensor_ptr(B, {0, 0, 0}),
                builder::tensor_ptr(C, {0, 0}), blocks, M, N, K, K, N, N, M * K,
                K * N, datatypes::f32.as_etype_int(),
                datatypes::f32.as_etype_int(), ir_nullptr, ir_nullptr,
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr);
        _return_(true);
    }
    // auto c = create_c_generator(std::cout);
    // c(brgemm_test);
    auto fptr = jit_engine_t::make(get_default_context())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                get_default_context(), brgemm_test));
    std::vector<float> A(blocks * M * K, 1.f);
    std::vector<float> B(blocks * N * K, 1.f);
    std::vector<float> C(M * N);
    fptr->call_default(A.data(), B.data(), C.data());
    for (auto i : C) {
        EXPECT_EQ(i, 160.f);
    }
}
