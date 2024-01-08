/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#if SC_BUILTIN_JIT_ENABLED
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <util/any_map.hpp>

#include <cstdint>
#include "compiler/ir/builder.hpp"
#include "compiler/ir/transform/simplify.hpp"
#include "compiler/jit/xbyak/ir/transform/fp16_legalizer.hpp"
#include "context.hpp"
#include "util/fp16.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

TEST(GCCore_CPU_fp16_legalizer, TestFp16Transform) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f16, {123, 321}),
            _arg_("B", datatypes::f16, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B);
        B[0] = A[0];
        B[1] = make_expr<constant_node>(32.f, datatypes::f16);
        _var_(c, datatypes::f16);
        c = make_expr<constant_node>(32.f, datatypes::f16);
        B[2] = builder::make_max(
                A[2], make_expr<constant_node>(0.f, datatypes::f16));
        B[3] = c;
    }

    _function_(datatypes::void_t, expected,
            _arg_("A", datatypes::f16, {123, 321}),
            _arg_("B", datatypes::f16, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        builder::make_reinterpret(B, datatypes::u16.get_pointerof())[0]
                = builder::make_reinterpret(
                        builder::make_reinterpret(
                                builder::make_reinterpret(
                                        A, datatypes::u16.get_pointerof())[0],
                                datatypes::f16),
                        datatypes::u16);
        builder::make_reinterpret(B, datatypes::u16.get_pointerof())[1]
                = builder::make_reinterpret(
                        make_expr<constant_node>(32.f, datatypes::f16),
                        datatypes::u16);

        _var_(c, datatypes::f16);
        c = make_expr<constant_node>(32.f, datatypes::f16);
        builder::make_reinterpret(B, datatypes::u16.get_pointerof())[2]
                = builder::make_reinterpret(
                        builder::make_max(
                                builder::make_reinterpret(
                                        builder::make_reinterpret(A,
                                                datatypes::u16
                                                        .get_pointerof())[2],
                                        datatypes::f16),
                                make_expr<constant_node>(0.f, datatypes::f16)),
                        datatypes::u16);
        builder::make_reinterpret(B, datatypes::u16.get_pointerof())[3]
                = builder::make_reinterpret(c, datatypes::u16);
    }
    auto ctx = get_test_ctx();
    auto mh = ctx->machine_;
    mh.cpu_flags_.fAVX512AMXFP16 = false;
    mh.cpu_flags_.fAVX512FP16 = false;
    fp16_legalizer_t f(mh);
    auto out = f(aaa);
    ir_simplifier_t f_sim(false);
    out = f_sim(out);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
#endif
