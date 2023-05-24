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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <compiler/ir/transform/value_numbering.hpp>
#include <compiler/jit/xbyak/ir/transform/intrinsics_combine.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

constexpr auto s32 = datatypes::s32;
constexpr auto f32 = datatypes::f32;
constexpr auto f32x16 = sc_data_type_t::f32(16);

TEST(GCCore_CPU_test_intrinsics_combine, TestFmaddCombine) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);

        _var_(v1, f32);
        v1 = A[0] * B[0];
        _var_(v2, f32);
        v2 = v1 + C[0]; // can combine
        D[0] = v2;

        _var_(v3, f32);
        v3 = A[1] * B[1];
        _var_(v4, f32);
        v4 = C[1] + v3; // can combine
        D[1] = v4;

        _var_(v5, f32);
        v5 = A[2] * B[2];
        _var_(v6, f32);
        v6 = v5 + C[2]; // cannot combine
        D[2] = v6;
        D[3] = v5;
    }
    ssa_transform_t s;
    auto out = s(ccc);
    xbyak::intrinsics_combine_t ic;
    out = ic(out);

    _function_(datatypes::void_t, expected, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);

        _var_init_(i0, s32, 0);
        _var_init_(t0, f32, A[i0]);
        _var_init_(i1, s32, 0);
        _var_init_(t1, f32, B[i1]);
        _var_init_(v0, f32, t0 * t1);
        _var_init_(i2, s32, 0);
        _var_init_(t2, f32, C[i2]);
        _var_init_(v1, f32, builder::make_fmadd(t0, t1, t2));
        _var_init_(i3, s32, 0);
        D[i3] = v1;

        _var_init_(i4, s32, 1);
        _var_init_(t3, f32, A[i4]);
        _var_init_(i5, s32, 1);
        _var_init_(t4, f32, B[i5]);
        _var_init_(v2, f32, t3 * t4);
        _var_init_(i6, s32, 1);
        _var_init_(t5, f32, C[i6]);
        _var_init_(v3, f32, builder::make_fmadd(t3, t4, t5));
        _var_init_(i7, s32, 1);
        D[i7] = v3;

        _var_init_(i8, s32, 2);
        _var_init_(t6, f32, A[i8]);
        _var_init_(i9, s32, 2);
        _var_init_(t7, f32, B[i9]);
        _var_init_(v4, f32, t6 * t7);
        _var_init_(i10, s32, 2);
        _var_init_(t8, f32, C[i10]);
        _var_init_(v5, f32, v4 + t8);
        _var_init_(i11, s32, 2);
        D[i11] = v5;
        _var_init_(i12, s32, 3);
        D[i12] = v4;
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_test_intrinsics_combine, TestFmaddCombineLoop) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);
        _var_(v7, f32);
        v7 = A[2] * B[2];
        _for_(i, 0, 10, 1) {
            _var_(v8, f32);
            v8 = C[i] + v7; // cannot combine
            D[i] = v8;
        }
    }
    ssa_transform_t s;
    auto out = s(ccc);
    value_numbering_t vn;
    out = vn(out);
    xbyak::intrinsics_combine_t ic;
    out = ic(out);

    _function_(datatypes::void_t, expected, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);

        _var_init_(t21, f32, A[2]);
        _var_init_(t23, f32, B[2]);
        _var_init_(t24, f32, (t21 * t23));
        _for_(i, 0, 10, 1) {
            _var_init_(t28, f32, C[i]);
            _var_init_(v210, f32, (t28 + t24));
            D[i] = v210;
        }
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_test_intrinsics_combine, TestBroadcastCombine) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);

        _var_(v1, f32);
        v1 = A[0]; // can combine
        B[span_t({0}, 16)] = builder::make_broadcast(v1, 16);

        _var_(v2, f32);
        v2 = A[1]; // cannot combine
        C[span_t({0}, 16)] = builder::make_broadcast(v2, 16);
        D[0] = v2;
    }
    ssa_transform_t s;
    auto out = s(ccc);
    xbyak::intrinsics_combine_t ic;
    out = ic(out);

    _function_(datatypes::void_t, expected, _arg_("A", f32, {10000}),
            _arg_("B", f32, {10000}), _arg_("C", f32, {10000}),
            _arg_("D", f32, {10000})) {
        _bind_(A, B, C, D);

        _var_init_(i0, s32, 0);
        _var_init_(t0, f32, A[i0]);
        _var_init_(i1, s32, 0);
        auto bcast_idx
                = builder::make_x86_intrin(x86_intrin_type::avx_broadcast_idx,
                        {A, i0, 1}, {{"lanes", 16}});
        _var_init_(v0, f32x16, bcast_idx);
        B[span_t({i1}, 16)] = v0;

        _var_init_(i2, s32, 1);
        _var_init_(t1, f32, A[i2]);
        _var_init_(i3, s32, 0);
        _var_init_(v1, f32x16, builder::make_broadcast(t1, 16));
        C[span_t({i3}, 16)] = v1;
        _var_init_(i4, s32, 0);
        D[i4] = t1;
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
