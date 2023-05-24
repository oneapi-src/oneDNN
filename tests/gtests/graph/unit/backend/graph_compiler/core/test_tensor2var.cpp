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
#include <compiler/ir/transform/tensor2var.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
TEST(GCCore_CPU_tensor2var_cpp, TestTensor2Var) {
    builder::ir_builder_t builder;
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123}),
            _arg_("B", datatypes::f32, {122}), _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(tmp, datatypes::f32, 32);
        // arg tensor, don't change
        A[0] = 1.0f;
        // local tensor with different SIMD len
        tmp[0] = 1.0f;
        tmp[span_t({0}, 8)] = tmp[span_t({0}, 8)];
        _tensor_(tmpb, datatypes::f32, 32);
        _var_(t, datatypes::pointer);
        // directly use tensor node
        t = tmpb;
        tmpb[span_t({0}, 8)] = tmpb[span_t({8}, 8)];

        // tensorptr
        _tensor_(tmpc, datatypes::f32, 32);
        t = builder::tensor_ptr(tmpc, {0});
        tmpc[span_t({0}, 8)] = tmpc[span_t({8}, 8)];

        // tensor too large
        _tensor_(tmpd, datatypes::f32, 4096);
        tmpd[span_t({0}, 8)] = tmpd[span_t({8}, 8)];

        // bad index
        _tensor_(tmpe, datatypes::f32, 32);
        tmpe[span_t({0}, 8)] = tmpe[span_t({11}, 8)];

        // good
        _tensor_(tmpf, datatypes::f32, 32);
        tmpf[span_t({0}, 8)] = tmpf[span_t({16}, 8)];

        // good
        _tensor_(tmpg, datatypes::f32, 2);
        tmpg[0] = 3.0f;
        tmpg[1] = tmpg[0] + 3.0f;

        _tensor_(tmph, datatypes::f32, 32);

        _for_(i, 0, len, 1, for_type::PARALLEL) {
            // in parallel
            tmph[0] = 1.0f;
        }

        _return_(12);
    }

    tensor2var_t f;
    auto outf = f(aaa);

    _function_(datatypes::s32, expected, _arg_("A", datatypes::f32, {123}),
            _arg_("B", datatypes::f32, {122}), _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(tmp, datatypes::f32, 32);
        // arg tensor, don't change
        A[0] = 1.0f;
        // local tensor with different SIMD len
        tmp[0] = 1.0f;
        tmp[span_t({0}, 8)] = tmp[span_t({0}, 8)];
        _tensor_(tmpb, datatypes::f32, 32);
        _var_(t, datatypes::pointer);
        // directly use tensor node
        t = tmpb;
        tmpb[span_t({0}, 8)] = tmpb[span_t({8}, 8)];

        // tensorptr
        _tensor_(tmpc, datatypes::f32, 32);
        t = builder::tensor_ptr(tmpc, {0});
        tmpc[span_t({0}, 8)] = tmpc[span_t({8}, 8)];

        // tensor too large
        _tensor_(tmpd, datatypes::f32, 4096);
        tmpd[span_t({0}, 8)] = tmpd[span_t({8}, 8)];

        // bad index
        _tensor_(tmpe, datatypes::f32, 32);
        tmpe[span_t({0}, 8)] = tmpe[span_t({11}, 8)];

        // good
        _var_(tmpf0, sc_data_type_t(sc_data_etype::F32, 8));
        _var_(tmpf2, sc_data_type_t(sc_data_etype::F32, 8));
        tmpf0 = tmpf2;

        // good
        _var_(tmpg0, datatypes::f32);
        _var_(tmpg1, datatypes::f32);
        tmpg0 = 3.0f;
        tmpg1 = tmpg0 + 3.0f;

        _tensor_(tmph, datatypes::f32, 32);

        _for_(i, 0, len, 1, for_type::PARALLEL) {
            // in parallel
            tmph[0] = 1.0f;
        }

        _return_(12);
    }
    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(outf, expected, false));
}
