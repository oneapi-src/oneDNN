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

#include <functional>
#include <iostream>
#include <vector>
#include "context.hpp"
#include "test_utils.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/bf16_legalize.hpp>
#include <compiler/ir/transform/cpu/target_specific_lower.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/jit/jit.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::utils;
using namespace dnnl::impl::graph::gc::test_utils;

#define DBF16 sc_data_type_t::bf16(8)
#define DF32 sc_data_type_t::f32(8)
#define BF16(x) builder::make_cast(DBF16, x)
#define F32(x) builder::make_cast(DF32, x)
void dotest_bf16_promote_binary(expr (*OP)(const expr_c &, const expr_c &)) {
    builder::ir_builder_t bld;
    stmt aaa, bbb;
    ir_comparer cmper(true);
    bld.push_scope();
    builder::make_var(DBF16, "c")
            = OP(builder::make_var(DBF16, "a"), builder::make_var(DBF16, "b"));
    aaa = bld.pop_scope();

    bld.push_scope();
    builder::make_var(DBF16, "c") = BF16(OP(F32(builder::make_var(DBF16, "a")),
            F32(builder::make_var(DBF16, "b"))));
    bbb = bld.pop_scope();

    bf16_promote_impl_t pass;
    auto ccc = pass.dispatch(aaa);
    EXPECT_TRUE(cmper.compare(ccc, bbb, false));
}

void dotest_bf16_promote_cmp(expr (*OP)(const expr_c &, const expr_c &)) {
    builder::ir_builder_t bld;
    stmt aaa, bbb;
    ir_comparer cmper(true);
    bld.push_scope();
    builder::make_var(DBF16, "c")
            = OP(builder::make_var(DBF16, "a"), builder::make_var(DBF16, "b"));
    aaa = bld.pop_scope();
    bld.push_scope();
    builder::make_var(DBF16, "c") = OP(F32(builder::make_var(DBF16, "a")),
            F32(builder::make_var(DBF16, "b")));
    bbb = bld.pop_scope();

    bf16_promote_impl_t pass;
    auto ccc = pass.dispatch(aaa);
    EXPECT_TRUE(cmper.compare(ccc, bbb, false));
}

void dotest_bf16_promote_single(expr (*OP)(const expr_c &)) {
    builder::ir_builder_t bld;
    stmt aaa, bbb;
    ir_comparer cmper(true);
    bld.push_scope();
    builder::make_var(DBF16, "c") = OP(builder::make_var(DBF16, "a"));
    aaa = bld.pop_scope();
    bld.push_scope();
    builder::make_var(DBF16, "c")
            = BF16(OP(F32(builder::make_var(DBF16, "a"))));
    bbb = bld.pop_scope();

    bf16_promote_impl_t pass;
    auto ccc = pass.dispatch(aaa);
    EXPECT_TRUE(cmper.compare(ccc, bbb, false));
}

void dotest_bf16_promote_assign() {
    builder::ir_builder_t bld;
    stmt aaa, bbb;
    ir_comparer cmper(true);
    bld.push_scope();
    builder::make_var(DBF16, "c") = builder::make_var(DF32, "a");
    builder::make_var(DF32, "b") = builder::make_var(DBF16, "c");
    aaa = bld.pop_scope();
    bld.push_scope();
    builder::make_var(DBF16, "c") = BF16(builder::make_var(DF32, "a"));
    builder::make_var(DF32, "b") = F32(builder::make_var(DBF16, "c"));

    bbb = bld.pop_scope();

    bf16_promote_impl_t pass;
    auto ccc = pass.dispatch(aaa);
    EXPECT_TRUE(cmper.compare(ccc, bbb, false));
}

TEST(GCCore_bf16legalize_cpp, TestBF16Promote) {
    dotest_bf16_promote_binary(builder::make_add);
    dotest_bf16_promote_binary(builder::make_sub);
    dotest_bf16_promote_binary(builder::make_mul);
    dotest_bf16_promote_binary(builder::make_div);
    dotest_bf16_promote_cmp(builder::make_cmp_eq);
    dotest_bf16_promote_cmp(builder::make_cmp_ne);
    dotest_bf16_promote_cmp(builder::make_cmp_ge);
    dotest_bf16_promote_cmp(builder::make_cmp_gt);
    dotest_bf16_promote_cmp(builder::make_cmp_le);
    dotest_bf16_promote_cmp(builder::make_cmp_lt);
    dotest_bf16_promote_binary(builder::make_max);
    dotest_bf16_promote_binary(builder::make_min);
    dotest_bf16_promote_single(builder::make_abs);
    dotest_bf16_promote_assign();
}

TEST(GCCore_bf16legalize_cpp, TestBF16CastElimination) {
    builder::ir_builder_t builder;
    ir_comparer cmper(true);
    _function_(datatypes::s32, aaa, _arg_("a", DBF16), _arg_("b", DBF16),
            _arg_("c", DBF16)) {
        _bind_(a, b, c);
        _var_(e, DBF16);
        e = builder::make_max(a + b, a - b);
        _var_(d, DBF16);
        d = builder::make_max(a, e);
        c = d;
        _return_(1);
    }
    _function_(datatypes::s32, bbb, _arg_("a", DBF16), _arg_("b", DBF16),
            _arg_("c", DBF16)) {
        _bind_(a, b, c);
        _var_(e, DBF16);
        e = BF16(builder::make_max(
                F32(BF16(F32(a) + F32(b))), F32(BF16(F32(a) - F32(b)))));
        _var_(d, DBF16);
        d = BF16(builder::make_max(F32(a), F32(e)));
        c = d;
        _return_(1);
    }
    _function_(datatypes::s32, ccc, _arg_("a", DBF16), _arg_("b", DBF16),
            _arg_("c", DBF16)) {
        _bind_(a, b, c);
        _var_(e, DF32);
        e = builder::make_max(F32(a) + F32(b), F32(a) - F32(b));
        _var_(d, DBF16);
        d = BF16(builder::make_max(F32(a), e));
        c = d;
        _return_(1);
    }
    bf16_promote_impl_t promote_pass;
    bf16_elimination_analyzer_t analysis_pass(get_test_ctx());
    bf16_cast_elimination_impl_t elimination_pass(
            get_test_ctx(), analysis_pass.var_use_cnt_);
    auto ddd = promote_pass.dispatch(aaa);
    EXPECT_TRUE(cmper.compare(ddd, bbb, false));
    cmper.reset();
    auto eee = analysis_pass.dispatch(ddd);
    auto fff = elimination_pass.dispatch(eee);
    EXPECT_TRUE(cmper.compare(fff, ccc, false));
}

float get_rand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) // NOLINT
            * 2 // NOLINT
            - 1.0; // NOLINT
}

TEST(GCCore_bf16legalize_cpp, TestBF16Lower) {
    builder::ir_builder_t builder;
    _function_(datatypes::bf16, aaa, _arg_("a", datatypes::bf16),
            _arg_("b", datatypes::bf16)) {
        _bind_(a, b);
        _return_((a + b) * (a - b)
                + builder::make_cast(datatypes::bf16, 3.14159F));
    }
    float fa = get_rand(), fb = get_rand(), fc, fc_ref;
    bf16_t a = fa;
    bf16_t b = fb;
    bf16_t c;
    fc_ref = float(
            bf16_t((float(a) + float(b)) * (float(a) - float(b)) + 3.14159F));
    auto fptr = jit_engine_t::make(get_default_context())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                                 get_default_context(), aaa),
                                false);
    // due to MSVC ABI, we cannot return bf16_t, because it is a struct and
    // MSVC will treat bf16_t differently with uint16_t when returning it
    c = bf16_t::from_storage(fptr->call<uint16_t>(a, b));
    fc = float(c);
    EXPECT_TRUE(std::abs(fc - fc_ref) < 1e-5);
    // vector type
    auto ctx = get_default_context();
    int lanes = 8;
    if (ctx->machine_.device_type_ == runtime::target_machine_t::type::cpu
            && ctx->machine_.cpu_flags_.fAVX512F) {
        _function_(datatypes::s32, bbb, _arg_("a", datatypes::bf16, {lanes}),
                _arg_("b", datatypes::bf16, {lanes}),
                _arg_("c", datatypes::bf16, {lanes})) {
            _bind_(a, b, c);
            c[span_t({0}, lanes)]
                    = (a[span_t({0}, lanes)] + b[span_t({0}, lanes)])
                            * (a[span_t({0}, lanes)] - b[span_t({0}, lanes)])
                    + builder::make_constant(
                            std::vector<union_val>(lanes, 3.14159F), DBF16);
            _return_(1);
        }

        std::vector<float> vfa(lanes), vfb(lanes), vfc(lanes), vfc_ref(lanes);
        std::vector<bf16_t> va(lanes), vb(lanes), vc(lanes);
        for (auto i = 0; i < lanes; i++) {
            vfa[i] = get_rand();
            vfb[i] = get_rand();
            va[i] = bf16_t(vfa[i]);
            vb[i] = bf16_t(vfb[i]);
            vfc_ref[i] = float(bf16_t((float(va[i]) + float(vb[i]))
                            * (float(va[i]) - float(vb[i]))
                    + 3.14159));
        }

        fptr = jit_engine_t::make(get_default_context())
                       ->get_entry_func(ir_module_t::from_entry_func(
                                                get_default_context(), bbb),
                               false);
        fptr->call<int>(&va[0], &vb[0], &vc[0]);

        vfc = std::vector<float>(vc.begin(), vc.end());
        test_utils::compare_data(vfc, vfc_ref);
    }
}
