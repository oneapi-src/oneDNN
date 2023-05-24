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

#include "exception_util.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/cpu/target_specific_lower.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

void check(const func_t &aaa, uint64_t contents_size, uint64_t seq_size,
        int idx, const char *name) {
    target_specific_lowering_cpu_t pass {get_default_context()};
    auto mod = ir_module_t::from_entry_func(get_default_context(), aaa);
    auto retmod = pass(mod);
    ASSERT_EQ(retmod->get_contents().size(), contents_size);
    auto ret = retmod->get_func("aaa");
    ASSERT_TRUE(ret);
    auto seq = ret->body_.checked_as<stmts>();
    ASSERT_EQ(seq->seq_.size(), seq_size);
    auto &body = seq->seq_;
    auto call_n = body[idx].checked_as<assign>()->value_.as<call>();
    ASSERT_TRUE(call_n.defined());
    auto funct = std::dynamic_pointer_cast<func_base>(call_n->func_);
    ASSERT_TRUE(funct);
    EXPECT_EQ(funct->name_, name);
    EXPECT_EQ(call_n->attr().get_or_else("inline_level", 0), 2);
    EXPECT_EQ(retmod->get_func(name), call_n->func_);
}

TEST(GCCore_CPU_target_specific_lower_cpp, TestLowerIntrinsics) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        A[0] = builder::make_exp(A[span_t({0}, 16)]);
        A[0] = builder::make_exp(A[span_t({0}, 8)]);
        A[0] = builder::make_exp(A[span_t({0}, 4)]);
        A[0] = builder::make_exp(A[span_t({0}, 1)]);
        A[0] = builder::make_exp(A[span_t({0}, 16)]);
    }
    check(aaa, 5, 5, 0, "_should_inline_exp_f32x16");
    check(aaa, 5, 5, 1, "_should_inline_exp_f32x8");
    check(aaa, 5, 5, 2, "_should_inline_exp_f32x4");
    check(aaa, 5, 5, 3, "_should_inline_exp_f32");
    // exp_f32x16 is used twice, check if it is duplicated.
    check(aaa, 5, 5, 4, "_should_inline_exp_f32x16");
}

TEST(GCCore_CPU_target_specific_lower_cpp, TestLowerIntrinsics2) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        _var_(B, sc_data_type_t::boolean(16));
        _var_(C, sc_data_type_t::boolean(8));
        _var_(D, sc_data_type_t::boolean(4));
        _var_(E, sc_data_type_t::boolean(1));
        B[0] = builder::make_isnan(A[span_t({0}, 16)]);
        C[0] = builder::make_isnan(A[span_t({0}, 8)]);
        D[0] = builder::make_isnan(A[span_t({0}, 4)]);
        E[0] = builder::make_isnan(A[span_t({0}, 1)]);
    }
    check(aaa, 5, 8, 4, "_should_inline_isnan_boolx16");
    check(aaa, 5, 8, 5, "_should_inline_isnan_boolx8");
    check(aaa, 5, 8, 6, "_should_inline_isnan_boolx4");
    check(aaa, 5, 8, 7, "_should_inline_isnan_bool");
}

static expr make_const(int64_t v, uint32_t lanes) {
    return make_expr<constant_node>(v, sc_data_type_t::s32(lanes));
}

TEST(GCCore_CPU_target_specific_lower_cpp, TestLowerSaturatedCast) {
    builder::ir_builder_t builder;

    _function_(datatypes::void_t, aaa) {
        _var_(a, datatypes::f32);
        _var_(b, datatypes::s32);
        _var_(c, sc_data_type_t::s32(16));
        _var_(d, sc_data_type_t::f32(16));

        _var_(e, datatypes::s8);
        _var_(f, datatypes::u8);
        _var_(g, sc_data_type_t::s8(16));
        _var_(h, sc_data_type_t::u8(16));

        e = builder::make_saturated_cast(a, datatypes::s8);
        e = builder::make_saturated_cast(b, datatypes::s8);
        f = builder::make_saturated_cast(a, datatypes::u8);
        f = builder::make_saturated_cast(b, datatypes::u8);

        g = builder::make_saturated_cast(c, sc_data_type_t::s8(16));
        g = builder::make_saturated_cast(d, sc_data_type_t::s8(16));
        h = builder::make_saturated_cast(c, sc_data_type_t::u8(16));
        h = builder::make_saturated_cast(d, sc_data_type_t::u8(16));
    }
    target_specific_lowering_cpu_t pass {get_default_context()};
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->machine_.cpu_flags_.fAVX512F = true;
    auto mod = ir_module_t::from_entry_func(ctx, aaa);
    auto retmod = pass(mod);
    auto ret = retmod->get_func("aaa");
    ASSERT_TRUE(ret);
    using namespace builder;
    _function_(datatypes::void_t, expected) {
        _var_(a, datatypes::f32);
        _var_(b, datatypes::s32);
        _var_(c, sc_data_type_t::s32(16));
        _var_(d, sc_data_type_t::f32(16));

        _var_(e, datatypes::s8);
        _var_(f, datatypes::u8);
        _var_(g, sc_data_type_t::s8(16));
        _var_(h, sc_data_type_t::u8(16));

        e = make_cast(datatypes::s8,
                make_max(make_min(make_round_and_cast(a, datatypes::s32),
                                 make_const(127, 1)),
                        make_const(-128, 1)));
        e = make_cast(datatypes::s8,
                make_max(make_min(b, make_const(127, 1)), make_const(-128, 1)));

        f = make_cast(datatypes::u8,
                make_max(make_min(make_round_and_cast(a, datatypes::s32),
                                 make_const(255, 1)),
                        make_const(0, 1)));
        f = make_cast(datatypes::u8,
                make_max(make_min(b, make_const(255, 1)), make_const(0, 1)));

        g = builder::make_saturated_cast(c, sc_data_type_t::s8(16));
        g = builder::make_saturated_cast(
                make_round_and_cast(d, sc_data_type_t::s32(16)),
                sc_data_type_t::s8(16));
        h = builder::make_saturated_cast(
                make_max(c, make_const(0, 16)), sc_data_type_t::u8(16));
        h = builder::make_saturated_cast(
                make_round_and_cast(make_max(d,
                                            make_expr<constant_node>(0.0f,
                                                    sc_data_type_t::f32(16))),
                        sc_data_type_t::s32(16)),
                sc_data_type_t::u8(16));
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(ret, expected, false));
}

TEST(GCCore_CPU_target_specific_lower_cpp, TestLowerGetTidGid) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        A[0] = builder::make_get_group_thread_id(-1);
    }
    _function_(datatypes::void_t, expected,
            _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        A[0] = builtin::get_thread_id_func()();
    }
    target_specific_lowering_cpu_t pass {get_default_context()};
    auto ret = pass(ir_module_t::from_entry_func(get_default_context(), aaa));
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(ret->get_entry_func(), expected, false));

    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        A[0] = builder::make_get_group_thread_id(0);
    }
    auto cccmod = ir_module_t::from_entry_func(get_default_context(), ccc);
    EXPECT_SC_ERROR(pass(cccmod), "get_group_thread_id");

    _function_(datatypes::void_t, ddd, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        A[0] = builder::make_get_group_id(0);
    }
    auto ddddmod = ir_module_t::from_entry_func(get_default_context(), ddd);
    EXPECT_SC_ERROR(pass(ddddmod), "get_group_id");
}
