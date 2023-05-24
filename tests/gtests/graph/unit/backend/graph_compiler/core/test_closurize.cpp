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
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>

#include <iostream>
#include "context.hpp"
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <runtime/config.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_closurize_cpp, TestSingleCore) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, tester) {
        _tensor_(t, datatypes::f32, {100});
        _for_(i, 2, 10, 2, for_type::PARALLEL) { t[i] = 2; }
    }
    _function_(datatypes::void_t, expected) {
        _tensor_(t, datatypes::f32, {100});
        _for_(i, 2, 10, 2) { t[i] = 2; }
    }

    auto mod = closurizer_cpu_t(true)(
            ir_module_t::from_entry_func(get_default_context(), tester));
    auto func = mod->get_func("tester");
    ir_comparer cmper {};

    ASSERT_TRUE(func);
    ASSERT_TRUE(cmper.compare(func, expected));
}

TEST(GCCore_CPU_closurize_cpp, TestClosurizeCPU) {
    builder::ir_builder_t builder;
    auto m = std::make_shared<ir_module_t>(get_default_context());
    _global_var_(m, gv, datatypes::s32, 1);
    _function_(datatypes::void_t, tester) {
        _var_(b, datatypes::s32);
        _tensor_(t, datatypes::f32, {100});
        _for_(i, 2, 10, 2, for_type::PARALLEL) {
            gv = 2;
            t[b + i] = gv;
        }
        _for_(i, 2, 10, 2, for_type::PARALLEL) {
            gv = 2;
            t[b + i] = gv + 1;
        }
    }
    m->add_func({tester});
    bool use_managed = runtime_config_t::get().managed_thread_pool_;
    m->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = use_managed;

    auto testerout = closurizer_cpu_t(false)(m);

    auto outfuncs = testerout->get_contents();
    ASSERT_EQ(outfuncs.size(), 5u);
    // 0 -> tester
    // 1 -> closure0
    // 2 -> closure0_wrapper
    // 3 -> closure1
    // 4 -> closure1_wrapper

    _function_(datatypes::void_t, closure1, _arg_("i", datatypes::index),
            _arg_("t", datatypes::f32, {100}), _arg_("b", datatypes::s32)) {
        _bind_(i, t, b);
        gv = 2;
        t[b + i] = gv;
    }

    _function_(datatypes::void_t, closure2, _arg_("i", datatypes::index),
            _arg_("t", datatypes::f32, {100}), _arg_("b", datatypes::s32)) {
        _bind_(i, t, b);
        gv = 2;
        t[b + i] = gv + 1;
    }
    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(outfuncs[1], closure1));
    EXPECT_TRUE(cmp.compare(outfuncs[3], closure2));

    auto u64_0 = make_expr<constant_node>(UINT64_C(0));
    auto pointer_0 = make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
    auto u8_pointer_0 = make_expr<constant_node>(
            UINT64_C(0), datatypes::s8.get_pointerof());

    _function_(datatypes::void_t, tester2) {
        _var_(b, datatypes::s32);
        _tensor_(t, datatypes::f32, {100});
        builder.push_scope();
        {
            _tensor_(args, datatypes::generic, {2UL});
            args[0UL] = builder::make_cast(datatypes::generic, t);
            args[1UL] = builder::make_cast(datatypes::generic, b);
            expr callnode = builder::make_call(
                    get_parallel_call_with_env_func(use_managed),
                    {builder::make_func_addr(outfuncs[2]), u64_0, pointer_0,
                            u8_pointer_0, 2UL, 10UL, 2UL, args});
            builder.push_evaluate(callnode);
        }
        builder.emit(builder.pop_scope());

        builder.push_scope();
        {
            _tensor_(args, datatypes::generic, {2UL});
            args[0UL] = builder::make_cast(datatypes::generic, t);
            args[1UL] = builder::make_cast(datatypes::generic, b);
            expr callnode = builder::make_call(
                    get_parallel_call_with_env_func(use_managed),
                    {builder::make_func_addr(outfuncs[4]), u64_0, pointer_0,
                            u8_pointer_0, 2UL, 10UL, 2UL, args});
            builder.push_evaluate(callnode);
        }
        builder.emit(builder.pop_scope());
    }
    EXPECT_TRUE(cmp.compare(outfuncs[0], tester2, false));
}

static optional<uint64_t> get_parallel_call_flag(const func_t f, int idx = 0) {
    return f->body_.static_as<stmts>()
            ->seq_.at(idx)
            .cast<stmts>()
            .map([](const stmts &v) { return v->seq_.back().as<evaluate>(); })
            .map([](const evaluate &v) { return v->value_.as<call>(); })
            .map([](const call &v) { return v->args_.at(1).as<constant>(); })
            .map([](const constant &v) { return v->get_index(); });
}

TEST(GCCore_CPU_closurize_cpp, TestClosurizeCPURemoveBarrier) {
    builder::ir_builder_t builder;
    if (!runtime_config_t::get().managed_thread_pool_) { GTEST_SKIP(); }
    _function_(datatypes::boolean, aaa) {
        _for_(i, 0, 10, 2, for_type::PARALLEL) {}
        _return_(true);
    }

    {
        _function_(datatypes::void_t, tester1) {
            _evaluate_call_(aaa);
            _tensor_(b, datatypes::index, 1);
        }
        tester1->attr()[function_attrs::is_main] = true;
        auto m1 = ir_module_t::from_entry_func(get_test_ctx(), tester1);
        m1->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = true;
        auto testerout1 = closurizer_cpu_t(false)(m1);
        auto f = testerout1->get_func("aaa");
        ASSERT_TRUE(f);
        ASSERT_EQ(f->body_.static_as<stmts>()->seq_.size(), 2UL);
        auto flag = get_parallel_call_flag(f);
        ASSERT_TRUE(flag.has_value() && flag.get() == 0);
    }

    {
        _function_(datatypes::void_t, tester1) { _evaluate_call_(aaa); }
        tester1->attr()[function_attrs::is_main] = true;
        auto m1 = ir_module_t::from_entry_func(get_test_ctx(), tester1);
        m1->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = true;
        auto testerout1 = closurizer_cpu_t(false)(m1);
        auto f = testerout1->get_func("aaa");
        ASSERT_TRUE(f);
        ASSERT_EQ(f->body_.static_as<stmts>()->seq_.size(), 2UL);
        auto flag = get_parallel_call_flag(f);
        ASSERT_TRUE(flag.has_value() && flag.get() == 4UL);
    }

    _function_(datatypes::void_t, bbb) {
        _for_(i, 0, 10, 2, for_type::PARALLEL) {}
        _tensor_(b, datatypes::index, 1);
    }
    {
        _function_(datatypes::void_t, tester1) { _evaluate_call_(bbb); }
        tester1->attr()[function_attrs::is_main] = true;
        auto m1 = ir_module_t::from_entry_func(get_test_ctx(), tester1);
        m1->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = true;
        auto testerout1 = closurizer_cpu_t(false)(m1);
        auto f = testerout1->get_func("bbb");
        ASSERT_TRUE(f);
        ASSERT_EQ(f->body_.static_as<stmts>()->seq_.size(), 2UL);
        auto flag = get_parallel_call_flag(f);
        ASSERT_TRUE(flag.has_value() && flag.get() == 0);
    }
}

TEST(GCCore_CPU_closurize_cpp, TestClosurizeCPURemoveBarrierPinMemory) {
    builder::ir_builder_t builder;
    if (!runtime_config_t::get().managed_thread_pool_) { GTEST_SKIP(); }
    {
        expr bbb_A, aaa_A, tester_A, tester_B, tester_T;
        _function_(datatypes::boolean, aaa, _arg_("t", datatypes::f32, {100})) {
            _bind_(t);
            _tensor_(A, datatypes::f32, 100);
            aaa_A = A;
            _for_(i, 0, 10, 2, for_type::PARALLEL) {
                t[0] = 1;
                A[i] = 0.0f;
            }
            _return_(true);
        }
        _function_(datatypes::boolean, bbb, _arg_("t", datatypes::f32, {100}),
                _arg_("t2", datatypes::f32, {100})) {
            _bind_(t, t2);
            _tensor_(A, datatypes::f32, 100);
            bbb_A = A;
            _for_(i, 0, 10, 2, for_type::PARALLEL) {
                t[0] = 1;
                t2[0] = 1;
                A[i] = 0.0f;
            }
            _return_(true);
        }

        _function_(
                datatypes::void_t, tester1, _arg_("t", datatypes::f32, {100})) {
            _bind_(t);
            tester_T = t;
            _tensor_(A, datatypes::f32, 100);
            tester_A = A;
            _tensor_(B, datatypes::f32, 100);
            tester_B = B;
            _tensor_(C, datatypes::f32, 100);
            builder.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(B, {0UL});
            _evaluate_call_(aaa, A);
            _evaluate_call_(bbb, C, t);
        }
        tester1->attr()[function_attrs::is_main] = true;
        auto m1 = ir_module_t::from_entry_func(get_test_ctx(), tester1);
        m1->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = true;
        auto testerout1 = closurizer_cpu_t(false)(m1);
        ASSERT_TRUE(bbb_A->attr().get<bool>(attr_keys::runtime_stack_alloc));
        ASSERT_TRUE(tester_B->attr().get<bool>(attr_keys::runtime_stack_alloc));
        ASSERT_FALSE(tester_A->attr().get_or_else<bool>(
                attr_keys::runtime_stack_alloc, false));
    }

    {
        _function_(datatypes::boolean, aaa,
                _arg_("t", datatypes::pointer, {100})) {
            _bind_(t);
            _tensor_(A, datatypes::f32, 100);
            builder.get_current_scope().body.back().checked_as<define>()->init_
                    = t[0];
            _for_(i, 0, 10, 2, for_type::PARALLEL) { A[i] = 0.0f; }
            _return_(true);
        }

        _function_(
                datatypes::void_t, tester1, _arg_("t", datatypes::f32, {100})) {
            _bind_(t);
            _tensor_(A, datatypes::f32, 100);
            _evaluate_call_(aaa, A);
        }
        tester1->attr()[function_attrs::is_main] = true;
        auto m1 = ir_module_t::from_entry_func(get_test_ctx(), tester1);
        m1->attr_[ir_module_t::attr_key_t::MANAGED_THREAD_POOL] = true;
        auto testerout1 = closurizer_cpu_t(false)(m1);
        auto f = testerout1->get_func("aaa");
        ASSERT_TRUE(f);
        ASSERT_EQ(get_parallel_call_flag(f, 1).get(), 0UL);
    }
}
