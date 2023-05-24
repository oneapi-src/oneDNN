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

#include "context.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/transform/tensor_inplace_info.hpp>
#include <util/any_map.hpp>

#include <unordered_set>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static context_ptr make_ctx() {
    auto ret = std::make_shared<context_t>(*get_test_ctx());
    ret->flags_.buffer_schedule_ = 1;
    return ret;
}
static context_ptr cur_ctx = make_ctx();

TEST(GCCore_CPU_buffer_schedule_cpp, TestSimpleSchedule) {
    ir_builder_t bld;
    bld.push_scope();
    _tensor_(external, datatypes::f32, {100});
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {50});
        // make a staic local tensor here. This tensor should not be touched
        _tensor_(static_v, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->linkage_
                = linkage::static_local;

        _var_(scalar, datatypes::f32);
        _var_(ptr, datatypes::pointer);
        ptr = a;
        a[0] = external[0];
        b[0] = a[0];

        _tensor_(c, datatypes::f32, {100});
        c[0] = b[0];

        _tensor_(d, datatypes::f32, {100});
        d[0] = c[0];

        _tensor_(e, datatypes::f32, {100});
        // overwriting b in the below line prevents reusing b for d
        b[10] = 2;
        e[0] = d[0];

        _tensor_(f, datatypes::f32, {100});
        scalar = e[0];

        _tensor_(int8, datatypes::s8, {100});
        int8[0] = external[0];
        external[0] = int8[0];
    }
    auto body = bld.pop_scope();
    body->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_WHOLE;

    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        // check dimension extension
        _tensor_(b, datatypes::f32, {100UL});
        // make a staic local tensor here. This tensor should not be touched
        _tensor_(static_v, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->linkage_
                = linkage::static_local;

        _var_(scalar, datatypes::f32);
        _var_(ptr, datatypes::pointer);
        ptr = a;
        a[0] = external[0];
        b[0] = a[0];

        // tensor c
        bld.push_scope();
        bld.emit(bld.pop_scope());
        b[0] = b[0];

        _tensor_(d, datatypes::f32, {100});
        d[0] = b[0];

        // tensor e
        bld.push_scope();
        bld.emit(bld.pop_scope());
        b[10] = 2;
        d[0] = d[0];

        bld.push_scope();
        bld.emit(bld.pop_scope());
        scalar = d[0];

        _tensor_(int8, datatypes::s8, {100});
        int8[0] = external[0];
        external[0] = int8[0];
    }
    auto body2 = bld.pop_scope();
    buffer_scheduler_t sch(cur_ctx, false);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(body), body2));

    ////Test for aggressive mode
    bld.push_scope();
    {
        _tensor_(rescheduled, datatypes::s8, {704UL});
        _tensor_(a, datatypes::f32, {100});
        // tensor b
        _tensor_(b, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {0UL});
        // make a staic local tensor here. This tensor should not be touched
        _tensor_(static_v, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->linkage_
                = linkage::static_local;

        _var_(scalar, datatypes::f32);
        _var_(ptr, datatypes::pointer);
        ptr = a;
        a[0] = external[0];
        b[0] = a[0];

        // tensor c
        _tensor_(c, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {256UL});
        c[0] = b[0];

        // tensor d
        _tensor_(d, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {256UL});
        d[0] = c[0];

        // tensor e
        _tensor_(e, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {0UL});
        b[10] = 2;
        e[0] = d[0];

        bld.push_scope();
        bld.emit(bld.pop_scope());
        scalar = e[0];

        _tensor_(int8, datatypes::s8, {100});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {0UL});
        int8[0] = external[0];
        external[0] = int8[0];
    }
    auto body_aggresive = bld.pop_scope();
    body->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_SIZE;
    EXPECT_TRUE(cmper.compare(sch(body), body_aggresive));

    // test2, cannot reuse because of interleaved access
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {10});
        _tensor_(b, datatypes::f32, {50});
        a[0] = external[0]; // 2...1
        b[0] = a[0]; // 4...3
        b[2] = a[2]; // 6...5
        b[1] = external[2]; // 8...7
        a[0] = b[1]; // 10...9
    }
    body = bld.pop_scope();
    body->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_WHOLE;
    // A: FirstAccessTick=FAT~2, LastReadTick=LRT~5
    // B: FAT~4, LRT~9
    EXPECT_TRUE(sch(body).ptr_same(body));

    bld.push_scope();
    {
        _tensor_(rescheduled, datatypes::s8, {320UL});
        _tensor_(a, datatypes::f32, {10});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {0UL});
        _tensor_(b, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(rescheduled, {64UL});
        a[0] = external[0]; // 2...1
        b[0] = a[0]; // 4...3
        b[2] = a[2]; // 6...5
        b[1] = external[2]; // 8...7
        a[0] = b[1]; // 10...9
    }
    body_aggresive = bld.pop_scope();
    body->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_SIZE;
    EXPECT_TRUE(cmper.compare(sch(body), body_aggresive));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestFuncSchedule) {
    ir_builder_t bld;
    bld.push_scope();

    _decl_func_(datatypes::f32, somefunc, _arg_("a", datatypes::f32, {100}),
            _arg_("b", datatypes::f32, {100}));
    somefunc->params_[0]->attr()["write_buffer"] = true;
    somefunc->params_[0]->attr()["read_buffer"] = true;

    _decl_func_(datatypes::f32, unary_func, _arg_("a", datatypes::f32, {100}));
    unary_func->params_[0]->attr()["write_buffer"] = true;
    _tensor_(external, datatypes::f32, {100});
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {50});

        a[0] = external[0];
        _evaluate_call_(somefunc, b, a);

        _tensor_(c, datatypes::f32, {100});
        _evaluate_call_(somefunc, c, builder::tensor_ptr(b, {1}));

        _tensor_(d, datatypes::f32, {100});
        _evaluate_call_(somefunc, builder::tensor_ptr(d, {1}), c);

        _tensor_(e, datatypes::f32, {100});
        _evaluate_call_(somefunc, e, d);
    }
    auto body = bld.pop_scope();

    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {100UL});
        a[0] = external[0];
        _evaluate_call_(somefunc, b, a);

        bld.push_scope();
        bld.emit(bld.pop_scope());
        _evaluate_call_(somefunc, a, builder::tensor_ptr(b, {1}));

        bld.push_scope();
        bld.emit(bld.pop_scope());
        _evaluate_call_(somefunc, builder::tensor_ptr(b, {1}), a);

        bld.push_scope();
        bld.emit(bld.pop_scope());
        _evaluate_call_(somefunc, a, b);
    }
    auto body2 = bld.pop_scope();
    buffer_scheduler_t sch(cur_ctx, false);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(body), body2));

    // original
    bld.push_scope();
    {
        _tensor_(b, datatypes::f32, {50});
        _tensor_(a, datatypes::f32, {100});
        b[0] = b[0];
        _evaluate_call_(unary_func, a);
    }
    body = bld.pop_scope();
    // expected when unary_func is write only: b can reuse a
    bld.push_scope();
    {
        bld.push_scope();
        bld.emit(bld.pop_scope());
        _tensor_(a, datatypes::f32, {100});
        a[0] = a[0];
        _evaluate_call_(unary_func, a);
    }
    body2 = bld.pop_scope();
    EXPECT_TRUE(cmper.compare(sch(body), body2));

    // expected when unary_func is read only or rw: a can reuse b
    bld.push_scope();
    {
        _tensor_(b, datatypes::f32, {100UL});
        bld.push_scope();
        bld.emit(bld.pop_scope());
        b[0] = b[0];
        _evaluate_call_(unary_func, b);
    }
    body2 = bld.pop_scope();
    unary_func->params_[0]->attr().remove("write_buffer");
    // assume readwrite
    EXPECT_TRUE(cmper.compare(sch(body), body2));

    unary_func->params_[0]->attr()["read_buffer"] = true;
    EXPECT_TRUE(cmper.compare(sch(body), body2));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestScope) {
    ir_builder_t bld;
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _var_(scalar, datatypes::f32);
        bld.push_scope();
        {
            _tensor_(b, datatypes::f32, {50});
            scalar = b[0];
            a[0] = 1;
        }
        bld.emit(bld.pop_scope());
        scalar = a[2];
    }
    auto body = bld.pop_scope();
    // b.LRT < a.FAT
    // but a cannot reuse buffer of b, because a outlives b

    buffer_scheduler_t sch(cur_ctx, false);
    EXPECT_TRUE(sch(body).ptr_same(body));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestThreadLocal) {
    ir_builder_t bld;
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {50UL});
        a[0] = 0;
        _for_(i, 0, 10) {
            _tensor_(b, datatypes::f32, {50UL});
            b[0] = 1;
            _tensor_(c, datatypes::f32, {20UL});
            c[10] = b[0];
            _tensor_(f, datatypes::f32, {50UL});
            _for_(j, 0, 100, 1, for_type::PARALLEL) {
                _tensor_(d, datatypes::f32, {50UL});
                d[10] = 2;
                _tensor_(e, datatypes::f32, {20UL});
                e[0] = d[10];
                f[0] = 1;
                c[0] = 1;
                _for_(k, 0, 200) {
                    _tensor_(g, datatypes::f32, {50UL});
                    _tensor_(h, datatypes::f32, {50UL});
                    g->attr().set(attr_keys::hint_first_access_tick,
                            static_cast<int64_t>(1));
                    g->attr().set(attr_keys::hint_last_access_tick,
                            static_cast<int64_t>(3));
                    h->attr().set(attr_keys::hint_first_access_tick,
                            static_cast<int64_t>(5));
                    h->attr().set(attr_keys::hint_last_access_tick,
                            static_cast<int64_t>(7));
                    e[0] = 3;
                    g[0] = e[10];
                    h[0] = g[0];
                }
            }
        }
        _for_(i, 0, 200, 1, for_type::PARALLEL) {
            _tensor_(m, datatypes::f32, {50UL});
            _tensor_(n, datatypes::f32, {20UL});
            m[0] = 3;
            n[0] = m[0];
            _for_(j, 0, 100) {
                _tensor_(p, datatypes::f32, {50UL});
                _tensor_(q, datatypes::f32, {10UL});
                p[0] = 3;
                q[0] = p[0];
            }
        }
    }
    auto body = bld.pop_scope();
    body->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_HOT;

    buffer_scheduler_t sch(cur_ctx, false);
    auto result = sch(body);
    bld.push_scope();
    {
        _tensor_(resch, datatypes::s8, {640UL});
        _tensor_(a, datatypes::f32, {50UL});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(resch.get(), {0UL});
        a[0] = 0;
        _for_(i, 0, 10) {
            _tensor_(b, datatypes::f32, {50UL});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(resch.get(), {0UL});
            b[0] = 1;
            _tensor_(c, datatypes::f32, {20UL});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(resch.get(), {512UL});
            c[10] = b[0];
            _tensor_(f, datatypes::f32, {50UL});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(resch.get(), {256UL});
            _for_(j, 0, 100, 1, for_type::PARALLEL) {
                // parallel scope 1
                _tensor_(resch_in_parallel_0, datatypes::s8, {384UL});
                _tensor_(d, datatypes::f32, {50UL});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(resch_in_parallel_0.get(), {0UL});
                d[10] = 2;
                _tensor_(e, datatypes::f32, {20UL});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(resch_in_parallel_0.get(), {0UL});
                e[0] = d[10];
                f[0] = 1;
                c[0] = 1;
                _for_(k, 0, 200) {
                    _tensor_(g, datatypes::f32, {50UL});
                    bld.get_current_scope()
                            .body.back()
                            .checked_as<define>()
                            ->init_
                            = builder::tensor_ptr(
                                    resch_in_parallel_0.get(), {128UL});
                    _tensor_(h, datatypes::f32, {50UL});
                    bld.get_current_scope()
                            .body.back()
                            .checked_as<define>()
                            ->init_
                            = builder::tensor_ptr(
                                    resch_in_parallel_0.get(), {128UL});
                    e[0] = 3;
                    g[0] = e[10];
                    h[0] = g[0];
                }
            }
        }
        _for_(i, 0, 200, 1, for_type::PARALLEL) {
            // parallel scope 2
            _tensor_(resch_in_parallel_2, datatypes::s8, {320UL});
            _tensor_(m, datatypes::f32, {50UL});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(resch_in_parallel_2.get(), {0UL});
            _tensor_(n, datatypes::f32, {20UL});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(resch_in_parallel_2.get(), {0UL});
            m[0] = 3;
            n[0] = m[0];
            _for_(j, 0, 100) {
                _tensor_(p, datatypes::f32, {50UL});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(resch_in_parallel_2.get(), {0UL});
                _tensor_(q, datatypes::f32, {10UL});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(
                                resch_in_parallel_2.get(), {256UL});
                p[0] = 3;
                q[0] = p[0];
            }
        }
    }
    auto expected = bld.pop_scope();
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(result, expected, false));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestFor) {
    ir_builder_t bld;
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {50});
        _for_(i, 0, 10) {
            b[i] = b[i + UINT64_C(100)];
            a[i] = 0;
        }
        a[0] = a[10];
    }
    auto body = bld.pop_scope();
    // b.LRT < a.FAT in the loop
    // but a cannot reuse buffer of b, because they are in a loop
    // the ticks will be adjusted to: b.LRT = a.LWT > a.FAT

    buffer_scheduler_t sch(cur_ctx, false);
    EXPECT_TRUE(sch(body).ptr_same(body));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestArgBuff) {
    ir_builder_t bld;

    _function_(datatypes::void_t, aaa, _arg_("a", datatypes::f32, {100})) {
        _bind_(a);
        a->attr()["write_buffer"] = true;
        _tensor_(b, datatypes::f32, {50});
        b[10] = 123.f;
        _tensor_(c, datatypes::f32, {1000});
        c[100] = 0.f;
        a[0] = c[10];
    }

    _function_(
            datatypes::void_t, aaa_check, _arg_("a", datatypes::f32, {100})) {
        _bind_(a);
        bld.push_scope();
        bld.emit(bld.pop_scope());
        a[10] = 123.f;
        _tensor_(c, datatypes::f32, {1000});
        c[100] = 0.f;
        a[0] = c[10];
    }
    buffer_scheduler_t sch(cur_ctx, false);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(aaa), aaa_check));

    _function_(datatypes::void_t, aaa2, _arg_("a", datatypes::f32, {100})) {
        _bind_(a);
        a->attr()["write_buffer"] = true;
        _tensor_(b, datatypes::f32, {50});
        a[0] = 1.f;
        // cannot reuse a for b, because b may write to the final result of a
        b[10] = 123.f;
    }
    EXPECT_TRUE(sch(aaa2) == aaa2);
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestDeadWriteEliminate) {
    ir_builder_t bld;
    _function_(datatypes::void_t, aaa, _arg_("a", datatypes::f32, {100}),
            _arg_("b", datatypes::f32, {100})) {
        _bind_(a, b);
        b[10] = 123;
        a->attr()["write_buffer"] = true;
        _tensor_(c, datatypes::f32, {50});
        c[10] = 123.f;
        _tensor_(d, datatypes::f32, {1000});
        d[100] = c[10];
        _if_(true) { d[0] = 2; }
        _tensor_(e, datatypes::f32, {1000});
        _for_(i, 0, 10) {
            i = e[0];
            e[0] = 20;
        }
        a[0] = c[10];
    }
    _function_(datatypes::void_t, aaa_check, _arg_("a", datatypes::f32, {100}),
            _arg_("b", datatypes::f32, {100})) {
        _bind_(a, b);
        b[10] = 123;
        a->attr()["write_buffer"] = true;
        bld.push_scope();
        bld.emit(bld.pop_scope());
        a[10] = 123.f;
        bld.push_scope();
        bld.emit(bld.pop_scope());
        bld.push_scope();
        bld.emit(bld.pop_scope());
        _if_(true) {
            bld.push_scope();
            bld.emit(bld.pop_scope());
        }
        _tensor_(e, datatypes::f32, {1000});
        _for_(i, 0, 10) {
            i = e[0];
            e[0] = 20;
        }
        a[0] = a[10];
    }
    buffer_scheduler_t sch(cur_ctx, true);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(aaa), aaa_check));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestDeadWriteDiffScope) {
    ir_builder_t bld;
    _function_(datatypes::void_t, aaa, _arg_("a", datatypes::f32, {100}),
            _arg_("b", datatypes::f32, {100})) {
        _bind_(a, b);
        _tensor_(c, datatypes::f32, {20UL});
        _tensor_(d, datatypes::f32, {50UL});
        _tensor_(e, datatypes::f32, {50UL});
        _for_(i, 0, 10, 1, for_type::PARALLEL) {
            // normal schedule
            d[10] = a[0];
            // deadwrite
            c[0] = 0;
        }
        // normal schedule
        e[20] = d[1];
        b[40] = e[3];
    }
    aaa->attr()[attr_keys::buf_sched_type] = attr_keys::BUF_SCHED_HOT;

    _function_(datatypes::void_t, expected, _arg_("a", datatypes::f32, {100}),
            _arg_("b", datatypes::f32, {100})) {
        _bind_(a, b);
        _tensor_(resch_0, datatypes::s8, {256UL});
        bld.push_scope();
        bld.emit(bld.pop_scope());
        _tensor_(d, datatypes::f32, {50UL});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(resch_0.get(), {0UL});
        _tensor_(e, datatypes::f32, {50UL});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(resch_0.get(), {0UL});
        _for_(i, 0, 10, 1, for_type::PARALLEL) {
            // normal schedule
            d[10] = a[0];
            bld.push_scope();
            bld.emit(bld.pop_scope());
        }
        // normal schedule
        e[20] = d[1];
        b[40] = e[3];
    }
    buffer_scheduler_t sch(cur_ctx, true);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(aaa), expected));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestRecursive) {
    // test the case that a->b, b->c, d->c
    // a should reuse c instead of b
    ir_builder_t bld;
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {50});
        _tensor_(c, datatypes::f32, {100});
        _tensor_(d, datatypes::f32, {100});
        a[0] = b[0];
        c[0] = 1;
        b[0] = 1;
        c[0] = d[0];
    }
    auto body = bld.pop_scope();

    bld.push_scope();
    {
        bld.push_scope();
        bld.emit(bld.pop_scope());
        bld.push_scope();
        bld.emit(bld.pop_scope());
        _tensor_(c, datatypes::f32, {100});
        bld.push_scope();
        bld.emit(bld.pop_scope());

        c[0] = c[0];
        c[0] = 1;
        c[0] = 1;
        c[0] = c[0];
    }
    auto body2 = bld.pop_scope();

    buffer_scheduler_t sch(cur_ctx, false);
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(sch(body), body2));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestListBrgemm) {
    ir_builder_t bld;
    bld.push_scope();
    {
        _tensor_(a, datatypes::f32, {100});
        _tensor_(b, datatypes::f32, {50});
        _tensor_(c, datatypes::f32, {50});
        _tensor_(d, datatypes::f32, {50});
        _tensor_(e, datatypes::f32, {50});

        _tensor_(l_a, datatypes::pointer, {1});
        l_a[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(a, {1}));
        _tensor_(l_b, datatypes::pointer, {1});
        l_b[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(b, {1}));
        _tensor_(l_c, datatypes::pointer, {1});
        l_c[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(c, {1}));
        _tensor_(l_d, datatypes::pointer, {1});
        l_d[0] = builder::make_cast(datatypes::pointer, d);

        builtin::brgemm_list_update(l_a, l_b, c, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
        builtin::brgemm_list_update(l_c, l_b, d, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
        builtin::brgemm_list_update(l_d, l_b, e, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
    }
    auto body = bld.pop_scope();

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.buffer_schedule_ = 3;
    buffer_scheduler_t sch(ctx, false);
    auto out = sch(body);

    bld.push_scope();
    {
        _tensor_(scheduled, datatypes::s8, {1216UL});
        _tensor_(a, datatypes::f32, {100});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {256UL});
        _tensor_(b, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {704UL});
        _tensor_(c, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {960UL});
        _tensor_(d, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {256UL});
        _tensor_(e, datatypes::f32, {50});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {960UL});

        _tensor_(l_a, datatypes::pointer, {1});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {0UL});
        l_a[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(a, {1}));
        _tensor_(l_b, datatypes::pointer, {1});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {64UL});
        l_b[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(b, {1}));
        _tensor_(l_c, datatypes::pointer, {1});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {128UL});
        l_c[0] = builder::make_cast(
                datatypes::pointer, builder::tensor_ptr(c, {1}));
        _tensor_(l_d, datatypes::pointer, {1});
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(scheduled.get(), {192UL});
        l_d[0] = builder::make_cast(datatypes::pointer, d);

        builtin::brgemm_list_update(l_a, l_b, c, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
        builtin::brgemm_list_update(l_c, l_b, d, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
        builtin::brgemm_list_update(l_d, l_b, e, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                datatypes::f32, datatypes::f32);
    }
    auto expected = bld.pop_scope();
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(expected, expected, false));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestForScope) {
    ir_builder_t bld;
    bld.push_scope();
    {
        for_loop loop;
        _named_for_(loop, i, 0, 10) {
            _tensor_(a, datatypes::f32, {50});
            _tensor_(b, datatypes::f32, {50});
            a[0] = 1;
            b[0] = a[0];
            builtin::print_float(a[0]);
            _tensor_(c, datatypes::f32, {50});
            c[0] = b[0];
            builtin::print_float(b[0]);
            builtin::print_float(c[0]);
        }
        loop->attr()[attr_keys::buf_sched_top_scope] = true;
    }
    auto body = bld.pop_scope();

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.buffer_schedule_ = 3;
    buffer_scheduler_t sch(ctx, false);
    auto out = sch(body);

    bld.push_scope();
    {
        _tensor_(scheduled, datatypes::s8, {512UL});
        _for_(i, 0, 10) {
            _tensor_(a, datatypes::f32, {50});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(scheduled.get(), {0UL});
            _tensor_(b, datatypes::f32, {50});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(scheduled.get(), {256UL});
            a[0] = 1;
            b[0] = a[0];
            builtin::print_float(a[0]);
            _tensor_(c, datatypes::f32, {50});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(scheduled.get(), {0UL});
            c[0] = b[0];
            builtin::print_float(b[0]);
            builtin::print_float(c[0]);
        }
    }
    auto expected = bld.pop_scope();
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(expected, expected, false));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestTempBufferInplace) {
    ir_builder_t bld;

    _function_(
            datatypes::void_t, main_entry, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(A, datatypes::f32, 100);
        auto id_a = alias_info::get_or_create_alias_info(*A.get());
        _tensor_(B, datatypes::f32, 100);
        auto id_b = alias_info::get_or_create_alias_info(*B.get());
        _tensor_(temp1, datatypes::f32, 100);
        A[0] = 1;
        B[0] = 1;
        _for_(i, 0, 100) { temp1[i] = A[i] + B[i]; }

        temp1->attr()[attr_keys::tensor_inplace_hint]
                = std::vector<temp_tensor_inplace_info_t> {
                        {id_a, inplace_kind::FREE}};
        temp1[0] = 1;
    }
    auto ctx = make_ctx();
    ctx->flags_.buffer_schedule_ = 2;
    buffer_scheduler_t pass {ctx, false, true};

    _function_(datatypes::void_t, expected, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(sched, datatypes::s8, UINT64_C(896));
        _tensor_(A, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        _tensor_(B, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(448)});
        _tensor_(temp1, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        A[0] = 1;
        B[0] = 1;
        _for_(i, 0, 100) { temp1[i] = A[i] + B[i]; }
        temp1[0] = 1;
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(pass(main_entry), expected, false));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestAlreadyScheduled) {
    ir_builder_t bld;

    _function_(
            datatypes::void_t, main_entry, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(A_base, datatypes::f32, 100);
        A_base.get()->attr()[attr_keys::can_be_scheduled] = true;
        _tensor_(A, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(A_base, {UINT64_C(0)});
        A[0] = 1;

        _tensor_(B_base, datatypes::f32, 100);
        B_base.get()->attr()[attr_keys::can_be_scheduled] = true;
        _tensor_(B, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(B_base, {UINT64_C(0)});
        B[0] = 1;
        C[0] = A[0];
        C[0] = B[0];

        _tensor_(D_base, datatypes::f32, 100);
        D_base.get()->attr()[attr_keys::can_be_scheduled] = true;
        _tensor_(D, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(D_base, {UINT64_C(0)});
        C[0] = D[0];
    }
    auto ctx = make_ctx();
    ctx->flags_.buffer_schedule_ = 2;
    buffer_scheduler_t pass {ctx, false, true};

    _function_(datatypes::void_t, expected, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(sched, datatypes::s8, UINT64_C(896));
        _tensor_(A_base, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        _tensor_(A, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(A_base, {UINT64_C(0)});
        A[0] = 1;

        _tensor_(B_base, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(448)});
        _tensor_(B, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(B_base, {UINT64_C(0)});
        B[0] = 1;
        C[0] = A[0];
        C[0] = B[0];

        // D can reuse A
        _tensor_(D_base, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        _tensor_(D, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(D_base, {UINT64_C(0)});
        C[0] = D[0];
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(pass(main_entry), expected, false));
}

TEST(GCCore_CPU_buffer_schedule_cpp, TestInplaceOutputArg) {
    ir_builder_t bld;
    _function_(datatypes::void_t, main_entry,
            _arg_("out", datatypes::f32, {100})) {
        _bind_(out);
        out->attr()["write_buffer"] = true;
        _tensor_(A, datatypes::f32, 100);
        A[0] = 1;
        _tensor_(B, datatypes::f32, 100);
        auto id_B = alias_info::get_or_create_alias_info(*B.get());
        B[0] = 1;
        _tensor_(C, datatypes::f32, 50);
        auto id_C = alias_info::get_or_create_alias_info(*C.get());
        C->attr()[attr_keys::tensor_inplace_hint]
                = std::vector<temp_tensor_inplace_info_t> {
                        {id_B, inplace_kind::ZERO_OFFSET}};
        C[0] = 1;
        B[0] = 1;
        out->attr()[attr_keys::tensor_inplace_hint]
                = std::vector<temp_tensor_inplace_info_t> {
                        {id_C, inplace_kind::ZERO_OFFSET}};
    }
    _function_(
            datatypes::void_t, expected, _arg_("out", datatypes::f32, {100})) {
        _bind_(out);
        out->attr()["write_buffer"] = true;
        _tensor_(sched, datatypes::s8, UINT64_C(448));
        _tensor_(A, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        A[0] = 1;
        _tensor_(B, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(out, {UINT64_C(0)});
        auto id_B = alias_info::get_or_create_alias_info(*B.get());
        B[0] = 1;
        _tensor_(C, datatypes::f32, 50);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(out, {UINT64_C(0)});
        auto id_C = alias_info::get_or_create_alias_info(*C.get());
        C->attr()[attr_keys::tensor_inplace_hint]
                = std::vector<temp_tensor_inplace_info_t> {
                        {id_B, inplace_kind::ZERO_OFFSET}};
        C[0] = 1;
        B[0] = 1;
        out->attr()[attr_keys::tensor_inplace_hint]
                = std::vector<temp_tensor_inplace_info_t> {
                        {id_C, inplace_kind::ZERO_OFFSET}};
    }

    auto ctx = make_ctx();
    ctx->flags_.buffer_schedule_ = 2;
    buffer_scheduler_t pass {ctx, false, true};
    auto out = pass(main_entry);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
