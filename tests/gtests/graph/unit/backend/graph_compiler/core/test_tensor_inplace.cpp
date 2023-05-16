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
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/transform/tensor_inplace.hpp>
#include <util/any_map.hpp>

#include <unordered_set>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static context_ptr make_ctx() {
    auto ret = std::make_shared<context_t>(*get_test_ctx());
    ret->flags_.buffer_schedule_ = 2;
    return ret;
}

TEST(GCCore_CPU_tensor_inplace_cpp, TestSimpleSchedule) {
    ir_builder_t bld;
    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {100}),
            _arg_("B", datatypes::f32, {100}),
            _arg_("C", datatypes::f32, {100})) {}
    aaa->attr()[function_attrs::inplace_hint]
            = std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>> {
                    {2, {{0, inplace_kind::FREE}}}};
    // make sure decl and def does not share the same tensor node
    for (auto &arg : aaa->decl_->params_) {
        arg = arg->remake();
    }

    _function_(datatypes::void_t, bbb, _arg_("A", datatypes::f32, {100}),
            _arg_("B", datatypes::f32, {100}),
            _arg_("C", datatypes::f32, {100})) {}
    bbb->attr()[function_attrs::inplace_hint]
            = std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>> {
                    {2, {{0, inplace_kind::FREE}}}};
    // make sure decl and def does not share the same tensor node
    for (auto &arg : bbb->decl_->params_) {
        arg = arg->remake();
    }

    _function_(
            datatypes::void_t, main_entry, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);
        _tensor_(temp1, datatypes::f32, 100);
        A[0] = 1;
        B[0] = 1;
        _evaluate_call_(aaa, A, B, temp1);
        // check that we cannot reuse A, because call of aaa is not the last use
        // of A
        C[0] = A[0];

        _tensor_(D, datatypes::f32, 100);
        // check that we cannot reuse D for temp1, because call of aaa is not
        // the first use of temp1
        _evaluate_call_(aaa, D, B, temp1);

        _tensor_(E, datatypes::f32, 100);
        E[0] = 1;
        _tensor_(temp2, datatypes::f32, 100);
        // we can reuse E for temp1
        _evaluate_call_(bbb, E, B, temp2);
        C[1] = temp2[1];
    }
    auto ctx = make_ctx();
    auto mod = ir_module_t::from_entry_func(ctx, main_entry);
    auto out_mod = tensor_inplace_t(ctx)(mod);

    _function_(datatypes::void_t, expected, _arg_("C", datatypes::f32, {100})) {
        _bind_(C);
        _tensor_(sched, datatypes::s8, UINT64_C(1344));
        _tensor_(A, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        _tensor_(B, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(448)});
        _tensor_(temp1, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(896)});
        A[0] = 1;
        B[0] = 1;
        _evaluate_call_(aaa->decl_, A, B, temp1);
        // check that we cannot reuse A, because call of aaa is not the last use
        // of A
        C[0] = A[0];

        _tensor_(D, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(0)});
        // check that we cannot reuse D for temp1, because call of aaa is not
        // the first use of temp1
        _evaluate_call_(aaa->decl_, D, B, temp1);

        _tensor_(E, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(896)});
        E[0] = 1;
        _tensor_(temp2, datatypes::f32, 100);
        bld.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(sched, {UINT64_C(896)});
        // we can reuse E for temp1
        _evaluate_call_(bbb->decl_, E, B, temp2);
        C[1] = temp2[1];
    }

    EXPECT_FALSE(aaa->params_[0]->attr_);
    EXPECT_FALSE(aaa->params_[1]->attr_);
    EXPECT_FALSE(aaa->params_[2]->attr_);

    // check that alias info is propagated back to func definition
    ASSERT_TRUE(bbb->params_[0]->attr_);
    auto group1 = alias_info::get_alias_info(*bbb->params_[0]);
    ASSERT_TRUE(group1);
    EXPECT_FALSE(bbb->params_[1]->attr_);
    ASSERT_TRUE(bbb->params_[2]->attr_);
    auto group2 = alias_info::get_alias_info(*bbb->params_[2]);
    ASSERT_TRUE(group2);

    ASSERT_NE(group1, group2);
    ASSERT_EQ(group1->alias_cliques_.size(), 1UL);
    ASSERT_EQ(group2->alias_cliques_.size(), 1UL);
    ASSERT_EQ(group1->alias_cliques_[0], group1->alias_cliques_[0]);
    ASSERT_EQ(group1->alias_cliques_[0]->set_.size(), 2UL);
    auto &theset = group1->alias_cliques_[0]->set_;
    auto in_set = theset.find(group2->shared_from_this()) != theset.end();
    ASSERT_TRUE(in_set);

    buffer_scheduler_t scheduler {out_mod->ctx_, true, true};
    auto after_sched = scheduler(out_mod->get_entry_func());
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(after_sched, expected, false));
}
