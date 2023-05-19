/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include <iostream>
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/parallel_workload_dispatch.hpp>
#include <test_utils.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_parallel_workload_dispatch, TestValidateWorkload) {
    builder::ir_builder_t builder;
    parallel_workload_dispatcher_t pass(true);
    for_loop lk, lm, lp;
    _function_(datatypes::void_t, aaa, _arg_("args", datatypes::f32)) {
        _bind_(args);
        _for_(i, 0, 123) {
            _tensor_(temp, datatypes::f32, {100});
            _for_(j, 0, 456) {
                args = 1;
                _if_(args > 1) {
                    _named_for_(lk, k, 0, 1024, 16) { args = 2; }
                }
                _else_ {
                    _tensor_(temp1, datatypes::f32, {100});
                    _named_for_(lm, m, 0, 16, 16) { args = 3; }
                }
            }
            _named_for_(lp, p, 0, 789) { args = 4; }
        }
    }
    lk->body_->attr().set(
            op_traits::workload_computable_t::workload_number, size_t(2));
    lm->body_->attr().set(
            op_traits::workload_computable_t::workload_number, size_t(5));
    lp->body_->attr().set(
            op_traits::workload_computable_t::workload_number, size_t(3));
    auto bbb = pass(aaa);
    size_t wkld_j = (size_t(2) * 1024) * 456;
    size_t wkld_i = (wkld_j + size_t(3) * 789) * 123;
    stmt li = bbb->body_.checked_as<stmts>()->seq_[0];
    stmt lj = li.checked_as<for_loop>()->body_.checked_as<stmts>()->seq_[1];
    EXPECT_EQ(pass.stmt_workload_map_[li], wkld_i);
    EXPECT_EQ(pass.stmt_workload_map_[lj], wkld_j);
}

TEST(GCCore_CPU_parallel_workload_dispatch, TestParallelElimination) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;
    parallel_workload_dispatcher_t pass;
    _function_(datatypes::void_t, aaa, _arg_("args", datatypes::f32)) {
        _bind_(args);
        auto assign_stmt = builder::make_assign_unattached(args, 4);
        assign_stmt->attr().set(
                op_traits::workload_computable_t::workload_number, size_t(4));
        // will reduce the number of threads
        _for_(i, 0UL, 64UL, 1, for_type::PARALLEL) {
            _for_(k, 0, 256, 1) { builder.emit(assign_stmt); }
        }
        _for_(i, 0, 1024, 1, for_type::PARALLEL) {
            _var_(a, datatypes::s32);
            a = builder::make_get_group_thread_id(-1);
            _for_(j, 0, 16, 16) {
                _for_(k, 0, 512, 1) { builder.emit(assign_stmt); }
            }
        }
    }
    auto bbb = pass(aaa);

    _function_(datatypes::void_t, expected, _arg_("args", datatypes::f32)) {
        _bind_(args);
        auto assign_stmt = builder::make_assign_unattached(args, 4);
        assign_stmt->attr().set(
                op_traits::workload_computable_t::workload_number, size_t(4));
        // will be split.
        _for_(tid, 0UL, 2UL, 1UL, for_type::PARALLEL) {
            _var_init_(start, datatypes::index, tid * UINT64_C(32));
            _var_init_(end, datatypes::index, start + UINT64_C(32));
            _for_(i, start, end, 1) {
                _for_(k, 0, 256, 1) { builder.emit(assign_stmt); }
            }
        }
        _for_(tid, 0UL, 16UL, 1UL, for_type::PARALLEL) {
            _var_init_(start, datatypes::index, tid * UINT64_C(64));
            _var_init_(end, datatypes::index, start + UINT64_C(64));
            _for_(i, start, end, 1) {
                _var_(a, datatypes::s32);
                a = builder::make_cast(datatypes::s32, tid);
                _for_(j, 0, 16, 16) {
                    _for_(k, 0, 512, 1) { builder.emit(assign_stmt); }
                }
            }
        }
    }
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(bbb, expected, false));
}
