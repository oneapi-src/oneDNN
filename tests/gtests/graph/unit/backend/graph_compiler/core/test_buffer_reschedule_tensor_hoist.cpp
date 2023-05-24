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

#include "context.hpp"
#include "test_utils.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/module_pass.hpp>
#include <compiler/ir/transform/buffer_reschedule_tensor_hoist.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/barrier.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_buffer_reschedule_tensor_hoist_cpp,
        TestHoistTensorOutOfParallel_Levels1) {
    auto ctx = get_test_ctx();
    builder::ir_builder_t bld;

    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {4096}),
            _arg_("B", datatypes::f32, {4096})) {
        _bind_(A, B);
        _for_(i, 0, 1024, 1, for_type::PARALLEL, 4) { // inner-most
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            _for_(j, 0, 1024, 1) { // normal for
                C[i] = A[i] * 2 + 3.0f;
            }
            B[i * 2] = C[i] / 2;
        }
    }

    buffer_rescheduling_tensor_hoisting_t pass(ctx, true);
    func_c out = pass(aaa);

    // tensor C cannot be hoisted because it is in inner-most
    _function_(datatypes::void_t, expected, _arg_("A", datatypes::f32, {4096}),
            _arg_("B", datatypes::f32, {4096})) {
        _bind_(A, B);
        _for_(i, 0, 1024, 1, for_type::PARALLEL, 4) {
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            _for_(j, 0, 1024, 1) { C[i] = A[i] * 2 + 3.0f; }
            B[i * 2] = C[i] / 2;
        }
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_buffer_reschedule_tensor_hoist_cpp,
        TestHoistTensorOutOfParallel_Levels2) {
    auto ctx = get_test_ctx();
    builder::ir_builder_t bld;

    _function_(datatypes::void_t, aaa, _arg_("A", datatypes::f32, {4096}),
            _arg_("B", datatypes::f32, {4096})) {
        _bind_(A, B);
        _for_(i, 0, 1024, 1, for_type::PARALLEL, 4) {
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            _for_(j, 0, 128, 1, for_type::PARALLEL, 2) {
                _tensor_(D, datatypes::f32, {UINT64_C(256)});
                B[i * 2] = C[i] + D[j] + A[i * 3];
            }
        }
    }

    buffer_rescheduling_tensor_hoisting_t pass(ctx, true);
    func_c out = pass(aaa);

    _function_(datatypes::void_t, expected, _arg_("A", datatypes::f32, {4096}),
            _arg_("B", datatypes::f32, {4096})) {
        _bind_(A, B);
        bld.push_scope();
        _tensor_(hoisted_rescheduled_0, datatypes::s8, {UINT64_C(16384)});
        _for_(i, 0, 1024, 1, for_type::PARALLEL, 4) {
            _tensor_(rescheduled_0, datatypes::s8, {UINT64_C(4096)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(hoisted_rescheduled_0.get(),
                            {(builder::make_get_group_id(UINT64_C(0))
                                    * UINT64_C(4096))});
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(rescheduled_0.get(), {UINT64_C(0)});
            _for_(j, 0, 128, 1, for_type::PARALLEL, 2) {
                _tensor_(rescheduled_1, datatypes::s8, {UINT64_C(1024)});
                _tensor_(D, datatypes::f32, {UINT64_C(256)});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(
                                rescheduled_1.get(), {UINT64_C(0)});
                B[i * 2] = C[i] + D[j] + A[i * 3];
            }
        }
        bld.emit(bld.pop_scope());
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_buffer_reschedule_tensor_hoist_cpp,
        TestHoistTensorOutOfParallel_Levels_InitTensorD) {
    auto ctx = get_test_ctx();
    builder::ir_builder_t bld;

    _function_(datatypes::void_t, aaa, _arg_("A_Input", datatypes::f32, {4096}),
            _arg_("B_Output", datatypes::f32, {4096})) {
        _bind_(A_Input, B_Output);

        _for_(i, 0, 1024, 1, for_type::PARALLEL, 2) {
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            _for_(j, 0, 1024, 1, for_type::PARALLEL, 2) {
                _tensor_(D, datatypes::f32, {UINT64_C(1024)});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(A_Input, {UINT64_C(1024)});
                B_Output[j] = C[i] + D[j];
            }
        }
    }

    buffer_rescheduling_tensor_hoisting_t pass(ctx, true);
    func_c out = pass(aaa);

    _function_(datatypes::void_t, expected,
            _arg_("A_Input", datatypes::f32, {4096}),
            _arg_("B_Output", datatypes::f32, {4096})) {
        _bind_(A_Input, B_Output);
        bld.push_scope();
        _tensor_(hoisted_C, datatypes::f32, {UINT64_C(2048)});
        _for_(i, 0, 1024, 1, for_type::PARALLEL, 2) {
            _tensor_(C, datatypes::f32, {UINT64_C(1024)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(hoisted_C.get(),
                            {(builder::make_get_group_id(UINT64_C(0))
                                    * UINT64_C(1024))});
            _for_(j, 0, 1024, 1, for_type::PARALLEL, 2) {
                _tensor_(D, datatypes::f32, {UINT64_C(1024)});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(A_Input, {UINT64_C(1024)});
                B_Output[j] = C[i] + D[j];
            }
        }
        bld.emit(bld.pop_scope());
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_buffer_reschedule_tensor_hoist_cpp,
        TestHoistTensorOutOfParallel_Levels2_Parallel) {
    auto ctx = get_test_ctx();
    builder::ir_builder_t bld;

    _function_(datatypes::void_t, aaa) {
        _for_(i, 0, 100, 1, for_type::PARALLEL, 4) {
            _tensor_(A, datatypes::f32, {UINT64_C(100)});
            _for_(j, 0, 100, 1, for_type::PARALLEL, 2) {
                _tensor_(B, datatypes::f32, {UINT64_C(100)});
                B[j] = A[j] + 1;
            }
            _tensor_(A2, datatypes::f32, {UINT64_C(100)});
            _for_(j, 0, 100, 1, for_type::PARALLEL, 2) {
                _tensor_(B2, datatypes::f32, {UINT64_C(100)});
                B2[j] = A2[j] + 2;
            }
        }
    }
    buffer_rescheduling_tensor_hoisting_t pass(ctx, false);
    func_c out = pass(aaa);

    _function_(datatypes::void_t, expected) {
        bld.push_scope();
        _tensor_(hoisted_rescheduled_0, datatypes::s8, {UINT64_C(1792)});
        _for_(i, 0, 100, 1, for_type::PARALLEL, 4) {
            _tensor_(rescheduled_0, datatypes::s8, {UINT64_C(448)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(hoisted_rescheduled_0.get(),
                            {(builder::make_get_group_id(UINT64_C(0))
                                    * UINT64_C(448))});
            _tensor_(A, datatypes::f32, {UINT64_C(100)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(rescheduled_0.get(), {UINT64_C(0)});
            _for_(j, 0, 100, 1, for_type::PARALLEL, 2) {
                _tensor_(rescheduled_1, datatypes::s8, {UINT64_C(448)});
                _tensor_(B, datatypes::f32, {UINT64_C(100)});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(
                                rescheduled_1.get(), {UINT64_C(0)});
                B[j] = A[j] + 1;
            }
            _tensor_(A2, datatypes::f32, {UINT64_C(100)});
            bld.get_current_scope().body.back().checked_as<define>()->init_
                    = builder::tensor_ptr(rescheduled_0.get(), {UINT64_C(0)});
            _for_(j, 0, 100, 1, for_type::PARALLEL, 2) {
                _tensor_(rescheduled_2, datatypes::s8, {UINT64_C(448)});
                _tensor_(B2, datatypes::f32, {UINT64_C(100)});
                bld.get_current_scope().body.back().checked_as<define>()->init_
                        = builder::tensor_ptr(
                                rescheduled_2.get(), {UINT64_C(0)});
                B2[j] = A2[j] + 2;
            }
        }
        bld.emit(bld.pop_scope());
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected));
}
