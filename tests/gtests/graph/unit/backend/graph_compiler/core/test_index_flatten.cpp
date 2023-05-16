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

#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/index_flatten.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_index_flatten_cpp, TestIndexFlatten) {
    builder::ir_builder_t builder;
    for_loop li, lj, lk, lp;
    auto ir_mod = std::make_shared<ir_module_t>(
            get_default_context(), std::vector<func_t> {}, 0);
    _global_tensor_(ir_mod, gv, datatypes::f32, 2, 2);
    int arr[] = {1, 2, 3, 4};
    gv.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(arr, sizeof(arr));
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222, 333}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(D, datatypes::f32, 10, 20);
        _tensor_(F, datatypes::f32, 100, len);
        _var_(ptr, datatypes::pointer);
        gv[{1, 1}] = 123;
        _for_(i, 0, len) {
            A[{i, 10}] = B[{10, i, 4}] + D[{1, i}] + F[{11, i}];
            ptr = builder::tensor_ptr(A, {i, 2});
            expr ptr1 = builder::tensor_ptr(A, {i, 2}, {4, 4});
            ptr1[{1, 1}] = 3;
            expr ptr2 = builder::tensor_ptr(ptr1, {1, 1}, {4, 4});
            ptr2[{2, 1}] = 3;
            // slice mode
            expr ptr3 = builder::tensor_ptr(A, {i, 2}, {}, true);
            ptr3[{1, 1}] = 3;
            expr ptr4 = builder::tensor_ptr(ptr3, {1, 1}, {}, true);
            ptr4[{2, 1}] = 3;
            // reshape slice
            expr ptr5 = builder::tensor_ptr(A, {1, 1}, {120, 120}, true);
            expr ptr6 = builder::tensor_ptr(ptr5, {0, 0}, {1, 14400}, false);
            ptr6[{0, 123}] = 5;
        }
    }

    expr expected_gv = builder::make_tensor("gv",
            std::vector<expr> {(2 - expr(1)) * UINT64_C(1)
                    + (((2 - expr(1)) * UINT64_C(2)) + 1)},
            datatypes::f32, address_space::automatic,
            gv.static_as<tensor>()->init_value_);

    _function_(datatypes::s32, expected,
            _arg_("A", datatypes::f32, {123 * 321}),
            _arg_("B", datatypes::f32, {111}), _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(D, datatypes::f32,
                (20 - expr(1)) * UINT64_C(1)
                        + ((10 - expr(1)) * UINT64_C(20) + 1));
        _tensor_(F, datatypes::f32,
                (len - 1) * UINT64_C(1)
                        + ((100 - expr(1))
                                        * builder::make_cast(
                                                datatypes::index, len)
                                + 1));
        _var_(ptr, datatypes::pointer);
        expected_gv[1 * expr(UINT64_C(2)) + 1 * expr(UINT64_C(1))] = 123;
        _for_(i, 0, len) {
            A[i * expr(UINT64_C(321)) + 10 * expr(UINT64_C(1))]
                    = B[10 * expr(UINT64_C(73926))
                              + (i * expr(UINT64_C(333))
                                      + 4 * expr(UINT64_C(1)))]
                    + D[1 * expr(UINT64_C(20)) + i * UINT64_C(1)]
                    + F[11 * builder::make_cast(datatypes::index, len)
                            + i * UINT64_C(1)];
            ptr = builder::tensor_ptr(
                    A, {i * expr(UINT64_C(321)) + 2 * expr(UINT64_C(1))});
            A[(i * expr(UINT64_C(321)) + 2 * expr(UINT64_C(1)))
                    + (1 * (1 * expr(4)) + 1 * expr(1))]
                    = 3;
            A[(i * expr(UINT64_C(321)) + 2 * expr(UINT64_C(1)))
                    + (1 * (1 * expr(4)) + 1 * expr(1))
                    + (2 * (1 * expr(4)) + 1 * expr(1))]
                    = 3;
            // slice mode
            A[(i * expr(UINT64_C(321)) + 2 * expr(UINT64_C(1)))
                    + (1 * expr(UINT64_C(321)) + 1 * expr(UINT64_C(1)))]
                    = 3;
            A[(i * expr(UINT64_C(321)) + 2 * expr(UINT64_C(1)))
                    + (1 * expr(UINT64_C(321)) + 1 * expr(UINT64_C(1)))
                    + (2 * expr(UINT64_C(321)) + 1 * expr(UINT64_C(1)))]
                    = 3;
            A[((((1 * expr(321UL)) + (1 * expr(1UL)))
                       + ((0 * expr(321UL)) + (0 * expr(1UL))))
                    + (((((123 * expr(1)) + expr(0UL)) / expr(120UL))
                               * expr(321UL))
                            + ((((((123 * expr(1)) + expr(0UL)) / expr(1UL))
                                        % expr(120UL))
                                       * expr(1UL))
                                    + expr(0))))]
                    = 5;
        }
    }
    expected->name_ = "aaa";
    expected->params_[0].as<tensor>()->dims_ = {(321 - expr(1)) * UINT64_C(1)
            + ((123 - expr(1)) * expr(UINT64_C(321)) + 1)};
    expected->params_[1].as<tensor>()->dims_ = {(333 - expr(1)) * UINT64_C(1)
            + ((222 - expr(1)) * expr(UINT64_C(333))
                    + ((111 - expr(1)) * expr(UINT64_C(73926)) + 1))};
    ir_mod->add_func({aaa});
    auto out = index_flattener_t()(ir_mod);
    ir_comparer cmper(false, true);
    expr gv_new = out->get_module_vars().at(0)->var_;
    EXPECT_TRUE(cmper.compare(gv_new, expected_gv, false));
    EXPECT_TRUE(cmper.compare(out->get_entry_func(), expected));
}
