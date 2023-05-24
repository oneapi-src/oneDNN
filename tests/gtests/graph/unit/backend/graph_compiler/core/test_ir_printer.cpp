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

#include <sstream>
#include "context.hpp"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/pass/printer.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_printer_cpp, TestTrackPosPrinter) {
    builder::ir_builder_t builder;
    const int shape1 = 128;
    for_loop li, lj, lk, lp;
    std::stringstream ss;

    stmt the_assign;

    _function_(datatypes::void_t, aaa,
            _arg_("A", datatypes::f32, {shape1, shape1}),
            _arg_("B", datatypes::f32, {shape1, shape1}),
            _arg_("C", datatypes::f32, {shape1, shape1}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, C, len);
        _tensor_(D, datatypes::f32, 2, 10);
        _tensor_(E, datatypes::f32, 100, 20);
        _tensor_(F, datatypes::f32, len);
        _tensor_(F_view, datatypes::s32, 10);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(F, {3});
        builder.get_current_scope().body.back()->attr()["comments"]
                = std::vector<std::string> {"hello", "hi"};
        _named_for_(li, i, 0, shape1) {
            _named_for_(lj, j, 0, shape1) {
                _named_for_(lk, k, 0, shape1) {
                    C[{i, j}] = C[{i, j}] + A[{i, k}] * B[{k, j}];
                    the_assign = builder.get_current_scope().body.back();
                }
            }
        }
    }

    ir_module_ptr m = ir_module_t::from_entry_func(get_test_ctx(), aaa);

    std::string expected1
            = R"(func aaa(A: [f32 * 128 * 128], B: [f32 * 128 * 128], C: [f32 * 128 * 128], len: s32): void {
  tensor D: [f32 * 2 * 10]
  tensor E: [f32 * 100 * 20]
  tensor F: [f32 * len]
  // hello
  // hi
  tensor F_view: [s32 * 10] = &F[3]
  for i in (0, 128, 1) {
    for j in (0, 128, 1) {
      for k in (0, 128, 1) {
        C[i, j] = (C[i, j] + (A[i, k] * B[k, j]))
      }
    }
  }
}
)";
    print_ir_and_annotate_source_pos(*m, ss);
    EXPECT_EQ(ss.str(), expected1);
    auto pos = aaa->attr().get_or_null<source_pos>("source_pos");
    ASSERT_TRUE(pos);
    ASSERT_EQ(*pos, (source_pos {1, 1}));
    pos = li->attr().get_or_null<source_pos>("source_pos");
    ASSERT_TRUE(pos);
    ASSERT_EQ(*pos, (source_pos {2, 8}));
    pos = the_assign->attr().get_or_null<source_pos>("source_pos");
    ASSERT_TRUE(pos);
    ASSERT_EQ(*pos, (source_pos {8, 11}));

    pos = the_assign.static_as<assign>()
                  ->value_.static_as<add>()
                  ->r_->attr()
                  .get_or_null<source_pos>("source_pos");
    ASSERT_TRUE(pos);
    ASSERT_EQ(*pos, (source_pos {29, 11}));
}
