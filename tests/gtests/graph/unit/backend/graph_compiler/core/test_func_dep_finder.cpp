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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/pass/func_dependency.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_func_dep_finder_cpp, TestFuncDepFinder) {
    builder::ir_builder_t builder;
    _decl_func_(datatypes::void_t, bbb, _arg_("A", datatypes::f32, {100, 100}),
            _arg_("len", datatypes::s32), _arg_("tsr", datatypes::pointer));
    _function_(datatypes::s32, ccc, _arg_("A", datatypes::f32, {100, 100}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, len);
        _for_(i, 0, 100) {
            _for_(j, 0, 100) { A[{i, j}] = 0; }
            builder.push_evaluate(bbb(A, len, A + 100));
            builtin::print_str("1234");
        }
        builder.push_evaluate(bbb(A, len, A + 100));
        builtin::print_str("1234");
        builtin::print_int(1234);
        _return_(12);
    }
    std::vector<func_t> f;
    func_dependency_finder_t finder(f);
    finder(ccc);
    ASSERT_EQ(f.size(), 3u);
    EXPECT_EQ(f[0]->name_, "bbb");
    EXPECT_EQ(f[1]->name_, "print_str");
    EXPECT_EQ(f[2]->name_, "print_int");
}
