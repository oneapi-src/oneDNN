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

#include <iostream>
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_cast_cpp, TestCast) {
    expr a = expr(1) + 2;
    builder::ir_builder_t b;
    b.push_scope();
    b.push_assign(a, 9);
    auto v = b.pop_scope();
    auto assn = v.checked_as<stmts>()->seq_[0].checked_as<assign>();
    assign_c assn_c = assn;
    assign_c assn2 = std::move(assn_c);
    EXPECT_FALSE(assn_c.defined());
    auto val = assn->value_.dyn_as<constant>()->get_s32();
    stmt stm = v.checked_as<stmts>();
    // should fail:
    // stmts stm2 = v;
    EXPECT_EQ(val, 9);
}
