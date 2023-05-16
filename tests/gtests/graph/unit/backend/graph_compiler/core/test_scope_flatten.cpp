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
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_scope_flatten_cpp, TestScopeFlatten) {
    builder::ir_builder_t builder;
    auto mymake = [&]() {
        builder.push_scope();
        {
            builder.push_scope();
            {
                _var_(len1, datatypes::s32);
                _var_(len, datatypes::f32);
            }
            builder.emit(builder.pop_scope());
            builder.push_scope();
            {
                _var_(len2, datatypes::s32);
                _var_(len3, datatypes::f32);
            }
            builder.emit(builder.pop_scope());
        }
        return builder.pop_scope().as<stmts>();
    };

    /////// expected
    builder.push_scope();
    {
        _var_(len1, datatypes::s32);
        _var_(len, datatypes::f32);
        builder.push_scope();
        {
            _var_(len2, datatypes::s32);
            _var_(len3, datatypes::f32);
        }
        builder.emit(builder.pop_scope());
    }
    auto exp = builder.pop_scope();

    ir_comparer cmp(true);
    auto body = mymake();
    scope_flatten(body, 0);
    EXPECT_TRUE(cmp.compare(body, exp));

    builder.push_scope();
    {
        _var_(len1, datatypes::s32);
        _var_(len, datatypes::f32);
        _var_(len2, datatypes::s32);
        _var_(len3, datatypes::f32);
    }
    exp = builder.pop_scope();
    body = mymake();
    scope_flatten(body, -1);
    EXPECT_TRUE(cmp.compare(body, exp));
}
