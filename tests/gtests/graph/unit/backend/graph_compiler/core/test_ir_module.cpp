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
#include <compiler/ir/ir_module.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

TEST(GCCore_CPU_ir_module_cpp, TestIRModule) {
    builder::ir_builder_t builder;

    _function_(datatypes::f32, AAA) { _return_(1.0f); }

    _function_(datatypes::f32, BBB) { _return_(AAA()); }

    _function_(datatypes::f32, CCC) { _return_(AAA() + BBB()); }

    _function_(datatypes::f32, DDD) { _return_(BBB() + BBB()); }
    DDD->name_ = "CCC"; // make a nameing conflict

    _function_(datatypes::f32, entry) { _return_(CCC() + DDD()); }

    auto modu = ir_module_t::from_entry_func(get_default_context(), entry);

    _global_var_(modu, val1, datatypes::f32, expr());
    {
        auto newfunc = modu->get_contents();
        ASSERT_EQ(newfunc.size(), 5u);

        func_t newDDD = std::const_pointer_cast<func_base>(newfunc[2]);
        _function_(datatypes::f32, entry_expected) {
            _return_(CCC->decl_() + newDDD->decl_());
        }
        EXPECT_TRUE(newfunc[0]->equals(entry_expected));
        EXPECT_EQ(newfunc[1], CCC);
        EXPECT_EQ(newfunc[2], DDD);
        EXPECT_EQ(newfunc[3], AAA);
        EXPECT_EQ(newfunc[4], BBB);
        EXPECT_EQ(DDD->name_.find_first_of("CCC_"), 0u);
    }

    _function_(datatypes::f32, EEE) { _return_(1.0f); }
    auto modu2
            = ir_module_t::from_entry_func(get_default_context(), {AAA, EEE});
    _module_var_(modu2, val2, datatypes::f32, expr());
    // make a name conflict
    val2.get().as<var>()->name_ = "val1";
    {
        auto newfunc = modu2->get_contents();
        ASSERT_EQ(newfunc.size(), 2u);

        EXPECT_EQ(newfunc[0], AAA);
        EXPECT_EQ(newfunc[1], EEE);
    }
    modu->merge(*modu2);
    auto newfunc = modu->get_contents();
    ASSERT_EQ(newfunc.size(), 6u);
    EXPECT_EQ(newfunc[5], EEE);
    EXPECT_GT(val2.get().as<var>()->name_.size(), std::string("val1").size());
    EXPECT_EQ(val2.get().as<var>()->name_.find_first_of("val1"), 0u);
}
