/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include <compiler/ir/sc_stmt.hpp>
#include <compiler/ir/transform/dead_func_eliminate.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_dead_func_elimination, TestDFE) {
    builder::ir_builder_t builder;
    // indirectly referenced
    _function_(datatypes::void_t, eee) {}
    _function_(datatypes::void_t, aaa) { _evaluate_call_(eee); }
    aaa->body_.as<stmts>()->seq_.emplace_back(
            builder::make_evaluate_unattached(aaa()));
    // not referenced
    _function_(datatypes::void_t, bbb) {}
    _function_(datatypes::void_t, ccc) { _evaluate_call_(bbb); }
    // referenced by pointer
    _function_(datatypes::void_t, ddd) {}
    _function_(datatypes::void_t, mainf, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _evaluate_call_(aaa);
        _var_(a, datatypes::pointer);
        a = builder::make_func_addr(ddd);
    }

    aaa->attr()[function_attrs::private_] = true;
    bbb->attr()[function_attrs::private_] = true;
    ccc->attr()[function_attrs::private_] = true;
    ddd->attr()[function_attrs::private_] = true;
    eee->attr()[function_attrs::private_] = true;

    dead_func_eliminate_t dfe;
    auto out = dfe(ir_module_t::from_entry_func(get_default_context(), mainf));
    auto newmain = out->get_entry_func();
    ASSERT_EQ(newmain, mainf);
    ASSERT_EQ(out->get_contents().size(), 4UL);
    ASSERT_EQ(out->get_func("mainf"), mainf);
    ASSERT_EQ(out->get_func("aaa"), aaa);
    ASSERT_EQ(out->get_func("eee"), eee);
    ASSERT_EQ(out->get_func("ddd"), ddd);
}
