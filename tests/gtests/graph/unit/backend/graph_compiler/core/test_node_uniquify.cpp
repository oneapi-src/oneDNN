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
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/node_uniquify.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_set>

using namespace dnnl::impl::graph::gc;

class node_uniquify_checker_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    std::unordered_set<expr_c> exprs;
    std::unordered_set<stmt_c> stmts;
    expr_c dispatch(expr_c v) override {
        if (!v.isa<var>() && !v.isa<tensor>()) {
            EXPECT_TRUE(exprs.find(v) == exprs.end());
            if (exprs.find(v) != exprs.end()) { std::cout << v << "\n"; }
            exprs.insert(v);
            return ir_viewer_t::dispatch(v);
        }
        return v;
    }

    stmt_c dispatch(stmt_c v) override {
        EXPECT_TRUE(stmts.find(v) == stmts.end());
        stmts.insert(v);
        return ir_viewer_t::dispatch(v);
    }
};

TEST(GCCore_CPU_node_uniquify_cpp, TestNodeUniquify) {
    builder::ir_builder_t builder;
    expr temp;
    for_loop lo;
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321})) {
        _bind_(A);
        _tensor_(D, datatypes::f32, 10, 20); // stmt0
        _var_(d, datatypes::s32); // stmt1
        _named_for_(lo, i, 0, 100) { A[{i, 10}] = D[{1, i}]; } // stmt2
        // the loop is duplicated
        builder.emit(lo); // stmt3
        temp = d + 32;
        // expr temp is used twice
        A[{0, 10}] = temp; // stmt4
        _return_(temp); // stmt5
    }

    ir_comparer cmper(false, true);
    auto result = node_uniquifier_t()(aaa);
    node_uniquify_checker_t().dispatch(result);
}
