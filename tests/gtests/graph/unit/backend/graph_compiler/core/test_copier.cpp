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
#include <compiler/ir/pass/ir_copy.hpp>
#include <util/any_map.hpp>

#include <unordered_set>

#include <iostream>
#include <utility>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

namespace copier_test {
// Flattens the original IR tree into a set and checks if the copied IR nodes
// are in the old set
class copy_validator_t : public ir_viewer_t {
    std::unordered_set<expr_c> met_expr;
    std::unordered_set<stmt_c> met_stmt;

public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    copy_validator_t() = default;
    enum { FLATTENING = 0, CHECKING } state;
    void push(const expr_c &v) {
        if (state == CHECKING) {
            auto itr = met_expr.find(v);
            EXPECT_TRUE(itr == met_expr.end());
        } else {
            met_expr.insert(v);
        }
    }

    void push(const stmt_c &v) {
        if (state == CHECKING) {
            auto itr = met_stmt.find(v);
            EXPECT_TRUE(itr == met_stmt.end());
        } else {
            met_stmt.insert(v);
        }
    }

    void view(constant_c v) override { push(v); }
    void view(var_c v) override { push(v); }
    void view(cast_c v) override {
        push(v);
        dispatch(v->in_);
    }
    void view(binary_c v) override {
        push(v);
        dispatch(v->l_);
        dispatch(v->r_);
    }

    void view(cmp_c v) override {
        push(v);
        dispatch(v->l_);
        dispatch(v->r_);
    }

    void view(logic_c v) override {
        push(v);
        dispatch(v->l_);
        dispatch(v->r_);
    }

    void view(logic_not_c v) override {
        push(v);
        dispatch(v->in_);
    }

    void view(indexing_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(call_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(intrin_call_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(func_addr_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(ssa_phi_c v) override {
        push(v);
        ir_viewer_t::dispatch_expr_arr(v->values_);
    }
    void view(tensor_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }

    void view(tensorptr_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }

    void view(assign_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(stmts_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(if_else_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(evaluate_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(returns_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(define_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
    void view(for_loop_c v) override {
        push(v);
        ir_viewer_t::view(v);
    }
};

template <typename T>
void check_impl(T a) {
    ir_comparer cmper(true, true);
    std::unordered_map<expr_c, expr> map;
    ir_copier_t c(map);
    auto copied = c(a);
    std::unordered_set<expr_c> orig_flattened;
    copy_validator_t val;
    val.state = copy_validator_t::FLATTENING;
    val.dispatch(a);
    val.state = copy_validator_t::CHECKING;
    val.dispatch(copied);

    EXPECT_TRUE(cmper.compare(a, copied));
}

void check(expr v) {
    check_impl(std::move(v));
}
void check(stmt v) {
    check_impl(std::move(v));
}
void check(func_t v) {
    check_impl(std::move(v));
}

} // namespace copier_test

using namespace copier_test;

TEST(GCCore_CPU_copier_cpp, TestCopierExpr) {
    expr a = 1;
    check(a);
    check(make_cast(datatypes::f32, 1));
    check(make_min(2, 1));
    check(make_max(2, 1));
    check(make_add(2, 1));
    check(make_sub(2, 1));
    check(make_mul(2, 1));
    check(make_div(2, 1));
    check(make_mod(2, 1));

    check(make_cmp_eq(2, 1));
    check(make_cmp_lt(2, 1));
    check(make_cmp_le(2, 1));
    check(make_cmp_gt(2, 1));
    check(make_cmp_ge(2, 1));
    check(make_cmp_ne(2, 1));
    check(make_logic_and(true, false));
    check(make_logic_or(true, false));
    check(make_logic_not(false));
    check(make_tensor("a", {100}, datatypes::f32, address_space::device)[10]);
    check(make_tensor("a", {100}, datatypes::f32)[10]);
    auto tmp_tsr = make_tensor("a", {100}, datatypes::f32);
    int a_val = 123;
    tmp_tsr.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(&a_val, sizeof(int));
    check(tmp_tsr);
    check(tensor_ptr(make_tensor("a", {100}, datatypes::f32), {10}));
    _decl_func_(datatypes::f32, aaa, _arg_("v1", datatypes::f32));
    check(aaa(1.23f));
    check(make_func_addr(aaa));
    check(make_expr<ssa_phi_node>(std::vector<expr> {1, 2, 3}, false));

    intrin_call intrinnode = make_expr<intrin_call_node>(
            intrin_type::min, std::vector<expr> {1, 2}, any_map_t());
    check(intrinnode);

    std::unordered_map<expr_c, expr> map;
    ir_copier_t c(map);
    expr callnode = aaa(1.23f);
    callnode->attr()["someattr"] = 123;
    EXPECT_EQ(c(callnode)->attr_->get<int>("someattr"), 123);

    intrinnode = builder::make_reinterpret(1, datatypes::f32)
                         .checked_as<intrin_call>();
    auto result = c(intrinnode);
    EXPECT_EQ(result.checked_as<intrin_call>()->intrin_attrs_->get_or_else(
                      intrin_attr::out_dtype, datatypes::undef),
            datatypes::f32);
    EXPECT_EQ(result->dtype_, datatypes::f32);
}

TEST(GCCore_CPU_copier_cpp, TestCopierStmt) {
    ir_builder_t bld;
    bld.push_scope();
    _var_(a, datatypes::f32);
    a = 1.23f;
    check(bld.get_current_scope().body.back());

    _if_(a == 1.23f) { a = 1.23f; }
    _else_ { a = 123.0f; }
    check(bld.get_current_scope().body.back());

    _if_(a == 1.23f) { a = 1.23f; }
    check(bld.get_current_scope().body.back());

    _return_();
    check(bld.get_current_scope().body.back());

    _return_(1.2345f);
    check(bld.get_current_scope().body.back());

    _for_(i, 0, 10, 20, for_type::PARALLEL) { a = i; }
    check(bld.get_current_scope().body.back());

    _var_(va, datatypes::boolean);
    check(bld.get_current_scope().body.back());

    _var_ex_(vb, datatypes::boolean, linkage::local, expr(true));
    check(bld.get_current_scope().body.back());
}

TEST(GCCore_CPU_copier_cpp, TestCopierVar) {
    auto v = builder::make_var(datatypes::f32, "abc");
    auto vmapped = builder::make_var(datatypes::f32, "vvv");
    check(v);
    std::unordered_map<expr_c, expr> map = {{v, vmapped}};
    ir_copier_t c(map);
    expr_c copied = c(v + 1.23f);
    EXPECT_TRUE(copied.checked_as<add>()->l_.ptr_same(vmapped));
}

TEST(GCCore_CPU_copier_cpp, TestCopierTensor) {
    auto v = builder::make_tensor("abc", {1, 2, 3}, datatypes::f32);
    auto vmapped = builder::make_tensor("ccc", {1, 2, 3}, datatypes::f32);
    check(v);
    std::unordered_map<expr_c, expr> map = {{v, vmapped}};
    ir_copier_t c(map);
    expr_c copied = c(v[10].get());
    EXPECT_TRUE(copied.checked_as<indexing>()->ptr_.ptr_same(vmapped));
}

TEST(GCCore_CPU_copier_cpp, TestCopierFunc) {
    ir_builder_t bld;
    _function_(datatypes::f32, AAA, _arg_("a", datatypes::f32),
            _arg_("A", datatypes::f32, {100, 100})) {
        _bind_(a, A);
        _for_(i, 0, 100) {
            _for_(j, 0, 100) { A[{i, j}] = a; }
        }
    }
    check(AAA);
}
