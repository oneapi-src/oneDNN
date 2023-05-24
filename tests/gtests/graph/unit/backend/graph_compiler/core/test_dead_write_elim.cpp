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
#include <compiler/ir/pass/dependency_analyzer.hpp>
#include <compiler/ir/sc_stmt.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/viewer.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using dependency_analysis::stmt_weak_set;
namespace ut {
using namespace dnnl::impl::graph::gc;
class dep_printer_t : public ir_viewer_t {
    void view(assign_c v) override {
        std::cout << "IR=" << v << "\ndepended by:\n";
        auto &dep = dependency_analysis::get_dep_info(v.get());
        for (auto v : dep.depended_by_) {
            auto ptr = v.lock();
            assert(ptr);
            std::cout << '\t' << ptr.get() << '\n';
        }
        std::cout << "depending on:\n";
        for (auto v : dep.depends_on_) {
            auto ptr = v.lock();
            assert(ptr);
            std::cout << '\t' << ptr.get() << '\n';
        }
    }
};

} // namespace ut

static std::shared_ptr<stmt_base_t> get_builder_top() {
    return builder::get_current_builder()->get_current_scope().body.back().impl;
}

static void check_dep(const std::shared_ptr<stmt_base_t> &ptr,
        const stmt_weak_set &depended_by, const stmt_weak_set &depends_on) {
    auto &dep = dependency_analysis::get_dep_info(ptr.get());
    bool result = dep.depended_by_ == depended_by;
    if (!result) { std::cout << "depended_by failed:" << ptr.get() << "\n"; }
    EXPECT_TRUE(result);
    result = dep.depends_on_ == depends_on;
    if (!result) { std::cout << "depends_on failed:" << ptr.get() << "\n"; }
    EXPECT_TRUE(result);
}

#define SAVE(v) v = get_builder_top();
TEST(GCCore_CPU_dependency_analyzer, TestDependency) {
    builder::ir_builder_t builder;
    std::shared_ptr<stmt_base_t> b_eq_3, c_eq_3, c_eq_4, c_eq_5, bc_eq_ab,
            bc_eq_2, c_eq_bb, ab_eq_3, if_else_s, c_eq_ab, t_eq_ai, ai_eq_t,
            loop_i;
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10000}),
            _arg_("b", datatypes::s32)) {
        _bind_(A, b);
        _tensor_(B, datatypes::f32, 100);
        _tensor_(D, datatypes::f32, 100);
        b = 3;
        SAVE(b_eq_3);
        _var_(c, datatypes::s32);
        c = 3;
        SAVE(c_eq_3);
        _if_(b == 2) {
            c = 4;
            SAVE(c_eq_4);
        }
        _else_ {
            c = 5;
            SAVE(c_eq_5);
        }
        SAVE(if_else_s);
        B[c] = A[b];
        SAVE(bc_eq_ab);

        _for_(i, 0, 10) {
            B[c] = 2;
            SAVE(bc_eq_2);
            c = B[b];
            SAVE(c_eq_bb);
            A[b] = 3;
            SAVE(ab_eq_3);
            c = A[b];
            SAVE(c_eq_ab);
        }
        _for_(i, 0, 10) {
            _var_(t, datatypes::s32);
            t = D[i];
            SAVE(t_eq_ai);
            D[i] = t + 1;
            SAVE(ai_eq_t);
        }
        SAVE(loop_i);
    }
    dependency_analyzer_t ana;
    ana(ccc);
    // ut::dep_printer_t p;
    // p.dispatch(ccc);

    check_dep(b_eq_3, {c_eq_ab, if_else_s, ab_eq_3, c_eq_bb, bc_eq_ab}, {});
    check_dep(c_eq_3, {c_eq_bb, bc_eq_2, bc_eq_ab, c_eq_4, c_eq_5}, {});
    check_dep(c_eq_4, {c_eq_bb, bc_eq_2, bc_eq_ab}, {c_eq_3});
    check_dep(c_eq_5, {c_eq_bb, bc_eq_2, bc_eq_ab}, {c_eq_3});
    check_dep(bc_eq_ab, {c_eq_ab, c_eq_bb, bc_eq_2, ab_eq_3},
            {b_eq_3, c_eq_5, c_eq_3, c_eq_4});
    check_dep(bc_eq_2, {c_eq_bb}, {c_eq_bb, bc_eq_ab, c_eq_5, c_eq_3, c_eq_4});
    check_dep(c_eq_bb, {c_eq_ab, bc_eq_2},
            {bc_eq_2, bc_eq_ab, c_eq_5, c_eq_3, c_eq_4, b_eq_3});
    check_dep(ab_eq_3, {c_eq_ab}, {b_eq_3, bc_eq_ab, c_eq_ab});
    check_dep(c_eq_ab, {ab_eq_3}, {ab_eq_3, bc_eq_ab, c_eq_bb, b_eq_3});
    check_dep(t_eq_ai, {ai_eq_t}, {loop_i});
    check_dep(ai_eq_t, {}, {loop_i, t_eq_ai});
}

TEST(GCCore_CPU_dead_write_elimination, TestDWE) {
    builder::ir_builder_t builder;
    auto mod = std::make_shared<ir_module_t>(get_default_context());
    _global_tensor_(mod, G, datatypes::f32, 100);
    _function_(datatypes::f32, ccc, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _tensor_(B, datatypes::f32, 100);
        _tensor_(C, datatypes::f32, 100);
        _tensor_(D, datatypes::f32, 100);
        _tensor_(E, datatypes::f32, 100);

        // constant index, required by the code in the loop
        C[1] = 1.0f;
        _for_(i, 0, 100) {
            _var_(t, datatypes::f32);
            _var_(t2, datatypes::f32);
            _for_(j, 0, 100) {
                t = D[j];
                D[j] = t + 1;
            }
            t2 = B[i];
            t = t + A[i] + G[i];
            t = t * t2;
            A[i] = C[1];
            G[i] = t;
            B[i] = t;
            // constant index, dead write
            C[0] = 1.0f;
            // constant index, required by the code after the loop
            C[2] = 1.0f;
        }
        _for_(i, 0, 100) {
            _var_(t, datatypes::f32);
            t = E[i];
            t = t + 1;
            E[i] = t;
        }
        _return_(C[2]);
    }
    dead_write_eliminator_t dwe;
    auto out = dwe(ccc);
    _function_(datatypes::f32, expected, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _tensor_(B, datatypes::f32, 100);
        _tensor_(C, datatypes::f32, 100);
        _tensor_(D, datatypes::f32, 100);
        _tensor_(E, datatypes::f32, 100);
        C[1] = 1.0f;

        _for_(i, 0, 100) {
            _var_(t, datatypes::f32);
            _var_(t2, datatypes::f32);
            _for_(j, 0, 100) {
                t = D[j];
                D[j] = t + 1;
            }
            t2 = B[i];
            t = t + A[i] + G[i];
            t = t * t2;
            A[i] = C[1];
            G[i] = t;
            builder.push_scope();
            builder.emit(builder.pop_scope());
            builder.push_scope();
            builder.emit(builder.pop_scope());
            C[2] = 1.0f;
        }
        _for_(i, 0, 100) {
            _var_(t, datatypes::f32);
            t = E[i];
            t = t + 1; // successful elimination
            builder.push_scope();
            builder.emit(builder.pop_scope());
        }
        _return_(C[2]);
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_dead_write_elimination, TestDWELoopIndependent) {
    builder::ir_builder_t builder;
    _function_(datatypes::f32, ccc, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _tensor_(B, datatypes::f32, 100);
        _var_(tid, datatypes::f32);
        tid = 122;
        B[tid] = 1.0f;
        _for_(i, 0, 100) {
            _var_(t, datatypes::f32);
            t = B[tid];
            _if_(t == 1.0f) {
                _var_(t2, datatypes::f32);
                t2 = 2.0f;
                B[tid] = t2;
            }
        }
        _return_(0);
    }
    dead_write_eliminator_t dwe;
    auto out = dwe(ccc);
    EXPECT_TRUE(out == ccc);
}
