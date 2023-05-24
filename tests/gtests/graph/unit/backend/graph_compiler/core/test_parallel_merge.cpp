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

#include "context.hpp"
#include "test_utils.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/parallel_merge.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

static void make_set_idle(const expr &v) {
    builder::get_current_builder()->push_evaluate(
            make_expr<intrin_call_node>(intrin_type::set_thread_idle_func,
                    std::vector<expr> {get_ir_null(), v}, any_map_t {}));
}

static func_t get_func(
        const char *name, const char *A, const char *B, int threads) {
    _function_(datatypes::s32, aaa, _arg_(A, datatypes::f32, {100}),
            _arg_(B, datatypes::f32, {100})) {
        _bind_(A, B);
        _for_(i, UINT64_C(0), uint64_t(threads), UINT64_C(1),
                for_type::PARALLEL) {
            A[i] = B[i];
        }
        _return_(12);
    }
    aaa->name_ = name;
    aaa->decl_->name_ = name;
    return aaa;
}

static void set_no_barrier(const char *next_func) {
    builder::get_current_builder()
            ->get_current_scope()
            .body.back()
            ->attr()[attr_keys::no_post_barrier]
            = next_func;
}

TEST(GCCore_CPU_parallel_merge, TestMergeOK) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;

    auto aaa = get_func("aaa", "A", "B", 16);
    auto bbb = get_func("bbb", "C", "D", 16);
    auto ccc = get_func("ccc", "E", "F", 16);
    _function_(datatypes::void_t, mainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa, A, B);

        make_set_idle(A);
        _evaluate_call_(aaa, A, B);
        set_no_barrier("bbb");
        make_set_idle(A);
        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(bbb, C, B);
        set_no_barrier("ccc");

        make_set_idle(A);
        _evaluate_call_(ccc, A, B);

        make_set_idle(A);
        _evaluate_call_(bbb, A, B);
    }

    auto irm = ir_module_t::from_entry_func(get_test_ctx(), mainf);
    parallel_merge_t pass;
    auto retmod = pass(irm);

    _function_(datatypes::void_t, expected_merged,
            _arg_("A", datatypes::f32, {100}),
            _arg_("B", datatypes::f32, {100}),
            _arg_("C", datatypes::f32, {100}),
            _arg_("D", datatypes::f32, {100}),
            _arg_("E", datatypes::f32, {100}),
            _arg_("F", datatypes::f32, {100})) {
        _bind_(A, B, C, D, E, F);
        _for_(i, UINT64_C(0), uint64_t(16), UINT64_C(1), for_type::PARALLEL) {
            A[i] = B[i];
            C[i] = D[i];
            E[i] = F[i];
        }
    }

    auto mergedf = retmod->get_func("parallel__aaa__bbb__ccc");
    ASSERT_TRUE(mergedf);
    _function_(datatypes::void_t, expectedmainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa->decl_, A, B);
        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(mergedf->decl_, A, B, C, B, A, B);
        make_set_idle(A);
        _evaluate_call_(bbb->decl_, A, B);
    }

    auto new_mainf = retmod->get_func("mainf");
    ASSERT_TRUE(new_mainf);

    ir_comparer cmp {true};
    EXPECT_TRUE(cmp.compare(new_mainf, expectedmainf));
    EXPECT_TRUE(cmp.compare(mergedf, expected_merged));
}

TEST(GCCore_CPU_parallel_merge, TestMergeThreads) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;

    auto aaa1 = get_func("aaa1", "A", "B", 10);
    auto aaa2 = get_func("aaa2", "C", "D", 2);
    auto aaa3 = get_func("aaa3", "E", "F", 3);
    auto aaa4 = get_func("aaa4", "G", "H", 10);
    auto aaa5 = get_func("aaa5", "I", "J", 3);
    auto aaa6 = get_func("aaa6", "K", "L", 16);
    _function_(datatypes::void_t, mainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa1, A, B);
        set_no_barrier("aaa2");

        make_set_idle(A);
        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(aaa2, C, B);
        set_no_barrier("aaa3");

        _tensor_(D, datatypes::f32, 100);
        _evaluate_call_(aaa3, D, B);
        set_no_barrier("aaa4");

        _tensor_(E, datatypes::f32, 100);
        _evaluate_call_(aaa4, E, B);
        set_no_barrier("aaa5");

        _tensor_(F, datatypes::f32, 100);
        _evaluate_call_(aaa5, F, B);
        set_no_barrier("aaa6");

        _tensor_(G, datatypes::f32, 100);
        _evaluate_call_(aaa6, G, B);
        set_no_barrier("");
    }

    auto irm = ir_module_t::from_entry_func(get_test_ctx(), mainf);
    parallel_merge_t pass;
    auto retmod = pass(irm);

    _function_(datatypes::void_t, expected_merged,
            _arg_("K", datatypes::f32, {100}),
            _arg_("L", datatypes::f32, {100}),
            _arg_("A", datatypes::f32, {100}),
            _arg_("B", datatypes::f32, {100}),
            _arg_("E", datatypes::f32, {100}),
            _arg_("F", datatypes::f32, {100}),
            _arg_("I", datatypes::f32, {100}),
            _arg_("J", datatypes::f32, {100}),
            _arg_("G", datatypes::f32, {100}),
            _arg_("H", datatypes::f32, {100}),
            _arg_("C", datatypes::f32, {100}),
            _arg_("D", datatypes::f32, {100})) {
        _bind_(K, L, A, B, E, F, I, J, G, H, C, D);
        _for_(i, UINT64_C(0), uint64_t(16), UINT64_C(1), for_type::PARALLEL) {
            K[i] = L[i];
            _if_(i >= UINT64_C(0) && i < UINT64_C(10)) { A[i] = B[i]; }
            _if_(i >= UINT64_C(10) && i < UINT64_C(13)) {
                E[i - UINT64_C(10)] = F[i - UINT64_C(10)];
            }
            _if_(i >= UINT64_C(13) && i < UINT64_C(16)) {
                I[i - UINT64_C(13)] = J[i - UINT64_C(13)];
            }
            _if_(i >= UINT64_C(6) && i < UINT64_C(16)) {
                G[i - UINT64_C(6)] = H[i - UINT64_C(6)];
            }
            _if_(i >= UINT64_C(4) && i < UINT64_C(6)) {
                C[i - UINT64_C(4)] = D[i - UINT64_C(4)];
            }
        }
    }

    auto mergedf
            = retmod->get_func("parallel__aaa6__aaa1__aaa3__aaa5__aaa4__aaa2");
    ASSERT_TRUE(mergedf);

    ir_comparer cmp {true};
    EXPECT_TRUE(cmp.compare(mergedf, expected_merged));
}

TEST(GCCore_CPU_parallel_merge, TestMergeFailThreads) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;

    auto aaa1 = get_func("aaa1", "A", "B", 10);
    auto aaa2 = get_func("aaa2", "A", "B", 1000);
    _function_(datatypes::void_t, mainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa1, A, B);
        set_no_barrier("aaa2");

        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(aaa2, C, B);
        set_no_barrier("aaa2");

        _tensor_(D, datatypes::f32, 100);
        _evaluate_call_(aaa2, D, B);
    }
    auto irm = ir_module_t::from_entry_func(get_test_ctx(), mainf);
    parallel_merge_t pass;
    auto retmod = pass(irm);
    EXPECT_TRUE(retmod == irm);
}

TEST(GCCore_CPU_parallel_merge, TestMergeFailComplexBody) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;

    auto aaa1 = get_func("aaa1", "A", "B", 10);
    _function_(datatypes::s32, complexf, _arg_("A", datatypes::f32, {100}),
            _arg_("B", datatypes::f32, {100})) {
        _bind_(A, B);
        _var_(a, datatypes::f32);
        a = 10;
        _for_(i, UINT64_C(0), uint64_t(16), UINT64_C(1), for_type::PARALLEL) {
            A[i] = B[i];
        }
        _return_(12);
    }
    _function_(datatypes::void_t, mainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa1, A, B);
        set_no_barrier("complexf");

        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(complexf, C, B);
        set_no_barrier("aaa1");

        _tensor_(D, datatypes::f32, 100);
        _evaluate_call_(aaa1, D, B);
    }
    auto irm = ir_module_t::from_entry_func(get_test_ctx(), mainf);
    parallel_merge_t pass;
    auto retmod = pass(irm);
    EXPECT_TRUE(retmod == irm);
}

TEST(GCCore_CPU_parallel_merge, TestMergeFailComplexMain) {
    SET_THREADS_OR_SKIP(16);
    builder::ir_builder_t builder;

    auto aaa1 = get_func("aaa1", "A", "B", 10);
    auto aaa2 = get_func("aaa2", "A", "B", 10);
    _function_(datatypes::void_t, mainf) {
        _tensor_(A, datatypes::f32, 100);
        _tensor_(B, datatypes::f32, 100);

        make_set_idle(A);
        _evaluate_call_(aaa1, A, B);
        set_no_barrier("aaa2");
        _var_(a, datatypes::f32);
        a = 10;
        _tensor_(C, datatypes::f32, 100);
        _evaluate_call_(aaa2, C, B);
    }
    auto irm = ir_module_t::from_entry_func(get_test_ctx(), mainf);
    parallel_merge_t pass;
    auto retmod = pass(irm);
    EXPECT_TRUE(retmod == irm);
}
