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

#include <array>
#include "context.hpp"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/dynamic_parallel_transform.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_threadpool_c.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
static constexpr auto s32 = datatypes::s32;

#define U64 UINT64_C

TEST(GCCore_CPU_dyn_parallel_transform, TestDynamicParallelTransform) {
    SET_THREADS_OR_SKIP(32);
    builder::ir_builder_t builder;
    ir_module_ptr mod = std::make_shared<ir_module_t>(get_test_ctx());
    expr G = mod->make_global_tensor(s32, "G", {1000});
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _tensor_(B, s32, 100);
        _for_(i, UINT64_C(0), UINT64_C(100), UINT64_C(1), for_type::PARALLEL,
                2) {
            _for_(j, UINT64_C(0), UINT64_C(40), UINT64_C(1), for_type::PARALLEL,
                    4) {
                A[j] = i + j;
                // captured
                _tensor_(C, s32, 100);
                C[0] = 10;
                // no need to capture
                _tensor_(D, s32, 100);
                D[0] = 10;
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    B[k] = i + j + k;
                }
                C[i + j] = 9;
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    C[k] = i + j + 1;
                }
            }
            G[i] = i;
        }
        _return_(0);
    }

    mod->add_func({ccc});
    dynamic_parallel_transform_t pass {true};
    auto mod2 = pass(mod);

    uint64_t num_threads = runtime_config_t::get().get_num_threads();
    static const std::array<int, 6> closure_names_id {1, 2, 3, 4, 5, 7};
    std::vector<func_t> wrappers, bodies;
    wrappers.reserve(closure_names_id.size());
    bodies.reserve(closure_names_id.size());
    for (auto v : closure_names_id) {
        auto f = mod2->get_func("ccc_0_closure_N" + std::to_string(v));
        ASSERT_TRUE(f);
        bodies.emplace_back(f);
        f = mod2->get_func(f->name_ + "_wrapper");
        ASSERT_TRUE(f);
        wrappers.emplace_back(f);
    }
    using namespace dnnl::impl::graph::gc::runtime::dynamic_threadpool;
    _function_(s32, ccc_expected, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _tensor_(B, s32, 100);
        builder.push_scope();
        {
            _tensor_(captures, datatypes::generic, U64(2));
            captures[U64(0)] = builder::make_cast(datatypes::generic, A);
            captures[U64(1)] = builder::make_cast(datatypes::generic, B);
            _evaluate_call_(get_dyn_threadpool_init_func(), get_ir_null(),
                    get_ir_null(), captures, /*num_roots*/ U64(1),
                    /*queue_size*/ U64(256), num_threads);
            _evaluate_call_(get_dyn_threadpool_submit_func(),
                    builder::make_func_addr(wrappers[0]->decl_), get_ir_null(),
                    /*num_iter*/ U64(0), /*loop_len*/ U64(100),
                    /*num_blocks*/ U64(2), /*outer_loop_hash*/ U64(0),
                    /*num_buffers*/ U64(0),
                    /*buffers*/ get_ir_null(), /*flags*/
                    uint64_t(work_item_flags::is_root
                            | work_item_flags::bind_last_level | 16));
            _evaluate_call_(get_dyn_threadpool_run_func());
            _evaluate_call_(get_dyn_threadpool_destroy_func());
        }
        auto scope = builder.pop_scope();
        builder.get_current_scope().emit(scope);
        _return_(0);
    }
    ir_comparer cmper {true};
    auto new_main = mod2->get_func("ccc");
    ASSERT_TRUE(new_main);
    EXPECT_TRUE(cmper.compare(new_main, ccc_expected, true));

    _function_(datatypes::void_t, closure1, _arg_("i", datatypes::index),
            _arg_("A", s32, {10000}), _arg_("B", s32, {100})) {
        _bind_(i, A, B);
        _evaluate_call_(get_dyn_threadpool_submit_func(),
                builder::make_func_addr(wrappers[1]->decl_), get_ir_null(),
                /*num_iter*/ U64(1), /*loop_len*/ U64(40),
                /*num_blocks*/ U64(4), /*outer_loop_hash*/ U64(0),
                /*num_buffers*/ U64(0),
                /*buffers*/ get_ir_null(),
                /*flags*/ uint64_t(work_item_flags::bind_last_level | 4));
    }
    ASSERT_TRUE(cmper.compare(bodies[0], closure1, true));

    _function_(datatypes::void_t, closure2, _arg_("i", datatypes::index),
            _arg_("j", datatypes::index), _arg_("A", s32, {10000}),
            _arg_("B", s32, {100})) {
        _bind_(i, j, A, B);
        A[j] = i + j;
        // captured
        _tensor_(C, s32, 100);
        builder.get_current_scope().body->seq_.back().static_as<define>()->init_
                = get_dyn_threadpool_shared_buffer_func()(U64(400));
        C[0] = 10;
        // no need to capture
        _tensor_(D, s32, 100);
        D[0] = 10;
        _tensor_(shared, datatypes::generic, U64(1));
        shared[U64(0)] = builder::make_cast(datatypes::generic, C);
        _evaluate_call_(get_dyn_threadpool_submit_func(),
                builder::make_func_addr(wrappers[2]->decl_), get_ir_null(),
                /*num_iter*/ U64(2), /*loop_len*/ U64(16),
                /*num_blocks*/ U64(4), /*outer_loop_hash*/ U64(0),
                /*num_buffers*/ U64(1),
                /*buffers*/ builder::make_cast(datatypes::pointer, shared),
                /*flags*/ uint64_t(work_item_flags::bind_last_level | 1));
    }
    EXPECT_TRUE(cmper.compare(bodies[1], closure2, true));

    _function_(datatypes::void_t, closure3, _arg_("i", datatypes::index),
            _arg_("j", datatypes::index), _arg_("k", datatypes::index),
            _arg_("C", s32, {100}), _arg_("A", s32, {10000}),
            _arg_("B", s32, {100})) {
        _bind_(i, j, k, C, A, B);
        B[k] = i + j + k;
        _var_init_(handle, datatypes::index,
                get_dyn_threadpool_loop_end_func()(U64(0), U64(0)));
        _if_(handle != U64(0)) {
            //     _tensor_(shared, datatypes::generic, U64(1));
            //     shared[U64(0)] = builder::make_cast(datatypes::generic, C);
            _evaluate_call_(get_dyn_threadpool_submit_func(),
                    builder::make_func_addr(wrappers[3]->decl_), get_ir_null(),
                    /*num_iter*/ U64(2), /*loop_len*/ U64(1),
                    /*num_blocks*/ U64(1), /*outer_loop_hash*/ U64(0),
                    /*num_buffers*/ U64(1),
                    /*buffers*/ get_ir_null(),
                    /*flags*/ uint64_t(work_item_flags::bind_last_level | 4));
        }
    }
    EXPECT_TRUE(cmper.compare(bodies[2], closure3, true));

    _function_(datatypes::void_t, closure4, _arg_("i", datatypes::index),
            _arg_("j", datatypes::index), _arg_("C", s32, {100}),
            _arg_("A", s32, {10000}), _arg_("B", s32, {100})) {
        _bind_(i, j, C, A, B);
        C[i + j] = 9;
        // _tensor_(shared, datatypes::generic, U64(1));
        // shared[U64(0)] = builder::make_cast(datatypes::generic, C);
        _evaluate_call_(get_dyn_threadpool_submit_func(),
                builder::make_func_addr(wrappers[4]->decl_), get_ir_null(),
                /*num_iter*/ U64(2), /*loop_len*/ U64(16),
                /*num_blocks*/ U64(4), /*outer_loop_hash*/ U64(0),
                /*num_buffers*/ U64(1),
                /*buffers*/ get_ir_null(),
                /*flags*/ uint64_t(work_item_flags::bind_last_level | 1));
    }
    EXPECT_TRUE(cmper.compare(bodies[3], closure4, true));

    _function_(datatypes::void_t, closure5, _arg_("i", datatypes::index),
            _arg_("j", datatypes::index), _arg_("k", datatypes::index),
            _arg_("C", s32, {100}), _arg_("A", s32, {10000}),
            _arg_("B", s32, {100})) {
        _bind_(i, j, k, C, A, B);
        C[k] = i + j + 1;
        _var_init_(handle, datatypes::index,
                get_dyn_threadpool_loop_end_func()(U64(0), U64(0)));
        _if_(handle != U64(0)) {
            _var_init_(handle2, datatypes::index,
                    get_dyn_threadpool_loop_end_func()(handle, U64(3)));
            _if_(handle2 != U64(0)) {
                _evaluate_call_(get_dyn_threadpool_submit_func(),
                        builder::make_func_addr(wrappers[5]->decl_),
                        get_ir_null(),
                        /*num_iter*/ U64(1), /*loop_len*/ U64(1),
                        /*num_blocks*/ U64(1), /*outer_loop_hash*/ U64(0),
                        /*num_buffers*/ U64(0),
                        /*buffers*/ get_ir_null(),
                        /*flags*/
                        uint64_t(work_item_flags::bind_last_level | 16));
            }
        }
    }
    EXPECT_TRUE(cmper.compare(bodies[4], closure5, true));

    _function_(datatypes::void_t, closure7, _arg_("i", datatypes::index),
            _arg_("A", s32, {10000}), _arg_("B", s32, {100})) {
        _bind_(i, A, B);
        G[i] = i;
    }
    EXPECT_TRUE(cmper.compare(bodies[5], closure7, true));

    // compare the wrappers
    _function_(datatypes::void_t, closure1_wrapper,
            _arg_("itr", datatypes::index, {1UL}),
            _arg_("buffers", datatypes::generic, {0UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[0]->decl_, itr[U64(0)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[0], closure1_wrapper, true));

    _function_(datatypes::void_t, closure2_wrapper,
            _arg_("itr", datatypes::index, {2UL}),
            _arg_("buffers", datatypes::generic, {0UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[1]->decl_, itr[U64(0)], itr[U64(1)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[1], closure2_wrapper, true));

    _function_(datatypes::void_t, closure3_wrapper,
            _arg_("itr", datatypes::index, {3UL}),
            _arg_("buffers", datatypes::generic, {1UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[2]->decl_, itr[U64(0)], itr[U64(1)], itr[U64(2)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), buffers[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[2], closure3_wrapper, true));

    _function_(datatypes::void_t, closure4_wrapper,
            _arg_("itr", datatypes::index, {2UL}),
            _arg_("buffers", datatypes::generic, {1UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[3]->decl_, itr[U64(0)], itr[U64(1)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), buffers[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[3], closure4_wrapper, true));

    _function_(datatypes::void_t, closure5_wrapper,
            _arg_("itr", datatypes::index, {3UL}),
            _arg_("buffers", datatypes::generic, {1UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[4]->decl_, itr[U64(0)], itr[U64(1)], itr[U64(2)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), buffers[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[4], closure5_wrapper, true));

    _function_(datatypes::void_t, closure7_wrapper,
            _arg_("itr", datatypes::index, {1UL}),
            _arg_("buffers", datatypes::generic, {0UL}),
            _arg_("args", datatypes::generic, {2UL})) {
        _bind_(itr, buffers, args);
        _evaluate_call_(bodies[5]->decl_, itr[U64(0)],
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(0)]),
                builder::make_cast(
                        datatypes::s32.get_pointerof(), args[U64(1)]));
    }
    EXPECT_TRUE(cmper.compare(wrappers[5], closure7_wrapper, true));
}

TEST(GCCore_CPU_dyn_parallel_transform, TestInlineAndBufferPassing) {
    using namespace dnnl::impl::graph::gc::runtime::dynamic_threadpool;
    SET_THREADS_OR_SKIP(32);
    builder::ir_builder_t builder;
    ir_module_ptr mod = std::make_shared<ir_module_t>(get_test_ctx());
    expr G = mod->make_global_tensor(s32, "G", {1000});
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _tensor_(B, s32, 100);
        _for_(j, UINT64_C(0), UINT64_C(40), UINT64_C(1), for_type::PARALLEL,
                4) {
            // captured
            _tensor_(C, s32, 100);
            C[0] = 10;
            _tensor_(D, s32, 100);
            _for_(i, UINT64_C(0), UINT64_C(40), UINT64_C(1), for_type::PARALLEL,
                    2) {
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    C[k] = k;
                    D[k] = k;
                }
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    D[k] = 9;
                }
            }
            C[0] = 1;
        }
        _return_(0);
    }

    mod->add_func({ccc});
    dynamic_parallel_transform_t pass {true};
    auto mod2 = pass(mod);

    auto result3 = mod2->get_func("ccc_0_closure_N3");
    auto result4 = mod2->get_func("ccc_0_closure_N4");
    auto result5 = mod2->get_func("ccc_0_closure_N5");
    auto wrapper7 = mod2->get_func("ccc_0_closure_N7_wrapper");
    ASSERT_TRUE(result3);
    ASSERT_TRUE(result4);
    ASSERT_TRUE(result5);
    ASSERT_TRUE(wrapper7);
    _function_(datatypes::void_t, closure3, _arg_("j", datatypes::index),
            _arg_("i", datatypes::index), _arg_("k", datatypes::index),
            _arg_("C", s32, {100}), _arg_("D", s32, {100})) {
        _bind_(j, i, k, C, D);
        C[k] = k;
        D[k] = k;
        _var_init_(handle, datatypes::index,
                get_dyn_threadpool_loop_end_func()(U64(0), U64(0)));
        // check that closure4 is inlined (instead of another job)
        _if_(handle != U64(0)) { _evaluate_call_(result4->decl_, j, i, D, C); }
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(result3, closure3, true));

    _function_(datatypes::void_t, closure5, _arg_("j", datatypes::index),
            _arg_("i", datatypes::index), _arg_("k", datatypes::index),
            _arg_("D", s32, {100}), _arg_("C", s32, {100})) {
        _bind_(j, i, k, D, C);
        D[k] = k;
        _var_init_(handle, datatypes::index,
                get_dyn_threadpool_loop_end_func()(U64(0), U64(0)));
        _if_(handle != U64(0)) {
            _var_init_(handle2, datatypes::index,
                    get_dyn_threadpool_loop_end_func()(handle, U64(3)));
            _if_(handle2 != U64(0)) {
                _tensor_(shared, datatypes::generic, U64(1));
                shared[U64(0)] = builder::make_cast(datatypes::generic, C);
                _evaluate_call_(get_dyn_threadpool_submit_func(),
                        builder::make_func_addr(wrapper7->decl_), get_ir_null(),
                        /*num_iter*/ U64(1), /*loop_len*/ U64(1),
                        /*num_blocks*/ U64(1), /*outer_loop_hash*/ U64(0),
                        /*num_buffers*/ U64(0),
                        /*buffers*/
                        builder::make_cast(datatypes::pointer, shared),
                        /*flags*/
                        uint64_t(work_item_flags::bind_last_level | 8));
            }
        }
    }
}

struct tp_reseter {
    thread_pool_mode_t old_;
    tp_reseter() {
        old_ = runtime_config_t::get().managed_thread_pool_;
        runtime_config_t::get().managed_thread_pool_
                = thread_pool_mode_t::DYNAMIC;
    }
    ~tp_reseter() { runtime_config_t::get().managed_thread_pool_ = old_; }
};

TEST(GCCore_CPU_dyn_parallel_transform, TestExecute) {
    REQUIRE_AVX2();
    SET_THREADS_OR_SKIP(32);
    tp_reseter reseter;
    builder::ir_builder_t builder;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());

    ir_module_ptr mod = std::make_shared<ir_module_t>(ctx);
    expr G = mod->make_global_tensor(s32, "G", {16});
    _function_(s32, ccc, _arg_("A", s32, {10, 40}), _arg_("a", s32)) {
        _bind_(A, a);
        _tensor_(B, s32, 10, 40, 16);
        _for_(i, UINT64_C(0), UINT64_C(10), UINT64_C(1), for_type::PARALLEL,
                2) {
            _for_(j, UINT64_C(0), UINT64_C(40), UINT64_C(1), for_type::PARALLEL,
                    4) {
                A[{i, j}] = builder::make_cast(s32, i + j);
                // captured
                _tensor_(C, s32, 16);
                // no need to capture
                _tensor_(D, s32, 16);
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    C[k] = builder::make_select(
                            A[{i, j}] == i + j, builder::make_cast(s32, k), -1);
                }
                // non-parallel, check the C buffer
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1)) {
                    _if_(C[k] == k) { A[{i, j}] = A[{i, j}] + 10; }
                }
                A[{i, j}] = A[{i, j}] - 3;
                _for_(k, UINT64_C(0), UINT64_C(16), UINT64_C(1),
                        for_type::PARALLEL, 4) {
                    B[{i, j, k}]
                            = builder::make_select(A[{i, j}] == i + j + 160 - 3,
                                    builder::make_cast(s32, k), -2);
                }
            }
            G[i] = builder::make_cast(s32, A[{i, 0}]) + a;
            _for_(j, UINT64_C(0), UINT64_C(40)) {
                _for_(k, UINT64_C(0), UINT64_C(16)) {
                    _if_(B[{i, j, k}] != k) { G[i] = -100; }
                }
            }
        }
        _return_(0);
    }

    mod->add_func({ccc});
    auto jitm = jit_engine_t::make(ctx)->make_jit_module(mod, true);
    auto jitf = jitm->get_function("ccc");
    auto buf = alloc_array<int32_t>(10 * 40);
    generic_val args[] = {buf.data(), 17};
    jitf->call_generic_default(args);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 40; j++) {
            ASSERT_EQ(buf[i * 40 + j], i + j + 160 - 3);
        }
    }
    auto global_tsr = (int32_t *)jitm->get_address_of_symbol("G");
    ASSERT_TRUE(global_tsr);
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(global_tsr[i], i + 160 - 3 + 17);
    }
}
