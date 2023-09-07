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

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "gtest/gtest.h"
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <runtime/managed_thread_pool.hpp>
#include <runtime/managed_thread_pool_exports.hpp>
#include <runtime/parallel.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <test_thread.hpp>
#define dnnl_thread_env() \
    dnnl::testing::scoped_tp_activation_t unused_raii {}
#else
#define dnnl_thread_env()
#endif

using namespace dnnl::impl::graph::gc;

#if SC_CPU_THREADPOOL > 0

TEST(GCCore_CPU_thread_pool, TestBarrier) {
    dnnl_thread_env();
    runtime::barrier_t bar[2];
    sc_init_barrier(bar, 2, 16);
    int data[16 * 16] = {0};
    std::vector<std::thread> threads;
    bool result = true;
    for (int t = 0; t < 16; t++) {
        threads.emplace_back(
                [&data, &bar, &result](int tid) {
                    for (int i = 0; i < 500; i++) {
                        data[tid * 16] = i;
                        sc_arrive_at_barrier(bar, nullptr, nullptr);
                        // check if all of data has been updated
                        for (int j = 0; j < 16; j++) {
                            if (data[j * 16] != i) result = false;
                        }
                        sc_arrive_at_barrier(bar, nullptr, nullptr);
                    }

                    for (int i = 0; i < 100; i++) {
                        data[tid * 16] = i;
                        sc_arrive_at_barrier(bar + 1, nullptr, nullptr);
                        for (int j = 0; j < 16; j++) {
                            if (data[j * 16] != i) result = false;
                        }
                        sc_arrive_at_barrier(bar + 1, nullptr, nullptr);
                    }
                },
                t);
    }
    for (int t = 0; t < 16; t++) {
        threads[t].join();
    }
    EXPECT_TRUE(result);
}

TEST(GCCore_CPU_thread_pool, TestThreadPool) {
    dnnl_thread_env();
    auto &cfg = runtime_config_t::get();
    std::vector<std::atomic<int>> v(100000);
    std::vector<int> counts(
            runtime_config_t::get().thread_pool_table_->get_num_threads());
    struct env_t {
        std::vector<std::atomic<int>> &v;
        std::vector<int> &counts;
    } env {v, counts};
    auto funct = [](runtime::stream_t *s, void *mod_data,
                         generic_val *args) noexcept {
        env_t *penv = (env_t *)mod_data;
        auto pcall = runtime_config_t::get().thread_pool_table_->parallel_call;
        if (runtime_config_t::get().managed_thread_pool_) {
            pcall = runtime_config_t::get()
                            .thread_pool_table_->parallel_call_managed;
        }
        pcall(
                [](void *a, void *b, int64_t idx, generic_val *args) {
                    std::vector<std::atomic<int>> &v
                            = *(std::vector<std::atomic<int>> *)b;

                    if (idx % 2 != 0) throw std::runtime_error("Bad index");
                    v.at(idx - 2)++;
                    if (args) throw std::runtime_error("Bad arg");
                    std::vector<int> &counts = *(std::vector<int> *)a;
                    counts.at(runtime_config_t::get()
                                      .thread_pool_table_->get_thread_id())++;
                },
                0, (void *)&penv->counts, (void *)&penv->v, 2, 100002, 2,
                nullptr);
    };

    if (cfg.managed_thread_pool_) {
        runtime::thread_manager::cur_mgr.run_main_function(
                funct, nullptr, &env, nullptr);
    } else {
        funct(nullptr, &env, nullptr);
    }
    // int actual_num_threads = 0;
    // for (auto &v : counts) {
    //     if (v > 0) { actual_num_threads++; }
    // }
    // EXPECT_GT(actual_num_threads, 3);

    for (size_t i = 0; i < v.size(); i++) {
        if (i % 2 == 0) {
            ASSERT_EQ(v[i].load(), 1);
        } else {
            ASSERT_EQ(v[i].load(), 0);
        }
    }
    int old_num_threads = cfg.thread_pool_table_->get_num_threads();
    cfg.thread_pool_table_->set_num_threads(12);
    EXPECT_EQ(cfg.thread_pool_table_->get_num_threads(), 12);
    cfg.thread_pool_table_->set_num_threads(old_num_threads);
    EXPECT_EQ(cfg.thread_pool_table_->get_num_threads(), old_num_threads);
}
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
TEST(GCCore_CPU_thread_pool, TestThreadNum) {
    dnnl_thread_env();
    auto &cfg = runtime_config_t::get();
    int nthreads
            = runtime_config_t::get().thread_pool_table_->get_num_threads();
    std::vector<int> counts(nthreads);

    auto funct = [](runtime::stream_t *s, void *mod_data,
                         generic_val *args) noexcept {
        std::vector<int> *penv = (std::vector<int> *)mod_data;
        int nthreads
                = runtime_config_t::get().thread_pool_table_->get_num_threads();
        auto pcall = runtime_config_t::get().thread_pool_table_->parallel_call;
        if (runtime_config_t::get().managed_thread_pool_) {
            pcall = runtime_config_t::get()
                            .thread_pool_table_->parallel_call_managed;
        }
        pcall(
                [](void *a, void *mod_data, int64_t idx, generic_val *args) {
                    std::this_thread::sleep_for(
                            std::chrono::milliseconds(1000));
                    std::vector<int> *penv = (std::vector<int> *)mod_data;
                    penv->at(runtime_config_t::get()
                                     .thread_pool_table_->get_thread_id())++;
                },
                0, nullptr, (void *)penv, 0, nthreads * 2, 1, nullptr);
    };

    if (cfg.managed_thread_pool_) {
        runtime::thread_manager::cur_mgr.run_main_function(
                funct, nullptr, &counts, nullptr);
    } else {
        funct(nullptr, &counts, nullptr);
    }
    for (auto &v : counts) {
        ASSERT_EQ(v, 2);
    }
}
#endif
