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

#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "test_utils.hpp"
#include "test_utils_arr_fill.hpp"
#include "gtest/gtest.h"
#include <runtime/context.hpp>
#include <runtime/dynamic_threadpool.hpp>
#include <runtime/dynamic_threadpool_c.hpp>
#include <runtime/generic_val.hpp>
#include <runtime/low_level_threadpool_wrapper.hpp>
#include <util/parallel.hpp>
#include <util/scoped_timer.hpp>

using namespace dnnl::impl::graph::gc;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

using namespace runtime::dynamic_threadpool;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include <test_thread.hpp>
#define dnnl_thread_env() \
    dnnl::testing::scoped_tp_activation_t unused_raii {}
#else
#define dnnl_thread_env()
#endif

namespace dyn_tp_test {
// a testing threadpool assuming there are N+4 threads, when there are actually
// N threads
struct threadpool_adapter_for_test_t : threadpool_adapter_t {
    static threadpool_scheduler *all_thread_prepare(
            threadpool_scheduler *ths, runtime::stream_t *stream, int threads) {
        return threadpool_adapter_t::all_thread_prepare(
                ths, stream, threads + 6);
    }
};

static void thread_main_testing(main_func_t f, runtime::stream_t *stream,
        void *mod_data, generic_val *args) {
    runtime::call_threadpool<threadpool_adapter_for_test_t,
            threadpool_scheduler>(nullptr, f, stream, mod_data, args);
}
} // namespace dyn_tp_test

namespace dyn_tp_test {
static void layer0_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args);
static void layer1_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args);
static void layer2_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args);
static void layer3_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args);
static void layer3outer_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args);
static void layer_start_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    void *buf = sc_dyn_threadpool_shared_buffer(128 * sizeof(float));
    sc_dyn_threadpool_create_work_items(layer0_func, itr, /*num_iter**/ 1,
            /*loop_len*/ 128, /*num_blocks*/ 16, /*outer_loop_hash*/ bs,
            /*num_buffers*/ 1, &buf, 1);
}
static void layer0_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    auto i = itr[1];
    float *input = (float *)args[0].v_ptr;
    float *out = (float *)args[1].v_ptr;
    // float *buf = (float *)buffer[0];
    for (int j = 0; j < 1024; j++) {
        out[bs * 128 * 1024 + i * 1024 + j] = 1.0f;
    }
    if (sc_dyn_threadpool_loop_end(nullptr, 0)) {
        sc_dyn_threadpool_create_work_items(layer1_func, nullptr,
                /*num_iter**/ 1,
                /*loop_len*/ 128, /*num_blocks*/ 16, /*outer_loop_hash*/ bs,
                /*num_buffers*/ 1, buffer, 1);
    }
}

static void layer1_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    auto i = itr[1];
    float *input = (float *)args[0].v_ptr;
    float *out = (float *)args[1].v_ptr;
    float *buf = (float *)buffer[0];
    buf[i] = 0;
    if (sc_dyn_threadpool_loop_end(nullptr, 0)) {
        sc_dyn_threadpool_create_work_items(layer2_func, nullptr,
                /*num_iter**/ 1,
                /*loop_len*/ 128, /*num_blocks*/ 16, /*outer_loop_hash*/ bs,
                /*num_buffers*/ 1, buffer, 1);
    }
};

static void layer2_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    auto i = itr[1];
    // auto j = itr[2];
    float *input = (float *)args[0].v_ptr;
    float *out = (float *)args[1].v_ptr;
    float *buf = (float *)buffer[0];
    assert(buf[i] == 0);
    for (int j = 0; j < 1024; j++) {
        buf[i] += input[bs * 128 * 1024 + i * 1024 + j];
    }
    if (sc_dyn_threadpool_loop_end(nullptr, 0)) {
        sc_dyn_threadpool_create_work_items(layer3outer_func, nullptr,
                /*num_iter**/ 1,
                /*loop_len*/ 128, /*num_blocks*/ 16, /*outer_loop_hash*/ bs,
                /*num_buffers*/ 1, nullptr, 1);
    }
}
static void layer3outer_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    auto i = itr[1];
    float *input = (float *)args[0].v_ptr;
    float *out = (float *)args[1].v_ptr;
    float *buf = (float *)buffer[0];
    ///// checking
    for (int j = 0; j < 1024; j++) {
        assert(out[bs * 128 * 1024 + i * 1024 + j] == 1.0f);
    }
    ///// end of checking
    sc_dyn_threadpool_create_work_items(layer3_func, itr,
            /*num_iter**/ 2,
            /*loop_len*/ 1024, /*num_blocks*/ 4,
            /*outer_loop_hash*/ bs * 23 + i,
            /*num_buffers*/ 1, nullptr, 1);
}
static void layer3_func(void *stream, void *mod_data, uint64_t *itr,
        void **buffer, generic_val *args) {
    auto bs = itr[0];
    auto i = itr[1];
    auto j = itr[2];
    float *input = (float *)args[0].v_ptr;
    float *out = (float *)args[1].v_ptr;
    float *buf = (float *)buffer[0];
    assert(buf[i] != 0);
    out[bs * 128 * 1024 + i * 1024 + j] += buf[i] + std::cos(j);
}
} // namespace dyn_tp_test

static constexpr int num_expected_threads = 16;
static constexpr int num_real_threads = num_expected_threads - 6;
TEST(GCCore_CPU_dyn_thread_pool, TestBarrier) {
    /*
    float input[32,128,1024]
    float out[32,128,1024]
    parallelfor(int bs=0;bs<32;bs++) {
        float buf[128];
        # loop 1: triggered=true
        parallelfor(int i=0;i<128;i++) {
            for(int j=0;j<1024;j++) {
                out[bs,i,j] = 1.0
            }
        }
        # loop 2: triggered=true
        parallelfor(int i=0;i<128;i++) {
            buf[i] = 0
        }
        # loop 3: dep=loop2, sync_level=1
        parallelfor(int i=0;i<128;i++) {
            for(int j=0;j<1024;j++) {
                buf[i] += input[bs,i,j]
            }
        }

        # loop 4: dep=loop1-loop3, sync_level=1
        parallelfor(int i=0;i<128;i++) {
            parallelfor(int j=0;j<1024;j++) {
                out[bs,i,j] = buf[i] + cos(j)
            }
        }
    }
    */

    test_buffer<float> input = alloc_array<float>(32 * 128 * 1024);
    test_buffer<float> out = alloc_array<float>(32 * 128 * 1024);
    generic_val buffers[] = {input.data(), out.data()};
    auto run = [](decltype(
                          runtime::dynamic_threadpool::thread_main) thread_main,
                       generic_val *buffers) {
        thread_main(
                [](runtime::stream_t *, void *, generic_val *buffers) {
                    sc_dyn_threadpool_sched_init(runtime::get_default_stream(),
                            nullptr, buffers, 1, /*queue size*/ 1024,
                            num_expected_threads);
                    sc_dyn_threadpool_create_work_items(
                            dyn_tp_test::layer_start_func, nullptr, 0, 32, 32,
                            0, 0, nullptr,
                            work_item_flags::bind_last_level
                                    | work_item_flags::is_root | 1);

                    sc_dyn_threadpool_run();
                    sc_dyn_threadpool_sched_destroy();
                },
                runtime::get_default_stream(), nullptr, buffers);
    };

    test_buffer<float> expected;
    {
        dnnl_thread_env();
        expected = alloc_array<float>(32 * 128 * 1024);
        utils::parallel_for(0, 32, 1, [&](int bs) {
            for (int i = 0; i < 128; i++) {
                float sum = 0;
                for (int j = 0; j < 1024; j++) {
                    sum += input[bs * 128 * 1024 + i * 1024 + j];
                }
                for (int j = 0; j < 1024; j++) {
                    expected[bs * 128 * 1024 + i * 1024 + j]
                            = 1.0f + sum + std::cos(j);
                }
            }
        });
        run(runtime::dynamic_threadpool::thread_main, buffers);
    }
    test_utils::compare_data(out, expected);
    SET_THREADS_OR_SKIP(num_real_threads);
    out.zeroout();
    {
        dnnl_thread_env();
        run(dyn_tp_test::thread_main_testing, buffers);
    }
    test_utils::compare_data(out, expected);
}

namespace test2 {
static void layer3_func(void *stream, void *mod_data, uint64_t *itr, void **,
        generic_val *args) {
    auto i = itr[0];
    std::atomic<int> *input = (std::atomic<int> *)args[0].v_ptr;
    int *result = (int *)args[1].v_ptr;
    int accu = i;
    for (int j = 0; j < 8; j++) {
        int check = i * 3 + j;
        for (int k = 0; k < 4; k++) {
            accu += (check + k);
        }
    }
    result[i] = (accu == input[i]) ? 2 : 1;
};

static void layer2_func(void *stream, void *mod_data, uint64_t *itr,
        void **bufs, generic_val *args) {
    auto i = itr[0];
    auto j = itr[1];
    auto k = itr[2];
    std::atomic<int> *input = (std::atomic<int> *)args[0].v_ptr;
    int *result = (int *)args[1].v_ptr;
    int *check = (int *)bufs[0];
    input[i] += (check[0] + k);
    if (auto scope = sc_dyn_threadpool_loop_end(nullptr, 0)) {
        if (sc_dyn_threadpool_loop_end(scope, 1)) {
            sc_dyn_threadpool_create_work_items(layer3_func, nullptr,
                    /*num_iter*/ 1, /*looplen*/ 1, /*num_blocks*/ 1,
                    /*loop_hash*/ i, /*numbuffers*/ 0, nullptr, /*flags*/ 1);
        }
    }
};

static void layer1_func(void *stream, void *mod_data, uint64_t *itr,
        void **bufs, generic_val *args) {
    auto i = itr[0];
    auto j = itr[1];
    std::atomic<int> *input = (std::atomic<int> *)args[0].v_ptr;
    int *result = (int *)args[1].v_ptr;
    int *check = (int *)sc_dyn_threadpool_shared_buffer(sizeof(int));
    check[0] = i * 3 + j;
    sc_dyn_threadpool_create_work_items(layer2_func, nullptr,
            /*num_iter*/ 2, /*looplen*/ 4, /*num_blocks*/ 4,
            /*loop_hash*/ i * 8 + j, /*numbuffers*/ 1, (void **)&check,
            /*flags*/ 1);
};

static void layer0_func(void *stream, void *mod_data, uint64_t *itr,
        void **bufs, generic_val *args) {
    auto i = itr[0];
    std::atomic<int> *input = (std::atomic<int> *)args[0].v_ptr;
    int *result = (int *)args[1].v_ptr;
    input[i] = i;
    sc_dyn_threadpool_create_work_items(layer1_func, nullptr,
            /*num_iter*/ 1, /*looplen*/ 8, /*num_blocks*/ 4,
            /*loop_hash*/ i * 8, /*numbuffers*/ 0, nullptr,
            /*flags*/ 1);
};
} // namespace test2

TEST(GCCore_CPU_dyn_thread_pool, TestMultiLevelSync) {
    /*
    atomic<int> data[4];
    parallelfor(int i=0;i<4;i++) {
        // scope0
        data[i]=i;
        parallelfor(int j=0;j<8;j++) {
            int check[1];
            check[0] = i*3+j; // scope1
            parallelfor(int k=0;k<4;k++) {
                data[i] += (check[0]+k) // scope2
            }
        }
        assert(data[i]==XXX) // scope3
    }
    */

    test_buffer<int> input = alloc_array<int>(4, INIT_ZERO);
    test_buffer<int> result = alloc_array<int>(4, INIT_ZERO);
    generic_val buffers[] = {input.data(), result.data()};
    auto run = [](decltype(
                          runtime::dynamic_threadpool::thread_main) thread_main,
                       generic_val *buffers) {
        thread_main(
                [](runtime::stream_t *, void *, generic_val *buffers) {
                    sc_dyn_threadpool_sched_init(runtime::get_default_stream(),
                            nullptr, buffers,
                            /*num_roots*/ 1,
                            /*queue size*/ 512, num_expected_threads);

                    sc_dyn_threadpool_create_work_items(test2::layer0_func,
                            nullptr,
                            /*num_iter*/ 0, /*looplen*/ 4, /*num_blocks*/ 4,
                            /*loop_hash*/ 0, /*numbuffers*/ 0, nullptr,
                            /*flags*/
                            runtime::dynamic_threadpool::work_item_flags::
                                            bind_last_level
                                    | runtime::dynamic_threadpool::
                                            work_item_flags::is_root
                                    | 1);
                    sc_dyn_threadpool_run();
                    sc_dyn_threadpool_sched_destroy();
                },
                runtime::get_default_stream(), nullptr, buffers);
    };
    {
        dnnl_thread_env();
        run(runtime::dynamic_threadpool::thread_main, buffers);
    }
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(result[i], 2);
    }
    result.zeroout();
    {
        dnnl_thread_env();
        SET_THREADS_OR_SKIP(num_real_threads);
        run(dyn_tp_test::thread_main_testing, buffers);
    }
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(result[i], 2);
    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
