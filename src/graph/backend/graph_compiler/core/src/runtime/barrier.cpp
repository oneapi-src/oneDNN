/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <assert.h>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include "barrier.hpp"
#include "trace.hpp"
#include <runtime/microkernel/cpu/kernel_timer.hpp>
#include <runtime/runtime.hpp>

#ifdef SC_KERNEL_PROFILE
static void make_trace(int in_or_out, int count) {
    if (sc_is_trace_enabled()) { sc_make_trace_kernel(3, in_or_out, count); }
}
static void make_trace_prefetch(int in_or_out, int count) {
    if (sc_is_trace_enabled()) { sc_make_trace_kernel(4, in_or_out, count); }
}
#else
#define make_trace(v, count) SC_UNUSED(count)
#define make_trace_prefetch(v, count) SC_UNUSED(count)
#endif

namespace gc = dnnl::impl::graph::gc;

extern "C" SC_API void sc_arrive_at_barrier(gc::runtime::barrier_t *b,
        gc::runtime::barrier_idle_func idle_func, void *idle_args) {
    make_trace(0, 0);
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt = --b->pending_;
    assert(cnt >= 0);
    int count = 0;
    if (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } else {
        if (idle_func) {
            if (cur_round != b->rounds_.load()) {
                make_trace(1, 0);
                return;
            }
            auto ret = idle_func(&b->rounds_, cur_round + 1, -1, idle_args);
            count = ret & 0xffffffff;
        }
        while (cur_round == b->rounds_.load()) {
            _mm_pause();
        }
    }
    make_trace(1, count);
}

static_assert(sizeof(gc::runtime::barrier_t) == 64,
        "size of barrier_t should be 64-byte");

extern "C" SC_API void sc_init_barrier(
        gc::runtime::barrier_t *b, int num_barriers, uint64_t thread_count) {
    for (int i = 0; i < num_barriers; i++) {
        b[i].total_ = thread_count;
        b[i].pending_.store(thread_count);
        b[i].rounds_.store(0);
    }
}
