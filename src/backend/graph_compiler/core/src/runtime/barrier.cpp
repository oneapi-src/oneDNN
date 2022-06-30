/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include <immintrin.h>
#include "barrier.hpp"

extern "C" SC_API void sc_arrive_at_barrier(sc::runtime::barrier_t *b) {
    auto cur_round = b->rounds_.load();
    auto cnt = --b->pending_;
    assert(cnt >= 0);
    if (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } else {
        while (cur_round == b->rounds_.load()) {
            _mm_pause();
        }
    }
}

extern "C" SC_API void sc_init_barrier(
        sc::runtime::barrier_t *b, int num_barriers, uint64_t thread_count) {
    for (int i = 0; i < num_barriers; i++) {
        b[i].total_ = thread_count;
        b[i].pending_.store(thread_count);
        b[i].rounds_.store(0);
    }
}
