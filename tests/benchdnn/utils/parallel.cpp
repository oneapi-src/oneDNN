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

#include "tests/test_thread.hpp"

#include "utils/parallel.hpp"
#include "src/common/counting_barrier.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#define ACTIVATE_THREADPOOL \
    dnnl::testing::scoped_tp_activation_t scoped_activation
#else
#define ACTIVATE_THREADPOOL
#endif

void sync() {
    // For async threadpool that behave in order, we add a sync
    dnnl::impl::counting_barrier_t b(1);
    dnnl::impl::parallel(1, [&b](int,int){b.notify();});
    b.wait();
}

// Note: no need in deactivation as `scoped_activation` object will deactivate
// it automatically at destruction.
void benchdnn_parallel_nd(int64_t D0, const std::function<void(int64_t)> &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, f);
    sync();
}

void benchdnn_parallel_nd(int64_t D0, int64_t D1,
        const std::function<void(int64_t, int64_t)> &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, D1, f);
    sync();
}

void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2,
        const std::function<void(int64_t, int64_t, int64_t)> &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, D1, D2, f);
    sync();
}

void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        const std::function<void(int64_t, int64_t, int64_t, int64_t)> &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, D1, D2, D3, f);
    sync();
}

void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        int64_t D4,
        const std::function<void(int64_t, int64_t, int64_t, int64_t, int64_t)>
                &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, D1, D2, D3, D4, f);
    sync();
}

void benchdnn_parallel_nd(int64_t D0, int64_t D1, int64_t D2, int64_t D3,
        int64_t D4, int64_t D5,
        const std::function<void(
                int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)> &f) {
    ACTIVATE_THREADPOOL;
    dnnl::impl::parallel_nd(D0, D1, D2, D3, D4, D5, f);
    sync();
}

int benchdnn_get_max_threads() {
    return dnnl_get_max_threads();
}
