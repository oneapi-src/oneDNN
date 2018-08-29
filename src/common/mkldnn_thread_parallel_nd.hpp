/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef MKLDNN_THREAD_PARALLEL_ND_HPP
#define MKLDNN_THREAD_PARALLEL_ND_HPP

/* This header must be included by mkldnn_thread.hpp only */

namespace mkldnn {
namespace impl {

/* Threading based on nd_iterator.
 * "Copy-paste" approach because of performance issues under Intel Compiler:
 * More aggressive usage of templates/lambda usually causes performance
 * degradation on Intel(R) Xeon Phi(TM).
 * In particular, #pagma omp parallel if(cond) may bring significant
 * performance issue on Intel Xeon Phi.
 */

template <typename T0, typename T1, typename F>
void parallel_nd(const T0 D0, const T1 D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;

#   pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0}; T1 d1{0};
        utils::nd_iterator_init(start, d0, D0, d1, D1);
        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1);
            utils::nd_iterator_step(d0, D0, d1, D1);
        }
    }
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_nd(const T0 D0, const T1 D1, const T2 D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;

#   pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0}; T1 d1{0}; T2 d2{0};
        utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1, d2);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd(const T0 D0, const T1 D1, const T2 D2, const T3 D3, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;

#   pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0};
        utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1, d2, d3);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
    typename F>
void parallel_nd(const T0 D0, const T1 D1, const T2 D2, const T3 D3,
        const T4 D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;

#   pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0};
        utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1, d2, d3, d4);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename F>
void parallel_nd(const T0 D0, const T1 D1, const T2 D2, const T3 D3,
        const T4 D4, const T5 D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;

#   pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0}; T5 d5{0};
        utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
            d5, D5);
        for (size_t iwork = start; iwork < end; ++iwork) {
            f(d0, d1, d2, d3, d4, d5);
            utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
                d5, D5);
        }
    }
}

/* For use inside of parallel section */

template <typename T0, typename T1, typename F>
void parallel_nd_in_omp(const T0 D0, const T1 D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;

    const int ithr = mkldnn_get_thread_num();
    const int nthr = mkldnn_get_num_threads();
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);
    T0 d0{0}; T1 d1{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        utils::nd_iterator_step(d0, D0, d1, D1);
    }
}
template <typename T0, typename T1, typename T2, typename F>
void parallel_nd_in_omp(const T0 D0, const T1 D1, const T2 D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;

    const int ithr = mkldnn_get_thread_num();
    const int nthr = mkldnn_get_num_threads();
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);
    T0 d0{0}; T1 d1{0}; T2 d2{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
    }
}
template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_nd_in_omp(const T0 D0, const T1 D1, const T2 D2, const T3 D3,
    F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;

    const int ithr = mkldnn_get_thread_num();
    const int nthr = mkldnn_get_num_threads();
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);
    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
    typename F>
void parallel_nd_in_omp(const T0 D0, const T1 D1, const T2 D2, const T3 D3,
    const T4 D4, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;

    const int ithr = mkldnn_get_thread_num();
    const int nthr = mkldnn_get_num_threads();
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);
    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
    typename T5, typename F>
void parallel_nd_in_omp(const T0 D0, const T1 D1, const T2 D2, const T3 D3,
    const T4 D4, const T5 D5, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;

    const int ithr = mkldnn_get_thread_num();
    const int nthr = mkldnn_get_num_threads();
    size_t start{0}, end{0};
    balance211(work_amount, nthr, ithr, start, end);
    T0 d0{0}; T1 d1{0}; T2 d2{0}; T3 d3{0}; T4 d4{0}; T5 d5{0};
    utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
        d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1, d2, d3, d4, d5);
        utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

} // namespace impl
} // namespace mkldnn

#endif
