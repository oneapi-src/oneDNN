/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef MKLDNN_THREAD_HPP
#define MKLDNN_THREAD_HPP

#include "utils.hpp"
#include "z_magic.hpp"

#if defined(_OPENMP)
#include <omp.h>
#else // defined(_OPENMP)
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_in_parallel() { return 0; }
#endif

/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#   define PRAGMA_OMP_SIMD(...)
#else
#   define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif // defined(_MSC_VER) && !defined(__INTEL_COMPILER)

namespace mkldnn {
namespace impl {

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if (n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = utils::div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

/*
Threading based on nd_iterator.
"Copy-paste" approach because of performance issues under Intel Compiler:
More aggressive usage of templates/lambda usually causes performance degradation
on Intel(R) Xeon Phi(TM).
In particular, #pagma opm parallel if(cond) may bring significant performance
issue on Intel Xeon Phi.
*/
template <typename T0, typename T1, typename F>
void parallel_nd(const T0 D0, const T1 D1, F f) {
#pragma omp parallel
    {
        size_t work_amount = (size_t)D0 * D1;
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
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
#pragma omp parallel
    {
        size_t work_amount = (size_t)D0 * D1 * D2;
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
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
#pragma omp parallel
    {
        size_t work_amount = (size_t)D0 * D1 * D2 * D3;
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
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
#pragma omp parallel
    {
        size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
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
#pragma omp parallel
    {
        size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
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
    size_t work_amount = (size_t)D0 * D1;
    const int ithr = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
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
    size_t work_amount = (size_t)D0 * D1 * D2;
    const int ithr = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
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
    size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    const int ithr = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
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
    size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    const int ithr = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
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
    size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    const int ithr = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
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

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
