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

#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_UTILS_ARR_FILL_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_UTILS_ARR_FILL_HPP

#include <algorithm>
#include <limits>
#include <random>
#include <stdlib.h>
#include <typeinfo>
#include <vector>
#include <checked_ptr.hpp>
#include <runtime/aligned_ptr.hpp>
#include <util/bf16.hpp>
#include <util/fp16.hpp>
#include <util/parallel.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#include <omp.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace test_utils {

// rand number [0,0x7FFF]
inline uint32_t randint_for_test(uint32_t &seed) {
    seed = (214013 * seed + 2531011);
    return (seed >> 16) & 0x7FFF;
}

template <typename T>
T rand_for_test(uint32_t &seed, T a = 0, T b = 0x7FFF) {
    return randint_for_test(seed) % (b - a) + a;
}

// rand number in [-1,1]
template <>
inline float rand_for_test<float>(uint32_t &seed, float a, float b) {
    uint32_t rnd_int = rand_for_test<uint32_t>(seed);
    // map to float
    return rnd_int / float(0x7FFF) * (b - a) + a;
}

template <>
inline double rand_for_test<double>(uint32_t &seed, double a, double b) {
    uint32_t rnd_int = rand_for_test<uint32_t>(seed);
    // map to double
    return rnd_int / double(0x7FFF) * (b - a) + a;
}

// rand number in [-1,1]
template <>
inline bf16_t rand_for_test<bf16_t>(uint32_t &seed, bf16_t a, bf16_t b) {
    return rand_for_test<float>(seed, a, b);
}

// rand number in [-1,1]
template <>
inline fp16_t rand_for_test<fp16_t>(uint32_t &seed, fp16_t a, fp16_t b) {
    return rand_for_test<float>(seed, a, b);
}

template <typename T>
inline void fill_data(T *data, size_t size, T a, T b) {
    size_t num_thread = runtime_config_t::get().get_num_threads();
    auto workload = size / num_thread;
    if (workload < 64 / (sizeof(T))) {
        uint32_t seed = rand(); // NOLINT
        for (size_t i = 0; i < size; i++) {
            data[i] = rand_for_test<T>(seed, a, b);
        }
        return;
    }
    dnnl::impl::graph::gc::utils::parallel_for(
            0, num_thread, 1, [&](int64_t t0) {
                uint64_t t = t0;
                auto start = t * workload;
                auto end = (t == num_thread - 1) ? size : (t + 1) * workload;
                uint32_t seed = rand() + (uint32_t)t; // NOLINT
                for (auto i = start; i < end; i++) {
                    data[i] = rand_for_test<T>(seed, a, b);
                }
            });
}

template <typename T,
        bool is_float
        = std::is_floating_point<T>::value || std::is_same<T, bf16_t>::value>
struct fill_data_impl_t {
    // default: is_float=true, value range from -1 to 1
    static void call(T *data, size_t size) {
        fill_data(data, size, (T)-1.0, (T)1.0);
    }
};

template <typename T>
struct fill_data_impl_t<T, false> {
    // default: is_float=false
    static void call(T *data, size_t size) {
        fill_data(data, size, std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max());
    }
};

template <typename T>
inline void fill_data(T *data, size_t size) {
    fill_data_impl_t<T>::call(data, size);
}
} // namespace test_utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#if defined(NDEBUG) || defined(_WIN32)
template <typename T>
using test_buffer = dnnl::impl::graph::gc::aligned_ptr_t<T>;
#else
template <typename T>
using test_buffer = dnnl::impl::graph::gc::checked_ptr_t<T>;
#endif

enum init_action { INIT_NOOP, INIT_ZERO, INIT_RANDOM, INIT_RANGE };

template <typename T>
test_buffer<T> alloc_array(
        size_t size, init_action action = INIT_RANDOM, T a = 0, T b = 1) {
    test_buffer<T> data(size);
    if (action == INIT_RANDOM) {
        dnnl::impl::graph::gc::test_utils::fill_data(data.data(), size);
    } else if (action == INIT_ZERO) {
        data.zeroout();
    } else if (action == INIT_RANGE) {
        // for bf16, range is [approximate a , approximate b]
        // for others, range is [a, b]
        dnnl::impl::graph::gc::test_utils::fill_data(data.data(), size, a, b);
    }
    return data;
}

template <typename T>
dnnl::impl::graph::gc::aligned_ptr_t<T> alloc_array2(
        size_t size, bool fill_rnd = true) {
    dnnl::impl::graph::gc::aligned_ptr_t<T> data(size);
    if (fill_rnd) {
        dnnl::impl::graph::gc::test_utils::fill_data(data.data(), size);
    }
    return data;
}

#endif
