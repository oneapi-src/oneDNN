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

#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_UTILS_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_TEST_UTILS_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "test_utils_arr_fill.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/dimensions.hpp>
#include <util/parallel.hpp>
#include <util/utils.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#include <omp.h>
#endif

#define SKIP_BF16(dtype) \
    if (dtype == datatypes::bf16 \
            && !::dnnl::impl::graph::gc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512AMXBF16 \
            && !::dnnl::impl::graph::gc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512BF16) { \
        return; \
    }

#define REQUIRE_BF16() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512AMXBF16 \
            && !::dnnl::impl::graph::gc::get_default_context() \
                        ->machine_.cpu_flags_.fAVX512BF16) { \
        GTEST_SKIP(); \
    }

#define REQUIRE_VNNI() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512VNNI) { \
        GTEST_SKIP(); \
    }

#define REQUIRE_AVX512() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512F) { \
        GTEST_SKIP(); \
    }

#define REQUIRE_AVX512VBMI() \
    if (!::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512VBMI) { \
        GTEST_SKIP(); \
    }

#define IS_AMX_AVAILABLE() \
    (::dnnl::impl::graph::gc::get_default_context() \
                    ->machine_.cpu_flags_.fAVX512AMXTILE \
            && ::dnnl::impl::graph::gc::get_default_context() \
                       ->flags_.brgemm_use_amx_)

#define REQUIRE_AMX() \
    if (!IS_AMX_AVAILABLE()) { GTEST_SKIP(); }

#define SKIP_AMX() \
    if (IS_AMX_AVAILABLE()) { GTEST_SKIP(); }

#define SKIP_BOUNDARY_CHECK()

#define SKIP_ON_XBYAK() \
    if (::dnnl::impl::graph::gc::get_default_context()->flags_.jit_kind_ \
            == jit_kind::xbyak) { \
        GTEST_SKIP(); \
    }

#if defined(SC_LLVM_BACKEND)
#define SKIP_ON_LLVM() \
    if (::dnnl::impl::graph::gc::get_default_context()->flags_.jit_kind_ \
            == jit_kind::llvm) { \
        GTEST_SKIP(); \
    }
#else
#define SKIP_ON_LLVM()
#endif

#if defined(_MSC_VER)
#define SKIP_ON_WINDOWS() GTEST_SKIP();
#else
#define SKIP_ON_WINDOWS()
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#define SC_OMP_CLAUSE(CONTENT) _Pragma(CONTENT)
#else
#define SC_OMP_CLAUSE(CONTENT)
#endif

namespace test_utils {
#define TEST_ASSERT(cond, ...) \
    if (!(cond)) { \
        ::std::cerr << __FILE__ << "[" << __LINE__ << "]: " << __VA_ARGS__ \
                    << "\n"; \
        ::std::abort(); \
    }

template <typename T>
std::vector<T> reorder_low_accuracy_format(
        const std::vector<T> &input, unsigned blocks, unsigned K, unsigned N) {
    if (input.size() != blocks * K * N) {
        std::cerr << "wrong input shapes." << std::endl;
        exit(1);
    }
    unsigned inner_blocks = 2;
    if (std::is_same<T, uint16_t>() || std::is_same<T, bf16_t>()) {
        inner_blocks = 2;
    } else if (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>()) {
        inner_blocks = 4;
    } else {
        std::cerr << "unsupported datatype reorder!" << std::endl;
        exit(1);
    }
    std::vector<T> results;
    results.resize(blocks * utils::divide_and_ceil(K, inner_blocks)
            * inner_blocks * N);
    for (unsigned b = 0; b < blocks; b++) {
        for (unsigned k = 0; k < utils::divide_and_ceil(K, inner_blocks); k++) {
            for (unsigned n = 0; n < N; n++) {
                for (unsigned i = 0; i < inner_blocks; i++) {
                    if (k * inner_blocks + i < K) {
                        results[b * utils::divide_and_ceil(K, inner_blocks)
                                        * inner_blocks * N
                                + k * N * inner_blocks + n * inner_blocks + i]
                                = input[b * K * N + (k * inner_blocks + i) * N
                                        + n];
                    } else {
                        results[b * utils::divide_and_ceil(K, inner_blocks) * N
                                        * inner_blocks
                                + k * N * inner_blocks + n * inner_blocks + i]
                                = 0;
                    }
                }
            }
        }
    }
    return results;
}

inline std::pair<float, int8_t> get_scale_and_zeropoint(
        const std::vector<float> &input, bool symmetric = true) {
    if (input.size() < 2) {
        std::cerr << "too few elements in input tensor." << std::endl;
        exit(1);
    }
    float xmin = FLT_MAX, xmax = FLT_MIN;
    for (auto &it : input) {
        xmin = std::min(xmin, it);
        xmax = std::max(xmax, it);
    }
    float scale = 0.f;
    int8_t zero_point = 0;
    if (symmetric) {
        scale = std::max(std::abs(xmin), std::abs(xmax)) / 127;
        return std::make_pair(scale, zero_point);
    } else {
        int8_t qmin = -128;
        scale = (xmax - xmin) / (float)255;
        zero_point = qmin - xmin / scale;
        return std::make_pair(scale, zero_point);
    }
}

inline int8_t f32_to_int8(float x, float scale) {
    return x / scale;
}

inline uint8_t f32_to_uint8(float x, float scale, int8_t zero_point = 0) {
    return (int)(x / scale) + (int)zero_point;
}

inline float int8_to_f32(int x, float scale, int8_t zero_point = 0) {
    return (x - zero_point) * scale;
}

template <class T>
std::vector<float> convert_int8_to_f32(const std::vector<T> &ref_int8) {
    std::vector<float> ref_f32;
    ref_f32.reserve(ref_int8.size());
    std::for_each(ref_int8.begin(), ref_int8.end(),
            [&](T x) { ref_f32.push_back(static_cast<float>(x)); });
    return ref_f32;
}

template <class T>
std::vector<T> convert_f32_to_int8(const std::vector<float> &ref_f32) {
    int clip_max = 0, clip_min = 0;
    std::vector<T> ref_int8;
    ref_int8.reserve(ref_f32.size());
    std::for_each(ref_f32.begin(), ref_f32.end(), [&](float x) {
        if (std::is_same<T, uint8_t>()) {
            clip_max = 255;
            clip_min = 0;
        } else if (std::is_same<T, int8_t>()) {
            clip_max = 127;
            clip_min = -128;
        }
        ref_int8.push_back(static_cast<T>(std::max(
                std::min(static_cast<int>(std::round(x)), 255), clip_min)));
    });
    return ref_int8;
}

/**
 * Helper function to get the product result for a given dims.
 * @param dims the shape vector
 * @return the total number of elements for t tensor
 * */
inline size_t product(const std::vector<int64_t> &dims) {
    size_t ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    return ret;
}

/**
 * Helper function to calculate rmse given two arrays of float number
 * @param a float array 1
 * @param b float array 2
 * @return the rmse of two arrays
 * */
inline float cal_rmse(
        const std::vector<float> &a, const std::vector<float> &b) {
    COMPILE_ASSERT(a.size() && a.size() == b.size(),
            "Two vector should have same size and can not be empty.");
    float sum = 0.f;
    SC_OMP_CLAUSE("omp parallel for reduction(+ : sum)")
    for (int64_t i = 0; i < (int64_t)a.size(); i++) {
        auto e = a[i] - b[i];
        sum = sum + e * e;
    }
    return std::sqrt(sum / a.size());
}

/**
 * Helper function to calculate RMSRE(Root Mean Squared Relative Errors) given
 * two arrays of float number
 * @param a float array 1
 * @param b float array 2
 * @return the RMSRE of two arrays
 * */
inline float cal_rmsre(
        const std::vector<float> &a, const std::vector<float> &b) {
    COMPILE_ASSERT(a.size() && a.size() == b.size(),
            "Two vector should have same size and can not be empty.");
    float sum_sre = 0.f;
    SC_OMP_CLAUSE("omp parallel for reduction(+ : sum_sre)")
    for (int64_t i = 0; i < (int64_t)a.size(); i++) {
        auto e = a[i] - b[i];
        // Use Relative Percent Difference(RPD) to avoid floating point error
        auto re = e < 1e-10 ? e : 2.f * e / (fabs(a[i]) + fabs(b[i]));
        sum_sre += re * re;
    }
    // printf("rmsre = %f\n", std::sqrt(sum_sre / a.size()));
    return std::sqrt(sum_sre / a.size());
}

// TODO(xxx): Data type specification, might be mergeed and align with
// implementation part.
enum class data_type {
    /// Undefined data type (used for void).
    undef = 0,
    /// 16-bit/half-precision floating point.
    f16,
    /// non-standard 16-bit floating point with 7-bit mantissa.
    bf16,
    /// 32-bit/single-precision floating point.
    f32,
    /// 32-bit signed integer.
    s32,
    /// 8-bit signed integer.
    s8,
    /// 8-bit unsigned integer.
    u8,
    /// boolean
    boolean,
};

template <typename T>
struct data_traits {};

template <>
struct data_traits<int32_t> {
    static const auto dtype = data_type::s32;
};

template <>
struct data_traits<bf16_t> {
    static const auto dtype = data_type::bf16;
};

template <>
struct data_traits<uint8_t> {
    static const auto dtype = data_type::u8;
};

template <>
struct data_traits<int8_t> {
    static const auto dtype = data_type::s8;
};

template <>
struct data_traits<float> {
    static const auto dtype = data_type::f32;
};

template <typename T>
inline size_t div_up(const T a, const T b) {
    assert(b);
    return static_cast<size_t>((a + b - 1) / b);
}

/**
 * parallel threading section, from oneDNN
 */
template <typename T>
struct remove_reference {
    typedef T type;
};
template <typename T>
struct remove_reference<T &> {
    typedef T type;
};
template <typename T>
struct remove_reference<T &&> {
    typedef T type;
};

template <typename T>
inline T &&forward(typename remove_reference<T>::type &t) {
    return static_cast<T &&>(t);
}
template <typename T>
inline T &&forward(typename remove_reference<T>::type &&t) {
    return static_cast<T &&>(t);
}

template <typename T>
inline T nd_iterator_init(T start) {
    return start;
}
template <typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&...tuple) {
    start = nd_iterator_init(start, forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() {
    return true;
}
template <typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&...tuple) {
    if (nd_iterator_step(forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

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
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template <typename T0, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, F f) {
    T0 start {0}, end {0};
    balance211(D0, nthr, ithr, start, end);
    for (T0 d0 = start; d0 < end; ++d0)
        f(d0);
}

template <typename T0, typename T1, typename F>
void for_nd(const int ithr, const int nthr, const T0 &D0, const T1 &D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    T0 d0 {0};
    T1 d1 {0};
    nd_iterator_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        f(d0, d1);
        nd_iterator_step(d0, D0, d1, D1);
    }
}

// Skip a lambda function in the parameter pack.
template <typename T>
constexpr size_t get_work_amount(const T &v) {
    return 1;
}
template <typename T, typename... Args>
constexpr size_t get_work_amount(const T &v, Args &&...args) {
    return (size_t)v * get_work_amount(forward<Args>(args)...);
}

template <typename... Args>
void parallel_nd(Args &&...args) {
    const bool do_parallel = get_work_amount(forward<Args>(args)...) > 1;
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
    auto the_nd_func = std::bind(for_nd<typename std::decay<Args>::type...>,
            std::placeholders::_1, std::placeholders::_2,
            forward<Args>(args)...);
    auto func = [&](int64_t i) {
        const int nthr
                = !do_parallel ? 1 : runtime_config_t::get().get_num_threads();
        const int ithr = !do_parallel ? 0 : i;
        the_nd_func(ithr, nthr);
    };
    if (do_parallel) {
        utils::parallel_for(
                0, runtime_config_t::get().get_num_threads(), 1, func);
    } else {
        func(0);
    }
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#pragma omp parallel if (do_parallel)
    {
        const int nthr = !do_parallel ? 1 : omp_get_max_threads();
        const int ithr = !do_parallel ? 0 : omp_get_thread_num();
        for_nd(ithr, nthr, forward<Args>(args)...);
    }
#else
    for_nd(0, 1, forward<Args>(args)...);
#endif
}
/** end oneDNN */

template <typename T>
void rand_fill_stable(T *v, size_t sz, uint32_t &seed) {
    for (size_t i = 0; i < sz; i++) {
        v[i] = test_utils::rand_for_test<T>(seed);
    }
}

template <typename T, typename A>
static void fill_data(T *buf, size_t size, A val) {
    parallel_nd(size, [&](size_t i) { buf[i] = static_cast<T>(val); });
}

// without gtest interface so we can return count
template <typename T>
static int compare_data_count(T *dst, T *ref, size_t size, float rtol = 1e-4,
        float atol = 1e-6, std::function<void()> on_error = nullptr) {
    bool pass = true;
    std::atomic<int> count(0);
    parallel_nd(size, [&](size_t i) {
        // early stopping to avoid verbose outputs
        if (data_traits<T>::dtype == data_type::f32
                || data_traits<T>::dtype == data_type::bf16) {
            const float ref_f32 = static_cast<float>(ref[i]);
            const float dst_f32 = static_cast<float>(dst[i]);
            const float diff_f32 = dst_f32 - ref_f32;
            const float gap = rtol
                            * (std::abs(ref_f32) > std::abs(dst_f32)
                                            ? std::abs(ref_f32)
                                            : std::abs(dst_f32))
                    + atol;
            if (std::abs(diff_f32) > gap) {
                pass = false;
                count = count + 1;
            }
        } else {
            EXPECT_EQ(ref[i], dst[i]) << "Index: " << i;
            if (ref[i] != dst[i]) {
                pass = false;
                count = count + 1;
            }
        }
    });
    if (!pass && on_error) { on_error(); }
    return count;
}

template <typename T>
inline void compare_data_single(
        const T *dst, const T *ref, int i, float rtol, float atol, bool &pass) {
    EXPECT_EQ(ref[i], dst[i]) << "Index: " << i;
    if (ref[i] != dst[i]) { pass = false; }
}

template <typename T>
inline void compare_data_fp(
        const T *dst, const T *ref, int i, float rtol, float atol, bool &pass) {
    const float ref_f32 = static_cast<float>(ref[i]);
    const float dst_f32 = static_cast<float>(dst[i]);
    const double diff_f32 = dst_f32 - ref_f32;
    const double gap = double(rtol)
                    * (std::abs(ref_f32) > std::abs(dst_f32)
                                    ? std::abs(ref_f32)
                                    : std::abs(dst_f32))
            + atol;
    bool good = std::abs(diff_f32) <= gap;
    EXPECT_TRUE(good) << "Index: " << i << ", ref_f32=" << ref_f32
                      << ", dst_f32=" << dst_f32;
    if (!good) { pass = false; }
}

template <>
inline void compare_data_single(const float *dst, const float *ref, int i,
        float rtol, float atol, bool &pass) {
    compare_data_fp(dst, ref, i, rtol, atol, pass);
}

template <>
inline void compare_data_single(const bf16_t *dst, const bf16_t *ref, int i,
        float rtol, float atol, bool &pass) {
    compare_data_fp(dst, ref, i, rtol, atol, pass);
}

template <typename T>
inline void compare_data(const T *dst, const T *ref, size_t size,
        float rtol = 1e-4, float atol = 1e-6,
        std::function<void()> on_error = nullptr) {
    bool pass = true;
    parallel_nd(size, [&](size_t i) {
        // early stopping to avoid verbose outputs
        if (!pass) { return; }
        compare_data_single(dst, ref, i, rtol, atol, pass);
    });
    if (!pass && on_error) { on_error(); }
}

template <typename T, bool is_struct = std::is_class<T>::value>
struct is_vector_like {
    static constexpr bool value
            = std::is_same<decltype(std::declval<T>().size()), size_t>::value
            && std::is_pointer<decltype(std::declval<T>().data())>::value;
};

template <typename T>
struct is_vector_like<T, false> {
    static constexpr bool value = false;
};

// compares to vector like containers
template <typename T1, typename T2>
inline void compare_data(const T1 &dst, const T2 &ref, float rtol = 1e-4,
        typename std::enable_if<is_vector_like<T1>::value
                        && is_vector_like<T2>::value,
                float>::type atol
        = 1e-6,
        std::function<void()> on_error = nullptr) {
    ASSERT_NE(ref.size(), 0u) << "The ref size is 0";
    ASSERT_EQ(dst.size(), ref.size())
            << "The dst and ref size is not equal (" << ref.size() << " vs "
            << dst.size() << ").";
    compare_data(dst.data(), ref.data(), ref.size(), rtol, atol, on_error);
}

template <typename T>
inline void check_fp_data(T *dat, size_t size, bool check_nan = true,
        bool check_inf = true, std::function<void()> on_error = nullptr) {
    bool pass = true;
    parallel_nd(size, [&](size_t i) {
        // early stopping to avoid verbose outputs
        if (!pass) { return; }

        bool is_nan = check_nan && std::isnan(dat[i]);
        bool is_inf = check_inf && std::isinf(dat[i]);

        bool good = !is_nan && !is_inf;

        EXPECT_TRUE(good) << "Index: " << i << ", dat_fp=" << dat[i];

        if (!good) { pass = false; }
    });
    if (!pass && on_error) { on_error(); }
}

template <typename T>
inline void check_fp_data(std::vector<T> &dat, bool check_nan = true,
        bool check_inf = true, std::function<void()> on_error = nullptr) {
    ASSERT_NE(dat.size(), 0u) << "Data size is 0";
    ASSERT_TRUE(data_traits<T>::dtype == data_type::f16
            || data_traits<T>::dtype == data_type::bf16
            || data_traits<T>::dtype == data_type::f32)
            << "Data must be floating point";
    check_fp_data(dat.data(), dat.size(), check_nan, check_inf, on_error);
}

template <typename T>
inline void dump_data(const std::vector<T> &ref) {
    // static_assert(sizeof(T) == 4, "Expecting size_t(T)==4");
    // not sure if the string representaion of float is stable
    // use int32 bit pattern
    for (auto &v : ref) {
        std::cout << std::hex << *(int32_t *)(&v) << ' ';
    }
    std::cout << '\n';
}

template <typename T>
inline T cal_size(
        const std::vector<T> &input, const std::vector<T> &strides = {}) {
    if (input.empty()) { return 0; }
    T result = 1;
    if (strides.empty()) {
        for (auto it : input) {
            result *= it;
        }
    } else {
        for (size_t i = 0; i < input.size(); ++i) {
            result += (input[i] - 1) * strides[i];
        }
    }
    return result;
}

inline sc_dims compute_dense_stride(const sc_dims &dim) {
    sc_dims result(dim.size(), 1);
    for (int i = dim.size() - 2; i >= 0; --i) {
        result[i] = result[i + 1] * dim[i + 1];
    }
    return result;
}

inline uint8_t get_dyn_mask(const sc_dims &in) {
    uint8_t ret = 0;
    for (size_t i = 0; i < in.size(); i++) {
        ret |= (is_dynamic_dim(in[i]) << i);
    }
    return ret;
}
} // namespace test_utils

struct thread_num_reset {
    int old_;
    thread_num_reset() : old_(runtime_config_t::get().get_num_threads()) {}
    ~thread_num_reset() { runtime_config_t::get().set_num_threads(old_); }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
