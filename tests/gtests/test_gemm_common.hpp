/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef TEST_GEMM_COMMON_H
#define TEST_GEMM_COMMON_H

#include "dnnl_test_common.hpp"
#include "dnnl_thread.hpp"
#include "gtest/gtest.h"

#include "dnnl.h"
#include "dnnl_types.h"

#include <utility>
#include <vector>
#include <type_traits>

#define CONCAT_WITH_UNDERSCORE_(a, b) a##_##b
#define CONCAT_WITH_UNDERSCORE(a, b) CONCAT_WITH_UNDERSCORE_(a, b)

#define INST_TEST_CASE_(str, ...) \
    INSTANTIATE_TEST_SUITE_P(str, gemm_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) \
    INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(str, TEST_CASE_NAME_PREFIX), __VA_ARGS__)

#define CPU_INST_TEST_CASE_(str, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(str, gemm_test, ::testing::Values(__VA_ARGS__))
#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE_( \
            CONCAT_WITH_UNDERSCORE(str, TEST_CASE_NAME_PREFIX), __VA_ARGS__)

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL

// Declare OpenCL GEMM interfaces for testing
extern "C" {
dnnl_status_t dnnl_ocl_sgemm(cl_command_queue queue, char transa, char transb,
        dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k, cl_float alpha, cl_mem a,
        dnnl_dim_t offset_a, dnnl_dim_t lda, cl_mem b, dnnl_dim_t offset_b,
        dnnl_dim_t ldb, cl_float beta, cl_mem c, dnnl_dim_t offset_c,
        dnnl_dim_t ldc);

dnnl_status_t dnnl_ocl_hgemm(cl_command_queue queue, char transa, char transb,
        dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k, cl_float alpha, cl_mem a,
        dnnl_dim_t offset_a, dnnl_dim_t lda, cl_mem b, dnnl_dim_t offset_b,
        dnnl_dim_t ldb, cl_float beta, cl_mem c, dnnl_dim_t offset_c,
        dnnl_dim_t ldc);

dnnl_status_t dnnl_ocl_gemm_f16f16f32(cl_command_queue queue, char transa,
        char transb, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k, cl_float alpha,
        cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda, cl_mem b,
        dnnl_dim_t offset_b, dnnl_dim_t ldb, cl_float beta, cl_mem c,
        dnnl_dim_t offset_c, dnnl_dim_t ldc);

dnnl_status_t dnnl_ocl_gemm_bf16bf16f32(cl_command_queue queue, char transa,
        char transb, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k, cl_float alpha,
        cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda, cl_mem b,
        dnnl_dim_t offset_b, dnnl_dim_t ldb, cl_float beta, cl_mem c,
        dnnl_dim_t offset_c, dnnl_dim_t ldc);

dnnl_status_t dnnl_ocl_gemm_bf16bf16bf16(cl_command_queue queue, char transa,
        char transb, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k, cl_float alpha,
        cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda, cl_mem b,
        dnnl_dim_t offset_b, dnnl_dim_t ldb, cl_float beta, cl_mem c,
        dnnl_dim_t offset_c, dnnl_dim_t ldc);

dnnl_status_t dnnl_ocl_gemm_s8s8s32(cl_command_queue queue, char transa,
        char transb, char offsetc, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k,
        cl_float alpha, cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda,
        int8_t ao, cl_mem b, dnnl_dim_t offset_b, dnnl_dim_t ldb, int8_t bo,
        cl_float beta, cl_mem c, dnnl_dim_t offset_c, dnnl_dim_t ldc, cl_mem co,
        dnnl_dim_t offset_co);
dnnl_status_t dnnl_ocl_gemm_u8s8s32(cl_command_queue queue, char transa,
        char transb, char offsetc, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k,
        cl_float alpha, cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda,
        uint8_t ao, cl_mem b, dnnl_dim_t offset_b, dnnl_dim_t ldb, int8_t bo,
        cl_float beta, cl_mem c, dnnl_dim_t offset_c, dnnl_dim_t ldc, cl_mem co,
        dnnl_dim_t offset_co);
dnnl_status_t dnnl_ocl_gemm_s8u8s32(cl_command_queue queue, char transa,
        char transb, char offsetc, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k,
        cl_float alpha, cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda,
        int8_t ao, cl_mem b, dnnl_dim_t offset_b, dnnl_dim_t ldb, uint8_t bo,
        cl_float beta, cl_mem c, dnnl_dim_t offset_c, dnnl_dim_t ldc, cl_mem co,
        dnnl_dim_t offset_co);
dnnl_status_t dnnl_ocl_gemm_u8u8s32(cl_command_queue queue, char transa,
        char transb, char offsetc, dnnl_dim_t m, dnnl_dim_t n, dnnl_dim_t k,
        cl_float alpha, cl_mem a, dnnl_dim_t offset_a, dnnl_dim_t lda,
        uint8_t ao, cl_mem b, dnnl_dim_t offset_b, dnnl_dim_t ldb, uint8_t bo,
        cl_float beta, cl_mem c, dnnl_dim_t offset_c, dnnl_dim_t ldc, cl_mem co,
        dnnl_dim_t offset_co);
}
#endif

// Declare bfloat16 GEMM interfaces for testing
extern "C" {
dnnl_status_t dnnl_gemm_bf16bf16f32(char transa, char transb, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const bfloat16_t *A,
        dnnl_dim_t lda, const bfloat16_t *B, dnnl_dim_t ldb, float beta,
        float *C, dnnl_dim_t ldc);
}

// Declare packed GEMM interfaces for testing
namespace dnnl {
namespace impl {
namespace cpu {

// Get pack-size functions.
extern dnnl_status_t sgemm_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

extern dnnl_status_t gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

extern dnnl_status_t gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

extern dnnl_status_t gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, size_t *size,
        bool *pack = nullptr);

// Pack functions.
extern dnnl_status_t sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const dim_t *lda, const dim_t *ldb, const float *src, float *dst);

extern dnnl_status_t gemm_bf16bf16f32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb,
        const bfloat16_t *src, bfloat16_t *dst);

extern dnnl_status_t gemm_s8u8s32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, const void *src,
        void *dst);

extern dnnl_status_t gemm_s8s8s32_pack(const char *identifier,
        const char *transa, const char *transb, const dim_t *M, const dim_t *N,
        const dim_t *K, const dim_t *lda, const dim_t *ldb, const void *src,
        void *dst);

// Compute functions.
extern dnnl_status_t sgemm_compute(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *A,
        const dim_t *lda, const float *B, const dim_t *ldb, const float *beta,
        float *C, const dim_t *ldc);

extern dnnl_status_t gemm_bf16bf16f32_compute(const char *transa,
        const char *transb, const dim_t *M, const dim_t *N, const dim_t *K,
        const bfloat16_t *A, const dim_t *lda, const bfloat16_t *B,
        const dim_t *ldb, const float *beta, float *C, const dim_t *ldc);

extern dnnl_status_t gemm_s8u8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const int8_t *A, const dim_t *lda, const uint8_t *B,
        const dim_t *ldb, const float *beta, int32_t *C, const dim_t *ldc,
        const int32_t *co);

extern dnnl_status_t gemm_s8s8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const dim_t *M, const dim_t *N,
        const dim_t *K, const int8_t *A, const dim_t *lda, const int8_t *B,
        const dim_t *ldb, const float *beta, int32_t *C, const dim_t *ldc,
        const int32_t *co);

} // namespace cpu
} // namespace impl
} // namespace dnnl

namespace dnnl {

struct test_igemm_params {
    char offsetc;
    bool nonzero_oa;
    bool nonzero_ob;
    bool nonzero_oc;

    int8_t oa() const { return (int8_t)(nonzero_oa ? 4 : 0); }
    int8_t ob() const { return (int8_t)(nonzero_ob ? 3 : 0); }
};

struct test_pack_params {
    bool pack_a;
    bool pack_b;
};

struct gemm_offset {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t co;
};

struct test_params {
    char transA;
    char transB;
    int64_t M;
    int64_t N;
    int64_t K;
    float alpha;
    float beta;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    test_igemm_params igemm_params;
    test_pack_params pack_params;
    bool expect_to_fail;
    dnnl_status_t expected_status;

    gemm_offset off;

    bool tr_a() const { return transA == 'T' || transA == 't'; }
    bool tr_b() const { return transB == 'T' || transB == 't'; }
    int64_t sizeC() const { return M * ldc; }

    bool oc_is_R() const {
        auto c = igemm_params.offsetc;
        return c == 'R' || c == 'r';
    }
    bool oc_is_C() const {
        auto c = igemm_params.offsetc;
        return c == 'C' || c == 'c';
    }
    int64_t size_oc() const { return oc_is_R() ? N : oc_is_C() ? M : 1; }
};

template <typename... TArgs>
inline test_params make_test_params_with_offset(
        const gemm_offset &off, TArgs &&... args) {
    test_params params {std::forward<TArgs>(args)...};
    params.off = off;
    return params;
}

template <typename... TArgs>
inline test_params make_test_params_pack(
        const test_pack_params &pack_params, TArgs &&... args) {
    test_params params {std::forward<TArgs>(args)...};
    params.pack_params = pack_params;
    return params;
}
/* Test implementation description.
 *
 * To reduce the time spent in GEMM validation the test matrices A, B, and C
 * are generated from sub-matrices (A', B', and C') of smaller size:
 * - A(M, K) <-> A'(M_test, K)
 * - B(K, N) <-> B'(K, N_test)
 * - C(M, N) <-> C'(M_test, N_test)
 *
 * The matrices A', B', and C' are generated randomly. Then:
 * - A(m, k) := A'(mapper_m[m], k),
 * - B(k, n) := B'(k, mapper_n[n]),
 * - C(m, n) := C'(mapper_m[m], mapper_n[n]);
 *
 * Here `mapper_x[]` is surjection of {0, ..., X-1} onto {0, ..., X_test-1}.
 * For simplicity mapper_x[x] = x, for x in {0, ..., X_test-1}.
 *
 * This technique allows reducing the complexity of the validation code from
 * O(M*N*K) to O(M_test * N_test * K).
 *
 * X_test := min(X, X_test_max), where X_test_max is prime number around 50.
 *
 * To make the test robust the surjective functions mapper_m and mapper_n
 * should randomly map the elements {X_test, ..., X-1} onto {0, ..., X_test-1}.
 *
 * The validation itself looks as follows:
 * 0.  Prepare mapper_m and mapper_n
 * 1.a Generate random matrices A', B', C'
 * 1.b Prepare matrices A, B, C based on A', B', and C' respectively
 * 2.  Compute C_calc := Op(M, N, K, A, B, C)
 * 3.  Compute C'_ref := Op_REF(M_test, N_test, K, A', B', C')
 * 4.  Expand C'_ref to C_ref, by applying mapper_m and mapper_n
 * 5.  Compare C_calc and C_ref
 */

const int M_test_max = 47;
const int N_test_max = 53;

/** Mapper:
 * a surjective function from {0, ..., dim-1} onto {0, ..., dim_test-1}.
 */
struct mapper_t {
    mapper_t(int64_t dim, int64_t dim_test_max, int64_t gen = 7,
            int64_t gen_start = 13)
        : dim_(dim)
        , dim_test_((std::min)(dim, dim_test_max))
        , gen_(gen)
        , gen_start_(gen_start)
        , mapper_(dim) {
        for (int64_t d = 0; d < dim_test_; ++d)
            mapper_[d] = d;
        for (int64_t g = gen_start_ % dim_test_, d = dim_test_; d < dim_; ++d) {
            mapper_[d] = mapper_[g];
            g = g * gen_ % dim_test_;
        }
    }

    int64_t dim() const { return dim_; }
    int64_t dim_test() const { return dim_test_; }
    int64_t operator[](int64_t d) const { return mapper_[d]; }

private:
    const int64_t dim_;
    const int64_t dim_test_;
    const int64_t gen_, gen_start_;
    std::vector<int64_t> mapper_;
};

enum class layout_t { ROW_MAJOR, COL_MAJOR };

/** Prepares matrix A or B according to the dimension mapper.
 * The K dimension is always assumed to be columns, hence:
 * - A layout = A_is_transposed ? ROW_MAJOR : COL_MAJOR
 * - B layout = B_is_transposed ? COL_MAJOR : ROW_MAJOR
 */
template <typename data_t>
void prepare_matrix(const test_memory &M_mem, int64_t off_beg, layout_t layout,
        int64_t R, int64_t C, int64_t LD, const mapper_t &mapper) {
    auto M = map_memory<data_t>(M_mem);
    auto dt = data_traits<data_t>::data_type;
    bool is_fp = (false || dt == memory::data_type::f16
            || dt == memory::data_type::bf16 || dt == memory::data_type::f32);
    const data_t mean = (data_t)(is_fp ? 1.f : 4);
    const data_t var = (data_t)(is_fp ? 2e-1f : 3);

    ASSERT_EQ(R, mapper.dim());
    const int R_test = mapper.dim_test();

    if (layout == layout_t::COL_MAJOR) {
        dnnl::impl::parallel_nd(C, R_test, [&](int64_t c, int64_t r) {
            const int64_t off = c * LD + r;
            M[off_beg + off] = set_value<data_t>(off, mean, var, 1.);
        });
        if (R > R_test) {
            const int64_t R_rest = R - R_test;
            dnnl::impl::parallel_nd(C, R_rest, [&](int64_t c, int64_t r_) {
                const int64_t r = R_test + r_;
                const int64_t off = c * LD + r;
                const int64_t off0 = c * LD + mapper[r];
                M[off_beg + off] = M[off_beg + off0];
            });
        }
    } else {
        dnnl::impl::parallel_nd(R_test, C, [&](int64_t r, int64_t c) {
            const int64_t off = r * LD + c;
            M[off_beg + off] = set_value<data_t>(off, mean, var, 1.);
        });
        if (R > R_test) {
            const int64_t R_rest = R - R_test;
            dnnl::impl::parallel_nd(R_rest, C, [&](int64_t r_, int64_t c) {
                const int64_t r = R_test + r_;
                const int64_t off = r * LD + c;
                const int64_t off0 = mapper[r] * LD + c;
                M[off_beg + off] = M[off_beg + off0];
            });
        }
    }
}

/** Extends columns of the matrix M according to the mapper_c */
template <typename data_t>
void extend_matrix_cols(const test_memory &M_mem, int64_t off, int64_t R,
        int64_t C, int64_t LD, const mapper_t &mapper_c) {
    auto M = map_memory<data_t>(M_mem);
    ASSERT_EQ(C, mapper_c.dim());
    const int64_t C_test = mapper_c.dim_test();
    if (C_test == C) return;

    dnnl::impl::parallel_nd(R, C - C_test, [&](int64_t r, int64_t c_) {
        const int64_t c = C_test + c_;
        const int64_t c0 = mapper_c[c];
        M[off + r * LD + c] = M[off + r * LD + c0];
    });
}

/** Extends rows of the matrix M according to the mapper_r */
template <typename data_t>
void extend_matrix_rows(const test_memory &M_mem, int64_t off, int64_t R,
        int64_t C, int64_t LD, const mapper_t &mapper_r) {
    auto M = map_memory<data_t>(M_mem);
    ASSERT_EQ(R, mapper_r.dim());
    const int64_t R_test = mapper_r.dim_test();
    if (R_test == R) return;

    dnnl::impl::parallel_nd(R - R_test, [&](int64_t r_) {
        const int64_t r = R_test + r_;
        const int64_t r0 = mapper_r[r];
        for (int64_t c = 0; c < C; ++c)
            M[off + r * LD + c] = M[off + r0 * LD + c];
    });
}

/** Extends matrix M according to the mapper_r and mapper_c */
template <typename data_t>
void extend_matrix(const test_memory &M_mem, int64_t off, int64_t R, int64_t C,
        int64_t LD, const mapper_t &mapper_r, const mapper_t &mapper_c) {
    ASSERT_EQ(R, mapper_r.dim());
    ASSERT_EQ(C, mapper_c.dim());
    extend_matrix_rows<data_t>(M_mem, off, R, C, LD, mapper_r);
    extend_matrix_cols<data_t>(M_mem, off, R, C, LD, mapper_c);
}

template <typename a_dt, typename b_dt, typename c_dt>
struct ref_gemm {
    static void call(const test_params &p, int64_t M, int64_t N,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &) {
        auto a = map_memory<a_dt>(a_mem);
        auto b = map_memory<b_dt>(b_mem);
        auto c = map_memory<c_dt>(c_mem);

        const bool tr_a = p.transA && (p.transA == 'T' || p.transA == 't');
        const bool tr_b = p.transB && (p.transB == 'T' || p.transB == 't');

        auto pa = [&](int64_t i, int64_t j) {
            return a[p.off.a + i * p.lda + j];
        };
        auto pb = [&](int64_t i, int64_t j) {
            return b[p.off.b + i * p.ldb + j];
        };
        auto pc = [&](int64_t i, int64_t j) -> c_dt & {
            return c[p.off.c + i * p.ldc + j];
        };

        dnnl::impl::parallel_nd(M, N, [&](int64_t im, int64_t in) {
            c_dt c_elem = (p.beta == 0.) ? 0. : pc(im, in) * p.beta;

            for (int64_t ik = 0; ik < p.K; ik++) {
                const a_dt a_elem = tr_a ? pa(ik, im) : pa(im, ik);
                const b_dt b_elem = tr_b ? pb(in, ik) : pb(ik, in);
                c_elem += p.alpha * a_elem * b_elem;
            }
            pc(im, in) = c_elem;
        });
    }
};

template <typename a_dt, typename b_dt>
struct ref_gemm<a_dt, b_dt, int32_t> {
    static void call(const test_params &p, int64_t M, int64_t N,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        auto A = map_memory<a_dt>(a_mem);
        auto B = map_memory<b_dt>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        const bool tr_a = p.transA && (p.transA == 'T' || p.transA == 't');
        const bool tr_b = p.transB && (p.transB == 'T' || p.transB == 't');
        bool OCisR = (p.igemm_params.offsetc == 'R'
                || p.igemm_params.offsetc == 'r');
        bool OCisC = (p.igemm_params.offsetc == 'C'
                || p.igemm_params.offsetc == 'c');

        auto pa = [&](int64_t i, int64_t j) {
            return (double)A[p.off.a + i * p.lda + j];
        };
        auto pb = [&](int64_t i, int64_t j) {
            return (double)B[p.off.b + i * p.ldb + j];
        };
        auto pc = [&](int64_t i, int64_t j) -> int32_t & {
            return C[p.off.c + i * p.ldc + j];
        };

        int8_t oa = p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();

        dnnl::impl::parallel_nd(M, N, [&](int64_t m, int64_t n) {
            double c_elem = 0;
            for (int64_t k = 0; k < p.K; k++) {
                const double a_elem = (tr_a ? pa(k, m) : pa(m, k)) - oa;
                const double b_elem = (tr_b ? pb(n, k) : pb(k, n)) - ob;
                c_elem += a_elem * b_elem;
            }

            double coffset = OCisR ? oc[n] : OCisC ? oc[m] : oc[0];
            double val = (p.beta == 0.f ? 0. : p.beta * (double)pc(m, n))
                    + p.alpha * c_elem + coffset;
            pc(m, n) = static_cast<int32_t>(
                    nearbyint(saturate<int32_t, double>(val)));
        });
    }
};

template <typename a_dt, typename c_dt>
void compare(const test_params &p, const test_memory &c_mem,
        const test_memory &c_ref_mem) {
    using data_type = memory::data_type;
    auto c = map_memory<c_dt>(c_mem);
    auto c_ref = map_memory<c_dt>(c_ref_mem);
    dnnl::impl::parallel_nd(p.M, p.ldc, [&](int64_t i, int64_t j) {
        if (is_current_test_failed()) return;

        c_dt ref = c_ref[p.off.c + i * p.ldc + j];
        c_dt got = c[p.off.c + i * p.ldc + j];
        c_dt diff = got - ref;

        if (data_traits<a_dt>::data_type == data_type::f16) {
            const float eps = 1e-3 * p.K;
            float e = (std::abs(ref) > eps) ? diff / ref : float(diff);
            ASSERT_NEAR(e, 0.0, eps) << "Row: " << i << " Col: " << j;
        } else if (data_traits<a_dt>::data_type == data_type::bf16) {
            const float eps = 1e-2 * p.K;
            float e = (std::abs(ref) > eps) ? diff / ref : float(diff);
            ASSERT_NEAR(e, 0.0, eps) << "Row: " << i << " Col: " << j;
        } else if (data_traits<a_dt>::data_type == data_type::f32) {
            c_dt e = (std::abs(ref) > 1e-4) ? c_dt(diff / ref) : diff;
            ASSERT_NEAR(e, 0.0, 1e-4) << "Row: " << i << " Col: " << j;
        } else {
            // igemm
            c_dt eps = 0;
            if (p.alpha == 1.0f) {
                eps = 1;
            } else if (data_traits<a_dt>::data_type == data_type::u8) {
                eps = p.K / 700 + 1;
            } else if (data_traits<a_dt>::data_type == data_type::s8) {
                eps = p.K / 350 + 1;
            }
            ASSERT_NEAR(diff, 0, eps) << "Row: " << i << " Col: " << j;
        }
    });
}

inline void get_matrix_size(
        const test_params &p, size_t &sizeA, size_t &sizeB, size_t &sizeC) {
    const bool tr_a = (p.transA == 'T' || p.transA == 't');
    const bool tr_b = (p.transB == 'T' || p.transB == 't');
    sizeA = tr_a ? p.lda * p.K : p.lda * p.M,
    sizeB = tr_b ? p.ldb * p.N : p.ldb * p.K, sizeC = p.ldc * p.M;
}

template <typename T>
inline test_memory get_matrix_memory(
        memory::dim n, memory::dim off, engine &eng) {
    auto d = create_md(
            {n + off}, data_traits<T>::data_type, memory::format_tag::x);
    return test_memory(d, eng);
}

template <typename a_dt, typename b_dt, typename c_dt>
void fill_matrices(const test_params &p, const mapper_t &mapper_m,
        const mapper_t &mapper_n, const test_memory &a_mem,
        const test_memory &b_mem, const test_memory &c_mem,
        const test_memory &c_ref_mem, const test_memory &oc_mem) {
    prepare_matrix<a_dt>(a_mem, p.off.a,
            p.tr_a() ? layout_t::COL_MAJOR : layout_t::ROW_MAJOR, p.M, p.K,
            p.lda, mapper_m);
    prepare_matrix<b_dt>(b_mem, p.off.b,
            p.tr_b() ? layout_t::ROW_MAJOR : layout_t::COL_MAJOR, p.N, p.K,
            p.ldb, mapper_n);

    fill_data<c_dt>(p.off.c + p.sizeC(), c_mem.get());
    extend_matrix<c_dt>(c_mem, p.off.c, p.M, p.N, p.ldc, mapper_m, mapper_n);
    {
        auto C = map_memory<c_dt>(c_mem);
        auto C_ref = map_memory<c_dt>(c_ref_mem);
        dnnl::impl::parallel_nd(p.sizeC(),
                [&](int64_t i) { C_ref[p.off.c + i] = C[p.off.c + i]; });
    }

    if (oc_mem.get_size() == 0) return;

    if (p.igemm_params.nonzero_oc) {
        fill_data<c_dt>(p.size_oc(), oc_mem.get(), (c_dt)1, (c_dt)0);
        if (p.oc_is_R()) {
            extend_matrix_cols<c_dt>(oc_mem, 0, 1, p.N, p.N, mapper_n);
        } else if (p.oc_is_C()) {
            extend_matrix_rows<c_dt>(oc_mem, 0, p.M, 1, 1, mapper_m);
        }
    } else {
        auto oc = map_memory<c_dt>(oc_mem);
        for (int64_t i = 0; i < p.size_oc(); i++)
            oc[i] = 0;
    }
}

template <typename a_dt, typename b_dt, typename c_dt>
struct dnnl_gemm {
    static dnnl_status_t call(test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem) {
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<float16_t, float16_t, float16_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = get_test_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_hgemm(q, p.transA, p.transB, p.M, p.N, p.K,
                    p.alpha, a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    b_mem.get().get_ocl_mem_object(), p.off.b, p.ldb, p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc);
            s.wait();
            return status;
        }
#endif
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<float, float, float> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        std::vector<float> a_pack_buf, b_pack_buf;
        float *A = map_memory<float>(b_mem), *a_eff = A;
        float *B = map_memory<float>(a_mem), *b_eff = B;
        float *C = map_memory<float>(c_mem);

        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = sgemm_pack_get_size("A", &trans_a, &trans_b, &m, &n, &k,
                    &lda, &ldb, &a_sz, &pack_a);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz / sizeof(float));
                a_eff = a_pack_buf.data();

                status = sgemm_pack("A", &trans_a, &trans_b, &m, &n, &k, &lda,
                        &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;
            status = sgemm_pack_get_size("B", &trans_a, &trans_b, &m, &n, &k,
                    &lda, &ldb, &b_sz, &pack_b);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz / sizeof(float));
                b_eff = b_pack_buf.data();

                status = sgemm_pack("B", &trans_a, &trans_b, &m, &n, &k, &lda,
                        &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = sgemm_compute(&trans_a, &trans_b, &m, &n, &k, a_eff, &lda,
                b_eff, &ldb, &p.beta, C, &ldc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem);

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = a_mem.get().get_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_sgemm(q, p.transA, p.transB, p.M, p.N, p.K,
                    p.alpha, a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    b_mem.get().get_ocl_mem_object(), p.off.b, p.ldb, p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc);
            s.wait();
            return status;
        }
#endif
        auto A = map_memory<float>(a_mem);
        auto B = map_memory<float>(b_mem);
        auto C = map_memory<float>(c_mem);

        return dnnl_sgemm(p.transA, p.transB, p.M, p.N, p.K, p.alpha, A, p.lda,
                B, p.ldb, p.beta, C, p.ldc);
    }
};

template <>
struct dnnl_gemm<int8_t, int8_t, int32_t> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);
        assert(p.igemm_params.oa() == 0);
        assert(p.igemm_params.ob() == 0);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        int8_t *A = map_memory<int8_t>(b_mem), *a_eff = A;
        int8_t *B = map_memory<int8_t>(a_mem), *b_eff = B;

        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        char offset_c = '\0';
        switch (p.igemm_params.offsetc) {
            case 'R': offset_c = 'C'; break;
            case 'r': offset_c = 'c'; break;
            case 'C': offset_c = 'R'; break;
            case 'c': offset_c = 'r'; break;
            default: offset_c = p.igemm_params.offsetc;
        }

        std::vector<int8_t> a_pack_buf;
        std::vector<int8_t> b_pack_buf;
        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_s8s8s32_pack_get_size(
                    "A", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &a_sz);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz);
                a_eff = a_pack_buf.data();

                status = gemm_s8s8s32_pack("A", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;

            status = gemm_s8s8s32_pack_get_size(
                    "B", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &b_sz);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz);
                b_eff = b_pack_buf.data();

                status = gemm_s8s8s32_pack("B", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_s8s8s32_compute(&trans_a, &trans_b, &offset_c, &m, &n, &k,
                a_eff, &lda, b_eff, &ldb, &p.beta, C, &ldc, oc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = get_test_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_s8s8s32(q, p.transA, p.transB,
                    p.igemm_params.offsetc, p.M, p.N, p.K, p.alpha,
                    a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    p.igemm_params.oa(), b_mem.get().get_ocl_mem_object(),
                    p.off.b, p.ldb, p.igemm_params.ob(), p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc,
                    oc_mem.get().get_ocl_mem_object(), p.off.co);
            s.wait();
            return status;
        }
#endif
        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem, oc_mem);

        auto A = map_memory<int8_t>(a_mem);
        auto B = map_memory<int8_t>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);
        int8_t oa = p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();
        return dnnl_gemm_s8s8s32(p.transA, p.transB, p.igemm_params.offsetc,
                p.M, p.N, p.K, p.alpha, A, p.lda, oa, B, p.ldb, ob, p.beta, C,
                p.ldc, oc);
    }
};

template <>
struct dnnl_gemm<int8_t, uint8_t, int32_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = get_test_engine();
            stream s(eng);
            cl_command_queue q2 = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_s8u8s32(q2, p.transA, p.transB,
                    p.igemm_params.offsetc, p.M, p.N, p.K, p.alpha,
                    a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    p.igemm_params.oa(), b_mem.get().get_ocl_mem_object(),
                    p.off.b, p.ldb, (uint8_t)p.igemm_params.ob(), p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc,
                    oc_mem.get().get_ocl_mem_object(), p.off.co);
            s.wait();
            return status;
        }
#endif
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<uint8_t, uint8_t, int32_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = get_test_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_u8u8s32(q, p.transA, p.transB,
                    p.igemm_params.offsetc, p.M, p.N, p.K, p.alpha,
                    a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    (uint8_t)p.igemm_params.oa(),
                    b_mem.get().get_ocl_mem_object(), p.off.b, p.ldb,
                    (uint8_t)p.igemm_params.ob(), p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc,
                    oc_mem.get().get_ocl_mem_object(), p.off.co);
            s.wait();
            return status;
        }
#endif
        throw error(dnnl_runtime_error, "unknown gemm");
    }
};

template <>
struct dnnl_gemm<uint8_t, int8_t, int32_t> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem, const test_memory &oc_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);
        assert(p.igemm_params.oa() == 0);
        assert(p.igemm_params.ob() == 0);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        int8_t *A = map_memory<int8_t>(b_mem), *a_eff = A;
        uint8_t *B = map_memory<uint8_t>(a_mem), *b_eff = B;

        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);

        char offset_c = '\0';
        switch (p.igemm_params.offsetc) {
            case 'R': offset_c = 'C'; break;
            case 'r': offset_c = 'c'; break;
            case 'C': offset_c = 'R'; break;
            case 'c': offset_c = 'r'; break;
            default: offset_c = p.igemm_params.offsetc;
        }

        std::vector<int8_t> a_pack_buf;
        std::vector<uint8_t> b_pack_buf;
        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_s8u8s32_pack_get_size(
                    "A", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &a_sz);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz);
                a_eff = a_pack_buf.data();

                status = gemm_s8u8s32_pack("A", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;

            status = gemm_s8u8s32_pack_get_size(
                    "B", &trans_a, &trans_b, &m, &n, &k, &lda, &ldb, &b_sz);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz);
                b_eff = b_pack_buf.data();

                status = gemm_s8u8s32_pack("B", &trans_a, &trans_b, &m, &n, &k,
                        &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_s8u8s32_compute(&trans_a, &trans_b, &offset_c, &m, &n, &k,
                a_eff, &lda, b_eff, &ldb, &p.beta, C, &ldc, oc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &oc_mem) {
        assert(p.igemm_params.oa() >= 0);

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = get_test_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_u8s8s32(q, p.transA, p.transB,
                    p.igemm_params.offsetc, p.M, p.N, p.K, p.alpha,
                    a_mem.get().get_ocl_mem_object(), p.off.a, p.lda,
                    p.igemm_params.oa(), b_mem.get().get_ocl_mem_object(),
                    p.off.b, p.ldb, p.igemm_params.ob(), p.beta,
                    c_mem.get().get_ocl_mem_object(), p.off.c, p.ldc,
                    oc_mem.get().get_ocl_mem_object(), p.off.co);
            s.wait();
            return status;
        }
#endif

        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem, oc_mem);

        auto A = map_memory<uint8_t>(a_mem);
        auto B = map_memory<int8_t>(b_mem);
        auto C = map_memory<int32_t>(c_mem);
        auto oc = map_memory<int32_t>(oc_mem);
        uint8_t oa = (uint8_t)p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();

        return dnnl_gemm_u8s8s32(p.transA, p.transB, p.igemm_params.offsetc,
                p.M, p.N, p.K, p.alpha, A, p.lda, oa, B, p.ldb, ob, p.beta, C,
                p.ldc, oc);
    }
};

template <>
struct dnnl_gemm<float16_t, float16_t, float> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = a_mem.get().get_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_f16f16f32(q, p.transA, p.transB, p.M,
                    p.N, p.K, p.alpha, a_mem.get().get_ocl_mem_object(),
                    p.off.a, p.lda, b_mem.get().get_ocl_mem_object(), p.off.b,
                    p.ldb, p.beta, c_mem.get().get_ocl_mem_object(), p.off.c,
                    p.ldc);
            s.wait();
            return status;
        }
#endif
        return dnnl_unimplemented;
    }
};

template <>
struct dnnl_gemm<bfloat16_t, bfloat16_t, float> {
    static dnnl_status_t call_packed(const test_params &p,
            const test_memory &a_mem, const test_memory &b_mem,
            const test_memory &c_mem) {
        /* Alas, the internal API still uses Fortran notation.
         * So in addition to the changes for pack API, we also need to take
         * care of conversions and layouts */

        using namespace dnnl::impl::cpu;

        assert(p.alpha == 1.f);

        /* Prepare for Fortran style, hence A <-> B */
        char trans_a = p.transB, trans_b = p.transA;

        int64_t m = p.N, n = p.M, k = p.K;
        int64_t lda = p.ldb, ldb = p.lda, ldc = p.ldc;

        std::vector<bfloat16_t> a_pack_buf, b_pack_buf;
        bfloat16_t *A = map_memory<bfloat16_t>(b_mem), *a_eff = A;
        bfloat16_t *B = map_memory<bfloat16_t>(a_mem), *b_eff = B;
        float *C = map_memory<float>(c_mem);

        bool pack_a = p.pack_params.pack_b;
        bool pack_b = p.pack_params.pack_a;

        dnnl_status_t status = dnnl_success;

        if (pack_a) {
            size_t a_sz;
            status = gemm_bf16bf16f32_pack_get_size("A", &trans_a, &trans_b, &m,
                    &n, &k, &lda, &ldb, &a_sz, &pack_a);
            if (status != dnnl_success) return status;

            if (pack_a) {
                a_pack_buf.resize(a_sz / sizeof(*a_eff));
                a_eff = a_pack_buf.data();

                status = gemm_bf16bf16f32_pack("A", &trans_a, &trans_b, &m, &n,
                        &k, &lda, &ldb, A, a_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_b) {
            size_t b_sz;
            status = gemm_bf16bf16f32_pack_get_size("B", &trans_a, &trans_b, &m,
                    &n, &k, &lda, &ldb, &b_sz, &pack_b);
            if (status != dnnl_success) return status;

            if (pack_b) {
                b_pack_buf.resize(b_sz / sizeof(*b_eff));
                b_eff = b_pack_buf.data();

                status = gemm_bf16bf16f32_pack("B", &trans_a, &trans_b, &m, &n,
                        &k, &lda, &ldb, B, b_eff);
                if (status != dnnl_success) return status;
            }
        }

        if (pack_a) trans_a = 'P';
        if (pack_b) trans_b = 'P';

        status = gemm_bf16bf16f32_compute(&trans_a, &trans_b, &m, &n, &k, a_eff,
                &lda, b_eff, &ldb, &p.beta, C, &ldc);

        return status;
    }

    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = a_mem.get().get_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_bf16bf16f32(q, p.transA, p.transB, p.M,
                    p.N, p.K, p.alpha, a_mem.get().get_ocl_mem_object(),
                    p.off.a, p.lda, b_mem.get().get_ocl_mem_object(), p.off.b,
                    p.ldb, p.beta, c_mem.get().get_ocl_mem_object(), p.off.c,
                    p.ldc);
            s.wait();
            return status;
        }
#endif
        if (p.pack_params.pack_a || p.pack_params.pack_b)
            return call_packed(p, a_mem, b_mem, c_mem);

        auto A = map_memory<bfloat16_t>(a_mem);
        auto B = map_memory<bfloat16_t>(b_mem);
        auto C = map_memory<float>(c_mem);
        return dnnl_gemm_bf16bf16f32(p.transA, p.transB, p.M, p.N, p.K, p.alpha,
                A, p.lda, B, p.ldb, p.beta, C, p.ldc);
    }
};

template <>
struct dnnl_gemm<bfloat16_t, bfloat16_t, bfloat16_t> {
    static dnnl_status_t call(const test_params &p, const test_memory &a_mem,
            const test_memory &b_mem, const test_memory &c_mem,
            const test_memory &) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (get_test_engine_kind() == engine::kind::gpu) {
            engine eng = a_mem.get().get_engine();
            stream s(eng);
            cl_command_queue q = s.get_ocl_command_queue();
            auto status = dnnl_ocl_gemm_bf16bf16bf16(q, p.transA, p.transB, p.M,
                    p.N, p.K, p.alpha, a_mem.get().get_ocl_mem_object(),
                    p.off.a, p.lda, b_mem.get().get_ocl_mem_object(), p.off.b,
                    p.ldb, p.beta, c_mem.get().get_ocl_mem_object(), p.off.c,
                    p.ldc);
            s.wait();
            return status;
        }
#endif
        return dnnl_unimplemented;
    }
};

template <typename a_dt, typename b_dt, typename c_dt>
struct run_test_gemm {
    static void call(const test_params &p) {
        if (p.expect_to_fail) {
            engine eng = get_test_engine();
            test_memory zero_mem({}, eng);
            auto status = dnnl_gemm<a_dt, b_dt, c_dt>::call(
                    p, zero_mem, zero_mem, zero_mem, zero_mem);
            if (status != dnnl_success)
                throw error(status, "oneDNN gemm returned error");
            return;
        }

        size_t sizeA, sizeB, sizeC;
        get_matrix_size(p, sizeA, sizeB, sizeC);

        engine eng = get_test_engine();
        test_memory a_mem = get_matrix_memory<a_dt>(sizeA, p.off.a, eng);
        test_memory b_mem = get_matrix_memory<b_dt>(sizeB, p.off.b, eng);
        test_memory c_mem = get_matrix_memory<c_dt>(sizeC, p.off.c, eng);
        test_memory c_ref_mem = get_matrix_memory<c_dt>(sizeC, p.off.c, eng);
        test_memory oc_mem
                = get_matrix_memory<c_dt>(p.size_oc(), p.off.co, eng);

        mapper_t mapper_m(p.M, M_test_max), mapper_n(p.N, N_test_max);
        const int64_t M_test = mapper_m.dim_test();
        const int64_t N_test = mapper_n.dim_test();

        fill_matrices<a_dt, b_dt, c_dt>(
                p, mapper_m, mapper_n, a_mem, b_mem, c_mem, c_ref_mem, oc_mem);

        auto status = dnnl_gemm<a_dt, b_dt, c_dt>::call(
                p, a_mem, b_mem, c_mem, oc_mem);

        if (status == dnnl_success) {
            ref_gemm<a_dt, b_dt, c_dt>::call(
                    p, M_test, N_test, a_mem, b_mem, c_ref_mem, oc_mem);
            extend_matrix<c_dt>(
                    c_ref_mem, p.off.c, p.M, p.N, p.ldc, mapper_m, mapper_n);
            compare<a_dt, c_dt>(p, c_mem, c_ref_mem);
        }

        if (status != dnnl_success)
            throw error(status, "oneDNN gemm returned error");
    }
};

template <typename a_dt, typename b_dt, typename c_dt>
class gemm_test_common : public ::testing::TestWithParam<test_params> {
protected:
    virtual void SetUp() {
        const auto &p = ::testing::TestWithParam<test_params>::GetParam();

        bool zero_off = (p.off.a == 0 && p.off.b == 0 && p.off.c == 0);
        SKIP_IF(!zero_off && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support non-zero offsets.");

        SKIP_IF(unsupported_data_type(data_traits<a_dt>::data_type),
                "Engine does not support this data type.");

        bool pack = (p.pack_params.pack_a || p.pack_params.pack_b);
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu && pack,
                "GPU does not support packed GEMM.");
#ifndef DNNL_X86_64
        SKIP_IF(pack, "Packed GEMM does not support non-x86 CPUs.");
#endif
        SKIP_IF((p.alpha != 1.f || p.igemm_params.oa() != 0
                        || p.igemm_params.ob() != 0)
                        && pack,
                "Packed GEMM doesn't support alpha or non-zero offset{A,B}.");
        SKIP_IF(data_traits<b_dt>::data_type == memory::data_type::u8
                        && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support s8u8s32 and u8u8s32 GEMM.");
        SKIP_IF(data_traits<c_dt>::data_type == memory::data_type::bf16
                        && get_test_engine_kind() == engine::kind::cpu,
                "CPU does not support bf16bf16bf16 GEMM.");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }
    void Test() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
        struct scoped_threadpool {
            scoped_threadpool() {
                impl::threadpool_utils::activate_threadpool(
                        impl::threadpool_utils::get_active_threadpool());
            }
            ~scoped_threadpool() {
                impl::threadpool_utils::deactivate_threadpool();
            }
        };
        scoped_threadpool stp;
#endif
        const auto &p = ::testing::TestWithParam<test_params>::GetParam();
        run_test_gemm<a_dt, b_dt, c_dt>::call(p);
    }
};
} // namespace dnnl
#endif
