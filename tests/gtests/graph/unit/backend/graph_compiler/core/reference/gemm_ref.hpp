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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_GEMM_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_GEMM_REF_HPP

#include <stdlib.h>
#include <time.h>
#include <vector>
#include <test_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct gemm_params {
    bool transA;
    bool transB;
    int64_t M;
    int64_t N;
    int64_t K;
    float alpha;
    float beta;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
};

template <typename a_t, typename b_t, typename c_t>
static void ref_gemm(const gemm_params &param, const a_t *A, const b_t *B,
        c_t *C, const c_t *bias = nullptr) {
    const bool transA = param.transA;
    const bool transB = param.transB;

    const auto lda = transA ? param.M : param.K;
    const auto ldb = transB ? param.K : param.N;
    const auto ldc = param.N;
    TEST_ASSERT((param.lda == lda),
            "lda is " << param.lda << ", but expected to be " << lda);
    TEST_ASSERT((param.ldb == ldb),
            "ldb is " << param.ldb << ", but expected to be " << ldb);
    TEST_ASSERT((param.ldc == ldc),
            "ldc is " << param.ldc << ", but expected to be " << ldc);

    auto pa = [&](int i, int j) { return A[i * param.lda + j]; };
    auto pb = [&](int i, int j) { return B[i * param.ldb + j]; };
    auto pc = [&](int i, int j) -> c_t & { return C[i * param.ldc + j]; };
    auto pbias = [&](int i) {
        if (bias)
            return bias[i];
        else
            return (c_t)0.0;
    };

    test_utils::parallel_nd(param.M, param.N, [&](int64_t im, int64_t in) {
        c_t c_elem = (param.beta == 0.) ? 0. : pc(im, in) * param.beta;

        for (int ik = 0; ik < param.K; ik++) {
            const a_t a_elem = transA ? pa(ik, im) : pa(im, ik);
            const b_t b_elem = transB ? pb(in, ik) : pb(ik, in);
            c_elem += param.alpha * a_elem * b_elem;
        }
        pc(im, in) = c_elem + pbias(in);
    });
}

template <typename T>
T MK2MKmk(T &input, int M, int K, int m, int k, int origin_m = 0,
        int origin_k = 0) {
    origin_k = origin_k ? origin_k : K * k;
    origin_m = origin_m ? origin_m : M * m;
    T output(test_utils::product({M, K, m, k}));
    int dim3 = K * m * k, dim2 = m * k, dim1 = k;
    utils::parallel_for(0, M, 1, [&](int64_t m_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto m_i = 0; m_i < m; ++m_i) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    if ((m_o * m + m_i < origin_m)
                            && (k_o * k + k_i < origin_k)) {
                        output[m_o * dim3 + k_o * dim2 + m_i * dim1 + k_i]
                                = input[(m_o * m + m_i) * origin_k + k_o * k
                                        + k_i];
                    } else {
                        output[m_o * dim3 + k_o * dim2 + m_i * dim1 + k_i] = 0;
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T MKmk2MK(T &input, int M, int K, int m, int k, int origin_m = 0,
        int origin_k = 0) {
    origin_k = origin_k ? origin_k : K * k;
    origin_m = origin_m ? origin_m : M * m;
    T output(origin_m * origin_k);
    int dim3 = K * m * k, dim2 = m * k, dim1 = k;
    utils::parallel_for(0, M, 1, [&](int64_t m_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto m_i = 0; m_i < m; ++m_i) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    if ((m_o * m + m_i) < origin_m
                            && (k_o * k + k_i) < origin_k) {
                        output[(m_o * m + m_i) * origin_k + k_o * k + k_i]
                                = input[m_o * dim3 + k_o * dim2 + m_i * dim1
                                        + k_i];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T batch_MKmk2MK(T &input, int batch, int M, int K, int m, int k) {
    T output(input.size());
    int dim4 = M * K * m * k, dim3 = K * m * k, dim2 = m * k, dim1 = k;
    utils::parallel_for(0, batch, 1, [&](int64_t b) {
        for (auto m_o = 0; m_o < M; ++m_o) {
            for (auto k_o = 0; k_o < K; ++k_o) {
                for (auto m_i = 0; m_i < m; ++m_i) {
                    for (auto k_i = 0; k_i < k; ++k_i) {
                        output[b * dim4 + (m_o * m + m_i) * K * k + k_o * k
                                + k_i]
                                = input[b * dim4 + m_o * dim3 + k_o * dim2
                                        + m_i * dim1 + k_i];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T KN2NKkn(T &input, int N, int K, int k, int n, int origin_k = 0,
        int origin_n = 0, int dtype_block = 1) {
    origin_k = origin_k ? origin_k : K * k;
    origin_n = origin_n ? origin_n : N * n;
    int pad_k = utils::divide_and_ceil(k, dtype_block);
    T output(test_utils::product({N, K, pad_k, n, dtype_block}));
    int dim4 = K * pad_k * n * dtype_block, dim3 = pad_k * n * dtype_block,
        dim2 = n * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto n_o = 0; n_o < N; ++n_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto k_i = 0; k_i < pad_k; ++k_i) {
                for (auto n_i = 0; n_i < n; ++n_i) {
                    for (auto k_b = 0; k_b < dtype_block; ++k_b) {
                        if ((n_o * n + n_i < origin_n)
                                && (k_o * k + k_i * dtype_block + k_b
                                        < origin_k)
                                && (k_i * dtype_block + k_b) < k) {
                            output[n_o * dim4 + k_o * dim3 + k_i * dim2
                                    + n_i * dim1 + k_b]
                                    = input[(k_o * k + k_i * dtype_block + k_b)
                                                    * origin_n
                                            + n_o * n + n_i];
                        } else {
                            output[n_o * dim4 + k_o * dim3 + k_i * dim2
                                    + n_i * dim1 + k_b]
                                    = 0;
                        }
                    }
                }
            }
        }
    }
    return output;
}
template <typename T>
T ACBD2ABDCcd(T &input, int A, int B, int D, int C, int c, int d, int origin_A,
        int origin_C, int origin_B, int origin_D, int dtype_block = 1) {
    int pad_c = utils::divide_and_ceil(c, dtype_block);
    T output(test_utils::product({A, B, C, D, pad_c, d, dtype_block}));
    int dim6 = B * C * D * pad_c * d * dtype_block,
        dim5 = C * D * pad_c * d * dtype_block,
        dim4 = C * pad_c * d * dtype_block, dim3 = pad_c * d * dtype_block,
        dim2 = d * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto a_o = 0; a_o < A; ++a_o) {
        for (auto b_o = 0; b_o < B; ++b_o)
            for (auto d_o = 0; d_o < D; ++d_o) {
                for (auto c_o = 0; c_o < C; ++c_o)
                    for (auto c_i = 0; c_i < pad_c; ++c_i) {
                        for (auto d_i = 0; d_i < d; ++d_i) {
                            for (auto c_b = 0; c_b < dtype_block; ++c_b) {
                                if ((d_o * d + d_i < origin_D)
                                        && (c_o * c + c_i * dtype_block + c_b
                                                < origin_C)
                                        && (c_i * dtype_block + c_b) < c) {
                                    auto input_idx = (d_o * d + d_i)
                                            + (b_o)*origin_D
                                            + (c_o * c + c_i * dtype_block
                                                      + c_b)
                                                    * (origin_B * origin_D)
                                            + (a_o)
                                                    * (origin_C * origin_B
                                                            * origin_D);
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = input[input_idx];
                                } else {
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = 0;
                                }
                            }
                        }
                    }
            }
    }
    return output;
}
template <typename T>
T ADBC2ABDCcd(T &input, int A, int B, int D, int C, int c, int d, int origin_A,
        int origin_D, int origin_B, int origin_C, int dtype_block = 1) {
    int pad_c = utils::divide_and_ceil(c, dtype_block);
    T output(test_utils::product({A, B, C, D, pad_c, d, dtype_block}));
    int dim6 = B * C * D * pad_c * d * dtype_block,
        dim5 = C * D * pad_c * d * dtype_block,
        dim4 = C * pad_c * d * dtype_block, dim3 = pad_c * d * dtype_block,
        dim2 = d * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto a_o = 0; a_o < A; ++a_o) {
        for (auto b_o = 0; b_o < B; ++b_o)
            for (auto d_o = 0; d_o < D; ++d_o) {
                for (auto c_o = 0; c_o < C; ++c_o)
                    for (auto c_i = 0; c_i < pad_c; ++c_i) {
                        for (auto d_i = 0; d_i < d; ++d_i) {
                            for (auto c_b = 0; c_b < dtype_block; ++c_b) {
                                if ((d_o * d + d_i < origin_D)
                                        && (c_o * c + c_i * dtype_block + c_b
                                                < origin_C)
                                        && (c_i * dtype_block + c_b) < c) {
                                    auto input_idx
                                            = (c_o * c + c_i * dtype_block
                                                      + c_b)
                                            + (b_o)*origin_C
                                            + (d_o * d + d_i) * origin_C
                                                    * origin_B
                                            + (a_o)
                                                    * (origin_C * origin_B
                                                            * origin_D);
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = input[input_idx];
                                } else {
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = 0;
                                }
                            }
                        }
                    }
            }
    }
    return output;
}
template <typename T>
T ACBD2ABCDcd(T &input, int A, int B, int C, int D, int c, int d, int origin_A,
        int origin_C, int origin_B, int origin_D, int dtype_block = 1) {
    int pad_c = utils::divide_and_ceil(c, dtype_block);
    T output(test_utils::product({A, B, C, D, pad_c, d, dtype_block}));
    int dim6 = B * C * D * pad_c * d * dtype_block,
        dim5 = C * D * pad_c * d * dtype_block,
        dim4 = D * pad_c * d * dtype_block, dim3 = pad_c * d * dtype_block,
        dim2 = d * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto a_o = 0; a_o < A; ++a_o) {
        for (auto b_o = 0; b_o < B; ++b_o)
            for (auto c_o = 0; c_o < C; ++c_o)
                for (auto d_o = 0; d_o < D; ++d_o) {
                    for (auto c_i = 0; c_i < pad_c; ++c_i) {
                        for (auto d_i = 0; d_i < d; ++d_i) {
                            for (auto c_b = 0; c_b < dtype_block; ++c_b) {
                                if ((d_o * d + d_i < origin_D)
                                        && (c_o * c + c_i * dtype_block + c_b
                                                < origin_C)
                                        && (c_i * dtype_block + c_b) < c) {
                                    auto input_idx = (d_o * d + d_i)
                                            + (b_o)*origin_D
                                            + (c_o * c + c_i * dtype_block
                                                      + c_b)
                                                    * (origin_B * origin_D)
                                            + (a_o)
                                                    * (origin_C * origin_B
                                                            * origin_D);
                                    output[a_o * dim6 + b_o * dim5 + c_o * dim4
                                            + d_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = input[input_idx];
                                } else {
                                    output[a_o * dim6 + b_o * dim5 + c_o * dim4
                                            + d_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = 0;
                                }
                            }
                        }
                    }
                }
    }
    return output;
}

template <typename T>
T ABDC2ABCDcd(T &input, int A, int B, int C, int D, int c, int d, int origin_A,
        int origin_B, int origin_D, int origin_C, int dtype_block = 1) {
    int pad_c = utils::divide_and_ceil(c, dtype_block);
    T output(test_utils::product({A, B, C, D, pad_c, d, dtype_block}));
    int dim6 = B * C * D * pad_c * d * dtype_block,
        dim5 = C * D * pad_c * d * dtype_block,
        dim4 = D * pad_c * d * dtype_block, dim3 = pad_c * d * dtype_block,
        dim2 = d * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto a_o = 0; a_o < A; ++a_o) {
        for (auto b_o = 0; b_o < B; ++b_o)
            for (auto c_o = 0; c_o < C; ++c_o)
                for (auto d_o = 0; d_o < D; ++d_o) {
                    for (auto c_i = 0; c_i < pad_c; ++c_i) {
                        for (auto d_i = 0; d_i < d; ++d_i) {
                            for (auto c_b = 0; c_b < dtype_block; ++c_b) {
                                if ((d_o * d + d_i < origin_D)
                                        && (c_o * c + c_i * dtype_block + c_b
                                                < origin_C)
                                        && (c_i * dtype_block + c_b) < c) {
                                    auto input_idx
                                            = (c_o * c + c_i * dtype_block
                                                      + c_b)
                                            + (d_o * d + d_i) * origin_C
                                            + (b_o) * (origin_D * origin_C)
                                            + (a_o)
                                                    * (origin_B * origin_D
                                                            * origin_C);
                                    output[a_o * dim6 + b_o * dim5 + c_o * dim4
                                            + d_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = input[input_idx];
                                } else {
                                    output[a_o * dim6 + b_o * dim5 + c_o * dim4
                                            + d_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = 0;
                                }
                            }
                        }
                    }
                }
    }
    return output;
}

template <typename T>
T ABCD2ABDCcd(T &input, int A, int B, int C, int D, int c, int d, int origin_A,
        int origin_B, int origin_C, int origin_D, int dtype_block = 1) {
    int pad_c = utils::divide_and_ceil(c, dtype_block);
    T output(test_utils::product({A, B, C, D, pad_c, d, dtype_block}));
    int dim6 = B * C * D * pad_c * d * dtype_block,
        dim5 = C * D * pad_c * d * dtype_block,
        dim4 = C * pad_c * d * dtype_block, dim3 = pad_c * d * dtype_block,
        dim2 = d * dtype_block, dim1 = dtype_block;
    // #pragma omp parallel for
    for (auto a_o = 0; a_o < A; ++a_o) {
        for (auto b_o = 0; b_o < B; ++b_o)
            for (auto d_o = 0; d_o < D; ++d_o) {
                for (auto c_o = 0; c_o < C; ++c_o)
                    for (auto c_i = 0; c_i < pad_c; ++c_i) {
                        for (auto d_i = 0; d_i < d; ++d_i) {
                            for (auto c_b = 0; c_b < dtype_block; ++c_b) {
                                if ((d_o * d + d_i < origin_D)
                                        && (c_o * c + c_i * dtype_block + c_b
                                                < origin_C)
                                        && (c_i * dtype_block + c_b) < c) {
                                    auto input_idx = (d_o * d + d_i)
                                            + (c_o * c + c_i * dtype_block
                                                      + c_b)
                                                    * origin_D
                                            + (b_o) * (origin_D * origin_C)
                                            + (a_o)
                                                    * (origin_B * origin_D
                                                            * origin_C);
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = input[input_idx];
                                } else {
                                    output[a_o * dim6 + b_o * dim5 + d_o * dim4
                                            + c_o * dim3 + c_i * dim2
                                            + d_i * dim1 + c_b]
                                            = 0;
                                }
                            }
                        }
                    }
            }
    }
    return output;
}

template <typename T, typename A>
std::vector<T, A> KN2KNkn(
        std::vector<T, A> &input, int K, int N, int k, int n) {
    std::vector<T, A> output(input.size());
    int dim3 = N * k * n, dim2 = k * n, dim1 = n;
    utils::parallel_for(0, K, 1, [&](int64_t k_o) {
        for (auto n_o = 0; n_o < N; ++n_o) {
            for (auto k_i = 0; k_i < k; ++k_i) {
                for (auto n_i = 0; n_i < n; ++n_i) {
                    output[k_o * dim3 + n_o * dim2 + k_i * dim1 + n_i]
                            = input[(k_o * k + k_i) * N * n + n_o * n + n_i];
                }
            }
        }
    });
    return output;
}

template <typename T>
T NK2NKknk(T &input, int N, int K, int k, int n, int k2) {
    T output(input.size());
    int dim4 = K * k * n * k2, dim3 = k * n * k2, dim2 = n * k2, dim1 = k2;
    utils::parallel_for(0, N, 1, [&](int64_t n_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto k_i = 0; k_i < k; ++k_i) {
                for (auto n_i = 0; n_i < n; ++n_i) {
                    for (auto k2_i = 0; k2_i < k2; ++k2_i) {
                        output[n_o * dim4 + k_o * dim3 + k_i * dim2 + n_i * dim1
                                + k2_i]
                                = input[(n_o * n + n_i) * K * k * k2
                                        + k_o * k * k2 + k_i * k2 + k2_i];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T ABC2ABCcb4c(T &input, int B, int N, int K, int k, int n, int k2) {
    T output(input.size());
    int dim5 = N * K * k * n * k2, dim4 = K * k * n * k2, dim3 = k * n * k2,
        dim2 = n * k2, dim1 = k2;
    utils::parallel_for(0, B, 1, [&](int64_t b_o) {
        for (auto n_o = 0; n_o < N; ++n_o) {
            for (auto k_o = 0; k_o < K; ++k_o) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    for (auto n_i = 0; n_i < n; ++n_i) {
                        for (auto k2_i = 0; k2_i < k2; ++k2_i) {
                            output[b_o * dim5 + n_o * dim4 + k_o * dim3
                                    + k_i * dim2 + n_i * dim1 + k2_i]
                                    = input[b_o * dim5
                                            + (n_o * n + n_i) * K * k * k2
                                            + k_o * k * k2 + k_i * k2 + k2_i];
                        }
                    }
                }
            }
        }
    });

    return output;
}

template <typename T>
T batch_NK2NKkn(T &input, int batch, int N, int K, int k, int n,
        int origin_k = 0, int origin_n = 0, int dtype_block = 1) {
    origin_k = origin_k ? origin_k : K * k;
    origin_n = origin_n ? origin_n : N * n;
    int pad_k = utils::divide_and_ceil(k, dtype_block);
    T output(test_utils::product({N, K, pad_k, n, dtype_block}));
    int dim5 = N * K * pad_k * n * dtype_block,
        dim4 = K * pad_k * n * dtype_block, dim3 = pad_k * n * dtype_block,
        dim2 = n * dtype_block, dim1 = dtype_block;
    utils::parallel_for(0, batch, 1, [&](int64_t b_o) {
        for (auto n_o = 0; n_o < N; ++n_o) {
            for (auto k_o = 0; k_o < K; ++k_o) {
                for (auto k_i = 0; k_i < pad_k; ++k_i) {
                    for (auto n_i = 0; n_i < n; ++n_i) {
                        for (auto k_b = 0; k_b < dtype_block; ++k_b) {
                            if ((n_o * n + n_i < origin_n)
                                    && (k_o * k + k_i * dtype_block + k_b
                                            < origin_k)
                                    && (k_i * dtype_block + k_b) < k) {
                                output[batch * dim5 + n_o * dim4 + k_o * dim3
                                        + k_i * dim2 + n_i * dim1 + k_b]
                                        = input[batch * origin_k * origin_n
                                                + (k_o * k + k_i * dtype_block
                                                        + k_b)
                                                + (n_o * n + n_i) * origin_k];
                            } else {
                                output[batch * dim5 + n_o * dim4 + k_o * dim3
                                        + k_i * dim2 + n_i * dim1 + k_b]
                                        = 0;
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T NK2NKkn(T &input, int N, int K, int k, int n, int origin_k = 0,
        int origin_n = 0) {
    T output(N * K * k * n);
    if (origin_k == 0) { origin_k = K * k; }
    if (origin_n == 0) { origin_n = N * n; }
    int dim3 = K * k * n, dim2 = k * n, dim1 = n;
    utils::parallel_for(0, N, 1, [&](int64_t n_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto k_i = 0; k_i < k; ++k_i) {
                for (auto n_i = 0; n_i < n; ++n_i) {
                    if (k_o * k + k_i < origin_k && n_o * n + n_i < origin_n) {
                        output[n_o * dim3 + k_o * dim2 + k_i * dim1 + n_i]
                                = input[(n_o * n + n_i) * origin_k + k_o * k
                                        + k_i];
                    } else {
                        output[n_o * dim3 + k_o * dim2 + k_i * dim1 + n_i] = 0;
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T NKkn2KN(T &input, int N, int K, int k, int n, int origin_k = 0,
        int origin_n = 0) {
    origin_k = origin_k ? origin_k : K * k;
    origin_n = origin_n ? origin_n : N * n;
    T output(origin_k * origin_n);
    int dim3 = K * k * n, dim2 = k * n, dim1 = n;
    utils::parallel_for(0, N, 1, [&](int64_t n_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            for (auto k_i = 0; k_i < k; ++k_i) {
                for (auto n_i = 0; n_i < n; ++n_i) {
                    if ((n_o * n + n_i) < origin_n
                            && (k_o * k + k_i) < origin_k) {
                        output[(k_o * k + k_i) * origin_n + n_o * n + n_i]
                                = input[n_o * dim3 + k_o * dim2 + k_i * dim1
                                        + n_i];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T, typename A>
std::vector<T, A> batch_NKkn2KN(
        std::vector<T, A> &input, int batch, int N, int K, int k, int n) {
    std::vector<T, A> output(input.size());
    int dim4 = N * K * k * n, dim3 = K * k * n, dim2 = k * n, dim1 = n;
    utils::parallel_for(0, batch, 1, [&](int64_t b) {
        for (auto n_o = 0; n_o < N; ++n_o) {
            for (auto k_o = 0; k_o < K; ++k_o) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    for (auto n_i = 0; n_i < n; ++n_i) {
                        output[b * dim4 + (k_o * k + k_i) * N * n + n_o * n
                                + n_i]
                                = input[b * dim4 + n_o * dim3 + k_o * dim2
                                        + k_i * dim1 + n_i];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T, typename A>
std::vector<T, A> padding_weight(
        const std::vector<T, A> &input, int N, int K, int dim = 0) {
    std::vector<T, A> output(N * K, 0);
    int insize = input.size();
    auto inN = (dim == 1 ? N : insize / K), inK = (dim == 1 ? insize / N : K);
    utils::parallel_for(0, N, 1, [&](int64_t n_o) {
        for (int k_o = 0; k_o < K; ++k_o) {
            if ((dim == 1 && k_o < inK) || (dim == 0 && n_o < inN)) {
                output[n_o * K + k_o] = input[n_o * inK + k_o];
            }
        }
    });
    return output;
}

template <typename T, typename A>
std::vector<T, A> drop_padding_weight(
        std::vector<T, A> &input, int N, int K, int insize, int dim = 0) {
    std::vector<T, A> output(insize, 0);
    auto inN = (dim == 1 ? N : insize / K), inK = (dim == 1 ? insize / N : K);
    utils::parallel_for(0, N, 1, [&](int64_t n_o) {
        for (auto k_o = 0; k_o < K; ++k_o) {
            if ((dim == 1 && k_o < inK) || (dim == 0 && n_o < inN)) {
                output[n_o * inK + k_o] = input[n_o * K + k_o];
            }
        }
    });
    return output;
}

template <typename T, typename A>
std::vector<T, A> transpose(std::vector<T, A> &input, int M, int N) {
    std::vector<T, A> output(M * N);
    utils::parallel_for(0, M, 1, [&](int64_t i) {
        for (auto j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    });
    return output;
}

template <typename T>
T ref_reduce_sum_MKmk(T &input, int M, int K, int m, int k,
        int axis = 0) { // axis should be 0(M) or 1(K)
    int A, B, a, b;
    std::tie(A, B, a, b) = axis ? std::tie(M, K, m, k) : std::tie(K, M, k, m);
    using TElem = typename std::remove_reference<decltype(input[0])>::type;
    T output = alloc_array<TElem>(A * a, INIT_ZERO);
    int stride0 = K * m * k, stride1 = m * k, stride2 = k;
    utils::parallel_for(0, A, 1, [&](int64_t a_o) {
        for (int a_i = 0; a_i < a; a_i++) {
            float reduce_out = 0.f;
            for (int b_o = 0; b_o < B; b_o++) {
                for (int b_i = 0; b_i < b; b_i++) {
                    if (axis == 1) {
                        reduce_out += input[a_o * stride0 + b_o * stride1
                                + a_i * stride2 + b_i];
                    } else {
                        reduce_out += input[b_o * stride0 + a_o * stride1
                                + b_i * stride2 + a_i];
                    }
                }
            }
            output[a_o * a + a_i] = reduce_out;
        }
    });
    return output;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
