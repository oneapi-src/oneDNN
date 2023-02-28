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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_BERTBMM_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_BERTBMM_REF_HPP

#include <stdlib.h>
#include <vector>
#include <test_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename T0, typename T1, typename TOut>
inline void ref_bertBMM_QK(TOut *out, const T0 *A, const T1 *B,
        const sc_dims &dims_2D, const int batch_size, const int num_head) {
    // reshape + transpose + BMM
    const auto size = test_utils::product(dims_2D);
    std::vector<T0> A_trans(size, 0.f);
    std::vector<T1> B_trans(size, 0.f);
    const auto seq_len = dims_2D[0] / batch_size;
    const auto head_dim = dims_2D[1] / num_head;
    const sc_dims stride
            = {num_head * seq_len * head_dim, seq_len * head_dim, head_dim, 1};
    const sc_dims orig_stride
            = {seq_len * num_head * head_dim, head_dim, num_head * head_dim, 1};
    const sc_dims c_stride
            = {seq_len * num_head * seq_len, seq_len, num_head * seq_len, 1};
    // do transpose [0, 2, 1, 3]
    utils::parallel_for(0, batch_size, 1, [&](int64_t i) {
        for (int j = 0; j < num_head; ++j) {
            for (int ii = 0; ii < seq_len; ++ii) {
                for (int jj = 0; jj < head_dim; ++jj) {
                    A_trans[i * stride[0] + j * stride[1] + ii * stride[2] + jj]
                            = A[i * orig_stride[0] + j * orig_stride[1]
                                    + ii * orig_stride[2] + jj];
                    B_trans[i * stride[0] + j * stride[1] + ii * stride[2] + jj]
                            = B[i * orig_stride[0] + j * orig_stride[1]
                                    + ii * orig_stride[2] + jj];
                }
            }
        }
    });

    // compute BMM
    utils::parallel_for(0, batch_size, 1, [&](int64_t i) {
        for (int j = 0; j < num_head; ++j) {
            auto offset = i * stride[0] + j * stride[1];
            T0 *A_base = &A_trans[offset];
            T1 *B_base = &B_trans[offset];

            for (int m = 0; m < seq_len; ++m) {
                TOut *out_base = &out[i * c_stride[0] + j * c_stride[1]
                        + m * c_stride[2]];
                for (int n = 0; n < seq_len; ++n) {
                    TOut sum = 0.f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += (TOut)A_base[m * head_dim + k]
                                * B_base[n * head_dim + k];
                    }
                    out_base[n] = sum;
                }
            }
        }
    });
}

template <typename T0, typename T1, typename TOut>
inline void ref_bertBMM_V(TOut *out, const T0 *A, const T1 *B,
        const sc_dims &A_dims_2D, const sc_dims &B_dims_2D,
        const int batch_size, const int num_head) {
    const auto A_size = test_utils::product(A_dims_2D);
    const auto B_size = test_utils::product(B_dims_2D);
    std::vector<T0> A_trans(A_size, 0.f);
    std::vector<T1> B_trans(B_size, 0.f);
    const auto seq_len = A_dims_2D[0] / batch_size;
    const auto A_dim = A_dims_2D[1] / num_head;
    const auto B_dim = B_dims_2D[1] / num_head;
    const sc_dims a_stride
            = {num_head * seq_len * A_dim, seq_len * A_dim, A_dim, 1};
    const sc_dims orig_a_stride
            = {seq_len * num_head * A_dim, A_dim, num_head * A_dim, 1};

    const sc_dims b_stride
            = {num_head * seq_len * B_dim, seq_len * B_dim, B_dim, 1};
    const sc_dims orig_b_stride
            = {seq_len * num_head * B_dim, B_dim, num_head * B_dim, 1};

    const sc_dims c_stride
            = {seq_len * num_head * B_dim, B_dim, num_head * B_dim, 1};

    // do transpose [0, 2, 1, 3]
    utils::parallel_for(0, batch_size, 1, [&](int64_t i) {
        for (int j = 0; j < num_head; ++j) {
            for (int ii = 0; ii < seq_len; ++ii) {
                for (int jj = 0; jj < A_dim; ++jj) {
                    A_trans[i * a_stride[0] + j * a_stride[1] + ii * a_stride[2]
                            + jj]
                            = A[i * orig_a_stride[0] + j * orig_a_stride[1]
                                    + ii * orig_a_stride[2] + jj];
                }
                for (int jj = 0; jj < B_dim; ++jj) {
                    B_trans[i * b_stride[0] + j * b_stride[1] + ii * b_stride[2]
                            + jj]
                            = B[i * orig_b_stride[0] + j * orig_b_stride[1]
                                    + ii * orig_b_stride[2] + jj];
                }
            }
        }
    });

    // compute BMM
    utils::parallel_for(0, batch_size, 1, [&](int64_t i) {
        for (int j = 0; j < num_head; ++j) {
            T0 *A_base = &A_trans[i * a_stride[0] + j * a_stride[1]];
            T1 *B_base = &B_trans[i * b_stride[0] + j * b_stride[1]];

            for (int m = 0; m < seq_len; ++m) {
                TOut *out_base = &out[i * c_stride[0] + j * c_stride[1]
                        + m * c_stride[2]];
                for (int n = 0; n < B_dim; ++n) {
                    TOut sum = 0.f;
                    for (int k = 0; k < A_dim; ++k) {
                        sum += (TOut)A_base[m * A_dim + k]
                                * B_base[k * B_dim + n];
                    }
                    out_base[n] = sum;
                }
            }
        }
    });
}

template <typename T>
static void ref_bertBMM_softmax_mask(T *C, T *mask, const sc_dims &dims_C,
        const sc_dims &dims_mask, int size_per_head = 64) {
    /** Only suitable for BMM_QK
     * adder = (1.0 - attention_mask) * (-10000.0)
     * attention_score("ATTEN_QK") =
     *     attention_score * (1/sqrt(SIZE_PER_HEAD)) + adder
     */
    COMPILE_ASSERT(dims_C[2] == dims_C[3],
            "softmax mask fusion should only occurs in bmm_qk mode.");
    int qk_M_num_block = dims_C[0], qk_N_num_block = dims_C[1],
        M_block = dims_C[2];
    int batch_size = dims_mask[0], seq_len = dims_mask[2];
    int M_num_block = qk_M_num_block / batch_size,
        N_num_block = seq_len / M_block;
    int num_head = qk_N_num_block / N_num_block;
    COMPILE_ASSERT(
            dims_mask[1] == 1, "dims[1] of softmax mask is expected to be 1.");
    COMPILE_ASSERT(M_num_block == seq_len / M_block,
            "seq_len should equal to M_num_block *  M_block");
    COMPILE_ASSERT(M_num_block == N_num_block,
            "M_num_block should be equal to N_num_block");
    // compute mask
    utils::parallel_for(0, batch_size, 1, [&](int64_t i) {
        for (int ii = 0; ii < seq_len; ++ii) {
            for (int jj = 0; jj < seq_len; ++jj) {
                mask[i * seq_len * seq_len + ii * seq_len + jj]
                        = (1.0
                                  - mask[i * seq_len * seq_len + ii * seq_len
                                          + jj])
                        * (-10000.0);
            }
        }
    });
    // compute bmm add mask
    utils::parallel_for(0, batch_size, 1, [&](int64_t bs) {
        for (int nh = 0; nh < num_head; ++nh) {
            for (int m_o = 0; m_o < M_num_block; ++m_o) {
                for (int n_o = 0; n_o < N_num_block; ++n_o) {
                    for (int m_i = 0; m_i < M_block; ++m_i) {
                        for (int n_i = 0; n_i < M_block; ++n_i) {
                            C[(bs * M_num_block + m_o) * qk_N_num_block
                                            * M_block * M_block
                                    + (nh * N_num_block + n_o) * M_block
                                            * M_block
                                    + m_i * M_block + n_i]
                                    = C[(bs * M_num_block + m_o)
                                                      * qk_N_num_block * M_block
                                                      * M_block
                                              + (nh * N_num_block + n_o)
                                                      * M_block * M_block
                                              + m_i * M_block + n_i]
                                            / std::sqrt(size_per_head)
                                    + mask[bs * seq_len * seq_len
                                            + (m_o * M_block + m_i) * seq_len
                                            + (n_o * M_block + n_i)];
                        }
                    }
                }
            }
        }
    });
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
