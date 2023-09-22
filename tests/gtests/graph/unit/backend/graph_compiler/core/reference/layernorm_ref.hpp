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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_LAYERNORM_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_LAYERNORM_REF_HPP

#include <cmath>
#include <stdlib.h>
#include <test_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// TODO(xxx): last dims is default axis to normlization
inline void ref_layernorm_fwd_with_mean_var(float *out, float *mean, float *var,
        float *in, float *gamma, float *beta, int N, int T, int C,
        float eps = 1e-5f, bool use_scale_shift = false) {
    utils::parallel_for(0, N, 1, [&](int64_t i) {
        for (uint64_t j = 0; j < (uint64_t)T; ++j) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < (uint64_t)C; ++k) {
                sum += in[i * T * C + j * C + k];
            }
            mean[i * T + j] = sum / C;
            float sqd_diff = 0.0f;
            for (uint64_t k = 0; k < (uint64_t)C; ++k) {
                sqd_diff += (in[i * T * C + j * C + k] - mean[i * T + j])
                        * (in[i * T * C + j * C + k] - mean[i * T + j]);
            }
            var[i * T + j] = sqd_diff / C;
            for (uint64_t k = 0; k < (uint64_t)C; ++k) {
                out[i * T * C + j * C + k]
                        = ((in[i * T * C + j * C + k] - mean[i * T + j])
                                  / std::sqrt(var[i * T + j] + eps))
                                * (use_scale_shift ? gamma[k] : 1)
                        + (use_scale_shift ? beta[k] : 0);
            }
        }
    });
}

inline void ref_layernorm_fwd(float *out, float *in, float *gamma, float *beta,
        int N, int T, int C, float eps = 1e-5f, bool use_scale_shift = false) {
    float mean[N * T];
    float var[N * T];

    ref_layernorm_fwd_with_mean_var(
            out, mean, var, in, gamma, beta, N, T, C, eps, use_scale_shift);
}

// TODO(xxx): K and k axis are default axis to normlization
inline void ref_layernorm_block_fwd_with_mean_var(float *out, float *mean,
        float *var, float *in, float *gamma, float *beta, int M, int K, int m,
        int k, float eps = 1e-5f, bool use_scale_shift = false) {
    utils::parallel_for(0, M, 1, [&](int64_t i) {
        for (uint64_t j = 0; j < (uint64_t)m; ++j) {
            float sum = 0.0f;
            for (uint64_t u = 0; u < (uint64_t)K; ++u) {
                for (uint64_t v = 0; v < (uint64_t)k; ++v) {
                    sum += in[i * K * m * k + u * m * k + j * k + v];
                }
            }
            mean[i * m + j] = sum / (K * k);
            float sqd_diff = 0.0f;
            for (uint64_t u = 0; u < (uint64_t)K; ++u) {
                for (uint64_t v = 0; v < (uint64_t)k; ++v) {
                    sqd_diff += (in[i * K * m * k + u * m * k + j * k + v]
                                        - mean[i * m + j])
                            * (in[i * K * m * k + u * m * k + j * k + v]
                                    - mean[i * m + j]);
                }
            }
            var[i * m + j] = sqd_diff / (K * k);
            for (uint64_t u = 0; u < (uint64_t)K; ++u) {
                for (uint64_t v = 0; v < (uint64_t)k; ++v) {
                    out[i * K * m * k + u * m * k + j * k + v]
                            = ((in[i * K * m * k + u * m * k + j * k + v]
                                       - mean[i * m + j])
                                      / std::sqrt(var[i * m + j] + eps))
                                    * (use_scale_shift ? gamma[u * k + v] : 1)
                            + (use_scale_shift ? beta[u * k + v] : 0);
                }
            }
        }
    });
}

inline void ref_layernorm_block_fwd(float *out, float *in, float *gamma,
        float *beta, int M, int K, int m, int k, float eps = 1e-5f,
        bool use_scale_shift = false) {
    float mean[M * 1 * m * 1]; // NOLINT
    float var[M * 1 * m * 1]; // NOLINT

    ref_layernorm_block_fwd_with_mean_var(out, &mean[0], &var[0], in, gamma,
            beta, M, K, m, k, eps, use_scale_shift);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
