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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_INSTANCENORM_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_INSTANCENORM_REF_HPP

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <test_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// TODO(xxx): last dims is default axis to normlization
static void ref_instancenorm_fwd(float *out, float *in, float *gamma,
        float *beta, int N, int C, int H, int W, float eps = 1e-5f,
        bool use_scale_shift = false) {
    std::vector<float> mean(N * C);
    std::vector<float> var(N * C);

    utils::parallel_for(0, N, 1, [&](int64_t i) {
        for (uint64_t j = 0; j < (uint64_t)C; ++j) {
            float sum = 0.0f;
            for (uint64_t u = 0; u < (uint64_t)H; ++u) {
                for (uint64_t v = 0; v < (uint64_t)W; ++v) {
                    sum += in[i * C * H * W + j * H * W + u * W + v];
                }
            }
            mean[i * C + j] = sum / (H * W);
            float sqd_diff = 0.0f;
            for (uint64_t u = 0; u < (uint64_t)H; ++u) {
                for (uint64_t v = 0; v < (uint64_t)W; ++v) {
                    sqd_diff += (in[i * C * H * W + j * H * W + u * W + v]
                                        - mean[i * C + j])
                            * (in[i * C * H * W + j * H * W + u * W + v]
                                    - mean[i * C + j]);
                }
            }
            var[i * C + j] = sqd_diff / (H * W);
            for (uint64_t u = 0; u < (uint64_t)H; ++u) {
                for (uint64_t v = 0; v < (uint64_t)W; ++v) {
                    out[i * C * H * W + j * H * W + u * W + v]
                            = ((in[i * C * H * W + j * H * W + u * W + v]
                                       - mean[i * C + j])
                                      / std::sqrt(var[i * C + j] + eps))
                                    * (use_scale_shift ? gamma[u * W + v] : 1)
                            + (use_scale_shift ? beta[u * W + v] : 0);
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
