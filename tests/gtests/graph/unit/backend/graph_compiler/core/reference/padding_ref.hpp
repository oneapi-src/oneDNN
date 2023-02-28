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

#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_PADDING_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_PADDING_REF_HPP

#include <stdlib.h>
#include <test_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
template <typename T>
static void ref_padding_2d(
        T *out, T *in, sc_dims out_dims, sc_dims pads_begin, sc_dims pads_end) {
    auto N = out_dims[0], C = out_dims[1], H = out_dims[2], W = out_dims[3];

    test_utils::parallel_nd(
            static_cast<int>(N * C * H * W), [&](int64_t i) { out[i] = 0; });

    auto ow_ = W - pads_begin[1] - pads_end[1];
    auto oh_ = H - pads_begin[0] - pads_end[0];

    int dim1 = C * oh_ * ow_, dim2 = oh_ * ow_, dim3 = ow_;
    utils::parallel_for(0, N, 1, [&](int64_t n) {
        int offset = n * C * H * W;
        for (auto c = 0; c < C; ++c) {
            offset += pads_begin[0] * W + pads_begin[1];
            for (auto h = 0; h < oh_; ++h) {
                for (auto w = 0; w < ow_; ++w) {
                    out[offset + w] = in[n * dim1 + c * dim2 + h * dim3 + w];
                }
                offset += W;
            }
            offset -= pads_begin[1];
            offset += pads_end[0] * W;
        }
    });
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
