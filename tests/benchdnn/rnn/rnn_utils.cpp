/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "rnn/rnn_aux.hpp"
#include "src/common/dnnl_thread.hpp"

namespace rnn {

void copy(int64_t dimc, int64_t dimr, int64_t ld_src, int64_t ld_dst,
        const float *src_, float *dst_, rnn_action_t action) {
    AOC<const float> src(src_, dimc, ld_src);
    AOC<float> dst(dst_, dimc, ld_dst);

    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++) {
            dst(i, j)
                    = action == action_sum ? dst(i, j) + src(i, j) : src(i, j);
        }
    });
}

void shift(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_, float shift,
        bool round) {
    AOC<float> src(src_, dimc, ld_src);
    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++) {
            float fp = src(i, j) + shift;
            src(i, j) = round ? saturate<dnnl_u8>(fp) : fp;
        }
    });
}

void scale(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_, float scale,
        bool round) {
    AOC<float> src(src_, dimc, ld_src);
    dnnl::impl::parallel_nd(dimc, [&](int64_t i) {
        for (int64_t j = 0; j < dimr; j++) {
            float fp = src(i, j) * scale;
            src(i, j) = round ? saturate<dnnl_u8>(fp) : fp;
        }
    });
}

} // namespace rnn
