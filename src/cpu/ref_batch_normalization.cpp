/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "ref_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_batch_normalization_fwd_t<data_type>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto scaleshift = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = reinterpret_cast<data_t*>(this->memory(1));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper scaleshift_d(conf_.weights_pd());

    const int N = conf_.MB();
    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();

    const bool is_training = ws != nullptr;
    const double eps = conf_.desc()->batch_norm_epsilon;

    data_t *ws_mean = is_training ? &ws[0] : nullptr;
    data_t *ws_variance = is_training ? &ws[C] : nullptr;

#   pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        data_t v_mean, v_variance;
        data_t &mean = is_training ? ws_mean[c] : v_mean;
        data_t &variance = is_training ? ws_variance[c] : v_variance;

        mean = variance = 0;

        for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            mean += src[data_d.off(n, c, h, w)];
        mean /= W * N * H;

        for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            data_t m = src[data_d.off(n,c,h,w)] - mean;
            variance += m*m;
        }
        variance = 1. / sqrt(variance/(W * H * N) + eps);

        for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            auto d_off = data_d.off(n,c,h,w);
            auto sm_off = scaleshift_d.off(0, c);
            auto sv_off = scaleshift_d.off(1, c);
            dst[d_off] = scaleshift[sm_off] * (src[d_off] - mean) * variance +
                scaleshift[sv_off];
        }
    }
}

template struct ref_batch_normalization_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
