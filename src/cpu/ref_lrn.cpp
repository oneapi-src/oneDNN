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

#include "ref_lrn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_lrn_fwd_t<data_type>::execute_forward() {
    using namespace alg_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = reinterpret_cast<data_t*>(this->memory(1));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper ws_d(conf_.workspace_pd());

    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();
    const bool across_channels = conf_.desc()->alg_kind == lrn_across_channels;

    auto ker = [=](data_t *d, int mb, int oc, int oh, int ow) {
        const double alpha = conf_.desc()->lrn_alpha;
        const double beta = conf_.desc()->lrn_beta;

        const int size = conf_.desc()->local_size;
        const int CSIZE = across_channels ? size : 1;
        const int HWSIZE = size + 1 - CSIZE;

        data_t sum = 0.0;
        int summands = across_channels ? size : size*size;
        for (int c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (int h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (int w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src[data_d.off(mb, c - (CSIZE - 1) / 2,
                            h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2)];
                    sum += s * s;
                }
            }
        }
        data_t k = pow(1 + alpha * sum / summands, beta);
        d[0] = src[data_d.off(mb, oc, oh, ow)] / k;
        if (ws)
            ws[ws_d.off(mb, oc, oh, ow)]
                = 1 / (k * (1 + alpha * sum / summands)); // for back prop
    };

    const int MB = conf_.MB();
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    ker(&dst[data_d.off(mb, c, h, w)], mb, c, h, w);
                }
            }
        }
    }
}

template struct ref_lrn_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
