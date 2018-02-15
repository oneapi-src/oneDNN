/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
        const float alpha = static_cast<float>(conf_.desc()->lrn_alpha);
        const float beta = static_cast<float>(conf_.desc()->lrn_beta);
        const float k = static_cast<float>(conf_.desc()->lrn_k);

        const int size = conf_.desc()->local_size;
        const int CSIZE = across_channels ? size : 1;
        const int HWSIZE = size + 1 - CSIZE;

        float sum = 0;
        const int summands = across_channels ? size : size * size;
        for (int c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (int h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (int w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    const float s = src[data_d.off(mb, c - (CSIZE - 1) / 2,
                            h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2)];
                    sum += s * s;
                }
            }
        }
        sum = k + alpha * sum / summands;
        if (ws)
            ws[ws_d.off(mb, oc, oh, ow)] = static_cast<data_t>(sum);
        d[0] = static_cast<data_t>(src[data_d.off(mb, oc, oh, ow)]
                / powf(sum, beta));
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

template <impl::data_type_t data_type>
void ref_lrn_bwd_t<data_type>::execute_backward() {
    using namespace alg_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper diff_data_d(conf_.diff_dst_pd());

    const int MB = conf_.MB();
    const int C = conf_.C();
    const int H = conf_.H();
    const int W = conf_.W();

    const float alpha = static_cast<float>(conf_.desc()->lrn_alpha);
    const float beta = static_cast<float>(conf_.desc()->lrn_beta);
    const float k = static_cast<float>(conf_.desc()->lrn_k);
    const int kernel_size = conf_.desc()->local_size;

    auto get_omega = [=](data_t c_k, int kernel_size, int C, const data_t *src,
            int n, int c, int h, int w) -> float {
        float sum = 0.0;

        int half_kernel_size = (kernel_size - 1) / 2;
        int c_start = (c < half_kernel_size) ? 0 : c - half_kernel_size;
        int c_end = c + kernel_size - half_kernel_size;
        c_end = c_end < C ? c_end : C;
        for (int i = c_start; i < c_end; ++i) {
            const float value = src[data_d.off(n, i, h, w)];
            sum += value * value;
        }
        return static_cast<float>(c_k + sum * alpha / kernel_size);
    };

    auto ker = [=](data_t *d, int mb, int oc, int oh, int ow) {
        int ks_start = kernel_size/2 > oc ? kernel_size/2 - oc : 0;
        int ks_stop = C - oc <= kernel_size/2
            ? C - oc + kernel_size/2 : kernel_size;

        float A = 0, B = 0, omega_mid = 0;
        // intermediate calcs data_t --> float (check if OK for int data_t)
        for (int ks = ks_start; ks < ks_stop; ks++) {
            int _t = oc + ks - (kernel_size/2);
            const float omega
                = get_omega(k, kernel_size, C, src, mb, _t, oh, ow);

            if (ks == kernel_size/2) omega_mid = omega;

            float t = src[data_d.off(mb, _t, oh, ow)] / powf(omega, beta);
            B += 1.0f / omega * t * diff_dst[diff_data_d.off(mb, _t, oh, ow)];
        }

        A = 1.0f / powf(omega_mid, beta)
            * diff_dst[diff_data_d.off(mb, oc, oh, ow)];
        B *= src[data_d.off(mb, oc, oh, ow)];
        B *= (2.0f * alpha * beta) / kernel_size;
        *d = static_cast<data_t>(A - B); // final cast down to data_t
    };

#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    ker(&diff_src[diff_data_d.off(mb, c, h, w)], mb, c, h, w);
                }
            }
        }
    }

}

template struct ref_lrn_fwd_t<data_type::f32>;
template struct ref_lrn_bwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
