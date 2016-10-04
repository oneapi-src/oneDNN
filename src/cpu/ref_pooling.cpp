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

/* FIXME: get rid of this! */
#include <limits>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "ref_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_pooling_fwd_t<data_type>::execute_forward() {
    using namespace alg_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = conf_.desc()->alg_kind == alg_kind::pooling_avg ? nullptr
        : reinterpret_cast<int*>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper ws_d(conf_.workspace_pd());

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int SH = conf_.KSH();
    const int SW = conf_.KSW();
    const int padT = conf_.padT();
    const int padL = conf_.padL();

    auto ker_max = [=](data_t *d, int mb, int oc, int oh, int ow) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const int ih = oh * SH - padT + kh;
                const int iw = ow * SW - padL + kw;

                if (ih < 0 || ih >= IH) continue;
                if (iw < 0 || iw >= IW) continue;

                auto s = src[src_d.off(mb, oc, ih, iw)];
                if (s > d[0]) {
                    d[0] = s;
                    if (ws) ws[ws_d.off(mb, oc, oh, ow)] = kh*KW + kw;
                }
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int oc, int oh, int ow) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const int ih = oh * SH - padT + kh;
                const int iw = ow * SW - padL + kw;

                if (ih < 0 || ih >= IH) continue;
                if (iw < 0 || iw >= IW) continue;

                d[0] += src[src_d.off(mb, oc, ih, iw)];
            }
        }
    };

    const int MB = conf_.MB();
    const int OC = conf_.C();
    const int OH = conf_.OH();
    const int OW = conf_.OW();

    if (conf_.desc()->alg_kind == alg_kind::pooling_max) {
#       pragma omp parallel for collapse(4) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                        d[0] = -std::numeric_limits<data_t>::infinity();
                        ker_max(d, mb, oc, oh, ow);
                    }
                }
            }
        }
    } else {
#       pragma omp parallel for collapse(4) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                        d[0] = 0;
                        ker_avg(d, mb, oc, oh, ow);
                        d[0] /= KW*KH;
                    }
                }
            }
        }
    }
}

template struct ref_pooling_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
