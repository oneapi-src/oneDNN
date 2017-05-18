/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_traits.hpp"

#include "ref_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
         data_type_t acc_type, data_type_t dst_type>
void _ref_convolution_fwd_t<with_relu, src_type, wei_type, acc_type, dst_type>
        ::execute_forward() {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int IH = conf_.IH();
    const int IW = conf_.IW();

    const int OC = conf_.OC() / G;
    const int IC = conf_.IC() / G;
    const int KH = conf_.KH();
    const int KW = conf_.KW();

    const int KSH = conf_.KSH();
    const int KSW = conf_.KSW();

    const int padT = conf_.padT();
    const int padL = conf_.padL();

    const double nslope = conf_.negative_slope();

    auto ker = [=](acc_data_t &d, int g, int mb, int oc, int oh, int ow) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int ih = oh * KSH - padT + kh;
                    const int iw = ow * KSW - padL + kw;

                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    d += (acc_data_t)src[src_d.off(mb, g*IC + ic, ih, iw)]
                        * (with_groups
                                ? weights[weights_d.off(g, oc, ic, kh, kw)]
                                : weights[weights_d.off(oc, ic, kh, kw)]);
                }
            }
        }
    };

    auto get_bias = [=](size_t off) -> acc_data_t {
        switch (conf_.cdesc()->bias_desc.data_type) {
        case data_type::s8: return *((const int8_t *)bias + off);
        case data_type::u8: return *((const uint8_t *)bias + off);
        case data_type::s32: return *((const int *)bias + off);
        case data_type::f32: return *((const float *)bias + off);
        default: assert(!"unimplemented");
        }
        return 0;
    };

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        acc_data_t a = bias
                            ? get_bias(bias_d.off(g*OC + oc)) : (acc_data_t)0;
                        ker(a, g, mb, oc, oh, ow);
                        if (with_relu && a < (acc_data_t)0) a *= nslope;
                        dst[dst_d.off(mb, g*OC + oc, oh, ow)] = (dst_data_t)a;
                    }
                }
            }
        }
    }
}

template <impl::data_type_t data_type>
void ref_convolution_bwd_data_t<data_type>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int IH = conf_.IH();
    const int IW = conf_.IW();

    const int OC = conf_.OC() / G;
    const int IC = conf_.IC() / G;
    const int KH = conf_.KH();
    const int KW = conf_.KW();

    const int KSH = conf_.KSH();
    const int KSW = conf_.KSW();

    const int padT = conf_.padT();
    const int padL = conf_.padL();


    auto ker = [=](data_t *d, int g, int mb, int ic, int ih, int iw) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    if (iw + padL < kw || ih + padT < kh)
                        continue;
                    int ow = iw - kw + padL;
                    int oh = ih - kh + padT;
                    if (ow % KSW != 0 || oh % KSH != 0)
                        continue;
                    ow /= KSW;
                    oh /= KSH;

                    if (oh < OH && ow < OW) {
                        *d += diff_dst[diff_dst_d.off(mb, g*OC + oc, oh, ow)] *
                            (with_groups ?
                             weights[weights_d.off(g, oc, ic, kh, kw)] :
                             weights[weights_d.off(oc, ic, kh, kw)]);
                    }
                }
            }
        }
    };

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int ic = 0; ic < IC; ++ic) {
                for (int ih = 0; ih < IH; ++ih) {
                    for (int iw = 0; iw < IW; ++iw) {
                        auto ds_idx = diff_src_d.off(mb, g*IC + ic, ih, iw);
                        data_t *d = &diff_src[ds_idx];
                        *d = data_t(0);
                        ker(d, g, mb, ic, ih, iw);
                    }
                }
            }
        }
    }
}

template <impl::data_type_t data_type>
void ref_convolution_bwd_weights_t<data_type>::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd(0));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int IH = conf_.IH();
    const int IW = conf_.IW();

    const int OC = conf_.OC() / G;
    const int IC = conf_.IC() / G;
    const int KH = conf_.KH();
    const int KW = conf_.KW();

    const int KSH = conf_.KSH();
    const int KSW = conf_.KSW();

    const int padT = conf_.padT();
    const int padL = conf_.padL();


    auto ker = [=](data_t *d, int g, int oc, int ic, int kh, int kw) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    if (ow*KSW + kw < padL
                            || oh*KSH + kh < padT
                            || ow*KSW + kw >= IW + padL
                            || oh*KSH + kh >= IH + padT)
                        continue;

                    int ih = oh*KSH - padT + kh;
                    int iw = ow*KSW - padL + kw;

                    *d += diff_dst[diff_dst_d.off(mb, g*OC + oc, oh, ow)] *
                        src[src_d.off(mb, g*IC + ic, ih, iw)];
                }
            }
        }
    };

    auto ker_bias = [=](data_t *d, int g, int oc) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    *d += diff_dst[diff_dst_d.off(mb, g*OC + oc, oh, ow)];
                }
            }
        }
    };

#   pragma omp parallel for collapse(2) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int oc = 0; oc < OC; ++oc) {
            if (diff_bias) {
                data_t *db = &diff_bias[diff_bias_d.off(g*OC+oc)];
               *db = data_t(0);
                ker_bias(db, g, oc);
            }

            for (int ic = 0; ic < IC; ++ic) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        data_t *d = with_groups
                            ? &diff_weights[diff_weights_d.off(g, oc, ic, kh, kw)]
                            : &diff_weights[diff_weights_d.off(oc, ic, kh, kw)];
                        *d = data_t(0);
                        ker(d, g, oc, ic, kh, kw);
                    }
                }
            }
        }
    }
}

template struct _ref_convolution_fwd_t<false, data_type::f32>;
template struct _ref_convolution_fwd_t<true, data_type::f32>;
template struct _ref_convolution_fwd_t<false, data_type::u8, data_type::s8,
         data_type::s32, data_type::u8>;
template struct _ref_convolution_fwd_t<true, data_type::u8, data_type::s8,
         data_type::s32, data_type::u8>;
template struct _ref_convolution_fwd_t<false, data_type::s16, data_type::s16,
         data_type::s32, data_type::s32>;
template struct _ref_convolution_fwd_t<true, data_type::s16, data_type::s16,
         data_type::s32, data_type::s32>;

template struct ref_convolution_bwd_data_t<data_type::f32>;
template struct ref_convolution_bwd_weights_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
