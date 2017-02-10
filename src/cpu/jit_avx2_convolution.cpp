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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_convolution.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <bool with_relu>
void _jit_avx2_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    auto ker = [&](int g, int n, int oc, int ic, int oh) {
        jit_conv_call_s par_conv = {};

        const int ij = oh * jcp.stride_h;
        const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
        const int i_b_overflow = nstl::max(jcp.ih, ij + jcp.kh - jcp.t_pad)
            - jcp.ih;

        const size_t _oc = g * jcp.nb_oc + oc;
        const size_t _ic = g * jcp.nb_ic + ic;

        const int ih = nstl::max(ij - jcp.t_pad, 0);
        par_conv.src = const_cast<data_t *>(&src[src_d.blk_off(n,
                    jcp.ic == 3 ? 0 : _ic, ih, 0)]);

        par_conv.dst = &dst[dst_d.blk_off(n, _oc, oh, 0)];

        const int wh = i_t_overflow;
        par_conv.filt = &weights[conf_.with_groups()
            ? weights_d.blk_off(g, oc, jcp.ic == 3 ? 0 : ic, wh, 0)
            : weights_d.blk_off(oc, jcp.ic == 3 ? 0 : ic, wh, 0)];

        if (ic == 0) {
            if (bias)
                par_conv.bias = &bias[bias_d.blk_off(_oc * jcp.oc_block)];
            par_conv.ic_flag |= jit_avx2_conv_fwd_kernel_f32::IC_FLAG_FIRST;
        }

        if (with_relu && ic + 1 == jcp.nb_ic) {
            par_conv.ic_flag |= jit_avx2_conv_fwd_kernel_f32::IC_FLAG_LAST;
        }

        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kw_padding = 0;

        par_conv.oc_blocks
            = nstl::min(oc + jcp.nb_oc_blocking, jcp.nb_oc) - oc;

        kernel_->jit_ker(&par_conv);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int g = 0; g < jcp.ngroups; ++g) {
        for (int n = 0; n < jcp.mb; ++n) {
            for (int oc = 0; oc < jcp.nb_oc; oc += jcp.nb_oc_blocking) {
                for (int ic = 0; ic < jcp.nb_ic; ++ic) {
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        ker(g, n, oc, ic, oh);
                    }
                }
            }
        }
    }
}

template void _jit_avx2_convolution_fwd_t<true>::execute_forward();
template void _jit_avx2_convolution_fwd_t<false>::execute_forward();

void jit_avx2_convolution_bwd_data_t::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    auto ker = [&](int g, int n, int ic, int oc, int ih) {
        jit_conv_call_s par_conv = {};

        const int i_t_overflow = nstl::max(0, jcp.kh - 1 - ih - jcp.t_pad);
        const int b_pad = jcp.ihp - jcp.ih - jcp.t_pad;
        const int i_b_overflow = nstl::max(0, jcp.kh - 1 - (jcp.ih - 1 - ih) - b_pad);
        const int oh = ih + jcp.t_pad - i_b_overflow;

        const int simd_w = 8;

        par_conv.src = &diff_src[diff_src_d.blk_off(n,
                    /*jcp.ic == 3 ? 0 :*/
                g * jcp.nb_ic + jcp.nb_ic_blocking*ic, ih, 0)];
        par_conv.dst = const_cast<data_t *>(&diff_dst[diff_dst_d.blk_off(n,
                g * jcp.nb_oc + oc, oh, 0)]);
        par_conv.filt = const_cast<data_t *>(&weights[conf_.with_groups()
            ? weights_d.blk_off(g, oc, jcp.ic == 3 ? 0 : jcp.nb_ic_blocking*ic, i_b_overflow, 0)
            : weights_d.blk_off(oc, jcp.ic == 3 ? 0 : jcp.nb_ic_blocking*ic, i_b_overflow, 0)]);
        par_conv.src_prf  = nullptr;
        par_conv.dst_prf  = nullptr;
        par_conv.filt_prf = nullptr;
        // TODO: move initialization into the kernel
        if (oc == 0)
        {
            for (int iw = 0; iw < jcp.iw; iw++)
            {
                for (int b = 0; b < jcp.nb_ic_blocking; b++)
                {
                    int current_ic = (jcp.ic == 3 ? 0 : g*jcp.nb_ic)
                        + jcp.nb_ic_blocking*ic+b;
                    int current_idx = diff_src_d.blk_off(n, current_ic, ih, iw);
                    for (int v = 0; v < simd_w; v++)
                        diff_src[current_idx + v] = 0.0;
                }
            }
        }

        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kw_padding = 0;

        kernel_->jit_ker(&par_conv);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < jcp.mb; ++n) {
        for (int g = 0; g < jcp.ngroups; ++g) {
            for (int ic = 0; ic < (jcp.nb_ic/jcp.nb_ic_blocking); ++ic) {
                for (int oc = 0; oc < jcp.nb_oc; ++oc) {
                    for (int ih = 0; ih < jcp.ih; ++ih) {
                        ker(g, n, ic, oc, ih);
                    }
                }
            }
        }
    }
}

void jit_avx2_convolution_bwd_weights_t::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd(0));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const auto &jcp = kernel_->jcp;

    auto ker = [&](int g, int n, int oc, int ic) {
        jit_conv_call_s par_conv = {};

        par_conv.src = &src[src_d.blk_off(n, g * jcp.nb_ic + ic)];
        par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, g * jcp.nb_oc + oc)];

        const size_t wdiff_offset = conf_.with_groups()
            ? diff_weights_d.blk_off(g, oc, ic, 0, 0)
            : diff_weights_d.blk_off(oc, ic, 0, 0);
        par_conv.filt = &diff_weights[wdiff_offset];

        // TODO: move initialization into the kernel
        if (n == 0) {
            const size_t sz = jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
            for (size_t i = 0; i < sz; ++i) diff_weights[wdiff_offset + i] = 0.0;
        }

        if (diff_bias && ic == 0) {
            const size_t _c = g*jcp.nb_oc + oc;
            auto db = &diff_bias[diff_bias_d.blk_off(_c*jcp.oc_block)];

            if (n == 0) {
                for (int cb = 0; cb < jcp.oc_block; ++cb) db[cb] = 0.0;
            }

            for (int h = 0; h < jcp.oh; ++h) {
                for (int w = 0; w < jcp.ow; ++w) {
                    auto dd = &diff_dst[diff_dst_d.blk_off(n,
                            g * jcp.nb_oc + oc, h, w)];
                    for (int cb = 0; cb < jcp.oc_block; ++cb) {
                        db[cb] += dd[cb];
                    }
                }
            }
        }

        kernel_->jit_ker(&par_conv);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int g = 0; g < jcp.ngroups; ++g) {
        for (int oc = 0; oc < jcp.nb_oc; ++oc) {
            for (int ic = 0; ic < jcp.nb_ic; ++ic) {
                for (int n = 0; n < jcp.mb; ++n) {
                    ker(g, n, oc, ic);
                }
            }
        }
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
