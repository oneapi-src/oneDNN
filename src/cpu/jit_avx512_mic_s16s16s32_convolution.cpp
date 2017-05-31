/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "jit_avx512_mic_s16s16s32_convolution.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _jit_avx512_mic_s16s16s32_convolution_fwd_t<with_relu>::execute_forward()
{
    auto src = reinterpret_cast<const data_input_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_input_t*>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_output_t*>(this->input_memory(2));
    auto dst = reinterpret_cast<data_output_t*>(this->memory());
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        const int oc_dim = jcp.nb_oc / jcp.nb_oc_blocking;
        const size_t work_amount = jcp.mb * jcp.ngroups * oc_dim;
        size_t n{0}, g{0}, oc{0};
        jit_conv_call_s par_conv = {};

        balance211(work_amount, nthr, ithr, start, end);
        assert(jcp.loop_order == loop_ngc);
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, oc, oc_dim);

        par_conv.src_prf = NULL;
        par_conv.dst_prf = NULL;
        par_conv.filt_prf = NULL;
        par_conv.bias_prf = NULL;

        for (size_t iwork = start; iwork < end; ++iwork) {
            for (int ic = 0; ic < jcp.nb_ic; ++ic) {
                for (int oh = 0; oh < jcp.oh; ++oh) {

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                    const int i_b_overflow
                            = nstl::max(jcp.ih, ij + jcp.kh - jcp.t_pad)
                            - jcp.ih;

                    par_conv.src = par_conv.src_prf;
                    par_conv.dst = par_conv.dst_prf;
                    par_conv.filt = par_conv.filt_prf;
                    par_conv.bias = par_conv.bias_prf;
                    par_conv.channel = par_conv.channel_prf;

                    const int ih = nstl::max(ij - jcp.t_pad, 0);
                    const int oc_b = jcp.nb_oc_blocking * oc;
                    par_conv.src_prf = &src[src_d.blk_off(
                            n, g * jcp.nb_ic + ic, ih, 0)];
                    par_conv.dst_prf = &dst[dst_d.blk_off(
                            n, g * jcp.nb_oc + oc_b, oh, 0)];
                    if (bias)
                        par_conv.bias_prf = &bias[bias_d.blk_off(
                                    (g * jcp.nb_oc + oc_b) * jcp.oc_block)];
                    par_conv.filt_prf = &weights[conf_.with_groups() ?
                                            weights_d.blk_off(g,
                                                oc_b, ic, i_t_overflow, 0) :
                                            weights_d.blk_off(
                                                oc_b, ic, i_t_overflow, 0)];
                    par_conv.kh_padding = par_conv.kh_padding_prf;
                    par_conv.kh_padding_prf
                            = jcp.kh - i_t_overflow - i_b_overflow;
                    par_conv.kw_padding = 0;
                    par_conv.channel_prf = ic;

                    if (par_conv.src != NULL)
                        kernel_->jit_ker(&par_conv);
                }
            }
            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, oc, oc_dim);
        }

        par_conv.src = par_conv.src_prf;
        par_conv.dst = par_conv.dst_prf;
        par_conv.filt = par_conv.filt_prf;
        par_conv.bias = par_conv.bias_prf;
        par_conv.channel = par_conv.channel_prf;

        par_conv.src_prf = NULL;
        par_conv.dst_prf = NULL;
        par_conv.filt_prf = NULL;
        par_conv.bias_prf = NULL;

        par_conv.kh_padding = par_conv.kh_padding_prf;
        par_conv.kw_padding = 0;

        if (par_conv.src != NULL)
            kernel_->jit_ker(&par_conv);
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_avx512_mic_s16s16s32_convolution_fwd_t<false>
                                                            ::execute_forward();
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
