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
#include "jit_avx512_mic_1x1_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _jit_avx512_mic_1x1_convolution_fwd_t<with_relu>::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    const size_t work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, osb{0};
        int iwork = (int)start;
        while (iwork < (int)end) {
            nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                             jcp.nb_bcast);

            int bcast_step_rem = jcp.nb_bcast - osb;
            int bcast_step = (bcast_step_rem <= jcp.nb_bcast_blocking_max)
                                     ? bcast_step_rem
                                     : jcp.nb_bcast_blocking;
            if ((iwork + bcast_step) > (int)end)
                bcast_step = (int)end - iwork;

            int ocb = 0;
            while (ocb < jcp.nb_load) {
                int load_step_rem = jcp.nb_load - ocb;
                int load_step = (load_step_rem < jcp.nb_load_blocking_max)
                                        ? load_step_rem
                                        : jcp.nb_load_blocking;

                for (int icb = 0; icb < jcp.nb_reduce;
                     icb += jcp.nb_reduce_blocking) {
                    jit_1x1_conv_call_s par_conv = {};

                    // TODO (Roma): remove this restriction
                    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

                    int nb_ic = jcp.nb_reduce, nb_oc = jcp.nb_load;
                    int os_block = jcp.bcast_block;
                    int nb_ic_blocking = jcp.nb_reduce_blocking;
                    int nb_oc_blocking = load_step;

                    int os = osb * os_block;
                    int nb_os_blocking = bcast_step;

                    int ow = os % jcp.ow;
                    int oh = os / jcp.ow;
                    size_t _ocb = g * nb_oc + ocb;
                    size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);
                    par_conv.output_data = &dst[dst_off];

                    int iw = nstl::max(ow * jcp.stride_w - jcp.l_pad, 0);
                    int ih = nstl::max(oh * jcp.stride_h - jcp.t_pad, 0);
                    size_t _icb = g * nb_ic + icb;
                    size_t src_off = src_d.blk_off(n, _icb, ih, iw);
                    par_conv.bcast_data = &src[src_off];

                    par_conv.load_data =
                            &weights[conf_.with_groups()
                                             ? weights_d.blk_off(g, ocb, icb)
                                             : weights_d.blk_off(ocb, icb)];

                    if (icb == 0 && bias)
                        par_conv.bias_data =
                                &bias[bias_d.blk_off(_ocb * jcp.oc_block)];

                    par_conv.reduce_pos_flag
                            = (icb == 0 ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                  REDUCE_FLAG_FIRST
                                        : 0)
                              | (icb + nb_ic_blocking >= nb_ic
                                         ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                   REDUCE_FLAG_LAST
                                         : 0);

                    par_conv.load_dim
                            = this_block_size(ocb * jcp.oc_block, jcp.oc,
                                              nb_oc_blocking * jcp.oc_block);
                    par_conv.bcast_dim = this_block_size(
                            os, jcp.os, nb_os_blocking * os_block);
                    par_conv.reduce_dim
                            = this_block_size(icb * jcp.ic_block, jcp.ic,
                                              nb_ic_blocking * jcp.ic_block);

                    kernel_->jit_ker(&par_conv);
                }
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_avx512_mic_1x1_convolution_fwd_t<true>::execute_forward();
template void _jit_avx512_mic_1x1_convolution_fwd_t<false>::execute_forward();

void jit_avx512_mic_1x1_convolution_bwd_data_t::execute_backward_data()
{
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const auto &jcp = kernel_->jcp;

    const size_t work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        balance211(work_amount, nthr, ithr, start, end);

        int icb = 0;
        while (icb < jcp.nb_load) {
            int load_step_rem = jcp.nb_load - icb;
            int load_step = (load_step_rem < jcp.nb_load_blocking_max)
                                    ? load_step_rem
                                    : jcp.nb_load_blocking;

            size_t n{0}, g{0}, isb{0};
            int iwork = (int)start;
            while (iwork < (int)end) {
                nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, isb,
                                 jcp.nb_bcast);
                int bcast_step = jcp.nb_bcast_blocking;
                int bcast_step_rem = jcp.nb_bcast - isb;
                if (bcast_step_rem <= jcp.nb_bcast_blocking_max)
                    bcast_step = bcast_step_rem;
                if ((iwork + bcast_step) > (int)end)
                    bcast_step = (int)end - iwork;

                for (int ocb = 0; ocb < jcp.nb_reduce;
                     ocb += jcp.nb_reduce_blocking) {
                    jit_1x1_conv_call_s par_conv = {};

                    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

                    int nb_ic = jcp.nb_load, nb_oc = jcp.nb_reduce;
                    int is_block = jcp.bcast_block;
                    int nb_ic_blocking = load_step;
                    int nb_oc_blocking = jcp.nb_reduce_blocking;

                    int is = isb * is_block;
                    int nb_is_blocking = bcast_step;

                    int iw = is % jcp.iw;
                    int ih = is / jcp.iw;
                    size_t _icb = g * nb_ic + icb;
                    size_t diff_src_off = diff_src_d.blk_off(n, _icb, ih, iw);
                    par_conv.output_data = &diff_src[diff_src_off];

                    int ow = iw;
                    int oh = ih;
                    size_t _ocb = g * nb_oc + ocb;
                    size_t diff_dst_off = diff_dst_d.blk_off(n, _ocb, oh, ow);
                    par_conv.bcast_data = &diff_dst[diff_dst_off];

                    par_conv.load_data =
                            &weights[conf_.with_groups()
                                             ? weights_d.blk_off(g, ocb, icb)
                                             : weights_d.blk_off(ocb, icb)];

                    par_conv.reduce_pos_flag
                            = (ocb == 0 ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                  REDUCE_FLAG_FIRST
                                        : 0)
                              | (ocb + nb_oc_blocking >= nb_oc
                                         ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                   REDUCE_FLAG_LAST
                                         : 0);

                    par_conv.reduce_dim
                            = this_block_size(ocb * jcp.oc_block, jcp.oc,
                                              nb_oc_blocking * jcp.oc_block);
                    par_conv.bcast_dim = this_block_size(
                            is, jcp.is, nb_is_blocking * is_block);
                    par_conv.load_dim
                            = this_block_size(icb * jcp.ic_block, jcp.ic,
                                              nb_ic_blocking * jcp.ic_block);

                    kernel_->jit_ker(&par_conv);
                }
                iwork += bcast_step;
            }
            icb += load_step;
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

void jit_avx512_mic_1x1_convolution_bwd_weights_t::execute_backward_weights()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int load_work = div_up(jcp.nb_load, jcp.nb_load_blocking);
    int bcast_work = div_up(jcp.nb_bcast, jcp.nb_bcast_blocking);
    const size_t work_amount = jcp.ngroups * load_work * bcast_work;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        size_t g{0}, load_i{0}, bcast_i{0};

        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, load_i, load_work, bcast_i,
                         bcast_work);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb, icb;
            icb = jcp.nb_bcast_blocking * bcast_i;
            ocb = jcp.nb_load_blocking * load_i;
            for (int n = 0; n < jcp.mb; ++n) {
                for (int osb = 0; osb < jcp.nb_reduce;
                     osb += jcp.nb_reduce_blocking) {
                    jit_1x1_conv_call_s par_conv = {};

                    int nb_ic = jcp.nb_bcast, nb_oc = jcp.nb_load;
                    int nb_os = jcp.nb_reduce, os_block = jcp.reduce_block;
                    int nb_os_blocking = jcp.nb_reduce_blocking;
                    int nb_oc_blocking = jcp.nb_load_blocking;
                    int nb_ic_blocking = jcp.nb_bcast_blocking;

                    int os = osb * os_block;

                    // TODO (Roma): remove this restriction
                    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

                    int oh = os / jcp.ow;
                    int ow = os % jcp.ow;
                    size_t _ocb = g * nb_oc + ocb;
                    size_t diff_dst_off = diff_dst_d.blk_off(n, _ocb, oh, ow);
                    par_conv.load_data = &diff_dst[diff_dst_off];

                    int iw = ow;
                    int ih = oh;
                    size_t _icb = g * nb_ic + icb;
                    size_t src_off = src_d.blk_off(n, _icb, ih, iw);
                    par_conv.bcast_data = &src[src_off];

                    par_conv.output_data =
                            &diff_weights[conf_.with_groups()
                                                  ? diff_weights_d.blk_off(
                                                            g, ocb, icb)
                                                  : diff_weights_d.blk_off(
                                                            ocb, icb)];

                    par_conv.reduce_pos_flag
                            = (osb == 0 && n == 0
                                       ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                 REDUCE_FLAG_FIRST
                                       : 0)
                              | (osb + nb_os_blocking >= nb_os && n + 1
                                                                  == jcp.mb
                                         ? jit_avx512_mic_1x1_conv_kernel_f32::
                                                   REDUCE_FLAG_LAST
                                         : 0);

                    par_conv.reduce_dim = this_block_size(
                            os, jcp.os, nb_os_blocking * os_block);
                    par_conv.bcast_dim
                            = this_block_size(icb * jcp.ic_block, jcp.ic,
                                              nb_ic_blocking * jcp.ic_block);
                    par_conv.load_dim
                            = this_block_size(ocb * jcp.oc_block, jcp.oc,
                                              nb_oc_blocking * jcp.oc_block);
                    par_conv.output_stride = jcp.ic * jcp.oc_block
                                             * sizeof(float);

                    par_conv.bias_data
                            = (diff_bias && icb == 0)
                                      ? &diff_bias[diff_bias_d.blk_off(
                                                 _ocb * jcp.oc_block)]
                                      : 0;

                    kernel_->jit_ker(&par_conv);
                }
            }
            nd_iterator_step(g, jcp.ngroups, load_i, load_work, bcast_i,
                             bcast_work);
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}
}
}
}
