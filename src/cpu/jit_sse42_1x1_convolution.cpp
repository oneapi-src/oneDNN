/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "jit_sse42_1x1_convolution.hpp"
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
void _jit_sse42_1x1_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        jit_1x1_conv_call_s par_conv = {};

        const int nb_oc = jcp.nb_load;
        const int nb_ic = jcp.nb_reduce;
        const int nb_ic_blocking = jcp.nb_reduce_blocking;
        const int os_block = jcp.bcast_block;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, osb{0};
            nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                    jcp.nb_bcast);

            const int bcast_step_rem = jcp.nb_bcast - osb;
            int bcast_step = bcast_step_rem <= jcp.nb_bcast_blocking_max
                ? bcast_step_rem : jcp.nb_bcast_blocking;
            bcast_step = nstl::min<int>(bcast_step, end - iwork);

            const int os = osb * os_block;
            const int ow = os % jcp.ow;
            const int oh = os / jcp.ow;
            const int iw = nstl::max<int>(ow * jcp.stride_w - jcp.l_pad, 0);
            const int ih = nstl::max<int>(oh * jcp.stride_h - jcp.t_pad, 0);

            par_conv.bcast_dim = this_block_size(os, jcp.os,
                    bcast_step * os_block);

            int ocb = 0;
            while (ocb < jcp.nb_load) {
                const int load_step_rem = jcp.nb_load - ocb;
                const int load_step = load_step_rem < jcp.nb_load_blocking_max
                    ? load_step_rem : jcp.nb_load_blocking;

                const size_t _ocb = g * nb_oc + ocb;
                par_conv.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                        load_step * jcp.oc_block);

                const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);
                par_conv.output_data = &dst[dst_off];

                par_conv.bias_data = &bias[_ocb * jcp.oc_block];

                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    par_conv.reduce_pos_flag = 0
                        | (icb == 0) * FLAG_REDUCE_FIRST
                        | (icb + nb_ic_blocking >= nb_ic) * FLAG_REDUCE_LAST;

                    par_conv.reduce_dim = this_block_size(icb * jcp.ic_block,
                            jcp.ic, nb_ic_blocking * jcp.ic_block);

                    const size_t _icb = g * nb_ic + icb;
                    const size_t src_off = src_d.blk_off(n, _icb, ih, iw);
                    par_conv.bcast_data = &src[src_off];

                    par_conv.load_data = &weights[conf_.with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    kernel_->jit_ker(&par_conv);
                }

                ocb += load_step;
            }

            iwork += bcast_step;
        }
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_sse42_1x1_convolution_fwd_t<true>::execute_forward();
template void _jit_sse42_1x1_convolution_fwd_t<false>::execute_forward();

}
}
}
