/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "dnnl_types.h"

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "jit_sse41_1x1_convolution.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

#define data_blk_off(f, n, c, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) : (f).blk_off(n, c, h, w))

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

void jit_sse41_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad();
}

void jit_sse41_1x1_convolution_fwd_t::execute_forward_thr(const int ithr,
        const int nthr, const data_t *src, const data_t *weights,
        const data_t *bias, data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int ndims = src_d.ndims();

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    auto par_conv = jit_1x1_conv_call_s();

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;
    const int os_block = jcp.bcast_block;

    int start {0}, end {0};
    balance211(work_amount, nthr, ithr, start, end);

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step, int &oh,
                              int &ow, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);

        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, end - iwork);

        const int os = osb * os_block;
        ow = os % jcp.ow;
        oh = os / jcp.ow;

        ih = oh * jcp.stride_h;
        iw = ow * jcp.stride_w;

        par_conv.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
    };

    auto init_load = [&](int ocb, int &load_step) {
        load_step = step(jcp.nb_load_blocking, jcp.nb_load - ocb,
                jcp.nb_load_blocking_max);
        par_conv.load_dim = this_block_size(
                ocb * jcp.oc_block, jcp.oc, load_step * jcp.oc_block);
    };

    auto inner_ker = [&](int ocb, int icb, int n, int g, int oh, int ow, int ih,
                             int iw) {
        const size_t _ocb = g * nb_oc + ocb;
        const size_t _icb = g * nb_ic + icb;

        par_conv.output_data = &dst[data_blk_off(dst_d, n, _ocb, oh, ow)];

        par_conv.bias_data = &bias[_ocb * jcp.oc_block];

        par_conv.first_last_flag = 0 | (icb == 0) * FLAG_REDUCE_FIRST
                | (icb + nb_ic_blocking >= nb_ic) * FLAG_REDUCE_LAST;

        par_conv.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking * jcp.ic_block);

        const size_t src_off = data_blk_off(src_d, n, _icb, ih, iw);
        par_conv.bcast_data = &src[src_off];

        par_conv.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        kernel_->jit_ker(&par_conv);
    };

    int iwork = start;
    while (iwork < end) {
        int n, g, bcast_step, oh, ow, ih, iw;
        init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
        int ocb = 0;
        while (ocb < jcp.nb_load) {
            int load_step;
            init_load(ocb, load_step);
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
            }
            ocb += load_step;
        }
        iwork += bcast_step;
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
