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
void _jit_avx2_convolution_t<with_relu>::execute_forward() {
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

        const int ih = nstl::max(ij - jcp.t_pad, 0);
        par_conv.src = const_cast<data_t *>(&src[src_d.blk_off(n,
                    jcp.ic == 3 ? 0 : g * jcp.nb_ic + ic, ih, 0)]);

        par_conv.dst = &dst[dst_d.blk_off(n,
                g * jcp.nb_oc + oc * jcp.nb_oc_blocking, oh, 0)];

        const int wcb = jcp.nb_oc_blocking*oc;
        const int wh = i_t_overflow;
        par_conv.filt = &weights[conf_.with_groups()
            ? weights_d.blk_off(g, wcb, jcp.ic == 3 ? 0 : ic, wh, 0)
            : weights_d.blk_off(wcb, jcp.ic == 3 ? 0 : ic, wh, 0)];

        if (ic == 0) {
            if (bias) {
                const size_t _c = g*jcp.nb_oc + jcp.nb_oc_blocking*oc;
                par_conv.bias = &bias[bias_d.blk_off(_c*jcp.oc_block)];
            }
            par_conv.ic_flag |= jit_avx2_conv_kernel_f32::IC_FLAG_FIRST;
        }

        if (with_relu && ic + 1 == jcp.nb_ic) {
            par_conv.ic_flag |= jit_avx2_conv_kernel_f32::IC_FLAG_LAST;
        }

        par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
        par_conv.kw_padding = 0;

        kernel_->jit_ker(&par_conv);
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int g = 0; g < jcp.ngroups; ++g) {
        for (int n = 0; n < jcp.mb; ++n) {
            for (int oc = 0; oc < (jcp.nb_oc/jcp.nb_oc_blocking); ++oc) {
                for (int ic = 0; ic < jcp.nb_ic; ++ic) {
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        ker(g, n, oc, ic, oh);
                    }
                }
            }
        }
    }
}

template void _jit_avx2_convolution_t<true>::execute_forward();
template void _jit_avx2_convolution_t<false>::execute_forward();

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
