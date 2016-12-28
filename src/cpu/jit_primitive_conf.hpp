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

#ifndef JIT_PRIMITIVE_CONF_HPP
#define JIT_PRIMITIVE_CONF_HPP

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_conv_conf_t {
    int mb;
    int ngroups, ic, oc;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    double relu_negative_slope;

    int ihp, iwp, ohp, owp;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    int ur_h, ur_w;
    int ur_w_tail;
};

struct __attribute__((__packed__)) jit_conv_call_s {
    const float *src; /* hack, non-const for backward_data */
    const float *dst; /* hack, non-const for forward */
    const float *filt; /* hack, non-const for backward_weights */
    const float *bias; /* hack, non-const for backward_bias */
    const float *src_prf;
    const float *dst_prf;
    const float *filt_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    int ic_flag;
};


}
}
}

#endif
